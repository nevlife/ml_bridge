# Copyright 2026 nevlife
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Node that runs a model on live topics.

The model module provides infer(state, sample) and optionally load(context), whose
result is the state passed to infer. Training runs without ROS: see training.py.
"""

import functools
import importlib
import json
import threading
from types import SimpleNamespace

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosbag2_dataset import LiveSampler
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import String

from .spec import check_version, Decoder, load_spec


def _stamp_ns(message):
    stamp = message.header.stamp
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class BridgeNode(Node):
    """Run a model on subscriptions."""

    def __init__(self, **kwargs):
        super().__init__('ml_bridge', **kwargs)
        self.spec = load_spec(self.declare_parameter('spec', Parameter.Type.STRING).value)
        self.model = importlib.import_module(
            self.declare_parameter('model', Parameter.Type.STRING).value)
        self.model_context = SimpleNamespace(
            spec=self.spec, logger=self.get_logger(),
            weights=self.declare_parameter('weights', '').value)
        if self.model_context.weights:
            check_version(self.spec, self.model_context.weights)
        self.decoder = Decoder(self.spec)
        self.newest_ns = 0
        self.received = 0
        self.used = 0
        load = getattr(self.model, 'load', None)
        self.state = load(self.model_context) if load else None
        self.live = LiveSampler(self.spec.rule())
        transport = self.declare_parameter('transport', 'ros').value
        device = self.declare_parameter('flux_device', 'cpu').value
        if transport not in ('ros', 'flux') or device not in ('cpu', 'cuda'):
            raise ValueError(f'unknown transport {transport!r} or flux_device {device!r}')
        single = len(self.spec.inputs) == 1 and self.spec.inputs[0].window is None
        # LiveSampler holds what it may still choose, and a held view would stop the channel.
        self.in_callback = transport == 'flux' and single
        if device == 'cuda' and not self.in_callback:
            raise ValueError('flux_device cuda needs one input without a window')
        self.output = None
        if self.spec.output_type:
            self.output = self.create_publisher(
                get_message(self.spec.output_type), self.spec.output_topic, 10)
        self.input_info = self.create_publisher(String, '~/input_info', 10)
        # A callback that blocks on the model loses messages of the other inputs.
        self.pending = None
        self.stopping = False
        self.wake = threading.Condition()
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()
        # Inference wants the newest data: no retransmission, short queues. A slow model on a
        # single input would otherwise work through old frames.
        qos = QoSProfile(depth=1 if single else 10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.flux_inputs = []
        for item in self.spec.inputs:
            callback = functools.partial(self._on_message, item.topic)
            if transport == 'flux' and item.type == 'sensor_msgs/msg/Image':
                from .flux_input import subscribe
                self.flux_inputs.append(subscribe(self, item.topic, callback,
                                                  copy=not self.in_callback, device=device))
            else:
                self.create_subscription(get_message(item.type), item.topic, callback, qos)

    def _on_message(self, topic, message):
        receive_ns = self.get_clock().now().nanoseconds
        clock_ns = _stamp_ns(message) if self.spec.time == 'header' else receive_ns
        if self.in_callback:
            self.received += 1
            self._run({'@time': clock_ns, topic: (receive_ns, message)}, 0)
            return
        try:
            self.live.add(topic, clock_ns, message, log_ns=receive_ns)
        except ValueError as error:
            self.get_logger().warn(f'{topic}: {error}', throttle_duration_sec=5.0)
            return
        # Waiting is measured on the sample clock: header stamps of a replayed bag are far
        # behind the wall clock.
        self.newest_ns = max(self.newest_ns, clock_ns)
        samples = self.live.ready(now_ns=self.newest_ns, max_wait_ns=self.spec.max_wait_ns)
        if not samples:
            return
        self.received += len(samples)
        with self.wake:
            skipped = len(samples) - 1 + (self.pending[1] + 1 if self.pending else 0)
            self.pending = (samples[-1], skipped)
            self.wake.notify()

    def _work(self):
        while True:
            with self.wake:
                while self.pending is None and not self.stopping:
                    self.wake.wait()
                if self.stopping:
                    return
                sample, skipped = self.pending
                self.pending = None
            self._run(sample, skipped)

    def destroy_node(self):
        with self.wake:
            self.stopping = True
            self.wake.notify()
        self.worker.join()
        super().destroy_node()

    def _run(self, raw_sample, skipped):
        self.used += 1
        sample = self.decoder(raw_sample)
        result = self.model.infer(self.state, sample)
        self.input_info.publish(String(data=json.dumps({
            'time': sample['@time'], 'skipped': skipped, 'early': self.live.early,
            'done': self.get_clock().now().nanoseconds,
            'inputs': {k: v[0] for k, v in sample.items()
                       if k != '@time' and isinstance(v, tuple)}})))
        for subscription in self.flux_inputs:
            if subscription.refused.max_borrow:
                self.get_logger().error('the model keeps a flux frame after infer, which '
                                        'stops the channel', throttle_duration_sec=5.0)
        if result is None or self.output is None:
            return
        if hasattr(result, 'header') and not _stamp_ns(result):
            result.header.stamp.sec, result.header.stamp.nanosec = divmod(sample['@time'],
                                                                          1_000_000_000)
        self.output.publish(result)


def main(args=None):
    rclpy.init(args=args)
    node = BridgeNode()
    try:
        if node.flux_inputs:
            from .flux_input import spin
            spin(node, node.flux_inputs)
        else:
            rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

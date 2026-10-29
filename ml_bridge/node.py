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
Node that trains a model on bags or runs it on live topics with the same samples.

The model module provides train(dataset, context), infer(state, sample), and
optionally load(context) whose result is the state passed to infer.
"""

import functools
import importlib
import json
from types import SimpleNamespace

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosbag2_dataset import LiveSampler
from rosbag2_dataset.torch import BagDataset
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import String

from .spec import Decoder, load_spec

MODES = ('train', 'infer')


def _stamp_ns(message):
    stamp = message.header.stamp
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class BridgeNode(Node):
    """Train on bags (mode train) or run on subscriptions (mode infer)."""

    def __init__(self, **kwargs):
        super().__init__('ml_bridge', **kwargs)
        self.spec = load_spec(self.declare_parameter('spec', Parameter.Type.STRING).value)
        self.mode = self.declare_parameter('mode', 'infer').value
        if self.mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}, not {self.mode!r}')
        self.model = importlib.import_module(
            self.declare_parameter('model', Parameter.Type.STRING).value)
        self.model_context = SimpleNamespace(
            spec=self.spec, logger=self.get_logger(),
            weights=self.declare_parameter('weights', '').value,
            out_dir=self.declare_parameter('out_dir', '').value,
            bags=list(self.declare_parameter('bags', ['']).value))
        self.model_context.bags = [bag for bag in self.model_context.bags if bag]
        self.decoder = Decoder(self.spec)
        self.received = 0
        self.used = 0
        if self.mode == 'infer':
            self._start_infer()

    def train(self):
        """Hand the model a dataset of the bags; without bags, its own data."""
        dataset = None
        if self.model_context.bags:
            dataset = BagDataset(self.model_context.bags, self.spec.topics, sync=self.spec.sync(),
                                 transform=self.decoder)
            self.get_logger().info(
                f'{len(dataset)} samples from {len(self.model_context.bags)} bags')
        self.model.train(dataset, self.model_context)

    def _start_infer(self):
        load = getattr(self.model, 'load', None)
        self.state = load(self.model_context) if load else None
        self.live = LiveSampler(self.spec.rule())
        self.output = None
        if self.spec.output_type:
            self.output = self.create_publisher(
                get_message(self.spec.output_type), self.spec.output_topic, 10)
        self.input_info = self.create_publisher(String, '~/input_info', 10)
        # Inference wants the newest data: no retransmission, short queues.
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        for item in self.spec.inputs:
            self.create_subscription(get_message(item.type), item.topic,
                                     functools.partial(self._on_message, item.topic), qos)

    def _on_message(self, topic, message):
        receive_ns = self.get_clock().now().nanoseconds
        clock_ns = _stamp_ns(message) if self.spec.time == 'header' else receive_ns
        try:
            self.live.add(topic, clock_ns, message, log_ns=receive_ns)
        except ValueError as error:
            self.get_logger().warn(f'{topic}: {error}', throttle_duration_sec=5.0)
            return
        samples = self.live.ready(now_ns=receive_ns, max_wait_ns=self.spec.max_wait_ns)
        if not samples:
            return
        self.received += len(samples)
        self.used += 1
        sample = self.decoder(samples[-1])
        result = self.model.infer(self.state, sample)
        self.input_info.publish(String(data=json.dumps({
            'time': sample['@time'], 'skipped': len(samples) - 1,
            'inputs': {k: v[0] for k, v in sample.items()
                       if k != '@time' and isinstance(v, tuple)}})))
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
        if node.mode == 'train':
            node.train()
        else:
            rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

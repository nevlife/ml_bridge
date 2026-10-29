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

"""Check that inference on topics receives the samples training reads from the bag."""

import os

import numpy as np
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter

MS = 1_000_000
SPEC = {
    'inputs': {
        '/cam': {'type': 'sensor_msgs/msg/Image'},
        '/odom': {'type': 'nav_msgs/msg/Odometry', 'tolerance_ms': 30,
                  'window': {'every_ms': 200, 'past': 3, 'tolerance_ms': 30}},
    },
    'time': 'header',
    'max_wait_ms': 10_000,
}


def _set_stamp(message, ns):
    message.header.stamp.sec, message.header.stamp.nanosec = divmod(ns, 10**9)


@pytest.fixture
def bag_events():
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image
    rng = np.random.default_rng(3)
    events = []
    for i in range(30):
        stamp = 10**9 + i * 100 * MS + int(rng.integers(0, 5)) * MS
        image = Image(height=2, width=3, encoding='bgr8', step=12)
        image.data = bytes((i + k) % 256 for k in range(24))
        _set_stamp(image, stamp)
        events.append(('/cam', stamp + 20 * MS, image))
    for i in range(60):
        stamp = 10**9 + i * 50 * MS + int(rng.integers(0, 4)) * MS
        odom = Odometry()
        odom.pose.pose.position.x = float(i)
        _set_stamp(odom, stamp)
        events.append(('/odom', stamp + 5 * MS, odom))
    events.sort(key=lambda event: event[1])
    return events


@pytest.fixture
def bag(bag_events, tmp_path):
    rosbag2_py = pytest.importorskip('rosbag2_py')
    from rclpy.serialization import serialize_message
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(tmp_path / 'bag'), storage_id='mcap'),
                rosbag2_py.ConverterOptions('', ''))
    for name, kind in [('/cam', 'sensor_msgs/msg/Image'), ('/odom', 'nav_msgs/msg/Odometry')]:
        writer.create_topic(rosbag2_py.TopicMetadata(id=0, name=name, type=kind,
                                                     serialization_format='cdr'))
    for name, time, message in bag_events:
        writer.write(name, serialize_message(message), time)
    del writer
    return tmp_path / 'bag'


@pytest.fixture
def context():
    context = rclpy.Context()
    rclpy.init(context=context, domain_id=100 + os.getpid() % 100)
    yield context
    rclpy.try_shutdown(context=context)


def _node(context, mode, spec_path, bags=()):
    from ml_bridge.node import BridgeNode
    return BridgeNode(context=context, parameter_overrides=[
        Parameter('spec', value=str(spec_path)), Parameter('mode', value=mode),
        Parameter('model', value='fake_model'),
        Parameter('bags', value=[str(b) for b in bags] or [''])])


def _keys(sample):
    def key(entry):
        if entry is None:
            return None
        value = entry[1]
        return value.tobytes() if isinstance(value, np.ndarray) else value.pose.pose.position.x

    return (sample['@time'], key(sample['/cam']), key(sample['/odom']),
            tuple(key(e) for e in sample['/odom@past']))


def test_infer_matches_train(context, bag, bag_events, tmp_path):
    import fake_model
    import yaml
    spec_path = tmp_path / 'spec.yaml'
    spec_path.write_text(yaml.safe_dump(SPEC))
    fake_model.trained.clear()
    fake_model.inferred.clear()

    _node(context, 'train', spec_path, [bag]).train()
    trained = {s['@time']: _keys(s) for s in fake_model.trained}
    assert len(trained) >= 25
    assert all(isinstance(s['/cam'][1], np.ndarray) and s['/cam'][1].shape == (2, 3, 3)
               for s in fake_model.trained)

    bridge = _node(context, 'infer', spec_path)
    source = rclpy.create_node('source', context=context)
    publishers = {name: source.create_publisher(type(message), name, 100)
                  for name, _, message in bag_events}
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(bridge)
    executor.add_node(source)
    while bridge.count_publishers('/cam') == 0 or not all(
            p.get_subscription_count() for p in publishers.values()):
        executor.spin_once(timeout_sec=0.1)
    for name, _, message in bag_events:
        publishers[name].publish(message)
        for _ in range(3):
            executor.spin_once(timeout_sec=0.01)
    for _ in range(20):
        executor.spin_once(timeout_sec=0.05)

    inferred = [_keys(s) for s in fake_model.inferred]
    assert len(inferred) >= 20
    assert all(trained[key[0]] == key for key in inferred)
    assert any(None in key[3] for key in inferred)
    executor.shutdown()
    source.destroy_node()
    bridge.destroy_node()

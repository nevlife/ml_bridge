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
    'max_wait_ms': 300,
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
        events.append(('/odom', stamp + 25 * MS, odom))
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


def _node(context, spec_path):
    from ml_bridge.node import BridgeNode
    return BridgeNode(context=context, parameter_overrides=[
        Parameter('spec', value=str(spec_path)), Parameter('model', value='fake_model')])


def _train(spec_path, bag, remap=None):
    import fake_model
    from ml_bridge.spec import load_spec
    from ml_bridge.training import make_dataset
    fake_model.train(make_dataset(load_spec(str(spec_path)), [bag], remap), None)


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

    _train(spec_path, bag)
    trained = {s['@time']: _keys(s) for s in fake_model.trained}
    assert len(trained) >= 25
    assert all(isinstance(s['/cam'][1], np.ndarray) and s['/cam'][1].shape == (2, 3, 3)
               for s in fake_model.trained)
    renamed_spec = dict(SPEC, inputs={'/camera': SPEC['inputs']['/cam'],
                                      '/odom': SPEC['inputs']['/odom']})
    renamed_path = tmp_path / 'renamed.yaml'
    renamed_path.write_text(yaml.safe_dump(renamed_spec))
    fake_model.trained.clear()
    _train(renamed_path, bag, {'/camera': '/cam'})
    assert [_keys(dict(s, **{'/cam': s['/camera']})) for s in fake_model.trained] == list(
        trained.values())
    assert all('/cam' not in s for s in fake_model.trained)

    bridge = _node(context, spec_path)
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
    assert len(inferred) >= len(trained) - 2
    assert all(trained[key[0]] == key for key in inferred)
    assert any(None in key[3] for key in inferred)
    executor.shutdown()
    source.destroy_node()
    bridge.destroy_node()


def _in_shared_memory(array):
    address = array.ctypes.data
    with open('/proc/self/maps') as maps:
        for line in maps:
            span, *rest = line.split()
            low, high = (int(part, 16) for part in span.split('-'))
            if low <= address < high:
                return rest[-1].startswith('/dev/shm/') and rest[-1] != span
    return False


def _flux_run(context, tmp_path, spec, parameters, count, on_infer, lag=0):
    """Publish rgb8 frames 10 ms apart on /cam; a sample comes lag frames after its own."""
    import threading
    import time
    import fake_model
    import yaml
    from ml_bridge.node import BridgeNode
    flux = pytest.importorskip('flux')
    image = pytest.importorskip('sensor_msgs_flux.image').Image
    spec_path = tmp_path / 'spec.yaml'
    spec_path.write_text(yaml.safe_dump(spec))
    fake_model.on_infer = on_infer
    fake_model.inferred.clear()
    bridge = BridgeNode(context=context, parameter_overrides=[
        Parameter('spec', value=str(spec_path)), Parameter('model', value='fake_model'),
        Parameter('transport', value='flux')] + parameters)
    source = rclpy.create_node('source', context=context)
    publisher = flux.Publisher(source, '/cam', fingerprint=image.FINGERPRINT__,
                               slot_size=1 << 16, slot_count=4)
    executor = flux.Executor()
    for subscription in bridge.flux_inputs:
        executor.add(subscription)
    executor.add_node(bridge)
    spinner = threading.Thread(target=executor.spin)
    spinner.start()
    try:
        deadline = time.monotonic() + 5.0
        while not all(s.attached for s in bridge.flux_inputs) and time.monotonic() < deadline:
            time.sleep(0.001)
        assert all(s.attached for s in bridge.flux_inputs)
        for i in range(count):
            deadline = time.monotonic() + 5.0
            while True:
                builder = image.build__(publisher)
                if builder:
                    break
                assert time.monotonic() < deadline, 'the channel stopped: a view is held'
                time.sleep(0.001)
            builder.set__header__stamp(1, i * 10 * MS)
            builder.height, builder.width, builder.step = 2, 3, 9
            builder.encoding = 'rgb8'
            builder.alloc__data(18)[:] = np.arange(18, dtype=np.uint8) + i
            assert not flux.faulted(builder.commit__())
            deadline = time.monotonic() + 2.0
            while bridge.used <= i - lag and time.monotonic() < deadline:
                time.sleep(0.001)
            assert bridge.used > i - lag, f'frame {i} did not arrive'
    finally:
        executor.stop()
        spinner.join()
        fake_model.on_infer = None
    return bridge, source


def test_flux_single_input_reads_the_slot(context, tmp_path):
    seen = []

    def check(sample):
        _, pixels = sample['/cam']
        seen.append((sample['@time'], pixels.shape, int(pixels[0, 0, 0]),
                     _in_shared_memory(pixels)))

    spec = {'inputs': {'/cam': {'type': 'sensor_msgs/msg/Image'}}, 'time': 'header'}
    bridge, source = _flux_run(context, tmp_path, spec, [], 30, check)
    assert bridge.in_callback
    assert seen == [(10**9 + i * 10 * MS, (2, 3, 3), i, True) for i in range(30)]
    bridge.destroy_node()
    source.destroy_node()


def test_flux_held_frames_are_copied(context, tmp_path):
    import fake_model
    spec = {'inputs': {'/cam': {'type': 'sensor_msgs/msg/Image',
                                'window': {'every_ms': 10, 'past': 1, 'tolerance_ms': 1}}},
            'time': 'header'}
    # More frames than the 4 slots, all held by the sampler or by fake_model.inferred.
    bridge, source = _flux_run(context, tmp_path, spec, [], 10, None, lag=1)
    assert not bridge.in_callback
    assert [int(s['/cam'][1][0, 0, 0]) for s in fake_model.inferred] == list(range(9))
    bridge.destroy_node()
    source.destroy_node()

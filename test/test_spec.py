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

"""Check spec parsing."""

import subprocess
import sys

from ml_bridge.spec import Decoder, load_spec
import numpy as np
import pytest


def test_spec_options():
    spec = load_spec({'inputs': {'/a': {'type': 'sensor_msgs/msg/Image'},
                                 '/b': {'type': 'nav_msgs/msg/Odometry', 'tolerance_ms': 20,
                                        'optional': True,
                                        'window': {'every_ms': 500, 'past': 4}}},
                      'time': 'log'})
    assert spec.reference == '/a' and spec.topics == ('/a', '/b')
    sync = spec.sync()
    assert sync['tolerance_ns'] == {'/b': 20_000_000} and sync['optional'] == ['/b']
    assert sync['windows']['/b'].past == 4 and sync['time_source'] == 'log'
    assert spec.rule().reference == '/a'


@pytest.mark.parametrize('data,match', [
    ({'inputs': {}}, 'at least one'),
    ({'inputs': {'/a': {}}}, 'type is required'),
    ({'inputs': {'/a': {'type': 'x', 'tolerance': 1}}}, 'unknown keys'),
    ({'inputs': {'/a': {'type': 'x'}}, 'reference': '/b'}, 'reference'),
    ({'inputs': {'/a': {'type': 'x'}}, 'time': 'wall'}, 'time'),
])
def test_spec_errors(data, match):
    with pytest.raises(ValueError, match=match):
        load_spec(data)


def test_images_become_rgb():
    from sensor_msgs.msg import Image
    spec = load_spec({'inputs': {'/a': {'type': 'sensor_msgs/msg/Image'}}})
    image = Image(height=1, width=2, encoding='bgra8', step=8, data=bytes(range(8)))
    sample = Decoder(spec)({'@time': 0, '/a': (5, image), '/a@past': [None, (4, image)]})
    np.testing.assert_array_equal(sample['/a'][1], [[[2, 1, 0], [6, 5, 4]]])
    assert sample['/a@past'][0] is None and sample['/a@past'][1][0] == 4


def test_training_does_not_need_ros():
    code = ("import sys; import ml_bridge.training; "
            "assert 'rclpy' not in sys.modules, 'training imported rclpy'")
    subprocess.run([sys.executable, '-c', code], check=True)

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

"""Image input over a flux channel; imported only when the node runs with transport flux."""

import flux
from sensor_msgs_flux.image import Image

_CHANNELS = {'mono8': 1, 'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}


class _Stamp:
    __slots__ = ('sec', 'nanosec')

    def __init__(self, stamp):
        self.sec, self.nanosec = stamp


class _Header:
    __slots__ = ('stamp',)

    def __init__(self, stamp):
        self.stamp = _Stamp(stamp)


class FluxImage:
    """
    The sensor_msgs/Image fields of a frame, which image_to_numpy reads like a message.

    Without copy, data points into the slot: the slot stays held while this object lives.
    device_pixels is the HxWxC tensor of a device subscription, valid during the callback.
    """

    __slots__ = ('header', 'height', 'width', 'encoding', 'is_bigendian', 'step', 'data',
                 'device_pixels')

    def __init__(self, view, data, device_pixels=None):
        self.header = _Header(view.header__stamp)
        self.height = int(view.height)
        self.width = int(view.width)
        self.encoding = view.encoding
        self.is_bigendian = int(view.is_bigendian)
        self.step = int(view.step)
        self.data = data
        self.device_pixels = device_pixels


def _on_device(frame):
    import torch
    host = frame._host()
    view = Image.View(host)
    channels = _CHANNELS.get(view.encoding)
    if channels is None:
        raise ValueError(f'No device image for encoding {view.encoding!r}')
    span = view.data
    offset = span.ctypes.data - host.ctypes.data
    height, step = int(view.height), int(view.step)
    rows = torch.from_dlpack(frame)[offset:offset + height * step].view(height, step)
    pixels = rows[:, :int(view.width) * channels].view(height, int(view.width), channels)
    return FluxImage(view, span, pixels)


def subscribe(node, topic, callback, *, copy, device):
    """Call callback with a FluxImage per frame; copy keeps it valid after the callback."""
    def on_frame(frame):
        if device == 'cuda':
            callback(_on_device(frame))
            return
        view = Image.View(frame)
        callback(FluxImage(view, view.data.copy() if copy else view.data))

    # A view that only lives through the callback wants the newest frame; a held one needs all.
    qos = flux.QoS() if copy else flux.QoS(depth=1)
    return flux.Subscription(node, topic, callback=on_frame, fingerprint=Image.FINGERPRINT__,
                             qos=qos, device=device)


def spin(node, subscriptions):
    executor = flux.Executor()
    for subscription in subscriptions:
        executor.add(subscription)
    executor.add_node(node)
    executor.spin()

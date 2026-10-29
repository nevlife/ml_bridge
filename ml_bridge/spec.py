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

"""Read a spec: the topics of one sample, how they align, and the output."""

from dataclasses import dataclass
from pathlib import Path

from rosbag2_dataset import image_to_numpy, SampleRule, Window
import yaml

MS = 1_000_000
_INPUT_KEYS = {'type', 'tolerance_ms', 'optional', 'window'}
_WINDOW_KEYS = {'every_ms', 'past', 'future', 'tolerance_ms'}
_SPEC_KEYS = {'inputs', 'reference', 'time', 'max_wait_ms', 'output'}
_RGB = {'bgr8': [2, 1, 0], 'bgra8': [2, 1, 0], 'rgba8': [0, 1, 2]}


def _image(message):
    pixels = image_to_numpy(message)
    order = _RGB.get(message.encoding)
    return pixels if order is None else pixels[..., order]


_DECODERS = {'sensor_msgs/msg/Image': _image}


def _check_keys(where, mapping, allowed):
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f'{where}: unknown keys {sorted(unknown)}')


@dataclass(frozen=True)
class Input:
    """One input topic of a sample."""

    topic: str
    type: str  # noqa: A003
    tolerance_ns: int = 0
    optional: bool = False
    window: Window = None


@dataclass(frozen=True)
class Spec:
    """Inputs, alignment, and output of a model."""

    inputs: tuple
    reference: str
    time: str
    max_wait_ns: int
    output_topic: str
    output_type: str

    @property
    def topics(self):
        return tuple(i.topic for i in self.inputs)

    def sync(self, names=None):
        """Return the Bag.synced and BagDataset sync options, topics renamed by names."""
        def name(topic):
            return names[topic] if names else topic

        return {
            'reference': name(self.reference),
            'tolerance_ns': {name(i.topic): i.tolerance_ns for i in self.inputs
                             if i.topic != self.reference},
            'optional': [name(i.topic) for i in self.inputs if i.optional],
            'windows': {name(i.topic): i.window for i in self.inputs if i.window},
            'time_source': self.time,
        }

    def rule(self):
        """Return the rule that live samples follow."""
        sync = self.sync()
        del sync['time_source']
        return SampleRule(self.topics, **sync)


def load_spec(source):
    """Load a spec from a YAML path or a mapping."""
    data = source if isinstance(source, dict) else yaml.safe_load(Path(source).read_text())
    _check_keys('spec', data, _SPEC_KEYS)
    if not data.get('inputs'):
        raise ValueError('spec: inputs must name at least one topic')
    inputs = []
    for topic, options in data['inputs'].items():
        options = options or {}
        _check_keys(f'inputs.{topic}', options, _INPUT_KEYS)
        if 'type' not in options:
            raise ValueError(f'inputs.{topic}: type is required')
        window = options.get('window')
        if window is not None:
            _check_keys(f'inputs.{topic}.window', window, _WINDOW_KEYS)
            window = Window(every_ns=round(window['every_ms'] * MS),
                            past=window.get('past', 0), future=window.get('future', 0),
                            tolerance_ns=round(window.get('tolerance_ms', 0) * MS))
        inputs.append(Input(topic, options['type'],
                            round(options.get('tolerance_ms', 0) * MS),
                            bool(options.get('optional', False)), window))
    reference = data.get('reference', inputs[0].topic)
    if reference not in {i.topic for i in inputs}:
        raise ValueError('spec: reference must be one of the inputs')
    time = data.get('time', 'header')
    if time not in ('header', 'log'):
        raise ValueError("spec: time must be 'header' or 'log'")
    output = data.get('output') or {}
    _check_keys('output', output, {'topic', 'type'})
    return Spec(tuple(inputs), reference, time, round(data.get('max_wait_ms', 100) * MS),
                output.get('topic', '~/output'), output.get('type'))


class Renamed:
    """Give the keys of a bag sample the spec's topic names, then decode."""

    def __init__(self, names, decoder):
        self.spec_names = {bag: spec for spec, bag in names.items()}
        self.decoder = decoder

    def __call__(self, sample):
        renamed = {}
        for key, value in sample.items():
            topic, at, suffix = key.partition('@')
            renamed[self.spec_names.get(topic, topic) + at + suffix] = value
        return self.decoder(renamed)


class Decoder:
    """
    Turn the messages of a sample into arrays where the type has a decoder.

    Images become HxW or HxWxC NumPy arrays; bgr8, bgra8, and rgba8 become RGB
    copies, other encodings stay views in their channel order. The same decoder runs
    on bag samples and on live samples, so both give equal arrays.
    """

    def __init__(self, spec):
        self.types = {i.topic: i.type for i in spec.inputs}

    def __call__(self, sample):
        decoded = {}
        for key, value in sample.items():
            decode = _DECODERS.get(self.types.get(key.split('@')[0]))
            if decode is None or value is None:
                decoded[key] = value
            elif isinstance(value, list):
                decoded[key] = [None if v is None else (v[0], decode(v[1])) for v in value]
            else:
                decoded[key] = (value[0], decode(value[1]))
        return decoded

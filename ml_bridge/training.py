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
Train a model on bags without ROS, with the sample rule that inference uses.

The model module provides train(dataset, context). Without bags the dataset is None
and the module trains on its own data. This module must not import rclpy.
"""

import argparse
import importlib
import logging
from types import SimpleNamespace

from rosbag2_dataset.torch import BagDataset

from .spec import Decoder, load_spec, Renamed


def make_dataset(spec, bags, remap=None):
    """Return the samples of the bags; remap maps spec topics to bag topics."""
    names = {topic: (remap or {}).get(topic, topic) for topic in spec.topics}
    return BagDataset(list(bags), [names[t] for t in spec.topics], sync=spec.sync(names),
                      transform=Renamed(names, Decoder(spec)))


def _remap(arguments):
    pairs = [argument.split(':=', 1) for argument in arguments]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError('--remap takes spec_topic:=bag_topic')
    return dict(pairs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--model', required=True, help='module with train(dataset, context)')
    parser.add_argument('--bags', nargs='*', default=[])
    parser.add_argument('--remap', nargs='*', default=[], help='spec_topic:=bag_topic')
    parser.add_argument('--weights', default='')
    parser.add_argument('--out-dir', default='')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger('ml_bridge')
    spec = load_spec(args.spec)
    dataset = None
    if args.bags:
        dataset = make_dataset(spec, args.bags, _remap(args.remap))
        logger.info(f'{len(dataset)} samples from {len(args.bags)} bags')
    context = SimpleNamespace(spec=spec, logger=logger, weights=args.weights,
                              out_dir=args.out_dir, bags=args.bags)
    importlib.import_module(args.model).train(dataset, context)


if __name__ == '__main__':
    main()

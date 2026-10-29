# ml_bridge

Run a learning model on ROS topics, and train it on recorded bags with the same
samples. One spec names the input topics and how they align; the training dataset
and the live subscriptions both build samples with the
[rosbag2_dataset](https://github.com/nevlife/rosbag2_dataset) `SampleRule`, and
decode messages with the same code.

## Spec

```yaml
inputs:
  /camera/image: {type: sensor_msgs/msg/Image}
  /odom:
    type: nav_msgs/msg/Odometry
    tolerance_ms: 30          # nearest message within 30 ms of the reference
    optional: false           # true keeps the sample with None when beyond tolerance
    window: {every_ms: 500, past: 4, future: 0, tolerance_ms: 30}
reference: /camera/image      # default: first input
time: header                  # header (header.stamp) or log (bag log time, receive time live)
max_wait_ms: 100              # live: release a sample after this wait even if a topic lags
output: {topic: ~/output, type: vision_msgs/msg/Detection2DArray}
```

A sample is a dictionary: `'@time'` (reference clock), each input's
`(time, value)` or `None`, and `'<topic>@past'` / `'<topic>@future'` lists for
windows. Images are NumPy arrays (`HxW` or `HxWxC`): `bgr8`, `bgra8`, and `rgba8`
become RGB copies, other encodings keep their channel order as views (read-only
for bags). Other messages stay messages.

## Model module

```python
def load(context): ...            # optional; returns the state given to infer
def train(dataset, context): ...  # dataset: map-style samples of the bags, or None
def infer(state, sample): ...     # returns an output message or None
```

`context` has `spec`, `logger`, `weights`, `out_dir`, and `bags`. Batching,
shuffling, augmentation, and optimization stay in the model's own training code,
for example a PyTorch `DataLoader` over `dataset`.

## Run

```bash
ros2 run ml_bridge run --ros-args -p spec:=spec.yaml -p model:=my_models.detector \
  -p mode:=train -p bags:="['/data/run1']" -p out_dir:=/data/runs/detector
ros2 run ml_bridge run --ros-args -p spec:=spec.yaml -p model:=my_models.detector \
  -p mode:=infer -p weights:=/data/runs/detector/best.pt
```

Topic remapping (`-r /camera/image:=/my_camera/image_raw`) applies to both modes:
`train` reads the remapped topics from the bags, and samples keep the spec's names.

In `infer` mode the node subscribes best effort with short queues, runs the model
on the newest finished sample, and publishes the result on the output topic and
`~/input_info` (JSON: reference time, skipped samples, input times). Messages
without a stamp get the sample's reference time.

## License

MIT. See [LICENSE](LICENSE).

# ml_bridge

Run a learning model on ROS topics, and train it on recorded bags without ROS, with
the same samples. One spec names the input topics and how they align; the training
dataset and the live subscriptions both build samples with the
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
max_wait_ms: 100              # live: release a sample once any input is this much newer
output: {topic: ~/output, type: vision_msgs/msg/Detection2DArray}
```

A sample is a dictionary: `'@time'` (reference clock), each input's
`(time, value)` or `None`, and `'<topic>@past'` / `'<topic>@future'` lists for
windows. Images are NumPy arrays (`HxW` or `HxWxC`): `bgr8`, `bgra8`, and `rgba8`
become RGB copies, other encodings keep their channel order as views (read-only
for bags). Other messages stay messages.

## Model package

A model is one ROS 2 package with a spec, a launch file, and two modules:

```python
def train(dataset, context): ...  # training module, no ROS: dataset of the bags, or None
def load(context): ...            # inference module, optional; returns the state
def infer(state, sample): ...     # inference module: returns an output message or None
```

The training `context` has `spec`, `logger`, `weights`, `out_dir`, and `bags`; the
inference `context` has `spec`, `logger`, and `weights`. Batching, shuffling,
augmentation, and optimization stay in the model's own training code.

A launch file names the package and its topic arguments:

```python
from ml_bridge.launch import infer_launch


def generate_launch_description():
    return infer_launch('my_model_ros', {'camera': '/camera/image'})
```

## Run

```bash
python -m ml_bridge.training --spec spec.yaml --model my_model_ros.training \
  --bags /data/run1 --remap /camera/image:=/my_camera --out-dir /data/runs
ros2 launch my_model_ros infer.launch.py camera:=/my_camera weights:=/data/runs/best.pt
```

`ml_bridge.training` does not import rclpy. Put `rosbag2_dataset`, `ml_bridge`, and the
model package on `PYTHONPATH` to train on a machine without ROS.

The node subscribes best effort with short queues, runs the model on the newest
finished sample, and publishes the result on the output topic and `~/input_info` (JSON:
reference time, skipped samples, input times). Messages without a stamp get the
sample's reference time.

## License

MIT. See [LICENSE](LICENSE).

# ml_bridge

Run a learning model on ROS topics, and train it on recorded bags without ROS, with
the same samples. One spec names the input topics and how they align; the training
dataset and the live subscriptions both build samples with the
[rosbag2_dataset](https://github.com/nevlife/rosbag2_dataset) `SampleRule`, and
decode messages with the same code.

## Spec

```yaml
version: 1                    # raise when the inputs change; old weights are then refused
inputs:
  /camera/image: {type: sensor_msgs/msg/Image}
  /goal: {type: geometry_msgs/msg/PoseStamped, align: hold, tolerance_ms: 60000}
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

`align: hold` takes the last message at or before the reference time, no older than
`tolerance_ms`, for an input that arrives rarely. The default, `nearest`, can pick a later
message, so live samples wait for it.

Training writes the spec version to `ml_bridge_spec_version` in `--out-dir`. The node reads
that file from a folder above `weights` and refuses other versions. A model can keep a
training spec beside its inference spec, for a future window that only training needs; both
carry the same version.

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
finished sample in its own thread, because a callback that blocks loses messages of the
other inputs, and publishes the result on the output topic and `~/input_info` (JSON:
reference time, skipped samples, the time inference ended, the count of samples released
by `max_wait_ms` that can differ from a bag's, input times). Messages without a stamp get the
sample's reference time.

## Middleware

rmw_zenoh with shared memory is the baseline for inference. The node does not choose an RMW,
so set it in every process and start a router:

```bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true'
ros2 run rmw_zenoh_cpp rmw_zenohd
```

Cyclone DDS is not supported for several large image topics: best-effort images are lost
under bursts, even with a larger socket receive buffer.

## flux input

`transport:=flux` receives `sensor_msgs/msg/Image` inputs over flux
channels of the same name; other inputs stay ROS topics. ROS remains the default, and flux
is imported only with this setting. Source the flux workspace after this one.

| Spec | Frame | `flux_device` |
| --- | --- | --- |
| One input, no window | The model reads the slot during `infer`, no copy | `cpu`, or `cuda` for a tensor on the GPU |
| Anything else | One copy in the callback: a held view would stop the channel | `cpu` |

Without a copy, `infer` must not keep the pixels: a kept view stops the channel, and the
node logs an error. Copy them first when a library holds on to its input.

A model that takes `flux_device:=cuda` handles an `HxWxC` uint8 tensor in `sample[topic][1]`.
A CPU publisher reaches a discrete GPU with one
copy made by flux; on an integrated GPU there is none.

## License

MIT. See [LICENSE](LICENSE).

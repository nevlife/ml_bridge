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

"""Launch description shared by model packages: the node with the package's spec."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def infer_launch(package, inputs):
    """
    Run the package's inference module on its config/spec.yaml.

    inputs maps a launch argument to the spec topic that it remaps.
    """
    arguments = [DeclareLaunchArgument('weights', default_value=''),
                 DeclareLaunchArgument('transport', default_value='ros'),
                 DeclareLaunchArgument('flux_device', default_value='cpu')]
    arguments += [DeclareLaunchArgument(name, default_value=topic)
                  for name, topic in inputs.items()]
    spec = str(Path(get_package_share_directory(package), 'config', 'spec.yaml'))
    node = Node(package='ml_bridge', executable='run', name=package, output='screen',
                parameters=[{'spec': spec, 'model': f'{package}.inference',
                             'weights': LaunchConfiguration('weights'),
                             'transport': LaunchConfiguration('transport'),
                             'flux_device': LaunchConfiguration('flux_device')}],
                remappings=[(topic, LaunchConfiguration(name))
                            for name, topic in inputs.items()])
    return LaunchDescription(arguments + [node])

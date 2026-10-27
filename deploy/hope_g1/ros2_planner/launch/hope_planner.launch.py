"""Launch the planner implementation and parameters owned by this deploy repo."""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
  params_file = (
    Path(get_package_share_directory("hope_g1_deploy_planner"))
    / "config"
    / "hope_planner_optimized_2.yaml"
  )
  return LaunchDescription(
    [
      Node(
        package="hope_g1_deploy_planner",
        executable="hope_g1_deploy_planner_node",
        name="hope_planner_optimized_2",
        output="screen",
        parameters=[str(params_file)],
      )
    ]
  )

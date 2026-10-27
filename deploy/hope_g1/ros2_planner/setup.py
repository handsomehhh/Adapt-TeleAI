from glob import glob
import os

from setuptools import setup


package_name = "hope_g1_deploy_planner"

setup(
  name=package_name,
  version="0.1.0",
  packages=[package_name],
  data_files=[
    (
      "share/ament_index/resource_index/packages",
      ["resource/" + package_name],
    ),
    ("share/" + package_name, ["package.xml"]),
    (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
  ],
  install_requires=["setuptools"],
  zip_safe=True,
  maintainer="deploy_g1_pingpong",
  maintainer_email="lxz@example.com",
  description="Standalone ROS 2 wrapper for the deploy-owned A30808 planner.",
  license="Apache-2.0",
  entry_points={
    "console_scripts": [
      "hope_g1_deploy_planner_node = hope_g1_deploy_planner.planner_node:main",
    ],
  },
)

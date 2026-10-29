import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/lxz/Adapt-Deploy/deploy/ros2_install/hope_g1_deploy_planner'

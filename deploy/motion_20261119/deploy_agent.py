"""Parallel dual deployment mode reproducing real_20261119_095258."""
import argparse
import hashlib
import json
import signal
from pathlib import Path
import torch
import yaml
from .reference.hope_g1.configuration import load_deployment
from .reference.hope_g1.controller import HopeController
from .transport import MODE, MotionMocap, RuntimeObserver

PACKAGE=Path(__file__).resolve().parent
CONFIG=PACKAGE/'configs'


def load():
    task=yaml.safe_load((CONFIG/'task.yaml').read_text());task.pop('task')
    for key in ('checkpoint','motion_dir'):
        task[key]=str((CONFIG/task[key]).resolve())
    options=yaml.safe_load((CONFIG/'rally.yaml').read_text())
    options['hope_planner_config']=str(CONFIG/'planner.yaml')
    runtime=yaml.safe_load((CONFIG/'real.yaml').read_text())
    runtime['stable_policy']['asset_dir']=str((CONFIG/runtime['stable_policy']['asset_dir']).resolve())
    # Check every pinned artifact, including motion ordering metadata and the
    # actual robot/AMP assets, before constructing the actor or opening DDS.
    manifest=json.loads((PACKAGE/'ASSETS.json').read_text())
    present={str(p.relative_to(PACKAGE)) for p in (PACKAGE/'artifacts').rglob('*')
             if p.is_file() and '__pycache__' not in p.parts}
    if present != set(manifest['files']):
        raise ValueError('Historical asset inventory differs from ASSETS.json')
    for name,expected in manifest['files'].items():
        p=PACKAGE/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Missing or changed historical asset: {p}')
    deployment=load_deployment(**task,ball_config=options['ball'],world_config=options['world'])
    return deployment,runtime,options


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env',choices=['check','real'],default='check')
    parser.add_argument('--mocap-config',type=Path)
    parser.add_argument('--network-interface')
    parser.add_argument('--duration',type=float,default=6000.)
    parser.add_argument('--enable-motors',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--torch-threads',type=int,default=1)
    parser.add_argument('--live-record-only',action='store_true')
    parser.add_argument('--live-view',action='store_true')
    parser.add_argument('--viewer',choices=['web'],default='web')
    parser.add_argument('--web-host',default='0.0.0.0');parser.add_argument('--web-port',type=int,default=8081)
    args=parser.parse_args()
    if args.live_view:parser.error('Use scripts/run.py viewer for the shared ball/table/P1/P2 window')
    if args.torch_threads<1 or not 0<args.duration<float('inf'):parser.error('Invalid threads/duration')
    if args.enable_motors and args.env!='real':parser.error('Motors require real mode')
    torch.set_num_threads(args.torch_threads)
    deployment,runtime,options=load()
    controller=HopeController(deployment,runtime,options,home_index=0,allow=list(range(len(deployment.motions))),selection='auto')
    print(json.dumps(dict(mode=MODE,source_run='real_20261119_095258',**deployment.report()),indent=2))
    print(controller.describe())
    if args.env=='check':return
    if args.mocap_config is None or args.output is None:parser.error('real requires generated mocap config and output')
    cfg=yaml.safe_load(args.mocap_config.read_text());side=cfg.get('robot_id')
    if side not in ('P1','P2') or cfg.get('deployment_mode')!=MODE or cfg.get('ball_source')!='shared':
        parser.error('Use the motion_20261119 dual launcher and generated config')
    ns=Path('/run/netns')/f'g1_{side.lower()}'
    if not ns.exists() or ns.stat().st_ino!=Path('/proc/self/ns/net').stat().st_ino:
        parser.error(f'Run the complete actor inside {ns.name}')
    from .reference.backends.sdk import require_sdk,network_interfaces
    require_sdk()
    if args.network_interface not in network_interfaces():parser.error('Configured robot interface is absent')
    runtime.update(network_interface=args.network_interface,robot_id=side)
    args.output.mkdir(parents=True,exist_ok=True)
    resolved=args.output/'rally_config.yaml';resolved.write_text(yaml.safe_dump(options,sort_keys=False))
    (args.output/'real_config.yaml').write_text(yaml.safe_dump(runtime,sort_keys=False))
    (args.output/'source.json').write_text((PACKAGE/'SOURCE.json').read_text())
    viewer=None
    if args.live_view or args.live_record_only:
        from .telemetry import TelemetryRecorder
        viewer=TelemetryRecorder(deployment,args.output,side)
    observer=RuntimeObserver(cfg,viewer)
    from .reference.backends.unitree import run_control_loop
    def shutdown(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,shutdown)
    try:
        run_control_loop(deployment,runtime,args.mocap_config,enable_motors=args.enable_motors,
            motion_index=0,motion_select='auto',motion_allow=list(range(len(deployment.motions))),
            duration=args.duration,output=args.output,simulation_config=resolved,controller_factory=HopeController,
            mocap_factory=MotionMocap,visualization=observer)
    finally:observer.close()


if __name__=='__main__':main()

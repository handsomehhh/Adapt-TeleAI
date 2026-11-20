import json
from pathlib import Path
import socket
import time
import pytest
import yaml
from deploy.motion_20261119.transport import MotionMocap,RuntimeObserver


def config():
    cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/mocap.yaml').read_text())
    cfg.update(robot_id='P2',deployment_mode='motion_20261119',ball_source='shared')
    cfg['udp']=dict(host='127.0.0.1',port=0)
    cfg['planner_feedback']=dict(host='127.0.0.1',port=9)
    return cfg


def test_receiver_identity_order_age_and_hold_preserve_root_conditioning():
    cfg=config();receiver=MotionMocap(cfg)
    try:
        packet=dict(mode='motion_20261119',robot_id='P2',kind='root',frame='hope_table',stamp=100.,position=[.2,-.3,.8],quaternion_wxyz=[1.,0.,0.,0.])
        receiver.accept(packet,wall_time=100.,monotonic_time=10.)
        assert receiver.root.get(10.) is not None
        assert receiver.root.conditioner is not None
        with pytest.raises(ValueError):receiver.accept(dict(packet,robot_id='P1'),wall_time=100.,monotonic_time=10.)
        with pytest.raises(ValueError):receiver.accept(dict(packet,mode='hitter'),wall_time=100.,monotonic_time=10.)
        with pytest.raises(ValueError):receiver.accept(dict(packet,kind='ball'),wall_time=100.,monotonic_time=10.)
        original=json.loads(Path(__file__).with_name('recorded_commands.json').read_text())['packets'][0]['packet']
        command=dict(original,mode='motion_20261119',robot_id='P2',stamp=100.,bridge_monotonic=10.)
        receiver.accept(command,wall_time=100.,monotonic_time=10.)
        assert receiver.latest_planner_command()[0]['center_index']==original['center_index']
        assert len(receiver.latest_planner_command()[0]['commands'])==len(original['commands'])
        receiver.accept(dict(mode='motion_20261119',robot_id='P2',kind='planner_hold',emitted_monotonic=10.01),wall_time=100.01,monotonic_time=10.01)
        assert receiver.latest_planner_command()==(None,None)
        receiver.accept(command,wall_time=100.02,monotonic_time=10.02)
        assert receiver.latest_planner_command()==(None,None)
        with pytest.raises(ValueError):receiver.accept(dict(command,stamp=101.),wall_time=102.,monotonic_time=12.)
    finally:receiver.close()


def test_status_expires_independently_and_never_enables_in_positioning():
    sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);sock.bind(('127.0.0.1',0));sock.settimeout(.2)
    cfg=config();cfg['planner_feedback']['port']=sock.getsockname()[1]
    observer=RuntimeObserver(cfg)
    try:
        row=dict(motor_output_enabled=True,state='POSITIONING',root_age=.01,dds_state_age=.01)
        observer.publish('state',row)
        assert not json.loads(sock.recv(8192))['enabled']
        observer.publish('state',dict(row,state='ARMED'))
        assert json.loads(sock.recv(8192))['enabled']
        observer.publish('state',dict(row,state='EXECUTE',root_age=.3))
        assert not json.loads(sock.recv(8192))['enabled']
    finally:observer.close();sock.close()


def test_telemetry_for_unified_viewer_without_extra_estimator(tmp_path):
    from deploy.motion_20261119.deploy_agent import load
    from deploy.motion_20261119.telemetry import TelemetryRecorder
    deployment,_,_=load()
    recorder=TelemetryRecorder(deployment,tmp_path,'P2')
    try:
        recorder.publish('state',dict(q=[0.]*29,root_pos=[0.,-.7,.9],root_quat=[1.,0.,0.,0.],root_age=.01,dds_state_age=.01,state='WAIT'))
        target=tmp_path/'visualization/latest_state.json';deadline=time.monotonic()+5
        while not target.exists() and time.monotonic()<deadline:time.sleep(.02)
        snapshot=json.loads(target.read_text())
        assert snapshot['robot_id']=='P2' and snapshot['schema']=='dual_robot_view_v1'
        metadata=json.loads((tmp_path/'visualization/metadata.json').read_text())
        assert len(metadata['joint_names'])==29 and metadata['mode']=='motion_20261119'
    finally:recorder.close()
    assert recorder.process.exitcode==0

import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
torch.set_num_threads(1)
from deploy.motion_20261119.deploy_agent import load
from deploy.motion_20261119.reference.hope_g1.motion_executor import HopeActorController,RacketGoal
from deploy.motion_20261119.reference.g1_pingpong.motions import MotionLibrary,RebasedMotion
from deploy.motion_20261119.reference.g1_pingpong.core import rotation,quaternion_multiply
from deploy.motion_20261119.reference.backends.mujoco import MujocoBackend

def test_recorded_observations_actions_and_targets():
    fixture=json.loads(Path(__file__).with_name('recorded_samples.json').read_text())
    dep,_,_=load();library=MotionLibrary(dep)
    c=HopeActorController(dep.settings,dep.actor)
    errors=dict(obs=0.,action=0.,target=0.);worst=None
    backend=MujocoBackend(dep.model,dep.settings)
    for row in fixture['rows']:
     home=row['home'];r=rotation(home['quaternion_wxyz']);translation=np.array(home['translation'])
     motion=library.motions[row['motion_index']];_,p,q=motion.sample(0)
     ref=RebasedMotion(motion,r@p+translation,quaternion_multiply(home['quaternion_wxyz'],q))
     if row['state']=='EXECUTE' and row['motion_index']!=0:
      anchor=row['execution_anchor'];ref=RebasedMotion(motion,np.array(anchor['position']),np.array(anchor['quaternion']))
     state=SimpleNamespace(**{k:np.array(row[k]) for k in ['q','dq','root_pos','root_quat','anchor_pos','anchor_quat','gyro','root_velocity_world']})
     i=backend.free_qpos
     backend.data.qpos[i:i+3]=state.root_pos
     backend.data.qpos[i+3:i+7]=state.root_quat
     backend.data.qpos[backend.qpos]=state.q
     backend.data.qvel[backend.qvel]=state.dq
     backend.forward_kinematics()
     measured=backend.state()
     for key in ('anchor_pos','racket_pos','racket_normal'):
      np.testing.assert_allclose(getattr(measured,key),row[key],atol=1e-6,rtol=1e-6)
     goal=RacketGoal(*[np.array(row[k]) for k in ['goal_position','goal_velocity','goal_normal','goal_base_xy']])
     c.previous_joint=np.array(row['previous_action'],dtype=np.float32)
     target=c.step(state,ref,row['reference_frame'],goal,row['time_to_strike'],row['time'])
     for key,new,old in [('obs',c.debug['actor_obs'],row['actor_obs']),('action',c.debug['joint_action'],row['joint_action']),('target',target,row['policy_target_q'])]:
      err=float(np.max(np.abs(np.array(new)-old)));errors[key]=max(errors[key],err)
      if key=='obs' and err>1e-5:print('bad row',row['time'],row['state'],row['motion_index'],err,'index',np.argmax(np.abs(new-old)))
    print('PARITY',len(fixture['rows']),errors)
    assert len(fixture['rows']) == 70
    assert errors['obs'] < 1e-6
    assert errors['action'] < 2e-5
    assert errors['target'] < 2e-5


def test_recorded_motion_selection_and_live_updates():
    from deploy.motion_20261119.reference.hope_g1.controller import HopeController
    fixture=json.loads(Path(__file__).with_name('recorded_samples.json').read_text())
    commands=json.loads(Path(__file__).with_name('recorded_commands.json').read_text())
    dep,runtime,options=load()
    controller=HopeController(dep,runtime,options,home_index=0,allow=list(range(162)),selection='auto')
    def state(row):
        return SimpleNamespace(**{k:np.array(row[k]) for k in ['q','dq','root_pos','root_quat','anchor_pos','anchor_quat','gyro','root_velocity_world']})
    controller.executor.prepare(state(fixture['rows'][0]),fixture['rows'][0]['time'])
    rows={row['time']:row for row in fixture['rows']}
    packets={d['packet']['stamp']:d for d in commands['packets']}
    for i,e in enumerate(commands['events']):
        row=rows[e['time']];raw=e.get('raw_command',e.get('raw_racket_command'));packet=packets[raw['planner_stamp']]
        controller.update_external_planner_command(packet['packet'],packet['arrival_time'])
        batch=controller.latest_planner_command(row['time'])
        assert batch is not None
        if i==0:
            choice=controller.executor.choose(state(row),batch,row['time'])
            assert choice is not None
            result=controller.executor.begin(choice,row['time'])
        else:
            result=controller.executor.update_command(state(row),batch,row['time'])
        assert result is not None
        assert result.event['motion_index']==e['motion_index']
        assert result.event['command_index']==e['command_index']
        assert controller.executor.frame==row['reference_frame']
        np.testing.assert_allclose(result.command.position,raw['position'],atol=1e-10)
        assert abs(result.command.deadline-e['deadline'])<1e-8
        controller.executor.frame+=1

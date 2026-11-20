"""Bounded telemetry recorder for the existing unified dual viewer, without a fit."""
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import queue
import signal
import time
import traceback
from deploy.io.live_visualization import STATE_KEYS
from deploy.io.viewer_snapshot import boot_id, write_snapshot
from deploy.io.diagnostics import json_default


def worker(directory, side, boot, events, stop, ready):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    path=Path(directory)
    try:
        with (path/'events.jsonl').open('w',buffering=1) as stream:
            ready.set()
            while True:
                try:event=events.get(timeout=.02)
                except queue.Empty:
                    if stop.is_set():break
                    continue
                stream.write(json.dumps(event,default=json_default,allow_nan=False)+'\n')
                if event['kind']=='state':
                    write_snapshot(path/'latest_state.json',dict(schema='dual_robot_view_v1',robot_id=side,
                        boot_id=boot,time=event['time'],data=event['data']))
    except Exception:
        (path/'worker_error.txt').write_text(traceback.format_exc())
        raise


class TelemetryRecorder:
    def __init__(self,deployment,output,side):
        import mujoco
        self.directory=Path(output)/'visualization';self.directory.mkdir(parents=True,exist_ok=False)
        model=self.directory/'scene.mjb';mujoco.mj_saveModel(deployment.model,str(model))
        boot=boot_id()
        metadata=dict(version=2,robot_id=side,boot_id=boot,mode='motion_20261119',world_frame='hope_table',
            model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
            joint_names=deployment.settings['robot']['joint_names'],world_translation=deployment.settings['simulation_world_translation'],
            checkpoint=str(deployment.checkpoint),mujoco_version=mujoco.__version__)
        (self.directory/'metadata.json').write_text(json.dumps(metadata,indent=2))
        ctx=mp.get_context('spawn');self.events=ctx.Queue(maxsize=2048);self.stop=ctx.Event();ready=ctx.Event()
        self.process=ctx.Process(target=worker,args=(str(self.directory),side,boot,self.events,self.stop,ready),daemon=True)
        self.process.start();self.accepted=self.dropped=0
        deadline=time.monotonic()+15.
        while not ready.wait(.1):
            if not self.process.is_alive() or time.monotonic()>deadline:
                self.close();raise RuntimeError('Motion telemetry worker failed to start')

    def publish(self,kind,data):
        row={k:data[k] for k in STATE_KEYS if k in data} if kind=='state' else dict(data)
        try:
            self.events.put_nowait(dict(kind=kind,time=time.monotonic(),data=row));self.accepted+=1
        except (queue.Full,ValueError,OSError):self.dropped+=1

    def close(self):
        self.stop.set();self.process.join(timeout=5)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=2)
        self.events.close();self.events.cancel_join_thread()
        (self.directory/'transport_summary.json').write_text(json.dumps(dict(accepted=self.accepted,dropped=self.dropped,worker_exitcode=self.process.exitcode)))

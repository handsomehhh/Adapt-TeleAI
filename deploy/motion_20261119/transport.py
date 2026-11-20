"""Mode/side checked shared estimates, with the original pose conditioning/decoder."""
import json
import math
import socket
import time
from .reference.io.transport import UDPMocap
from deploy.io.shared_ball import SharedBallCache

MODE='motion_20261119'


class MotionMocap(UDPMocap):
    def __init__(self, config):
        self.shared = SharedBallCache(config)
        self.last_command_stamp = -float('inf')
        self.last_hold = -float('inf')
        super().__init__(config)
        # The original socket thread never receives raw ball positions in this
        # mode; its original root cache and command-array decoder are retained.
        self.ball = self.shared

    def accept(self, message, *, wall_time=None, monotonic_time=None):
        wall_time=time.time() if wall_time is None else wall_time
        monotonic_time=time.monotonic() if monotonic_time is None else monotonic_time
        if message.get('mode') != MODE or message.get('robot_id') != self.cfg['robot_id']:
            raise ValueError('Motion mode/robot mismatch')
        if message.get('kind')=='ball_estimate':
            self.shared.accept(message,wall_time=wall_time,monotonic_time=monotonic_time)
            self.accepted+=1
            return
        if message.get('kind')=='planner_hold':
            emitted=message['emitted_monotonic']
            if not math.isfinite(emitted) or not 0 <= monotonic_time-emitted <= .15:return
            if emitted<=self.last_hold:return
            self.last_hold=emitted
            # Expire future updates. The already accepted swing remains owned
            # by the historical controller and completes its original follow-through.
            with self._planner_lock:self._planner_packet=self._planner_received=None
            return
        if message.get('kind')=='racket_command_array':
            stamp=message['stamp']
            bridge=message['bridge_monotonic']
            if not math.isfinite(stamp) or not 0 <= wall_time-stamp <= .1:
                raise ValueError('Expired motion command array')
            if stamp<=self.last_command_stamp or bridge<=self.last_hold:return
            super().accept(message,wall_time=wall_time,monotonic_time=monotonic_time)
            self.last_command_stamp=stamp
            return
        if message.get('kind')!='root':raise ValueError('Motion mode accepts shared ball only')
        super().accept(message,wall_time=wall_time,monotonic_time=monotonic_time)


class RuntimeObserver:
    def __init__(self, cfg, viewer=None):
        self.cfg,self.viewer=cfg,viewer
        self.sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);self.sock.setblocking(False)
        feedback=cfg['planner_feedback']
        self.destination=(feedback['host'],feedback['port'])

    def feedback(self, enabled):
        packet=dict(mode=MODE,robot_id=self.cfg['robot_id'],generated_monotonic=time.monotonic(),enabled=bool(enabled))
        try:self.sock.sendto(json.dumps(packet).encode(),self.destination)
        except (OSError,BlockingIOError):pass  # Missing feedback expires only this side.

    def publish(self, kind, row):
        if kind=='state':
            enabled=(row.get('motor_output_enabled') and row.get('state') in ('ARMED','EXECUTE','RECOVER')
                and row.get('root_age',float('inf'))<=.2 and row.get('dds_state_age',float('inf'))<=.3
                and not row.get('root_pose_frozen',False))
            self.feedback(enabled)
        if self.viewer is not None:self.viewer.publish(kind,row)

    def close(self):
        self.feedback(False);self.sock.close()
        if self.viewer is not None:self.viewer.close()

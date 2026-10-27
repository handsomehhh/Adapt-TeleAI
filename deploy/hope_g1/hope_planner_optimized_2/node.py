"""ROS 2 node wrapping the HOPE 7-DOF racket planner.

Subscribes to ball position from the mocap `/poses` stream (the avatar_pro relay
on the Avatar Pro / Chingmu VRPN path) and publishes the desired racket state on
`/racket/command` as the shared
hope_msgs/RacketCommand. Diagnostics are published at 10 Hz.

Per HOPE rules the racket pose is never measured by motion capture; the
humanoid must achieve the commanded racket state via its own forward
kinematics. See HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md.
"""

import csv
import os
from pathlib import Path

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.executors import ExternalShutdownException

from hope_msgs.msg import RacketCommand

from .constants import BallPhysics, PlannerConfig
from .planner import HOPEPlanner


use_print = False

class PlannerPredictionRecorder:
    """Append planner-owned command and trajectory records for exact replay."""

    def __init__(self, directory: str, config: PlannerConfig, physics: BallPhysics):
        self._commands = None
        self._points = None
        self._command_writer = None
        self._point_writer = None
        self._command_seq = 0
        self._pending_flushes = 0
        if not directory:
            return
        path = Path(directory).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        self._commands = (path / "planner_prediction_commands.csv").open(
            "w", newline="", buffering=65536
        )
        self._points = (path / "planner_prediction_points.csv").open(
            "w", newline="", buffering=65536
        )
        self._command_writer = csv.writer(self._commands)
        self._point_writer = csv.writer(self._points)
        self._command_writer.writerow([
            "planner_command_seq", "publish_ns", "source_ball_time", "estimate_time",
            "source_ball_x", "source_ball_y", "source_ball_z",
            "estimate_x", "estimate_y", "estimate_z",
            "estimate_vx", "estimate_vy", "estimate_vz",
            "intercept_x", "intercept_y", "intercept_z", "strike_time",
            "x_hit", "drag_k", "restitution_h", "restitution_v", "integration_dt_s",
        ])
        self._point_writer.writerow([
            "planner_command_seq", "prediction_time", "x", "y", "z",
        ])
        self._commands.flush()
        self._points.flush()

    @property
    def active(self) -> bool:
        return self._command_writer is not None

    def record(self, publish_ns: int, source_ball_time: float, source_ball: np.ndarray,
               command, planner: HOPEPlanner) -> None:
        if not self.active:
            return
        estimate = planner.latest_estimate
        velocity = planner.latest_velocity
        samples = planner.latest_prediction_samples
        if estimate is None or velocity is None or samples is None:
            return
        self._command_seq += 1
        seq = self._command_seq
        config, physics = planner.config, planner.physics
        self._command_writer.writerow([
            seq, publish_ns, source_ball_time, source_ball_time,
            *source_ball, *estimate, *velocity, *command.p_intercept, command.t_strike,
            config.x_hit, physics.k, physics.C_h, physics.C_v, config.dt_integrate,
        ])
        self._point_writer.writerows(
            (seq, float(sample[0]), float(sample[1]), float(sample[2]), float(sample[3]))
            for sample in samples
        )
        # Buffer normal high-rate writes, but make a recent planner record
        # durable at least every 20 valid publications if the process is stopped.
        self._pending_flushes += 1
        if self._pending_flushes >= 20:
            self._commands.flush()
            self._points.flush()
            self._pending_flushes = 0

    def close(self) -> None:
        for handle in (self._commands, self._points):
            if handle is not None:
                handle.flush()
                handle.close()
        self._commands = self._points = None


class HOPEPlannerNode(Node):
    """ROS 2 node for the HOPE model-based planner."""

    def __init__(self):
        super().__init__("hope_planner_optimized_2")

        self.declare_parameter("ball_rigid_body_name", "pingpong_ball")
        self.declare_parameter("ball_pose_index", 0)
        self.declare_parameter("x_hit", 0.0)
        self.declare_parameter("target_land_x", 2.055)
        self.declare_parameter("target_land_y", -0.7625)
        self.declare_parameter("delta_t_flight", 0.5)
        self.declare_parameter("time_to_strike_invalid_x", 0.0)
        self.declare_parameter("drag_k", 0.5)
        self.declare_parameter("restitution_h", 0.75)
        self.declare_parameter("restitution_v", 0.85)
        self.declare_parameter("restitution_racket", 0.88)
        # Stage 2 (incoming-ball trajectory prediction) integration step.
        self.declare_parameter("dt_integrate", 0.001)
        self.declare_parameter("stage3_dt_integrate", 0.001)
        self.declare_parameter("outgoing_velocity_tolerance_m", 1e-4)
        # Inclusive interception workspace. Commands outside this box are
        # filtered by HOPEPlanner before this node can publish them.
        self.declare_parameter("publish_intercept_x_min", 0.0)
        self.declare_parameter("publish_intercept_x_max", 0.0)
        self.declare_parameter("publish_intercept_y_min", 0.0)
        self.declare_parameter("publish_intercept_y_max", 1.5)
        self.declare_parameter("publish_intercept_z_min", 0.2)
        self.declare_parameter("publish_intercept_z_max", 0.6)
        # Empty disables recording.  The E2E launcher exports this default
        # before starting planner, but a real robot can set -p record_dir:=... .
        self.declare_parameter("record_dir", os.environ.get("HOPE_PLANNER_TRACE_DIR", ""))

        self._ball_index = int(self.get_parameter("ball_pose_index").value)

        config = PlannerConfig(
            x_hit=self.get_parameter("x_hit").value,
            target_land=np.array([
                self.get_parameter("target_land_x").value,
                self.get_parameter("target_land_y").value,
                0.0,
            ]),
            delta_t_flight=self.get_parameter("delta_t_flight").value,
            C_r=self.get_parameter("restitution_racket").value,
            dt_integrate=self.get_parameter("dt_integrate").value,
            stage3_dt_integrate=self.get_parameter("stage3_dt_integrate").value,
            outgoing_velocity_tolerance_m=self.get_parameter(
                "outgoing_velocity_tolerance_m"
            ).value,
            time_to_strike_invalid_x=self.get_parameter("time_to_strike_invalid_x").value,
            publish_intercept_x_min=self.get_parameter("publish_intercept_x_min").value,
            publish_intercept_x_max=self.get_parameter("publish_intercept_x_max").value,
            publish_intercept_y_min=self.get_parameter("publish_intercept_y_min").value,
            publish_intercept_y_max=self.get_parameter("publish_intercept_y_max").value,
            publish_intercept_z_min=self.get_parameter("publish_intercept_z_min").value,
            publish_intercept_z_max=self.get_parameter("publish_intercept_z_max").value,
        )
        publish_bounds = (
            config.publish_intercept_x_min, config.publish_intercept_x_max,
            config.publish_intercept_y_min, config.publish_intercept_y_max,
            config.publish_intercept_z_min, config.publish_intercept_z_max,
        )
        if (not np.all(np.isfinite(publish_bounds))
                or config.publish_intercept_x_min > config.publish_intercept_x_max
                or config.publish_intercept_y_min > config.publish_intercept_y_max
                or config.publish_intercept_z_min > config.publish_intercept_z_max):
            raise ValueError(
                "publish_intercept bounds must be finite and have min <= max for every axis"
            )
        physics = BallPhysics(
            k=self.get_parameter("drag_k").value,
            C_h=self.get_parameter("restitution_h").value,
            C_v=self.get_parameter("restitution_v").value,
        )

        self.planner = HOPEPlanner(physics=physics, config=config)
        self._prediction_recorder = PlannerPredictionRecorder(
            str(self.get_parameter("record_dir").value), config, physics
        )

        # Best-effort, depth-1 QoS for high-rate mocap topics (REP-2003 sensor style).
        mocap_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # Control setpoints should favor delivery over freshest-only sensor
        # semantics. Keep this separate from the high-rate mocap subscription.
        command_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(PoseArray, "/poses", self._poses_cb, mocap_qos)
        self.cmd_pub = self.create_publisher(RacketCommand, "/racket/command", command_qos)
        self.diag_pub = self.create_publisher(DiagnosticArray, "/planner/diagnostics", 1)

        # Diagnostics at 10 Hz.
        self._n_received = 0
        self._n_valid = 0
        self._last_valid = False
        self._last_tts = float("nan")
        self.create_timer(0.1, self._publish_diagnostics)

        self.get_logger().info(
            f"HOPE planner started - x_hit={config.x_hit:.2f}, "
            f"target={config.target_land}, ball_pose_index={self._ball_index}, "
            "publish_intercept_box="
            f"x=[{config.publish_intercept_x_min:.3f},{config.publish_intercept_x_max:.3f}] "
            f"y=[{config.publish_intercept_y_min:.3f},{config.publish_intercept_y_max:.3f}] "
            f"z=[{config.publish_intercept_z_min:.3f},{config.publish_intercept_z_max:.3f}]"
        )
        if self._prediction_recorder.active:
            self.get_logger().info(
                f"planner prediction trace enabled: {self.get_parameter('record_dir').value}"
            )

    def _poses_cb(self, msg: PoseArray) -> None:
        self._n_received += 1
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if len(msg.poses) <= self._ball_index:
            self.planner.note_input_not_pushed("pose_array_missing_ball_index")
            self._last_valid = False
            self._last_tts = float("nan")
            return

        # NOTE: PoseArray carries no names. Configure ball_pose_index to match
        # the ball's slot in the /poses ordering (the avatar_pro relay puts the
        # ball first), or swap this for a /tf lookup keyed on ball_rigid_body_name.
        pose = msg.poses[self._ball_index]
        p_ball = np.array([pose.position.x, pose.position.y, pose.position.z])
        if not np.isfinite(t) or not np.all(np.isfinite(p_ball)):
            self.planner.note_input_not_pushed("nonfinite_ball_timestamp_or_position")
            self._last_valid = False
            self._last_tts = float("nan")
            return

        cmd = self.planner.update(
            t,
            p_ball,
            collect_prediction_samples=self._prediction_recorder.active,
        )
        if cmd is None:
            # print("cmd is None")
            self._last_valid = False
            self._last_tts = float("nan")
            return

        self._last_valid = cmd.valid
        tts = self.planner.time_to_strike
        self._last_tts = tts if tts is not None else float("nan")
        if cmd.valid:
            self._n_valid += 1

        out = RacketCommand()
        # The command header is its own publication time, not the timestamp of
        # the ball sample that informed it.  This lets a downstream trace
        # distinguish planner publication from deploy receipt.  The trajectory
        # fit above still uses the source ball timestamp ``t``.
        out.header.frame_id = "world"
        out.position.x = float(cmd.p_intercept[0])
        out.position.y = float(cmd.p_intercept[1])
        out.position.z = float(cmd.p_intercept[2])
        out.velocity.x = float(cmd.v_racket[0])
        out.velocity.y = float(cmd.v_racket[1])
        out.velocity.z = float(cmd.v_racket[2])
        # RacketCommand.normal carries the unit face normal (Vector3), not an
        # orientation. IK-based controllers that need a full quaternion can call
        # quaternion_utils.normal_to_quaternion(cmd.n_racket, constrain_up=True).
        out.normal.x = float(cmd.n_racket[0])
        out.normal.y = float(cmd.n_racket[1])
        out.normal.z = float(cmd.n_racket[2])
        out.strike_time = float(cmd.t_strike)
        out.time_to_strike = float(self._last_tts)
        out.ball_velocity_outgoing.x = float(cmd.v_ball_outgoing[0])
        out.ball_velocity_outgoing.y = float(cmd.v_ball_outgoing[1])
        out.ball_velocity_outgoing.z = float(cmd.v_ball_outgoing[2])
        out.valid = bool(cmd.valid)
        out.clears_net = bool(cmd.clears_net)
        out.bypasses_net_posts = bool(cmd.bypasses_net_posts)
        out.predicted_bounces = int(cmd.num_bounces)
        # Stamp immediately before handing the command to ROS, rather than at
        # the beginning of message construction.
        out.header.stamp = self.get_clock().now().to_msg()
        # Preserve the exact /poses (latest ball) timestamp consumed by this
        # planning update.  Deploy tracing records it alongside publication and
        # trigger time so command end-to-end latency is directly measurable.
        out.source_ball_time = float(t)
        # Publish the matching raw ball observation in the same world frame as
        # the racket command.  Keep this distinct from the estimator state used
        # internally by the planner.
        out.source_ball_position.x = float(p_ball[0])
        out.source_ball_position.y = float(p_ball[1])
        out.source_ball_position.z = float(p_ball[2])
        out.stage1_fit_rmse_m = float(cmd.stage1_fit_rmse_m)
        if cmd.valid:
            publish_ns = int(out.header.stamp.sec) * 1_000_000_000 + int(out.header.stamp.nanosec)
            self._prediction_recorder.record(publish_ns, float(t), p_ball, cmd, self.planner)
            if use_print:
                print("---- valid command"," source_ball_position: ", p_ball, "pred: ",cmd.p_intercept  , " st: ", out.strike_time, " tts: ", out.time_to_strike,)
        # else:
            # print("---- invalid command")
        self.cmd_pub.publish(out)

    def destroy_node(self):
        self._prediction_recorder.close()
        return super().destroy_node()

    def _publish_diagnostics(self) -> None:
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        status = DiagnosticStatus()
        status.name = "hope_planner_optimized_2"
        status.hardware_id = "hope_planner_optimized_2"
        if self._n_received == 0:
            status.level = DiagnosticStatus.WARN
            status.message = "no /poses received yet"
        elif self._last_valid:
            status.level = DiagnosticStatus.OK
            status.message = "valid racket command"
        else:
            status.level = DiagnosticStatus.OK
            status.message = "running; no valid strike"
        frame = self.planner.last_update_diagnostics
        estimated_vx = frame["estimated_vx_m_s"]
        def diagnostic_number(key: str) -> str:
            value = frame[key]
            return "nan" if value is None else f"{float(value):.4f}"

        candidate_valid = frame["candidate_valid"]
        status.values = [
            KeyValue(key="poses_received", value=str(self._n_received)),
            KeyValue(key="valid_commands", value=str(self._n_valid)),
            KeyValue(key="last_valid", value=str(self._last_valid)),
            KeyValue(key="time_to_strike_s", value=f"{self._last_tts:.4f}"),
            KeyValue(key="buffer_points", value=str(frame["buffer_points"])),
            KeyValue(key="buffer_min_fit_samples", value=str(frame["buffer_min_fit_samples"])),
            KeyValue(key="current_point_pushed", value=str(frame["current_point_pushed"])),
            KeyValue(key="current_point_reason", value=str(frame["current_point_reason"])),
            KeyValue(key="buffer_reset_reason", value=str(frame["buffer_reset_reason"])),
            KeyValue(key="command_published", value=str(frame["command_published"])),
            KeyValue(key="command_reason", value=str(frame["command_reason"])),
            KeyValue(
                key="estimated_vx_m_s",
                value="nan" if estimated_vx is None else f"{float(estimated_vx):.4f}",
            ),
            # These are the current Stage-3 values even when the core planner
            # rejects them at the publish-workspace gate.  They are diagnostics
            # only: no rejected candidate is sent to /racket/command.
            KeyValue(key="candidate_intercept_x_m", value=diagnostic_number("candidate_intercept_x_m")),
            KeyValue(key="candidate_intercept_y_m", value=diagnostic_number("candidate_intercept_y_m")),
            KeyValue(key="candidate_intercept_z_m", value=diagnostic_number("candidate_intercept_z_m")),
            KeyValue(key="candidate_velocity_x_m_s", value=diagnostic_number("candidate_velocity_x_m_s")),
            KeyValue(key="candidate_velocity_y_m_s", value=diagnostic_number("candidate_velocity_y_m_s")),
            KeyValue(key="candidate_velocity_z_m_s", value=diagnostic_number("candidate_velocity_z_m_s")),
            KeyValue(key="candidate_time_to_strike_s", value=diagnostic_number("candidate_time_to_strike_s")),
            KeyValue(
                key="candidate_valid",
                value="unknown" if candidate_valid is None else str(candidate_valid),
            ),
        ]
        arr.status = [status]
        self.diag_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = HOPEPlannerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

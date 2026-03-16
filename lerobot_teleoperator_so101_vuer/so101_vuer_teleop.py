import numpy as np
import threading
import time
import asyncio
from typing import Any
from scipy.spatial.transform import Rotation as R

from lerobot.teleoperators.teleoperator import Teleoperator
from .config_so101_vuer_teleop import So101VuerTeleopConfig

from vuer import Vuer, VuerSession
from vuer.schemas import Hands, MotionControllers, ImageBackground, Scene, CoordsMarker
import cv2
import base64
import pyroki as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description
from .pyroki_snippets import solve_ik

class So101VuerTeleop(Teleoperator):
    config_class = So101VuerTeleopConfig
    name = "so101_vuer"

    def __init__(self, config: So101VuerTeleopConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        
        # Threading mechanisms
        self._ik_thread = None
        self._vuer_thread = None
        self._lock = threading.Lock()
        
        # Start at a safe default position (from your IK script)
        self._target_pos = np.array([0.00931305, -0.27034248, 0.26730747])
        self._target_wxyz = np.array([0.707, -0.707, 0.0, 0.0])
        self._target_gripper = 0.0
        
        # --- NEW: Visualizer State ---
        self._viz_pos = np.array([0.0, 0.0, 0.0])
        self._viz_rot = np.array([0.0, 0.0, 0.0])
        # -----------------------------
        
        self._latest_q_sol = None
        self._latest_frame_b64 = None

        self.scale = 50.0  
        self.ik_joint_mapping = {
            "1": "shoulder_pan", "2": "shoulder_lift", "3": "elbow_flex",
            "4": "wrist_flex", "5": "wrist_roll"
        }

    def compute_robot_target_matrix(self, hand_matrix_vr):
        """
        Transforms the VR hand matrix into the Robot's Base Coordinate System.
        """
        head_pos = np.array([0.0, self.config.user_height, 0.0])
        origin_pos = head_pos + np.array([0.0, 0.0, 0.10])
        
        # Shift to the respective shoulder
        right_vec = np.array([1.0, 0.0, 0.0])
        if self.config.user_hand == "left":
            origin_pos -= right_vec * 0.20
        elif self.config.user_hand == "right":
            origin_pos += right_vec * 0.20
            
        if self.config.target_coord_sys == "headset":
            origin_pos[1] = self.config.user_height
        elif self.config.target_coord_sys == "floor":
            origin_pos[1] = 0.0 
        elif self.config.target_coord_sys == "ribs":
            origin_pos[1] = self.config.user_height - 0.40
        elif self.config.target_coord_sys == "hip":
            origin_pos[1] = self.config.user_height - 0.70

        # Create Torso to World transform (VR Space)
        T_torso_vr = np.eye(4)
        T_torso_vr[:3, 3] = origin_pos
        
        # Hand relative to VR Torso
        T_hand_torso = np.linalg.inv(T_torso_vr) @ hand_matrix_vr
        
        # Matrix to rotate VR axes to Robot axes
        # VR: +X=Right, +Y=Up, -Z=Forward
        # Robot: +X=Forward, +Y=Left, +Z=Up
        R_vr_to_robot = np.array([
            [ 0,  0, -1],
            [-1,  0,  0],
            [ 0,  1,  0]
        ])
        yaw_offset = -np.pi / 2  
        
        R_yaw = np.array([
            [np.cos(yaw_offset), -np.sin(yaw_offset), 0],
            [np.sin(yaw_offset),  np.cos(yaw_offset), 0],
            [0,                  0,                   1]
        ])
        
        # Apply the yaw offset to the rotation matrix
        R_vr_to_robot = R_yaw @ R_vr_to_robot

        T_vr_to_robot = np.eye(4)
        T_vr_to_robot[:3, :3] = R_vr_to_robot
        
        # Final Hand relative to Robot Base
        T_hand_robot = T_vr_to_robot @ T_hand_torso
        
        # Apply a local rotation offset to align the hand's grip with the robot gripper
        # Adjusting roll/pitch/yaw locally so the thumbs align
        hand_offset = R.from_euler('z', -np.pi / 2).as_matrix() # 90 degree correction around Z (local forward in VR)
        T_offset = np.eye(4)
        T_offset[:3, :3] = hand_offset
        T_hand_robot = T_hand_robot @ T_offset
        
        return T_hand_robot

    def _vuer_worker(self):
        """Background thread for the Vuer asyncio event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        app = Vuer(host=self.config.vuer_host, cert=self.config.vuer_cert, key=self.config.vuer_key)

        @app.add_handler("HAND_MOVE")
        async def on_hand_move(event, session):
            hand_data = event.value.get(self.config.user_hand)
            if not hand_data or len(hand_data) < 16:
                return
                
            wrist_flat_array = hand_data[:16]
            hand_matrix_vr = np.array(wrist_flat_array).reshape(4, 4).T

            # --- THE GIZMO MATH (VR Space) ---
            # Completely raw, unperturbed hand coordinates directly from the headset
            viz_pos = hand_matrix_vr[:3, 3]
            viz_euler = R.from_matrix(hand_matrix_vr[:3, :3]).as_euler('xyz')
            # ---------------------------------
            
            # Extract pinch strength for the gripper
            hand_state = event.value.get(f"{self.config.user_hand}State", {})
            # Some WebXR implementations use 'pinch', others use 'pinchStrength'
            pinch_val = hand_state.get("pinch", hand_state.get("pinchStrength", 0.0))
            
            # Transform matrix
            T_robot = self.compute_robot_target_matrix(hand_matrix_vr)
            pos = T_robot[:3, 3]
            
            # Extract quaternion (Scipy gives xyzw, Pyroki needs wxyz)
            quat_xyzw = R.from_matrix(T_robot[:3, :3]).as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            
            with self._lock:
                self._target_pos = pos
                
                # --- NEW: Save visualizer state ---
                self._viz_pos = viz_pos
                self._viz_rot = viz_euler
                # ----------------------------------
                
                # Block rotation updates if pinch is below threshold to allow for more stable gripper control at close range
                if pinch_val < 0.2:
                    self._target_wxyz = quat_wxyz
                    
                self._target_gripper = 1.0 - float(pinch_val)

        @app.add_handler("CONTROLLER_MOVE")
        async def on_controller_move(event, session):
            controller_data = event.value.get(self.config.user_hand)
            if not controller_data or len(controller_data) < 16:
                return
                
            wrist_flat_array = controller_data[:16]
            hand_matrix_vr = np.array(wrist_flat_array).reshape(4, 4).T

            # --- THE GIZMO MATH (VR Space) ---
            # Completely raw, unperturbed controller coordinates
            viz_pos = hand_matrix_vr[:3, 3]
            viz_euler = R.from_matrix(hand_matrix_vr[:3, :3]).as_euler('xyz')
            # ---------------------------------
            
            # Extract trigger value for the gripper
            state = event.value.get(f"{self.config.user_hand}State", {})
            pinch_val = state.get("triggerValue", state.get("squeezeValue", 0.0))
            
            # Transform matrix (Note: compute_robot_target_matrix still has one Z-axis hand_offset inside it)
            T_robot = self.compute_robot_target_matrix(hand_matrix_vr)
            
            # ALL extra controller rotation perturbations have been removed from here!
            
            pos = T_robot[:3, 3]
            
            quat_xyzw = R.from_matrix(T_robot[:3, :3]).as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            
            with self._lock:
                self._target_pos = pos
                
                # --- NEW: Save visualizer state ---
                self._viz_pos = viz_pos
                self._viz_rot = viz_euler
                # ----------------------------------
                
                if pinch_val < 0.2:
                    self._target_wxyz = quat_wxyz
                    
                self._target_gripper = 1.0 - float(pinch_val)

        @app.spawn(start=True)
        async def main(session: VuerSession):
            session.set(Scene())
            session.upsert(Hands(stream=True, key="hands", showLeft=True, showRight=True), to="bgChildren")
            session.upsert(MotionControllers(stream=True, key="motionControllers", left=True, right=True), to="bgChildren")
            last_sent_b64 = None
            while self._is_connected:
                with self._lock:
                    current_img = self._latest_frame
                    # Safely copy the gizmo state
                    viz_pos = self._viz_pos.copy()
                    viz_rot = self._viz_rot.copy()
                    
                # --- NEW: Render the Vector Gizmo ---
                session.upsert(
                    CoordsMarker(
                        position=viz_pos.tolist(),
                        rotation=viz_rot.tolist(),
                        scale=0.15, # Sets the vectors to be 15cm long
                        key="ik_gizmo"
                    ),
                    to="bgChildren"
                )
                
                if current_img is not None:
                    session.upsert(
                        ImageBackground(
                            current_img,
                            format="jpeg",
                            quality=50,
                            fixed=True,             
                            # Pushes the screen further away from your face (80cm)
                            distanceToCamera=1,   
                            key="camera_feed",
                            # X=0 (centered)
                            # Y= drops the screen 60cm below your eye level
                            # Z=0 (depth is handled by distanceToCamera)
                            position=[0, self.config.user_height - 0.6, -3],
                        ),
                        to="bgChildren"
                    )
                    
                await asyncio.sleep(0.033)

        print("VR Server Started. Waiting for headset connection...")
        app.run()

    def _ik_worker(self):
        """Background thread that continuously solves IK based on the VR state."""
        while self._is_connected:
            with self._lock:
                target_pos = self._target_pos.copy()
                target_quat = self._target_wxyz.copy()

            q_sol = solve_ik(
                robot=self.robot,
                target_link_name=self.config.target_link,
                target_position=target_pos,
                target_wxyz=target_quat,
            )

            if q_sol is not None:
                with self._lock:
                    self._latest_q_sol = q_sol

            time.sleep(0.01)

    def _camera_worker(self):
        """Autonomously searches for the active robot in memory to copy its camera feed, bypassing LeRobot's closed CLI loop."""
        import gc
        from lerobot.robots.robot import Robot as BaseRobot
        active_robot = None
        
        while self._is_connected:
            try:
                if active_robot is None:
                    # Dynamically find the active robot initialized by the CLI
                    for obj in gc.get_objects():
                        if isinstance(obj, BaseRobot) and getattr(obj, "is_connected", False):
                            active_robot = obj
                            break
                
                if active_robot is not None and hasattr(active_robot, "get_observation"):
                    obs = active_robot.get_observation()
                    selected_img = None
                    # Find any camera matrix from MuJoCo or Realsense
                    for key, val in obs.items():
                        if isinstance(val, np.ndarray) and val.ndim == 3:
                            selected_img = val
                            break
                            
                    if selected_img is not None:
                        # Convert to uint8 if necessary
                        if selected_img.dtype != np.uint8:
                            selected_img = (np.clip(selected_img, 0, 1) * 255).astype(np.uint8)

                        # Deep copy and resize to save bandwidth (fixes the SegFault and Network Choking)
                        selected_img = cv2.resize(selected_img.copy(), (320, 240))
                        
                        with self._lock:
                            # Pass the raw numpy array directly! No Base64 or OpenCV conversion needed.
                            self._latest_frame = selected_img
            except Exception:
                pass
            time.sleep(0.033) # ~30 FPS polling

    def connect(self) -> None:
        self.urdf = load_robot_description(self.config.urdf_name)
        self.robot = pk.Robot.from_urdf(self.urdf)
        self.urdf_joints = [j.name for j in self.urdf.actuated_joints]
        
        print("\n--- Compiling JAX IK Solver ---")
        dummy_pos = np.array([0.3, 0.0, 0.2])
        dummy_quat = np.array([1.0, 0.0, 0.0, 0.0])
        solve_ik(
            robot=self.robot, target_link_name=self.config.target_link,
            target_position=dummy_pos, target_wxyz=dummy_quat,
        )
        print("--- JAX Compilation Complete! ---\n")

        self._is_connected = True

        self._ik_thread = threading.Thread(target=self._ik_worker, daemon=True)
        self._vuer_thread = threading.Thread(target=self._vuer_worker, daemon=True)
        self._cam_thread = threading.Thread(target=self._camera_worker, daemon=True)
        
        self._ik_thread.start()
        self._vuer_thread.start()
        self._cam_thread.start()

    def disconnect(self) -> None:
        self._is_connected = False
        if self._ik_thread:
            self._ik_thread.join(timeout=1.0)
        # Note: Vuer apps are notoriously difficult to kill gracefully from a thread,
        # but the daemon=True flag ensures it dies when the main LeRobot script exits.

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def get_action(self) -> dict:
        with self._lock:
            q_sol = self._latest_q_sol
            gripper_val = self._target_gripper

        action_dict = {
            "shoulder_pan.pos": 0.0, "shoulder_lift.pos": 0.0, "elbow_flex.pos": 0.0,
            "wrist_flex.pos": 0.0, "wrist_roll.pos": 0.0,
            "gripper.pos": gripper_val * self.scale,
        }

        if q_sol is not None:
            if "1" in self.urdf_joints: action_dict["shoulder_pan.pos"] = float(q_sol[self.urdf_joints.index("1")]) * self.scale
            if "2" in self.urdf_joints: action_dict["shoulder_lift.pos"] = float(q_sol[self.urdf_joints.index("2")]) * self.scale
            if "3" in self.urdf_joints: action_dict["elbow_flex.pos"] = float(q_sol[self.urdf_joints.index("3")]) * self.scale
            if "4" in self.urdf_joints: action_dict["wrist_flex.pos"] = float(q_sol[self.urdf_joints.index("4")]) * self.scale
            if "5" in self.urdf_joints: action_dict["wrist_roll.pos"] = float(q_sol[self.urdf_joints.index("5")]) * self.scale
            
        return action_dict
    
    @property
    def action_features(self) -> dict:
        return {
            "shoulder_pan.pos": float, "shoulder_lift.pos": float, "elbow_flex.pos": float,
            "wrist_flex.pos": float, "wrist_roll.pos": float, "gripper.pos": float,
        }

    @property
    def feedback_features(self) -> dict: return {}
    @property
    def is_calibrated(self) -> bool: return True
    def calibrate(self) -> None: pass
    def configure(self) -> None: pass
    
    def send_feedback(self, feedback: dict[str, Any]) -> None: pass

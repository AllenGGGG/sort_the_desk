# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example command:
```shell
python src/lerobot/async_inference/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
"""

import logging
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.import_utils import register_third_party_plugins

from .configs import RobotClientConfig
from .helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
    visualize_interpolation,
)


class YoloSafetyDetector:
    """YOLO-based safety detector: triggers emergency stop when a person enters camera1 view."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf: float = 0.5,
        required_hits: int = 2,
        device: str = "cpu",
    ):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "YOLO safety detection requires the 'ultralytics' package in the same Python "
                "environment used to run robot_client."
            ) from e

        self.model = YOLO(model_path)
        self.conf = conf
        self.danger_classes = {"person"}
        self.required_hits = required_hits
        self.hit_count = 0
        self.device = device

    def is_danger(self, frame: np.ndarray) -> bool:
        """检测画面中是否有危险目标。frame: (H, W, 3) BGR/RGB numpy array."""
        result = self.model(frame, verbose=False, device=self.device)[0]
        danger = False

        for box in result.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]

            if conf < self.conf:
                continue
            if cls_name not in self.danger_classes:
                continue

            # ROI 是整个画面 (640x480)，所以只要检测到就算危险
            danger = True
            break

        if danger:
            self.hit_count += 1
        else:
            self.hit_count = 0

        return self.hit_count >= self.required_hits


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        # Store configuration
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
        )
        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()

        # Initialize client side variables
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing

        # Interpolation: upsample policy actions to higher motor command frequency.
        # `fps` remains the policy/observation frequency. `control_fps` is the motor command frequency.
        self.policy_fps = self.config.fps
        self.control_fps = self.config.control_fps
        self.upsample_factor = self.control_fps // self.policy_fps
        self.control_dt = 1.0 / self.control_fps  # 电机控制周期
        self.prev_action: torch.Tensor | None = None
        self.sub_action_buffer: list[torch.Tensor] = []
        self.control_tick = 0  # 用于控制 observation 发送频率
        self.ema_alpha = 0.4
        self.prev_smoothed_action: torch.Tensor | None = None

        # Debug: record actions for interpolation visualization.
        self.debug_policy_actions: list[torch.Tensor] = []
        self.debug_motor_actions: list[torch.Tensor] = []

        # Safety: YOLO-based emergency stop
        self.safety_detector = YoloSafetyDetector()
        self.emergency_stop = False
        self.ignore_action_chunks_before = 0.0
        self.safety_frame_lock = threading.Lock()
        self.safety_frame: np.ndarray | None = None
        self.safety_danger = False
        self.safety_thread = threading.Thread(target=self._safety_loop, daemon=True)

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()
            self.safety_thread.start()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client"""
        self.shutdown_event.set()

        if self.safety_thread.is_alive():
            self.safety_thread.join(timeout=1.0)

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep} | ")

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                if self.emergency_stop:
                    self.logger.debug("Dropping received actions while emergency stop is active")
                    continue

                if (
                    len(timed_actions) > 0
                    and timed_actions[0].get_timestamp() < self.ignore_action_chunks_before
                ):
                    self.logger.debug("Dropping stale action chunk generated before emergency stop")
                    continue

                # Log device type of received actions
                if len(timed_actions) > 0:
                    received_device = timed_actions[0].get_action().device.type
                    self.logger.debug(f"Received actions on device: {received_device}")

                # Move actions to client_device (e.g., for downstream planners that need GPU)
                client_device = self.config.client_device
                if client_device != "cpu":
                    for timed_action in timed_actions:
                        if timed_action.get_action().device.type != client_device:
                            timed_action.action = timed_action.get_action().to(client_device)
                    self.logger.debug(f"Converted actions to device: {client_device}")
                else:
                    self.logger.debug(f"Actions kept on device: {client_device}")

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Log incoming actions
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # Update action queue
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

                if verbose:
                    # Get queue state after changes
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self):
        """Check if there are actions available in the queue or sub_action_buffer"""
        if self.sub_action_buffer:
            return True
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def _clear_action_buffers(self) -> None:
        with self.action_queue_lock:
            self.action_queue = Queue()
        self.sub_action_buffer.clear()
        self.prev_action = None
        self.prev_smoothed_action = None
        self.action_chunk_size = -1
        self.must_go.set()

    def _hold_current_position(self, raw_observation: RawObservation) -> None:
        hold_action = {
            key: raw_observation[key]
            for key in self.robot.action_features
            if key in raw_observation
        }
        if len(hold_action) != len(self.robot.action_features):
            missing = set(self.robot.action_features) - set(hold_action)
            self.logger.warning(f"Cannot hold current position, missing observation keys: {missing}")
            return

        self.robot.send_action(hold_action)

    def _queue_safety_frame(self, raw_observation: RawObservation) -> None:
        frame = raw_observation.get("camera1")
        if frame is None:
            return

        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()

        with self.safety_frame_lock:
            self.safety_frame = frame.copy()

    def _latest_safety_danger(self) -> bool:
        with self.safety_frame_lock:
            return self.safety_danger

    def _safety_loop(self) -> None:
        while self.running:
            with self.safety_frame_lock:
                frame = self.safety_frame
                self.safety_frame = None

            if frame is None:
                time.sleep(0.005)
                continue

            try:
                danger = self.safety_detector.is_danger(frame)
            except Exception as e:
                self.logger.error(f"Safety detector error: {e}")
                danger = True

            with self.safety_frame_lock:
                self.safety_danger = danger

    def _update_emergency_stop(self, raw_observation: RawObservation) -> None:
        self._queue_safety_frame(raw_observation)
        danger = self._latest_safety_danger()

        if danger:
            if not self.emergency_stop:
                self.logger.warning("EMERGENCY STOP: person detected in workspace!")
                self.ignore_action_chunks_before = time.time()
                self._clear_action_buffers()

            self.emergency_stop = True
            self._hold_current_position(raw_observation)
            return

        if self.emergency_stop:
            self.logger.info("Safety clear - waiting for fresh action chunk")
            self._clear_action_buffers()
            self.emergency_stop = False

    def _smooth_action_with_ema(self, action: torch.Tensor) -> torch.Tensor:
        if self.prev_smoothed_action is None:
            smoothed_action = action.clone()
        else:
            smoothed_action = self.ema_alpha * action + (1 - self.ema_alpha) * self.prev_smoothed_action
            # Keep gripper crisp; only smooth arm joints.
            smoothed_action[-1] = action[-1]

        self.prev_smoothed_action = smoothed_action.clone()
        return smoothed_action

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """Reading and performing actions in local queue, with linear interpolation for upsampling."""

        # Fill sub_action_buffer when empty
        if not self.sub_action_buffer:
            get_start = time.perf_counter()
            with self.action_queue_lock:
                self.action_queue_size.append(self.action_queue.qsize())
                self._current_timed_action = self.action_queue.get_nowait()
            get_end = time.perf_counter() - get_start

            current_action = self._current_timed_action.get_action()

            if self.config.debug_visualize_queue_size:
                self.debug_policy_actions.append(current_action.clone())

            # Linear interpolation between prev and current action
            # Gripper (last dim) skips interpolation — uses target value directly for crisp open/close
            if self.prev_action is not None:
                for i in range(1, self.upsample_factor + 1):
                    t = i / self.upsample_factor
                    interp = (1 - t) * self.prev_action + t * current_action
                    # Gripper: always use current target, no interpolation
                    interp[-1] = current_action[-1]
                    self.sub_action_buffer.append(interp)
            else:
                # First action, no interpolation possible
                self.sub_action_buffer = [current_action] * self.upsample_factor

            self.prev_action = current_action.clone()

            if verbose:
                with self.action_queue_lock:
                    current_queue_size = self.action_queue.qsize()
                self.logger.debug(
                    f"Ts={self._current_timed_action.get_timestamp()} | "
                    f"Action #{self._current_timed_action.get_timestep()} performed | "
                    f"Queue size: {current_queue_size}"
                )
                self.logger.debug(
                    f"Popping action from queue took {get_end:.6f}s | Queue size: {current_queue_size}"
                )

        # Execute one sub-action per control loop tick
        sub_action = self.sub_action_buffer.pop(0)
        smoothed_action = self._smooth_action_with_ema(sub_action)

        if self.config.debug_visualize_queue_size:
            self.debug_motor_actions.append(smoothed_action.clone())

        _performed_action = self.robot.send_action(
            self._action_tensor_to_action_dict(smoothed_action)
        )

        # Update latest_action only after all sub-actions for this timestep are consumed
        if not self.sub_action_buffer:
            with self.latest_action_lock:
                self.latest_action = self._current_timed_action.get_timestep()

        return _performed_action

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(
        self, task: str, verbose: bool = False, raw_observation: RawObservation | None = None
    ) -> RawObservation:
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            if raw_observation is None:
                raw_observation = self.robot.get_observation()
            raw_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            _ = self.send_observation(observation)

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations.

        Motor control runs at control_fps, observation sending runs at policy_fps.
        Safety detection runs on camera1 at observation frequency.
        """
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()

            """Control loop: (1) Safety check and observation streaming at policy_fps."""
            # Only send observation every upsample_factor ticks
            self.control_tick += 1
            if self.control_tick >= self.upsample_factor:
                self.control_tick = 0

                # Safety check is independent of policy queue state. Always inspect camera1 at policy_fps.
                try:
                    safety_observation = self.robot.get_observation()
                    self._update_emergency_stop(safety_observation)
                except Exception as e:
                    if not self.emergency_stop:
                        self.logger.error(f"EMERGENCY STOP: failed to read safety observation: {e}")
                        self._clear_action_buffers()
                    self.emergency_stop = True
                    safety_observation = None

                if (
                    safety_observation is not None
                    and not self.emergency_stop
                    and self._ready_to_send_observation()
                ):
                    _captured_observation = self.control_loop_observation(
                        task, verbose, raw_observation=safety_observation
                    )

            """Control loop: (2) Performing actions at control_fps, unless emergency stop."""
            if self.emergency_stop:
                self.logger.debug("Emergency stop active - holding position")
            elif self.actions_available():
                _performed_action = self.control_loop_action(verbose)

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Sleep at control_fps (60Hz), not policy_fps
            time.sleep(max(0, self.control_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    # TODO: Assert if checking robot support is still needed with the plugin system
    # if cfg.robot.type not in SUPPORTED_ROBOTS:
    #     raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")

        # Create and start action receiver thread
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)

        # Start action receiver thread
        action_receiver_thread.start()

        try:
            # The main thread runs the control loop
            client.control_loop(task=cfg.task)

        except KeyboardInterrupt:
            client.logger.info("KeyboardInterrupt received, shutting down...")

        finally:
            # Save debug plots BEFORE stopping (in case stop/join hangs)
            if cfg.debug_visualize_queue_size:
                try:
                    visualize_action_queue_size(client.action_queue_size)
                    joint_names = list(client.robot.action_features.keys())
                    visualize_interpolation(
                        client.debug_policy_actions,
                        client.debug_motor_actions,
                        joint_names=joint_names,
                    )
                except Exception as e:
                    print(f"[DEBUG] Failed to save plots: {e}")

            client.stop()
            action_receiver_thread.join(timeout=3.0)  # 最多等 3 秒，防止卡死
            client.logger.info("Client stopped")


if __name__ == "__main__":
    register_third_party_plugins()
    async_client()  # run the client


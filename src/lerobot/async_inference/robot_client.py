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
from queue import Empty, Queue
from typing import Any

import cv2
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
    compress_observation_images,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
    visualize_interpolation,
)


class YoloSafetyDetector:
    """YOLO-based safety detector: triggers emergency stop when a person enters the safety camera view.

    NOTE: this class does NOT call cv2.imshow. On macOS, GUI calls must run on the main thread,
    so rendering is the caller's responsibility. `detect()` returns (emergency, display_frame).
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf: float = 0.5,
        required_hits: int = 2,
        device: str = "cpu",
        visualize: bool = False,
        window_name: str = "YOLO safety camera",
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
        self.visualize = visualize
        self.window_name = window_name

    def detect(self, frame: np.ndarray) -> tuple[bool, np.ndarray | None]:
        """Run YOLO on one frame. Returns (emergency, annotated_bgr_frame_or_None).

        The annotated frame is returned only when `visualize` is True, so the main thread
        can call cv2.imshow on it. No GUI calls are made inside this method.
        """
        result = self.model(frame, verbose=False, device=self.device)[0]
        danger = False
        display_frame = None

        if self.visualize:
            display_frame = frame.copy()
            if display_frame.ndim == 3 and display_frame.shape[2] == 3:
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

        for box in result.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]

            if conf < self.conf:
                continue

            is_danger_class = cls_name in self.danger_classes
            if is_danger_class:
                danger = True

            if self.visualize and display_frame is not None:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                color = (0, 0, 255) if is_danger_class else (128, 128, 128)
                label = f"{cls_name} {conf:.2f}"
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    display_frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        if danger:
            self.hit_count += 1
        else:
            self.hit_count = 0

        emergency = self.hit_count >= self.required_hits

        if self.visualize and display_frame is not None:
            status = "EMERGENCY STOP" if emergency else "SAFE"
            status_color = (0, 0, 255) if emergency else (0, 180, 0)
            cv2.putText(
                display_frame,
                status,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2,
                cv2.LINE_AA,
            )

        return emergency, display_frame

    def annotate_passthrough(self, frame: np.ndarray, emergency: bool) -> np.ndarray:
        """Produce a BGR status-overlayed frame for non-detection ticks. No GUI calls."""
        display_frame = frame.copy()
        if display_frame.ndim == 3 and display_frame.shape[2] == 3:
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

        status = "EMERGENCY STOP" if emergency else "SAFE"
        status_color = (0, 0, 255) if emergency else (0, 180, 0)
        cv2.putText(
            display_frame,
            status,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
            cv2.LINE_AA,
        )
        return display_frame


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

        # Raw robot observation keys that carry camera frames (tuple-shaped features).
        # Motor/scalar features have `float`-like types. We compress only camera frames.
        self._robot_image_keys = {
            k for k, ft in self.robot.observation_features.items() if isinstance(ft, tuple)
        }
        self._obs_target_hw = (config.obs_target_height, config.obs_target_width)

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

        # Observation sender: decouples network IO from the motor control loop.
        # Main loop only push()s a ready-to-send (compressed) TimedObservation; the
        # sender thread owns the gRPC stub.SendObservations call. When the queue is
        # full the OLDEST frame is dropped so the sender can never backpressure the
        # control loop.
        self.obs_send_queue: "Queue[TimedObservation]" = Queue(maxsize=2)
        self.obs_drop_count = 0
        self.obs_timeout_count = 0
        self._obs_sender_thread = threading.Thread(
            target=self._observation_sender_loop, daemon=True
        )

        # Stale-action watchdog: track when we last accepted a fresh action chunk.
        # If no fresh chunk arrives within `stale_action_threshold`, the main loop
        # flushes the queue and holds position to avoid running open-loop against
        # a stale visual observation.
        self._last_action_received_at: float | None = None
        self._last_action_recv_lock = threading.Lock()
        self._action_stale = False

        # Safety: YOLO-based emergency stop (only constructed when enabled).
        self.yolo_enabled = bool(config.yolo_enabled)
        self.safety_detector: YoloSafetyDetector | None = None
        self.emergency_stop = False
        self.ignore_action_chunks_before = 0.0
        self.safety_frame_lock = threading.Lock()
        self.safety_frame: np.ndarray | None = None
        self.safety_danger = False
        self.safety_frame_count = 0
        # Handed off from safety thread to main thread for cv2.imshow (macOS requires main thread).
        self.safety_display_lock = threading.Lock()
        self.safety_display_frame: np.ndarray | None = None
        self.safety_thread: threading.Thread | None = None

        if self.yolo_enabled:
            # Fail-safe: require the safety camera to exist before we let the robot move.
            if config.safety_camera not in self._robot_image_keys:
                raise ValueError(
                    f"safety_camera '{config.safety_camera}' is not in the robot's camera keys "
                    f"({sorted(self._robot_image_keys)}). Either correct --safety_camera or "
                    f"disable --yolo_enabled."
                )

            self.safety_detector = YoloSafetyDetector(
                model_path=config.yolo_model_path,
                conf=config.yolo_conf,
                required_hits=config.yolo_required_hits,
                device=config.yolo_device,
                visualize=config.yolo_visualize,
            )
            self.safety_thread = threading.Thread(target=self._safety_loop, daemon=True)
        else:
            self.logger.warning(
                "YOLO safety loop is DISABLED (yolo_enabled=False). The robot will not auto-stop "
                "on person detection. Use --yolo_enabled=True for real deployments."
            )

        # Inform the user about the compression contract (cannot be auto-verified here because
        # the policy's image_features live on the server). The server will still re-run its own
        # resize to policy shape if needed; setting target_hw to the policy's declared shape
        # avoids a double resize and keeps distribution identical to training.
        if config.obs_compression_enabled:
            self.logger.info(
                f"Observation compression ON (JPEG q={config.obs_jpeg_quality}, "
                f"target_hw={self._obs_target_hw}). Make sure target_hw matches the policy's "
                f"image_features shape; mismatch will cause double resize on the server."
            )

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
            if self.safety_thread is not None:
                self.safety_thread.start()
            self._obs_sender_thread.start()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client"""
        self.shutdown_event.set()

        # Unblock the sender thread which may be waiting on the queue.
        try:
            self.obs_send_queue.put_nowait(None)  # type: ignore[arg-type]
        except Exception:
            pass

        if self._obs_sender_thread.is_alive():
            self._obs_sender_thread.join(timeout=1.0)

        if self.safety_thread is not None and self.safety_thread.is_alive():
            self.safety_thread.join(timeout=1.0)

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Hand off a (already-compressed) TimedObservation to the sender thread.

        This call is non-blocking: if the send queue is full, the OLDEST queued
        observation is dropped to make room. The main control loop is therefore
        never blocked by network IO or gRPC retries.
        """
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        # Non-blocking enqueue with drop-oldest policy.
        try:
            self.obs_send_queue.put_nowait(obs)
        except Exception:
            # Queue full: drop the oldest pending obs, then push the new one.
            try:
                dropped = self.obs_send_queue.get_nowait()
                self.obs_drop_count += 1
                if isinstance(dropped, TimedObservation):
                    self.logger.debug(
                        f"Dropped stale observation #{dropped.get_timestep()} "
                        f"(total dropped: {self.obs_drop_count})"
                    )
            except Exception:
                pass
            try:
                self.obs_send_queue.put_nowait(obs)
            except Exception:
                return False

        return True

    def _observation_sender_loop(self) -> None:
        """Dedicated thread: owns compression + the gRPC SendObservations call.

        JPEG resize/encode and any network hiccup / retry backoff / serialization
        cost lives here, isolated from the motor control loop. A per-call
        deadline (`obs_send_timeout`) bounds a single send so a bad network
        cannot pile up latency.
        """
        timeout = float(self.config.obs_send_timeout) if self.config.obs_send_timeout > 0 else None
        while self.running:
            try:
                obs = self.obs_send_queue.get(timeout=0.1)
            except Empty:
                continue

            if obs is None:
                # Sentinel from stop() — drain and exit.
                continue

            try:
                # Step 1: compress camera frames (off the motor-loop critical path).
                if self.config.obs_compression_enabled:
                    compress_start = time.perf_counter()
                    obs.observation = compress_observation_images(
                        obs.get_observation(),
                        image_keys=self._robot_image_keys,
                        target_hw=self._obs_target_hw,
                        quality=self.config.obs_jpeg_quality,
                    )
                    self.logger.debug(
                        f"Observation compression time: {time.perf_counter() - compress_start:.6f}s"
                    )

                # Step 2: serialize + send.
                start_time = time.perf_counter()
                observation_bytes = pickle.dumps(obs)
                serialize_time = time.perf_counter() - start_time
                self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

                observation_iterator = send_bytes_in_chunks(
                    observation_bytes,
                    services_pb2.Observation,
                    log_prefix="[CLIENT] Observation",
                    silent=True,
                )
                self.stub.SendObservations(observation_iterator, timeout=timeout)
                self.logger.debug(f"Sent observation #{obs.get_timestep()}")

            except grpc.RpcError as e:
                code = getattr(e, "code", lambda: None)()
                if code == grpc.StatusCode.DEADLINE_EXCEEDED:
                    self.obs_timeout_count += 1
                    self.logger.warning(
                        f"SendObservations timed out (#{obs.get_timestep()}, total timeouts: "
                        f"{self.obs_timeout_count}). Frame dropped."
                    )
                else:
                    self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            except Exception as e:  # noqa: BLE001
                self.logger.error(f"Unexpected error in sender loop: {e}")

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

                # Watchdog: a fresh chunk just landed.
                with self._last_action_recv_lock:
                    self._last_action_received_at = time.time()
                    if self._action_stale:
                        self.logger.info("Fresh action chunk received - clearing stale-action hold")
                        self._action_stale = False

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

    def _check_action_staleness(self) -> bool:
        """Return True if no fresh action chunk has landed within
        `stale_action_threshold`. When triggered, flushes the action buffer so
        the robot does not run open-loop against outdated visual context.

        Always False when `stale_action_threshold <= 0` (watchdog disabled) or
        before the first action chunk has been received.
        """
        threshold = float(self.config.stale_action_threshold)
        if threshold <= 0:
            return False

        with self._last_action_recv_lock:
            last_at = self._last_action_received_at
            already_stale = self._action_stale

        if last_at is None:
            # Haven't heard from the server yet (startup). Don't trip the watchdog.
            return False

        elapsed = time.time() - last_at
        if elapsed <= threshold:
            return False

        if not already_stale:
            self.logger.warning(
                f"STALE ACTIONS: no new chunk for {elapsed:.2f}s "
                f"(threshold={threshold:.2f}s). Flushing buffer and holding position."
            )
            with self._last_action_recv_lock:
                self._action_stale = True
            self._clear_action_buffers()

        return True

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
        frame = raw_observation.get(self.config.safety_camera)
        if frame is None:
            self.logger.warning(
                f"Safety camera '{self.config.safety_camera}' not found in observation. "
                f"Available keys: {list(raw_observation.keys())}"
            )
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

            self.safety_frame_count += 1
            should_detect = self.safety_frame_count % self.config.yolo_detect_every_n == 0

            display_frame = None
            if not should_detect:
                if self.config.yolo_visualize:
                    display_frame = self.safety_detector.annotate_passthrough(
                        frame, self.safety_danger
                    )
            else:
                try:
                    danger, display_frame = self.safety_detector.detect(frame)
                except Exception as e:
                    self.logger.error(f"Safety detector error: {e}")
                    danger = True
                    display_frame = None

                with self.safety_frame_lock:
                    self.safety_danger = danger

            if self.config.yolo_visualize and display_frame is not None:
                with self.safety_display_lock:
                    self.safety_display_frame = display_frame

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

            # Hand a SHALLOW-COPIED raw observation to the sender. JPEG resize/encode
            # runs in `_observation_sender_loop` so the motor control loop stays off
            # the hot path for image processing (cv2.resize + imencode ~4-8 ms/frame).
            obs_to_send = dict(raw_observation)

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=obs_to_send,
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
        When yolo_enabled is True, safety detection runs on the configured
        safety_camera at observation frequency. Network IO is offloaded to a
        dedicated sender thread, so the motor loop is never blocked by gRPC.
        """
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None
        # Most recent raw observation captured by the loop. Persisted across ticks
        # so the stale-action watchdog can hold the last known position even on
        # ticks that didn't refresh the observation.
        obs_for_send: RawObservation | None = None

        while self.running:
            control_loop_start = time.perf_counter()

            """Control loop: (1) Safety check (optional) and observation streaming at policy_fps."""
            # Only send observation every upsample_factor ticks
            self.control_tick += 1
            if self.control_tick >= self.upsample_factor:
                self.control_tick = 0

                if self.yolo_enabled:
                    # Safety check is independent of policy queue state. Always inspect
                    # safety_camera at policy_fps.
                    try:
                        obs_for_send = self.robot.get_observation()
                        self._update_emergency_stop(obs_for_send)
                    except Exception as e:
                        if not self.emergency_stop:
                            self.logger.error(f"EMERGENCY STOP: failed to read safety observation: {e}")
                            self._clear_action_buffers()
                        self.emergency_stop = True
                        obs_for_send = None
                else:
                    # YOLO disabled: skip safety detection entirely, but still capture
                    # the observation so the policy keeps running.
                    try:
                        obs_for_send = self.robot.get_observation()
                    except Exception as e:
                        self.logger.error(f"Failed to read observation: {e}")
                        obs_for_send = None

                if (
                    obs_for_send is not None
                    and not self.emergency_stop
                    and self._ready_to_send_observation()
                ):
                    _captured_observation = self.control_loop_observation(
                        task, verbose, raw_observation=obs_for_send
                    )

            """Control loop: (2) Performing actions at control_fps, unless emergency stop or stale actions."""
            # Watchdog: detect stale action chunks and hold position instead of
            # running open-loop on outdated visual context.
            action_stale = self._check_action_staleness()

            if self.emergency_stop:
                self.logger.debug("Emergency stop active - holding position")
            elif action_stale:
                # Stale: buffer already flushed by the watchdog. Hold the last
                # commanded position using the most recent raw observation.
                if obs_for_send is not None:
                    self._hold_current_position(obs_for_send)
            elif self.actions_available():
                _performed_action = self.control_loop_action(verbose)

            # Render YOLO visualization on the main thread (macOS requires GUI calls on main thread).
            if self.yolo_enabled and self.config.yolo_visualize:
                with self.safety_display_lock:
                    display_frame = self.safety_display_frame
                    self.safety_display_frame = None
                if display_frame is not None and self.safety_detector is not None:
                    cv2.imshow(self.safety_detector.window_name, display_frame)
                    cv2.waitKey(1)

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Sleep at control_fps, not policy_fps
            time.sleep(max(0, self.control_dt - (time.perf_counter() - control_loop_start)))

        if self.yolo_enabled and self.config.yolo_visualize and self.safety_detector is not None:
            try:
                cv2.destroyWindow(self.safety_detector.window_name)
            except cv2.error:
                pass

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


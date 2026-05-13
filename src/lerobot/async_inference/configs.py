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

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from lerobot.robots.config import RobotConfig

from .constants import (
    DEFAULT_FPS,
    DEFAULT_INFERENCE_LATENCY,
    DEFAULT_OBS_QUEUE_TIMEOUT,
)

# Aggregate function registry for CLI usage
AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


def get_aggregate_function(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get aggregate function by name from registry."""
    if name not in AGGREGATE_FUNCTIONS:
        available = list(AGGREGATE_FUNCTIONS.keys())
        raise ValueError(f"Unknown aggregate function '{name}'. Available: {available}")
    return AGGREGATE_FUNCTIONS[name]


@dataclass
class PolicyServerConfig:
    """Configuration for PolicyServer.

    This class defines all configurable parameters for the PolicyServer,
    including networking settings and action chunking specifications.
    """

    # Networking configuration
    host: str = field(default="localhost", metadata={"help": "Host address to bind the server to"})
    port: int = field(default=8080, metadata={"help": "Port number to bind the server to"})

    # Timing configuration
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second"})
    inference_latency: float = field(
        default=DEFAULT_INFERENCE_LATENCY, metadata={"help": "Target inference latency in seconds"}
    )

    obs_queue_timeout: float = field(
        default=DEFAULT_OBS_QUEUE_TIMEOUT, metadata={"help": "Timeout for observation queue in seconds"}
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {self.port}")

        if self.environment_dt <= 0:
            raise ValueError(f"environment_dt must be positive, got {self.environment_dt}")

        if self.inference_latency < 0:
            raise ValueError(f"inference_latency must be non-negative, got {self.inference_latency}")

        if self.obs_queue_timeout < 0:
            raise ValueError(f"obs_queue_timeout must be non-negative, got {self.obs_queue_timeout}")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PolicyServerConfig":
        """Create a PolicyServerConfig from a dictionary."""
        return cls(**config_dict)

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds"""
        return 1 / self.fps

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "fps": self.fps,
            "environment_dt": self.environment_dt,
            "inference_latency": self.inference_latency,
        }


@dataclass
class RobotClientConfig:
    """Configuration for RobotClient.

    This class defines all configurable parameters for the RobotClient,
    including network connection, policy settings, and control behavior.
    """

    # Policy configuration
    policy_type: str = field(metadata={"help": "Type of policy to use"})
    pretrained_name_or_path: str = field(metadata={"help": "Pretrained model name or path"})

    # Robot configuration (for CLI usage - robot instance will be created from this)
    robot: RobotConfig = field(metadata={"help": "Robot configuration"})

    # Policies typically output K actions at max, but we can use less to avoid wasting bandwidth (as actions
    # would be aggregated on the client side anyway, depending on the value of `chunk_size_threshold`)
    actions_per_chunk: int = field(metadata={"help": "Number of actions per chunk"})

    # Task instruction for the robot to execute (e.g., 'fold my tshirt')
    task: str = field(default="", metadata={"help": "Task instruction for the robot to execute"})

    # Network configuration
    server_address: str = field(default="localhost:8080", metadata={"help": "Server address to connect to"})

    # Device configuration
    policy_device: str = field(default="cpu", metadata={"help": "Device for policy inference"})
    client_device: str = field(
        default="cpu",
        metadata={
            "help": "Device to move actions to after receiving from server (e.g., for downstream planners)"
        },
    )

    # Control behavior configuration
    chunk_size_threshold: float = field(default=0.5, metadata={"help": "Threshold for chunk size control"})
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second"})
    control_fps: int = field(default=60, metadata={"help": "Motor command frequency on the robot client"})

    # YOLO safety detection configuration
    yolo_enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable the YOLO-based safety loop. When False, the safety thread and detector "
                "are not constructed, and `ultralytics` is not imported."
            )
        },
    )
    yolo_visualize: bool = field(
        default=False, metadata={"help": "Show safety camera with YOLO detection boxes"}
    )
    yolo_model_path: str = field(
        default="yolo11n.pt",
        metadata={"help": "Path to YOLO model used for safety detection"},
    )
    yolo_conf: float = field(default=0.5, metadata={"help": "YOLO confidence threshold"})
    yolo_required_hits: int = field(
        default=2,
        metadata={"help": "Consecutive person detections required to trigger emergency stop"},
    )
    yolo_device: str = field(
        default="cpu", metadata={"help": "Device used for YOLO safety detection"}
    )
    yolo_detect_every_n: int = field(
        default=1,
        metadata={"help": "Run YOLO once every N safety camera frames. 1 means every frame."},
    )
    safety_camera: str = field(
        default="camera1",
        metadata={"help": "Camera key in robot observation used for YOLO safety detection"},
    )

    # Observation compression (reduces network bandwidth between client and server).
    # When enabled, the client resizes camera frames to (obs_target_height, obs_target_width)
    # and JPEG-encodes them before sending. The server decodes on receive; the policy input
    # is unchanged as long as the target shape matches the policy's image_features.
    obs_compression_enabled: bool = field(
        default=False,
        metadata={"help": "Enable JPEG compression of camera frames before sending to server"},
    )
    obs_jpeg_quality: int = field(
        default=90,
        metadata={"help": "JPEG quality (1-100) when obs_compression_enabled is True"},
    )
    obs_target_height: int = field(
        default=480,
        metadata={"help": "Resize target height for camera frames before JPEG encoding"},
    )
    obs_target_width: int = field(
        default=640,
        metadata={"help": "Resize target width for camera frames before JPEG encoding"},
    )

    # Per-call gRPC deadline for SendObservations. Prevents a bad network from
    # blocking the sender thread indefinitely. Timeout errors are logged and the
    # frame is dropped; the main control loop is never blocked by network.
    obs_send_timeout: float = field(
        default=0.2,
        metadata={"help": "Per-call SendObservations timeout in seconds (<=0 disables)"},
    )

    # Stale-action watchdog: if no fresh action chunk arrives for this long, the
    # client flushes its buffer and holds position instead of running open-loop
    # against stale visual context. Set <=0 to disable the watchdog.
    stale_action_threshold: float = field(
        default=0.5,
        metadata={
            "help": (
                "Seconds without receiving a new action chunk before the client "
                "flushes its action buffer and holds position (<=0 disables)"
            )
        },
    )

    # Aggregate function configuration (CLI-compatible)
    aggregate_fn_name: str = field(
        default="weighted_average",
        metadata={"help": f"Name of aggregate function to use. Options: {list(AGGREGATE_FUNCTIONS.keys())}"},
    )

    # Debug configuration
    debug_visualize_queue_size: bool = field(
        default=False, metadata={"help": "Visualize the action queue size"}
    )

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds"""
        return 1 / self.fps

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.server_address:
            raise ValueError("server_address cannot be empty")

        if not self.policy_type:
            raise ValueError("policy_type cannot be empty")

        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path cannot be empty")

        if not self.policy_device:
            raise ValueError("policy_device cannot be empty")

        if not self.client_device:
            raise ValueError("client_device cannot be empty")

        if self.chunk_size_threshold < 0 or self.chunk_size_threshold > 1:
            raise ValueError(f"chunk_size_threshold must be between 0 and 1, got {self.chunk_size_threshold}")

        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

        if self.control_fps <= 0:
            raise ValueError(f"control_fps must be positive, got {self.control_fps}")

        if self.control_fps < self.fps:
            raise ValueError(f"control_fps must be >= fps, got control_fps={self.control_fps}, fps={self.fps}")

        if self.control_fps % self.fps != 0:
            raise ValueError(
                f"control_fps must be an integer multiple of fps, got control_fps={self.control_fps}, "
                f"fps={self.fps}"
            )

        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")

        if self.yolo_conf < 0 or self.yolo_conf > 1:
            raise ValueError(f"yolo_conf must be between 0 and 1, got {self.yolo_conf}")

        if self.yolo_required_hits <= 0:
            raise ValueError(f"yolo_required_hits must be positive, got {self.yolo_required_hits}")

        if self.yolo_detect_every_n <= 0:
            raise ValueError(f"yolo_detect_every_n must be positive, got {self.yolo_detect_every_n}")

        if not 1 <= self.obs_jpeg_quality <= 100:
            raise ValueError(f"obs_jpeg_quality must be in [1, 100], got {self.obs_jpeg_quality}")

        if self.obs_target_height <= 0 or self.obs_target_width <= 0:
            raise ValueError(
                f"obs_target_height/width must be positive, got "
                f"({self.obs_target_height}, {self.obs_target_width})"
            )

        self.aggregate_fn = get_aggregate_function(self.aggregate_fn_name)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RobotClientConfig":
        """Create a RobotClientConfig from a dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            "server_address": self.server_address,
            "policy_type": self.policy_type,
            "pretrained_name_or_path": self.pretrained_name_or_path,
            "policy_device": self.policy_device,
            "client_device": self.client_device,
            "chunk_size_threshold": self.chunk_size_threshold,
            "fps": self.fps,
            "control_fps": self.control_fps,
            "actions_per_chunk": self.actions_per_chunk,
            "task": self.task,
            "debug_visualize_queue_size": self.debug_visualize_queue_size,
            "aggregate_fn_name": self.aggregate_fn_name,
            "yolo_enabled": self.yolo_enabled,
            "yolo_visualize": self.yolo_visualize,
            "yolo_model_path": self.yolo_model_path,
            "yolo_conf": self.yolo_conf,
            "yolo_required_hits": self.yolo_required_hits,
            "yolo_device": self.yolo_device,
            "yolo_detect_every_n": self.yolo_detect_every_n,
            "safety_camera": self.safety_camera,
            "obs_compression_enabled": self.obs_compression_enabled,
            "obs_jpeg_quality": self.obs_jpeg_quality,
            "obs_target_height": self.obs_target_height,
            "obs_target_width": self.obs_target_width,
            "obs_send_timeout": self.obs_send_timeout,
            "stale_action_threshold": self.stale_action_threshold,
        }

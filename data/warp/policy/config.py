from dataclasses import dataclass, field
import numpy as np


@dataclass
class G1Config:

    # PD gains per joint
    Kp: np.ndarray = field(default_factory=lambda: np.array([
        40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
        40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
        40.179,
        14.251, 14.251, 14.251, 14.251, 14.251,
        14.251, 14.251, 14.251, 14.251, 14.251,
    ]))
    Kd: np.ndarray = field(default_factory=lambda: np.array([
        2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
        2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
        2.558,
        0.907, 0.907, 0.907, 0.907, 0.907,
        0.907, 0.907, 0.907, 0.907, 0.907,
    ]))

    # default positions
    default_base_pos: np.ndarray = field(default_factory=lambda: np.array([
        0, 0, 0.78, 1, 0, 0, 0,
    ]))
    default_joint_pos: np.ndarray = field(default_factory=lambda: np.array([
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
         0.0,
         0.35,  0.18, 0.0, 0.87, 0.0,
         0.35, -0.18, 0.0, 0.87, 0.0,
    ]))

    # action scale
    action_scale: np.ndarray = field(default_factory=lambda: np.array([
        0.548, 0.351, 0.548, 0.351, 0.439, 0.439,
        0.548, 0.351, 0.548, 0.351, 0.439, 0.439,
        0.548,
        0.439, 0.439, 0.439, 0.439, 0.439,
        0.439, 0.439, 0.439, 0.439, 0.439,
    ]))

    # control
    control_dt: float = 0.02

    # gait
    gait_period: float = 0.6
    stand_cmd_threshold: float = 0.0

    # velocity command [vx, vy, omega]
    cmd: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.0, 0.0]))

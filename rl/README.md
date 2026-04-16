# Reinforcement Learning
Run all scripts from the project root.

## Environments
Custom Brax environments in `rl/envs/`, all inheriting from `brax.envs.base.PipelineEnv`:

| Environment | File | Description |
|---|---|---|
| `hopper` | `hopper_env.py` | Forward locomotion for the hopper |
| `hopper_tracking` | `hopper_tracking_env.py` | Velocity tracking variant (joystick-compatible) |
| `paddle_ball` | `paddle_ball_env.py` | Ball-paddle control |
| `paddle_ball_tracking` | `paddle_ball_tracking_env.py` | Tracking variant (joystick-compatible) |

Each environment defines a config dataclass (e.g. `HopperConfig`) specifying the model path, physics timestep, control rate, reward weights, and initial condition ranges.

Tracking environments accept an external velocity command and include it in the observation. During playback, commands can be sent via a gamepad joystick.

## Training
```bash
python rl/train_rl.py
```
Uses Brax PPO (`brax.training.agents.ppo`) with custom policy and value networks defined in `rl/algorithms/custom_networks.py`. Select the environment and hyperparameters at the top of `train_rl.py` by uncommenting the desired config block.

The `PPO_Train` class (`rl/algorithms/ppo_train.py`) wraps the Brax PPO trainer, handles logging to TensorBoard, and saves the trained policy as a `.pkl` file to `rl/policy/`.

## Playback
```bash
python rl/play_rl.py
```
Loads a trained policy and runs it in the interactive MuJoCo viewer with real-time physics. For tracking environments, a gamepad joystick (`rl/utils/utils.py`) can be used to send velocity commands during playback.

## Structure
```
rl/
├── train_rl.py              # Entry point for training
├── play_rl.py               # Entry point for interactive playback
├── envs/                    # Brax environment definitions
├── algorithms/
│   ├── ppo_train.py         # PPO training wrapper
│   ├── ppo_play.py          # Policy loading and inference
│   └── custom_networks.py   # MLP policy/value networks
├── policy/                  # Trained policy checkpoints (.pkl)
├── utils/
│   └── utils.py             # Joystick utility
└── log/                     # TensorBoard logs
```

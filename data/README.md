# Data Generation Pipeline
Run all scripts from the project root.

## 1) Generate Raw Data
`parallel_sim.py` contains a class that can perform three types of rollouts:
- **Zero Input Rollouts**: a zero input vector goes into the system.
- **Policy Input Rollouts**: load an RL policy and use this for closed loop control.
- **Random Input Rollouts**: sample a random input sequence and apply random control during rollout.

There are two backends:
- `data/mjx/parallel_sim.py` — MuJoCo MJX (JAX GPU) for hopper or paddle_ball.
- `data/warp/parallel_sim.py` — MuJoCo Warp (GPU) for the G1 humanoid.

This produces numbered training files: `{env_name}_data_01.npz`, `{env_name}_data_02.npz`, etc. A standalone testing dataset (`{env_name}_data_testing.npz`) is also automatically generated at the end of each run.

## 2) Parse Data
Parsing scripts combine raw data files and extract Poincare section data (states at pre-impact moments):
- `data/mjx/parse_hopper_data.py` — for the hopper.
- `data/mjx/parse_paddle_ball_data.py` — for the paddle ball.
- `data/warp/parse_g1_23dof_data.py` — for the G1 humanoid.

Each script:
1. Iterates over each numbered `_data_XX.npz` file and parses contact events to extract Poincare section data (states at pre-impact moments).
2. Consolidates all parsed results and saves as `{env_name}_data_poincare.npz`.
4. If a `{env_name}_data_testing.npz` file exists, it is automatically parsed the same way and saved as `{env_name}_data_poincare_testing.npz`.

## 3) Plot Data
- `plot_sim_data.py`: plots raw trajectory data from a single `_data_01.npz` file.
- `plot_sim_data_poincare.py`: plots the parsed Poincare section data from `_data_poincare.npz`.

Both plotting scripts exist under `data/mjx/` and `data/warp/`.

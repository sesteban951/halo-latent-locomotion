# standard imports
import numpy as np
import time
import os, sys
import matplotlib.pyplot as plt

# mujoco imports
import mujoco
import mujoco.viewer

# brax imports
from brax import envs

# change directories to project root (so `from rl...` works even if run from /data)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# custom imports
from rl.envs.hopper_env import HopperEnv
from rl.envs.hopper_tracking_env import HopperTrackingEnv
from rl.envs.paddle_ball_env import PaddleBallEnv
from rl.envs.paddle_ball_tracking_env import PaddleBallTrackingEnv
from rl.algorithms.ppo_play import PPO_Play


################################################################################
# UTILS
################################################################################

# loads every contact pair possible from the mujoco XML model
def load_touch_sensors(mj_model):
    """
    Load all touch sensors from a MuJoCo model.

    Args:
        mj_model: mujoco.MjModel

    Returns:
        touch_sensor_ids: list of int, indices of touch sensors
        touch_sensor_names: list of str, names of touch sensors
        num_touch_sensors: int, number of touch sensors
    """
    touch_sensor_ids = []
    touch_sensor_names = []

    for i, stype in enumerate(mj_model.sensor_type):
        if stype == mujoco.mjtSensor.mjSENS_TOUCH:
            touch_sensor_ids.append(i)
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            touch_sensor_names.append(name)

    num_touch_sensors = len(touch_sensor_ids)

    print(f"Found {num_touch_sensors} touch sensors: {touch_sensor_names}")
    return touch_sensor_ids, touch_sensor_names, num_touch_sensors


################################################################################
# MAIN PLOTTING
################################################################################

if __name__ == "__main__":

    # choose the environment
    # env_name = "paddle_ball"
    env_name = "hopper"

    # create the environment to get config info
    env = envs.get_environment(env_name)
    config = env.config

    # number of random trajectories to plot (use n_plot = 1 to see one trajectory for debugging)
    n_plot = 100

    # load the mujoco model
    model_path = config.model_path
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # load touch sensors if available
    touch_sensor_ids, touch_sensor_names, num_touch_sensors = load_touch_sensors(mj_model)

    # load the data
    if "tracking" in env_name:
        file_name = f"./data/mjx/raw_data/{env_name}_data_cmd_01.npz"
    else:
        file_name = f"./data/mjx/raw_data/{env_name}_data_01.npz"
    data = np.load(file_name)

    # access the arrays
    sim_dt = data['sim_dt']
    q_traj = data['q_log']
    v_traj = data['v_log']
    u_traj = data['u_log']
    c_traj = data['c_log']
    if "tracking" in env_name:
        cmd = data['cmd_log']

    # get the shape of the data
    batch_size, N_state, nq = q_traj.shape
    _, _, nv = v_traj.shape
    _, N_input, nu = u_traj.shape
    _, _, nc = c_traj.shape

    # print log shapes
    print(f"Simulation dt: {sim_dt}")
    print(f"Full q_traj shape: {q_traj.shape}")
    print(f"Full v_traj shape: {v_traj.shape}")
    print(f"Full u_traj shape: {u_traj.shape}")
    print(f"Full c_traj shape: {c_traj.shape}")
    if "tracking" in env_name:
        print(f"Full cmd shape: {cmd.shape}")

    # percent segments of the trajectory to use
    traj_segment_percent = (0.0, 1.0)  # (start, end) as percent of trajectory length
    start_idx = int(traj_segment_percent[0] * N_state)
    end_idx = int(traj_segment_percent[1] * N_state)
    q_traj = q_traj[:, start_idx:end_idx, :]
    v_traj = v_traj[:, start_idx:end_idx, :]
    u_traj = u_traj[:, start_idx:end_idx, :]
    c_traj = c_traj[:, start_idx:end_idx, :]

    N_state = q_traj.shape[1]
    N_input = u_traj.shape[1]
    print(f"Using trajectory segment from {start_idx} to {end_idx} (N_state = {N_state})")

    # select one random trajectory to playback
    traj_idx = np.random.randint(batch_size)
    print(f"Playing back trajectory {traj_idx} of {batch_size}")

    # wall clock timing variables
    t_sim = 0.0
    wall_start = time.time()
    last_render = 0.0

    # start the interactive simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:

        # Set camera parameters
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.8])   # look at x, y, z
        viewer.cam.distance = 3.0                           # distance from lookat
        viewer.cam.elevation = -20.0                        # tilt down/up
        viewer.cam.azimuth = 90.0                           # rotate around lookat

        # show the initial state
        mj_data.qpos[:] = q_traj[traj_idx, 0]
        mj_data.qvel[:] = v_traj[traj_idx, 0]
        mj_data.ctrl[:] = u_traj[traj_idx, 0]

        mujoco.mj_forward(mj_model, mj_data)  # recompute derived quantities
        viewer.sync()

        while viewer.is_running():

            # get the current sim time and state
            t_sim = mj_data.time

            # fix the state
            step_idx = int(t_sim / sim_dt)

            if step_idx >= N_state:
                break
            
            # hardcode the trajectory state for playback
            mj_data.qpos = q_traj[traj_idx, step_idx]
            mj_data.qvel = v_traj[traj_idx, step_idx]

            # step the simulation
            mujoco.mj_step(mj_model, mj_data)

            # sync the viewer
            viewer.sync()

            # sync the sim time with the wall clock time
            wall_elapsed = time.time() - wall_start
            if t_sim > wall_elapsed:
                time.sleep(t_sim - wall_elapsed)


    # alpha value for plotting trajectories (0 = transparent, 1 = solid)
    alpha_traj = 0.5 

    rng = np.random.default_rng(0)
    idxs = rng.choice(batch_size, size=min(n_plot, batch_size), replace=False)

    # time bases
    T_state = N_state
    T_input = N_input if N_input > 0 else N_state
    ctrl_dt = float(sim_dt) * (T_state / T_input)
    t_state   = np.arange(T_state) * float(sim_dt)
    t_input   = np.arange(T_input) * ctrl_dt
    t_contact = t_state

    # optional contact shift (if logged post-step)
    c_traj_plot = c_traj

    nrows = max(q_traj.shape[2], v_traj.shape[2], u_traj.shape[2], c_traj.shape[2])

    fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(18, 2.6 * nrows), sharex=True)
    if axes.ndim == 1:
        axes = axes[None, :]

    axes[0, 0].set_title("Positions (q)")
    axes[0, 1].set_title("Velocities (v)")
    axes[0, 2].set_title("Controls (u)")
    axes[0, 3].set_title(touch_sensor_names[0] if len(touch_sensor_names) else "Contacts (c)")

    show_legends = (len(idxs) <= 6)

    for i in range(nrows):
        ax_q, ax_v, ax_u, ax_c = axes[i, 0], axes[i, 1], axes[i, 2], axes[i, 3]

        # q[i]
        if i < q_traj.shape[2]:
            for idx in idxs:
                ax_q.plot(t_state, q_traj[idx, :, i], alpha=alpha_traj, label=f"traj {idx}")
            ax_q.set_ylabel(f"q[{i}]"); ax_q.grid(True, alpha=0.3)
            if show_legends and i == 0: ax_q.legend(frameon=False, loc="best")
        else:
            ax_q.axis("off")

        # v[i]
        if i < v_traj.shape[2]:
            for idx in idxs:
                ax_v.plot(t_state, v_traj[idx, :, i], alpha=alpha_traj)
            ax_v.set_ylabel(f"v[{i}]"); ax_v.grid(True, alpha=0.3)
        else:
            ax_v.axis("off")

        # u[i]
        if i < u_traj.shape[2]:
            for idx in idxs:
                ax_u.plot(t_input, u_traj[idx, :, i], alpha=alpha_traj)
            ax_u.set_ylabel(f"u[{i}]"); ax_u.grid(True, alpha=0.3)
        else:
            ax_u.axis("off")

        # c[i]
        if i < c_traj_plot.shape[2]:
            for idx in idxs:
                ax_c.plot(t_contact, c_traj_plot[idx, :, i], alpha=alpha_traj)
            ax_c.set_ylabel(f"c[{i}]"); ax_c.grid(True, alpha=0.3)
        else:
            ax_c.axis("off")

    for ax in axes.ravel():
        ax.set_xlabel("Time [s]")
        ax.tick_params(labelbottom=True)

    plt.tight_layout()
    plt.show()

    # ---------------------------
    # Phase plots (same alpha)
    # ---------------------------

    fig, axes = plt.subplots(nrows=q_traj.shape[2], ncols=1, figsize=(7, 2.6 * q_traj.shape[2]), sharex=False)
    if q_traj.shape[2] == 1:
        axes = np.array([axes])

    for i in range(q_traj.shape[2]):
        ax = axes[i]
        if i < v_traj.shape[2]:
            for idx in idxs:
                ax.plot(q_traj[idx, :, i], v_traj[idx, :, i], alpha=alpha_traj, label=f"traj {idx}")
            ax.set_xlabel(f"q[{i}]"); ax.set_ylabel(f"v[{i}]")
            ax.set_title(f"Phase Plot: q[{i}] vs v[{i}]"); ax.grid(True, alpha=0.3)
            if show_legends and i == 0: ax.legend(frameon=False, loc="best")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()
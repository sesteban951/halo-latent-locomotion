##
#
#  Plot and playback G1 simulation data
#
##

# standard imports
import numpy as np
import time
import os
import matplotlib.pyplot as plt

# mujoco imports
import mujoco
import mujoco.viewer


############################################################################
# JOINT NAMES
############################################################################

# joint names in actuator order (matches qpos[7:30] and qvel[6:29])
JOINT_NAMES = [
    "L hip pitch", "L hip roll", "L hip yaw", "L knee", "L ankle pitch", "L ankle roll",
    "R hip pitch", "R hip roll", "R hip yaw", "R knee", "R ankle pitch", "R ankle roll",
    "Waist yaw",
    "L shoulder pitch", "L shoulder roll", "L shoulder yaw", "L elbow", "L wrist roll",
    "R shoulder pitch", "R shoulder roll", "R shoulder yaw", "R elbow", "R wrist roll",
]

# base state labels
BASE_POS_LABELS = ["x", "y", "z", "qw", "qx", "qy", "qz"]
BASE_VEL_LABELS = ["vx", "vy", "vz", "wx", "wy", "wz"]


############################################################################
# MAIN
############################################################################

if __name__ == "__main__":

    # ---------------------- Config ----------------------

    xml_path = "./models/g1_23dof.xml"
    data_path = "./data/warp/raw_data/g1_23dof_data_01.npz"

    n_plot = 20          # number of trajectories to plot
    playback = True      # launch MuJoCo viewer for playback

    # trajectory segment to use (fraction of full length)
    traj_segment = (0.0, 1.0)

    # ---------------------- Load ----------------------

    data = np.load(data_path)
    sim_dt = float(data["sim_dt"])
    control_dt = float(data["control_dt"])
    q_traj = data["q_log"]  # (B, T, nq=30)
    v_traj = data["v_log"]  # (B, T, nv=29)
    c_traj = data["c_log"]  # (B, T, nc=2)

    batch_size, T_full, nq = q_traj.shape
    _, _, nv = v_traj.shape
    _, _, nc = c_traj.shape
    nu = nq - 7  # 23 joints

    print(f"Loaded: {data_path}")
    print(f"  sim_dt={sim_dt}, control_dt={control_dt}")
    print(f"  q_traj: {q_traj.shape}")
    print(f"  v_traj: {v_traj.shape}")
    print(f"  c_traj: {c_traj.shape}")

    # slice trajectory segment
    start_idx = int(traj_segment[0] * T_full)
    end_idx = int(traj_segment[1] * T_full)
    q_traj = q_traj[:, start_idx:end_idx, :]
    v_traj = v_traj[:, start_idx:end_idx, :]
    c_traj = c_traj[:, start_idx:end_idx, :]
    T = q_traj.shape[1]
    print(f"  Using steps [{start_idx}:{end_idx}], T={T}")

    # ---------------------- Playback ----------------------

    if playback:
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        mj_data = mujoco.MjData(mj_model)

        traj_idx = np.random.randint(batch_size)
        print(f"\nPlaying back trajectory {traj_idx} of {batch_size}")

        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:

            viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -20.0
            viewer.cam.azimuth = 135.0

            # show initial state
            mj_data.qpos[:] = q_traj[traj_idx, 0]
            mj_data.qvel[:] = v_traj[traj_idx, 0]
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            render_dt = 1.0 / 50.0  # render at 50 Hz
            render_skip = max(1, int(round(render_dt / sim_dt)))

            wall_start = time.time()
            for step_idx in range(T):
                if not viewer.is_running():
                    break

                mj_data.qpos[:] = q_traj[traj_idx, step_idx]
                mj_data.qvel[:] = v_traj[traj_idx, step_idx]

                # only render every render_skip steps
                if step_idx % render_skip == 0:
                    mujoco.mj_forward(mj_model, mj_data)
                    viewer.sync()

                    # real-time sync
                    t_sim = step_idx * sim_dt
                    wall_elapsed = time.time() - wall_start
                    if t_sim > wall_elapsed:
                        time.sleep(t_sim - wall_elapsed)

    # ---------------------- Select trajectories ----------------------

    rng = np.random.default_rng(0)
    idxs = rng.choice(batch_size, size=min(n_plot, batch_size), replace=False)
    alpha = 0.5
    show_legends = (len(idxs) <= 6)
    t_arr = np.arange(T) * sim_dt

    # ---------------------- Base state ----------------------

    fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(14, 2.2 * 7), sharex=True)

    for i in range(7):
        ax = axes[i, 0]
        for idx in idxs:
            ax.plot(t_arr, q_traj[idx, :, i], alpha=alpha)
        ax.set_ylabel(BASE_POS_LABELS[i])
        ax.grid(True, alpha=0.3)

    for i in range(6):
        ax = axes[i, 1]
        for idx in idxs:
            ax.plot(t_arr, v_traj[idx, :, i], alpha=alpha)
        ax.set_ylabel(BASE_VEL_LABELS[i])
        ax.grid(True, alpha=0.3)
    axes[6, 1].axis("off")

    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    fig.suptitle("Base State", y=0.98)
    plt.tight_layout()

    # ---------------------- Joint positions ----------------------

    cols = 6  # 6 joints per row (left leg, right leg, waist, arms)
    rows = -(-nu // cols)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(3.2 * cols, 2.2 * rows), sharex=True)
    axes = np.atleast_2d(axes)

    for j in range(nu):
        r, c = divmod(j, cols)
        ax = axes[r, c]
        for idx in idxs:
            ax.plot(t_arr, q_traj[idx, :, 7 + j], alpha=alpha)
        ax.set_ylabel(JOINT_NAMES[j], fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(nu, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [s]")
    fig.suptitle("Joint Positions (q)", y=0.98)
    plt.tight_layout()

    # ---------------------- Joint velocities ----------------------

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(3.2 * cols, 2.2 * rows), sharex=True)
    axes = np.atleast_2d(axes)

    for j in range(nu):
        r, c = divmod(j, cols)
        ax = axes[r, c]
        for idx in idxs:
            ax.plot(t_arr, v_traj[idx, :, 6 + j], alpha=alpha)
        ax.set_ylabel(JOINT_NAMES[j], fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(nu, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [s]")
    fig.suptitle("Joint Velocities (dq)", y=0.98)
    plt.tight_layout()

    # ---------------------- Phase plots ----------------------

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(3.2 * cols, 2.8 * rows))
    axes = np.atleast_2d(axes)

    for j in range(nu):
        r, c = divmod(j, cols)
        ax = axes[r, c]
        for idx in idxs:
            ax.plot(q_traj[idx, :, 7 + j], v_traj[idx, :, 6 + j], alpha=alpha)
        ax.set_xlabel(f"q", fontsize=8)
        ax.set_ylabel(f"dq", fontsize=8)
        ax.set_title(JOINT_NAMES[j], fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(nu, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    fig.suptitle("Phase Plots (q vs dq)", y=0.98)
    plt.tight_layout()

    # ---------------------- Contact forces ----------------------

    CONTACT_NAMES = ["Left foot", "Right foot"]

    fig, axes = plt.subplots(nrows=nc, ncols=1, figsize=(10, 2.2 * nc), sharex=True)
    if nc == 1:
        axes = [axes]

    for i in range(nc):
        ax = axes[i]
        for idx in idxs:
            ax.plot(t_arr, c_traj[idx, :, i], alpha=alpha)
        label = CONTACT_NAMES[i] if i < len(CONTACT_NAMES) else f"c[{i}]"
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Foot Contact Forces", y=0.98)
    plt.tight_layout()

    plt.show()

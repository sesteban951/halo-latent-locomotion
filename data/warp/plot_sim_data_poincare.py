# standard imports
import numpy as np
import matplotlib.pyplot as plt
import time

################################################################################
# LABELS
################################################################################

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

################################################################################
# MAIN PLOTTING
################################################################################

if __name__ == "__main__":

    # set the seed
    seed = int(time.time())
    np.random.seed(seed)

    # number of trajectories to plot
    num_plots = 20

    # load the poincare data
    env_name = "g1_23dof"
    data_path = f"./data/warp/raw_data/{env_name}_data_poincare.npz"

    # load the data
    data = np.load(data_path)

    # extract the data
    t_preimpact = data["t_data"]       # (B, nc_switch, K)
    q_preimpact = data["q_data"]       # (B, nc_switch, K, nq)
    v_preimpact = data["v_data"]       # (B, nc_switch, K, nv)
    switching_idx = data["switching_idx"]

    # print some shapes to verify
    print("t_preimpact shape:", t_preimpact.shape)
    print("q_preimpact shape:", q_preimpact.shape)
    print("v_preimpact shape:", v_preimpact.shape)
    print("switching_idx:", switching_idx)

    # choose the desired horizon to plot (e.g., first few impacts)
    horizon = q_preimpact.shape[2]

    # truncate to current horizon
    t_preimpact = t_preimpact[:, :, :horizon]
    q_preimpact = q_preimpact[:, :, :horizon, :]
    v_preimpact = v_preimpact[:, :, :horizon, :]
    print("t_preimpact shape:", t_preimpact.shape)
    print("q_preimpact shape:", q_preimpact.shape)
    print("v_preimpact shape:", v_preimpact.shape)

    # compute the time-to-impact intervals
    T_I = t_preimpact[:, :, 1:] - t_preimpact[:, :, :-1]  # (B, nc_switch, K-1)

    # number of trajectories to plot
    num_trajectories = t_preimpact.shape[0]
    num_plots = min(num_plots, num_trajectories)

    # randomly select trajectories to plot
    traj_idx = np.random.choice(num_trajectories, size=num_plots, replace=False)
    contact_channel = 1  # 0 = left foot, 1 = right foot
    q_pre = q_preimpact[traj_idx, contact_channel, :, :]  # (num_plots, K, nq)
    v_pre = v_preimpact[traj_idx, contact_channel, :, :]  # (num_plots, K, nv)

    # ------------------------------------- PLOT -------------------------------------

    # plot the time to impact intervals
    T_I_sel = T_I[traj_idx, contact_channel, :]          # (num_plots, K-1)
    colors = [plt.cm.tab10(i % 10) for i in range(num_plots)]
    x_seg = np.arange(1, T_I_sel.shape[1] + 1)

    # 1) Time-to-impact
    plt.figure()
    for n in range(num_plots):
        plt.plot(x_seg, T_I_sel[n], marker='o', color=colors[n])
    plt.xlabel("interval k")
    plt.ylabel("Dt_k [s]")
    plt.title(f"{env_name}: Time-to-Impact (channel {contact_channel})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 2) Base phase plots v_i vs q_i
    nq, nv = q_pre.shape[-1], v_pre.shape[-1]
    base_nd = min(6, nq, nv)
    base_cols = min(3, base_nd)
    base_rows = -(-base_nd // base_cols)  # ceil

    fig, axes = plt.subplots(base_rows, base_cols, figsize=(4 * base_cols, 3 * base_rows))
    axes = np.array(axes).reshape(base_rows, base_cols)

    for i in range(base_nd):
        r, c = divmod(i, base_cols)
        ax = axes[r, c]
        for n in range(num_plots):
            ax.plot(q_pre[n, :, i], v_pre[n, :, i], marker='o', color=colors[n])
        ax.set_xlabel(BASE_POS_LABELS[i])
        ax.set_ylabel(BASE_VEL_LABELS[i])
        ax.grid(True, alpha=0.3)

    # hide empties
    for j in range(base_nd, base_rows * base_cols):
        axes.flat[j].axis("off")

    fig.suptitle(f"{env_name}: Base Phase Plots (channel {contact_channel})", y=0.98)
    plt.tight_layout()

    # 3) Joint phase plots v_i vs q_i
    joint_q_start = 7
    joint_v_start = 6
    joint_nd = min(nq - joint_q_start, nv - joint_v_start)
    joint_cols = min(4, joint_nd)
    joint_rows = -(-joint_nd // joint_cols)  # ceil

    fig, axes = plt.subplots(joint_rows, joint_cols, figsize=(4 * joint_cols, 3 * joint_rows))
    axes = np.array(axes).reshape(joint_rows, joint_cols)

    for j in range(joint_nd):
        r, c = divmod(j, joint_cols)
        ax = axes[r, c]
        q_idx = joint_q_start + j
        v_idx = joint_v_start + j
        for n in range(num_plots):
            ax.plot(q_pre[n, :, q_idx], v_pre[n, :, v_idx], marker='o', color=colors[n])
        ax.set_xlabel(JOINT_NAMES[j])
        ax.set_ylabel(f"d{JOINT_NAMES[j]}")
        ax.grid(True, alpha=0.3)

    # hide empties
    for j in range(joint_nd, joint_rows * joint_cols):
        axes.flat[j].axis("off")

    fig.suptitle(f"{env_name}: Joint Phase Plots (channel {contact_channel})", y=0.98)
    plt.tight_layout()
    plt.show()

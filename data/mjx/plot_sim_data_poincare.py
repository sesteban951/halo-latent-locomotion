# standard imports
import numpy as np
import matplotlib.pyplot as plt
import time

################################################################################
# MAIN PLOTTING
################################################################################

if __name__ == "__main__":

    # set the seed
    # seed = 42
    seed = int(time.time())
    np.random.seed(seed)

    # number of trajectories to plot
    num_plots = 16

    # load the poincare data
    # env_name = "paddle_ball"
    env_name = "hopper"

    ################################################################################

    # determine if tracking data
    is_tracking_data = "tracking" in env_name

    if is_tracking_data:
        data_path = f"./data/mjx/raw_data/{env_name}_data_poincare_cmd.npz"
    else:
        data_path = f"./data/mjx/raw_data/{env_name}_data_poincare.npz"

    # load the data
    data = np.load(data_path)

    # extract the data
    t_preimpact = data["t_data"]       # (B, nc_switch, K)
    q_preimpact = data["q_data"]       # (B, nc_switch, K, nq)
    v_preimpact = data["v_data"]       # (B, nc_switch, K, nv)
    if is_tracking_data:
        cmd_preimpact = data["cmd_data"]   # (B, )

    # print some shapes to verify
    print("t_preimpact shape:", t_preimpact.shape)
    print("q_preimpact shape:", q_preimpact.shape)
    print("v_preimpact shape:", v_preimpact.shape)
    if is_tracking_data:
        print("cmd_preimpact shape:", cmd_preimpact.shape)

    # choose the desired horizon to plot (e.g., first switch)
    # horizon = 5
    horizon = q_preimpact.shape[2]

    # current horizon
    t_preimpact = t_preimpact[:, :, :horizon]         # (B, K)
    q_preimpact = q_preimpact[:, :, :horizon, :]      # (B, K, nq)
    v_preimpact = v_preimpact[:, :, :horizon, :]      # (B, K, nv)
    print("t_preimpact shape:", t_preimpact.shape)
    print("q_preimpact shape:", q_preimpact.shape)
    print("v_preimpact shape:", v_preimpact.shape)

    # compute the time-to-impact intervals
    T_I = t_preimpact[:, :, 1:] - t_preimpact[:, :, :-1]  # (B, K-1)

    # number of trajectories to plot
    num_trajectories = t_preimpact.shape[0]
    
    # randomly select a trajectories to plot
    traj_idx = np.random.choice(num_trajectories, size=num_plots, replace=False)
    contact_channel = 0  # WARNING: only one channel for now
    q_pre = q_preimpact[traj_idx, contact_channel, :, :]  # (num_plots, K, nq)
    v_pre = v_preimpact[traj_idx, contact_channel, :, :]  # (num_plots, K, nv)
    if is_tracking_data:
        cmd_pre = cmd_preimpact[traj_idx]  # (num_plots, )

    # ------------------------------------- PLOT -------------------------------------

    # plot the time to impact intervals
    switch_idx = 0
    T_I_sel = T_I[traj_idx, switch_idx, :]                  # (num_plots, K-1)
    colors = [plt.cm.tab10(i % 10) for i in range(num_plots)]
    x_seg = np.arange(1, T_I_sel.shape[1] + 1)

    # 1) Time-to-impact
    plt.figure()
    for n in range(num_plots):
        plt.plot(x_seg, T_I_sel[n], marker='o', color=colors[n])
    plt.xlabel("interval k")
    plt.ylabel("Δt_k [s]")
    plt.title(f"{env_name}: Time-to-Impact")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 2) Phase plots v_i vs q_i
    nq, nv = q_pre.shape[-1], v_pre.shape[-1]
    nd = min(nq, nv)
    cols = min(4, nd)
    rows = -(-nd // cols)  # ceil

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(nd):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        for n in range(num_plots):
            ax.plot(q_pre[n, :, i], v_pre[n, :, i], marker='o', color=colors[n])
        ax.set_xlabel(f"q[{i}]")
        ax.set_ylabel(f"v[{i}]")
        ax.grid(True, alpha=0.3)

    # hide empties
    for j in range(nd, rows*cols):
        axes.flat[j].axis('off')

    fig.suptitle(f"{env_name}: Phase Plots", y=0.98)
    plt.tight_layout()
    plt.show()

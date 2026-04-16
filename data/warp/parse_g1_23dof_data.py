##
#
#  For parsing G1 Warp data and computing Poincare data
#
##

# standard imports
import numpy as np

# for mujoco
import mujoco

##################################################################################
# UTILS
##################################################################################

# Tuple ([undesired contact idx], [switching index])
contact_idx_dict = {
    "g1_23dof": ([], [0, 1]),  # (None, Left foot / Right foot)
}


# load the touch sensor IDs and names from the mujoco model
def load_touch_sensors(mj_model):

    # load the touch sensors
    touch_sensor_ids = []
    touch_sensor_names = []
    for i, stype in enumerate(mj_model.sensor_type):
        if stype == mujoco.mjtSensor.mjSENS_TOUCH:
            touch_sensor_ids.append(i)
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            touch_sensor_names.append(name)

    # print the touch sensor info
    print(f"Found [{len(touch_sensor_ids)}] touch sensors:")
    for i, name in zip(touch_sensor_ids, touch_sensor_names):
        print(f"  Touch Sensor ID: {i}, Name: {name}")

    return touch_sensor_ids, touch_sensor_names


# transform base state from world frame to yaw-aligned right foot frame
def transform_base_to_yaw_foot_frame(q_data, v_data, mj_model, foot_site_name="right_foot"):
    """
    Transform base state (position, orientation, velocity) from world frame
    to a yaw-aligned right foot frame.

    The yaw-aligned foot frame has:
      - Origin at the right foot site position
      - z-axis pointing up (no roll/pitch)
      - x-axis aligned with the foot's heading (yaw only)

    Joint-level entries (q[7:], v[6:]) are unchanged.

    Args:
        q_data: (..., nq)  generalized positions  (freejoint: q[0:3]=pos, q[3:7]=quat w,x,y,z)
        v_data: (..., nv)  generalized velocities  (freejoint: v[0:3]=lin_vel world, v[3:6]=ang_vel body)
        mj_model: MuJoCo model
        foot_site_name: name of the foot site to align with

    Returns:
        q_out, v_out with same shapes, base entries transformed.
    """
    original_shape_q = q_data.shape
    original_shape_v = v_data.shape
    nq = original_shape_q[-1]
    nv = original_shape_v[-1]

    # flatten to (N, nq) and (N, nv) for iteration
    q_flat = q_data.reshape(-1, nq)
    v_flat = v_data.reshape(-1, nv)
    N = q_flat.shape[0]

    q_out = q_flat.copy()
    v_out = v_flat.copy()

    mj_data = mujoco.MjData(mj_model)
    foot_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, foot_site_name)
    if foot_site_id == -1:
        raise ValueError(f"Site '{foot_site_name}' not found in MuJoCo model.")

    R_base_buf = np.zeros(9)

    for i in range(N):
        # --- FK to get foot pose ---
        mj_data.qpos[:] = q_flat[i]
        mujoco.mj_kinematics(mj_model, mj_data)

        p_foot = mj_data.site_xpos[foot_site_id].copy()           # (3,)
        R_foot = mj_data.site_xmat[foot_site_id].reshape(3, 3)    # (3,3)

        # --- build R_yaw (yaw from foot heading, no roll/pitch) ---
        yaw = np.arctan2(R_foot[1, 0], R_foot[0, 0])
        cy, sy = np.cos(yaw), np.sin(yaw)
        R_yaw = np.array([[cy, -sy, 0.0],
                          [sy,  cy, 0.0],
                          [0.0, 0.0, 1.0]])

        # --- base rotation matrix from quaternion ---
        mujoco.mju_quat2Mat(R_base_buf, q_flat[i, 3:7])
        R_base = R_base_buf.reshape(3, 3)

        # --- position: relative to foot, in yaw frame ---
        q_out[i, 0:3] = R_yaw.T @ (q_flat[i, 0:3] - p_foot)

        # --- orientation: relative to yaw frame ---
        R_rel = R_yaw.T @ R_base
        quat_rel = np.zeros(4)
        mujoco.mju_mat2Quat(quat_rel, R_rel.flatten())
        q_out[i, 3:7] = quat_rel

        # --- linear velocity: rotated into yaw frame ---
        v_out[i, 0:3] = R_yaw.T @ v_flat[i, 0:3]

        # --- angular velocity: body frame → world frame → yaw frame ---
        omega_world = R_base @ v_flat[i, 3:6]
        v_out[i, 3:6] = R_yaw.T @ omega_world

    # reshape back to original
    q_out = q_out.reshape(original_shape_q)
    v_out = v_out.reshape(original_shape_v)

    return q_out, v_out


##################################################################################
# MAIN
##################################################################################

if __name__ == "__main__":

    # choose the dataset
    env_name = "g1_23dof"

    ##################################################################################

    # define the data path
    data_path = "./data/warp/raw_data"

    # ----------------------------------- MJ MODEL -----------------------------------

    # load the mujoco model
    model_path = f"./models/{env_name}.xml"
    mj_model = mujoco.MjModel.from_xml_path(model_path)

    # load the touch sensors
    touch_sensor_ids, touch_sensor_names = load_touch_sensors(mj_model)

    # define the undesired contact indices
    contact_idx_dicts = contact_idx_dict[env_name]

    def build_poincare_dataset(sim_dt, control_dt, q_data, v_data, c_data):

        sim_dt = np.round(float(sim_dt), 6)
        control_dt = np.round(float(control_dt), 6)

        print("Raw data summary:")
        print(f"  sim_dt: {sim_dt}")
        print(f"  control_dt: {control_dt}")
        print(f"  q_data shape: {q_data.shape}")
        print(f"  v_data shape: {v_data.shape}")
        print(f"  c_data shape: {c_data.shape}")

        time_array = np.arange(q_data.shape[1]) * sim_dt
        nq = q_data.shape[2]
        nv = v_data.shape[2]

        epsilon = 1e-6
        c_bool = (c_data > epsilon).astype(np.float32)
        undesired_idx = np.asarray(contact_idx_dicts[0], dtype=int)

        if len(undesired_idx) > 0:
            print("Processing undesired contacts...")

            threshold = 0.5
            c_traj = c_bool[:, :, undesired_idx]
            undesired_traj = (c_traj > threshold).any(axis=(1, 2))

            keep = ~undesired_traj
            n_removed = int(undesired_traj.sum())
            n_keep = int(keep.sum())
            n_old = c_data.shape[0]

            q_data = q_data[keep, :, :]
            v_data = v_data[keep, :, :]
            c_data = c_data[keep, :, :]
            c_bool = c_bool[keep, :, :]

            print("Undesired contact pruning. Summary:")
            print(f"  Removed [{n_removed} / {n_old} ({100 * (n_removed / n_old):.2f}%)] trajectories; kept [{n_keep}].")
            print(f"  q_data shape: {q_data.shape}")
            print(f"  v_data shape: {v_data.shape}")
            print(f"  c_data shape: {c_data.shape}")
        else:
            print("No undesired contacts to process.")

        # --- prune trajectories where base z-height drops below threshold ---
        z_height_min = 0.3
        base_z = q_data[:, :, 2]                          # (B, T_raw)
        fallen = (base_z < z_height_min).any(axis=1)       # (B,)
        n_fallen = int(fallen.sum())
        n_before = q_data.shape[0]
        keep_z = ~fallen

        q_data = q_data[keep_z, :, :]
        v_data = v_data[keep_z, :, :]
        c_data = c_data[keep_z, :, :]
        c_bool = c_bool[keep_z, :, :]

        print(f"Z-height pruning (threshold={z_height_min}m):")
        print(f"  Removed [{n_fallen} / {n_before} ({100 * (n_fallen / n_before):.2f}%)] trajectories; kept [{q_data.shape[0]}].")

        switching_idx = np.asarray(contact_idx_dicts[1], dtype=int)
        nc_switch = len(switching_idx)
        switching_channel_names = [touch_sensor_names[j] for j in switching_idx]
        if nc_switch == 0:
            raise ValueError("No switching indices defined for this environment.")

        c_data = c_data[:, :, switching_idx]
        c_bool = c_bool[:, :, switching_idx]
        print(f"Filtered contact data to switching indices: {switching_idx.tolist()}. New c_data shape: {c_data.shape}")

        transition_idx = c_bool[:, 1:, :] - c_bool[:, :-1, :]
        print("transition_idx shape:", transition_idx.shape)

        preimpact_idx = (transition_idx == +1).astype(np.int8)
        print("preimpact_idx shape:", preimpact_idx.shape)

        perimpacts_per_channel = preimpact_idx.sum(axis=1).astype(np.int64)
        print("perimpacts_per_channel shape:", perimpacts_per_channel.shape)

        min_preimpacts_per_channel = perimpacts_per_channel.min(axis=0)
        max_preimpacts_per_channel = perimpacts_per_channel.max(axis=0)
        print("Impact range per channel:")
        for i, name in enumerate(switching_channel_names):
            print(f"  Channel [{i}, {name}]: "
                  f"min preimpacts = {int(min_preimpacts_per_channel[i])}, "
                  f"max preimpacts = {int(max_preimpacts_per_channel[i])}")

        B, _, _ = c_data.shape
        K = int(min_preimpacts_per_channel.min())
        if K <= 0:
            raise ValueError("No pre-impacts found across all channels/batches (K=0).")

        t_preimpact = np.empty((B, nc_switch, K), dtype=time_array.dtype)
        q_preimpact = np.empty((B, nc_switch, K, nq), dtype=q_data.dtype)
        v_preimpact = np.empty((B, nc_switch, K, nv), dtype=v_data.dtype)

        for j in range(nc_switch):
            A = preimpact_idx[:, :, j]
            for b in range(B):
                k_list = np.flatnonzero(A[b])
                if k_list.size >= K:
                    k_sel = k_list[:K]
                elif k_list.size == 0:
                    k_sel = np.zeros(K, dtype=int)
                else:
                    pad = np.full(K - k_list.size, k_list[-1], dtype=int)
                    k_sel = np.concatenate([k_list, pad], axis=0)

                t_preimpact[b, j, :] = time_array[k_sel]
                q_preimpact[b, j, :, :] = q_data[b, k_sel, :]
                v_preimpact[b, j, :, :] = v_data[b, k_sel, :]

        print("Poincare pre-impact tensors built:")
        print("  t_preimpact :", t_preimpact.shape)
        print("  q_preimpact :", q_preimpact.shape)
        print("  v_preimpact :", v_preimpact.shape)

        # --- prune trajectories where right foot last Dt interval is too small ---
        gait_period = 0.6
        dt_min = gait_period / 2.0
        right_foot_ch = 1  # switching_idx: [0=left, 1=right]
        last_dt = t_preimpact[:, right_foot_ch, -1] - t_preimpact[:, right_foot_ch, -2]  # (B,)
        too_short = (last_dt < dt_min)
        n_too_short = int(too_short.sum())
        n_before_dt = t_preimpact.shape[0]
        keep_dt = ~too_short

        t_preimpact = t_preimpact[keep_dt, :, :]
        q_preimpact = q_preimpact[keep_dt, :, :, :]
        v_preimpact = v_preimpact[keep_dt, :, :, :]

        print(f"Last-interval pruning (right foot, threshold={dt_min}s):")
        print(f"  Removed [{n_too_short} / {n_before_dt} ({100 * (n_too_short / n_before_dt):.2f}%)] trajectories; kept [{t_preimpact.shape[0]}].")

        # --- prune trajectories with consecutive short TTI (chatter detection) ---
        tti_min = 0.1
        tti = np.diff(t_preimpact, axis=2)                          # (B, nc_switch, K-1)
        short = tti < tti_min                                       # boolean
        consecutive_short = short[:, :, :-1] & short[:, :, 1:]     # two in a row
        chattery = consecutive_short.any(axis=(1, 2))               # (B,)
        n_chattery = int(chattery.sum())
        n_before_tti = t_preimpact.shape[0]
        keep_tti = ~chattery

        t_preimpact = t_preimpact[keep_tti, :, :]
        q_preimpact = q_preimpact[keep_tti, :, :, :]
        v_preimpact = v_preimpact[keep_tti, :, :, :]

        print(f"Consecutive short-TTI pruning (threshold={tti_min}s, 2+ in a row):")
        print(f"  Removed [{n_chattery} / {n_before_tti} ({100 * (n_chattery / n_before_tti):.2f}%)] trajectories; kept [{t_preimpact.shape[0]}].")

        # --- drop the first impact point (transient from sim init) ---
        t_preimpact = t_preimpact[:, :, 1:]
        q_preimpact = q_preimpact[:, :, 1:, :]
        v_preimpact = v_preimpact[:, :, 1:, :]
        print(f"Dropped first impact point. New K={t_preimpact.shape[2]}.")

        # transform base state to yaw-aligned right foot frame
        print("Transforming base state to yaw-aligned right foot frame...")
        q_preimpact, v_preimpact = transform_base_to_yaw_foot_frame(
            q_preimpact, v_preimpact, mj_model)
        print("  Done. Base entries now in yaw-aligned right foot frame.")

        return {
            "sim_dt": sim_dt,
            "control_dt": control_dt,
            "undesired_idx": undesired_idx,
            "switching_idx": switching_idx,
            "t_data": t_preimpact,
            "q_data": q_preimpact,
            "v_data": v_preimpact,
        }

    def save_poincare_dataset(parsed, save_path):
        np.savez(save_path,
                 sim_dt=parsed["sim_dt"],
                 control_dt=parsed["control_dt"],
                 undesired_idx=parsed["undesired_idx"],
                 switching_idx=parsed["switching_idx"],
                 t_data=parsed["t_data"],
                 q_data=parsed["q_data"],
                 v_data=parsed["v_data"])
        print(f"Saved Poincare data to: {save_path}")

    def consolidate_poincare_datasets(parsed_datasets):
        if len(parsed_datasets) == 0:
            raise ValueError("No Poincaré datasets available for consolidation.")

        sim_dt = float(parsed_datasets[0]["sim_dt"])
        control_dt = float(parsed_datasets[0]["control_dt"])
        undesired_idx = parsed_datasets[0]["undesired_idx"]
        switching_idx = parsed_datasets[0]["switching_idx"]
        K = min(int(data["t_data"].shape[2]) for data in parsed_datasets)

        print(f"Consolidating [{len(parsed_datasets)}] parsed datasets with K=[{K}]...")

        return {
            "sim_dt": sim_dt,
            "control_dt": control_dt,
            "undesired_idx": undesired_idx,
            "switching_idx": switching_idx,
            "t_data": np.concatenate([data["t_data"][:, :, :K] for data in parsed_datasets], axis=0),
            "q_data": np.concatenate([data["q_data"][:, :, :K, :] for data in parsed_datasets], axis=0),
            "v_data": np.concatenate([data["v_data"][:, :, :K, :] for data in parsed_datasets], axis=0),
        }

    def process_training_dataset():
        print(f"\n{'='*60}")
        print(f"Processing data for [{env_name}]...")
        print(f"{'='*60}")

        max_num_files = 100
        parsed_datasets = []
        for i in range(max_num_files):
            raw_path = f"{data_path}/{env_name}_data_{i+1:02d}.npz"
            try:
                data = np.load(raw_path)
            except FileNotFoundError:
                break

            print(f"\nParsing raw file [{i+1:02d}] from: {raw_path}")
            parsed = build_poincare_dataset(float(data["sim_dt"]),
                                            float(data["control_dt"]),
                                            data["q_log"],
                                            data["v_log"],
                                            data["c_log"])
            parsed_datasets.append(parsed)

        if len(parsed_datasets) == 0:
            print(f"No data files found at {data_path} for env {env_name}.")
            exit(0)

        final_save_path = f"{data_path}/{env_name}_data_poincare.npz"
        consolidated = consolidate_poincare_datasets(parsed_datasets)
        save_poincare_dataset(consolidated, final_save_path)

    def process_testing_dataset():
        raw_path = f"{data_path}/{env_name}_data_testing.npz"
        try:
            data = np.load(raw_path)
        except FileNotFoundError:
            print(f"No testing data file found at {raw_path}. Skipping.")
            return

        print(f"\n{'='*60}")
        print(f"Processing data for [{env_name}_testing]...")
        print(f"{'='*60}")
        parsed = build_poincare_dataset(float(data["sim_dt"]),
                                        float(data["control_dt"]),
                                        data["q_log"],
                                        data["v_log"],
                                        data["c_log"])
        final_save_path = f"{data_path}/{env_name}_data_poincare_testing.npz"
        save_poincare_dataset(parsed, final_save_path)

    process_training_dataset()
    process_testing_dataset()

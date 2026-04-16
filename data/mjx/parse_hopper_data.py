##
#
#  For parsing data and computing Poincare data
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
    "hopper"      : ([0], [1]), # (Torso, Foot)
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

# load the position sensor IDs and names from the mujoco model
def load_position_sensors(mj_model):

    # load the position sensors (frame/world positions and joint positions)
    position_sensor_ids = []
    position_sensor_names = []
    for i, stype in enumerate(mj_model.sensor_type):
        if stype in (getattr(mujoco.mjtSensor, "mjSENS_FRAMEPOS", -1),
                     getattr(mujoco.mjtSensor, "mjSENS_JOINTPOS", -1)):
            position_sensor_ids.append(i)
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            position_sensor_names.append(name)

    # print the position sensor info
    print(f"Found [{len(position_sensor_ids)}] position sensors:")
    for i, name in zip(position_sensor_ids, position_sensor_names):
        print(f"  Position Sensor ID: {i}, Name: {name}")

    return position_sensor_ids, position_sensor_names


##################################################################################
# MAIN
##################################################################################

if __name__ == "__main__":

    # choose the dataset
    env_name = "hopper"

    ##################################################################################

    # determine if this is tracking data
    if "tracking" in env_name.lower():
        is_tracking_data = True
    else:
        is_tracking_data = False

    # define the data path
    data_path = f"./data/mjx/raw_data"

    # ----------------------------------- MJ MODEL -----------------------------------

    # get the mujoco model name
    if is_tracking_data:
        model_env_name = env_name.replace("_tracking", "")
    else:
        model_env_name = env_name

    # load the mujoco model
    model_path = f"./models/{model_env_name}.xml"
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # set the sim_dt to something really small to have small forward steps
    mj_model.opt.timestep = 1e-6

    # load the touch sensors
    touch_sensor_ids, touch_sensor_names = load_touch_sensors(mj_model)

    # load position sensors
    pos_sensor_ids, pos_sensor_names = load_position_sensors(mj_model)

    # define the undesired contact indices
    if is_tracking_data:
        contact_idx_dicts = contact_idx_dict[env_name.replace("_tracking", "")]
    else:
        contact_idx_dicts = contact_idx_dict[env_name]

    # grab the sensor IDs and addresses for torso and foot positions
    sid_torso = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_pos")
    sid_foot  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR,  "foot_pos")
    a_t, d_t  = mj_model.sensor_adr[sid_torso], mj_model.sensor_dim[sid_torso]  # d_t == 3
    a_f, d_f  = mj_model.sensor_adr[sid_foot],  mj_model.sensor_dim[sid_foot]   # d_f == 3

    def build_poincare_dataset(sim_dt, q_data, v_data, u_data, c_data, cmd_data=None):

        sim_dt = np.round(float(sim_dt), 6)

        print("Raw data summary:")
        print(f"  sim_dt: {sim_dt}")
        print(f"  q_data shape: {q_data.shape}")
        print(f"  v_data shape: {v_data.shape}")
        print(f"  u_data shape: {u_data.shape}")
        print(f"  c_data shape: {c_data.shape}")
        if is_tracking_data:
            print(f"  cmd_data shape: {cmd_data.shape}")

        time_array = np.arange(q_data.shape[1]) * sim_dt
        nq = q_data.shape[2]
        nv = v_data.shape[2]

        epsilon = 1e-6
        c_bool = (c_data > epsilon).astype(np.float32)
        undesired_idx = np.asarray(contact_idx_dicts[0], dtype=int)

        if len(undesired_idx) > 0:
            print("Processing undesired contacts...")

            threshold = 0.05
            c_traj = c_bool[:, :, undesired_idx]
            undesired_traj = (c_traj > threshold).any(axis=(1, 2))

            keep = ~undesired_traj
            n_removed = int(undesired_traj.sum())
            n_keep = int(keep.sum())
            n_old = c_data.shape[0]

            q_data = q_data[keep, :, :]
            v_data = v_data[keep, :, :]
            u_data = u_data[keep, :, :]
            c_data = c_data[keep, :, :]
            c_bool = c_bool[keep, :, :]
            if is_tracking_data:
                cmd_data = cmd_data[keep]

            print("Undesired contact pruning. Summary:")
            print(f"  Removed [{n_removed} / {n_old} ({100 * (n_removed / n_old):.2f}%)] trajectories; kept [{n_keep}].")
            print(f"  q_data shape: {q_data.shape}")
            print(f"  v_data shape: {v_data.shape}")
            print(f"  u_data shape: {u_data.shape}")
            print(f"  c_data shape: {c_data.shape}")
            if is_tracking_data:
                print(f"  cmd_data shape: {cmd_data.shape}")
        else:
            print("No undesired contacts to process.")

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
        k_preimpact_idx = np.empty((B, nc_switch, K), dtype=np.int64)
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

                k_preimpact_idx[b, j, :] = k_sel
                t_preimpact[b, j, :] = time_array[k_sel]
                q_preimpact[b, j, :, :] = q_data[b, k_sel, :]
                v_preimpact[b, j, :, :] = v_data[b, k_sel, :]

        print("Computing base position relative to foot at pre-impacts...")
        for j in range(nc_switch):
            for b in range(B):
                for k in range(K):
                    kk = int(k_preimpact_idx[b, j, k])

                    mj_data.qpos[:mj_model.nq] = q_data[b, kk, :mj_model.nq]
                    mj_data.qvel[:mj_model.nv] = v_data[b, kk, :mj_model.nv]
                    mujoco.mj_forward(mj_model, mj_data)

                    p_b_W = mj_data.sensordata[a_t:a_t+d_t]
                    p_g_W = mj_data.sensordata[a_f:a_f+d_f]
                    p_b_foot = p_b_W - p_g_W

                    q_preimpact[b, j, k, 0] = p_b_foot[0]
                    q_preimpact[b, j, k, 1] = p_b_foot[2]

        parsed = {
            "sim_dt": sim_dt,
            "undesired_idx": undesired_idx,
            "switching_idx": switching_idx,
            "t_data": t_preimpact,
            "q_data": q_preimpact,
            "v_data": v_preimpact,
        }
        if is_tracking_data:
            parsed["cmd_data"] = cmd_data

        print("Poincaré pre-impact tensors built:")
        print("  t_preimpact :", t_preimpact.shape)
        print("  q_preimpact :", q_preimpact.shape)
        print("  v_preimpact :", v_preimpact.shape)
        if is_tracking_data:
            print("  cmd_data    :", cmd_data.shape)

        return parsed

    def save_poincare_dataset(parsed, save_path):
        if is_tracking_data:
            np.savez(save_path,
                     sim_dt=parsed["sim_dt"],
                     undesired_idx=parsed["undesired_idx"],
                     switching_idx=parsed["switching_idx"],
                     t_data=parsed["t_data"],
                     q_data=parsed["q_data"],
                     v_data=parsed["v_data"],
                     cmd_data=parsed["cmd_data"])
        else:
            np.savez(save_path,
                     sim_dt=parsed["sim_dt"],
                     undesired_idx=parsed["undesired_idx"],
                     switching_idx=parsed["switching_idx"],
                     t_data=parsed["t_data"],
                     q_data=parsed["q_data"],
                     v_data=parsed["v_data"])
        print(f"Saved Poincaré data to: {save_path}")

    def consolidate_poincare_datasets(parsed_datasets):
        if len(parsed_datasets) == 0:
            raise ValueError("No Poincaré datasets available for consolidation.")

        sim_dt = float(parsed_datasets[0]["sim_dt"])
        undesired_idx = parsed_datasets[0]["undesired_idx"]
        switching_idx = parsed_datasets[0]["switching_idx"]
        K = min(int(data["t_data"].shape[2]) for data in parsed_datasets)

        print(f"Consolidating [{len(parsed_datasets)}] parsed datasets with K=[{K}]...")

        consolidated = {
            "sim_dt": sim_dt,
            "undesired_idx": undesired_idx,
            "switching_idx": switching_idx,
            "t_data": np.concatenate([data["t_data"][:, :, :K] for data in parsed_datasets], axis=0),
            "q_data": np.concatenate([data["q_data"][:, :, :K, :] for data in parsed_datasets], axis=0),
            "v_data": np.concatenate([data["v_data"][:, :, :K, :] for data in parsed_datasets], axis=0),
        }

        if is_tracking_data:
            consolidated["cmd_data"] = np.concatenate([data["cmd_data"] for data in parsed_datasets], axis=0)

        return consolidated

    def process_training_dataset():
        print(f"\n{'='*60}")
        print(f"Processing data for [{env_name}]...")
        print(f"{'='*60}")

        max_num_files = 100
        parsed_datasets = []
        for i in range(max_num_files):
            if is_tracking_data:
                raw_path = f"{data_path}/{env_name}_data_cmd_{i+1:02d}.npz"
            else:
                raw_path = f"{data_path}/{env_name}_data_{i+1:02d}.npz"

            try:
                data = np.load(raw_path)
            except FileNotFoundError:
                break

            print(f"\nParsing raw file [{i+1:02d}] from: {raw_path}")
            parsed = build_poincare_dataset(float(data["sim_dt"]),
                                            data["q_log"],
                                            data["v_log"],
                                            data["u_log"],
                                            data["c_log"],
                                            data["cmd_log"] if is_tracking_data else None)
            parsed_datasets.append(parsed)

        if len(parsed_datasets) == 0:
            print(f"No data files found at {data_path} for env {env_name}.")
            exit(0)

        if is_tracking_data:
            final_save_path = f"{data_path}/{env_name}_data_poincare_cmd.npz"
        else:
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
                                        data["q_log"],
                                        data["v_log"],
                                        data["u_log"],
                                        data["c_log"],
                                        data["cmd_log"] if is_tracking_data else None)

        if is_tracking_data:
            final_save_path = f"{data_path}/{env_name}_data_poincare_cmd_testing.npz"
        else:
            final_save_path = f"{data_path}/{env_name}_data_poincare_testing.npz"
        save_poincare_dataset(parsed, final_save_path)

    process_training_dataset()
    process_testing_dataset()

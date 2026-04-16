##
#
# Dataloader module for MJX data
#
##

# standard imports
import math
import numpy as np
from dataclasses import dataclass

# jax imports
import jax
import jax.numpy as jnp

################################################################################
# DATALOADER
################################################################################

@dataclass
class DataLoaderConfig:

    file_path: str                    # full path to the Poincare dataset .npz file
    batch_size: int                   # mini-batch size for training
    val_fraction: float = 0.1         # fraction of data to use for validation, in (0, 1)

# MJX DataLoader class for Poincare section data
class DataLoader:

    # initializer
    def __init__(self, config):

        # dataset file path
        self.file_path = config.file_path
        self.env_name  = config.file_path.split("/")[-1].replace(".npz", "")

        # train/val split configuration
        assert 0 < config.val_fraction < 1, f"val_fraction must be in (0, 1), got {config.val_fraction}"
        self.val_fraction = config.val_fraction

        # load the data
        self.load_data()

        # split into train/val
        total_samples = self.q_data.shape[0]
        val_size = int(total_samples * self.val_fraction)
        train_size = total_samples - val_size

        self.t_data_train = self.t_data[:train_size]
        self.q_data_train = self.q_data[:train_size]
        self.v_data_train = self.v_data[:train_size]

        self.t_data_val = self.t_data[train_size:]
        self.q_data_val = self.q_data[train_size:]
        self.v_data_val = self.v_data[train_size:]

        print(f"Split data: train={train_size}, val={val_size} (val_fraction={self.val_fraction})")

        # convert to jax arrays
        self.t_data_train = jnp.array(self.t_data_train)
        self.q_data_train = jnp.array(self.q_data_train)
        self.v_data_train = jnp.array(self.v_data_train)

        self.t_data_val = jnp.array(self.t_data_val)
        self.q_data_val = jnp.array(self.q_data_val)
        self.v_data_val = jnp.array(self.v_data_val)

        # get model dimensions
        self.nq = self.q_data_train.shape[2]
        self.nv = self.v_data_train.shape[2]
        self.nx = self.nq + self.nv

        # compute the final dataset statistics
        self.B_train = self.q_data_train.shape[0]
        self.B_val = self.q_data_val.shape[0]
        self.T = self.q_data_train.shape[1]

        # mini batch size
        self.batch_size = config.batch_size
        if self.batch_size > self.B_train:
            raise ValueError(f"batch_size ({self.batch_size}) cannot exceed training dataset size ({self.B_train}).")

        # Training epoch state
        self.train_epoch_ptr = 0
        self.train_epoch_perm = None
        self.train_epoch_num = 0
        self._train_need_reset = True

        # Validation epoch state
        self.val_epoch_ptr = 0
        self.val_epoch_perm = None
        self.val_epoch_num = 0
        self._val_need_reset = True

        # print final data stats
        print(f"Final training data shape: [B = {self.B_train}, T = {self.T}]")
        print(f"  t_data_train shape: [{self.t_data_train.shape}]")
        print(f"  q_data_train shape: [{self.q_data_train.shape}]")
        print(f"  v_data_train shape: [{self.v_data_train.shape}]")
        print(f"Final validation data shape: [B = {self.B_val}, T = {self.T}]")
        print(f"  t_data_val shape: [{self.t_data_val.shape}]")
        print(f"  q_data_val shape: [{self.q_data_val.shape}]")
        print(f"  v_data_val shape: [{self.v_data_val.shape}]")
        print(f"Training:   [{math.floor(self.B_train / self.batch_size)}] batches/epoch (B=[{self.B_train}], batch_size=[{self.batch_size}])")
        print(f"Validation: [{math.floor(self.B_val / self.batch_size)}] batches/epoch (B=[{self.B_val}], batch_size=[{self.batch_size}])")

    # load the data
    def load_data(self):

        # try loading the data
        try:
            data = np.load(self.file_path)
        except FileNotFoundError:
            print(f"DataLoader: No data file found at path: [{self.file_path}].")
            exit(0)
        
        # extract the data
        t_preimpact = data["t_data"]         # (B, ns, K)
        q_preimpact = data["q_data"]         # (B, ns, K, nq)
        v_preimpact = data["v_data"]         # (B, ns, K, nv)

        # choose the contact channel to use, NOTE: just choose the default first one for now
        t_preimpact = t_preimpact[:, 0, :]    # (B, K)
        q_preimpact = q_preimpact[:, 0, :, :] # (B, K, nq)
        v_preimpact = v_preimpact[:, 0, :, :] # (B, K, nv)

        # store the data to the class
        self.t_data = t_preimpact
        self.q_data = q_preimpact
        self.v_data = v_preimpact

        # print data stats
        print("Data Loader: found data file.")
        print(f"  Data name: [{self.file_path}]")
        print(f"  Total trajectories: [{self.q_data.shape[0]}]")
        print(f"  Native Horizon length:  [{self.q_data.shape[1]}]")

    #####################################  UTILS  ##########################################

    # compute stats of the data
    def compute_data_stats(self):
        """Compute statistics from the training data"""
        
        # Always use the "train" version since that's where all data goes
        q_data_np = np.array(self.q_data_train)
        v_data_np = np.array(self.v_data_train)

        # compute the mean of the data
        q_mean = jnp.mean(q_data_np, axis=(0, 1))  # (nq,)
        v_mean = jnp.mean(v_data_np, axis=(0, 1))  # (nv,)
        x_mean = jnp.concatenate((q_mean, v_mean), axis=-1)  # (nx,)

        # compute the covariance matrix of the data
        q_std = jnp.std(q_data_np, axis=(0, 1))    # (nq,)
        v_std = jnp.std(v_data_np, axis=(0, 1))    # (nv,)
        x_std = jnp.concatenate((q_std, v_std), axis=-1)  # (nx,)

        # compute the upper and lower bound vectors
        q_lb = jnp.min(q_data_np, axis=(0, 1))  # (nq,)
        q_ub = jnp.max(q_data_np, axis=(0, 1))  # (nq,)
        v_lb = jnp.min(v_data_np, axis=(0, 1))  # (nv,)
        v_ub = jnp.max(v_data_np, axis=(0, 1))  # (nv,)
        x_lb = jnp.concatenate((q_lb, v_lb), axis=-1)  # (nx,)
        x_ub = jnp.concatenate((q_ub, v_ub), axis=-1)  # (nx,)

        return x_mean, x_std, x_lb, x_ub

    # reset training epoch
    def reset_train_epoch(self, rng):
        self.train_epoch_num += 1
        rng, rng_perm = jax.random.split(rng)
        self.train_epoch_perm = jax.random.permutation(rng_perm, self.B_train)
        self.train_epoch_ptr = 0
        self._train_need_reset = False
        return rng

    # reset validation epoch
    def reset_val_epoch(self, rng):
        self.val_epoch_num += 1
        rng, rng_perm = jax.random.split(rng)
        self.val_epoch_perm = jax.random.permutation(rng_perm, self.B_val)
        self.val_epoch_ptr = 0
        self._val_need_reset = False
        return rng

    # sample training data
    def sample_train_data(self, rng):

        if (self.train_epoch_perm is None) or self._train_need_reset:
            rng = self.reset_train_epoch(rng)

        mb = self.batch_size
        start = self.train_epoch_ptr
        end = start + mb

        if end <= self.B_train:
            idx = self.train_epoch_perm[start:end]
            self.train_epoch_ptr = end
            if self.train_epoch_ptr == self.B_train:
                self._train_need_reset = True
                self.train_epoch_ptr = 0
        else:
            tail_idx = self.train_epoch_perm[start:self.B_train]
            head_needed = end - self.B_train
            
            rng, rng_perm = jax.random.split(rng)
            next_perm = jax.random.permutation(rng_perm, self.B_train)
            head_idx = next_perm[:head_needed]
            idx = jnp.concatenate([tail_idx, head_idx], axis=0)
            
            self.train_epoch_perm = next_perm
            self.train_epoch_ptr = head_needed
            self._train_need_reset = False

        q_b = self.q_data_train[idx]
        v_b = self.v_data_train[idx]
        x_b = jnp.concatenate((q_b, v_b), axis=-1)

        return x_b, rng

    # sample validation data
    def sample_val_data(self, rng):

        if (self.val_epoch_perm is None) or self._val_need_reset:
            rng = self.reset_val_epoch(rng)

        mb = self.batch_size
        start = self.val_epoch_ptr
        end = start + mb

        if end <= self.B_val:
            idx = self.val_epoch_perm[start:end]
            self.val_epoch_ptr = end
            if self.val_epoch_ptr == self.B_val:
                self._val_need_reset = True
                self.val_epoch_ptr = 0
        else:
            tail_idx = self.val_epoch_perm[start:self.B_val]
            head_needed = end - self.B_val
            
            rng, rng_perm = jax.random.split(rng)
            next_perm = jax.random.permutation(rng_perm, self.B_val)
            head_idx = next_perm[:head_needed]
            idx = jnp.concatenate([tail_idx, head_idx], axis=0)
            
            self.val_epoch_perm = next_perm
            self.val_epoch_ptr = head_needed
            self._val_need_reset = False

        q_b = self.q_data_val[idx]
        v_b = self.v_data_val[idx]
        x_b = jnp.concatenate((q_b, v_b), axis=-1)
        return x_b, rng


################################################################################
# EXAMPLE USAGE
################################################################################

if __name__ == "__main__":

    # import matplotlib for plotting
    import matplotlib.pyplot as plt

    print("="*80)
    print("TEST 1: Train/Val split")
    print("="*80)

    # some example usage
    file_path = "./data/mjx/raw_data/paddle_ball_data_poincare.npz"
    # file_path = "./data/mjx/raw_data/hopper_data_poincare.npz"
    batch_size = 64
    dataset_label = file_path.split("/")[-1].replace(".npz", "")

    # create the dataloader config
    config = DataLoaderConfig(
        file_path=file_path,
        batch_size=batch_size,
        val_fraction=0.1
    )

    # create the dataloader
    dataloader2 = DataLoader(config=config)

    # sample training batches
    rng = jax.random.PRNGKey(42)
    x_train_1, rng = dataloader2.sample_train_data(rng)
    x_train_2, rng = dataloader2.sample_train_data(rng)

    # sample validation batches
    x_val_1, rng = dataloader2.sample_val_data(rng)
    x_val_2, rng = dataloader2.sample_val_data(rng)

    print(f"\nTraining batch shapes:")
    print(f"  x_train_1 shape: {x_train_1.shape}")
    print(f"  x_train_2 shape: {x_train_2.shape}")
    
    print(f"\nValidation batch shapes:")
    print(f"  x_val_1 shape: {x_val_1.shape}")
    print(f"  x_val_2 shape: {x_val_2.shape}")

    # verify train and val are different
    print(f"\nTrain vs Val check:")
    print(f"  x_train_1 equals x_val_1: {jnp.array_equal(x_train_1, x_val_1)}")
    print(f"  norm of (x_train_1 - x_val_1): {jnp.linalg.norm(x_train_1 - x_val_1):.2f}")

    # verify batches within same set are different
    print(f"\nBatch diversity check:")
    print(f"  x_train_1 equals x_train_2: {jnp.array_equal(x_train_1, x_train_2)}")
    print(f"  x_val_1 equals x_val_2: {jnp.array_equal(x_val_1, x_val_2)}")

    print("\n" + "="*80)
    print("TEST 2: compute_data_stats()")
    print("="*80)

    x_mean, x_std, x_lb, x_ub = dataloader2.compute_data_stats()

    print(f"\nData stats shapes (expected nx={dataloader2.nx}):")
    print(f"  x_mean shape: {x_mean.shape}  (expected ({dataloader2.nx},))")
    print(f"  x_std  shape: {x_std.shape}   (expected ({dataloader2.nx},))")
    print(f"  x_lb   shape: {x_lb.shape}    (expected ({dataloader2.nx},))")
    print(f"  x_ub   shape: {x_ub.shape}    (expected ({dataloader2.nx},))")

    shapes_ok      = all(a.shape == (dataloader2.nx,) for a in [x_mean, x_std, x_lb, x_ub])
    no_nans        = not any(jnp.any(jnp.isnan(a)) for a in [x_mean, x_std, x_lb, x_ub])
    std_nonneg     = jnp.all(x_std >= 0)
    bounds_ok      = jnp.all(x_lb <= x_ub)
    mean_in_bounds = jnp.all((x_mean >= x_lb) & (x_mean <= x_ub))

    print(f"\nSanity checks:")
    print(f"  All shapes == (nx,):         {shapes_ok}")
    print(f"  No NaNs in any stat:         {no_nans}")
    print(f"  x_std >= 0 everywhere:       {bool(std_nonneg)}")
    print(f"  x_lb <= x_ub everywhere:     {bool(bounds_ok)}")
    print(f"  x_mean within [x_lb, x_ub]: {bool(mean_in_bounds)}")

    print("\n" + "="*80)
    print("TEST 3: Epoch iteration test")
    print("="*80)

    # test iterating through a full epoch
    rng = jax.random.PRNGKey(99)
    num_train_batches = math.ceil(dataloader2.B_train / batch_size)
    num_val_batches = math.ceil(dataloader2.B_val / batch_size)
    
    print(f"\nIterating through full training epoch ({num_train_batches} batches)...")
    for i in range(int(num_train_batches * 20)):
        x_b, rng = dataloader2.sample_train_data(rng)
        if i == 0 or i == num_train_batches - 1:
            print(f"  Batch {i+1}/{num_train_batches}: shape {x_b.shape}")
    
    print(f"\nIterating through full validation epoch ({num_val_batches} batches)...")
    for i in range(int(num_val_batches * 20)):
        x_b, rng = dataloader2.sample_val_data(rng)
        if i == 0 or i == num_val_batches - 1:
            print(f"  Batch {i+1}/{num_val_batches}: shape {x_b.shape}")

    print("\n" + "="*80)
    print("TEST 4: Visualization")
    print("="*80)
    
    # plot phase plot of first trajectory in a training batch
    x_sample, _ = dataloader2.sample_train_data(jax.random.PRNGKey(0))
    q_ = x_sample[0, :, 0]  # (T,)
    v_ = x_sample[0, :, dataloader2.nq]  # velocity starts at index nq
    
    plt.figure(figsize=(10, 6))
    plt.plot(q_, v_, "o-", markersize=4, linewidth=1.5)
    plt.xlabel("Position q[0]")
    plt.ylabel("Velocity v[0]")
    plt.title(f"Phase Plot - {dataset_label} (Training Sample)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\n" + "="*80)
    print("TEST 5: Reproducibility")
    print("="*80)

    # same rng key + forced reset should produce identical batches
    dataloader2._train_need_reset = True
    x_a, _ = dataloader2.sample_train_data(jax.random.PRNGKey(7))
    dataloader2._train_need_reset = True
    x_b, _ = dataloader2.sample_train_data(jax.random.PRNGKey(7))

    dataloader2._val_need_reset = True
    x_val_a, _ = dataloader2.sample_val_data(jax.random.PRNGKey(7))
    dataloader2._val_need_reset = True
    x_val_b, _ = dataloader2.sample_val_data(jax.random.PRNGKey(7))

    print(f"\nSame rng key → same train batch: {jnp.array_equal(x_a, x_b)}")
    print(f"Same rng key → same val batch:   {jnp.array_equal(x_val_a, x_val_b)}")

    # different rng keys should (almost certainly) produce different batches
    dataloader2._train_need_reset = True
    x_c, _ = dataloader2.sample_train_data(jax.random.PRNGKey(0))
    dataloader2._train_need_reset = True
    x_d, _ = dataloader2.sample_train_data(jax.random.PRNGKey(1))
    print(f"Different rng keys → different train batches: {not jnp.array_equal(x_c, x_d)}")

    print("\n" + "="*80)
    print("TEST 6: Train/Val partition integrity")
    print("="*80)

    total = dataloader2.B_train + dataloader2.B_val
    actual_val_frac = dataloader2.B_val / total
    sizes_consistent = (dataloader2.t_data_train.shape[0] == dataloader2.B_train and
                        dataloader2.t_data_val.shape[0]   == dataloader2.B_val)

    print(f"\nPartition sizes:")
    print(f"  B_train: {dataloader2.B_train}")
    print(f"  B_val:   {dataloader2.B_val}")
    print(f"  Total:   {total}")
    print(f"\nVal fraction: {actual_val_frac:.4f} (requested {dataloader2.val_fraction:.4f})")
    print(f"t/q/v shapes consistent with B_train/B_val: {sizes_consistent}")

    # verify no raw-data row is shared between splits (sequential split means no overlap)
    t_train_np = np.array(dataloader2.t_data_train)
    t_val_np   = np.array(dataloader2.t_data_val)
    overlap = np.intersect1d(t_train_np[:, 0], t_val_np[:, 0])
    print(f"Overlapping first-timestamps between train and val: {len(overlap)} (expected 0)")

    print("\nAll tests completed!")

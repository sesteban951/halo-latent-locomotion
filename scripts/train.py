##
#
#  Main script to demonstrate ROM Learning
#
##

# standard imports
import time

# suppress jax warnings ("All configs were filtered out because...")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all,1=INFO,2=WARNING,3=ERROR

# jax imports
import jax

# add project root to path
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# custom imports
from utils.auto_encoder import AutoEncoderConfig, AutoEncoder
from utils.training import TrainConfig, Trainer
from utils.dataloader import DataLoaderConfig, DataLoader


############################################################################
# MAIN
############################################################################

if __name__ == "__main__":

    # print the device being used (gpu or cpu)
    device = jax.devices()[0]
    print("Device type:", device.platform)      # e.g. 'gpu' or 'cpu'
    print("Device name:", device.device_kind)   # e.g. 'NVIDIA GeForce RTX 4090'

    # -------------------------- Random Seed --------------------------- #

    # set random seed
    # seed = 42
    seed = int(time.time())
    rng = jax.random.PRNGKey(seed)

    # ------------------------- Dynamics Model -------------------------- #

    # Paddle Ball
    file_path = "./data/mjx/raw_data/paddle_ball_data_poincare.npz"
    num_steps = 5_000
    batch_size = 512
    T = 6
    nz = 2
    enc_layer_sizes = [64, 32, 16]
    dec_layer_sizes = [16, 32, 64]
    dyn_layer_sizes = [64, 64, 64]

    # Hopper
    # file_path = "./data/mjx/raw_data/hopper_data_poincare.npz"
    # num_steps = 15_000
    # batch_size = 1024
    # T = 8
    # nz = 4
    # enc_layer_sizes = [64, 32, 16]
    # dec_layer_sizes = [16, 32, 64]
    # dyn_layer_sizes = [64, 64, 64]

    # G1 Humanoid
    # file_path = "./data/warp/raw_data/g1_23dof_data_poincare.npz"
    # num_steps = 30_000
    # batch_size = 512
    # T = 8
    # nz = 12
    # enc_layer_sizes = [256, 128, 64]
    # dec_layer_sizes = [64, 128, 256]
    # dyn_layer_sizes = [128, 128, 128]

    # -------------------------- Training Setup -------------------------- #

    # training config
    train_config = TrainConfig(
        num_steps=num_steps,    # number of training steps
        batch_size=batch_size,  # training batch size
        T=T,                    # trajectory length (is clipped if shorter in dataloader)
        normalize_data=True,    # whether to normalize data or not
        std_floor=1e-6,         # std floor value for normalizer
        eps=1e-6,               # small constant to avoid division by zero
        learning_rate=1e-3,     # learning rate
        lambda_x=1.0,           # weight for x reconstruction loss
        lambda_z=1.0,           # weight for z reconstruction loss
        lambda_fwd=1.0,         # weight for latent forward conjugate loss
        lambda_bck=1.0,         # weight for latent backward conjugate loss
        lambda_pred=1.0,        # weight for prediction loss
        lambda_iso=1.0,         # weight for isometry loss
        lambda_reg=1e-6,        # weight for L2 regularization (not on biases)
        log_every=100,          # log training progress every n steps
        val_every=1_000,        # log validation metrics every n steps
        print_model=False,      # print model summary in the beginning
    )

    # DataLoader
    dl_config = DataLoaderConfig(
        file_path=file_path,
        batch_size=train_config.batch_size,
        val_fraction=0.10
    )
    dataloader = DataLoader(config=dl_config)

    # autoencoder config
    ae_config = AutoEncoderConfig(
        nx=dataloader.nx,
        nz=nz,
        enc_layer_sizes=enc_layer_sizes,
        dec_layer_sizes=dec_layer_sizes,
        dyn_layer_sizes=dyn_layer_sizes,
        ae_activation_fn_name="swish",
        dyn_activation_fn_name="swish"
    )

    # instantiate autoencoder model
    auto_encoder = AutoEncoder(config=ae_config)

    # instantiate trainer
    trainer = Trainer(
        rng=rng,                      # random number generator
        model=auto_encoder,           # autoencoder model
        training_config=train_config, # training hyperparameters
        dataloader=dataloader         # DataLoader for data generation
    )

    # ----------------------------- Train ----------------------------- #

    # start training
    params_trained = trainer.train()

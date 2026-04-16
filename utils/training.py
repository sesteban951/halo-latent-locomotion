##
#
# Training Class for ROM Learning using AE
#
##

# standard imports
import numpy as np
import math
from typing import Optional

# for logging
from tensorboardX import SummaryWriter  # tensorboard writer
from datetime import datetime           # timestamping logged data
import os                               # for path handling
import dataclasses                      # for dataclass conversion

# jax imports
import jax                      
import jax.numpy as jnp         # standard jax numpy
from functools import partial   # for partial function application

# flax imports
import flax
from flax import linen as nn           # neural network library
from flax import struct                # immutable dataclass
from flax.training import train_state  # simple train state for optimization

# optax imports
import optax         # gradient processing and optimization library

# for saving model parameters
import json

# add project root to path
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# custom imports
from utils.dataloader  import *  # dataloader for pre generated data


############################################################################
# DATA NORMALIZATION  (per-feature: shapes (1,1,nx))
############################################################################

@struct.dataclass
class NormalizerStats:
    mean: jnp.ndarray      # (1,1,nx)
    std:  jnp.ndarray      # (1,1,nx)
    std_floor: float = 1e-6
    eps: float = 1e-6

class Normalizer:
    def __init__(self,
                 stats: Optional[NormalizerStats] = None,
                 std_floor: float = 1e-6,
                 eps: float = 1e-6):
        self.std_floor = std_floor
        self.eps = eps
        self.stats: Optional[NormalizerStats] = stats

    @classmethod
    def compute_stats(cls, xb, eps: float = 1e-6, std_floor: float = 1e-6):
        # xb: (B, T, nx)
        if xb.ndim != 3:
            raise ValueError("xb must have shape (B, T, nx)")
        # Per-feature statistics over both batch and time -> (1,1,nx)
        mean = jnp.mean(xb, axis=(0, 1), keepdims=True)        # (1,1,nx)
        std  = jnp.std (xb, axis=(0, 1), keepdims=True)        # (1,1,nx)
        std  = jnp.maximum(std, std_floor)
        stats = NormalizerStats(mean=mean, std=std, std_floor=std_floor, eps=eps)

        print("Data Normalizer (per-feature) initialized.")
        print(f"  Mean min: {float(jnp.min(mean)):.4g}, Mean max: {float(jnp.max(mean)):.4g}")
        print(f"  Std  min: {float(jnp.min(std )):.4g}, Std  max: {float(jnp.max(std )):.4g}")
        print(f"  Std floor: {stats.std_floor}, Eps: {stats.eps}")
        return stats

    def set_stats(self, xb):
        self.stats = self.compute_stats(xb, eps=self.eps, std_floor=self.std_floor)

    def normalize(self, x_bt: jnp.ndarray) -> jnp.ndarray:
        # x_bt: (B,T,nx)
        return (x_bt - self.stats.mean) / (self.stats.std + self.stats.eps)

    def denormalize(self, xn_bt: jnp.ndarray) -> jnp.ndarray:
        return xn_bt * (self.stats.std + self.stats.eps) + self.stats.mean

############################################################################
# LOSS FUNCTION
############################################################################


# reconstruction loss function, left and right losses
def reconstruction_loss_fn(params, model, input_data, config):

    # sizes
    B, T, nx = input_data.shape  # (B, T, nx)

    # flatten data
    x = jnp.reshape(input_data, (B * T, nx))  # Original data

    # pass through encoder and decoder
    z     = model.apply({"params": params}, x, method=model.encode)   
    x_hat = model.apply({"params": params}, z, method=model.decode)   
    z_hat = model.apply({"params": params}, x_hat, method=model.encode)

    # compute residuals
    x_residuals_ = jnp.sum((x - x_hat) ** 2, axis=-1) # (B*T,)
    z_residuals_ = jnp.sum((z - z_hat) ** 2, axis=-1) # (B*T,)

    # denominator for averaging
    denom = B * T

    # x reconstruction loss, λ_x * (1/(B*T)) * Σ  ‖xₜ - x̂ₜ‖²
    loss_x = config.lambda_x * x_residuals_.sum() / denom

    # z reconstruction loss, λ_z * (1/(B*T)) * Σ  ‖zₜ - ẑₜ‖²
    loss_z = config.lambda_z * z_residuals_.sum() / denom

    return loss_x, loss_z


# dynamics loss function, forward and backward conjugacy losses
def dynamics_loss_fn(params, model, input_data, config):

    # sizes
    B, T, nx = input_data.shape  # (B, T, nx)

    # shift trajectory to get (x_t, x_t1) pairs
    x_t = input_data[:, :-1, :]    # shape (B, T-1, nx)
    x_t1 = input_data[:, 1:, :]    # shape (B, T-1, nx)

    # flatten data
    x_t = jnp.reshape(x_t, (B * (T-1), nx))   # Original data at time t
    x_t1 = jnp.reshape(x_t1, (B * (T-1), nx)) # Original data at time t+1

    # encode to latent space
    z_t  = model.apply({"params": params}, x_t,  method=model.encode)
    z_t1 = model.apply({"params": params}, x_t1, method=model.encode)

    # latent dynamics forward step
    z_t1_hat = model.apply({"params": params}, z_t, method=model.latent_step)

    # decode to reconstructed FOM state
    x_t1_hat = model.apply({"params": params}, z_t1_hat, method=model.decode)

    # compute residuals
    fwd_residuals_ = jnp.sum((z_t1 - z_t1_hat) ** 2, axis=-1)  # (B*(T-1),)
    bck_residuals_ = jnp.sum((x_t1 - x_t1_hat) ** 2, axis=-1)  # (B*(T-1),)

    # denominator for averaging
    denom = B * (T - 1)

    # Forward conjugacy loss, λ_fwd * (1/(B*(T-1))) * Σ  ‖zₜ₊₁ − ẑₜ₊₁‖²
    loss_fwd = config.lambda_fwd * fwd_residuals_.sum() / denom

    # Backward conjugacy loss, λ_bck * (1/(B*(T-1))) * Σ  ‖xₜ₊₁ − x̂ₜ₊₁‖²
    loss_bck = config.lambda_bck * bck_residuals_.sum() / denom

    return loss_fwd, loss_bck


# prediction loss function
def prediction_loss_fn(params, model, input_data, config):

    # sizes
    B, T, nx = input_data.shape  # (B, T, nx)
    nz = model.config.nz

    # get the initial conditions and encode
    x0 = input_data[:, 0, :]  # shape (B, nx)
    z0 = model.apply({"params": params}, x0, method=model.encode) # shape (B, nz)

    # rollout in latent space
    def scan_step(z, _):
        # single latent step
        z_next = model.apply({"params": params}, z, method=model.latent_step) # (B, nz)
        return z_next, z_next # carry, y_t
    
    # scan over time steps
    _, z_t = jax.lax.scan(scan_step, z0, None, length=T-1)  # z_t shape (T-1, B, nz)

    # swap axes
    z_t = jnp.transpose(z_t, (1, 0, 2))   # shape (B, T-1, nz)

    # add the initial condition at the start
    z_t = jnp.concatenate([jnp.expand_dims(z0, axis=1), z_t], axis=1)  # shape (B, T, nz)

    # decode all latent states to FOM space
    z_t_flat = z_t.reshape((B * T, nz))  # shape (B*T, nz)
    x_hat_t_flat = model.apply({"params": params}, z_t_flat, method=model.decode) # shape (B*T, nx)
    x_hat_t = x_hat_t_flat.reshape((B, T, nx))  # shape (B, T, nx)

    # get the true FOM states
    x_t = input_data  # shape (B, T, nx)

    # remove the initial condition from both
    x_t = x_t[:, 1:, :]          # shape (B, T-1, nx)
    x_hat_t = x_hat_t[:, 1:, :]  # shape (B, T-1, nx)
    
    # compute prediction residuals
    pred_residuals = jnp.sum((x_t - x_hat_t) ** 2, axis=-1) # (B, T-1)

    # denominator for averaging
    denom = B * (T - 1)

    # Prediction loss, λ_pred * (1/(B*(T-1))) * Σ  ‖xₜ − x̂ₜ‖²
    loss_pred = config.lambda_pred * pred_residuals.sum() / denom

    return loss_pred


# Isometry loss function (penalize deviation of latent covariance from identity)
def isometry_loss_fn(params, model, input_data, config):

    # sizes
    B, T, nx = input_data.shape
    nz = model.config.nz
    denom = B * T

    # encode
    x = input_data.reshape(B*T, nx)
    z = model.apply({"params": params}, x, method=model.encode)
    z = z.reshape(B, T, nz)

    # compute mean and center
    mu = jnp.sum(z, axis=(0, 1)) / denom  # (nz,)
    zc = z - mu
    
    # compute covariance matrix
    C = jnp.einsum('btn,btm->nm', zc, zc) / denom  # (nz, nz)

    # target: identity covariance, using the Frobenius norm
    I_z = jnp.eye(nz)
    loss_iso = config.lambda_iso * jnp.sum((C - I_z)**2) 

    return loss_iso


# Regularization losses
def regularization_loss_fn(params, config):

    # # L1 model param regularization (no reg on biases), λ_reg * Σ |θ|
    # l1 = sum(jnp.sum(jnp.abs(p)) for p in jax.tree_util.tree_leaves(params) if p.ndim > 1)
    # loss_reg = config.lambda_reg * l1
    
    # L2 model param regularization (no reg on biases), λ_reg * Σ ‖θ‖²
    l2 = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params) if p.ndim > 1)
    loss_reg = config.lambda_reg * l2

    return loss_reg


# main loss function
def loss_fn(params, model, input_data, config):
    """
    Returns:
        total_loss, aux
    """

    # Reconstruction losses
    loss_x, loss_z = reconstruction_loss_fn(params, model, input_data, config)

    # Dynamics losses
    loss_fwd, loss_bck = dynamics_loss_fn(params, model, input_data, config)

    # Prediction loss
    loss_pred = prediction_loss_fn(params, model, input_data, config)

    # Isometry loss
    loss_iso = isometry_loss_fn(params, model, input_data, config)

    # Regularization loss
    loss_reg = regularization_loss_fn(params, config)

    # total loss
    total_loss = (loss_x +    # x reconstruction loss
                  loss_z +    # z reconstruction loss
                  loss_fwd +  # forward conjugacy loss
                  loss_bck +  # backward conjugacy loss
                  loss_pred + # prediction loss
                  loss_iso +  # isometry loss
                  loss_reg)   # regularization on network parameters loss

    # auxiliary info
    aux = (loss_x, 
           loss_z, 
           loss_fwd, 
           loss_bck, 
           loss_pred,
           loss_iso, 
           loss_reg)

    return total_loss, aux

############################################################################
# TRAINER
############################################################################

# struct to hold training parameters
@struct.dataclass
class TrainConfig:

    # number of total steps
    num_steps: int     # total number of training steps
    batch_size: int    # mini-batch size for training

    # trajectory parameters
    T: int             # maximum trajectory length in the dataset

    # normalization params
    normalize_data: bool  # whether to normalize data or not
    std_floor: float      # std floor value for normalizer
    eps: float            # small constant to avoid division by zero

    # learning rate
    learning_rate: float

    # losses
    lambda_x: float    # x reconstruction loss weight (left)
    lambda_z: float    # z latent space loss weight   (right)
    lambda_fwd: float  # latent dynamics forward conjugacy loss weight
    lambda_bck: float  # latent dynamics backward conjugacy loss weight
    lambda_pred: float # prediction loss weight
    lambda_iso: float  # isometry loss weight
    lambda_reg: float  # L2 regularization loss weight

    # print printing
    print_model: bool    # print model summary at start of training
    log_every: int       # print metrics every n steps
    
    # validation parameters
    val_every: int       # evaluate validation every n steps


# struct to hold metrics
@struct.dataclass
class Metrics:

    # progress metrics
    step: int        # current training step

    # loss metrics
    loss: float      # total loss          
    loss_x: float    # x recon loss         L_x    = λ_x   · (1/(B·T)) · Σ_{b,t} ‖x_t - x̂_t‖²
    loss_z: float    # z recon loss         L_z    = λ_z   · (1/(B·T)) · Σ_{b,t} ‖z_t - ẑ_t‖²
    loss_fwd: float  # forward conj loss    L_fwd  = λ_fwd · (1/(B·(T-1))) · Σ_{b,t} ‖z_{t+1} − ẑ_{t+1}‖²
    loss_bck: float  # backward conj loss   L_bck  = λ_bck · (1/(B·(T-1))) · Σ_{b,t} ‖x_{t+1} − x̂_{t+1}‖²
    loss_pred: float # prediction loss      L_pred = λ_pred · (1/(B·(T-1))) · Σ_{b,t} ‖x_t − x̂_t‖²,  t ∈ [1,T-1]
    loss_iso: float  # isometry loss        L_iso  = λ_iso · ‖C - I‖_F²,  C = Cov(z), the sample covariance
    loss_reg: float  # regularization loss  L_reg  = λ_reg · Σ_{W} ‖W‖²  (weights only)

    # gradient norms
    grad_norm: float     # gradient norm          ‖g‖₂ = ‖∇_θ L‖₂
    update_norm: float   # parameter update norm  ‖Δθ‖₂ = ‖θₖ₊₁ − θₖ‖₂


# simple trainer class to handle training
class Trainer:

    # initialize trainer
    def __init__ (self,
                  rng: jax.random.PRNGKey,
                  model: nn.Module,
                  training_config: TrainConfig,
                  dataloader: DataLoader):

        # model to train
        self.model = model

        # load the configs
        self.training_config = training_config

        # dataloader
        self.dataloader = dataloader
        if self.training_config.T > self.dataloader.T:
            raise ValueError(f"Desired training trajectory length [T={self.training_config.T}] "
                             f"exceeds the maximum available trajectory length [T={self.dataloader.T}] in the dataset.")

        # get dynamical system name
        self.system_name = self.dataloader.env_name

        # print epoch info
        train_batches_per_epoch = math.floor(self.dataloader.B_train / self.dataloader.batch_size)
        val_batches_per_epoch = math.floor(self.dataloader.B_val / self.dataloader.batch_size)
        train_epochs = self.training_config.num_steps / max(train_batches_per_epoch, 1)
        val_epochs = self.training_config.num_steps / max(val_batches_per_epoch, 1)
        print(f"Training:   [{round(train_epochs)}] total epochs ({self.training_config.num_steps} steps / {train_batches_per_epoch} batches/epoch)")
        print(f"Validation: [{round(val_epochs)}] total epochs ({self.training_config.num_steps} steps / {val_batches_per_epoch} batches/epoch)")

        # random key
        rng_data_gen, rng_init, rng_tab = jax.random.split(rng, 3)
        self.rng_data_gen = rng_data_gen

        # initialize model parameters
        dummy_input_x = jnp.zeros((1, self.model.config.nx))
        dummy_input_z = jnp.zeros((1, self.model.config.nz))
        variables = self.model.init(rng_init, 
                                    dummy_input_x, dummy_input_z,
                                    method=self.model.init_modules)
        params = variables["params"]

        # setup the optimizer
        optimizer = optax.adam(learning_rate=self.training_config.learning_rate)
        # optimizer = optax.chain(
        #     optax.clip_by_global_norm(1.0),                        
        #     optax.adamw(learning_rate=self.training_config.learning_rate, weight_decay=1e-4),
        # )

        # create the train state
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer
        )
            
        # initialize data normalizer
        if self.training_config.normalize_data == True:
            
            print("Computing normalizer statistics from training set...")
            # collect all training data for better stats
            all_train_data = []
            temp_rng = jax.random.PRNGKey(0)
            
            num_train_batches = math.ceil(self.dataloader.B_train / self.dataloader.batch_size)
            for _ in range(num_train_batches):
                x_batch, temp_rng = self.generate_train_data(temp_rng)
                all_train_data.append(x_batch)
            
            # concatenate all training data
            all_train_data = jnp.concatenate(all_train_data, axis=0)  # (B_train, T, nx)
            
            # initialize the normalizer with all training data
            self.normalizer = Normalizer(std_floor=self.training_config.std_floor, 
                                        eps=self.training_config.eps)
            self.normalizer.set_stats(all_train_data)
            
            print(f"Normalizer computed from {all_train_data.shape[0]} training samples.")
        else:
            self.normalizer = None
        
        # logging with Tensorboard
        self.initialize_logging()

        # print model summary
        if training_config.print_model:
            print(self.model.tabulate({'params': rng_tab}, 
                                        dummy_input_x, dummy_input_z,
                                        method=self.model.init_modules, depth=2))


    # function that generates TRAINING data
    def generate_train_data(self, rng):
        """Generate a batch of training data"""
        # get a batch of training data
        x_t_batch, rng_new = self.dataloader.sample_train_data(rng)  # shape (batch_size, T_data, nx)

        # crop to desired trajectory length
        x_t_batch = x_t_batch[:, :self.training_config.T, :]

        return x_t_batch, rng_new


    # function that generates VALIDATION data
    def generate_val_data(self, rng):
        """Generate a batch of validation data"""
        # get a batch of validation data
        x_t_batch, rng_new = self.dataloader.sample_val_data(rng)  # shape (batch_size, T_data, nx)

        # crop to desired trajectory length
        x_t_batch = x_t_batch[:, :self.training_config.T, :]

        return x_t_batch, rng_new


    # single training step function
    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def train_step(self, state, input_data, step):
        """
        Single training step

        Args:
            state:  Current TrainState (model + optimizer)
            input_data: FOM Input state trajectory, shape (mini_batch_size, N, nx)
            step:   Current training step (for logging)
        Returns:
            new_state: Updated TrainState after applying gradients
            metrics:   Metrics dataclass with training info
        """

        # define a loss function wrapper to compute gradients
        def loss_fn_wrap(params):
            return loss_fn(params,               # model parameters to optimize
                           self.model,           # model instance
                           input_data,           # input data
                           self.training_config) # training config
        
        # make a function that computes the loss and its gradients
        grad_fn = jax.value_and_grad(loss_fn_wrap,      # function that only takes the model parameters as input
                                     has_aux=True)      # tells JAX that loss_fn returns auxiliary data in addition to the loss value
        
        # compute the loss and gradients w.r.t. the model parameters
        (loss, aux), grads = grad_fn(state.params)
        loss_x, loss_z, loss_fwd, loss_bck, loss_pred, loss_iso, loss_reg = aux

        # apply the gradients to update the model parameters
        new_state = state.apply_gradients(grads=grads)

        # compute the gradient norm and update norm for logging
        grad_norm = optax.global_norm(grads)              
        delta_params = jax.tree_util.tree_map(lambda new, old: new - old,
                                                     new_state.params, state.params)
        update_norm = optax.global_norm(delta_params)

        # update the metrics
        metrics = Metrics(step=step, 
                          loss=loss, 
                          loss_x=loss_x,
                          loss_z=loss_z,
                          loss_fwd=loss_fwd,
                          loss_bck=loss_bck,
                          loss_pred=loss_pred,
                          loss_iso=loss_iso,
                          loss_reg=loss_reg,
                          grad_norm=grad_norm,
                          update_norm=update_norm)

        return new_state, metrics


    # evaluation step function
    @partial(jax.jit, static_argnums=(0,))
    def eval_step(self, state, val_data):
        """
        Evaluate the model on validation data
        
        Args:
            state: Current TrainState
            val_data: Validation data, shape (batch_size, T, nx)
        Returns:
            val_metrics: Dictionary of validation metrics
        """
        # compute loss without gradients
        loss, aux = loss_fn(state.params, self.model, val_data, self.training_config)
        loss_x, loss_z, loss_fwd, loss_bck, loss_pred, loss_iso, loss_reg = aux
        
        return {
            'val_loss': loss,
            'val_loss_x': loss_x,
            'val_loss_z': loss_z,
            'val_loss_fwd': loss_fwd,
            'val_loss_bck': loss_bck,
            'val_loss_pred': loss_pred,
            'val_loss_iso': loss_iso,
            'val_loss_reg': loss_reg,
        }
    

    # main training loop
    def train(self):
        """
        Main training loop for the model.
        """

        # print starting info
        print("Starting training...")
        print("Training Start Time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        t0 = datetime.now()

        # loop over training steps
        for step in range(1, self.training_config.num_steps + 1):

            # generate a mini-batch of TRAINING data
            input_data, self.rng_data_gen = self.generate_train_data(self.rng_data_gen)  # shape (batch_size, T, nx)

            # normalize data if needed
            if self.training_config.normalize_data == True:
                input_data = self.normalizer.normalize(input_data)
            
            # perform a single training step
            self.state, metrics = self.train_step(self.state, input_data, step)

            # log the training metrics
            if (step % self.training_config.log_every == 0) or (step == 1):
                self.log_metrics(step, metrics, prefix="train")

            # log the validation metrics
            if (step % self.training_config.val_every == 0) or (step == 1):
                # generate a mini-batch of VALIDATION data
                val_data, self.rng_data_gen = self.generate_val_data(self.rng_data_gen)
                
                # normalize validation data if needed
                if self.training_config.normalize_data == True:
                    val_data = self.normalizer.normalize(val_data)
                
                # evaluate on validation data
                val_metrics = self.eval_step(self.state, val_data)
                self.log_val_metrics(step, val_metrics)

        # training done
        t1 = datetime.now()
        delta_t = t1 - t0
        minutes, seconds = divmod(delta_t.total_seconds(), 60)
        print("Training complete time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"Training total time: {int(minutes):02d}:{int(seconds):02d} (mm:ss)")

        # close the tensorboard writer
        self.writer.flush()
        self.writer.close()

        # save the training
        self.save_training()

        return self.state.params
    

    # initialize tensorboard logging
    def initialize_logging(self):

        # check that log directory exists, if not, create it
        log_path = "./scripts/log"
        if not os.path.exists(log_path):
            # create the directory
            os.makedirs(log_path)
            print(f"Created directory: [{log_path}]")

        # current date and time
        current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        # log path name
        log_file = f"{log_path}/ae_{self.system_name}_{current_datetime}_nz_{self.model.config.nz}"

        # create logging objects
        self.writer = SummaryWriter(log_file) # tensorboard writer
        self.times = [datetime.now()]         # time stamps
        self.history = {}                     # history dictionary for storing metrics

        # print logging path info
        print(f"Logging to: [{log_file}].")

    # log the metrics at the current step
    def log_metrics(self, step, metrics, prefix="train"):

        # print the metrics
        print(
            f"Step {step:05d} | "
            f"L_tot = {metrics.loss:.4f}, "
            f"L_x = {metrics.loss_x:.4f}, "
            f"L_z = {metrics.loss_z:.4f}, "
            f"L_fwd = {metrics.loss_fwd:.4f}, "
            f"L_bck = {metrics.loss_bck:.4f}, "
            f"L_pred = {metrics.loss_pred:.4f}, "
            f"L_iso = {metrics.loss_iso:.4f}, "
            f"L_reg = {metrics.loss_reg:.4f} | "
            f"‖g‖ = {metrics.grad_norm:.4f}, "
            f"‖Δθ‖ = {metrics.update_norm:.4f} "
        )

        # convert metrics to dictionary for logging
        metrics_dict = dataclasses.asdict(metrics)
        metrics_dict = jax.device_get(metrics_dict)  # move to CPU if needed

        # timestamps
        self.times.append(datetime.now())

        # sanitize
        step = int(step)
        clean = {k: float(np.asarray(v)) for k, v in metrics_dict.items() if k != "step"}

        # log to tensorboard
        for k, v in clean.items():
            self.writer.add_scalar(f"{prefix}/{k}", v, step)

        # flush the writer
        self.writer.flush()

        # save to history
        self.history.setdefault("step", []).append(step)
        for k, v in clean.items():
            self.history.setdefault(k, []).append(v)

    # log validation metrics
    def log_val_metrics(self, step, val_metrics):
        """Log validation metrics to tensorboard and print"""
        
        # print validation metrics
        print(
            f"Val Step {step:05d} | "
            f"L_tot = {val_metrics['val_loss']:.4f}, "
            f"L_x = {val_metrics['val_loss_x']:.4f}, "
            f"L_z = {val_metrics['val_loss_z']:.4f}, "
            f"L_fwd = {val_metrics['val_loss_fwd']:.4f}, "
            f"L_bck = {val_metrics['val_loss_bck']:.4f}, "
            f"L_pred = {val_metrics['val_loss_pred']:.4f}, "
            f"L_iso = {val_metrics['val_loss_iso']:.4f}, "
            f"L_reg = {val_metrics['val_loss_reg']:.4f}"
        )
        
        # move to CPU if needed
        val_metrics = jax.device_get(val_metrics)
        
        # log to tensorboard
        for k, v in val_metrics.items():
            self.writer.add_scalar(f"validation/{k}", float(v), step)
        
        # flush
        self.writer.flush()
        
        # save to history
        for k, v in val_metrics.items():
            self.history.setdefault(k, []).append(float(v))


    # save the training
    def save_training(self):

        # check if the save diretocry exits, if not, create it
        save_path = "./scripts/params"
        if not os.path.exists(save_path):
            # create the directory
            os.makedirs(save_path)
            print(f"Created directory: [{save_path}]")

        # current date and time
        current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        # log path name
        save_file = f"{save_path}/ae_{self.system_name}_{current_datetime}_nz_{self.model.config.nz}"
        os.makedirs(save_file, exist_ok=True)  
        
        # params -> msgpack
        params_host = jax.device_get(self.state.params)
        with open(os.path.join(save_file, "model_params.msgpack"), "wb") as f:
            f.write(flax.serialization.to_bytes(params_host))

        # model_config -> JSON
        model_config = dataclasses.asdict(self.model.config)
        with open(os.path.join(save_file, "model_config.json"), "w") as f:
            json.dump(model_config, f, indent=2)

        # training_config -> JSON
        training_config = dataclasses.asdict(self.training_config)
        with open(os.path.join(save_file, "training_config.json"), "w") as f:
            json.dump(training_config, f, indent=2)

        # normalizer_stats -> JSON (if present)
        if (self.normalizer is not None) and (self.normalizer.stats is not None):
            norm_path = os.path.join(save_file, "normalizer_stats.json")
            norm_stats = {
                "mean": jax.device_get(self.normalizer.stats.mean).tolist(),   # (1,1,nx)
                "std":  jax.device_get(self.normalizer.stats.std ).tolist(),   # (1,1,nx)
                "std_floor": float(self.normalizer.stats.std_floor),
                "eps": float(self.normalizer.stats.eps),
            }
            with open(norm_path, "w") as f:
                json.dump(norm_stats, f, indent=2)

        print(f"Training params saved to: [{save_file}].")
        return save_file
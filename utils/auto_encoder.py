##
#
# Auto Encoder Model for ROM Learning
#
##

# jax imports
import jax.numpy as jnp
from typing import Sequence

# flax imports
from flax import linen as nn           # neural network library
from flax import struct                # immutable dataclass


############################################################################
# UTILS
############################################################################

# activation function loader
def load_activation(name: str):
    act_name = name.lower()
    if   act_name == "relu":     return nn.relu
    elif act_name == "leaky_relu": return nn.leaky_relu
    elif act_name == "elu":      return nn.elu
    elif act_name == "gelu":     return nn.gelu
    elif act_name == "swish":    return nn.swish
    elif act_name == "tanh":     return nn.tanh
    elif act_name == "sigmoid":  return nn.sigmoid
    elif act_name == "selu":     return nn.selu
    raise ValueError(f"Unknown activation: {act_name}")

# kernel initializer loader
def load_kernel(act_name: str):
    """Sane initializer pairing."""
    act_name = act_name.lower()
    if act_name in ("relu", "leaky_relu", "elu", "gelu", "swish"):
        return nn.initializers.he_uniform()      # or he_normal()
    elif act_name == "tanh":
        return nn.initializers.glorot_normal()   # or glorot_uniform()
    elif act_name == "sigmoid":
        return nn.initializers.glorot_normal()
    elif act_name == "selu":
        return nn.initializers.lecun_uniform()
    
    # fallback
    return nn.initializers.glorot_normal()

############################################################################
# AUTOENCODER
############################################################################

# config for the AutoEncoder model
@struct.dataclass
class AutoEncoderConfig:
   
    nx: int                         # Dimension of the FOM state
    nz: int                         # Dimension of the ROM state
    enc_layer_sizes: Sequence[int]  # Encoder layer sizes
    dec_layer_sizes: Sequence[int]  # Decoder layer sizes
    dyn_layer_sizes: Sequence[int]  # Latent dynamics model layer sizes
    ae_activation_fn_name: str      # Activation function name
    dyn_activation_fn_name: str     # Activation function name


# AutoEncoder model
class AutoEncoder(nn.Module):

    # hyperparameters config
    config: AutoEncoderConfig

    # initialize all modules
    def init_modules(self, x, z):
        _ = self(x)              # encoder+decoder
        _ = self.latent_step(z)  # dynamics head
        return 0

    # setup the layers
    def setup(self):

        # Autoencoder activation and kernel init
        self.ae_activation_fn = load_activation(self.config.ae_activation_fn_name)
        ae_kernel_init = load_kernel(self.config.ae_activation_fn_name)

        # Dynamics activation and kernel init
        self.dyn_activation_fn = load_activation(self.config.dyn_activation_fn_name)
        dyn_kernel_init = load_kernel(self.config.dyn_activation_fn_name)

        # common bias initializer
        bias_init = nn.initializers.zeros

        # encoder
        self.enc_hidden = [
            nn.Dense(size, kernel_init=ae_kernel_init, bias_init=bias_init, name=f"enc_dense_{i}")
            for i, size in enumerate(self.config.enc_layer_sizes)
        ]
        self.enc_out = nn.Dense(self.config.nz, kernel_init=ae_kernel_init, bias_init=bias_init, name="enc_out")

        # decoder
        self.dec_hidden = [
            nn.Dense(size, kernel_init=ae_kernel_init, bias_init=bias_init, name=f"dec_dense_{i}")
            for i, size in enumerate(self.config.dec_layer_sizes)
        ]
        self.dec_out = nn.Dense(self.config.nx, kernel_init=ae_kernel_init, bias_init=bias_init, name="dec_out")

        # latent dynamics model
        self.dyn_hidden = [nn.Dense(s, kernel_init=dyn_kernel_init, bias_init=bias_init, name=f"dyn_dense_{i}")
                           for i, s in enumerate(self.config.dyn_layer_sizes)]
        self.dyn_out = nn.Dense(self.config.nz, kernel_init=dyn_kernel_init, bias_init=bias_init, name="dyn_out")

    # Encoder
    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Encoder network to map FOM state to latent space:
            zₜ = E(xₜ)
        """
        val = x
        for layer in self.enc_hidden:
            val = layer(val)
            val = self.ae_activation_fn(val)
        z = self.enc_out(val)
        return z  

    # Decoder
    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Decoder network to map latent space to reconstructed FOM state:
            x̂ₜ = D(zₜ)
        """
        val = z
        for layer in self.dec_hidden:
            val = layer(val)
            val = self.ae_activation_fn(val)
        x = self.dec_out(val)
        return x
    
    # Latent Dynamics Model (residual form)
    def latent_step(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Simple latent dynamics model as a feedforward NN:
            zₜ₊₁ = zₜ + f(zₜ) [residual]   OR   zₜ₊₁ = f(zₜ) [absolute]
        """
        val = z
        for layer in self.dyn_hidden:
            val = layer(val)
            val = self.dyn_activation_fn(val)
        # z_t1 = self.dyn_out(val)  # shape (..., nz) # NOTE: this would be the alternative
        fz = self.dyn_out(val)    # shape (..., nz)
        z_t1 = z + fz
        return z_t1

    # Forward pass
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return reconstructed x only (for now)."""
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec

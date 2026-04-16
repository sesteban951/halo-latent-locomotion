##
#
# Custom assortment of nerual networks and wrappers to be used with Brax PPO training.
#
##

# python imports
import numpy as np
from typing import Sequence, Callable, Union

# flax impports
import flax.linen as nn
from flax import struct

# jax imports
import jax
import jax.numpy as jnp

# brax imports
from brax.training import distribution, networks, types
from brax.training.acme import running_statistics
from brax.training.agents.ppo.networks import PPONetworks, make_inference_fn
from brax.training.types import Params


"""
===============================================================
 Activation ↔ Recommended Kernel Initializer Cheat Sheet
===============================================================

For stable training in PPO / RL:

| Activation   | Recommended Initializer(s) | Notes                                |
|--------------|-----------------------------|-------------------------------------|
| tanh         | lecun_uniform / lecun_normal | Best for bounded activations, avoids saturation |
| sigmoid      | lecun_uniform / lecun_normal | Same as tanh                       |
| relu         | he_uniform / he_normal       | Standard for ReLU family           |
| leaky_relu   | he_uniform / he_normal       | Same as ReLU, avoids dead neurons  |
| elu          | he_uniform / he_normal       | Smooth ReLU variant                |
| gelu         | he_uniform / he_normal       | Modern smooth activation           |
| swish        | he_uniform / he_normal       | Empirically stable for continuous control |
| softmax      | xavier_uniform / xavier_normal | Good for classification logits    |
| linear (id)  | xavier_uniform / xavier_normal | Safe default for value head       |
| orthogonal   | optional for RNNs / stability | Sometimes used in RL exploration  |

Rule of thumb:
- **tanh/sigmoid** → LeCun
- **ReLU-family (relu, leaky_relu, elu, gelu, swish)** → He/Kaiming
- **softmax/linear outputs** → Xavier
- **orthogonal** → special cases (e.g. RNNs)
===============================================================
"""


##################################### NETWORKS #########################################

# MLP config
@struct.dataclass
class MLPConfig:
    """
    # Example: (if activation_fn=nn.tanh, layer_sizes=[32, 16, 4], activate_final=True)
        - Input → Dense(32) → tanh → Dense(16) → tanh → Dense(4) → tanh → Output

    # Example initialization
        - Activation: tanh, sigmoid -> LeCun uniform/normal
        - Activation: relu, leaky_relu, elu, gelu -> Kaiming/He uniform/normal
        - softmax (for final layer) -> Xavier/Glorot uniform/normal
    """

    layer_sizes: Sequence[int]               # sizes of each hidden layer
    bias: bool = True                        # whether to use bias vector in dense layers
    kernel_init_name: str = "lecun_uniform"  # kernel initializer
    activate_final: bool = False             # whether to activate the final layer
    activation_fn_name: str = "tanh"         # activation function to use

    # get the kernel initializer function
    def kernel_init(self):
        if self.kernel_init_name == "lecun_uniform":
            return nn.initializers.lecun_uniform()
        elif self.kernel_init_name == "lecun_normal":
            return nn.initializers.lecun_normal()
        elif self.kernel_init_name == "he_uniform":
            return nn.initializers.variance_scaling(2.0, "fan_in", "uniform")
        elif self.kernel_init_name == "he_normal":
            return nn.initializers.variance_scaling(2.0, "fan_in", "truncated_normal")
        elif self.kernel_init_name == "xavier_uniform":
            return nn.initializers.xavier_uniform()
        elif self.kernel_init_name == "xavier_normal":
            return nn.initializers.xavier_normal()
        elif self.kernel_init_name == "orthogonal":
            return nn.initializers.orthogonal()
        else:
            raise ValueError(self.kernel_init_name)

    # get the activation function
    def activation_fn(self):
        if self.activation_fn_name == "tanh":
            return nn.tanh
        elif self.activation_fn_name == "sigmoid":
            return nn.sigmoid
        elif self.activation_fn_name == "relu":
            return nn.relu
        elif self.activation_fn_name == "leaky_relu":
            return nn.leaky_relu
        elif self.activation_fn_name == "elu":
            return nn.elu
        elif self.activation_fn_name == "gelu":
            return nn.gelu
        elif self.activation_fn_name == "swish":
            return nn.swish
        else:
            raise ValueError(self.activation_fn_name)


# basic MLP
class MLP(nn.Module):
    """
    Simple multi-layer perceptron (MLP) network.
    """

    # MLP configuration
    config: MLPConfig

    # main forward pass
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through this custom MLP network.
        """
        for i, layer_size in enumerate(self.config.layer_sizes):            
            
            # apply dense layer
            x = nn.Dense(
                features=layer_size,
                use_bias=self.config.bias,
                kernel_init=self.config.kernel_init(),
                name=f"dense_{i}",
            )(x)

            # apply the activation function at every layer and optionally at the final layer
            if i != len(self.config.layer_sizes) - 1 or self.config.activate_final:
                x = self.config.activation_fn()(x)

        return x


# RNN config
@struct.dataclass
class RNNConfig:
    """
    Configuration for recurrent neural networks (LSTM/GRU).
    
    Example: RNNConfig(
        hidden_size=128,
        num_layers=2,
        cell_type="lstm",
        activation_fn=nn.tanh
    )
    """

    hidden_size: int                                # number of hidden units per recurrent layer
    num_layers: int = 1                             # number of stacked recurrent layers
    cell_type: str = "lstm"                         # "lstm" or "gru"
    activation_fn: Callable = nn.tanh               # activation inside the cell
    use_bias: bool = True                           # whether to use bias in recurrent cells


# basic RNN
# TODO: test if this actually works, not sure if brax supports this
class RNN(nn.Module):
    """
    Simple recurrent neural network wrapper (supports LSTM/GRU).
    """
    config: RNNConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, carry=None):
        """
        Args:
            x: [batch, input_dim] or [batch, time, input_dim]
            carry: optional tuple of hidden/cell states for LSTM/GRU
        Returns:
            outputs, carry
        """
        # If no carry is provided, initialize
        if carry is None:
            if self.config.cell_type.lower() == "lstm":
                carry = nn.OptimizedLSTMCell.initialize_carry(
                    self.make_rng("params"), 
                    (x.shape[0],), self.config.hidden_size
                )
            elif self.config.cell_type.lower() == "gru":
                carry = nn.GRUCell.initialize_carry(
                    self.make_rng("params"), 
                    (x.shape[0],), self.config.hidden_size
                )
            else:
                raise ValueError(f"Unsupported cell type {self.config.cell_type}")

        # Select cell type
        if self.config.cell_type.lower() == "lstm":
            Cell = nn.OptimizedLSTMCell
        elif self.config.cell_type.lower() == "gru":
            Cell = nn.GRUCell
        else:
            raise ValueError(f"Unsupported cell type {self.config.cell_type}")

        # Stacked RNN layers
        for i in range(self.config.num_layers):
            cell = Cell(name=f"{self.config.cell_type}_layer_{i}")
            carry, x = cell(carry, x)

        return x, carry


# util to print some details about a flax model
def print_model_summary(module: nn.Module, input_shape: Sequence[int]):
    """
    Print a readable summary of a flax neural network module.

    Args:
        module: The flax module to summarize.
        input_shape: The shape of the input to the module.
    """

    # Create a dummy input
    dummy_rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones(input_shape)
    print(module.tabulate(dummy_rng, dummy_input, depth=1))


##################################### WRAPPER #########################################

@struct.dataclass
class BraxPPONetworksWrapper:
    """
    Thin wrapper to hold custom networks for PPO training in Brax.
    """

    policy_network: nn.Module # the policy network (default is Sequence[int] = (32,) * 4, swish activation)
    value_network: nn.Module  # the value network (default is Sequence[int] = (256,) * 5, swish activation)
    action_distribution: distribution.ParametricDistribution # distribution for actions


    def make_ppo_networks(
        self,
        obs_size: int,   # observations size
        act_size: int,   # actions size
        preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor # function to preprocess observations
                                                                                                            # default is identity function
    ) -> PPONetworks:
        """
        Create the PPO networks using the custom policy and value networks.

        Args:
            obs_size: Size of the observations.
            act_size: Size of the actions.
            preprocess_observations_fn: Function to preprocess observations.
        Returns:
            An instance of PPONetworks containing the policy and value networks.
        """

        # create action distribution. The policy network should output the parameters of this distribution
        action_dist = self.action_distribution(event_size=act_size)

        # create dummy observation for initialization
        dummy_obs = jnp.zeros(obs_size)

        # create a random key for initialization
        dummy_rng = jax.random.PRNGKey(0)
        policy_rng, value_rng = jax.random.split(dummy_rng)

        # check that the size of the policy network matches the size of the action distribution
        dummy_policy_params = self.policy_network.init(policy_rng, dummy_obs)
        dummy_policy_output = self.policy_network.apply(dummy_policy_params, dummy_obs)
        dummy_value_params = self.value_network.init(value_rng, dummy_obs)
        dummy_value_output = self.value_network.apply(dummy_value_params, dummy_obs)

        # shapes to make sure should match
        action_dist_params_shape = action_dist.param_size
        policy_output_shape = dummy_policy_output.shape[-1]
        value_output_shape = dummy_value_output.shape[-1]

        # assert the networks output shape matches the action distribution parameter size
        assert policy_output_shape == action_dist_params_shape, (
            f"Policy network output shape ({policy_output_shape}) does not match "
            f"action distribution parameter size ({action_dist_params_shape})."
        )

        # assert the value network output shape is correct
        assert value_output_shape == 1, (
            f"Value network output shape ({value_output_shape}) is not 1."
        )


        # create the Policy Network functions
        # init funcition
        def policy_init(key):
            return self.policy_network.init(key, dummy_obs)
        
        # apply function
        def policy_apply(processor_params, policy_params, obs):
            processed_obs = preprocess_observations_fn(obs, processor_params)
            return self.policy_network.apply(policy_params, processed_obs)
        
        # feedforward policy network
        policy_network = networks.FeedForwardNetwork(
            init=policy_init,
            apply=policy_apply
        )


        # create the Value Network functions
        # init function
        def value_init(key):
            return self.value_network.init(key, dummy_obs)
        
        # apply function
        def value_apply(processor_params, value_params, obs):
            processed_obs = preprocess_observations_fn(obs, processor_params)
            return jnp.squeeze(self.value_network.apply(value_params, processed_obs), axis=-1)

        # feedforward value network
        value_network = networks.FeedForwardNetwork(
            init=value_init,
            apply=value_apply
        )


        # bulid the PPONetworks dataclass
        ppo_networks = PPONetworks(
            policy_network=policy_network,
            value_network=value_network,
            parametric_action_distribution=action_dist,
        )

        return ppo_networks


# create the policy function
def make_policy_function(
        network_wrapper: BraxPPONetworksWrapper, # the networks wrapper
        params: Params,                          # the model parameters
        obs_size: int,                           # the observation size
        act_size: int,                           # the action size
        normalize_observations: bool = True,     # whether to normalize observations
        deterministic: bool = True               # whether to use deterministic actions
    ):
    """
    Create a policy from a trained model as a function that takes observations and returns actions.

    Args:
        network_wrapper: The BraxPPONetworksWrapper containing the policy and value networks.
        params: The trained model parameters.
        obs_size: The size of the observations.
        act_size: The size of the actions.
        normalize_observations: Whether to normalize observations using running statistics.
        deterministic: Whether to use deterministic actions (e.g., mean of the distribution).
    Returns:
        A function that takes observations and returns actions.
    """

    # preprocessing of observatiosn functions
    if normalize_observations:
        preprocess_observations_fn = running_statistics.normalize
    else:
        preprocess_observations_fn = types.identity_observation_preprocessor
    
    # create the PPO networks
    ppo_networks = network_wrapper.make_ppo_networks(
        obs_size=obs_size,
        act_size=act_size,
        preprocess_observations_fn=preprocess_observations_fn
    )

    # make the inference function
    inference_fn = make_inference_fn(ppo_networks=ppo_networks)
    policy = inference_fn(params=params, deterministic=deterministic)

    return policy

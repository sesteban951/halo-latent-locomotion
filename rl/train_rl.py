##
#
#  Simple Training Script for BRAX environments
#  NOTE: For some reason, my desktop (4090 GPU) is better at training.  
#        and produces better policies for the same hyperparameters.
#
##

# jax imports
import jax

# brax imports
from brax import envs
import brax.training.distribution as distribution

# custom imports
from envs.hopper_env import HopperEnv, HopperConfig
from envs.hopper_tracking_env import HopperTrackingEnv, HopperTrackingConfig
from envs.paddle_ball_env import PaddleBallEnv, PaddleBallConfig
from envs.paddle_ball_tracking_env import PaddleBallTrackingEnv, PaddleBallTrackingConfig
from algorithms.ppo_train import PPO_Train
from algorithms.custom_networks import BraxPPONetworksWrapper, MLPConfig, MLP


if __name__ == "__main__":

    # print the device being used (gpu or cpu)
    device = jax.devices()[0]
    print("Device type:", device.platform)      # e.g. 'gpu' or 'cpu'
    print("Device name:", device.device_kind)   # e.g. 'NVIDIA GeForce RTX 4090'

    #--------------------------------- SETUP ---------------------------------#

    # Initialize the environment and PPO hyperparameters
    # env = envs.get_environment("paddle_ball")
    # env = envs.get_environment("paddle_ball_tracking")
    # policy_network_config = MLPConfig(
    #     layer_sizes=(32, 32, 32, 32, 2*env.action_size),   # policy hidden layer sizes
    #     activation_fn_name="swish",                        # activation function
    # )
    # value_network_config = MLPConfig(
    #     layer_sizes=(256, 256, 256, 256, 256, 1),  # value hidden layer sizes
    #     activation_fn_name="swish",                # activation function
    # )
    # network_wrapper = BraxPPONetworksWrapper(
    #     policy_network=MLP(policy_network_config),
    #     value_network=MLP(value_network_config),
    #     action_distribution=distribution.NormalTanhDistribution
    # )
    # ppo_config = dict(
    #     num_timesteps=100_000_000,      
    #     num_evals=10,                  
    #     reward_scaling=0.1,            
    #     episode_length=500,            
    #     normalize_observations=True,   
    #     unroll_length=10,              
    #     num_minibatches=32,            
    #     num_updates_per_batch=8,       
    #     discounting=0.97,              
    #     learning_rate=5e-4,            
    #     clipping_epsilon=0.2,          
    #     entropy_cost=1e-3,             
    #     num_envs=2048,                 
    #     batch_size=2048,               
    #     seed=0,                        
    # )

    # Initialize the environment and PPO hyperparameters
    # env = envs.get_environment("hopper")
    env = envs.get_environment("hopper_tracking")
    policy_network_config = MLPConfig(
        layer_sizes=(32, 32, 32, 32, 2*env.action_size),   # policy hidden layer sizes
        activation_fn_name="swish",                        # activation function
    )
    value_network_config = MLPConfig(
        layer_sizes=(256, 256, 256, 256, 256, 1),  # value hidden layer sizes
        activation_fn_name="swish",                # activation function
    )
    network_wrapper = BraxPPONetworksWrapper(
        policy_network=MLP(policy_network_config),
        value_network=MLP(value_network_config),
        action_distribution=distribution.NormalTanhDistribution
    )
    ppo_config = dict(
        num_timesteps=60_000_000,      
        num_evals=10,                  
        reward_scaling=0.1,            
        episode_length=600,            
        normalize_observations=True,   
        unroll_length=5,              
        num_minibatches=64,            
        num_updates_per_batch=8,       
        discounting=0.97,              
        learning_rate=5e-4,            
        clipping_epsilon=0.2,          
        entropy_cost=3e-4,             
        num_envs=4096,                 
        batch_size=4096,               
        seed=0,                        
    )

    #--------------------------------- TRAIN ---------------------------------#

    # Create PPO training instance
    ppo_trainer = PPO_Train(env, network_wrapper, ppo_config)

    # start training
    ppo_trainer.train()
# standard imports
from datetime import datetime
import os

# jax imports
import jax

# brax imports
from brax.training.agents.ppo import train as ppo

# for saving results
import pickle

# for logging
from tensorboardX import SummaryWriter


# PPO Training class
class PPO_Train:
    """
    PPO Training class
    
    Args:
        env: Brax environment instance
        network_wrapper: BraxPPONetworksWrapper instance containing the policy and value networks
        ppo_config: dictionary containing PPO hyperparameters
    """
    def __init__(self, env, network_wrapper, ppo_config):

        # set the environment and PPO hyperparameters
        self.env = env
        self.network_wrapper = network_wrapper
        self.ppo_config = ppo_config

        # check if the save path exists, if not create it
        self.save_path = "./rl/policy"
        self.check_save_path()

        # create SummaryWriter object for logging RL progress
        self.create_summary_writer()

        # print info
        print(f"Created PPO training instance for: [{self.env.__class__.__name__}]")

    # check if the save file path exists
    def check_save_path(self):

        # check if the directory exists
        if not os.path.exists(self.save_path):

            # notify that the directory does not exist
            print(f"Directory [{self.save_path}] does not exist, creating it now...")

            # create the directory
            os.makedirs(self.save_path)
            print(f"Created directory: [{self.save_path}]")

    # create the summary writer for TensorBoard logging
    def create_summary_writer(self):

        # get current datetime for logging
        self.robot_name = self.env.env_name
        self.current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        log_file = f"./rl/log/{self.robot_name}_log_{self.current_datetime}"

        # create a SummaryWriter for logging
        self.writer = SummaryWriter(log_file)
        self.times = [datetime.now()]

        # print logging path info
        print(f"Logging to: [{log_file}].")

    # progress function for logging
    def progress(self, num_steps, metrics):

        # try to get eval reward if it exists
        reward = metrics.get("eval/episode_reward", None)
        if reward is not None:
            print(f"  Step: {num_steps:,}, Reward: {reward:.1f}")
        else:
            print(f"  Step: {num_steps}, Reward: N/A")

        # append the current time
        self.times.append(datetime.now())

        # Write all metrics to tensorboard
        for key, val in metrics.items():

            # convert jax arrays to floats
            if isinstance(val, jax.Array):
                val = float(val)

            # log to tensorboard
            self.writer.add_scalar(key, val, num_steps)

    # main training function
    def train(self):

        # train the PPO agent
        #   - (make_policy) makes the policy function
        #   - (params) are the trained parameters
        #   - (metrics) final training metrics
        print("Beginning Training...")
        make_policy, params, metrics = ppo.train(
            environment=self.env,
            network_factory=self.network_wrapper.make_ppo_networks,
            progress_fn=self.progress,
            **self.ppo_config
        )
        print("Training complete.")
        print(f"time to jit: {self.times[1] - self.times[0]}")
        print(f"time to train: {self.times[-1] - self.times[1]}")

        # save the trained policy
        save_file = f"{self.save_path}/{self.robot_name}_policy_{self.current_datetime}.pkl"
        policy_data = {
            "policy_config": self.network_wrapper.policy_network.config,
            "value_config": self.network_wrapper.value_network.config,
            "action_dist_class": self.network_wrapper.action_distribution.__name__,
            "params": params
        }
        with open(save_file, "wb") as f:
            pickle.dump(policy_data, f)
        print(f"Saved trained policy to: [{save_file}]")

        return make_policy, params, metrics

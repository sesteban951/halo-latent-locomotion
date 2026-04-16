# jax imports
import jax
import jax.numpy as jnp
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from brax import envs
from flax import struct

# mujoco imports
import mujoco


# struct to hold the configuration parameters
@struct.dataclass
class HopperConfig:
    """Config dataclass for hopper."""

    # model path (NOTE: relative the script that calls this class)
    model_path: str = "./models/hopper.xml"

    # number of "simulation steps" for every control input
    physics_steps_per_control_step: int = 4

    # Reward function coefficients
    reward_torso_height: float = 4.0   # reward for torso height
    reward_torso_angle: float = 0.5    # reward for torso angle
    reward_leg_pos: float = 0.01        # reward for leg position
    reward_torso_vel_x: float = 1.0      # reward for zero velocity
    reward_torso_vel_z: float = 0.01      # reward for zero velocity
    reward_torso_vel_angle: float = 0.01      # reward for zero velocity
    reward_leg_vel: float = 0.001        # reward for zero leg velocity
    reward_control: float = 1e-4       # cost for control effort

    # desired values
    desired_pos_z: float = 2.0  # desired torso height, achieved by hopping
    desired_vel_x: float = 1.0  # desired forward velocity

    # Ranges for sampling initial conditions
    lb_torso_height: float = 1.0
    ub_torso_height: float = 2.0
    lb_torso_vel: float = -5.0
    ub_torso_vel: float =  5.0
    
    lb_angle_pos: float = -jnp.pi
    ub_angle_pos: float =  jnp.pi
    lb_angle_vel: float = -5.0
    ub_angle_vel: float =  5.0

    lb_leg_pos: float = -0.25
    ub_leg_pos: float =  0.25
    lb_leg_vel: float = -5.0
    ub_leg_vel: float =  5.0

# environment class
class HopperEnv(PipelineEnv):
    """
    Environment for training a hopper hopping task.
    
    States: x = (pos_x, pos_z, theta, leg_pos, vel_x, vel_z, theta_dot, leg_vel), shape=(8,)
    Observations: o = (pos_z, cos(theta), sin(theta), leg_pos, vel_x, vel_z, theta_dot, leg_vel), shape=(8,)
    Actions: a = (tau_theta, force_leg) torque applied to the body and force applied to the leg, shape=(2,)
    """

    # initialize the environment
    def __init__(self, config: HopperConfig = HopperConfig()):

        # robot name
        self.robot_name = "hopper"

        # environment name
        self.env_name = "hopper"

        # load the config
        self.config = config

        # create the brax system
        mj_model = mujoco.MjModel.from_xml_path(self.config.model_path)
        sys = mjcf.load_model(mj_model)

        # insantiate the parent class
        super().__init__(
            sys=sys,                                             # brax system defining the kinematic tree and other properties
            backend="mjx",                                       # defining the physics pipeline
            n_frames=self.config.physics_steps_per_control_step  # number of times to step the physics pipeline per control step
                                                                 # for each environment step
        )
        # n_frames: number of sim steps per control step, dt = n_frames * xml_dt

        # print message
        print(f"Initialized HopperEnv with model [{self.config.model_path}].")

    # reset function
    def reset(self, rng):
        """
        Resets the environment to an initial state.

        Args:
            rng: jax random number generator (jax.Array)

        Returns:
            State: brax.envs.base.State object
                   Environment state for training and inference.
        """
        
        # split the rng to sample unique initial conditions
        rng, rng1, rng2 = jax.random.split(rng, 3)

        # set the state bounds for sampling initial conditions
        qpos_lb = jnp.array([-0.001, self.config.lb_torso_height, self.config.lb_angle_pos, self.config.lb_leg_pos])
        qpos_ub = jnp.array([ 0.001, self.config.ub_torso_height, self.config.ub_angle_pos, self.config.ub_leg_pos])
        qvel_lb = jnp.array([self.config.lb_torso_vel, self.config.lb_torso_vel, self.config.lb_angle_vel, self.config.lb_leg_vel])
        qvel_ub = jnp.array([self.config.ub_torso_vel, self.config.ub_torso_vel, self.config.ub_angle_vel, self.config.ub_leg_vel])

        # sample the initial state
        qpos = jax.random.uniform(rng1, (4,), minval=qpos_lb, maxval=qpos_ub)
        qvel = jax.random.uniform(rng2, (4,), minval=qvel_lb, maxval=qvel_ub)

        # reset the physics state
        data = self.pipeline_init(qpos, qvel)

        # reset the observation
        obs = self._compute_obs(data)

        # reset reward
        reward, done = jnp.zeros(2)

        # reset the metrics
        metrics = {"reward_torso_height": jnp.array(0.0),
                   "reward_torso_angle": jnp.array(0.0),
                   "reward_leg_pos": jnp.array(0.0),
                   "reward_torso_vel_x": jnp.array(0.0),
                   "reward_torso_vel_z": jnp.array(0.0),
                   "reward_torso_vel_angle": jnp.array(0.0),
                   "reward_leg_vel": jnp.array(0.0),
                   "reward_control": jnp.array(0.0)}

        # state info
        info = {"rng": rng,
                "step": 0}
        
        return State(pipeline_state=data,
                     obs=obs,
                     reward=reward,
                     done=done,
                     metrics=metrics,
                     info=info)

    # physics step function
    def step(self, state, action):
        """
        Step the environment by one timestep.

        Args:
            state: brax.envs.base.State object
                   The current state of the environment.
            action: jax.Array
                    The action to be applied to the environment.
        """

        # step the physics
        data = self.pipeline_step(state.pipeline_state, action)

        # update the observations
        obs = self._compute_obs(data)

        # data
        pos_z = data.qpos[1]
        theta = data.qpos[2]
        leg_pos = data.qpos[3]
        vel_x = data.qvel[0]
        vel_z = data.qvel[1]
        theta_dot = data.qvel[2]
        leg_vel = data.qvel[3]
        tau = data.ctrl

        # special angle error
        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)
        theta_angle_vec = jnp.array([cos_theta - 1.0, sin_theta]) # want (0, 0)

        # compute error terms
        torso_height_err = jnp.square(pos_z - self.config.desired_pos_z).sum()
        torso_angle_err = jnp.square(theta_angle_vec).sum()
        leg_pos_err = jnp.square(leg_pos).sum()
        torso_vel_x_err = jnp.square(vel_x - self.config.desired_vel_x).sum()
        torso_vel_z_err = jnp.square(vel_z).sum()
        torso_vel_angle_err = jnp.square(theta_dot).sum()
        leg_vel_err = jnp.square(leg_vel).sum()
        control_err = jnp.square(tau).sum()

        # compute the rewards
        reward_torso_height = -self.config.reward_torso_height * torso_height_err
        reward_torso_angle = -self.config.reward_torso_angle * torso_angle_err
        reward_leg_pos = -self.config.reward_leg_pos * leg_pos_err
        reward_torso_vel_x = -self.config.reward_torso_vel_x * torso_vel_x_err
        reward_torso_vel_z = -self.config.reward_torso_vel_z * torso_vel_z_err
        reward_torso_vel_angle = -self.config.reward_torso_vel_angle * torso_vel_angle_err
        reward_leg_vel = -self.config.reward_leg_vel * leg_vel_err
        reward_control  = -self.config.reward_control * control_err

        # compute the total reward
        reward = (reward_torso_height + reward_torso_angle + reward_leg_pos +
                  reward_torso_vel_x + reward_torso_vel_z + reward_torso_vel_angle + 
                  reward_leg_vel + reward_control)

        # update the metrics and info dictionaries
        state.metrics["reward_torso_height"] = reward_torso_height
        state.metrics["reward_torso_angle"] = reward_torso_angle
        state.metrics["reward_leg_pos"] = reward_leg_pos
        state.metrics["reward_torso_vel_x"] = reward_torso_vel_x
        state.metrics["reward_torso_vel_z"] = reward_torso_vel_z
        state.metrics["reward_torso_vel_angle"] = reward_torso_vel_angle
        state.metrics["reward_leg_vel"] = reward_leg_vel
        state.metrics["reward_control"] = reward_control
        state.info["step"] += 1

        return state.replace(pipeline_state=data,
                             obs=obs,
                             reward=reward)

    # internal function to compute the observation
    def _compute_obs(self, data):
        """
        Compute the observation from the physics state.

        Args:
            data: brax.physics.base.State object
                  The physics state of the environment.
        """

        # extract the relevant information from the data
        pos_z = data.qpos[1]
        theta = data.qpos[2]
        leg_pos = data.qpos[3]
        vel_x = data.qvel[0]
        vel_z = data.qvel[1]
        theta_dot = data.qvel[2]
        leg_vel = data.qvel[3]

        # compute the observation
        obs = jnp.array([pos_z,
                         jnp.cos(theta),
                         jnp.sin(theta),
                         leg_pos,
                         vel_x, 
                         vel_z,
                         theta_dot,
                         leg_vel])

        return obs
    
    @property
    def observation_size(self):
        """Returns the size of the observation space."""
        return 8

    @property
    def action_size(self):
        """Returns the size of the action space."""
        return 2


# register the environment
envs.register_environment("hopper", HopperEnv)
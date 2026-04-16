# jax imports
import jax
import jax.numpy as jnp
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from flax import struct
from brax import envs

# mujoco imports
import mujoco


# struct to hold the configuration parameters
@struct.dataclass
class PaddleBallTrackingConfig:
    """Config dataclass for paddle ball."""

    # model path (NOTE: relative the script that calls this class)
    model_path: str = "./models/paddle_ball.xml"

    # number of "simulation steps" for every control input
    physics_steps_per_control_step: int = 4

    # Reward function coefficients
    reward_ball_pos: float = 2.0
    reward_ball_vel: float = 0.01
    reward_paddle_pos: float = 1.0
    reward_paddle_vel: float = 0.015
    reward_control: float = 1e-4

    # Ranges for sampling initial conditions
    lb_ball_pos: float = 1.0
    ub_ball_pos: float = 2.0
    lb_ball_vel: float = -5.0
    ub_ball_vel: float =  0.5

    lb_paddle_pos: float = 0.01
    ub_paddle_pos: float = 1.0
    lb_paddle_vel: float = -5.0
    ub_paddle_vel: float =  5.0

    # sample position commands
    cmd_lb: float = 1.0        # lower bound of command sampling range
    cmd_ub: float = 2.0        # upper bound of command sampling range
    cmd_nom: float = 1.5       # nominal command (for bernoulli sampling)
    bernoulli_p: float = 0.1   # probability of sampling the nominal command, p ∈ [0, 1]


# environment class
class PaddleBallTrackingEnv(PipelineEnv):
    """
    Environment for training a paddle ball bouncing task (with tracking a height).

    States: x = (pos_ball, pos_paddle, vel_ball, vel_paddle), shape=(4,)
    Observations: o = (pos_ball, pos_paddle, vel_ball, vel_paddle), shape=(4,)
    Actions: a = tau, the force on the paddle, shape=(1,)
    """

    # initialize the environment
    def __init__(self, config: PaddleBallTrackingConfig = PaddleBallTrackingConfig()):

        # robot name
        self.robot_name = "paddle_ball"

        # environment name
        self.env_name = "paddle_ball_tracking"

        # load the config
        self.config = config

        # create the brax system
        # TODO: eventually refactor to use MJX fully instead since BRAX is now moving away from MJCF
        # see https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb
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
        print(f"Initialized PaddleBallTrackingEnv with model [{self.config.model_path}].")

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
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        # set the state bounds for sampling initial conditions
        qpos_lb = jnp.array([self.config.lb_ball_pos, self.config.lb_paddle_pos])
        qpos_ub = jnp.array([self.config.ub_ball_pos, self.config.ub_paddle_pos])
        qvel_lb = jnp.array([self.config.lb_ball_vel, self.config.lb_paddle_vel])
        qvel_ub = jnp.array([self.config.ub_ball_vel, self.config.ub_paddle_vel])

        # sample the initial state
        qpos = jax.random.uniform(rng1, (2,), minval=qpos_lb, maxval=qpos_ub)
        qvel = jax.random.uniform(rng2, (2,), minval=qvel_lb, maxval=qvel_ub)

        # sample new position command
        pos_cmd = jax.random.uniform(rng3, shape=(), 
                                     minval=self.config.cmd_lb, 
                                     maxval=self.config.cmd_ub)
        # with some probability, set the command to the nominal value
        bernoulli_sample = jax.random.bernoulli(rng, p=self.config.bernoulli_p, shape=())
        pos_cmd = jnp.where(bernoulli_sample, self.config.cmd_nom, pos_cmd)
        
        # reset the physics state
        data = self.pipeline_init(qpos, qvel)

        # reset the observation
        obs = self._compute_obs(data, pos_cmd)

        # reset reward
        reward, done = jnp.zeros(2)

        # reset the metrics
        metrics = {"reward_ball_pos": jnp.array(0.0),
                   "reward_paddle_pos": jnp.array(0.0),
                   "reward_ball_vel": jnp.array(0.0),
                   "reward_paddle_vel": jnp.array(0.0),
                   "reward_control": jnp.array(0.0)}

        # state info
        info = {"rng": rng,
                "step": 0,
                "pos_cmd": pos_cmd}
        
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

        # pull the command from the state info
        pos_cmd = state.info["pos_cmd"]

        # update the observations
        obs = self._compute_obs(data, pos_cmd)

        # data
        ball_pos = data.qpos[0]
        paddle_pos = data.qpos[1]
        ball_vel = data.qvel[0]
        paddle_vel = data.qvel[1]
        tau = data.ctrl

        # desired position of the ball and paddle
        des_ball_pos = pos_cmd
        des_paddle_pos = 0.5

        # compute error terms
        ball_pos_err = jnp.square(ball_pos - des_ball_pos).sum()
        paddle_pos_err = jnp.square(paddle_pos - des_paddle_pos).sum()
        ball_vel_err = jnp.square(ball_vel).sum()
        paddle_vel_err = jnp.square(paddle_vel).sum()
        control_err = jnp.square(tau).sum()

        # compute the rewards
        # reward_ball_pos = -self.config.reward_ball_pos * ball_pos_err
        reward_ball_pos = jnp.exp(-ball_pos_err) * self.config.reward_ball_pos
        reward_paddle_pos = -self.config.reward_paddle_pos * paddle_pos_err
        reward_ball_vel = -self.config.reward_ball_vel * ball_vel_err
        reward_paddle_vel = -self.config.reward_paddle_vel * paddle_vel_err
        reward_control  = -self.config.reward_control * control_err

        # compute the total reward
        reward = (reward_ball_pos + reward_paddle_pos + 
                  reward_ball_vel + reward_paddle_vel + 
                  reward_control)
        
        # update the metrics and info dictionaries
        state.metrics["reward_ball_pos"] = reward_ball_pos
        state.metrics["reward_paddle_pos"] = reward_paddle_pos
        state.metrics["reward_ball_vel"] = reward_ball_vel
        state.metrics["reward_paddle_vel"] = reward_paddle_vel
        state.metrics["reward_control"] = reward_control
        state.info["step"] += 1

        return state.replace(pipeline_state=data,
                             obs=obs,
                             reward=reward)

    # internal function to compute the observation
    def _compute_obs(self, data, pos_cmd):
        """
        Compute the observation from the physics state.

        Args:
            data: brax.physics.base.State object
                  The physics state of the environment.
            pos_cmd: float
                     The desired ball position command.
        """

        # extract the relevant information from the data
        ball_pos = data.qpos[0]
        paddle_pos = data.qpos[1]
        ball_vel = data.qvel[0]
        paddle_vel = data.qvel[1]

        # compute the observation
        obs = jnp.array([ball_pos,            # ball position
                         paddle_pos,          # paddle position
                         ball_vel,            # ball velocity
                         paddle_vel,          # paddle velocity
                         pos_cmd])            # ball position command

        return obs
    
    @property
    def observation_size(self):
        """Returns the size of the observation space."""
        return 5

    @property
    def action_size(self):
        """Returns the size of the action space."""
        return 1


# register the environment
envs.register_environment("paddle_ball_tracking", PaddleBallTrackingEnv)
##
#
#  Perform Parallel MJX Rollouts
#  
##

# standard imports
import os, sys
import numpy as np
import time
from dataclasses import dataclass

# jax imports
import jax
import jax.numpy as jnp
from jax import lax

# brax imports
from brax import envs

# mujoco imports
import mujoco
import mujoco.mjx as mjx

# suppress jax warnings ("All configs were filtered out because...")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all,1=INFO,2=WARNING,3=ERROR

# change directories to project root (so `from rl...` works even if run from /data)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# custom imports
from rl.envs.hopper_env import HopperEnv
from rl.envs.paddle_ball_env import PaddleBallEnv
from rl.algorithms.custom_networks import BraxPPONetworksWrapper, MLP
from rl.algorithms.ppo_play import PPO_Play


##################################################################################
# PARALLEL DYNAMICS ROLLOUT CLASS
##################################################################################

# struct to hold the parallel sim config
@dataclass
class ParallelSimConfig:

    rng: jax.random.PRNGKey           # random number generator key
    env_name: str                     # RL environment name
    batch_size: int                   # batch size for parallel rollout
    state_bounds: tuple               # tuple of (q_lb, q_ub, v_lb, v_ub) for initial condition sampling
    sim_dt_des: float                 # desired simulation time step (if None, use model XML dt)
    log_dtype: jnp.dtype=jnp.float16  # data type for logging (jnp.float16, jnp.bfloat16, jnp.float32, etc.)
    integrator: str="RK4"             # integrator type: "Euler" or "RK4"
    policy_params_path: str=None      # path to policy parameters file (if None, cannot run closed loop with policy)


# MJX Rollout class
class ParallelSim():
    """
    Class to perform parallel rollouts using mujoco mjx on GPU.

    Args:
        config: ParallelSimConfig, configuration for the parallel sim
    """

    # initialize the class
    def __init__(self, config: ParallelSimConfig):

        # assign the random seed
        self.rng = config.rng

        # assign the batch size
        self.batch_size = config.batch_size

        # load the enviornment
        self.env = envs.get_environment(config.env_name)
        env_config = self.env.config

        # logging data type
        self.log_dtype = config.log_dtype
        
        # load the control decimation (number of sim steps per control step)
        self.control_decimation = env_config.physics_steps_per_control_step

        # load the desired sim dt
        if config.sim_dt_des is not None:
            self.sim_dt = float(config.sim_dt_des)
        else:
            self.sim_dt = None

        # integrator
        self.integrator = config.integrator

        # load the mujoco model for parallel sim
        model_path = env_config.model_path
        mj_model = mujoco.MjModel.from_xml_path(model_path)
        mj_data = mujoco.MjData(mj_model)
        self.initialize_model(mj_model, mj_data, model_path)

        # load in the touch sensors if any
        self.initialize_touch_sensors(mj_model)

        # policy and observation function
        if config.policy_params_path is not None:
            self.initialize_policy_and_obs_fn(config.policy_params_path)
        else:
            print("No policy parameters path provided. Rollouts will use zero input.")
            self.policy_fn_batched = None
            self.obs_fn_batched = None

        # initialize jit functions for speed
        self.initialize_jit_functions()

        # set the initial condition state bounds
        state_bounds = config.state_bounds
        self.q_lb, self.q_ub, self.v_lb, self.v_ub = (
            jnp.asarray(state_bounds[0], dtype=jnp.float32),
            jnp.asarray(state_bounds[1], dtype=jnp.float32),
            jnp.asarray(state_bounds[2], dtype=jnp.float32),
            jnp.asarray(state_bounds[3], dtype=jnp.float32),
        )

        # zeros vector to use with zero input rollouts
        self.u_zero = jnp.zeros((self.batch_size, self.nu), dtype=jnp.float32)  # (batch, nu) zeros

        # sampling bounds. Should be [-1.0, 1.0] for all models beause of how XML is set up
        self.u_lb = -jnp.ones((self.nu,), dtype=jnp.float32)  # lower bound is -1.0, shape (nu,)
        self.u_ub =  jnp.ones((self.nu,), dtype=jnp.float32)  # upper bound is  1.0, shape (nu,)

    ######################################### INITIALIZATION #########################################

    # initialize model
    def initialize_model(self, mj_model, mj_data, model_path):
        """
        Initialize the mujoco model and data for parallel rollout.
        
        Args:
            mj_model: mujoco.MjModel, the mujoco model
            mj_data: mujoco.MjData, the mujoco data
        """
        # mapping from string to mujoco integrator type
        # https://mujoco.readthedocs.io/en/stable/computation/index.html#integrators
        integrator_map = {
            "euler":         mujoco.mjtIntegrator.mjINT_EULER,        
            "implicitfast":  mujoco.mjtIntegrator.mjINT_IMPLICITFAST, 
            "rk4":           mujoco.mjtIntegrator.mjINT_RK4,          
        }

        # set the integrator type
        if self.integrator.lower() in integrator_map:
            mj_model.opt.integrator = integrator_map[self.integrator.lower()]
        else:
            raise ValueError(f"Unknown integrator type: [{self.integrator}]. Supported types are: {list(integrator_map.keys())}")

        # original simulation timestep
        original_sim_dt = float(mj_model.opt.timestep)

        # overwrite the simulation time step of the mujoco XML if one is specified
        if self.sim_dt is not None:
            mj_model.opt.timestep = self.sim_dt

        # check that control dt is integer multiple of sim dt
        sim_dt = float(mj_model.opt.timestep)
        ctrl_dt = float(original_sim_dt * self.control_decimation)
        if (ctrl_dt / sim_dt) % 1.0 != 0.0:
            raise ValueError(f"Sim dt ({sim_dt}) is not integer multiple of Control dt ({ctrl_dt}) .")

        # put the model and data on GPU
        self.mjx_model = mjx.put_model(mj_model)
        self.mjx_data = mjx.put_data(mj_model, mj_data)

        # load sizes
        self.nq = self.mjx_model.nq
        self.nv = self.mjx_model.nv
        self.nu = self.mjx_model.nu

        # create the batched step function
        self.step_fn_batched = jax.jit(jax.vmap(lambda d: mjx.step(self.mjx_model, d), in_axes=0))

        # simulation parameters
        self.sim_dt = float(self.mjx_model.opt.timestep)  # simulation time step

        # print message
        print(f"Initialized batched MJX model from [{model_path}].")
        print(f"   integrator: {self.mjx_model.opt.integrator}")
        print(f"   sim_dt: {self.sim_dt:.4f}")
        print(f"   nq: {self.nq}")
        print(f"   nv: {self.nv}")
        print(f"   nu: {self.nu}")

    # initialize touch sensors
    def initialize_touch_sensors(self, mj_model):
        """
        Initialize touch sensors if any are present in the model.

        Args:
            mj_model: mujoco.MjModel, the mujoco model
        """
        
        # cache touch sensor IDs
        self.touch_sensor_ids = [
            i for i, stype in enumerate(mj_model.sensor_type)
            if stype == mujoco.mjtSensor.mjSENS_TOUCH
        ]

        # number of touch sensors
        self.nc = len(self.touch_sensor_ids)

        print(f"Found {self.nc} touch sensors.")

    # initialize the policy and observation functions
    def initialize_policy_and_obs_fn(self, policy_params_path):
        """
        Initialize the policy and observation functions.

        Args:
            policy_params_path: str, path to the policy parameters file
        Returns:
            policy_fn: function, policy function
            obs_fn: function, observation function
        """

        # create the policy and observation function
        ppo_player = PPO_Play(self.env, policy_params_path)

        # get the policy and observation functions
        policy_fn, obs_fn = ppo_player.policy_and_obs_functions()

        # create the batched policy and observation functions
        self.policy_fn_batched = jax.vmap(policy_fn, in_axes=0)
        self.obs_fn_batched = jax.vmap(obs_fn, in_axes=0)

        # print message
        print(f"Initialized batched policy and observation functions from:")
        print(f"   env:    [{self.env.robot_name}]")
        print(f"   policy: [{policy_params_path}]")


    # initialize jit functions
    def initialize_jit_functions(self):
        """
        Initialize the jit functions for rollout for speed.
        """

        # jit the rollout with zero inputs function
        self.rollout_zero_input_jit = jax.jit(self._rollout_zero_input, 
                                              static_argnames=('T',), 
                                              # donate_argnums=(0, 1)
                                              )
        
        # jit the rollout with policy inputs function
        self.rollout_policy_input_jit = jax.jit(self._rollout_policy_input,
                                                static_argnames=('T',),
                                                # donate_argnums=(0, 1)
                                                )
        
        # jit the rollout with random inputs function
        self.rollout_random_input_jit = jax.jit(self._rollout_random_input,
                                                static_argnames=('T',),
                                                # donate_argnums=(0, 1, 2)
                                                )


    ######################################### SAMPLING #########################################

    # sample initial conditions
    def sample_random_uniform_initial_conditions(self):
        """
        Sample initial conditions for the system.

        Returns:
            q0_batch: jnp.array, sampled initial positions (batch_size, nq)
            v0_batch: jnp.array, sampled initial velocities (batch_size, nv)
        """

        # split the rng
        self.rng, key1, key2 = jax.random.split(self.rng, 3)

        # sample initial conditions
        q0_batch = jax.random.uniform(key1, 
                                      (self.batch_size, self.nq), 
                                      minval=self.q_lb, 
                                      maxval=self.q_ub).astype(jnp.float32)
        v0_batch = jax.random.uniform(key2, 
                                      (self.batch_size, self.nv), 
                                      minval=self.v_lb, 
                                      maxval=self.v_ub).astype(jnp.float32)

        return q0_batch, v0_batch
    

    # sample inputs from a uniform distribution
    def sample_random_uniform_inputs(self, S):
        """
        Sample random sequence of inputs for the system.

        Args:
            S: int, length of input sequence
        Returns:
            u_batch: jnp.array, sampled input sequence (batch_size, S, nu)
        """

        # split the rng
        self.rng, subkey = jax.random.split(self.rng)

        # sample random control inputs
        u_seq_batch = jax.random.uniform(subkey, 
                                         (self.batch_size, S, self.nu), 
                                         minval=self.u_lb, 
                                         maxval=self.u_ub) # shape (batch, S, nu)
        
        return u_seq_batch


    ########################################## ZERO INPUT ROLLOUT ##########################################

    # rollout with zero input sequence (thin wrapper to allow usage of jitted functions)
    def rollout_zero_input(self, T):
        """
        Perform rollout with zero input sequence.

        Args:
            T: int, number of integrations steps
        Returns:
            q_log: jnp.array, logged positions (batch_size, T, nq)
            v_log: jnp.array, logged velocities (batch_size, T, nv)
            u_log: jnp.array, logged inputs (batch_size, T-1, nu) (all zeros)
            c_log: jnp.array, logged contacts (batch_size, T-1, nc) 
        """
    
        # sample initial conditions
        q0_batch, v0_batch = self.sample_random_uniform_initial_conditions()

        # perform rollout
        q_log, v_log, u_log, c_log = self.rollout_zero_input_jit(q0_batch, v0_batch, T)

        return q_log, v_log, u_log, c_log

    # rollout with zero input sequence (pure function to jit)
    def _rollout_zero_input(self, q0_batch, v0_batch, T):
        """
        Perform rollout with zero input sequence.
        Args:
            q0_batch: jnp.array, initial positions (batch_size, nq)
            v0_batch: jnp.array, initial velocities (batch_size, nv)
            T: int, number of integrations steps
        Returns:
            q_log: jnp.array, logged positions (batch_size, T, nq)
            v_log: jnp.array, logged velocities (batch_size, T, nv)
            u_log: jnp.array, logged inputs (batch_size, T-1, nu) (all zeros)
            c_log: jnp.array, logged contacts (batch_size, T-1, nc)
        """

        # number of integration steps
        S = T - 1

        # set the initial conditions in the batched data
        data_0 = jax.vmap(lambda q0, v0: self.mjx_data.replace(qpos=q0, qvel=v0))(q0_batch, v0_batch)

        # set the control to zero in the batched data
        data_0 = data_0.replace(ctrl=self.u_zero)

        # main step body
        def body(data, _):

            # take a step
            data = self.step_fn_batched(data)

            # extract contact pairs
            contact = self.parse_contact(data)

            return data, (data.qpos, data.qvel, contact)

        # forward propagation
        data_last, (q_log, v_log, c_log) = lax.scan(body, data_0, None, length=S)

        # add the initial condition to the logs
        q0 = data_0.qpos   # initial q
        v0 = data_0.qvel   # initial v
        c0 = self.parse_contact(data_0)  # initial contact
        q_log = jnp.concatenate((q0[None, :, :], q_log), axis=0)  # shape (T, batch_size, nq)
        v_log = jnp.concatenate((v0[None, :, :], v_log), axis=0)  # shape (T, batch_size, nv)
        c_log = jnp.concatenate((c0[None, :, :], c_log), axis=0)   # (T, batch, nc)

        # swap axis to get (batch_size, T, dim)
        q_log  = jnp.swapaxes(q_log, 0, 1)  # shape (batch_size, T, nq)
        v_log  = jnp.swapaxes(v_log, 0, 1)  # shape (batch_size, T, nv)
        c_log  = jnp.swapaxes(c_log, 0, 1)  # shape (batch_size, T, nc)

        # u_log is all zeros
        u_log = jnp.broadcast_to(self.u_zero[:, None, :], (self.batch_size, S, self.nu)) # shape (batch_size, T-1, nu)

        # cast logs to desired data type
        q_log, v_log, u_log, c_log = self._cast_logs(q_log, v_log, u_log, c_log)

        return q_log, v_log, u_log, c_log
    

    ######################################### POLICY INPUT ROLLOUT #########################################

    # rollout closed loop using RL policy (thin wrapper to allow usage of jitted functions)
    def rollout_policy_input(self, T):
        """
        Perform rollout with inputs from RL policy.

        Args:
            T: int, number of integrations steps
        Returns:
            q_log: jnp.array, logged positions (batch_size, T, nq)
            v_log: jnp.array, logged velocities (batch_size, T, nv)
            a_log: jnp.array, logged accelerations (batch_size, T, nv)
            u_log: jnp.array, logged inputs (batch_size, T-1, nu)
            c_log: jnp.array, logged contact pairs (batch_size, T, nc)
        """

        # check that policy function is available
        if (self.policy_fn_batched is None) or (self.obs_fn_batched is None):
            raise ValueError("Policy or Observation function is not set.")

        # sample initial conditions
        q0_batch, v0_batch = self.sample_random_uniform_initial_conditions()

        # perform rollout
        q_log, v_log, a_log, u_log, c_log = self.rollout_policy_input_jit(q0_batch, v0_batch, T)

        return q_log, v_log, a_log, u_log, c_log


    # rollout closed loop using RL policy (pure function to jit)
    def _rollout_policy_input(self, q0_batch, v0_batch, T):
        """
        Perform rollout with inputs from RL policy.
        Args:
            q0_batch: jnp.array, initial positions (batch_size, nq)
            v0_batch: jnp.array, initial velocities (batch_size, nv)
            T: int, number of integrations steps
        Returns:
            q_log: jnp.array, logged positions (batch_size, T, nq)
            v_log: jnp.array, logged velocities (batch_size, T, nv)
            a_log: jnp.array, logged accelerations (batch_size, T, nv)
            u_log: jnp.array, logged inputs (batch_size, T-1, nu)
            c_log: jnp.array, logged contact pairs (batch_size, T, nc)
        """

        # number of integration steps
        S = T - 1

        # set the initial conditions in the batched data
        data_0 = jax.vmap(lambda q0, v0: self.mjx_data.replace(qpos=q0, qvel=v0))(q0_batch, v0_batch)
        
        # Apply zero control and step once to populate qacc for initial state
        data_0 = data_0.replace(ctrl=jnp.zeros((self.batch_size, self.nu), dtype=jnp.float32))
        data_0 = self.step_fn_batched(data_0)

        # start with zero action 
        u0 = jnp.zeros((self.batch_size, self.nu), dtype=jnp.float32)  # (batch_size, nu) zeros

        # main step body
        def body(carry, _):
            # unpack carry
            data, u_curr, t = carry

            # update control input at specified decimation
            def compute_control(_):
                obs = self.obs_fn_batched(data)    # (batch_size, obs_dim)
                act = self.policy_fn_batched(obs)  # (batch_size, nu)
                return act

            u_next = lax.cond(
                (t % self.control_decimation) == 0,
                compute_control,
                lambda _: u_curr,
                operand=None
            )

            # apply control and take step
            data = data.replace(ctrl=u_next)
            data = self.step_fn_batched(data)

            # get acceleration directly from data
            a_next = data.qacc

            # extract contact pairs
            contact = self.parse_contact(data)

            return (data, u_next, t + 1), (data.qpos, data.qvel, a_next, u_next, contact)

        # forward propagation
        (data_last, u_last, _), (q_log, v_log, a_log_steps, u_log, c_log) = lax.scan(
            body, (data_0, u0, 0), None, length=S
        )

        # add the initial condition to the logs
        q0 = data_0.qpos                           # (batch_size, nq)
        v0 = data_0.qvel                           # (batch_size, nv)
        a0 = data_0.qacc                           # (batch_size, nv) - now properly populated
        c0 = self.parse_contact(data_0)            # (batch_size, nc)

        # stack to length T for q, v, a, c
        q_log = jnp.concatenate((q0[None, :, :], q_log), axis=0)       # (T, batch, nq)
        v_log = jnp.concatenate((v0[None, :, :], v_log), axis=0)       # (T, batch, nv)
        a_log = jnp.concatenate((a0[None, :, :], a_log_steps), axis=0) # (T, batch, nv)
        c_log = jnp.concatenate((c0[None, ...], c_log), axis=0)        # (T, batch, nc)

        # swap axis to get (batch, T, dim)
        q_log = jnp.swapaxes(q_log, 0, 1)  # (batch, T, nq)
        v_log = jnp.swapaxes(v_log, 0, 1)  # (batch, T, nv)
        a_log = jnp.swapaxes(a_log, 0, 1)  # (batch, T, nv)
        u_log = jnp.swapaxes(u_log, 0, 1)  # (batch, T-1, nu)
        c_log = jnp.swapaxes(c_log, 0, 1)  # (batch, T, nc)

        # cast logs to desired data type
        q_log, v_log, a_log, u_log, c_log = self._cast_logs(q_log, v_log, a_log, u_log, c_log)

        return q_log, v_log, a_log, u_log, c_log


    # rollout closed loop using RL policy given initial conditions
    def rollout_policy_input_x0(self, x0_batch, T):
        """
        Perform rollout with inputs from RL policy.

        Args:
            x0_batch: jnp.array, initial states (batch_size, nx)
            T: int, number of integrations steps
        Returns:
            q_log: jnp.array, logged positions (batch_size, T, nq)
            v_log: jnp.array, logged velocities (batch_size, T, nv)
            u_log: jnp.array, logged inputs (batch_size, T-1, nu)
            c_log: jnp.array, logged contact pairs (batch_size, T, nc)
        """

        # check that policy function is available
        if (self.policy_fn_batched is None) or (self.obs_fn_batched is None):
            raise ValueError("Policy or Observation function is not set.")

        # number of integration steps
        S = T - 1

        # get the batch size
        batch_size = x0_batch.shape[0]

        # extract q0 and v0
        q0_batch = x0_batch[:, :self.nq]
        v0_batch = x0_batch[:, self.nq:self.nq + self.nv]

        # set the initial conditions in the batched data
        data_0 = jax.vmap(lambda q0, v0: self.mjx_data.replace(qpos=q0, qvel=v0))(q0_batch, v0_batch)

        # start with zero action
        u0 = jnp.zeros((batch_size, self.nu), dtype=jnp.float32)  # (batch_size, nu) zeros

        # main step body
        def body(carry, _):

            # unpack carry
            data, u_curr, t = carry

            # update control input at specified decimation
            def compute_control(_):
                obs = self.obs_fn_batched(data)    # get observations          (batch_size, obs_dim)
                act = self.policy_fn_batched(obs)  # get actions from policy   (batch_size, nu)
                return act

            # if time to update control input
            u_next = lax.cond((t % self.control_decimation) == 0,
                               compute_control, 
                               lambda _: u_curr, operand=None)

            # apply control and take step
            data = data.replace(ctrl=u_next)
            data = self.step_fn_batched(data)

            # extract contact pairs
            contact = self.parse_contact(data)

            return (data, u_next, t + 1), (data.qpos, data.qvel, u_next, contact)

        # do the forward propagation
        (data_last, u_last, _), (q_log, v_log, u_log, c_log) = lax.scan(body, (data_0, u0, 0), None, length=S)

        # add the initial condition to the logs
        q0 = data_0.qpos   # initial q
        v0 = data_0.qvel   # initial v 
        c0 = self.parse_contact(data_0)  # initial contact
        q_log = jnp.concatenate((q0[None, :, :], q_log), axis=0)  # shape (T, batch_size, nq)
        v_log = jnp.concatenate((v0[None, :, :], v_log), axis=0)  # shape (T, batch_size, nv)
        c_log = jnp.concatenate((c0[None, ...], c_log), axis=0)   # (T, batch_size, nc)

        # swap axis to get (batch, T, dim)
        q_log = jnp.swapaxes(q_log, 0, 1)  # shape (batch_size, T, nq)
        v_log = jnp.swapaxes(v_log, 0, 1)  # shape (batch_size, T, nv)
        u_log = jnp.swapaxes(u_log, 0, 1)  # shape (batch_size, T-1, nu)
        c_log = jnp.swapaxes(c_log, 0, 1)  # shape (batch_size, T, nc)

        # cast logs to desired data type (a_log not used here, pass empty placeholder)
        a_log = jnp.empty((q_log.shape[0], 0, self.nv))
        q_log, v_log, a_log, u_log, c_log = self._cast_logs(q_log, v_log, a_log, u_log, c_log)

        return q_log, v_log, u_log, c_log


    ######################################### RANDOM INPUT ROLLOUT #########################################

    # rollout with random input sequence (thin wrapper to allow usage of jitted functions)
    def rollout_random_input(self, T):
        """
        Perform rollout with random input sequence.
        Args:
            T: int, number of integrations steps
        Returns:
            q_log: jnp.array, logged positions (batch_size, T, nq)
            v_log: jnp.array, logged velocities (batch_size, T, nv)
            u_log: jnp.array, logged inputs (batch_size, T-1, nu)
            c_log: jnp.array, logged contact forces (batch_size, T, nc)
        """
    
        # sample initial conditions
        q0_batch, v0_batch = self.sample_random_uniform_initial_conditions()

        # sample random input sequence
        u_seq_batch = self.sample_random_uniform_inputs(T-1)

        # perform rollout
        q_log, v_log, u_log, c_log = self.rollout_random_input_jit(q0_batch, v0_batch, u_seq_batch, T)

        # perform rollout
        return q_log, v_log, u_log, c_log

    # rollout with random input sequence (pure function to jit)
    def _rollout_random_input(self, q0_batch, v0_batch, u_seq_batch, T):
        """
        Perform rollout with random input sequence.
        Args:
            q0_batch: jnp.array, initial positions (batch_size, nq)
            v0_batch: jnp.array, initial velocities (batch_size, nv)
            u_seq_batch: jnp.array, input sequence (batch_size, T-1, nu)
            T: int, number of integrations steps
        Returns:
            q_log: jnp.array, logged positions (batch_size, T, nq)
            v_log: jnp.array, logged velocities (batch_size, T, nv)
            u_log: jnp.array, logged inputs (batch_size, T-1, nu)
            c_log: jnp.array, logged contact forces (batch_size, T, nc)
        """

        # number of integration steps
        S = T - 1

        # set the initial conditions in the batched data
        data_0 = jax.vmap(lambda q0, v0: self.mjx_data.replace(qpos=q0, qvel=v0))(q0_batch, v0_batch)

        # swap axis to get (T-1, batch, nu) for lax.scan
        u_seq_batch_swapped = jnp.swapaxes(u_seq_batch, 0, 1)  # (T-1, batch_size, nu)

        # main step body
        def body(data, u_t):

            # apply control and take step
            data = data.replace(ctrl=u_t)
            data = self.step_fn_batched(data)

            # extract contact pairs
            contact = self.parse_contact(data)

            return data, (data.qpos, data.qvel, u_t, contact)

        # forward propagation
        data_last, (q_log, v_log, u_log, c_log) = lax.scan(body, data_0, u_seq_batch_swapped, length=S)

        # add the initial condition to the logs
        q0 = data_0.qpos   # initial q
        v0 = data_0.qvel   # initial v
        c0 = self.parse_contact(data_0)  # initial contact
        q_log = jnp.concatenate((q0[None, :, :], q_log), axis=0)  # shape (T, batch_size, nq)
        v_log = jnp.concatenate((v0[None, :, :], v_log), axis=0)  # shape (T, batch_size, nv)
        c_log = jnp.concatenate((c0[None, :, :], c_log), axis=0)  # shape (T, batch_size, nc)

        # swap axis to get (batch_size, T, dim)
        q_log  = jnp.swapaxes(q_log, 0, 1)  # shape (batch_size, T, nq)
        v_log  = jnp.swapaxes(v_log, 0, 1)  # shape (batch_size, T, nv)
        c_log  = jnp.swapaxes(c_log, 0, 1)  # shape (batch_size, T, nc)

        # the input log is just the input sequence
        u_log  = u_seq_batch                # shape (batch_size, T-1, nu)

        # cast logs to desired data type
        q_log, v_log, u_log, c_log = self._cast_logs(q_log, v_log, u_log, c_log)

        return q_log, v_log, u_log, c_log
    
    
    ######################################### UTILS #########################################

    # generate time series array
    def create_time_array(self, dt, T):
        # build the time array
        t_traj = jnp.arange(T, dtype=jnp.float32) * dt  # shape (T,)
        
        return t_traj

    # parse contact forces from mjx.Data
    def parse_contact(self, data_batch):
        """
        Extract touch sensor values for a batch of mjx.Data.

        Args:
            data_batch: mjx.Data, batched mjx data (batch_size, ...)

        Returns:
            jnp.ndarray, shape (batch_size, nc)
            Contact forces at each touch sensor site.
        """
        if self.nc == 0:
            # no touch sensors defined
            return jnp.zeros((self.batch_size, 0), dtype=jnp.float32)
        
        # data_batch.sensordata has shape (batch_size, nsensor)
        return data_batch.sensordata[:, self.touch_sensor_ids]

    # cast logs to desired data type
    def _cast_logs(self, q_log, v_log, a_log, u_log, c_log):

        # check current log data type
        log_dtype = q_log.dtype

        # cast if needed
        if log_dtype == self.log_dtype:
            # cast not needed
            return (q_log, v_log, a_log, u_log, c_log)
        else:
            # cast to desired data type
            return (q_log.astype(self.log_dtype),
                    v_log.astype(self.log_dtype),
                    a_log.astype(self.log_dtype),
                    u_log.astype(self.log_dtype),
                    c_log.astype(self.log_dtype))

        
##################################################################################
# EXAMPLE USAGE
##################################################################################


if __name__ == "__main__":

    # print the device being used (gpu or cpu)
    device = jax.devices()[0]
    print("Device type:", device.platform)      # e.g. 'gpu' or 'cpu'
    print("Device name:", device.device_kind)   # e.g. 'NVIDIA GeForce RTX 4090'

    # create a random number generator
    # seed = 0
    seed = int(time.time())
    rng = jax.random.PRNGKey(seed)

    # choose batch size
    batch_size = 4096

    # desired data type for logging (for storage savings)
    log_dtype = jnp.float32  

    # choose environment, policy parameters, and state space bounds
    # env_name = "paddle_ball"
    # params_path = "./rl/policy/paddle_ball_policy.pkl"
    # q_lb = jnp.array([ 1.0,  0.2]) 
    # q_ub = jnp.array([ 2.0,  0.8])
    # v_lb = jnp.array([-2.5, -2.5])
    # v_ub = jnp.array([ 0.5,  2.5])
    # sim_dt_des = 0.001  # desired sim dt
    # T = 7500            # trajectory length

    env_name = "hopper"
    params_path = "./rl/policy/hopper_policy.pkl"
    q_lb = jnp.array([-0.001, 1.0, -jnp.pi/4, -0.1])  
    q_ub = jnp.array([ 0.001, 1.5,  jnp.pi/4,  0.1])  
    v_lb = jnp.array([-1.0, -1.0, -1.0, -1.0])
    v_ub = jnp.array([ 1.0,  1.0,  1.0,  1.0])
    sim_dt_des = 0.001  # desired sim dt
    T = 7500            # trajectory length

    # number of datasets to generate
    N_datasets = 10

    # assign the state bounds
    state_bounds = (q_lb, q_ub, v_lb, v_ub)

    # integration type
    # use "RK4" when you need good contact dynamics else use "implicitfast" for speed
    integrator = "rk4"  # "euler", "implicitfast", "rk4"

    # make the config
    config = ParallelSimConfig(env_name=env_name,
                               batch_size=batch_size,
                               sim_dt_des=sim_dt_des,
                               log_dtype=log_dtype,
                               state_bounds=state_bounds,
                               rng=rng,
                               integrator=integrator,
                               policy_params_path=params_path)
    
    # create the rollout instance
    r = ParallelSim(config)

    def rollout_and_save_dataset(save_path, label):

        print(f"\nGenerating {label}...")

        time_0 = time.time()
        q_log, v_log, a_log, u_log, c_log = r.rollout_policy_input(T)
        q_log.block_until_ready()
        v_log.block_until_ready()
        a_log.block_until_ready()
        u_log.block_until_ready()
        c_log.block_until_ready()
        rollout_time = time.time() - time_0
        print(f"Rollout time: {rollout_time:.3f}s")

        q_log_np = np.array(q_log)
        v_log_np = np.array(v_log)
        a_log_np = np.array(a_log)
        u_log_np = np.array(u_log)
        c_log_np = np.array(c_log)

        print(q_log_np.shape)
        print(v_log_np.shape)
        print(a_log_np.shape)
        print(u_log_np.shape)
        print(c_log_np.shape)

        np.savez(save_path,
                 sim_dt=float(r.sim_dt),
                 q_log=q_log_np,
                 v_log=v_log_np,
                 a_log=a_log_np,
                 u_log=u_log_np,
                 c_log=c_log_np)
        print(f"Saved data to: {save_path}")

        return rollout_time

    # generate numbered training datasets plus one standalone testing dataset
    testing_dataset_name = f"{env_name}_data_testing.npz"
    tot_time = 0.0
    raw_data_dir = "./data/mjx/raw_data"
    os.makedirs(raw_data_dir, exist_ok=True)

    for i in range(N_datasets):
        save_path = f"{raw_data_dir}/{env_name}_data_{i+1:02d}.npz"
        tot_time += rollout_and_save_dataset(save_path, f"dataset {i+1}/{N_datasets}")

    # generate testing dataset with 2x batch size (two rollouts concatenated)
    testing_save_path = f"{raw_data_dir}/{testing_dataset_name}"
    print(f"\nGenerating testing dataset (2x batch)...")
    time_0 = time.time()
    q1, v1, a1, u1, c1 = r.rollout_policy_input(T)
    q2, v2, a2, u2, c2 = r.rollout_policy_input(T)
    q2.block_until_ready()
    test_time = time.time() - time_0
    tot_time += test_time
    print(f"Testing rollout time: {test_time:.3f}s")

    q_test = np.concatenate([np.array(q1), np.array(q2)], axis=0)
    v_test = np.concatenate([np.array(v1), np.array(v2)], axis=0)
    a_test = np.concatenate([np.array(a1), np.array(a2)], axis=0)
    u_test = np.concatenate([np.array(u1), np.array(u2)], axis=0)
    c_test = np.concatenate([np.array(c1), np.array(c2)], axis=0)
    print(f"  q_log: {q_test.shape}")
    print(f"  v_log: {v_test.shape}")
    print(f"  a_log: {a_test.shape}")
    print(f"  u_log: {u_test.shape}")
    print(f"  c_log: {c_test.shape}")

    np.savez(testing_save_path,
             sim_dt=float(r.sim_dt),
             q_log=q_test,
             v_log=v_test,
             a_log=a_test,
             u_log=u_test,
             c_log=c_test)
    print(f"  Saved to: {testing_save_path}")

    print("\nFinished generating datasets.")
    print(f"Total rollout time: {tot_time:.3f}s")

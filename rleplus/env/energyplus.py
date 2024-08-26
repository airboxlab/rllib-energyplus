import abc
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from rleplus.env.utils import try_import_energyplus_api

EnergyPlusAPI, DataExchange, _ = try_import_energyplus_api()


@dataclass
class RunnerConfig:
    """Configuration for the runner."""

    # Path to the weather file (.epw)
    epw: Union[Path, str]
    # Path to the IDF file
    idf: Union[Path, str]
    # Path to the output directory
    output: Union[Path, str]
    # EnergyPlus variables to request
    variables: Dict[str, Tuple[str, str]]
    # EnergyPlus meters to request
    meters: Dict[str, str]
    # EnergyPlus actuators to actuate
    # a dict is used here to handle multiple actuators, but only one is possible in this example (discrete action space)
    actuators: Dict[str, Tuple[str, str, str]]
    # Generate eplusout.csv at end of simulation
    csv: bool = False
    # In verbose mode, EnergyPlus will print to stdout
    verbose: bool = False
    # EnergyPlus timestep duration, in fractional hour. Default is 0.25 (15 minutes)
    eplus_timestep_duration: float = 0.25

    def __post_init__(self):
        self.epw = str(self.epw)
        self.idf = str(self.idf)
        self.output = str(self.output)

        # check provided paths exist
        for path in [self.epw, self.idf]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

        # check variables, meters and actuators are not empty
        for name, data in [("variables/meters", {**self.variables, **self.meters}), ("actuators", self.actuators)]:
            if len(data) == 0:
                raise ValueError(f"No {name} provided")

        assert self.eplus_timestep_duration > 0.0, "E+ timestep duration must be > 0.0"


class EnergyPlusRunner:
    """EnergyPlus simulation runner.

    This class is responsible for running EnergyPlus in a separate thread and to interact
    with it through its API.
    """

    def __init__(self, episode: int, obs_queue: Queue, act_queue: Queue, runner_config: RunnerConfig) -> None:
        self.episode = episode
        self.runner_config = runner_config
        self.verbose = self.runner_config.verbose

        self.obs_queue = obs_queue
        self.act_queue = act_queue
        # protect act_queue from concurrent access that can happen at end of simulation
        self.act_queue_mutex = threading.Lock()

        self.energyplus_api = EnergyPlusAPI()
        self.x: DataExchange = self.energyplus_api.exchange
        self.energyplus_exec_thread: Optional[threading.Thread] = None
        self.energyplus_state: Any = None
        self.sim_results: Dict[str, Any] = {}
        self.initialized = False
        self.progress_value: int = 0
        self.simulation_complete = False
        # Zone timestep duration, in fractional hour. Default is 15 minutes
        # Make sure to set this value to reflect your simulation timestep (ie 4 steps per hour in IDF = 0.25)
        self.zone_timestep_duration = self.runner_config.eplus_timestep_duration

        # below is declaration of variables, meters and actuators
        # this simulation will interact with
        self.variables = runner_config.variables
        self.var_handles: Dict[str, int] = {}

        self.meters = runner_config.meters
        self.meter_handles: Dict[str, int] = {}

        self.actuators = runner_config.actuators
        self.actuator_handles: Dict[str, int] = {}
        self.last_action = 0.0

    def start(self) -> None:
        self.energyplus_state = self.energyplus_api.state_manager.new_state()
        runtime = self.energyplus_api.runtime

        # register callback used to track simulation progress
        def _report_progress(progress: int) -> None:
            self.progress_value = progress
            if self.verbose:
                print(f"Simulation progress: {self.progress_value}%")

        runtime.callback_progress(self.energyplus_state, _report_progress)

        runtime.set_console_output_status(self.energyplus_state, self.verbose)

        # register callback used to collect observations
        runtime.callback_end_zone_timestep_after_zone_reporting(self.energyplus_state, self._collect_obs)

        # register callback used to send actions
        runtime.callback_after_predictor_after_hvac_managers(self.energyplus_state, self._send_actions)

        # run EnergyPlus in a non-blocking way
        def _run_energyplus(rn, cmd_args, state, results):
            if self.verbose:
                print(f"running EnergyPlus with args: {cmd_args}")

            # start simulation
            results["exit_code"] = rn.run_energyplus(state, cmd_args)

            if not self.simulation_complete:
                # free consumers from waiting
                self.obs_queue.put(None)
                self.act_queue.put(None)
                self.stop()

        self.energyplus_exec_thread = threading.Thread(
            target=_run_energyplus,
            args=(self.energyplus_api.runtime, self.make_eplus_args(), self.energyplus_state, self.sim_results),
        )
        self.energyplus_exec_thread.start()

    def stop(self) -> None:
        if not self.simulation_complete:
            self.simulation_complete = True
            self._flush_queues()
            self.energyplus_exec_thread.join()
            self.energyplus_exec_thread = None
            self.energyplus_api.runtime.clear_callbacks()
            self.energyplus_api.state_manager.delete_state(self.energyplus_state)

    def failed(self) -> bool:
        return self.sim_results.get("exit_code", -1) > 0

    def make_eplus_args(self) -> List[str]:
        """Make command line arguments to pass to EnergyPlus."""
        eplus_args = ["-r"] if self.runner_config.csv else []
        eplus_args += [
            "-w",
            self.runner_config.epw,
            "-d",
            f"{self.runner_config.output}/episode-{self.episode:08}-{os.getpid():05}",
            self.runner_config.idf,
        ]
        return eplus_args

    def init_exchange(self, default_action: float) -> Dict[str, float]:
        self.last_action = default_action
        self.act_queue.put(default_action)
        return self.obs_queue.get()

    def _collect_obs(self, state_argument) -> None:
        """EnergyPlus callback that collects output variables/meters values and enqueue them."""
        if self.simulation_complete or not self._init_callback(state_argument):
            return

        self.next_obs = {
            **{key: self.x.get_variable_value(state_argument, handle) for key, handle in self.var_handles.items()},
            **{key: self.x.get_meter_value(state_argument, handle) for key, handle in self.meter_handles.items()},
        }
        self.obs_queue.put(self.next_obs)

    def _send_actions(self, state_argument):
        """EnergyPlus callback that sets actuator value from last decided action."""
        if self.simulation_complete or not self._init_callback(state_argument):
            return

        # E+ has zone and system timesteps, a zone timestep can be made of several system timesteps
        # (number varies on each iteration). We should send actions at least once per zone timestep, so we can
        # resend the last action if we are iterating over system timesteps, but we need to wait for a new action
        # when moving from one zone timestep to another.
        sys_timestep_duration = self.x.system_time_step(state_argument)
        if sys_timestep_duration < self.zone_timestep_duration and self.act_queue.empty():
            self.act_queue.put(self.last_action)

        # wait for next action
        with self.act_queue_mutex:
            if self.simulation_complete:
                return
            next_action = self.act_queue.get()

        # end of simulation
        if next_action is None:
            self.simulation_complete = True
            return

        assert isinstance(next_action, float)

        # keep last action to resend it if needed (see above)
        self.last_action = next_action
        # we only have one actuator in this example
        actuator_handle = list(self.actuator_handles.values())[0]
        self.x.set_actuator_value(state=state_argument, actuator_handle=actuator_handle, actuator_value=next_action)

    def _init_callback(self, state_argument) -> bool:
        """Initialize EnergyPlus handles and checks if simulation runtime is ready."""
        self.initialized = self._init_handles(state_argument) and not self.x.warmup_flag(state_argument)
        return self.initialized

    def _init_handles(self, state_argument):
        """Initialize sensors/actuators handles to interact with during simulation."""
        if not self.initialized:
            if not self.x.api_data_fully_ready(state_argument):
                return False

            self.var_handles = {
                key: self.x.get_variable_handle(state_argument, *var) for key, var in self.variables.items()
            }

            self.meter_handles = {
                key: self.x.get_meter_handle(state_argument, meter) for key, meter in self.meters.items()
            }

            self.actuator_handles = {
                key: self.x.get_actuator_handle(state_argument, *actuator) for key, actuator in self.actuators.items()
            }

            for handles in [self.var_handles, self.meter_handles, self.actuator_handles]:
                if any([v == -1 for v in handles.values()]):
                    available_data = self.x.list_available_api_data_csv(state_argument).decode("utf-8")
                    raise RuntimeError(
                        f"got -1 handle, check your var/meter/actuator names:\n"
                        f"> variables: {self.var_handles}\n"
                        f"> meters: {self.meter_handles}\n"
                        f"> actuators: {self.actuator_handles}\n"
                        f"> available E+ API data: {available_data}"
                    )

            self.initialized = True

        return True

    def _flush_queues(self):
        # release waiting threads (if any)
        if self.act_queue.empty():
            self.act_queue.put(None)

        while not self.obs_queue.empty():
            self.obs_queue.get()

        # flush actions queue after last callback was called
        with self.act_queue_mutex:
            while not self.act_queue.empty():
                self.act_queue.get()


class EnergyPlusEnv(gym.Env, metaclass=abc.ABCMeta):
    """Base, abstract EnergyPlus gym environment.

    This class implements the OpenAI gym (now gymnasium) API. It must be subclassed to
    implement the actual environment.
    """

    def __init__(self, env_config: Dict[str, Any]):
        self.spec = gym.envs.registration.EnvSpec(f"{self.__class__.__name__}")

        self.env_config = env_config
        self.episode = -1
        self.timestep = 0

        self.observation_space = self.get_observation_space()
        self.last_obs = {}

        self.action_space = self.get_action_space()
        self.default_action = self.post_process_action(self.action_space.sample())

        self.energyplus_runner: Optional[EnergyPlusRunner] = None
        self.obs_queue: Optional[Queue] = None
        self.act_queue: Optional[Queue] = None

        self.runner_config = RunnerConfig(
            epw=self.get_weather_file(),
            idf=self.get_idf_file(),
            output=self.env_config["output"],
            variables=self.get_variables(),
            meters=self.get_meters(),
            actuators=self.get_actuators(),
            csv=self.env_config.get("csv", False),
            verbose=self.env_config.get("verbose", False),
            eplus_timestep_duration=self.env_config.get("eplus_timestep_duration", 0.25),
        )

    @abc.abstractmethod
    def get_weather_file(self) -> Union[Path, str]:
        """Returns the path to a valid weather file (.epw).

        This method can be used to randomize training data by providing different weather
        files. It's called on each reset()
        """

    @abc.abstractmethod
    def get_idf_file(self) -> Union[Path, str]:
        """Returns the path to a valid IDF file."""

    @abc.abstractmethod
    def get_observation_space(self) -> gym.Space:
        """Returns the observation space of the environment."""

    @abc.abstractmethod
    def get_action_space(self) -> gym.Space:
        """Returns the action space of the environment."""

    @abc.abstractmethod
    def compute_reward(self, obs: Dict[str, float]) -> float:
        """Computes the reward for the given observation."""

    @abc.abstractmethod
    def get_variables(self) -> Dict[str, Tuple[str, str]]:
        """Returns the variables to track during simulation."""

    @abc.abstractmethod
    def get_meters(self) -> Dict[str, str]:
        """Returns the meters to track during simulation."""

    @abc.abstractmethod
    def get_actuators(self) -> Dict[str, Tuple[str, str, str]]:
        """Returns the actuators to control during simulation."""

    def post_process_action(self, action: Union[float, List[float]]) -> Union[float, List[float]]:
        """Post-processes the action(s) before sending it to EnergyPlus.

        This method can be used to implement constraints on the actions, for example.
        Default implementation returns the action unchanged.
        """
        return action

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        self.episode += 1
        self.last_obs = self.observation_space.sample()

        if self.energyplus_runner is not None:
            self.energyplus_runner.stop()

        # observations and actions queues for flow control
        # queues have a default max size of 1
        # as only 1 E+ timestep is processed at a time
        self.obs_queue = Queue(maxsize=1)
        self.act_queue = Queue(maxsize=1)

        self.energyplus_runner = EnergyPlusRunner(
            episode=self.episode,
            obs_queue=self.obs_queue,
            act_queue=self.act_queue,
            runner_config=self.runner_config,
        )
        self.energyplus_runner.start()

        # wait until E+ is ready.
        self.last_obs = obs = self.energyplus_runner.init_exchange(default_action=self.default_action)
        return np.array(list(obs.values())), {}

    def step(self, action):
        self.timestep += 1
        done = False

        # check for simulation errors
        if self.energyplus_runner.failed():
            raise RuntimeError(f"EnergyPlus failed with {self.energyplus_runner.sim_results['exit_code']}")

        # simulation_complete is likely to happen after last env step()
        # is called, hence leading to waiting on queue for a timeout
        if self.energyplus_runner.simulation_complete:
            done = True
            obs = self.last_obs
        else:
            # post-process action
            action_to_apply = self.post_process_action(action)
            # Enqueue action (sent to EnergyPlus through dedicated callback)
            # then wait to get next observation.
            # Timeout is set to 2s to handle end of simulation cases, which happens async
            # and materializes by worker thread waiting on this queue (EnergyPlus callback
            # not consuming anymore).
            # Timeout value can be increased if E+ timestep takes longer
            timeout = 2
            try:
                self.act_queue.put(action_to_apply, timeout=timeout)
                obs = self.obs_queue.get(timeout=timeout)
            except (Full, Empty):
                obs = None
                pass

            # obs can be None if E+ simulation is complete
            # this materializes by either an empty queue or a None value received from queue
            if obs is None:
                done = True
                obs = self.last_obs
            else:
                self.last_obs = obs

        # compute reward
        reward = self.compute_reward(obs)

        # print("obs", obs, "reward", reward, "done", done, "action", action)
        obs_vec = np.array(list(obs.values()))
        return obs_vec, reward, done, False, {}

    def close(self):
        if self.energyplus_runner is not None:
            self.energyplus_runner.stop()

    def render(self, mode="human"):
        pass

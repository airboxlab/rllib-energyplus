"""An example of how to use Pearl to train a Bootstrapped DQN agent on the Amphitheater
environment.

See https://github.com/facebookresearch/Pearl for more configuration options.
"""
import argparse
from tempfile import TemporaryDirectory

from pearl.action_representation_modules.identity_action_representation_module import (
    IdentityActionRepresentationModule,
)
from pearl.history_summarization_modules.lstm_history_summarization_module import (
    LSTMHistorySummarizationModule,
)
from pearl.neural_networks.common.value_networks import EnsembleQValueNetwork
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.bootstrapped_dqn import (
    BootstrappedDQN,
)
from pearl.replay_buffers.sequential_decision_making.bootstrap_replay_buffer import (
    BootstrapReplayBuffer,
)
from pearl.utils.functional_utils.train_and_eval.online_learning import online_learning
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace

from rleplus.examples.registry import env_creator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        help="The gym environment to use.",
        required=False,
        default="AmphitheaterEnv",
    )
    parser.add_argument(
        "--csv", help="Generate eplusout.csv at end of simulation", required=False, default=False, action="store_true"
    )
    parser.add_argument(
        "--verbose",
        help="In verbose mode, EnergyPlus will print to stdout",
        required=False,
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--output",
        help="EnergyPlus output directory. Default is a generated one in /tmp/",
        required=False,
        default=TemporaryDirectory().name,
    )
    parser.add_argument("--timesteps", "-t", help="Number of timesteps to train", required=False, default=1e6)

    built_args = parser.parse_args()
    print(f"Running with following CLI args: {built_args}")
    return built_args


def main():
    args = parse_args()

    # build the environment: we need to wrap the original gym environment in a Pearl environment
    env_cls = env_creator(args.env)
    env = GymEnvironment(
        env_or_env_name=env_cls(
            env_config=dict(
                csv=args.csv,
                verbose=args.verbose,
                output=args.output,
            )
        )
    )
    assert isinstance(env.action_space, DiscreteActionSpace)

    # declare some variables about environment dimensions
    num_actions = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.action_dim
    # Policy learner state dim, as well as hidden dim for the LSTM history summarization module.
    # Note that Pearl flow is: (LSTM) history summarization module -> Policy Learner, hence the LSTM output/hidden dim
    # is the same as the policy learner's state dim
    state_dim = 128

    # Bootstrapped DQN, is an extension of DQN that uses the so-called "deep exploration" mechanism.
    # The main idea is to keep an ensemble of k Q-value networks and on each episode, one of them is sampled and the
    # greedy policy associated with that network is used for exploration.
    # See: https://arxiv.org/abs/1602.04621
    k = 10
    policy_learner = BootstrappedDQN(
        q_ensemble_network=EnsembleQValueNetwork(
            state_dim=state_dim,
            action_dim=act_dim,
            ensemble_size=k,
            output_dim=1,
            hidden_dims=[64, 64],
            prior_scale=0.3,
        ),
        action_space=env.action_space,
        training_rounds=50,
        action_representation_module=IdentityActionRepresentationModule(
            max_number_actions=num_actions,
            representation_dim=act_dim,
        ),
    )

    # History summarization module: we use the LSTM history summarization module
    history_summarization_module = LSTMHistorySummarizationModule(
        observation_dim=obs_dim,
        action_dim=act_dim,
        hidden_dim=state_dim,
        history_length=8,
    )

    # Pearl agent
    agent = PearlAgent(
        policy_learner=policy_learner,
        history_summarization_module=history_summarization_module,
        replay_buffer=BootstrapReplayBuffer(100_000, 1.0, k),
        device_id=-1,
    )

    # run the online learning loop
    online_learning(
        agent=agent,
        env=env,
        number_of_steps=args.timesteps,
        print_every_x_steps=100,
        record_period=10000,
        learn_after_episode=True,
    )


if __name__ == "__main__":
    main()

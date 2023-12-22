import argparse
from tempfile import TemporaryDirectory

import torch
from pearl.action_representation_modules.identity_action_representation_module import (
    IdentityActionRepresentationModule,
)
from pearl.history_summarization_modules.lstm_history_summarization_module import (
    LSTMHistorySummarizationModule,
)
from pearl.neural_networks.common.value_networks import VanillaValueNetwork
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    VanillaActorNetwork,
)
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.ppo import (
    ProximalPolicyOptimization,
)
from pearl.replay_buffers.sequential_decision_making.on_policy_episodic_replay_buffer import (
    OnPolicyEpisodicReplayBuffer,
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
    parser.add_argument(
        "--use-lstm",
        action="store_true",
        help="Whether to use the LSTM history summarization module",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use a GPU for training",
    )

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

    # use the identity action representation module (input = output)
    action_representation_module = IdentityActionRepresentationModule(
        max_number_actions=env.action_space.n,
        representation_dim=env.action_space.action_dim,
    )

    # use the LSTM history summarization module if requested
    history_summarization_module = (
        LSTMHistorySummarizationModule(
            observation_dim=env.observation_space.shape[0],
            action_dim=1,
            hidden_dim=env.observation_space.shape[0],
            history_length=8,
        )
        if args.use_lstm
        else None
    )

    agent = PearlAgent(
        policy_learner=ProximalPolicyOptimization(
            state_dim=env.observation_space.shape[0],
            action_space=env.action_space,
            actor_network_type=VanillaActorNetwork,
            actor_hidden_dims=[256, 256],
            critic_network_type=VanillaValueNetwork,
            critic_hidden_dims=[256, 256],
            actor_learning_rate=0.003,
            critic_learning_rate=0.003,
            discount_factor=0.95,
            training_rounds=30,
            batch_size=1024,
            epsilon=0.1,
            entropy_bonus_scaling=0.01,
            action_representation_module=action_representation_module,
        ),
        replay_buffer=OnPolicyEpisodicReplayBuffer(10_000),
        history_summarization_module=history_summarization_module,
        device_id=0 if torch.cuda.is_available() and args.use_gpu else -1,
    )

    online_learning(
        agent=agent,
        env=env,
        number_of_steps=args.timesteps,
        print_every_x_steps=100,
        record_period=35040,
        learn_after_episode=True,
    )


if __name__ == "__main__":
    main()

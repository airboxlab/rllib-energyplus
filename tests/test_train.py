import unittest
from tempfile import TemporaryDirectory

import torch
from pearl.action_representation_modules.one_hot_action_representation_module import (
    OneHotActionTensorRepresentationModule,
)
from pearl.history_summarization_modules.lstm_history_summarization_module import (
    LSTMHistorySummarizationModule,
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


class TestTrain(unittest.TestCase):
    def test_ray_runner_config(self):
        from rleplus.train.rllib import parse_args

        args = parse_args()
        self.assertTrue(args.output.startswith("/tmp/"))
        self.assertEqual(args.timesteps, 1e6)
        self.assertEqual(args.num_workers, 2)
        self.assertEqual(args.num_gpus, 0)
        self.assertEqual(args.alg, "PPO")
        self.assertFalse(args.use_lstm)

    def test_pearl(self):
        env_cls = env_creator("AmphitheaterEnv")
        env = GymEnvironment(
            env_or_env_name=env_cls(
                env_config=dict(
                    csv=False,
                    verbose=False,
                    output=TemporaryDirectory().name,
                )
            )
        )
        assert isinstance(env.action_space, DiscreteActionSpace)

        # use the identity action representation module (input = output)
        # action_representation_module = IdentityActionRepresentationModule(
        #     max_number_actions=env.action_space.n,
        #     representation_dim=env.action_space.action_dim,
        # )
        action_representation_module = OneHotActionTensorRepresentationModule(
            max_number_actions=env.action_space.n,
        )

        # use the LSTM history summarization module if requested
        history_summarization_module = LSTMHistorySummarizationModule(
            observation_dim=env.observation_space.shape[0],
            action_dim=100,
            hidden_dim=env.observation_space.shape[0],
            history_length=8,
        )

        agent = PearlAgent(
            policy_learner=ProximalPolicyOptimization(
                env.observation_space.shape[0],
                env.action_space,
                actor_hidden_dims=[64, 64],
                critic_hidden_dims=[64, 64],
                training_rounds=50,
                batch_size=64,
                epsilon=0.1,
                action_representation_module=OneHotActionTensorRepresentationModule(
                    max_number_actions=env.action_space.n
                ),
            ),
            replay_buffer=OnPolicyEpisodicReplayBuffer(10_000),
            history_summarization_module=history_summarization_module,
            device_id=0,
        )
        online_learning(
            agent=agent,
            env=env,
            number_of_steps=100000,
            print_every_x_steps=100,
            record_period=10000,
            learn_after_episode=True,
        )

        # agent = PearlAgent(
        #     policy_learner=ProximalPolicyOptimization(
        #         state_dim=env.observation_space.shape[0],
        #         action_space=env.action_space,
        #         actor_network_type=VanillaActorNetwork,
        #         actor_hidden_dims=[256, 256],
        #         critic_network_type=VanillaValueNetwork,
        #         critic_hidden_dims=[256, 256],
        #         actor_learning_rate=0.003,
        #         critic_learning_rate=0.003,
        #         discount_factor=0.95,
        #         training_rounds=30,
        #         batch_size=1024,
        #         epsilon=0.1,
        #         entropy_bonus_scaling=0.01,
        #         action_representation_module=action_representation_module,
        #     ),
        #     replay_buffer=OnPolicyEpisodicReplayBuffer(10_000),
        #     history_summarization_module=history_summarization_module,
        #     device_id=0 if torch.cuda.is_available() and args.use_gpu else -1,
        # )
        #
        # online_learning(
        #     agent=agent,
        #     env=env,
        #     number_of_steps=args.timesteps,
        #     print_every_x_steps=100,
        #     learn_after_episode=True,
        # )

    def test_ppo(self):
        torch.autograd.set_detect_anomaly(True)

        env = GymEnvironment("CartPole-v1")
        assert isinstance(env.action_space, DiscreteActionSpace)
        num_actions = env.action_space.n

        history_summarization_module = LSTMHistorySummarizationModule(
            observation_dim=env.observation_space.shape[0],
            action_dim=2,
            hidden_dim=4,
            history_length=8,
        )

        # agent = PearlAgent(
        #     policy_learner=DeepQLearning(
        #         state_dim=env.observation_space.shape[0],
        #         action_space=env.action_space,
        #         hidden_dims=[64, 64],
        #         training_rounds=20,
        #         action_representation_module=OneHotActionTensorRepresentationModule(
        #              max_number_actions=num_actions
        #          ),
        #     ),
        #     history_summarization_module=history_summarization_module,
        #     replay_buffer=FIFOOffPolicyReplayBuffer(10_000),
        #     device_id=-1,
        # )

        agent = PearlAgent(
            policy_learner=ProximalPolicyOptimization(
                env.observation_space.shape[0],
                env.action_space,
                actor_hidden_dims=[64, 64],
                critic_hidden_dims=[64, 64],
                training_rounds=50,
                batch_size=64,
                epsilon=0.1,
                action_representation_module=OneHotActionTensorRepresentationModule(max_number_actions=num_actions),
            ),
            history_summarization_module=history_summarization_module,
            replay_buffer=OnPolicyEpisodicReplayBuffer(10_000),
            device_id=-1,
        )

        online_learning(
            agent=agent,
            env=env,
            number_of_steps=100000,
            print_every_x_steps=100,
            record_period=10000,
            learn_after_episode=True,
        )

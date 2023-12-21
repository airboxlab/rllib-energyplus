"""An example of using Ray RLlib to train a PPO agent on EnergyPlus."""

import argparse
from tempfile import TemporaryDirectory

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig

from rleplus.examples.registry import register_all


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
        "--num-workers",
        type=int,
        default=2,
        help="The number of workers to use",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="The number of GPUs to use",
    )
    parser.add_argument(
        "--alg",
        default="PPO",
        choices=["APEX", "DQN", "IMPALA", "PPO", "R2D2"],
        help="The algorithm to use",
    )
    parser.add_argument(
        "--use-lstm",
        action="store_true",
        help="Whether to auto-wrap the model with an LSTM. Only valid option for " "--run=[IMPALA|PPO|R2D2]",
    )
    built_args = parser.parse_args()
    print(f"Running with following CLI args: {built_args}")
    return built_args


def main():
    args = parse_args()

    ray.init()

    register_all()

    # Ray configuration. See Ray docs for tuning
    config = (
        PPOConfig()
        .environment(
            env=args.env,
            env_config=vars(args),
        )
        .training(
            gamma=0.95,
            lr=0.003,
            kl_coeff=0.3,
            train_batch_size=4000,
            sgd_minibatch_size=128,
            vf_loss_coeff=0.01,
            use_critic=True,
            use_gae=True,
            model={
                "use_lstm": args.use_lstm,
                "vf_share_layers": False,
            },
            _enable_learner_api=True,
        )
        .rl_module(_enable_rl_module_api=True)
        .framework(
            # to use tensorflow, you'll need install it first,
            # then set framework="tf2" and eager_tracing=True (for fast exec)
            framework="torch",
        )
        .resources(num_gpus=args.num_gpus)
        .rollouts(
            num_rollout_workers=args.num_workers,
            rollout_fragment_length="auto",
        )
    )

    print("PPO config:", config.to_dict())

    tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop={"timesteps_total": args.timesteps},
            failure_config=air.FailureConfig(max_failures=0, fail_fast=True),
        ),
        param_space=config.to_dict(),
    ).fit()

    ray.shutdown()


if __name__ == "__main__":
    main()

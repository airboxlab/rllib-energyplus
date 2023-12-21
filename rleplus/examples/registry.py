def env_creator(env_name):
    if env_name == "AmphitheaterEnv":
        from .amphitheater.env import AmphitheaterEnv

        return AmphitheaterEnv
    else:
        raise NotImplementedError


def ray_register(env_name):
    from ray.tune.registry import register_env

    env = env_creator(env_name)
    register_env(env_name, lambda cfg: env(cfg))


def register_all():
    try:
        ray_register("AmphitheaterEnv")
    except ImportError:
        pass

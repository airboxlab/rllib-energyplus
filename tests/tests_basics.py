import unittest
from pathlib import Path

from rllibenergyplus.run import EnergyPlusEnv


class TestEnv(unittest.TestCase):

    def test_env_reset_close(self):
        root_dir = Path(__file__).parent.parent
        env = EnergyPlusEnv({
            "idf": f"{root_dir}/model.idf",
            "epw": f"{root_dir}/LUX_LU_Luxembourg.AP.065900_TMYx.2004-2018.epw",
            "output": "/tmp/tests_output"
        })

        obs, _ = env.reset()
        self.assertIsNotNone(obs)
        self.assertEqual((7,), obs.shape)

        obs, rew, done, _, _ = env.step(0)
        self.assertIsNotNone(obs)
        self.assertEqual((7,), obs.shape)
        self.assertNotEquals(0, rew)
        self.assertFalse(done)

        env.close()

    def test_env_serializable(self):
        import ray

        serializable, _ = ray.util.inspect_serializability(EnergyPlusEnv({}))
        self.assertTrue(serializable)

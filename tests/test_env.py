import unittest
from pathlib import Path
from unittest.mock import patch

from rleplus.env.energyplus import RunnerConfig
from rleplus.examples.amphitheater.env import AmphitheaterEnv


class TestEnv(unittest.TestCase):
    def test_env_reset_close(self):
        env = AmphitheaterEnv({"output": "/tmp/tests_output"})

        obs, _ = env.reset()
        self.assertIsNotNone(obs)
        self.assertEqual((7,), obs.shape)

        obs, rew, done, _, _ = env.step(0)
        self.assertIsNotNone(obs)
        self.assertEqual((7,), obs.shape)
        self.assertNotEqual(0, rew)
        self.assertFalse(done)

        env.close()

    def test_demo_env_serializable(self):
        import ray

        serializable, _ = ray.util.inspect_serializability(AmphitheaterEnv({"output": "/tmp/tests_output"}))
        self.assertTrue(serializable)

    def test_env_runner_config(self):
        idf = Path(__file__).parent / "model.idf"
        epw = Path(__file__).parent / "LUX_LU_Luxembourg.AP.065900_TMYx.2004-2018.epw"
        output = "/tmp/tests_output"
        variables = {
            "oat": ("Site Outdoor Air DryBulb Temperature", "Environment"),
            "iat": ("Zone Mean Air Temperature", "TZ_Amphitheater"),
            "co2": ("Zone Air CO2 Concentration", "TZ_Amphitheater"),
            "htg_spt": ("Schedule Value", "HTG HVAC 1 ADJUSTED BY 1.1 F"),
            "clg_spt": ("Schedule Value", "CLG HVAC 1 ADJUSTED BY 0 F"),
        }
        meters = {
            "elec": "Electricity:HVAC",
        }
        actuators = {"sat_spt": ("System Node Setpoint", "Temperature Setpoint", "Node 3")}

        with self.assertRaises(FileNotFoundError):
            RunnerConfig(idf=idf, epw=epw, output=output, variables=variables, meters=meters, actuators=actuators)

        with self.assertRaises(ValueError) as e, patch("os.path.exists", return_value=True):
            RunnerConfig(idf=idf, epw=epw, output=output, variables={}, meters={}, actuators=actuators)
            self.assertEqual("No variables/meters provided", str(e.exception))

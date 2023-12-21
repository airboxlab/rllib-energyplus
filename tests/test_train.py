import unittest


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

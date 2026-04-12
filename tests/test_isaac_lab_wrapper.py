import numpy as np

from srl.envs.isaac_lab_wrapper import IsaacLabWrapper


class _FakeDeviceEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self.device = "cpu"
        self.closed = False

    def reset(self):
        return {
            "policy": np.zeros((2, 8, 8, 3), dtype=np.float32),
            "critic": np.ones((2, 4), dtype=np.float32),
        }, {"reset": True}

    def step(self, actions):
        assert tuple(actions.shape) == (2, 1)
        obs = {
            "policy": np.zeros((2, 8, 8, 3), dtype=np.float32),
            "critic": np.full((2, 4), 2.0, dtype=np.float32),
        }
        reward = np.array([1.0, 2.0], dtype=np.float32)
        terminated = np.array([False, True])
        truncated = np.array([False, False])
        info = {"is_success": np.array([0.0, 1.0], dtype=np.float32)}
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.closed = True


def test_isaac_lab_wrapper_preserves_group_keys_and_transposes_images() -> None:
    env = IsaacLabWrapper(_FakeDeviceEnv())

    obs, info = env.reset()

    assert info == {"reset": True}
    assert set(obs.keys()) == {"policy", "critic"}
    assert obs["policy"].shape == (2, 3, 8, 8)
    assert obs["critic"].shape == (2, 4)


def test_isaac_lab_wrapper_step_converts_outputs() -> None:
    env = IsaacLabWrapper(_FakeDeviceEnv())

    obs, reward, terminated, truncated, info = env.step(np.zeros((2, 1), dtype=np.float32))

    assert obs["policy"].shape == (2, 3, 8, 8)
    assert reward.tolist() == [1.0, 2.0]
    assert terminated.tolist() == [False, True]
    assert truncated.tolist() == [False, False]
    assert info["is_success"].tolist() == [0.0, 1.0]


def test_isaac_lab_wrapper_close_delegates() -> None:
    base_env = _FakeDeviceEnv()
    env = IsaacLabWrapper(base_env)

    env.close()

    assert base_env.closed is True
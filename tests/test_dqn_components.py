import unittest

import numpy as np
import torch

from dqn import DQNAgent, QNetwork, ReplayBuffer


class ReplayBufferTests(unittest.TestCase):
    def test_sample_returns_tensors_with_expected_shapes(self):
        buffer = ReplayBuffer(capacity=10, observation_dim=4)

        for index in range(6):
            state = np.array([index, 0.0, 0.5, 0.5 - index], dtype=np.float32)
            next_state = state + 0.1
            buffer.push(state, 2, 1.0, next_state, False)

        batch = buffer.sample(batch_size=4, device=torch.device("cpu"))

        self.assertEqual(batch.states.shape, (4, 4))
        self.assertEqual(batch.actions.shape, (4, 1))
        self.assertEqual(batch.rewards.shape, (4, 1))
        self.assertEqual(batch.next_states.shape, (4, 4))
        self.assertEqual(batch.terminated.shape, (4, 1))
        self.assertEqual(len(buffer), 6)


class QNetworkTests(unittest.TestCase):
    def test_forward_returns_one_q_value_per_action(self):
        network = QNetwork(observation_dim=4, action_dim=3)
        inputs = torch.zeros((5, 4), dtype=torch.float32)

        outputs = network(inputs)

        self.assertEqual(outputs.shape, (5, 3))


class DQNAgentTests(unittest.TestCase):
    def test_select_action_uses_greedy_policy_when_epsilon_is_zero(self):
        agent = DQNAgent(observation_dim=4, action_dim=3, seed=7)

        with torch.no_grad():
            for parameter in agent.online_network.parameters():
                parameter.zero_()
            agent.online_network.model[-1].bias.copy_(
                torch.tensor([0.0, 1.0, 3.0], dtype=torch.float32)
            )

        action = agent.select_action(
            np.zeros(4, dtype=np.float32),
            epsilon=0.0,
        )

        self.assertEqual(action, 2)

    def test_optimize_model_returns_loss_after_warmup(self):
        agent = DQNAgent(
            observation_dim=4,
            action_dim=3,
            batch_size=4,
            warmup_transitions=4,
            seed=11,
        )

        for index in range(5):
            state = np.array([0.0, 0.0, 0.2, 0.2], dtype=np.float32)
            next_state = np.array([0.1, 0.0, 0.2, 0.1], dtype=np.float32)
            agent.replay_buffer.push(
                state=state,
                action=index % 3,
                reward=1.0,
                next_state=next_state,
                terminated=False,
            )

        loss = agent.optimize_model()

        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(agent.optimization_steps, 1)


if __name__ == "__main__":
    unittest.main()

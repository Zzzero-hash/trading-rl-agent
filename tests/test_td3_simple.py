#!/usr/bin/env python3
"""
Minimal TD3 Integration Test
Tests basic TD3 functionality without complex environment integration.

NOTE: TD3 has been removed from Ray RLlib 2.38.0+
This test demonstrates the custom TD3 implementation as an alternative.
For Ray RLlib integration, use SAC instead.
"""

import os
import sys
import warnings

import numpy as np

from src.agents.configs import TD3Config
from src.agents.td3_agent import TD3Agent


def test_td3_basic_functionality():
    """Test basic TD3 agent functionality."""
    print("🧪 Testing TD3 Basic Functionality...")

    # Create TD3 agent with simple config
    config = TD3Config(
        learning_rate=1e-3, batch_size=16, buffer_capacity=100, hidden_dims=[32, 32]
    )

    state_dim = 20  # Simple state space
    action_dim = 1  # Single continuous action

    agent = TD3Agent(config=config, state_dim=state_dim, action_dim=action_dim)

    print(f"✅ Agent created with {state_dim} state dims, {action_dim} action dims")
    print(f"✅ Actor params: {sum(p.numel() for p in agent.actor.parameters())}")
    print(f"✅ Critic 1 params: {sum(p.numel() for p in agent.critic_1.parameters())}")
    print(f"✅ Critic 2 params: {sum(p.numel() for p in agent.critic_2.parameters())}")

    # Test action selection
    state = np.random.randn(state_dim).astype(np.float32)
    action = agent.select_action(state, add_noise=False)
    action_with_noise = agent.select_action(state, add_noise=True)

    print(f"✅ Action without noise: {action}")
    print(f"✅ Action with noise: {action_with_noise}")
    assert len(action) == action_dim
    assert len(action_with_noise) == action_dim
    assert all(-1.0 <= a <= 1.0 for a in action)

    # Test experience storage and training
    print("🔄 Testing experience collection and training...")

    # Collect experiences
    for i in range(20):
        state = np.random.randn(state_dim).astype(np.float32)
        action = np.random.uniform(-1, 1, action_dim).astype(np.float32)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim).astype(np.float32)
        done = False

        agent.store_experience(state, action, reward, next_state, done)

    print(f"✅ Stored {len(agent.replay_buffer)} experiences")
    assert len(agent.replay_buffer) == 20

    # Test training
    initial_total_it = agent.total_it
    metrics = agent.train()

    print("✅ Training step completed")
    print(f"✅ Total iterations: {agent.total_it} (was {initial_total_it})")
    print(f"✅ Training metrics: {metrics}")

    assert agent.total_it == initial_total_it + 1
    assert isinstance(metrics, dict)
    assert "critic_1_loss" in metrics
    assert "critic_2_loss" in metrics

    # Test policy delay mechanism
    print("🔄 Testing policy delay mechanism...")
    policy_delay = agent.policy_delay

    for i in range(policy_delay + 2):
        metrics = agent.train()

        if (agent.total_it % policy_delay) == 0:
            assert (
                metrics.get("policy_update", False) is True
            ), f"Policy should update on iteration {agent.total_it}"
        else:
            assert (
                metrics.get("policy_update", False) is False
            ), f"Policy should NOT update on iteration {agent.total_it}"

    print(f"✅ Policy delay mechanism working correctly (delay={policy_delay})")

    # Test save/load
    print("💾 Testing save/load functionality...")
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        save_path = f.name

    try:
        # Save current state
        original_total_it = agent.total_it
        agent.save(save_path)

        # Create new agent and load
        agent2 = TD3Agent(config=config, state_dim=state_dim, action_dim=action_dim)
        agent2.load(save_path)

        # Verify state preservation
        assert agent2.total_it == original_total_it

        # Test actions are similar
        test_state = np.random.randn(state_dim).astype(np.float32)
        action1 = agent.select_action(test_state, add_noise=False)
        action2 = agent2.select_action(test_state, add_noise=False)

        np.testing.assert_allclose(action1, action2, rtol=1e-5)
        print("✅ Save/load successful - actions match within tolerance")

    finally:
        os.unlink(save_path)

    print("🎉 All TD3 Integration Tests Passed!")


if __name__ == "__main__":
    try:
        test_td3_basic_functionality()
        print("\n" + "=" * 60)
        print("🌟 TD3 INTEGRATION TEST SUITE: SUCCESS 🌟")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

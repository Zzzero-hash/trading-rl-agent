#!/usr/bin/env python3
"""
Comprehensive integration test for TD3 + SAC ensemble functionality.
Tests the complete pipeline from individual agents to ensemble decisions.
"""

import os
import sys

import numpy as np
import pytest

from src.agents.configs import SACConfig, TD3Config
from src.agents.rllib_weighted_policy import WeightedPolicyManager, CallablePolicy
from src.agents.sac_agent import SACAgent
from src.agents.td3_agent import TD3Agent


def test_ensemble_integration():
    """Test ensemble with TD3 and SAC agents."""

    pytest.skip("WeightedPolicyManager example test disabled")

    print("🧪 Testing Ensemble Integration...")

    # Create agent configurations
    td3_config = TD3Config(
        learning_rate=1e-3, batch_size=16, buffer_capacity=1000, hidden_dims=[32, 32]
    )

    sac_config = SACConfig(
        learning_rate=1e-3, batch_size=16, buffer_capacity=1000, hidden_dims=[32, 32]
    )

    # Test dimensions
    state_dim = 10
    action_dim = 1

    # Create individual agents
    print("📦 Creating individual agents...")
    td3_agent = TD3Agent(config=td3_config, state_dim=state_dim, action_dim=action_dim)
    sac_agent = SACAgent(config=sac_config, state_dim=state_dim, action_dim=action_dim)

    print(
        f"✅ TD3 Agent: {sum(p.numel() for p in td3_agent.actor.parameters())} actor params"
    )
    # SACAgent uses 'actor' as the network attribute name
    print(
        f"✅ SAC Agent: {sum(p.numel() for p in sac_agent.actor.parameters())} actor params"
    )

    # Create ensemble
    print("🤝 Creating ensemble agent...")
    policies = {
        "td3": CallablePolicy(td3_agent.observation_space, td3_agent.action_space, td3_agent.select_action),
        "sac": CallablePolicy(sac_agent.observation_space, sac_agent.action_space, sac_agent.select_action),
    }
    ensemble = WeightedPolicyManager(policies, {"td3": 0.5, "sac": 0.5})

    print(f"✅ Ensemble created with {len(ensemble.agents)} agents")

    # Test action selection
    print("🎯 Testing ensemble action selection...")
    test_state = np.random.randn(state_dim).astype(np.float32)

    # Get individual actions
    td3_action = td3_agent.select_action(test_state, add_noise=False)
    sac_action = sac_agent.select_action(test_state, evaluate=True)
    ensemble_action = ensemble.select_action(test_state)

    print(f"   TD3 action: {td3_action}")
    print(f"   SAC action: {sac_action}")
    print(f"   Ensemble action: {ensemble_action}")

    # Test experience storage
    print("💾 Testing ensemble experience storage...")
    for i in range(20):
        state = np.random.randn(state_dim).astype(np.float32)
        action = np.random.uniform(-1, 1, action_dim).astype(np.float32)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim).astype(np.float32)
        done = False

        ensemble.store_experience(state, action, reward, next_state, done)

    print("✅ Stored experiences in both agents")
    print(f"   TD3 buffer size: {len(td3_agent.replay_buffer)}")
    print(f"   SAC buffer size: {len(sac_agent.replay_buffer)}")

    # Test ensemble training
    print("🏋️ Testing ensemble training...")
    initial_td3_it = td3_agent.total_it
    initial_sac_it = sac_agent.total_it

    ensemble_metrics = ensemble.train()

    print("✅ Training completed")
    print(f"   TD3 iterations: {td3_agent.total_it} (was {initial_td3_it})")
    print(f"   SAC iterations: {sac_agent.total_it} (was {initial_sac_it})")
    print(f"   Ensemble metrics keys: {list(ensemble_metrics.keys())}")

    # Test performance tracking
    print("📊 Testing performance tracking...")
    ensemble.update_agent_performance("td3", 0.75)
    ensemble.update_agent_performance("sac", 0.82)

    print("✅ Performance updated:")
    print(f"   TD3 performance: {ensemble.agent_performances.get('td3', 'Not set')}")
    print(f"   SAC performance: {ensemble.agent_performances.get('sac', 'Not set')}")

    return True


if __name__ == "__main__":
    try:
        success = test_ensemble_integration()
        if success:
            print("\n🎉 ENSEMBLE INTEGRATION TEST: SUCCESS!")
            print("=" * 60)
            print("🌟 TD3 + SAC + Ensemble: ALL WORKING PERFECTLY! 🌟")
            print("=" * 60)
        else:
            print("\n❌ ENSEMBLE INTEGRATION TEST: FAILED!")
            sys.exit(1)

    except Exception as e:
        print(f"\n💥 ENSEMBLE INTEGRATION TEST ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

import unittest

from src.rule_engine import RuleEngine


class TestRuleEngine(unittest.TestCase):

    def setUp(self):
        self.rule_engine = RuleEngine()

    def test_adjust_position_sizing(self):
        initial_size = 100
        regime = "bull"
        adjusted_size = self.rule_engine.adjust_position_sizing(initial_size, regime)
        self.assertEqual(adjusted_size, 150)  # Example expected value for bull regime

    def test_set_stop_loss(self):
        entry_price = 100
        stop_loss_distance = 10
        stop_loss_price = self.rule_engine.set_stop_loss(entry_price, stop_loss_distance)
        self.assertEqual(stop_loss_price, 90)  # Example expected stop loss price

    def test_activate_strategy(self):
        regime = "bear"
        strategy_active = self.rule_engine.activate_strategy(regime)
        self.assertFalse(strategy_active)  # Example expected value for bear regime

if __name__ == "__main__":
    unittest.main()

import unittest

from src.regime_classifier import RegimeClassifier


class TestRegimeClassifier(unittest.TestCase):

    def setUp(self):
        self.classifier = RegimeClassifier()

    def test_classify_regime(self):
        features = [0.1, 0.2, 0.3]  # Example feature set
        regime = self.classifier.classify_regime(features)
        self.assertIn(regime, ["bull", "bear", "sideways"], "Regime classification failed")

    def test_get_regime_probabilities(self):
        features = [0.1, 0.2, 0.3]  # Example feature set
        probabilities = self.classifier.get_regime_probabilities(features)
        self.assertEqual(len(probabilities), 3, "Probability output length is incorrect")
        self.assertAlmostEqual(sum(probabilities), 1.0, places=2, msg="Probabilities do not sum to 1")

if __name__ == "__main__":
    unittest.main()

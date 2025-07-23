import unittest

from src.feature_extractor import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):

    def setUp(self):
        self.feature_extractor = FeatureExtractor()

    def test_extract_volatility(self):
        # Sample data for testing
        sample_data = [100, 102, 101, 105, 103]
        volatility = self.feature_extractor.extract_volatility(sample_data)
        self.assertIsInstance(volatility, float)
        self.assertGreaterEqual(volatility, 0)

    def test_extract_volume(self):
        # Sample data for testing
        sample_data = [1000, 1500, 1200, 1300, 1100]
        volume = self.feature_extractor.extract_volume(sample_data)
        self.assertIsInstance(volume, float)
        self.assertGreaterEqual(volume, 0)

    def test_extract_order_book_imbalance(self):
        # Sample data for testing
        bid_volume = [100, 150, 200]
        ask_volume = [120, 130, 180]
        imbalance = self.feature_extractor.extract_order_book_imbalance(bid_volume, ask_volume)
        self.assertIsInstance(imbalance, float)
        self.assertGreaterEqual(imbalance, -1)  # Assuming imbalance can be negative

if __name__ == "__main__":
    unittest.main()

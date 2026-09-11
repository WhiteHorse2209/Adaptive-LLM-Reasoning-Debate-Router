import unittest
import os
import sys

# Add src to path so we can import it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.config import load_config, get_active_profile

class TestConfig(unittest.TestCase):
    def test_load_config(self):
        config = load_config("config.json")
        self.assertIn("active_profile", config)
        self.assertIn("profiles", config)
        self.assertIn("router", config)

    def test_get_active_profile(self):
        config = load_config("config.json")
        profile = get_active_profile(config)
        self.assertEqual(profile["provider"], "ollama")
        self.assertIn("model", profile)

if __name__ == "__main__":
    unittest.main()

import unittest


class ModelBRuntimeTests(unittest.TestCase):
    def test_frozen_ood_assets_and_chain_load(self):
        from src.phototrap_ood.chain import FrozenDecisionChain

        chain = FrozenDecisionChain()
        self.assertIsNotNone(chain.model)
        self.assertIn("policy_B", chain.model)
        self.assertIsInstance(chain.model["policy_B"], dict)

    def test_active_runtime_imports(self):
        import src.model_b_ood.runtime.bioclip
        import src.model_b_ood.runtime.features
        import src.model_b_ood.runtime.gate
        import src.model_b_ood.runtime.speciesnet


if __name__ == "__main__":
    unittest.main()

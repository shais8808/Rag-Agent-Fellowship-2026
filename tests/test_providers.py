import unittest

from providers import _get_openrouter_max_tokens, _get_openrouter_model_candidates


class OpenRouterProviderTests(unittest.TestCase):
    def test_claude_model_alias_uses_current_openrouter_id(self) -> None:
        candidates = _get_openrouter_model_candidates("anthropic/claude-3-5-sonnet")
        self.assertEqual(candidates[0], "anthropic/claude-3.5-sonnet")

    def test_mistral_model_falls_back_to_a_supported_option(self) -> None:
        candidates = _get_openrouter_model_candidates("mistralai/mistral-7b-instruct")
        self.assertIn("openai/gpt-4o-mini", candidates)

    def test_openrouter_max_tokens_uses_safe_default(self) -> None:
        self.assertEqual(_get_openrouter_max_tokens(None), 1000)
        self.assertEqual(_get_openrouter_max_tokens("800"), 800)
        self.assertEqual(_get_openrouter_max_tokens("9999"), 2000)


if __name__ == "__main__":
    unittest.main()

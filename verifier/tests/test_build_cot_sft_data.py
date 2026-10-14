import unittest

from utils.build_cot_sft_data import parse_teacher_answer, request_for, response_text, select_rows, soclaas_url, validate


ROW = {
    "id": 1,
    "gold": "SUPPORTED",
    "prompt": "Claim: The sky is blue.\n\nEvidence:\nE1 [Sky] The sky is blue.\nE2 [Earth] Earth is a planet.",
    "completion": '{"verdict": "SUPPORTED", "evidence": ["E1"], "confidence": 1.0}',
}


class CotDataTests(unittest.TestCase):
    def test_gold_and_reasoning_are_separate(self):
        request = request_for(ROW)
        self.assertEqual(request["generationConfig"]["responseFormat"]["text"]["mimeType"], "application/json")
        self.assertIn("Gold evidence IDs: [\"E1\"]", request["contents"][0]["parts"][0]["text"])

    def test_rejects_invalid_and_unfaithful_citation_structure(self):
        answer = {"reasoning": "E1 directly states that the sky is blue.",
                  "relation": "supports", "cited_evidence": ["E1"], "gold_justified": True}
        self.assertIsNone(validate(ROW, answer))
        self.assertEqual(validate(ROW, {**answer, "cited_evidence": ["E9"]}), "citation_mismatch")
        self.assertEqual(validate(ROW, {**answer, "gold_justified": False}), "gold_not_justified_by_context")
        self.assertEqual(validate(ROW, {**answer, "reasoning": "E2 is relevant to the claim about the sky.",
                                        "cited_evidence": ["E2"]}),
                         "gold_evidence_not_explained")
        self.assertIsNone(validate(ROW, {**answer, "reasoning": "E1 states the sky is blue; E2 is unrelated."}))
        self.assertEqual(validate(ROW, {**answer, "reasoning": "The shown evidence says the sky is blue."}),
                         "citation_mismatch")

    def test_markdown_fenced_json_is_parsed_without_accepting_trailing_text(self):
        self.assertEqual(parse_teacher_answer('```json\n{"relation":"supports"}\n```'),
                         {"relation": "supports"})
        with self.assertRaises(ValueError):
            parse_teacher_answer('```json\n{"relation":"supports"}\n``` extra')

    def test_balanced_selection_is_deterministic(self):
        rows = [{"id": f"{label}-{i}", "gold": label} for label in
                ("SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO") for i in range(3)]
        self.assertEqual(select_rows(rows, 2, 7), select_rows(rows, 2, 7))
        self.assertEqual(len(select_rows(rows, 2, 7)), 6)

    def test_gemini_response_text(self):
        body = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": '{"ok":true}'}]}}]}
        self.assertEqual(response_text(body), '{"ok":true}')
        body["candidates"][0]["finishReason"] = "MAX_TOKENS"
        self.assertEqual(response_text(body), "")

    def test_soclaas_request_and_response(self):
        request = request_for(ROW, "soclaas", "qwen3.5:9b")
        self.assertEqual(request["model"], "qwen3.5:9b")
        self.assertEqual(request["messages"][0]["role"], "system")
        body = {"choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]}
        self.assertEqual(response_text(body, "soclaas"), '{"ok":true}')
        self.assertEqual(soclaas_url("https://soclaas-api.comp.nus.edu.sg/v1", "models"),
                         "https://soclaas-api.comp.nus.edu.sg/v1/models")
        with self.assertRaises(ValueError):
            soclaas_url("https://example.org/v1", "models")


if __name__ == "__main__":
    unittest.main()

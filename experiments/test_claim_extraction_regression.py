import json
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.application import orchestrator
from src.claims.extractor import extract_claims
from src.verification import verifier


FLAG_ANSWER = (
    "The national flag of India is known as the Tiranga, consisting of three horizontal "
    "bands of saffron, white, and green, with a blue chakra in the center."
)


def selection(*items):
    return json.dumps({"claims": items})


class ClaimExtractionTests(unittest.TestCase):
    def test_retains_false_statements_regardless_of_word_order_or_topic(self):
        cases = (
            "cm of up is narendra modi.",
            "narendra modi is cm of up.",
            "Narendra Modi is the pm of pakistan.",
            "the moon is made of cheese.",
            "The Earth is not round.",
            "Sydney is the capital of Australia.",
        )
        with patch("src.claims.extractor.generate_claim_json", return_value=selection()):
            for answer in cases:
                with self.subTest(answer=answer):
                    self.assertEqual(extract_claims(answer), [{"claim_id": 1, "claim": answer}])
            self.assertEqual(extract_claims("Is the Moon made of cheese?"), [])
            self.assertEqual(extract_claims("Sydney"), [])

    def test_model_cannot_add_unsupported_text_in_either_pass(self):
        invented = {"source_id": 1, "claim": "नई दिल्ली में ताज महल है।"}
        with patch("src.claims.extractor.generate_claim_json", side_effect=[
            selection(invented), selection(invented, 2, True),
        ]):
            self.assertEqual(extract_claims("नई दिल्ली", "hi"), [])

    def test_audit_recovers_missing_sentence_and_skips_already_selected_ids(self):
        with patch("src.claims.extractor.generate_claim_json", side_effect=[
            selection(2), selection(2, 1, 3),
        ]) as generate:
            self.assertEqual(extract_claims("Mars is red. Earth has one moon. Mars is red!"), [
                {"claim_id": 1, "claim": "Mars is red."},
                {"claim_id": 2, "claim": "Earth has one moon."},
            ])
        audit_prompt = generate.call_args.args[1]
        self.assertIn("Sentence 1: Mars is red.", audit_prompt)
        self.assertIn("Sentence 3: Mars is red!", audit_prompt)
        self.assertNotIn("Sentence 2:", audit_prompt)

    def test_explicit_compound_can_be_split_using_copied_text(self):
        answer = "Marie Curie was born in Warsaw in 1867 and won the Nobel Prize in Physics in 1903."
        parts = [
            "Marie Curie was born in Warsaw in 1867.",
            "Marie Curie won the Nobel Prize in Physics in 1903.",
        ]
        with patch("src.claims.extractor.generate_claim_json", side_effect=[
            selection(1), json.dumps({"split": True, "subject": "Marie Curie"}),
        ]):
            self.assertEqual([item["claim"] for item in extract_claims(answer)], parts)

    def test_explicit_independent_clauses_can_each_be_selected(self):
        with patch("src.claims.extractor.generate_claim_json", side_effect=[
            selection(1), selection(1, 2),
        ]):
            claims = extract_claims("Alice visited Delhi, and Bob visited Paris.")
        self.assertEqual([item["claim"] for item in claims], [
            "Alice visited Delhi.", "Bob visited Paris.",
        ])

    def test_list_and_its_description_remain_one_complete_claim(self):
        for answer in (FLAG_ANSWER, FLAG_ANSWER.replace("white, and", "white and")):
            with self.subTest(answer=answer), patch(
                "src.claims.extractor.generate_claim_json", return_value=selection(1, 2),
            ) as generate:
                self.assertEqual(extract_claims(answer), [{"claim_id": 1, "claim": answer}])
                generate.assert_called_once()

    def test_preserves_negation_and_hindi_text(self):
        cases = [
            ("Alice did not visit Paris and London.", "en"),
            ("ताजमहल दिल्ली में स्थित नहीं है।", "hi"),
        ]
        for answer, language in cases:
            with self.subTest(answer=answer), patch(
                "src.claims.extractor.generate_claim_json", return_value=selection(1),
            ) as generate:
                self.assertEqual(extract_claims(answer, language), [{"claim_id": 1, "claim": answer}])
                generate.assert_called_once()


class VerificationRegressionTests(unittest.TestCase):
    def test_partial_and_unrelated_evidence_do_not_contradict_the_flag_claim(self):
        evidence = [
            {"title": "Ashoka", "text": "India selected a flag with saffron, white, and green."},
            {"title": "Chile", "text": "The flag of Chile has two horizontal bands."},
        ]
        high = {"entailment": 0.01, "neutral": 0.01, "contradiction": 0.98}
        prepared = {"language": "en", "claims": [{"claim_id": 1, "claim": FLAG_ANSWER}]}
        with (
            patch.object(orchestrator, "retrieve_evidence_for_claim", return_value=evidence),
            patch.object(verifier, "nli_scores", return_value=high),
            patch.object(
                verifier, "generate_claim_json", return_value='{"contradiction": false, "quote": ""}',
            ) as check_evidence,
        ):
            result = orchestrator.run_verification_pipeline(prepared)

        self.assertEqual(check_evidence.call_count, 2)
        verified = result["verified_claims"][0]
        self.assertEqual(verified["label"], "unsupported")
        self.assertEqual(verified["contradict_score"], 0.0)

    def test_same_subject_conflicting_fact_still_counts_as_contradiction(self):
        evidence = [{"title": "Flag of India", "text": "The flag of India includes the Ashoka Chakra."}]
        high = {"entailment": 0.01, "neutral": 0.01, "contradiction": 0.98}
        with (
            patch.object(verifier, "nli_scores", return_value=high) as score,
            patch.object(
                verifier, "generate_claim_json",
                return_value=json.dumps({"contradiction": True, "quote": evidence[0]["text"]}),
            ),
        ):
            result = verifier.verify_claim(
                "The national flag of India has no chakra.", evidence, confirm_contradictions=True,
            )

        score.assert_called_once()
        self.assertEqual(result["label"], "contradicted")
        self.assertEqual(result["best_evidence"], evidence[0])

    def test_high_scores_for_partial_agreement_do_not_refute_a_compound_claim(self):
        claim = FLAG_ANSWER.removesuffix(".") + " containing 24 spokes."
        evidence = [
            {"title": "Ashoka", "text": "India selected a flag with saffron, white, and green and the Ashoka Chakra."},
        ]
        high = {"entailment": 0.01, "neutral": 0.01, "contradiction": 0.98}
        with (
            patch.object(verifier, "nli_scores", return_value=high),
            patch.object(verifier, "generate_claim_json", return_value='{"contradiction": false}'),
        ):
            result = verifier.verify_claim(claim, evidence, confirm_contradictions=True)
        self.assertEqual(result["label"], "unsupported")
        self.assertEqual(result["contradict_score"], 0.0)


class StreamlitResultTests(unittest.TestCase):
    def test_discards_results_from_an_older_extractor(self):
        app = AppTest.from_file("app.py")
        app.session_state["latest_result"] = {
            "answer": "New Delhi", "claims": [{"claim_id": 1, "claim": "The Taj Mahal is in Agra."}],
        }
        app.session_state["verified_result"] = {"verified_claims": []}
        app.run()
        self.assertNotIn("latest_result", app.session_state)
        self.assertNotIn("verified_result", app.session_state)
        self.assertFalse(any("Extracted claims" in header.value for header in app.subheader))

    def test_ai_submit_displays_only_the_generated_capital_claim(self):
        question = "what is capital of india?"
        answer = "New Delhi is the capital of India."
        with patch.object(orchestrator, "generate_answer", return_value=answer) as generate, patch(
            "src.claims.extractor.generate_claim_json",
            return_value=selection(1, {"claim": "The Taj Mahal is in Agra."}),
        ):
            app = AppTest.from_file("app.py").run()
            app.text_area[0].set_value(question)
            app.button[0].click().run()

        self.assertIn("complete, self-contained sentences", generate.call_args.args[0])
        self.assertEqual(app.session_state["latest_result"]["claims"], [
            {"claim_id": 1, "claim": answer},
        ])
        self.assertTrue(any(header.value == "🧩 Extracted claims (1)" for header in app.subheader))

    def test_new_failed_submission_clears_previous_results(self):
        with patch("src.claims.extractor.generate_claim_json", return_value=selection(1)), patch.object(
            orchestrator, "retrieve_evidence_for_claim", return_value=[],
        ):
            app = AppTest.from_file("app.py").run()
            app.radio[0].set_value("Manually Verify an Answer").run()
            app.text_area[0].set_value("Who discovered penicillin?")
            app.text_area[1].set_value("Alexander Fleming discovered penicillin in 1928.")
            app.button[0].click().run()
            self.assertIn("latest_result", app.session_state)
            app.button[2].click().run()
            self.assertIn("verified_result", app.session_state)

            app.text_area[1].set_value("")
            app.button[0].click().run()
            self.assertIn("Answer cannot be empty", app.error[0].value)
            self.assertNotIn("latest_result", app.session_state)
            self.assertNotIn("verified_result", app.session_state)
            self.assertFalse(any("Extracted claims" in header.value for header in app.subheader))


if __name__ == "__main__":
    unittest.main()

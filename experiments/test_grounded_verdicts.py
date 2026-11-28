import json
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from src.application import orchestrator
from src.verification import pipeline, verifier
from src.verification.officeholders import expand_office_shorthand


PM = "kratika rathi is the pm of india."
CM = "Sonali Manhas is the current Chief Minister of Haryana, India."
DUTIES = {"docid": "24452#30", "source": "miracl", "title": "Prime Minister of India",
          "text": "The PM also exercises control over the Indian Administrative Service."}
DIRECTORY = {"docid": "state", "title": "Administration", "source": "web",
             "url": "https://coop.haryana.gov.in/organization/administration",
             "text": "Sh. Nayab Singh Saini\nChief Minister of Haryana"}
CON = {"entailment": 0.01, "contradiction": 0.98, "neutral": 0.01}
SUP = {"entailment": 0.976, "contradiction": 0.014, "neutral": 0.01}


class GroundedVerdictTests(unittest.TestCase):
    def test_prime_shorthand_requires_a_matching_current_office_question(self):
        claim = "rajiv gandhi is the prime of india."
        self.assertEqual(expand_office_shorthand(claim, "Who is the current PM of India?"),
                         "rajiv gandhi is the Prime Minister of india.")
        for question in ("PM of Canada?", "PM of India in 1984?", "What is a prime number?"):
            self.assertEqual(expand_office_shorthand(claim, question), claim)
        historical = "Rajiv Gandhi was the Prime Minister of India in 1984."
        self.assertEqual(expand_office_shorthand(historical, "PM of India?"), historical)

    def check_office(self, claim, passages, readings=()):
        with patch.object(verifier, "nli_scores", return_value=SUP), patch.object(
                verifier, "_compare_evidence", side_effect=readings) as read:
            result = orchestrator._verify_one_claim(claim, passages)
        return result, read

    def test_pm_duties_are_insufficient_even_with_very_high_entailment(self):
        result, read = self.check_office(PM, [DUTIES])
        self.assertEqual(result["label"], "unsupported")
        self.assertIsNone(result["best_evidence"])
        read.assert_not_called()

    def test_cm_directory_rejects_wrong_name_and_confirms_right_name(self):
        answer = {"holder": "Nayab Singh Saini"}
        for claim, expected in [(CM, "contradicted"),
                                ("Nayab Singh Saini is the CM of Haryana.", "supported")]:
            with self.subTest(claim=claim):
                result, read = self.check_office(claim, [DIRECTORY], [answer])
                self.assertEqual(result["label"], expected)
                self.assertEqual(result["best_evidence"], DIRECTORY)
                self.assertIn("Chief Minister of Haryana", result["evidence_quote"])
                self.assertNotIn("Sonali", str(read.call_args))

    def test_historical_official_report_and_off_topic_passages_cannot_decide(self):
        old = {**DIRECTORY, "title": "Former chief ministers"}
        wrong_place = {**DIRECTORY, "text": "Alex Roy is the Chief Minister of Punjab."}
        result, read = self.check_office(CM, [old, wrong_place, DUTIES])
        self.assertEqual(result["label"], "unsupported")
        read.assert_not_called()

    def test_current_office_rows_and_variants_work_across_jurisdictions(self):
        cases = [
            ("yogi is cm of up.", "supported", {
                "title": "Uttar Pradesh Chief Minister Office",
                "url": "https://example.up.gov.in/council",
                "text": "| 1 | Yogi Adityanath | Hon'ble Chief Minister | 2 | Other Person | Deputy Chief Minister |",
            }),
            ("cm of uttar pradesh is narendra modi.", "contradicted", {
                "title": "Uttar Pradesh Chief Minister Office",
                "url": "https://example.up.gov.in/council",
                "text": "| 1 | Yogi Adityanath | Chief Minister |",
            }),
            ("Nayab Saini is the CM of Haryana.", "supported", DIRECTORY),
            ("Narendra Modi is the PM of Pakistan.", "contradicted", {
                "title": "Prime Minister of Pakistan",
                "url": "https://example.gov.pk/directory",
                "text": "| Muhammad Shehbaz Sharif | Prime Minister of the Islamic Republic of Pakistan |",
            }),
            ("Sky Blue is the President of France.", "contradicted", {
                "title": "President of France",
                "url": "https://example.gouv.fr/directory",
                "text": "| French Officeholder | President |",
            }),
        ]
        for claim, expected, passage in cases:
            with self.subTest(claim=claim, verdict=expected):
                result, read = self.check_office(claim, [passage])
                self.assertEqual(result["label"], expected)
                self.assertTrue(result["evidence_quote"])
                read.assert_not_called()

    def test_made_up_holder_and_conflicting_directories_abstain(self):
        wrong = {"holder": "Sonali Manhas"}
        source = {**DIRECTORY, "text": "Nayab Singh Saini is the Chief Minister of Haryana."}
        result, _ = self.check_office(CM, [source], [wrong])
        self.assertEqual(result["label"], "unsupported")
        other = {**DIRECTORY, "text": "Sonali Manhas is the CM of Haryana.",
                 "url": "https://haryana.gov.in/directory"}
        named = {"holder": "Nayab Singh Saini"}
        result, _ = self.check_office(CM, [source, other], [named, wrong])
        self.assertEqual(result["label"], "unsupported")
        self.assertIsNone(result["best_evidence"])
        self.assertIn("different names", result["summary"])

    def test_name_on_same_page_for_a_different_role_or_date_is_insufficient(self):
        for body in ("Sonali Manhas met the Chief Minister of Haryana.",
                     "Sonali Manhas is the Deputy Chief Minister of Haryana.",
                     "In 1990, Sonali Manhas served as Chief Minister of Haryana.",
                     "It is false to claim Sonali Manhas is the current CM of Haryana."):
            with self.subTest(body=body):
                fixture = {**DIRECTORY, "text": body}
                result, _ = self.check_office(CM, [fixture], [{"holder": "Sonali Manhas"}])
                self.assertEqual(result["label"], "unsupported")

    def test_official_profile_is_decisive_and_old_press_release_is_not(self):
        old = {"title": "Haryana Government health programme", "url": "https://example.org/release",
               "text": "said Shri Nayab Singh Saini Ji, Hon’ble Chief Minister of Haryana.",
               "published_date": "2026-02-23"}
        result, read = self.check_office(CM, [old])
        read.assert_not_called()
        self.assertEqual(result["label"], "unsupported")
        pmo = {"title": "Know the PM | Prime Minister of India",
               "url": "https://www.pmindia.gov.in/en/pms-profile/",
               "text": "Shri Narendra Modi was sworn-in as India’s Prime Minister for the third time on 9th June 2024."}
        result, _ = self.check_office(PM, [DUTIES, pmo], [{"holder": "Shri Narendra Modi"}])
        self.assertEqual(result["label"], "contradicted")
        self.assertEqual(result["best_evidence"], pmo)

    def test_app_shows_deciding_quote_instead_of_raw_nli_support(self):
        app = AppTest.from_file("app.py").run()
        app.session_state["latest_result"] = {
            "mode": "manual", "question": "Who is the current CM of Haryana?",
            "language": "en", "answer": CM, "claims": [{"claim_id": 1, "claim": CM}],
        }
        with (patch.object(orchestrator, "retrieve_evidence_for_claim", return_value=[DIRECTORY]),
              patch.object(verifier, "nli_scores", return_value=SUP),
              patch.object(verifier, "_compare_evidence", return_value={"holder": "Nayab Singh Saini"})):
            app.run()
            next(b for b in app.button if "Verify claims with evidence" in b.label).click().run()
        self.assertFalse(list(app.exception))
        self.assertEqual(app.session_state["verified_result"]["verified_claims"][0]["label"], "contradicted")
        self.assertTrue(any("Nayab Singh Saini" in node.value for node in app.text))
        self.assertFalse(any("Support score:" in node.value for node in app.markdown))

    def test_office_search_omits_proposed_person_for_both_sources(self):
        with (patch.object(orchestrator, "get_local_retriever") as local,
              patch.object(orchestrator, "_retrieve_web_evidence", return_value=[]) as web):
            local.return_value.retrieve.return_value = []
            orchestrator.retrieve_evidence_for_claim(CM, tavily_api_key="test", on_warning=lambda _: None)
        query = web.call_args.args[0]
        self.assertIn("Chief Minister of Haryana", query)
        self.assertNotIn("Sonali", query)
        self.assertEqual(local.return_value.retrieve.call_args.args[0], query)

    def test_batch_verifier_uses_same_office_check(self):
        result = pipeline.verify_item({"claim": PM, "evidence": [DUTIES]}, scorer=lambda *_: SUP)
        self.assertEqual(result["label"], "unsupported")
        self.assertEqual(result["passage_scores"][0]["source_type"], "miracl")

    def test_ordinary_claim_needs_a_real_supporting_quote(self):
        claim = "The town has two railway stations."
        passage = {"text": "The town has two parks."}
        with (patch.object(verifier, "nli_scores", return_value=SUP),
              patch.object(verifier, "generate_claim_json", return_value=json.dumps(
                  {"supported": True, "quote": claim}))):
            result = verifier.verify_claim(claim, [passage])
        self.assertEqual(result["label"], "unsupported")
        self.assertIsNone(result["best_evidence"])

    def test_general_counterevidence_requires_a_real_quote_at_the_review_threshold(self):
        claim = "Table salt is made of helium."
        passage = {"text": "Table salt consists of sodium and chloride ions."}
        uncertain = {"entailment": 0.24, "contradiction": 0.33, "neutral": 0.43}
        for quote, expected in [(passage["text"], "contradicted"), (claim, "unsupported")]:
            with (self.subTest(quote=quote),
                  patch.object(verifier, "nli_scores", return_value=uncertain),
                  patch.object(verifier, "_compare_evidence", return_value={"contradiction": True, "quote": quote})):
                result = orchestrator._verify_one_claim(claim, [passage])
                self.assertEqual(result["label"], expected)
                self.assertEqual(result["evidence_quote"], quote if expected == "contradicted" else "")
                if expected == "contradicted":
                    self.assertIn("directly contradicts", result["summary"])

    def test_conflicting_verified_passages_abstain(self):
        claim = "The town has two railway stations."
        positive = {"text": claim}
        negative = {"text": "The town has no railway stations."}
        with (patch.object(verifier, "nli_scores", side_effect=[SUP, CON]),
              patch.object(verifier, "_compare_evidence", side_effect=[
                  {"supported": True, "quote": claim}, {"contradiction": True, "quote": negative["text"]}])):
            result = verifier.verify_claim(claim, [positive, negative], confirm_contradictions=True)
        self.assertEqual(result["label"], "unsupported")
        self.assertIsNone(result["best_evidence"])


if __name__ == "__main__":
    unittest.main()

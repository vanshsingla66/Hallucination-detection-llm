import json
import os
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest

from src.application import orchestrator
from src.verification import officeholders, verifier


TEST_KEY = "test-key-for-mocked-http"
CLAIMS = ["Mars has two moons.", "Earth has one moon."]
WEB_URL = "https://example.org/moons"


def web_response(session, request, **kwargs):
    if request.url == officeholders.INDIA_PM_PROFILE:
        response = requests.Response()
        response.status_code = 200
        response._content = (
            b"<p>Rajiv Gandhi became Prime Minister of India in 1984.</p>"
            b"<p><img alt='PM India'>Shri Narendra Modi was sworn-in as "
            b"India&rsquo;s Prime Minister on 9th June 2024.</p>"
        )
        response.url = request.url
        return response
    query = json.loads(request.body)["query"]
    result = {"title": "Moons", "content": f"Web evidence for {query}", "url": WEB_URL}
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps({"results": [result]}).encode()
    response.url = request.url
    return response


class TavilyRetrievalTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict(os.environ, {"TAVILY_API_KEY": ""}))
        self.local = stack.enter_context(patch.object(orchestrator, "get_local_retriever"))
        self.local.return_value.retrieve.return_value = [
            {"docid": "1", "text": "Local passage on moons."},
        ]
        stack.enter_context(patch.object(orchestrator, "_get_reranker", return_value=None))
        self.http = stack.enter_context(patch.object(
            requests.Session, "send", autospec=True, side_effect=web_response,
        ))
        stack.enter_context(patch.object(verifier, "nli_scores", return_value={
            "entailment": 0.8, "neutral": 0.1, "contradiction": 0.1,
        }))
        stack.enter_context(patch.object(verifier, "_support_quote", return_value=""))

    def test_missing_key_uses_only_local_evidence(self):
        passages = orchestrator.retrieve_evidence_for_claim(CLAIMS[0])
        self.assertEqual([p["source"] for p in passages], ["miracl"])
        self.http.assert_not_called()

    def test_indian_office_request_prefers_official_sources_and_omits_guessed_name(self):
        orchestrator.retrieve_evidence_for_claim("Kratika Rathi is the PM of India.", tavily_api_key=TEST_KEY)
        official, tavily = [call.args[1] for call in self.http.call_args_list]
        self.assertEqual(official.url, officeholders.INDIA_PM_PROFILE)
        self.assertNotIn("Authorization", official.headers)
        request = json.loads(tavily.body)
        self.assertNotIn("Kratika", request["query"])
        self.assertIn("Prime Minister", request["query"])
        self.assertEqual(request["include_domains"], ["gov.in", "nic.in"])
        self.assertEqual(request["include_domains_mode"], "prefer")

    def test_pmo_profile_resolves_both_screenshot_claims(self):
        claims = [
            ("kratika rathi is the prime minister of india.", "contradicted"),
            ("rajiv gandhi is the prime of india.", "contradicted"),
            ("Narendra Modi is the PM of India.", "supported"),
        ]
        for claim, expected in claims:
            with self.subTest(claim=claim), patch.object(verifier, "_compare_evidence") as compare:
                prepared = {"question": "pm of india?", "claims": [{"claim_id": 1, "claim": claim}]}
                verified = orchestrator.run_verification_pipeline(
                    prepared, tavily_api_key=TEST_KEY,
                )["verified_claims"][0]
                self.assertEqual(verified["label"], expected)
                self.assertEqual(verified["claim"], claim)
                self.assertEqual(verified["best_evidence"]["source"], "official")
                self.assertIn("Narendra Modi", verified["evidence_quote"])
                compare.assert_not_called()

    def test_current_pm_lookup_keeps_no_key_mode_offline(self):
        passages = orchestrator.retrieve_evidence_for_claim("Rajiv Gandhi is the PM of India.", tavily_api_key="")
        self.assertEqual([p["source"] for p in passages], ["miracl"])
        self.http.assert_not_called()

    def test_official_lookup_shares_web_budget_and_deduplicates_the_same_url(self):
        self.local.return_value.retrieve.return_value = [
            {"docid": f"local_{i}", "text": f"Local passage {i}."} for i in range(5)
        ]
        duplicate = {"source": "web", "url": officeholders.INDIA_PM_PROFILE.rstrip("/"), "text": "Repeat"}
        web = [duplicate] + [
            {"source": "web", "url": f"https://example.org/{i}", "text": f"Web passage {i}."}
            for i in range(4)
        ]
        with patch.object(orchestrator, "_retrieve_web_evidence", return_value=web):
            passages = orchestrator.retrieve_evidence_for_claim(
                "Narendra Modi is the PM of India.", tavily_api_key=TEST_KEY,
            )
        self.assertEqual(len(passages), 10)
        self.assertEqual([p["source"] for p in passages].count("official"), 1)
        self.assertNotIn(duplicate, passages)

    def test_relevant_question_searches_without_the_guessed_answer(self):
        prepared = {"question": "What is the capital of Australia?", "claims": [
            {"claim_id": 1, "claim": "Sydney is the capital of Australia."},
            {"claim_id": 2, "claim": "Mars has two moons."},
        ]}
        orchestrator.run_verification_pipeline(prepared, tavily_api_key=TEST_KEY)
        queries = [json.loads(call.args[1].body)["query"] for call in self.http.call_args_list]
        self.assertEqual(queries, [prepared["question"], "Mars has two moons."])

    def test_profile_failure_still_searches_tavily_and_does_not_accept_former_pm(self):
        def unavailable_profile(session, request, **kwargs):
            if request.url == officeholders.INDIA_PM_PROFILE:
                raise requests.Timeout()
            response = requests.Response()
            response.status_code = 200
            response._content = json.dumps({"results": [{
                "title": "Rajiv Gandhi",
                "url": "https://example.org/rajiv-gandhi",
                "content": "Rajiv Gandhi became prime minister of India in 1984.",
            }]}).encode()
            return response

        self.http.side_effect = unavailable_profile
        prepared = {"question": "pm of india?", "claims": [
            {"claim_id": 1, "claim": "rajiv gandhi is the prime of india."},
        ]}
        verified = orchestrator.run_verification_pipeline(prepared, tavily_api_key=TEST_KEY)["verified_claims"][0]
        self.assertEqual(self.http.call_count, 2)
        self.assertEqual(verified["label"], "unsupported")
        self.assertEqual(verified["verification_claim"], "rajiv gandhi is the Prime Minister of india.")
        self.assertEqual([p["source"] for p in verified["evidence"]], ["miracl", "web"])

    def test_configured_key_searches_both_sources_for_each_claim_and_verifies_both(self):
        os.environ["TAVILY_API_KEY"] = TEST_KEY
        prepared = {"language": "en", "claims": [
            {"claim_id": n, "claim": claim} for n, claim in enumerate(CLAIMS, start=1)
        ]}
        result = orchestrator.run_verification_pipeline(prepared)

        self.assertEqual(self.local.return_value.retrieve.call_count, len(CLAIMS))
        sent = [call.args[1] for call in self.http.call_args_list]
        self.assertEqual([json.loads(r.body)["query"] for r in sent], CLAIMS)
        for request in sent:
            self.assertEqual(request.url, "https://api.tavily.com/search")
            self.assertEqual(request.headers["Authorization"], f"Bearer {TEST_KEY}")
            self.assertNotIn("api_key", json.loads(request.body))
        for call in self.http.call_args_list:
            self.assertTrue(call.kwargs["verify"])
        for verified in result["verified_claims"]:
            self.assertEqual({p["source"] for p in verified["passage_scores"]}, {"miracl", "web"})
            web = next(p for p in verified["evidence"] if p["source"] == "web")
            self.assertEqual(web["text"], f"Web evidence for {verified['claim']}")
            self.assertEqual(web["url"], WEB_URL)

    def test_local_index_failure_still_allows_web_results(self):
        os.environ["TAVILY_API_KEY"] = TEST_KEY
        self.local.side_effect = OSError("Local index unavailable")
        passages = orchestrator.retrieve_evidence_for_claim(CLAIMS[0])
        self.assertEqual([p["source"] for p in passages], ["web"])
        self.http.assert_called_once()

    def test_rejected_web_key_keeps_local_passages(self):
        os.environ["TAVILY_API_KEY"] = TEST_KEY
        self.http.side_effect = None
        self.http.return_value = requests.Response()
        self.http.return_value.status_code = 401
        prepared = {"claims": [{"claim_id": 1, "claim": CLAIMS[0]}]}
        verified = orchestrator.run_verification_pipeline(prepared)["verified_claims"][0]
        self.http.assert_called_once()
        self.assertEqual([p["source"] for p in verified["evidence"]], ["miracl"])
        self.assertIn("401", verified["retrieval_warnings"][0])
        self.assertNotIn(TEST_KEY, str(verified))

    def test_empty_web_response_is_reported_and_keeps_local_evidence(self):
        self.http.side_effect = None
        self.http.return_value = requests.Response()
        self.http.return_value.status_code = 200
        self.http.return_value._content = b'{"results":[]}'
        warnings = []
        passages = orchestrator.retrieve_evidence_for_claim(
            CLAIMS[0], tavily_api_key=TEST_KEY, on_warning=warnings.append,
        )
        self.http.assert_called_once()
        self.assertEqual([p["source"] for p in passages], ["miracl"])
        self.assertIn("returned no passages", warnings[0])

    def test_https_failure_keeps_local_evidence_and_reports_the_cause(self):
        self.http.side_effect = requests.exceptions.SSLError("Certificate verification failed")
        warnings = []
        passages = orchestrator.retrieve_evidence_for_claim(
            CLAIMS[0], tavily_api_key=TEST_KEY, on_warning=warnings.append,
        )
        self.assertEqual([p["source"] for p in passages], ["miracl"])
        self.assertIn("certificate", warnings[0])

    def test_marathi_uses_its_local_retriever_with_web(self):
        passages = orchestrator.retrieve_evidence_for_claim(
            "मंगळाला दोन चंद्र आहेत.", language="mr", tavily_api_key=TEST_KEY,
        )
        self.local.return_value.retrieve.assert_called_once_with(
            "मंगळाला दोन चंद्र आहेत.", final_top_k=5,
        )
        self.assertEqual([p["source"] for p in passages], ["vaarta", "web"])

    def test_sidebar_key_can_be_added_and_cleared_for_the_same_answer(self):
        with patch("src.claims.extractor.generate_claim_json", return_value='{"claims":[1]}'):
            app = AppTest.from_file("app.py").run()
            app.radio[0].set_value("Manually Verify an Answer").run()
            app.text_area[0].set_value("How many moons does Mars have?")
            app.text_area[1].set_value(CLAIMS[0])
            app.button[0].click().run()

        for key, expected_sources in (
            ("", {"miracl"}), (TEST_KEY, {"miracl", "web"}), ("", {"miracl"}),
        ):
            with self.subTest(web_enabled=bool(key)):
                app.sidebar.text_input[0].set_value(key).run()
                next(b for b in app.button if "Verify claims with evidence" in b.label).click().run()
                self.assertFalse(list(app.exception))
                verified = app.session_state["verified_result"]["verified_claims"][0]
                self.assertEqual({p["source"] for p in verified["evidence"]}, expected_sources)
                self.assertFalse(os.environ["TAVILY_API_KEY"])
                if key:
                    self.assertTrue(any(WEB_URL in node.value for node in app.markdown))
        self.http.assert_called_once()


if __name__ == "__main__":
    unittest.main()

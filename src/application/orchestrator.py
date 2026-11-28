"""Shared entry points for AI-generated and user-provided answers.

Both modes (AI-generated and user-provided) run through the same downstream
pipeline:
  1. Claim extraction  (LLM-based, 2-pass)
  2. Evidence retrieval  (local corpus plus Tavily when a key is configured)
  3. NLI verification  (mDeBERTa multilingual)
  4. Confidence scoring and explanation generation

Public API
----------
prepare_ai_answer(question, language)     -> dict  (stage: "claims_extracted")
prepare_manual_answer(question, answer, language) -> dict
run_verification_pipeline(result)         -> dict  (stage: "complete")
run_full_pipeline(question, language, answer=None) -> dict
"""
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.claims.extractor import extract_claims
from src.explanation.confidence import calculate_confidence, generate_explanation
from src.generation.llm import generate_answer, MODEL_NAME, LANGUAGE_NAMES
from src.retrieval.official_profiles import retrieve_india_pm_profile
from src.verification.officeholders import current_office_claim, expand_office_shorthand, is_current_office_source
from src.verification.verifier import verify_claim as _nli_verify_claim

SUPPORTED_LANGUAGES = LANGUAGE_NAMES

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

# Cached singleton retriever and reranker instances
_LOCAL_RETRIEVERS: dict = {}
_RERANKER_INSTANCE = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} cannot be empty.")
    return value.strip()


def get_local_retriever(language: str):
    """Obtain or initialize the local dataset retriever for the specified language.

    Supported languages:
      - 'en': MIRACLEnglishShardRetriever
      - 'hi': MIRACLShardRetriever(language='hi')
      - 'bn': MIRACLBengaliShardRetriever
      - 'mr': VaartaMarathiRetriever (if corpus files exist)
    """
    language = (language or "en").lower().strip()
    if language in _LOCAL_RETRIEVERS:
        return _LOCAL_RETRIEVERS[language]

    retriever = None
    if language == "en":
        from src.retrieval.miracl_english_shard_retriever import MIRACLEnglishShardRetriever
        print("[orchestrator] Initializing local MIRACL English retriever...")
        retriever = MIRACLEnglishShardRetriever()
    elif language == "hi":
        from src.retrieval.miracl_hindi_shard_retriever import MIRACLShardRetriever
        print("[orchestrator] Initializing local MIRACL Hindi retriever...")
        retriever = MIRACLShardRetriever(language="hi")
    elif language == "bn":
        from src.retrieval.miracl_bengali_shard_retriever import MIRACLBengaliShardRetriever
        print("[orchestrator] Initializing local MIRACL Bengali retriever...")
        retriever = MIRACLBengaliShardRetriever()
    elif language == "mr":
        from src.retrieval.vaarta_marathi_retriever import VaartaMarathiRetriever
        print("[orchestrator] Initializing local Vaarta Marathi retriever...")
        retriever = VaartaMarathiRetriever()
    else:
        print(f"[orchestrator] No local corpus retriever configured for language '{language}'.")

    if retriever is not None:
        _LOCAL_RETRIEVERS[language] = retriever
    return retriever


def _get_reranker():
    global _RERANKER_INSTANCE
    if _RERANKER_INSTANCE is None:
        try:
            from src.retrieval.reranker import EvidenceReranker
            _RERANKER_INSTANCE = EvidenceReranker()
        except Exception as exc:
            print(f"[orchestrator] Warning: EvidenceReranker unavailable: {exc}")
            _RERANKER_INSTANCE = False
    return _RERANKER_INSTANCE if _RERANKER_INSTANCE is not False else None


def _retrieve_web_evidence(query: str, api_key: str, top_k: int, *, india_office=False) -> list[dict]:
    """Look up this claim with Tavily and keep the source links for verification."""
    payload = {"query": query, "search_depth": "basic", "max_results": min(top_k, 20)}
    if india_office:
        payload.update(include_domains=["gov.in", "nic.in"], include_domains_mode="prefer",
                       include_published_date=True)
    response = requests.post(
        "https://api.tavily.com/search",
        json=payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()
    results = response.json()["results"]

    passages = []
    for rank, r in enumerate(results, start=1):
        text = (r.get("content") or "").strip()
        url = r.get("url", "")
        title = r.get("title", "")
        if not text or not url:
            continue
        passages.append({
            "docid": f"web_{rank}",
            "title": title,
            "text": text,
            "url": url,
            "published_date": r.get("published_date"),
            "source": "web",
            "rank": rank,
            "relevance_score": float(r.get("score") or 0.0),
            "rerank_score": None,
        })
    return passages


def _rerank_evidence(claim: str, passages: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank passages by semantic similarity to the claim."""
    if not passages:
        return passages
    reranker = _get_reranker()
    if reranker is not None:
        try:
            return reranker.rerank(claim, passages, top_k=top_k)
        except Exception as exc:
            print(f"[orchestrator] Reranking failed, keeping source order: {exc}")
    return passages[:top_k]


def retrieve_evidence_for_claim(
    claim: str,
    language: str = "en",
    top_k: int = 5,
    *,
    question: str = "",
    tavily_api_key: str | None = None,
    on_warning=print,
) -> list[dict]:
    """Return up to top_k local and top_k web passages, including official lookups."""
    if top_k <= 0:
        return []
    language = (language or "en").lower().strip()
    if tavily_api_key is None:
        tavily_api_key = os.environ.get("TAVILY_API_KEY", "")
    tavily_api_key = tavily_api_key.strip()
    office = current_office_claim(claim)
    query = office.query if office else claim
    if not office and question.strip().endswith("?") and len(question) <= 240:
        question_words = set(re.findall(r"\b\w{3,}\b", question.casefold()))
        claim_words = set(re.findall(r"\b\w{3,}\b", claim.casefold()))
        shared = (question_words & claim_words) - {"the", "and", "was", "were", "has", "does", "did"}
        if len(shared) >= 2:
            query = question.strip()
    local_passages = []
    try:
        retriever = get_local_retriever(language)
        if retriever is not None:
            if language == "mr":
                # Vaarta already reranks its own candidates.
                local_passages = retriever.retrieve(query, final_top_k=top_k)
            else:
                candidate_k = 50 if language == "en" else 200
                candidates = retriever.retrieve(query, top_k=candidate_k)
                local_passages = _rerank_evidence(query, candidates, top_k=top_k)
            for passage in local_passages:
                if not passage.get("source"):
                    passage["source"] = "vaarta" if language == "mr" else "miracl"
    except Exception as exc:
        on_warning(f"Local {language.upper()} retrieval failed: {exc}")

    if not tavily_api_key:
        return local_passages[:top_k]

    # Search still runs for each claim. The current PM's own profile supplies
    # direct evidence when search snippets omit the incumbent or show past PMs.
    official_passages = []
    if office and office.role == "Prime Minister" and office.place.casefold() == "india":
        try:
            profile = retrieve_india_pm_profile()
        except requests.RequestException:
            profile = None
        if profile:
            official_passages.append(profile)
        else:
            on_warning("Could not read the incumbent from the official PMO profile.")

    try:
        india_office = office is not None and (
            office.place.casefold() in {"india", "uttar pradesh", "haryana"}
            or bool(re.search(r",\s*India\b", claim, re.I))
        )
        candidate_count = min(top_k * 2, 20) if office else top_k
        web_passages = _retrieve_web_evidence(query, tavily_api_key, candidate_count, india_office=india_office)
    except requests.HTTPError as exc:
        status = exc.response.status_code
        message = {
            401: "Check the API key.",
            403: "Check the key's access permissions.",
            429: "Too many requests. Try again later.",
            432: "Check your Tavily usage limit.",
            433: "Check your Tavily spending limit.",
        }.get(status, "Try again later.")
        on_warning(f"Tavily search failed (HTTP {status}). {message}")
    except requests.exceptions.SSLError:
        on_warning("Tavily search failed because the server's HTTPS certificate could not be verified.")
    except Exception:
        # Request errors can include the Authorization header. Keep credentials out of the UI.
        on_warning("Tavily search failed. Check the connection and try again.")
    else:
        if web_passages:
            ranked_web = _rerank_evidence(query, web_passages, top_k=len(web_passages))
            if office:
                ranked_web.sort(key=lambda p: is_current_office_source(p, office), reverse=True)
            web_evidence = official_passages[:]
            seen_urls = {p["url"].split("#", 1)[0].rstrip("/") for p in official_passages}
            for passage in ranked_web:
                if len(web_evidence) >= top_k:
                    break
                url = passage["url"].split("#", 1)[0].rstrip("/")
                if url not in seen_urls:
                    seen_urls.add(url)
                    web_evidence.append(passage)
            return local_passages[:top_k] + web_evidence
        on_warning("Tavily searched the web and returned no passages for this claim.")
    return local_passages[:top_k] + official_passages


def _verify_one_claim(claim_text: str, evidence: list[dict]) -> dict:
    """Run NLI on one claim and add confidence/explanation fields."""
    nli_result = _nli_verify_claim(claim_text, evidence, confirm_contradictions=True)

    confidence = calculate_confidence(
        nli_result.get("support_score"),
        nli_result.get("contradict_score"),
        decision_method=nli_result.get("decision_method", "nli"), label=nli_result["label"],
    )

    passage_scores = nli_result.get("passage_scores", [])

    explanation = generate_explanation(nli_result)

    return {
        "label": nli_result["label"],
        "support_score": nli_result.get("support_score", 0.0),
        "contradict_score": nli_result.get("contradict_score", 0.0),
        "best_evidence": nli_result.get("best_evidence"),
        "decision_method": nli_result.get("decision_method", "nli"),
        "evidence_quote": nli_result.get("evidence_quote", ""),
        "num_passages": nli_result.get("num_passages", 0),
        "passage_scores": passage_scores,
        "strongest_supporting_passage": explanation.get("strongest_supporting_passage"),
        "strongest_contradicting_passage": explanation.get("strongest_contradicting_passage"),
        "confidence_status": confidence["status"],
        "confidence_interpretation": confidence["interpretation"],
        "summary": explanation.get("summary", ""),
    }


# ---------------------------------------------------------------------------
# Step 1: Extract claims (Milestone 1 — unchanged behaviour)
# ---------------------------------------------------------------------------

def prepare_ai_answer(question: str, language: str = "en") -> dict:
    """Generate an AI answer and extract factual claims from it."""
    question = _validate_text(question, "Question")
    language = (language or "en").lower().strip()
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")

    prompt = (
        "Answer the user's question directly and concisely. "
        "Use complete, self-contained sentences that state the relevant subject and relationship. "
        "Express even a short answer as a sentence rather than a bare name, date, or number. "
        "Do not add unrelated details. Do not guess when uncertain.\n\n"
        f"Question:\n{question}\n\nAnswer:"
    )
    answer = generate_answer(prompt, max_new_tokens=300, language=language)
    claims = extract_claims(answer, language=language)
    return {
        "mode": "ai", "question": question, "answer": answer,
        "language": language, "model": MODEL_NAME,
        "claims": claims, "stage": "claims_extracted",
    }


def prepare_manual_answer(question: str, answer: str, language: str = "en") -> dict:
    """Accept a user-provided answer and extract factual claims from it."""
    question = _validate_text(question, "Question")
    answer = _validate_text(answer, "Answer")
    language = (language or "en").lower().strip()
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")

    claims = extract_claims(answer, language=language)
    return {
        "mode": "manual", "question": question, "answer": answer,
        "language": language, "model": None, "claims": claims,
        "stage": "claims_extracted",
    }


# ---------------------------------------------------------------------------
# Step 2+: Evidence retrieval + verification + explanation
# ---------------------------------------------------------------------------

def run_verification_pipeline(
    result: dict,
    progress_callback=None,
    *,
    tavily_api_key: str | None = None,
) -> dict:
    """Take a 'claims_extracted' result dict and run the full downstream pipeline.

    Parameters
    ----------
    result : dict
        Output of prepare_ai_answer() or prepare_manual_answer().
    progress_callback : callable, optional
        Called before each claim and once at completion.
    tavily_api_key : str, optional
        Session key for web search alongside the local corpus. An empty string
        selects local retrieval only. Omit the key to use TAVILY_API_KEY.

    Returns
    -------
    dict
        The same dict with 'verified_claims' added and 'stage' set to
        'complete'.  The original 'claims' list is preserved unchanged.
    """
    claims = result.get("claims", [])
    language = (result.get("language") or "en").lower().strip()
    verified = []

    for i, claim_item in enumerate(claims):
        claim_text = claim_item["claim"]
        verification_claim = expand_office_shorthand(claim_text, result.get("question") or "")

        if progress_callback:
            progress_callback(i, len(claims), claim_text)

        retrieval_warnings = []
        passages = retrieve_evidence_for_claim(
            verification_claim, language=language, tavily_api_key=tavily_api_key,
            on_warning=retrieval_warnings.append, question=result.get("question") or "",
        )

        # NLI verification + confidence/explanation.
        verification = _verify_one_claim(verification_claim, passages)

        verified.append({
            "claim_id": claim_item["claim_id"],
            "claim": claim_text,
            "verification_claim": verification_claim,
            "evidence": passages,
            "retrieval_warnings": retrieval_warnings,
            **verification,
        })

    if progress_callback:
        progress_callback(len(claims), len(claims), "")

    return {
        **result,
        "verified_claims": verified,
        "stage": "complete",
    }


# ---------------------------------------------------------------------------
# Convenience: run everything in one call
# ---------------------------------------------------------------------------

def run_full_pipeline(
    question: str,
    language: str = "en",
    answer: str | None = None,
    progress_callback=None,
    *,
    tavily_api_key: str | None = None,
) -> dict:
    """End-to-end pipeline.

    If *answer* is None, generates an AI answer first (Ask AI & Verify mode).
    If *answer* is provided, uses it directly (Manual Verification mode).
    """
    if answer:
        result = prepare_manual_answer(question, answer, language)
    else:
        result = prepare_ai_answer(question, language)

    return run_verification_pipeline(
        result, progress_callback=progress_callback, tavily_api_key=tavily_api_key,
    )

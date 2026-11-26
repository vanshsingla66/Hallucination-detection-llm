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
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.claims.extractor import extract_claims
from src.explanation.confidence import calculate_confidence, generate_explanation
from src.generation.llm import generate_answer, MODEL_NAME, LANGUAGE_NAMES
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


def _retrieve_web_evidence(claim: str, api_key: str, top_k: int) -> list[dict]:
    """Look up this claim with Tavily and keep the source links for verification."""
    response = requests.post(
        "https://api.tavily.com/search",
        json={"query": claim, "search_depth": "basic", "max_results": min(top_k, 20)},
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
    tavily_api_key: str | None = None,
    on_warning=print,
) -> list[dict]:
    """Return up to top_k passages per source. An explicit empty key selects local only."""
    if top_k <= 0:
        return []
    language = (language or "en").lower().strip()
    if tavily_api_key is None:
        tavily_api_key = os.environ.get("TAVILY_API_KEY", "")
    tavily_api_key = tavily_api_key.strip()
    local_passages = []
    try:
        retriever = get_local_retriever(language)
        if retriever is not None:
            if language == "mr":
                # Vaarta already reranks its own candidates.
                local_passages = retriever.retrieve(claim, final_top_k=top_k)
            else:
                candidate_k = 50 if language == "en" else 200
                candidates = retriever.retrieve(claim, top_k=candidate_k)
                local_passages = _rerank_evidence(claim, candidates, top_k=top_k)
            for passage in local_passages:
                if not passage.get("source"):
                    passage["source"] = "vaarta" if language == "mr" else "miracl"
    except Exception as exc:
        on_warning(f"Local {language.upper()} retrieval failed: {exc}")

    if not tavily_api_key:
        return local_passages

    try:
        web_passages = _retrieve_web_evidence(claim, tavily_api_key, top_k)
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
            return local_passages + _rerank_evidence(claim, web_passages, top_k=top_k)
        on_warning("Tavily searched the web and returned no passages for this claim.")
    return local_passages


def _verify_one_claim(claim_text: str, evidence: list[dict]) -> dict:
    """Run NLI on one claim and add confidence/explanation fields."""
    nli_result = _nli_verify_claim(claim_text, evidence, confirm_contradictions=True)

    confidence = calculate_confidence(
        nli_result.get("support_score"),
        nli_result.get("contradict_score"),
    )

    passage_scores = nli_result.get("passage_scores", [])

    explanation = generate_explanation({
        "claim": claim_text,
        "label": nli_result.get("label"),
        "support_score": nli_result.get("support_score"),
        "contradict_score": nli_result.get("contradict_score"),
        "best_evidence": nli_result.get("best_evidence"),
        "passage_scores": passage_scores,
    })

    return {
        "label": nli_result["label"],
        "support_score": nli_result.get("support_score", 0.0),
        "contradict_score": nli_result.get("contradict_score", 0.0),
        "best_evidence": nli_result.get("best_evidence"),
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

        if progress_callback:
            progress_callback(i, len(claims), claim_text)

        retrieval_warnings = []
        passages = retrieve_evidence_for_claim(
            claim_text, language=language, tavily_api_key=tavily_api_key,
            on_warning=retrieval_warnings.append,
        )

        # NLI verification + confidence/explanation.
        verification = _verify_one_claim(claim_text, passages)

        verified.append({
            "claim_id": claim_item["claim_id"],
            "claim": claim_text,
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

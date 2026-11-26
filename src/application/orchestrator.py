"""Shared entry points for AI-generated and user-provided answers.

Both modes (AI-generated and user-provided) run through the same downstream
pipeline:
  1. Claim extraction  (LLM-based, 2-pass)
  2. Evidence retrieval  (Tavily web search + semantic reranking)
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

from src.claims.extractor import extract_claims
from src.explanation.confidence import calculate_confidence, generate_explanation
from src.generation.llm import generate_answer, MODEL_NAME, LANGUAGE_NAMES
from src.verification.verifier import verify_claim as _nli_verify_claim

SUPPORTED_LANGUAGES = LANGUAGE_NAMES

# Tavily API key — set via env var or .env file.
_TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

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


def _retrieve_web_evidence(claim: str, top_k: int = 5) -> list[dict]:
    """Retrieve evidence for a single claim via Tavily web search.

    Returns an empty list when the API key is missing or the request fails,
    so the pipeline always continues.
    """
    api_key = os.environ.get("TAVILY_API_KEY", "").strip() or _TAVILY_API_KEY
    if not api_key:
        return []

    import urllib.request
    import urllib.error
    import json

    payload = json.dumps({
        "api_key": api_key,
        "query": claim,
        "search_depth": "basic",
        "max_results": top_k,
    }).encode()

    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            results = json.loads(resp.read()).get("results", [])
    except Exception as exc:
        print(f"[orchestrator] Web search request failed: {exc}")
        return []

    passages = []
    for rank, r in enumerate(results, start=1):
        text = (r.get("content") or "").strip()
        url = r.get("url", "")
        title = r.get("title", "")
        if not text:
            continue
        passages.append({
            "docid": f"web_{rank}",
            "title": title,
            "text": text,
            "url": url,
            "source": "web",
            "rank": rank,
            "relevance_score": float(r.get("score", 0.0)),
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
            print(f"[orchestrator] Reranking failed, using BM25 rank: {exc}")
    return passages[:top_k]


def retrieve_evidence_for_claim(
    claim: str,
    language: str = "en",
    top_k: int = 5,
) -> list[dict]:
    """Retrieve evidence for a single claim.

    Strategy:
      1. Retrieve local MIRACL / corpus evidence for the selected language.
      2. Rerank local candidates using the multilingual semantic reranker.
      3. If local retrieval returns no evidence and Tavily is configured,
         fall back to web search.
      4. If both sources yield no evidence, returns an empty list without
         crashing, allowing NLI to record 'Not Enough Evidence'.
    """
    language = (language or "en").lower().strip()
    local_passages = []

    # 1. Local retrieval
    retriever = get_local_retriever(language)
    if retriever is not None:
        try:
            candidate_k = 50 if language == "en" else 200
            candidates = retriever.retrieve(claim, top_k=candidate_k)
            if candidates:
                source_label = "vaarta" if language == "mr" else "miracl"
                for c in candidates:
                    if not c.get("source"):
                        c["source"] = source_label
                local_passages = _rerank_evidence(claim, candidates, top_k=top_k)
        except Exception as exc:
            print(f"[orchestrator] Local retriever error for '{language}': {exc}")
            raise RuntimeError(f"Local {language.upper()} retrieval failed: {exc}") from exc

    if local_passages:
        return local_passages

    # 2. Fallback to Tavily web search if local retrieval found nothing
    tavily_key = os.environ.get("TAVILY_API_KEY", "").strip() or _TAVILY_API_KEY
    if tavily_key:
        print(f"[orchestrator] Local retrieval returned no passages; attempting Tavily fallback...")
        web_passages = _retrieve_web_evidence(claim, top_k=top_k)
        if web_passages:
            return _rerank_evidence(claim, web_passages, top_k=top_k)

    return []


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
) -> dict:
    """Take a 'claims_extracted' result dict and run the full downstream pipeline.

    Parameters
    ----------
    result : dict
        Output of prepare_ai_answer() or prepare_manual_answer().
    progress_callback : callable, optional
        Called with (current: int, total: int, claim_text: str) after each
        claim is processed — useful for Streamlit progress bars.

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

        # Retrieve local evidence first (MIRACL/corpus), with optional web fallback.
        passages = retrieve_evidence_for_claim(claim_text, language=language)

        # NLI verification + confidence/explanation.
        verification = _verify_one_claim(claim_text, passages)

        verified.append({
            "claim_id": claim_item["claim_id"],
            "claim": claim_text,
            "evidence": passages,
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
) -> dict:
    """End-to-end pipeline.

    If *answer* is None, generates an AI answer first (Ask AI & Verify mode).
    If *answer* is provided, uses it directly (Manual Verification mode).
    """
    if answer:
        result = prepare_manual_answer(question, answer, language)
    else:
        result = prepare_ai_answer(question, language)

    return run_verification_pipeline(result, progress_callback=progress_callback)

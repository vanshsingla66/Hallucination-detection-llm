"""Streamlit UI. Run from the repository root: streamlit run app.py

Two modes share the same downstream verification pipeline:
  • Ask AI & Verify      — generate an AI answer, then verify it
  • Manually Verify      — submit an existing answer for verification

Requires Ollama to be running for fast generation:
  ollama serve   (in a separate terminal)
"""
import os
import streamlit as st

from src.application.orchestrator import (
    SUPPORTED_LANGUAGES,
    prepare_ai_answer,
    prepare_manual_answer,
    run_verification_pipeline,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Hallucination Detection System",
    page_icon="🔎",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LABEL_EMOJI = {
    "supported": "✅",
    "contradicted": "❌",
    "unsupported": "⚠️",
}

LABEL_COLOR = {
    "supported": "green",
    "contradicted": "red",
    "unsupported": "orange",
}

LABEL_TEXT = {
    "supported": "SUPPORTED",
    "contradicted": "CONTRADICTED",
    "unsupported": "NOT ENOUGH EVIDENCE",
}

STATUS_EMOJI = {
    "support_dominant": "🟢",
    "contradiction_dominant": "🔴",
    "conflicting_scores": "🟡",
    "weak_scores": "⚪",
    "insufficient_data": "⚫",
}


def _verdict_badge(label: str) -> str:
    emoji = LABEL_EMOJI.get(label, "❓")
    color = LABEL_COLOR.get(label, "grey")
    text = LABEL_TEXT.get(label, label.upper())
    return f":{color}[{emoji} {text}]"


def _summarize_verdicts(verified_claims: list[dict]) -> dict:
    counts = {"supported": 0, "contradicted": 0, "unsupported": 0}
    for vc in verified_claims:
        lbl = vc.get("label", "unsupported")
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts


def _source_name(source: str) -> str:
    return "Tavily Web" if source == "web" else source.upper()


def _clear_verification() -> None:
    st.session_state.pop("verified_result", None)


def _clear_results() -> None:
    st.session_state.pop("latest_result", None)
    _clear_verification()


# Discard results produced by older versions of the pipeline.
_CLAIM_RESULTS_VERSION = 5
if st.session_state.get("claim_results_version") != _CLAIM_RESULTS_VERSION:
    if st.session_state.get("claim_results_version") == 4:
        _clear_verification()
    else:
        _clear_results()
    st.session_state["claim_results_version"] = _CLAIM_RESULTS_VERSION


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Language")
    language_codes = list(SUPPORTED_LANGUAGES)
    language_labels = {
        code: f"{name} ({code})"
        for code, name in SUPPORTED_LANGUAGES.items()
    }
    selected_language = st.selectbox(
        "Answer language",
        language_codes,
        format_func=lambda c: language_labels[c],
        on_change=_clear_results,
    )

    st.divider()
    st.subheader("🔑 Evidence Retrieval Sources")
    st.markdown(
        "- Local corpus: MIRACL (`en`, `hi`, `bn`) and Marathi datasets (`mr`)\n"
        "- Tavily Web Search for every claim when a key is set"
    )
    tavily_input = st.text_input(
        "Tavily API key",
        type="password",
        placeholder="tvly-…",
        on_change=_clear_verification,
    )
    tavily_key = tavily_input.strip() or os.environ.get("TAVILY_API_KEY", "").strip()

    if tavily_key:
        st.success("Verification will search Tavily and the available local corpus.")
    else:
        st.info("Verification will use the available local corpus only.")

    st.divider()
    st.caption(
        "**Generation model**: llama3.2 via Ollama  \n"
        "**Verification model**: mDeBERTa-v3 multilingual NLI  \n"
        "**Reranker**: paraphrase-multilingual-MiniLM"
    )

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------
st.title("🔎 Hallucination Detection System")
st.caption(
    "Generate or submit an answer, extract factual claims, retrieve evidence, "
    "and get an NLI-based verdict for each claim."
)

# ---------------------------------------------------------------------------
# Mode selector
# ---------------------------------------------------------------------------
mode = st.radio(
    "Choose a mode",
    ["Ask AI & Verify", "Manually Verify an Answer"],
    horizontal=True,
    on_change=_clear_results,
)

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.form("hallucination_input_form"):
    question = st.text_area(
        "Your question",
        placeholder="e.g. Who discovered penicillin?",
        height=100,
    )
    manual_answer = ""
    if mode == "Manually Verify an Answer":
        manual_answer = st.text_area(
            "Answer to verify",
            placeholder="Paste the answer you want to fact-check…",
            height=180,
        )

    col_submit, col_clear = st.columns([3, 1])
    with col_submit:
        submitted = st.form_submit_button(
            "Generate answer & verify claims" if mode == "Ask AI & Verify"
            else "Verify my answer",
            type="primary",
            use_container_width=True,
        )
    with col_clear:
        clear = st.form_submit_button(
            "Clear",
            use_container_width=True,
        )

if clear:
    _clear_results()
    st.rerun()

# ---------------------------------------------------------------------------
# Step 1: Extract claims on form submit
# ---------------------------------------------------------------------------
if submitted:
    _clear_results()
    try:
        if mode == "Ask AI & Verify":
            with st.spinner("⏳ Generating answer and extracting claims…"):
                result = prepare_ai_answer(question, selected_language)
        else:
            with st.spinner("⏳ Extracting factual claims from your answer…"):
                result = prepare_manual_answer(
                    question, manual_answer, selected_language
                )
        st.session_state["latest_result"] = result
    except Exception as exc:
        st.error(f"Could not process this input: {exc}")
        st.exception(exc)

# ---------------------------------------------------------------------------
# Display claims and verification controls
# ---------------------------------------------------------------------------
result = st.session_state.get("latest_result")
if result:
    st.divider()

    # --- Answer display ---
    with st.expander("📝 Answer", expanded=True):
        st.write(result["answer"])
        st.caption(
            f"Generated by: **{result['model']}**" if result.get("model")
            else "Source: **User-provided answer**"
        )

    # --- Claims list ---
    st.subheader(f"🧩 Extracted claims ({len(result['claims'])})")
    if result["claims"]:
        for c in result["claims"]:
            st.markdown(f"**{c['claim_id']}.** {c['claim']}")
    else:
        st.warning("No claims were extracted. Try a more factual answer.")

    # --- Verify button ---
    verified_result = st.session_state.get("verified_result")
    if result["claims"] and not verified_result:
        if st.button("🔍 Verify claims with evidence", type="primary"):
            progress_bar = st.progress(0, text="Starting verification…")
            status_text = st.empty()

            def _progress(current, total, claim_text):
                pct = current / total if total else 0
                label = (
                    f"Verifying claim {current + 1}/{total}: "
                    f"{claim_text[:60]}…" if claim_text else "Done."
                )
                progress_bar.progress(pct, text=label)
                status_text.caption(label)

            try:
                with st.spinner("Running NLI verification…"):
                    verified = run_verification_pipeline(
                        result, progress_callback=_progress, tavily_api_key=tavily_key,
                    )
                st.session_state["verified_result"] = verified
                progress_bar.empty()
                status_text.empty()
                st.rerun()
            except Exception as exc:
                st.error(f"Verification failed: {exc}")
                st.exception(exc)

# ---------------------------------------------------------------------------
# Display verification results
# ---------------------------------------------------------------------------
verified_result = st.session_state.get("verified_result")
if verified_result:
    verified_claims = verified_result.get("verified_claims", [])
    counts = _summarize_verdicts(verified_claims)

    st.divider()
    st.subheader("🏁 Verification Results")

    # Summary row
    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Supported", counts.get("supported", 0))
    col2.metric("❌ Contradicted", counts.get("contradicted", 0))
    col3.metric("⚠️ Not Enough Evidence", counts.get("unsupported", 0))

    retrieval_sources = set()
    for vc in verified_claims:
        for ev in vc.get("evidence", []):
            if ev.get("source"):
                retrieval_sources.add(_source_name(ev["source"]))
    if retrieval_sources:
        st.success(f"Evidence retrieved from: {', '.join(sorted(retrieval_sources))}")
    else:
        st.info("No evidence passages were available for these claims.")
    for vc in verified_claims:
        for warning in vc.get("retrieval_warnings", []):
            st.warning(f"Claim {vc['claim_id']}: {warning}")

    st.divider()

    # Per-claim breakdown
    for vc in verified_claims:
        label = vc.get("label", "unsupported")
        emoji = LABEL_EMOJI.get(label, "❓")
        sup = vc.get("support_score", 0.0)
        con = vc.get("contradict_score", 0.0)
        status_ico = STATUS_EMOJI.get(vc.get("confidence_status", ""), "")

        with st.expander(
            f"{emoji} Claim {vc['claim_id']}: {vc['claim'][:100]}…"
            if len(vc["claim"]) > 100
            else f"{emoji} Claim {vc['claim_id']}: {vc['claim']}",
            expanded=(label == "contradicted"),
        ):
            st.markdown(f"**Verdict:** {_verdict_badge(label)}")
            st.markdown(
                f"**Support score:** `{sup:.3f}`   "
                f"**Contradiction score:** `{con:.3f}`"
            )

            if vc.get("summary"):
                st.markdown(f"**Explanation:** {vc['summary']}")

            confidence_txt = vc.get("confidence_interpretation", "")
            if confidence_txt:
                st.caption(f"{status_ico} {confidence_txt}")

            # Best evidence passage
            best_ev = vc.get("best_evidence")
            if best_ev:
                with st.container(border=True):
                    source_name = _source_name(best_ev.get("source", "miracl"))
                    docid = best_ev.get("docid", "")
                    docid_badge = f" (DocID: `{docid}`)" if docid else ""
                    st.caption(f"📄 Best matching evidence — **Source: {source_name}**{docid_badge}")
                    if best_ev.get("title"):
                        st.markdown(f"**{best_ev['title']}**")
                    ev_text = best_ev.get("text", "")
                    st.write(ev_text[:600] + "…" if len(ev_text) > 600 else ev_text)
                    if best_ev.get("url"):
                        st.markdown(f"[Source URL ↗]({best_ev['url']})")
            else:
                st.caption("No evidence passage retrieved for this claim.")

            # If both support and contradiction are notable, surface the contradicting passage explicitly
            strongest_con = vc.get("strongest_contradicting_passage")
            if con >= 0.5 and strongest_con and strongest_con.get("docid") != (best_ev or {}).get("docid"):
                with st.container(border=True):
                    con_docid = strongest_con.get("docid", "")
                    st.caption(f"⚠️ Passage with highest contradiction score (`{con:.3f}`) — **DocID: `{con_docid}`**")
                    if strongest_con.get("title"):
                        st.markdown(f"**{strongest_con['title']}**")
                    con_text = strongest_con.get("text") or strongest_con.get("text_preview", "")
                    st.write(con_text[:400] + "…" if len(con_text) > 400 else con_text)

            # All evidence passages (collapsed)
            all_ev = vc.get("evidence", [])
            if len(all_ev) > 1:
                with st.expander(f"All {len(all_ev)} evidence passages"):
                    for ev in all_ev:
                        src = _source_name(ev.get("source", "miracl"))
                        docid_txt = f" | DocID: `{ev.get('docid')}`" if ev.get("docid") else ""
                        st.caption(
                            f"**{ev.get('title', 'Untitled')}** ({src}{docid_txt})"
                        )
                        st.write(
                            ev.get("text", "")[:350] + "…"
                            if len(ev.get("text", "")) > 350
                            else ev.get("text", "")
                        )
                        if ev.get("url"):
                            st.markdown(f"[Source URL ↗]({ev['url']})")
                        st.divider()

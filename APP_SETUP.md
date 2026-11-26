# Multilingual LLM Hallucination Detection and Evidence Verification System

## System Architecture

The application provides an end-to-end hallucination detection and claim verification pipeline with two entry modes sharing the same downstream verification pipeline:

1. **Ask AI & Verify**: Generates a factual answer using a local fast LLM (`llama3.2:latest` via Ollama) and extracts atomic claims.
2. **Manually Verify an Answer**: Accepts user-supplied answers for factual verification without generating another answer.

### Downstream Verification Pipeline
- **Claim Extraction (`src/claims/`)**: 2-pass high-recall atomic claim extraction with format robustness and completeness audit. Supports multilingual queries (`en`, `hi`, `mr`, `bn`, `ta`, `te`).
- **Evidence Retrieval & Reranking (`src/retrieval/`)**:
  - Live Web evidence retrieval using Tavily API (optional, set `TAVILY_API_KEY` in UI sidebar or `.env`).
  - Semantic evidence reranking using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- **NLI Claim Verification (`src/verification/`)**:
  - Cross-lingual NLI using `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`.
  - Determines claim veracity: `supported`, `contradicted`, or `unsupported` (insufficient evidence).
- **Confidence Scoring & Explanation (`src/explanation/`)**:
  - Scores balance, confidence status (`support_dominant`, `contradiction_dominant`, `conflicting_scores`, `weak_scores`).
  - Generates clear, human-readable explanations and surfaces strongest evidence passages.
- **Unified Orchestrator (`src/application/orchestrator.py`)**:
  - Coordinates all components seamlessly.
- **Interactive Web Interface (`app.py`)**:
  - Modern, responsive Streamlit dashboard with progress tracking, claim breakdowns, score badges, and source links.

---

## Prerequisites & Installation

### 1. Fast Local LLM Generation via Ollama (Recommended)
Ollama runs quantized GGUF models locally on CPU with AVX2 acceleration:
```bash
# Start Ollama service in a terminal (if not already running):
ollama serve

# Ensure llama3.2 is available:
ollama pull llama3.2:latest
```
*(If Ollama is stopped, the application will automatically fall back to lazy-loading the Hugging Face Transformers model).*

### 2. Python Environment
Python 3.10 to 3.12 is supported:
```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Running the Application

From the project root:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Optional: Live Web Evidence Retrieval
In the sidebar of the Streamlit interface, paste your **Tavily API key** or set it in your environment:
```bash
export TAVILY_API_KEY="your-api-key"
```
Without an API key, the system runs local NLI verification on any supplied or cached evidence and marks unretrieved claims as `unsupported` with zero errors.


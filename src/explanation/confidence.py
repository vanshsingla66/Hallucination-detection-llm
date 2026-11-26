import json
from pathlib import Path
from collections import Counter


THRESHOLD = 0.5


def calculate_confidence(support_score, contradict_score):
    """
    Summarize support and contradiction scores.
    This is a heuristic assessment, not a calibrated probability.
    """

    if support_score is None or contradict_score is None:
        return {
            "score_balance": None,
            "status": "insufficient_data",
            "interpretation": "One or both scores are missing."
        }

    if support_score >= THRESHOLD and contradict_score >= THRESHOLD:
        return {
            "score_balance": round(
                abs(support_score - contradict_score), 3
            ),
            "status": "conflicting_scores",
            "interpretation": (
                "Both scores are high. Inspect the evidence to determine "
                "whether it genuinely conflicts."
            )
        }

    if support_score >= THRESHOLD:
        return {
            "score_balance": round(support_score, 3),
            "status": "support_dominant",
            "interpretation": (
                "The support score reaches the threshold, "
                "while the contradiction score does not."
            )
        }

    if contradict_score >= THRESHOLD:
        return {
            "score_balance": round(contradict_score, 3),
            "status": "contradiction_dominant",
            "interpretation": (
                "The contradiction score reaches the threshold, "
                "while the support score does not."
            )
        }

    return {
        "score_balance": 0.0,
        "status": "weak_scores",
        "interpretation": "Neither score reaches the threshold."
    }


def load_verification_data():
    project_root = Path(__file__).resolve().parents[2]
    file_path = (
        project_root / "data" / "processed" / "verification.json"
    )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def generate_explanation(record):
    claim = record["claim"]
    label = record["label"]

    support_score = record.get("support_score")
    contradict_score = record.get("contradict_score")

    passage_scores = record.get("passage_scores") or []
    best_evidence = record.get("best_evidence") or {}

    strongest_support = max(
        passage_scores,
        key=lambda p: p.get("entailment", 0.0),
        default=None
    )

    strongest_contradiction = max(
        passage_scores,
        key=lambda p: p.get("contradiction", 0.0),
        default=None
    )

    if support_score is None or contradict_score is None:
        summary = (
            "Insufficient score information to assess this claim."
        )
    elif (
        support_score >= THRESHOLD
        and contradict_score >= THRESHOLD
    ):
        summary = (
            "Both support and contradiction scores are high. "
            "Inspect the passages to determine whether they address "
            "the same fact and genuinely conflict."
        )
    elif support_score >= THRESHOLD:
        summary = (
            "The retrieved evidence provides support for this claim."
        )
    elif contradict_score >= THRESHOLD:
        summary = (
            "A passage has a high contradiction score. "
            "Check its meaning and context before treating the claim "
            "as false."
        )
    else:
        summary = (
            "Neither score reaches the threshold. "
            "The available evidence may be insufficient."
        )

    return {
        "claim": claim,
        "verification_label": label,
        "support_score": support_score,
        "contradict_score": contradict_score,
        "evidence_title": best_evidence.get("title"),
        "evidence_text": best_evidence.get("text"),
        "strongest_supporting_passage": strongest_support,
        "strongest_contradicting_passage": strongest_contradiction,
        "summary": summary
    }


def main():
    data = load_verification_data()
    all_results = []

    for record in data:
        confidence_result = calculate_confidence(
            record.get("support_score"),
            record.get("contradict_score")
        )

        explanation_result = generate_explanation(record)

        final_result = {
            **explanation_result,
            "sample_id": record.get("sample_id"),
            "claim_id": record.get("claim_id"),
            "score_balance": confidence_result["score_balance"],
            "confidence_status": confidence_result["status"],
            "confidence_interpretation": (
                confidence_result["interpretation"]
            )
        }

        all_results.append(final_result)

    project_root = Path(__file__).resolve().parents[2]
    output_path = (
        project_root / "data" / "processed" / "explanations.json"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2, ensure_ascii=False)

    # Reload saved output to verify the file.
    with open(output_path, "r", encoding="utf-8") as file:
        saved_data = json.load(file)

    print("Total claims processed:", len(all_results))
    print("Results saved to:", output_path)
    print("Saved records:", len(saved_data))

    # Confidence status distribution
    status_counts = Counter(
        item["confidence_status"] for item in saved_data
    )

    print("\nConfidence status distribution:")
    for status, count in sorted(status_counts.items()):
        print(f"{status}: {count}")

    # Original verification label distribution
    label_counts = Counter(
        item["verification_label"] for item in saved_data
    )

    print("\nOriginal verification label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"{label}: {count}")

    # Check evidence fields
    missing_support = sum(
        item["strongest_supporting_passage"] is None
        for item in saved_data
    )

    missing_contradiction = sum(
        item["strongest_contradicting_passage"] is None
        for item in saved_data
    )

    print("\nEvidence field check:")
    print("Claims without a supporting passage:", missing_support)
    print(
        "Claims without a contradicting passage:",
        missing_contradiction
    )

    # Identify claims that need manual review
    print("\nClaims needing review:")

    review_count = 0

    for item in saved_data:
        label = item["verification_label"]
        status = item["confidence_status"]

        needs_review = (
            status == "conflicting_scores"
            or (
                label == "supported"
                and status == "contradiction_dominant"
            )
            or (
                label == "contradicted"
                and status == "support_dominant"
            )
        )

        if needs_review:
            review_count += 1
            print(
                f"Sample {item['sample_id']}, "
                f"Claim {item['claim_id']}: "
                f"label={label}, status={status}"
            )

    print("\nTotal claims needing review:", review_count)


if __name__ == "__main__":
    main()

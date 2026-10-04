from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, LLM_ENABLED


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _tokens(text: str) -> set[str]:
    import re
    return set(re.findall(r"\w+", str(text).lower()))


def _overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta)


def _heuristic_eval(questions: list[str], answers: list[str],
                    contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Offline fallback metrics khi RAGAS/LLM không khả dụng (không cần API key)."""
    per_question: list[EvalResult] = []
    for q, a, ctxs, gt in zip(questions, answers, contexts, ground_truths):
        joined = " ".join(ctxs)
        faithfulness = _overlap(a, joined)
        answer_relevancy = _overlap(a, q)
        context_precision = (
            sum(_overlap(c, gt) for c in ctxs) / len(ctxs) if ctxs else 0.0
        )
        context_recall = _overlap(gt, joined)
        per_question.append(EvalResult(
            question=q, answer=a, contexts=ctxs, ground_truth=gt,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall,
        ))

    def _mean(attr: str) -> float:
        if not per_question:
            return 0.0
        return sum(getattr(r, attr) for r in per_question) / len(per_question)

    return {
        "faithfulness": _mean("faithfulness"),
        "answer_relevancy": _mean("answer_relevancy"),
        "context_precision": _mean("context_precision"),
        "context_recall": _mean("context_recall"),
        "per_question": per_question,
        "engine": "heuristic",
    }


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation, fallback sang heuristic metrics khi thiếu API/model."""
    if not LLM_ENABLED:
        print("  ⚠️  Không có LLM provider hợp lệ — dùng heuristic metrics offline.")
        return _heuristic_eval(questions, answers, contexts, ground_truths)
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        from src.llm import get_ragas_llm, get_ragas_embeddings

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=get_ragas_llm(),
            embeddings=get_ragas_embeddings(),
        )
        df = result.to_pandas()
        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=list(row["contexts"]),
                ground_truth=row["ground_truth"],
                faithfulness=float(row.get("faithfulness", 0.0)),
                answer_relevancy=float(row.get("answer_relevancy", 0.0)),
                context_precision=float(row.get("context_precision", 0.0)),
                context_recall=float(row.get("context_recall", 0.0)),
            )
            for _, row in df.iterrows()
        ]

        import math
        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        if not per_question or all(
            not math.isfinite(getattr(r, m)) for r in per_question for m in metrics
        ):
            print("  ⚠️  RAGAS trả về toàn NaN (judge lỗi?); dùng heuristic fallback.")
            return _heuristic_eval(questions, answers, contexts, ground_truths)

        # RAGAS có thể trả NaN lẻ tẻ cho vài metric/câu → vá bằng giá trị heuristic.
        heuristic = _heuristic_eval(questions, answers, contexts, ground_truths)
        for r, hr in zip(per_question, heuristic["per_question"]):
            for m in metrics:
                if not math.isfinite(getattr(r, m)):
                    setattr(r, m, getattr(hr, m))

        aggregate = {
            m: sum(getattr(r, m) for r in per_question) / len(per_question) for m in metrics
        }
        return {**aggregate, "per_question": per_question, "engine": "ragas"}
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed ({e}); falling back to heuristic metrics.")
        return _heuristic_eval(questions, answers, contexts, ground_truths)


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": (
            "LLM hallucinating — answer chứa claim không được context hỗ trợ",
            "Siết prompt (CHỈ dựa trên context), giảm temperature, yêu cầu trích dẫn",
        ),
        "context_recall": (
            "Missing relevant chunks — retrieval không lấy được evidence đúng",
            "Cải thiện chunking (hierarchical/structure-aware) hoặc thêm BM25 lexical",
        ),
        "context_precision": (
            "Too many irrelevant chunks — context bị nhiễu",
            "Thêm cross-encoder reranking hoặc metadata filtering",
        ),
        "answer_relevancy": (
            "Answer doesn't match question — trả lời lạc đề/thiếu ý",
            "Cải thiện prompt template và định dạng câu trả lời",
        ),
    }

    analyzed: list[dict] = []
    for r in eval_results:
        metrics = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        avg = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "contexts": r.contexts,
            "worst_metric": worst_metric,
            "score": round(metrics[worst_metric], 4),
            "avg_score": round(avg, 4),
            "diagnosis": f"{diagnosis} (worst={worst_metric}, score={metrics[worst_metric]:.2f})",
            "suggested_fix": suggested_fix,
            "error_tree": (
                f"Output sai? → Context đúng? (recall={r.context_recall:.2f}, "
                f"precision={r.context_precision:.2f}) → Query OK? → Fix: {suggested_fix}"
            ),
        })

    analyzed.sort(key=lambda x: x["avg_score"])
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")

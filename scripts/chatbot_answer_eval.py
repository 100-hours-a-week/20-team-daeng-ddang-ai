#!/usr/bin/env python3
"""Capture and compare chatbot answers for qualitative evaluation."""

from __future__ import annotations

import argparse
import difflib
import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


DEFAULT_CASES: list[dict[str, Any]] = [
    {
        "id": "water_polyuria",
        "question": "강아지가 물을 많이 마시고 소변도 자주 봐요. 어떤 질환을 의심해야 하나요?",
        "user_context": {"dog_age_years": 9, "dog_weight_kg": 5.2, "breed": "말티즈"},
    },
    {
        "id": "patella_walk",
        "question": "슬개골이 안 좋은 아이 산책은 어느 정도가 적절한가요?",
        "user_context": {"dog_age_years": 6, "dog_weight_kg": 4.6, "breed": "포메라니안"},
    },
    {
        "id": "limping_redflag",
        "question": "강아지가 갑자기 절뚝거릴 때 응급으로 봐야 하는 신호가 있나요?",
    },
    {
        "id": "diarrhea_two_days",
        "question": "강아지 설사가 이틀째인데 병원 가야 하나요?",
    },
    {
        "id": "night_cough",
        "question": "강아지가 밤에 기침을 자주 해요. 심장 문제일 수도 있나요?",
    },
]


@dataclass
class AnswerResult:
    case_id: str
    question: str
    ok: bool
    latency_ms: float
    status_code: int
    answer: str
    answer_chars: int
    citations: list[dict[str, Any]]
    error: str | None = None


def load_cases(cases_file: str | None) -> list[dict[str, Any]]:
    if cases_file:
        with open(cases_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("cases file must be a JSON list")
        return data

    default_path = Path("scripts/chatbot_bench_cases.example.json")
    if default_path.is_file():
        with open(default_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data

    return DEFAULT_CASES


def build_payload(case: dict[str, Any], idx: int) -> dict[str, Any]:
    message = str(case.get("question", "")).strip()
    if not message:
        raise ValueError(f"case[{idx}] missing 'question'")
    return {
        "dog_id": int(case.get("dog_id", 1)),
        "conversation_id": str(case.get("conversation_id", f"answer-eval-{idx}")),
        "message": {"role": "user", "content": message},
        "history": case.get("history", []),
        "user_context": case.get("user_context"),
    }


def run_eval(args: argparse.Namespace) -> int:
    endpoint = args.base_url.rstrip("/") + "/api/vet/chat"
    cases = load_cases(args.cases_file)
    results: list[AnswerResult] = []

    if args.warmup > 0 and cases:
        warmup_payload = build_payload(cases[0], 0)
        for _ in range(args.warmup):
            try:
                requests.post(endpoint, json=warmup_payload, timeout=args.timeout)
            except Exception:
                pass

    started_at = datetime.now().isoformat()
    for i, case in enumerate(cases):
        case_id = str(case.get("id", f"case_{i}"))
        payload = build_payload(case, i)
        start = time.perf_counter()
        try:
            resp = requests.post(endpoint, json=payload, timeout=args.timeout)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if resp.status_code != 200:
                results.append(
                    AnswerResult(
                        case_id=case_id,
                        question=payload["message"]["content"],
                        ok=False,
                        latency_ms=elapsed_ms,
                        status_code=resp.status_code,
                        answer="",
                        answer_chars=0,
                        citations=[],
                        error=resp.text[:500],
                    )
                )
                continue

            body = resp.json()
            answer = str(body.get("answer") or "")
            citations = body.get("citations") or []
            if not isinstance(citations, list):
                citations = []
            results.append(
                AnswerResult(
                    case_id=case_id,
                    question=payload["message"]["content"],
                    ok=True,
                    latency_ms=elapsed_ms,
                    status_code=200,
                    answer=answer,
                    answer_chars=len(answer),
                    citations=[c for c in citations if isinstance(c, dict)],
                )
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            results.append(
                AnswerResult(
                    case_id=case_id,
                    question=payload["message"]["content"],
                    ok=False,
                    latency_ms=elapsed_ms,
                    status_code=0,
                    answer="",
                    answer_chars=0,
                    citations=[],
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    ok_results = [r for r in results if r.ok]
    latencies = [r.latency_ms for r in ok_results]
    output = {
        "summary": {
            "run_started_at": started_at,
            "run_finished_at": datetime.now().isoformat(),
            "tag": args.tag,
            "base_url": args.base_url,
            "endpoint": endpoint,
            "cases_file": args.cases_file or "scripts/chatbot_bench_cases.example.json or built-in",
            "timeout_seconds": args.timeout,
            "warmup": args.warmup,
            "total_requests": len(results),
            "success_count": len(ok_results),
            "failure_count": len(results) - len(ok_results),
            "success_rate": round((len(ok_results) / len(results)) if results else 0.0, 4),
            "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
            "avg_answer_chars": round(statistics.mean([r.answer_chars for r in ok_results]), 2) if ok_results else 0.0,
            "avg_citation_count": round(statistics.mean([len(r.citations) for r in ok_results]), 2) if ok_results else 0.0,
        },
        "results": [r.__dict__ for r in results],
    }

    output_path = Path(args.output) if args.output else Path("scripts/bench_results") / f"chatbot_answer_eval_{args.tag}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[Answer Eval: {args.tag}]")
    print(f"- success_rate: {output['summary']['success_rate']*100:.1f}% ({output['summary']['success_count']}/{output['summary']['total_requests']})")
    print(f"- avg_latency_ms: {output['summary']['avg_latency_ms']}")
    print(f"- avg_answer_chars: {output['summary']['avg_answer_chars']}")
    print(f"- avg_citation_count: {output['summary']['avg_citation_count']}")
    print(f"- saved: {output_path}")
    return 0


def _load_run(path: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        body = json.load(f)
    results = body.get("results", [])
    by_id = {str(item["case_id"]): item for item in results if isinstance(item, dict) and "case_id" in item}
    return body.get("summary", {}), by_id


def _citation_titles(citations: list[dict[str, Any]]) -> list[str]:
    titles: list[str] = []
    for citation in citations:
        title = str(citation.get("title") or citation.get("doc_id") or "").strip()
        if title:
            titles.append(title)
    return titles


def compare_runs(args: argparse.Namespace) -> int:
    baseline_summary, baseline = _load_run(args.baseline)
    candidate_summary, candidate = _load_run(args.candidate)
    case_ids = sorted(set(baseline) | set(candidate))

    lines = [
        "[Answer Compare]",
        f"- baseline : {args.baseline}",
        f"- candidate: {args.candidate}",
        f"- baseline_tag: {baseline_summary.get('tag', '')}",
        f"- candidate_tag: {candidate_summary.get('tag', '')}",
        "",
    ]

    for case_id in case_ids:
        base = baseline.get(case_id)
        cand = candidate.get(case_id)
        if not base or not cand:
            lines.append(f"[{case_id}] missing in one run")
            lines.append("")
            continue

        base_answer = str(base.get("answer") or "")
        cand_answer = str(cand.get("answer") or "")
        similarity = difflib.SequenceMatcher(None, base_answer, cand_answer).ratio()
        base_titles = _citation_titles(base.get("citations") or [])
        cand_titles = _citation_titles(cand.get("citations") or [])
        added_titles = [title for title in cand_titles if title not in base_titles]
        removed_titles = [title for title in base_titles if title not in cand_titles]

        lines.append(f"[{case_id}]")
        lines.append(f"question: {base.get('question') or cand.get('question') or ''}")
        lines.append(f"answer_similarity: {similarity:.3f}")
        lines.append(
            f"latency_ms: baseline={float(base.get('latency_ms', 0.0)):.2f} candidate={float(cand.get('latency_ms', 0.0)):.2f}"
        )
        lines.append(
            f"answer_chars: baseline={int(base.get('answer_chars', 0))} candidate={int(cand.get('answer_chars', 0))}"
        )
        lines.append(f"baseline_answer: {base_answer[:args.preview_chars]}")
        lines.append(f"candidate_answer: {cand_answer[:args.preview_chars]}")
        lines.append(f"baseline_citations: {', '.join(base_titles[:args.max_citations]) or '-'}")
        lines.append(f"candidate_citations: {', '.join(cand_titles[:args.max_citations]) or '-'}")
        lines.append(f"added_citations: {', '.join(added_titles[:args.max_citations]) or '-'}")
        lines.append(f"removed_citations: {', '.join(removed_titles[:args.max_citations]) or '-'}")
        lines.append("")

    report = "\n".join(lines)
    print(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"\n- saved: {output_path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run and compare chatbot answer evaluations.")
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="save raw answers and citations for a question set")
    run_p.add_argument("--tag", required=True, help="result tag, e.g. baseline_ko_sroberta")
    run_p.add_argument("--base-url", default="http://localhost:8300", help="chatbot-service base URL")
    run_p.add_argument("--cases-file", default=None, help="optional JSON list of question cases")
    run_p.add_argument("--warmup", type=int, default=1, help="warm-up request count")
    run_p.add_argument("--timeout", type=int, default=120, help="per-request timeout seconds")
    run_p.add_argument("--output", default=None, help="optional output JSON path")
    run_p.set_defaults(func=run_eval)

    cmp_p = sub.add_parser("compare", help="compare two answer evaluation JSON files")
    cmp_p.add_argument("--baseline", required=True, help="baseline JSON path")
    cmp_p.add_argument("--candidate", required=True, help="candidate JSON path")
    cmp_p.add_argument("--preview-chars", type=int, default=220, help="answer preview length per case")
    cmp_p.add_argument("--max-citations", type=int, default=3, help="max citation titles to display")
    cmp_p.add_argument("--output", default=None, help="optional text report path")
    cmp_p.set_defaults(func=compare_runs)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

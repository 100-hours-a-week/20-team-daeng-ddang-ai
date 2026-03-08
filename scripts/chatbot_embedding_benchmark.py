#!/usr/bin/env python3
"""Chatbot embedding benchmark utility.

Use this to compare chatbot performance/quality before and after changing
embedding model settings.

Example:
    # 1) run baseline
    python scripts/chatbot_embedding_benchmark.py run \
      --tag baseline_ko_sroberta \
      --base-url http://localhost:8300

    # 2) switch embedding + rebuild vector DB, then run again
    python scripts/chatbot_embedding_benchmark.py run \
      --tag candidate_e5_base \
      --base-url http://localhost:8300

    # 3) compare two run outputs
    python scripts/chatbot_embedding_benchmark.py compare \
      --baseline scripts/bench_results/chatbot_embedding_baseline_ko_sroberta.json \
      --candidate scripts/bench_results/chatbot_embedding_candidate_e5_base.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


DEFAULT_CASES: list[dict[str, Any]] = [
    {
        "id": "c01",
        "question": "강아지가 물을 많이 마시고 소변도 자주 봐요. 어떤 질환을 의심해야 하나요?",
        "expected_keywords": ["당뇨", "신장", "내원", "검사"],
    },
    {
        "id": "c02",
        "question": "슬개골이 안 좋은 아이 산책은 어느 정도가 적절한가요?",
        "expected_keywords": ["슬개골", "산책", "통증", "내원"],
    },
    {
        "id": "c03",
        "question": "노령견이 밥을 잘 안 먹고 기운이 없어요. 집에서 먼저 뭘 확인해야 할까요?",
        "expected_keywords": ["식욕", "탈수", "내원", "증상"],
    },
    {
        "id": "c04",
        "question": "강아지가 갑자기 절뚝거릴 때 응급으로 봐야 하는 신호가 있나요?",
        "expected_keywords": ["절뚝", "통증", "응급", "내원"],
    },
    {
        "id": "c05",
        "question": "강아지 설사가 이틀째인데 병원 가야 하나요?",
        "expected_keywords": ["설사", "탈수", "내원", "검사"],
    },
    {
        "id": "c06",
        "question": "보행이 비틀거리는 강아지에게 집에서 하면 안 되는 행동이 뭔가요?",
        "expected_keywords": ["보행", "안정", "무리", "내원"],
    },
    {
        "id": "c07",
        "question": "체중이 많이 늘어난 소형견의 관절 관리 방법 알려줘.",
        "expected_keywords": ["체중", "관절", "운동", "식이"],
    },
    {
        "id": "c08",
        "question": "강아지가 밤에 기침을 자주 해요. 심장 문제일 수도 있나요?",
        "expected_keywords": ["기침", "심장", "검사", "내원"],
    },
]


@dataclass
class CaseResult:
    case_id: str
    ok: bool
    latency_ms: float
    status_code: int
    answer_chars: int
    citation_count: int
    answer_keyword_hit: int
    citation_keyword_hit: int
    error: str | None = None


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = max(0, min(len(arr) - 1, math.ceil((p / 100) * len(arr)) - 1))
    return arr[idx]


def load_cases(cases_file: str | None) -> list[dict[str, Any]]:
    if not cases_file:
        return DEFAULT_CASES
    with open(cases_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("cases file must be a JSON list")
    return data


def build_payload(case: dict[str, Any], idx: int) -> dict[str, Any]:
    message = str(case.get("question", "")).strip()
    if not message:
        raise ValueError(f"case[{idx}] missing 'question'")
    return {
        "dog_id": int(case.get("dog_id", 1)),
        "conversation_id": str(case.get("conversation_id", f"embed-bench-{idx}")),
        "message": {
            "role": "user",
            "content": message,
        },
        "history": case.get("history", []),
        "user_context": case.get("user_context"),
    }


def text_contains_any(text: str, keywords: list[str]) -> int:
    if not text or not keywords:
        return 0
    lower_text = text.lower()
    for kw in keywords:
        if kw.lower() in lower_text:
            return 1
    return 0


def run_benchmark(args: argparse.Namespace) -> int:
    endpoint = args.base_url.rstrip("/") + "/api/vet/chat"
    cases = load_cases(args.cases_file)
    results: list[CaseResult] = []

    if args.warmup > 0:
        warmup_case = cases[0]
        for _ in range(args.warmup):
            payload = build_payload(warmup_case, 0)
            try:
                requests.post(endpoint, json=payload, timeout=args.timeout)
            except Exception:
                pass

    run_started = datetime.now().isoformat()
    for rep in range(args.repeat):
        for i, case in enumerate(cases):
            case_id = str(case.get("id", f"case_{i}"))
            expected_keywords = case.get("expected_keywords", []) or []
            payload = build_payload(case, i)

            start = time.perf_counter()
            try:
                resp = requests.post(endpoint, json=payload, timeout=args.timeout)
                elapsed_ms = (time.perf_counter() - start) * 1000
                if resp.status_code != 200:
                    results.append(
                        CaseResult(
                            case_id=case_id,
                            ok=False,
                            latency_ms=elapsed_ms,
                            status_code=resp.status_code,
                            answer_chars=0,
                            citation_count=0,
                            answer_keyword_hit=0,
                            citation_keyword_hit=0,
                            error=resp.text[:300],
                        )
                    )
                    continue

                body = resp.json()
                answer = str(body.get("answer") or "")
                citations = body.get("citations") or []
                cite_text = " ".join(
                    str(c.get("title", "")) + " " + str(c.get("snippet", ""))
                    for c in citations
                    if isinstance(c, dict)
                )

                results.append(
                    CaseResult(
                        case_id=case_id,
                        ok=True,
                        latency_ms=elapsed_ms,
                        status_code=200,
                        answer_chars=len(answer),
                        citation_count=len(citations),
                        answer_keyword_hit=text_contains_any(answer, expected_keywords),
                        citation_keyword_hit=text_contains_any(cite_text, expected_keywords),
                    )
                )
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                results.append(
                    CaseResult(
                        case_id=case_id,
                        ok=False,
                        latency_ms=elapsed_ms,
                        status_code=0,
                        answer_chars=0,
                        citation_count=0,
                        answer_keyword_hit=0,
                        citation_keyword_hit=0,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

    ok_results = [r for r in results if r.ok]
    latencies = [r.latency_ms for r in ok_results]
    total = len(results)
    success = len(ok_results)

    summary = {
        "run_started_at": run_started,
        "run_finished_at": datetime.now().isoformat(),
        "tag": args.tag,
        "base_url": args.base_url,
        "endpoint": endpoint,
        "cases_file": args.cases_file or "built-in",
        "repeat": args.repeat,
        "warmup": args.warmup,
        "timeout_seconds": args.timeout,
        "total_requests": total,
        "success_count": success,
        "failure_count": total - success,
        "success_rate": round((success / total) if total else 0.0, 4),
        "latency_ms": {
            "avg": round(statistics.mean(latencies), 2) if latencies else 0.0,
            "p50": round(percentile(latencies, 50), 2) if latencies else 0.0,
            "p95": round(percentile(latencies, 95), 2) if latencies else 0.0,
            "p99": round(percentile(latencies, 99), 2) if latencies else 0.0,
            "max": round(max(latencies), 2) if latencies else 0.0,
        },
        "response_quality": {
            "avg_answer_chars": round(statistics.mean([r.answer_chars for r in ok_results]), 2) if ok_results else 0.0,
            "avg_citation_count": round(statistics.mean([r.citation_count for r in ok_results]), 2) if ok_results else 0.0,
            "answer_keyword_hit_rate": round(
                (sum(r.answer_keyword_hit for r in ok_results) / len(ok_results)) if ok_results else 0.0, 4
            ),
            "citation_keyword_hit_rate": round(
                (sum(r.citation_keyword_hit for r in ok_results) / len(ok_results)) if ok_results else 0.0, 4
            ),
        },
    }

    output = {
        "summary": summary,
        "per_request": [r.__dict__ for r in results],
    }

    output_path = Path(args.output) if args.output else Path("scripts/bench_results") / f"chatbot_embedding_{args.tag}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[Run Summary: {args.tag}]")
    print(f"- success_rate: {summary['success_rate']*100:.1f}% ({success}/{total})")
    print(
        f"- latency_ms: avg={summary['latency_ms']['avg']}, p50={summary['latency_ms']['p50']}, "
        f"p95={summary['latency_ms']['p95']}, p99={summary['latency_ms']['p99']}, max={summary['latency_ms']['max']}"
    )
    print(
        f"- quality: citation_hit={summary['response_quality']['citation_keyword_hit_rate']}, "
        f"answer_hit={summary['response_quality']['answer_keyword_hit_rate']}, "
        f"avg_citations={summary['response_quality']['avg_citation_count']}"
    )
    print(f"- saved: {output_path}")
    return 0


def compare_runs(args: argparse.Namespace) -> int:
    with open(args.baseline, "r", encoding="utf-8") as f:
        baseline = json.load(f)["summary"]
    with open(args.candidate, "r", encoding="utf-8") as f:
        candidate = json.load(f)["summary"]

    def diff(c: float, b: float) -> float:
        return round(c - b, 4)

    print("\n[Compare Summary]")
    print(f"- baseline : {args.baseline}")
    print(f"- candidate: {args.candidate}")
    print("")
    print("Success Rate")
    print(f"  baseline={baseline['success_rate']:.4f} candidate={candidate['success_rate']:.4f} diff={diff(candidate['success_rate'], baseline['success_rate']):+.4f}")
    print("Latency (ms)")
    for key in ["avg", "p50", "p95", "p99", "max"]:
        b = float(baseline["latency_ms"][key])
        c = float(candidate["latency_ms"][key])
        print(f"  {key:>3}: baseline={b:.2f} candidate={c:.2f} diff={c-b:+.2f}")
    print("Quality")
    for key in ["citation_keyword_hit_rate", "answer_keyword_hit_rate", "avg_citation_count", "avg_answer_chars"]:
        b = float(baseline["response_quality"][key])
        c = float(candidate["response_quality"][key])
        print(f"  {key}: baseline={b:.4f} candidate={c:.4f} diff={diff(c, b):+.4f}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark chatbot embedding model changes.")
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run benchmark once and save JSON result")
    run_p.add_argument("--tag", required=True, help="result tag, e.g. baseline_ko or candidate_e5")
    run_p.add_argument("--base-url", default="http://localhost:8300", help="chatbot-service base URL")
    run_p.add_argument("--cases-file", default=None, help="optional JSON list of test cases")
    run_p.add_argument("--repeat", type=int, default=1, help="repeat full case set N times")
    run_p.add_argument("--warmup", type=int, default=1, help="warm-up request count")
    run_p.add_argument("--timeout", type=int, default=120, help="per-request timeout seconds")
    run_p.add_argument("--output", default=None, help="optional output JSON path")
    run_p.set_defaults(func=run_benchmark)

    cmp_p = sub.add_parser("compare", help="compare two saved benchmark JSON files")
    cmp_p.add_argument("--baseline", required=True, help="baseline JSON path")
    cmp_p.add_argument("--candidate", required=True, help="candidate JSON path")
    cmp_p.set_defaults(func=compare_runs)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

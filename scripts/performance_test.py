"""Simple benchmark script for ai-orchestrator endpoints.

Usage examples:
    # single endpoint sequentially
    python scripts/performance_test.py -e missions --count 10

    # multiple endpoints sequentially
    python scripts/performance_test.py -e missions face healthcare --count 5

    # concurrent requests (using async client)
    python scripts/performance_test.py -e face --count 50 --async
    python scripts/performance_test.py -e face healthcare --count 20 --async

Each endpoint will receive the specified number of requests.
- sync mode: endpoints are exercised one by one.
- async mode: by default endpoints are also exercised one by one (fair mode),
  with in-flight concurrency controlled by --concurrency.
- async + --mix-endpoints: all endpoints are mixed concurrently (stress mode).
Results are reported separately per endpoint.

Before running, ensure ai-orchestrator (and any downstream services) are
running locally on http://localhost:8000.
"""

import argparse
import time
import json
import asyncio
import math
from datetime import datetime
from pathlib import Path

import requests
import httpx

BASE = "http://localhost:8000"


def endpoint_url(endpoint: str) -> str:
    if endpoint == "missions":
        return f"{BASE}/api/missions/judge"
    if endpoint == "chatbot":
        return f"{BASE}/api/vet/chat"
    return f"{BASE}/api/{endpoint}/analyze"

def build_payload(endpoint: str):
    """Return a minimal valid request body for the given endpoint.

    Uses real test videos from S3 and actual user messages.
    """
    if endpoint == "missions":
        # MissionInput requires int mission_id, enum mission_type, and walk_id
        return {
            "analysis_id": "bench-missions",
            "walk_id": 1,
            "missions": [
                {
                    "mission_id": 1,
                    "mission_type": "PAW",
                    "video_url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/PAW_03.mp4"
                }
            ]
        }
    elif endpoint == "face":
        # FaceAnalyzeRequest needs an analysis_id and optional video_url
        return {
            "analysis_id": "bench-face",
            "video_url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/face_test/testVideo01.mp4"
        }
    elif endpoint == "healthcare":
        # HealthcareAnalyzeRequest with real test video
        return {
            "analysis_id": "bench-healthcare",
            "dog_id": 123,
            "video_url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/healthcare_test/testVideo02.mp4"
        }
    elif endpoint == "chatbot":
        # ChatRequest with real user question
        return {
            "dog_id": 1,
            "conversation_id": "bench-conv1",
            "message": {
                "role": "user",
                "content": "최근 들어 우리 아이가 물을 평소보다 엄청 많이 마시고 오줌도 자주 싸요. 단순히 더워서 그런 걸까요, 아니면 당뇨 같은 병을 의심해 봐야 할까요?"
            },
            "history": []
        }
    else:
        raise ValueError(endpoint)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(0, min(len(sorted_values) - 1, math.ceil((p / 100) * len(sorted_values)) - 1))
    return sorted_values[index]


def sync_worker(
    endpoints: list[str], count: int, timeout_seconds: int
) -> tuple[dict[str, list[float]], dict[str, int]]:
    # perform `count` requests for each endpoint sequentially
    overall_stats: dict[str, list[float]] = {ep: [] for ep in endpoints}
    failures: dict[str, int] = {ep: 0 for ep in endpoints}

    for ep in endpoints:
        url = endpoint_url(ep)
        payload = build_payload(ep)
        for _ in range(count):
            start = time.time()
            try:
                r = requests.post(url, json=payload, timeout=timeout_seconds)
                elapsed = time.time() - start
                if r.status_code != 200:
                    failures[ep] += 1
                    # show full response for debugging
                    print(f"error ({ep}) {r.status_code} -> {r.text}")
                    continue
                overall_stats[ep].append(elapsed)
            except Exception as exc:
                failures[ep] += 1
                print(f"exception ({ep}) {type(exc).__name__} {exc}")

    # print per-endpoint summaries
    for ep, times in overall_stats.items():
        print(f"\nEndpoint '{ep}':")
        print_summary(times, "sync", failures[ep], count)

    return overall_stats, failures


def print_summary(times: list[float], mode: str, failures: int, requested: int):
    count = len(times)
    if count == 0:
        print(f"{mode} results: 0 requests (all failed, failures={failures}/{requested})")
        return
    total = sum(times)
    success_rate = (count / requested) * 100 if requested > 0 else 0.0
    print(f"{mode} results: {count} requests")
    print(f"  success : {success_rate:.1f}% ({count}/{requested})")
    print(f"  failures: {failures}")
    print(f"  total   : {total:.2f}s")
    print(f"  average : {total/count:.2f}s")
    print(f"  min     : {min(times):.2f}s")
    print(f"  p50     : {percentile(times, 50):.2f}s")
    print(f"  p95     : {percentile(times, 95):.2f}s")
    print(f"  p99     : {percentile(times, 99):.2f}s")
    print(f"  max     : {max(times):.2f}s")


def save_results(
    overall_stats: dict[str, list[float]],
    failures: dict[str, int],
    requested_per_endpoint: int,
    mode: str,
    output_path: str,
    concurrency: int,
    mix_endpoints: bool,
):
    """Save benchmark results to JSON file."""
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "concurrency": concurrency,
        "mix_endpoints": mix_endpoints,
        "endpoints": {}
    }
    
    for ep, times in overall_stats.items():
        count = len(times)
        failure_count = failures.get(ep, 0)
        success_rate = (count / requested_per_endpoint) if requested_per_endpoint > 0 else 0.0
        result_data["endpoints"][ep] = {
                "count": count,
                "requested": requested_per_endpoint,
                "failures": failure_count,
                "success_rate": round(success_rate, 4),
                "total_seconds": round(sum(times), 2),
                "average_seconds": round(sum(times) / count, 2) if count else 0.0,
                "min_seconds": round(min(times), 2) if count else 0.0,
                "p50_seconds": round(percentile(times, 50), 2) if count else 0.0,
                "p95_seconds": round(percentile(times, 95), 2) if count else 0.0,
                "p99_seconds": round(percentile(times, 99), 2) if count else 0.0,
                "max_seconds": round(max(times), 2) if count else 0.0,
        }
    
    # ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")


async def async_worker(
    endpoints: list[str],
    count: int,
    timeout_seconds: int,
    concurrency: int,
    mix_endpoints: bool,
) -> tuple[dict[str, list[float]], dict[str, int]]:
    # send `count` requests per endpoint with controlled concurrency
    overall_stats: dict[str, list[float]] = {ep: [] for ep in endpoints}
    failures: dict[str, int] = {ep: 0 for ep in endpoints}
    sem = asyncio.Semaphore(max(1, concurrency))

    timeout = httpx.Timeout(timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async def make_task(ep: str):
            url = endpoint_url(ep)
            payload = build_payload(ep)
            async with sem:
                start = time.time()
                try:
                    r = await client.post(url, json=payload)
                    elapsed = time.time() - start
                    if r.status_code != 200:
                        print(f"error ({ep}) {r.status_code} {r.text}")
                        return ep, None
                    return ep, elapsed
                except Exception as exc:  # could be httpx.ReadTimeout etc.
                    print(f"exception ({ep}) {type(exc).__name__} {exc}")
                    return ep, None

        def apply_results(results: list[tuple[str, float | None] | Exception]):
            for item in results:
                if isinstance(item, Exception):
                    print(f"task-level exception {type(item).__name__}: {item}")
                    continue
                ep, elapsed = item
                if elapsed is None:
                    failures[ep] += 1
                else:
                    overall_stats[ep].append(elapsed)

        if mix_endpoints:
            tasks = [make_task(ep) for ep in endpoints for _ in range(count)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            apply_results(results)
        else:
            # Fair comparison mode: endpoint interference 없이 endpoint별로 분리 실행
            for ep in endpoints:
                tasks = [make_task(ep) for _ in range(count)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                apply_results(results)

        for ep in endpoints:
            if failures[ep]:
                print(f"{failures[ep]} failed/{ep} requests (see earlier errors)")

    # print per-endpoint summaries
    for ep, times in overall_stats.items():
        print(f"\nEndpoint '{ep}':")
        print_summary(times, "async", failures[ep], count)

    return overall_stats, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", "-e", nargs="+",
                        choices=["missions", "face", "healthcare", "chatbot"],
                        default=["missions"],
                        help="one or more endpoints to test")
    parser.add_argument("--count", type=int, default=10,
                        help="number of requests per endpoint")
    parser.add_argument("--async", dest="use_async", action="store_true",
                        help="send requests concurrently")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="save results to JSON file (e.g., results.json)")
    parser.add_argument("--timeout-seconds", type=int, default=60,
                        help="request timeout in seconds (default: 60)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="max in-flight requests in async mode (default: 4)")
    parser.add_argument("--mix-endpoints", action="store_true",
                        help="mix multiple endpoints concurrently (default: false)")
    args = parser.parse_args()
    
    mode = "async" if args.use_async else "sync"
    
    if args.use_async:
        results, failures = asyncio.run(
            async_worker(
                args.endpoints,
                args.count,
                args.timeout_seconds,
                args.concurrency,
                args.mix_endpoints,
            )
        )
    else:
        results, failures = sync_worker(args.endpoints, args.count, args.timeout_seconds)
    
    # save results if output path is specified
    if args.output:
        save_results(
            results,
            failures,
            args.count,
            mode,
            args.output,
            args.concurrency if args.use_async else 1,
            args.mix_endpoints if args.use_async else False,
        )


if __name__ == "__main__":
    main()

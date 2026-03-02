"""Simple benchmark script for ai-orchestrator endpoints.

Usage examples:
    # single endpoint sequentially
    python scripts/performance_test.py -e missions --count 10

    # multiple endpoints sequentially
    python scripts/performance_test.py -e missions face healthcare --count 5

    # concurrent requests (using async client)
    python scripts/performance_test.py -e face --count 50 --async
    python scripts/performance_test.py -e face healthcare --count 20 --async

Each endpoint will receive the specified number of requests.  If multiple
endpoints are given the script will exercise each one in turn (sync mode) or
mix them concurrently (async mode).  Results are reported separately per
endpoint.

Before running, ensure ai-orchestrator (and any downstream services) are
running locally on http://localhost:8000.
"""

import argparse
import time
import json
import asyncio
from datetime import datetime
from pathlib import Path

import requests
import httpx

BASE = "http://localhost:8000"

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


def sync_worker(endpoints: list[str], count: int) -> dict[str, list[float]]:
    # perform `count` requests for each endpoint sequentially
    overall_stats: dict[str, list[float]] = {ep: [] for ep in endpoints}

    for ep in endpoints:
        if ep == "missions":
            url = f"{BASE}/api/missions/judge"
        elif ep == "chatbot":
            url = f"{BASE}/api/vet/chat"  # chatbot endpoint
        else:
            url = f"{BASE}/api/{ep}/analyze"
        payload = build_payload(ep)
        for i in range(count):
            start = time.time()
            r = requests.post(url, json=payload, timeout=60)
            elapsed = time.time() - start
            overall_stats[ep].append(elapsed)
            if r.status_code != 200:
                # show full response for debugging
                print(f"error ({ep}) {r.status_code} -> {r.text}")

    # print per-endpoint summaries
    for ep, times in overall_stats.items():
        print(f"\nEndpoint '{ep}':")
        print_summary(times, "sync")
    
    return overall_stats


def print_summary(times, mode: str):
    count = len(times)
    total = sum(times)
    print(f"{mode} results: {count} requests")
    print(f"  total   : {total:.2f}s")
    print(f"  average : {total/count:.2f}s")
    print(f"  min      : {min(times):.2f}s")
    print(f"  max      : {max(times):.2f}s")


def save_results(overall_stats: dict[str, list[float]], mode: str, output_path: str):
    """Save benchmark results to JSON file."""
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "endpoints": {}
    }
    
    for ep, times in overall_stats.items():
        if times:
            result_data["endpoints"][ep] = {
                "count": len(times),
                "total_seconds": round(sum(times), 2),
                "average_seconds": round(sum(times) / len(times), 2),
                "min_seconds": round(min(times), 2),
                "max_seconds": round(max(times), 2)
            }
    
    # ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")


async def async_worker(endpoints: list[str], count: int) -> dict[str, list[float]]:
    # send `count` concurrent requests to each endpoint
    overall_stats: dict[str, list[float]] = {ep: [] for ep in endpoints}

    async with httpx.AsyncClient(timeout=60) as client:
        async def make_task(ep: str):
            if ep == "missions":
                url = f"{BASE}/api/missions/judge"
            elif ep == "chatbot":
                url = f"{BASE}/api/vet/chat"
            else:
                url = f"{BASE}/api/{ep}/analyze"
            payload = build_payload(ep)
            start = time.time()
            r = await client.post(url, json=payload)
            elapsed = time.time() - start
            if r.status_code != 200:
                print(f"error ({ep})", r.status_code, r.text)
            return ep, elapsed

        tasks = []
        for ep in endpoints:
            for _ in range(count):
                tasks.append(make_task(ep))

        results = await asyncio.gather(*tasks)
        for ep, elapsed in results:
            overall_stats[ep].append(elapsed)

    # print per-endpoint summaries
    for ep, times in overall_stats.items():
        print(f"\nEndpoint '{ep}':")
        print_summary(times, "async")
    
    return overall_stats


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
    args = parser.parse_args()
    
    mode = "async" if args.use_async else "sync"
    
    if args.use_async:
        results = asyncio.run(async_worker(args.endpoints, args.count))
    else:
        results = sync_worker(args.endpoints, args.count)
    
    # save results if output path is specified
    if args.output:
        save_results(results, mode, args.output)


if __name__ == "__main__":
    main()

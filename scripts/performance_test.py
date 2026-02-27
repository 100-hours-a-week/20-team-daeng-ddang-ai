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

import requests
import httpx

BASE = "http://localhost:8000"

def build_payload(endpoint: str):
    if endpoint == "missions":
        return {
            "analysis_id": "test",
            "walk_id": "walk1",
            "missions": [
                {"mission_id": "m1", "mission_type": "sit", "video_url": "http://example.com/video.mp4"}
            ]
        }
    elif endpoint == "face":
        return {"video_url": "http://example.com/video.mp4"}
    elif endpoint == "healthcare":
        return {"video_url": "http://example.com/walk.mp4", "dog_id": "123"}
    elif endpoint == "chatbot":
        # simple chat payload
        return {"conversation_id": "conv1", "message": {"content": "Hello"}}
    else:
        raise ValueError(endpoint)


def sync_worker(endpoints: list[str], count: int):
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
                print(f"error ({ep})", r.status_code, r.text)

    # print per-endpoint summaries
    for ep, times in overall_stats.items():
        print(f"\nEndpoint '{ep}':")
        print_summary(times, "sync")


def print_summary(times, mode: str):
    count = len(times)
    total = sum(times)
    print(f"{mode} results: {count} requests")
    print(f"  total   : {total:.2f}s")
    print(f"  average : {total/count:.2f}s")
    print(f"  min      : {min(times):.2f}s")
    print(f"  max      : {max(times):.2f}s")


async def async_worker(endpoints: list[str], count: int):
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
    args = parser.parse_args()
    if args.use_async:
        asyncio.run(async_worker(args.endpoints, args.count))
    else:
        sync_worker(args.endpoints, args.count)


if __name__ == "__main__":
    main()

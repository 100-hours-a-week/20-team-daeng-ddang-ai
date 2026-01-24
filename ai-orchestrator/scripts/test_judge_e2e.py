# scripts/test_judge_e2e.py
import os, sys, time, requests
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 테스트 대상 API 엔드포인트
# 로컬 테스트 시: "http://localhost:8000/api/missions/judge"
# 원격 서버 테스트 시: "http://<SERVER_IP>:8000/api/missions/judge"
API_URL = os.getenv("TEST_API_URL", "http://localhost:8000/api/missions/judge")

# 메인 테스트 함수
def test_judge():
    # 테스트 케이스 정의 (S3 URL 사용)
    test_cases = [
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/down.mp4", "mission": "DOWN"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/jump.mp4", "mission": "JUMP"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/paw.mp4", "mission": "PAW"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/sit_small.mp4", "mission": "SIT"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/spin.mp4", "mission": "TURN"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/IMG_6135.mov", "mission": "DOWN"}, #실패케이스
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/IMG_6137.mov", "mission": "DOWN"}, #실패케이스
        # {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/DOWN_02.mp4", "mission": "DOWN"},
        # {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/DOWN_03.mp4", "mission": "DOWN"},
        # {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/DOWN_04.mp4", "mission": "DOWN"},
        # {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/DOWN_05.mp4", "mission": "DOWN"},
        # {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/DOWN_06.mp4", "mission": "DOWN"},
        # {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/JUMP_02.mp4", "mission": "JUMP"},
        # {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/JUMP_03.mp4", "mission": "JUMP"},
        # {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/JUMP_04.mp4", "mission": "JUMP"},
        # {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/JUMP_05.mp4", "mission": "JUMP"},
        # {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/JUMP_06.mp4", "mission": "JUMP"},
        # {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/PAW_02.mp4", "mission": "PAW"},
        # {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/PAW_03.mp4", "mission": "PAW"},
        # {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/PAW_04.mp4", "mission": "PAW"},
        # {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/PAW_05.mp4", "mission": "PAW"},
        # {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/PAW_06.mp4", "mission": "PAW"},
        # {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/SIT_02.mp4", "mission": "SIT"},
        # {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/SIT_03.mp4", "mission": "SIT"},
        # {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/SIT_04.mp4", "mission": "SIT"},
        # {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/SIT_05.mp4", "mission": "SIT"},
        # {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/SIT_06.mp4", "mission": "SIT"},
        # {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/TURN_02.mp4", "mission": "TURN"},
        # {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/TURN_03.mp4", "mission": "TURN"},
        # {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/TURN_04.mp4", "mission": "TURN"},
        # {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/TURN_05.mp4", "mission": "TURN"},
        # {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/TURN_06.mp4", "mission": "TURN"}
    ]

    print(f"[Test] Starting sequential test for {len(test_cases)} videos...\n")
    print(f"[Target] {API_URL}\n")

    total_start_time = time.time()

    # 각 테스트 케이스 순회
    for i, case in enumerate(test_cases):
        video_url = case["url"]
        mission_type = case["mission"]
        
        print(f"=== [Test Case {i+1}] Mission: {mission_type} ===")
        print(f"URL: {video_url}")

        # 요청 페이로드 구성
        payload = {
            "analysis_id": f"test-analysis-{i+1}",
            "walk_id": f"test-walk-{i+1}",
            "missions": [
                {
                    "mission_id": f"m_{i+1}",
                    "mission_type": mission_type,
                    "video_url": video_url
                }
            ]
        }

        print(f"Sending Request...", end=" ", flush=True)
        try:
            # API 호출 및 시간 측정
            start_time = time.time()
            response = requests.post(API_URL, json=payload)
            response.raise_for_status() # 4xx/5xx 에러 발생 시 예외 발생
            elapsed = time.time() - start_time
            
            # 응답 파싱
            result = response.json()
            mission_res = result["missions"][0]
            status = "SUCCESS" if mission_res["success"] else "FAIL"
            
            print(f"DONE ({elapsed:.2f}s)")
            print(f"Result: {status} (Confidence: {mission_res['confidence']:.2f})")
            
            # Reason 확인
            reason_text = (mission_res.get("reason") or "").strip()
            if not reason_text:
                reason_text = "[서버에서 Reason 반환 안함]"

            print(f"Reason: {reason_text}")
            
        except requests.exceptions.HTTPError as e:
            print(f"FAILED (HTTP {response.status_code})")
            print(f"Response: {response.text}")
        except Exception as e:
            print(f"ERROR: {e}")
        
        print("-" * 50)
        # Gemini API Rate Limit 방지를 위한 지연
        time.sleep(1)

    total_elapsed = time.time() - total_start_time
    print(f"\n[Test Completed] Total Time: {total_elapsed:.2f}s")

if __name__ == "__main__":
    test_judge()
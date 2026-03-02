#!/usr/bin/env python
"""Direct chatbot endpoint testing script.

Tests the chatbot functionality at different levels:
1. Direct chatbot-service health check
2. Direct chatbot-service chat endpoint
3. Via ai-orchestrator /api/vet/chat endpoint
"""

import requests
import json
import sys

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def test_chatbot_service_health():
    """Test if chatbot-service is running on port 8300"""
    print(f"\n{YELLOW}1. Testing chatbot-service health (port 8300)...{RESET}")
    try:
        r = requests.get("http://localhost:8300/health", timeout=5)
        if r.status_code == 200:
            print(f"{GREEN}✓ chatbot-service is running{RESET}")
            print(f"  Response: {r.json()}")
            return True
        else:
            print(f"{RED}✗ chatbot-service returned {r.status_code}{RESET}")
            return False
    except requests.ConnectionError:
        print(f"{RED}✗ Cannot connect to chatbot-service on port 8300{RESET}")
        print(f"  Make sure chatbot-service is running: python chatbot-service/run.py")
        return False
    except Exception as e:
        print(f"{RED}✗ Error: {e}{RESET}")
        return False


def test_chatbot_service_direct():
    """Test chatbot-service /api/vet/chat endpoint directly"""
    print(f"\n{YELLOW}2. Testing chatbot-service /api/vet/chat endpoint directly...{RESET}")
    
    payload = {
        "dog_id": 1,
        "conversation_id": "direct-test-1",
        "message": {
            "role": "user",
            "content": "우리 강아지가 자주 물을 마셔요"
        },
        "history": []
    }
    
    try:
        r = requests.post(
            "http://localhost:8300/api/vet/chat",
            json=payload,
            timeout=30
        )
        if r.status_code == 200:
            print(f"{GREEN}✓ chatbot-service /api/vet/chat responded{RESET}")
            response = r.json()
            print(f"  Status: {r.status_code}")
            print(f"  Answer: {response.get('answer', 'N/A')[:100]}...")
            return True
        else:
            print(f"{RED}✗ chatbot-service returned {r.status_code}{RESET}")
            print(f"  Response: {r.text}")
            return False
    except requests.Timeout:
        print(f"{RED}✗ Request timed out (>30s){RESET}")
        return False
    except requests.ConnectionError:
        print(f"{RED}✗ Cannot connect to chatbot-service on port 8300{RESET}")
        return False
    except Exception as e:
        print(f"{RED}✗ Error: {e}{RESET}")
        return False


def test_orchestrator_chatbot():
    """Test ai-orchestrator /api/vet/chat endpoint"""
    print(f"\n{YELLOW}3. Testing ai-orchestrator /api/vet/chat endpoint...{RESET}")
    
    payload = {
        "dog_id": 1,
        "conversation_id": "orchestrator-test-1",
        "message": {
            "role": "user",
            "content": "우리 강아지가 자주 물을 마셔요"
        },
        "history": []
    }
    
    try:
        r = requests.post(
            "http://localhost:8000/api/vet/chat",
            json=payload,
            timeout=30
        )
        if r.status_code == 200:
            print(f"{GREEN}✓ ai-orchestrator /api/vet/chat responded{RESET}")
            response = r.json()
            print(f"  Status: {r.status_code}")
            print(f"  Answer: {response.get('answer', 'N/A')[:100]}...")
            if response.get('error_code'):
                print(f"{YELLOW}  Warning - error_code: {response.get('error_code')}{RESET}")
            return True
        else:
            print(f"{RED}✗ ai-orchestrator returned {r.status_code}{RESET}")
            print(f"  Response: {r.text}")
            return False
    except requests.Timeout:
        print(f"{RED}✗ Request timed out (>30s){RESET}")
        return False
    except requests.ConnectionError:
        print(f"{RED}✗ Cannot connect to ai-orchestrator on port 8000{RESET}")
        return False
    except Exception as e:
        print(f"{RED}✗ Error: {e}{RESET}")
        return False


def main():
    print(f"\n{YELLOW}=== Chatbot Functionality Test ==={RESET}")
    print("This script tests if chatbot is working at different levels.")
    
    # Test 1: chatbot-service health
    health_ok = test_chatbot_service_health()
    
    if not health_ok:
        print(f"\n{RED}Chatbot service is not running. Cannot continue.{RESET}")
        sys.exit(1)
    
    # Test 2: chatbot-service direct
    direct_ok = test_chatbot_service_direct()
    
    # Test 3: via orchestrator
    orchestrator_ok = test_orchestrator_chatbot()
    
    # Summary
    print(f"\n{YELLOW}=== Summary ==={RESET}")
    print(f"chatbot-service health: {GREEN if health_ok else RED}{'✓ OK' if health_ok else '✗ FAILED'}{RESET}")
    print(f"chatbot-service /api/vet/chat: {GREEN if direct_ok else RED}{'✓ OK' if direct_ok else '✗ FAILED'}{RESET}")
    print(f"ai-orchestrator /api/vet/chat: {GREEN if orchestrator_ok else RED}{'✓ OK' if orchestrator_ok else '✗ FAILED'}{RESET}")
    
    if direct_ok and orchestrator_ok:
        print(f"\n{GREEN}All tests passed! Chatbot is working correctly.{RESET}")
        return 0
    else:
        print(f"\n{RED}Some tests failed. Check the output above for details.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

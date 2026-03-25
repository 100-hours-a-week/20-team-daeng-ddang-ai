const fs = require('fs');
const path = require('path');

const BASE_URL = "GCPID";
const CHAT_URL = `${BASE_URL}/api/vet/chat`;
const HEALTH_URL = `${BASE_URL}/api/healthcare/analyze`;

const LOG_FILE = path.join(__dirname, 'test_results.log');

// 로그 파일 초기화 (기존 내용 삭제)
fs.writeFileSync(LOG_FILE, `Test started at ${new Date().toLocaleString()}\n${'='.repeat(50)}\n\n`);

// 1. 챗봇 상담 페이로드
const chatPayload = {
  "dog_id": 123,
  "conversation_id": "conv_abc_123",
  "message": {
    "role": "user",
    "content": "우리 강아지가 밥을 안 먹어요."
  },
  "history": [
    { "role": "user", "content": "어제는 산책을 잘 했어요." },
    { "role": "assistant", "content": "그렇군요. 다른 증상은 없나요?" }
  ],
  "image_url": null,
  "user_context": {
    "dog_age_years": 8.5,
    "dog_weight_kg": 5.0,
    "breed": "포메라니안"
  }
};

// 2. 헬스케어 분석 페이로드
const healthPayload = {
  "video_url": "s3_video_url",
  "dog_id": 123
};

/**
 * 공통 요청 함수
 */
async function sendRequest(name, url, body) {
  const startTime = Date.now();
  console.log(`[${name}] Sending to ${url}...`);
  
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    
    const data = await response.json();
    const duration = Date.now() - startTime;
    
    console.log(`[${name}] Received in ${duration}ms`);

    // 로그 파일에 기록 및 콘솔 출력
    const logEntry = {
      timestamp: new Date().toISOString(),
      name,
      url,
      duration: `${duration}ms`,
      status: response.status,
      response: data
    };
    const logString = JSON.stringify(logEntry, null, 2);
    fs.appendFileSync(LOG_FILE, logString + "\n---\n");
    console.log(`\n[${name}] Response Details:\n${logString}\n`);

    if (data.error_code) {
      console.warn(`[${name}] Error returned:`, data.error_code);
    }
  } catch (error) {
    const errorEntry = {
      timestamp: new Date().toISOString(),
      name,
      url,
      error: error.message
    };
    const errorString = JSON.stringify(errorEntry, null, 2);
    fs.appendFileSync(LOG_FILE, errorString + "\n---\n");
    console.error(`\n[${name}] Request FAILED:\n${errorString}\n`);
  }
}

async function runTest() {
  console.log("Starting concurrency test (Total 12 requests: 6 Chat + 6 Health)...");
  
  const tasks = [];

  // 챗봇 요청 6개 추가
  for (let i = 1; i <= 6; i++) {
    tasks.push(sendRequest(`Chat-${i}`, CHAT_URL, chatPayload));
  }

  // // 헬스케어 요청 6개 추가
  // for (let i = 1; i <= 6; i++) {
  //   tasks.push(sendRequest(`Health-${i}`, HEALTH_URL, healthPayload));
  // }

  // 12개 요청 병렬 실행
  await Promise.all(tasks);
  
  console.log("\nAll 12 requests have been processed.");
}

runTest();

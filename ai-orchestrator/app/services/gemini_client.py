import os
import json
import time
import tempfile
import requests
import google.generativeai as genai
from typing import Dict, Any, Optional

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class GeminiClient:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in environment variables.")

        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    def _download_video(self, video_url: str) -> str:
        try:
            with tempfile.NamedTemporaryFile(delete = False, suffix = ".mp4") as tmp_file:
                response = requests.get(video_url, stream = True, timeout = 30)
                response.raise_for_status()

                for chunk in response.iter_content(chunk_size = 8192):
                    tmp_file.write(chunk)

                return tmp_file.name

        except Exception as e:
            raise RuntimeError(f"Failed to download video from {video_url}: {e}")

    def call_gemini_judge(self, video_url: str, prompt: str) -> Dict[str, Any]:
        video_path = None
        upload_file = None

        try:
            video_path = self._download_video(video_url)
            upload_file = genai.upload_file(video_path)
            
            start_time = time.time()
            timeout_seconds = 60

            while upload_file.state.name == "PROCESSING":
                if time.time() - start_time > timeout_seconds:
                    raise TimeoutError("Gemini file processing timed out.")
                    
                time.sleep(1)
                upload_file = genai.get_file(upload_file.name)
            
            if upload_file.state.name == "FAILED":
                raise RuntimeError("Gemini file upload failed processing.")

            response = self.model.generate_content(
                [upload_file, prompt],
                generation_config = {"response_mime_type": "application/json"}
            )
            
            try:
                result = json.loads(response.text)
                return result
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse JSON from Gemini response: {response.text}")

        except Exception as e:
            raise e
            
        finally:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
            
            if upload_file:
                try:
                    genai.delete_file(upload_file.name)
                except:
                    pass

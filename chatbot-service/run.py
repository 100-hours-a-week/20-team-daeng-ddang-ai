import uvicorn
import os

if __name__ == "__main__":
    # Chatbot Service runs on port 8300 by default
    port = int(os.getenv("PORT", 8300))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

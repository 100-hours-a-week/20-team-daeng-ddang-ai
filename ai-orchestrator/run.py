import uvicorn
import os

if __name__ == "__main__":
    # AI Orchestrator runs on port 8000 by default
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False, workers=1)
import uvicorn
import os

if __name__ == "__main__":
    # AI Orchestrator runs on port 8000 by default
    port = int(os.getenv("PORT", 8000))
    # 'reload=True' is convenient for dev, but usually off in strict prod. 
    # Can be toggled with an env var or kept simple for now.
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

import os
import uvicorn


if __name__ == "__main__":
    # Development launcher with autoreload
    port = int(os.getenv("PORT", "8200"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True, workers=1)

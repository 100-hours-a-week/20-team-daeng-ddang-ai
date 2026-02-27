import uvicorn
import os

if __name__ == "__main__":
    # Development launcher (face service already uses reload=True by default)
    port = int(os.getenv("PORT", 8100))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

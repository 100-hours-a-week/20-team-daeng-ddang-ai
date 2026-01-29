
import uvicorn
import os

if __name__ == "__main__":
    # Face Service runs on port 8100 by default (as per plan)
    port = int(os.getenv("PORT", 8100))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

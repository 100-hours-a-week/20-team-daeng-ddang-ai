import uvicorn
import os

if __name__ == "__main__":
    host = os.getenv("FACE_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("FACE_SERVER_PORT", "8100"))
    
    print(f"Starting Face Server on {host}:{port}...")
    uvicorn.run("app.face_server:app", host=host, port=port, reload=True)

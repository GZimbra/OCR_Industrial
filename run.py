import os
import uvicorn

if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    uvicorn.run("htr_local.web:app", host="127.0.0.1", port=8000, reload=False)

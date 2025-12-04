from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import sys

# Add parent directory to path to import inference.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import ActionRecognizer


from contextlib import asynccontextmanager

# Global Model
recognizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global recognizer
    # Assuming model is in root directory
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "best_model.pth")
    recognizer = ActionRecognizer(model_path)
    print("Model loaded.")
    yield
    # Shutdown logic (if any)

from fastapi.staticfiles import StaticFiles

app = FastAPI(lifespan=lifespan)

# Mount static files
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Use absolute path for OpenPose
        abs_temp_file = os.path.abspath(temp_file)
        result = recognizer.predict_video(abs_temp_file)
        return result
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import requests
from fastapi.middleware.cors import CORSMiddleware
import base64

app = FastAPI()

# ✅ CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Colab API URL — update this if ngrok URL changes
COLAB_API_URL = "https://0fef-34-169-80-20.ngrok-free.app"

# ✅ Input format
class OutfitRequest(BaseModel):
    age: int
    gender: str
    occasion: str
    season: str
    style: str
    color: str

# ✅ POST route to connect to Colab backend
@app.post("/generate/")
async def generate_outfit(data: OutfitRequest):
    try:
        # 🔁 Forward request to Colab API
        response = requests.post(f"{COLAB_API_URL}/generate/", json=data.dict())
        
        if response.status_code != 200:
            return JSONResponse(status_code=500, content={"error": "Failed to fetch from Colab"})

        # ✅ Expecting JSON with base64 image and description
        result = response.json()

        return {
            "description": result["description"],
            "image_base64": result["image_base64"]
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


#colab link https://colab.research.google.com/drive/1Q-Ggn3qoefa5pjirBVSspNRer6Qe1SCU

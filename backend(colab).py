#as the project requires gpu for image generation ( stable diffusion ), i have used google colab
#colab link https://colab.research.google.com/drive/1Q-Ggn3qoefa5pjirBVSspNRer6Qe1SCU



!pip install -q fastapi uvicorn pyngrok transformers accelerate bitsandbytes diffusers scipy safetensors
!pip install -q "huggingface_hub[cli]"

from huggingface_hub import login
login(token="enter you huggingface token here")

from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline
import torch

model_id = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map="auto"
)

sd_model = "stabilityai/stable-diffusion-2"
pipe = StableDiffusionPipeline.from_pretrained(
    sd_model,
    torch_dtype=torch.float16,
    revision="fp16"
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")


from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
import base64
from fastapi.responses import JSONResponse
from io import BytesIO
import traceback

app = FastAPI()

class FashionRequest(BaseModel):
    age: int
    gender: str
    occasion: str
    season: str
    style: str
    color: str

def generate_outfit_description(age, gender, occasion, season, style, color):
    prompt = (
        f"Instruction: Suggest a fashionable outfit for a {age}-year-old {gender} "
        f"attending a {occasion} in the {season} season. The style should be {style} and the color theme should be {color}. "
        f"Include details for clothing items, accessories, and footwear. Make it sound stylish and appealing.\n\nResponse:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Response:")[-1].strip()

@app.post("/generate/")
async def generate_outfit(data: FashionRequest):
    try:
        description = generate_outfit_description(
            data.age, data.gender, data.occasion, data.season, data.style, data.color
        )

        safe_prompt = description[:200]  # Truncate to avoid errors
        print("🧵 Prompt for SD:", safe_prompt)

        image = pipe(safe_prompt).images[0]

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return JSONResponse(content={
            "description": description,
            "image_base64": encoded_image
        })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


!pip install pyngrok --quiet
from pyngrok import ngrok

!ngrok config add-authtoken "enter you ngrok token here"

import uvicorn
import threading

def run():
    uvicorn.run(app, host="0.0.0.0", port=8000)

thread = threading.Thread(target=run)
thread.start()

public_url = ngrok.connect(8000)
print("✅ Your Colab FastAPI is live at:", public_url)

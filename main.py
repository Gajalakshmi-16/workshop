from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import os
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# --- Initialize the FastAPI lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan events"""
    try:
        print("🚀 FastAPI starting up...")
        print("✅ Loading lightweight Vision-Language model for image analysis (CPU mode)...")

        # --- Small model ---
        global model, feature_extractor, tokenizer
        model_id = "nlpconnect/vit-gpt2-image-captioning"

        feature_extractor = ViTImageProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = VisionEncoderDecoderModel.from_pretrained(model_id)

        print("✅ Model loaded successfully on CPU!")

    except Exception as e:
        print(f"❌ Error initializing model: {e}")

    yield
    print("🛑 FastAPI shutting down...")

# --- App Setup ---
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="reverser-base-template/templates")
app.mount("/static", StaticFiles(directory="reverser-base-template/static"), name="static")

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "user": "admin"})

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return templates.TemplateResponse("settings.html", {"request": request, "user": "admin"})

@app.get("/default-instructions", response_class=HTMLResponse)
async def default_instructions_page(request: Request):
    return templates.TemplateResponse("default_instructions.html", {"request": request, "user": "admin"})

@app.get("/rules", response_class=HTMLResponse)
async def rules_page(request: Request):
    return templates.TemplateResponse("rules.html", {"request": request, "user": "admin"})

# --- Vision Analysis Function ---
async def analyze_image(image_bytes: bytes, user_query: str) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

        output_ids = model.generate(pixel_values, max_length=50, num_beams=4)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if "dust" in user_query.lower() or "dirty" in user_query.lower():
            caption += " | Note: You asked about cleanliness — please review the image carefully."

        return caption
    except Exception as e:
        return f"Error analyzing image: {e}"

# --- Chat Endpoint ---
@app.post("/chat")
async def chat(user_query: str = Form(""), image: UploadFile = File(None)):
    if image and image.filename:
        image_bytes = await image.read()
        analysis_result = await analyze_image(image_bytes, user_query)
        return JSONResponse({"response": analysis_result})
    elif user_query:
        return JSONResponse({"response": f"🧠 Text-only mode active. You asked: '{user_query}'."})
    else:
        raise HTTPException(status_code=400, detail="No query or image provided.")

# --- Run App ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8002))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)

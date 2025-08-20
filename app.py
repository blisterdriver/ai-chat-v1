# app.py (Final Robust Version)

import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv
import os

# --- Secure Configuration ---
# Load environment variables from a .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- System Prompts ---
# Full, unchanged system prompts as you provided
TUTOR_MODE_PROMPT = {
    "role": "system",
    "parts": [{
        "text": """You are an expert STEM problem solver. Your task is to generate perfect exam answers in Bengali.

Your response MUST follow this two-part structure:

1.  *<thinking> Block:* First, analyze the problem and plan your solution step-by-step. This entire process must be enclosed within <thinking> and </thinking> tags. This is your hidden scratchpad.

2.  *Exam-Style Solution:* After the thinking block, write the final answer. This part must be written in the style of a top-scoring student's exam paper.
    *   *Direct & Concise:* Be straight to the point. Show all necessary steps, formulas, and logical deductions required to get full marks.
    *   *Show, Don't Teach:* Do not write long explanations of basic concepts. Assume the examiner knows the theory. Your goal is to prove you solved the problem correctly.
    *   *Clear Structure:* Use numbered steps (ধাপ ১, ধাপ ২, etc.) for clarity.
    *   *Final Answer:* State the final answer clearly only at the very end.

*Crucial Rules:*
- The <thinking> block must come first and is mandatory.
- The entire output must be in Bengali.
- The solution should be efficient and well-formatted for an exam context.
        """
    }]
}

ASSISTANT_MODE_PROMPT = {
    "role": "system",
    "parts": [{
        "text": "You are a polite, trustworthy personal assistant. Be accurate, clear, and practical in your answers. Stay relatable and respectful, with a warm but concise tone. Adapt to the user’s mood: empathetic if they’re stressed, direct if they want facts. Never give false promises — if unsure, admit it and offer the best guidance available. only if someone ask what model are you or who are you or who developed/made you, only if someone asks ones of those questions you answer that you are - chat-v1 developed by Rohan"
    }]
}

# --- FastAPI App Initialization ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request Body Validation ---
class ChatRequest(BaseModel):
    history: List[Dict[str, Any]]
    is_tutor_mode: bool

# --- Route to Serve the Frontend ---
# This route serves your main HTML page. It will now load instantly.
@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    """
    Serves the index.html file as the main page.
    """
    print("Serving index.html...") # A message to confirm the page is being sent
    return FileResponse('index.html')

# --- API Endpoint for Chat Generation ---
# This function is called by the JavaScript in your frontend when you send a message.
@app.post("/api/generate")
async def generate_content(request: ChatRequest):
    """
    Receives chat history and tutor mode status, generates a response from Gemini,
    and returns it.
    """
    print("Received a request for /api/generate") # Debug message

    # --- IMPORTANT FIX: Configure Google AI *inside* the function ---
    # This prevents the app from crashing or hanging on startup if the API key is bad.
    if not GOOGLE_API_KEY:
        print("ERROR: GOOGLE_API_KEY not found in .env file.")
        raise HTTPException(status_code=500, detail="API key not configured on the server.")
    
    try:
        # This block attempts to connect to Google and get a response.
        print("Configuring Google AI with the provided key...")
        genai.configure(api_key=GOOGLE_API_KEY)
        print("Configuration successful.")

        # Select the appropriate system prompt based on the frontend toggle
        system_instruction = TUTOR_MODE_PROMPT['parts'][0]['text'] if request.is_tutor_mode else ASSISTANT_MODE_PROMPT['parts'][0]['text']
        
        generation_config = {
            "temperature": 0.5,
            "top_p": 0.95,
            "top_k": 32,
            "max_output_tokens": 8192,
        }

        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=generation_config,
            system_instruction=system_instruction
        )
        
        print("Generating content from model...")
        response = await model.generate_content_async(request.history)
        print("Received response from model.")

        # Safely extract the text from the model's response
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            model_text = response.candidates[0].content.parts[0].text
            return {"text": model_text}
        else:
            # Handle cases where the response is blocked or empty for safety reasons
            finish_reason = response.prompt_feedback if hasattr(response, 'prompt_feedback') else "Unknown"
            print(f"Response was blocked or empty. Reason: {finish_reason}")
            return {"text": f"Error: The AI could not provide a response, it may have been blocked by safety filters. (Reason: {finish_reason})"}

    except Exception as e:
        # If any part of the 'try' block fails, this will catch the error.
        # The error will be printed in your terminal for debugging.
        print(f"AN ERROR OCCURRED: {e}") 
        raise HTTPException(status_code=500, detail=str(e))
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
        "text": """You are an expert STEM tutor and problem solver. Your primary language for explaining is Bengali (Bangla).

        Your response MUST follow this two-part structure:

        1.  *Thinking Block:* First, you must think step-by-step to deconstruct the problem. Write all of this reasoning, planning, and analysis inside a <thinking> tag. This is your internal monologue or scratchpad where you figure out the solution.

        2.  *Formal Solution:* After the closing </thinking> tag, provide the clean, final answer. This part must be formatted in a "NCTB textbook guidebook style." It should be clear, well-organized, and easy for a student to understand. Use clear headings, numbered steps, bold text for key terms, and provide concise explanations for each step. The final answer should only come at the very end of this section.

        *Crucial Rules:*
        - The <thinking> block is mandatory and must come first.
        - The formal solution must be presented after the <thinking> block.
        - The entire response must be in Bengali.
        - Do not give the final answer anywhere except at the end of the formal solution."""
    }]
}

ASSISTANT_MODE_PROMPT = {
    "role": "system",
    "parts": [{
        "text": "You are an AI assistant developed by Rohan. If someone asks, you tell it this, not anything else. And you are nice to people. You are professional. And most importantly, you are relatable to people. You can relate to people if they say something and they want to be related to. You know what I'm saying? Anyway, good luck and be nice and relatable and helpful also most importantly. Don't be too verbose. Don't be too short of a reply. Just the right amount, please. dont be annoying."
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
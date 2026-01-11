import openai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
from helper import *
from huggingface_hub import InferenceClient


# Request model for the API endpoint
class DiagnosticRequest(BaseModel):
    user_choice: str # This is all you actually need from the frontend


app = FastAPI(title="FixIT.core Diagnostic API")

# Define which "origins" (URLs) are allowed to talk to this API
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, GET, OPTIONS, etc.
    allow_headers=["*"],  # Allows Custom Headers
)

# # Enable CORS for frontend integration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# In-memory session (Note: In production, use a database or Redis)
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=HF_TOKEN
)
agent = DiagnosticAgent(client)

@app.post("/diagnostic/next", response_model=AgentResponse)
async def next_step(request: DiagnosticRequest):
    """Process a user choice and get the next diagnostic step."""
    return agent.process_step(request.user_choice)

@app.post("/diagnostic/rewind")
async def rewind():
    """Undo the last step in the diagnostic history."""
    success = agent.handle_correction()
    if success:
        return {"status": "success", "message": "State rewound successfully."}
    return {"status": "error", "message": "Cannot rewind any further."}

@app.get("/diagnostic/reset")
async def reset():
    """Reset the diagnostic session to the initial state."""
    global agent,client
    agent = DiagnosticAgent(client)
    return {"status": "reset", "message": "Session restarted."}

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
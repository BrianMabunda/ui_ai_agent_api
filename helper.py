import pandas as pd
import numpy as np
import lancedb
import openai
from pydantic import BaseModel
from pydantic import Field
from typing import List
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PICKLE_PATH = os.path.join(BASE_DIR, "cleaned_embeddings_dataframe.pkl")
GLOBAL_DFRAME = pd.read_pickle(PICKLE_PATH)

from sentence_transformers import SentenceTransformer

# METHOD A: Using sentence-transformers (Best for Hugging Face Spaces)
# This loads the model into the Space's memory so it doesn't need your local PC.
# Note: nomic-ai/nomic-embed-text-v1 is the standard Hugging Face equivalent.
class LocalEmbedder:
    def __init__(self):
        # This will download the model to the HF Space the first time it runs
        self.model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)

    def create_embeddings(self, text):
        # nomic-embed requires a prefix for certain tasks
        # For search/retrieval: "search_query: " or "search_document: "
        prefix = "search_query: " 
        embedding = self.model.encode(prefix + text)
        return embedding.tolist()




def create_embeddings(text):
    embedder = LocalEmbedder()
    return embedder.create_embeddings(text)

# def create_embeddings(text):
#     client = openai.OpenAI(
#     base_url="http://localhost:11434/v1", 
#     api_key="ollama"  # Placeholder key
# )
#     response = client.embeddings.create(
#         input=text,
#         model="nomic-embed-text"
#     )
#     # print(response)
#     embeddings = response.data[0].embedding
#     return embeddings


class ContextExtractor:
    def __init__(self, prompt):
        

        self.dframe = GLOBAL_DFRAME.copy()
        self.prompt = prompt
        # CRITICAL FIX: Ensure create_embeddings function is defined globally or imported
        self.question_embeddings = create_embeddings(prompt) 
        self.context = ""
        self.db = lancedb.connect("./lancedb_store")
        self.table = self.db.create_table("my_table", data=self.dframe,mode="overwrite") 

    # This helper method is useful for clarity, but not strictly needed if inlining the apply
    def change_prompt(self, promt):
        self.prompt=promt
    
    def get_context(self):
        

        top_k_test = self.table.search(self.question_embeddings).limit(5).to_pandas()
        
        # Cleaner way to build the final context string
        context_list = top_k_test['cleaned_content'].tolist()
        self.context = "\n\n".join(context_list) # Join with double newline for readability
        
        # Removed the redundant self.dframe['distance']=0 line
        
        return self.context

class RAGAgent():
    user_prompt =""
    messages=[]
    context=""
    response=""
    

    def __init__(self,client):
        self.client=client
    
    def change_prompt(self,prompt):
        self.user_prompt=prompt

    def update_message(self):
        system_prompt = """
                        ### Role
                        You are a Senior Network Systems Engineer with expert-level knowledge of networking hardware, specifically switches (L2/L3), routers, and physical infrastructure. Your goal is to provide precise, technical, and actionable troubleshooting advice.

                        ### Context Usage
                        - You will be provided with retrieved technical documentation (Context). 
                        - Always prioritize the information found in the Context.
                        - If the Context does not contain the answer, explicitly state: "I do not have enough specific documentation to answer this accurately." Do not make up hardware specifications or commands.

                        ### Response Guidelines
                        1. **Clarity**: Use technical terminology (e.g., VLAN tagging, SFP+ compatibility, backplane capacity) correctly.
                        2. **Safety**: When suggesting hardware changes, include necessary safety warnings (e.g., static discharge, power cycling).
                        3. **Troubleshooting Steps**: Provide answers in a logical, step-by-step format (Step 1: Physical Check, Step 2: Configuration, etc.).
                        4. **Commands**: If applicable, provide specific CLI commands (Cisco IOS, Juniper Junos, etc.) within code blocks.

                        ### Constraints
                        - Do not mention that you are an AI or that you are using retrieved documents.
                        - Maintain a professional, technical, and helpful tone.
                        """

        formatted_context = f"--- RETRIEVED TECHNICAL DOCUMENTATION ---\n{self.context}\n----------------------------------------"
        
        # 2. Refine the User Prompt to include the context
        # This clearly separates the 'evidence' from the 'question'
        user_prompt_with_context = f"""
        
            Use the following context to answer the question. 
            If the answer is not in the context, say you do not know.

            CONTEXT:
            {formatted_context}

            QUESTION:
            {self.user_prompt}
        """

        # 3. Final Messages List
        self.messages = [
            {
                "role": "system", 
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": user_prompt_with_context
            }
        ]

    def pre_run(self):
        context_finder=ContextExtractor(self.user_prompt)
        context_finder.change_prompt(self.user_prompt)
        self.context=context_finder.get_context()
    
    def run(self):
        self.pre_run()
        self.update_message()

        response = self.client.chat.completions.create( 
          
            # Define the messages. Remember this is meant to be a user prompt!
            model="meta-llama/Llama-3.2-1B-Instruct", # Use this exact string
            messages=self.messages,
            # Keep responses creative
            max_tokens=500, 
        )
        self.response=response.choices[0].message.content

        return self.response


class AgentResponse(BaseModel):
    current_step: str = Field(alias="status") # Maps 'status' from LLM to 'current_step' for UI
    message: str
    options: List[str]
    confidence: str # Changed to str since LLM sends "Low/High" or "0.5" as strings

    class Config:
        populate_by_name = True # Allows us to initialize using 'status'

class DiagnosticAgent:
    def __init__(self,client):
        self.rag_response=""
        self.prompt=""
        self.client  = client
        self.rag_agent=RAGAgent(client)
        self.history = []
    def update_message(self):
        # Keep the system prompt + the last 6 turns of conversation
        if len(self.history) > 7:
            self.history = [self.history[0]] + self.history[-6:]
        system_prompt=f"""
                # ROLE
                You are a User Interface Agent for Network Support. Your job is to take technical troubleshooting data and turn it into a simple, single-step instruction for a customer.

                # INPUT DATA
                Technical Solution: {self.rag_response}

                # OPERATIONAL RULES
                1. MESSAGE: Simplified instruction. Maximum 120 characters. 
                2. OPTIONS: Provide 2-4 buttons for the user to click next (e.g., "It worked", "Still no power"). Max 20 characters per button.
                3. STATUS: Categorize the current stage as "Power", "Cables", "Config", or "Resolved".
                4. CONFIDENCE: Rate how likely this step is to fix the issue as a number from 0 to 100 .
                5. JSON ONLY: Respond only with valid JSON. No markdown backticks.

        """+"""#OUTPUT SCHEMA
            {

            "status": "Power | Cables | Config | Resolved",
            "condifence" :"How close are we to a solution"
            "message": "Short instruction here",
            "options": ["Option 1", "Option 2"]
            }
            """
        # 1. If history is empty, initialize it with the System Prompt
        if not self.history:
            self.history.append({"role": "system", "content": system_prompt})
        else:
            # 2. Update the system prompt with the LATEST RAG info 
            # (This ensures the Agent knows the new technical answer)
            self.history[0] = {"role": "system", "content": system_prompt}

        # 3. Add the User's click to history so the AI "remembers" it
        self.history.append({"role": "user", "content": f"User selected: {self.prompt}"})

    def process_step(self, user_choice):
        self.prompt = user_choice
        self.rag_agent.change_prompt(user_choice)
        self.rag_response = self.rag_agent.run()
        
        # Update the system prompt with new RAG context at history[0]
        self.update_message() 

        completion = self.client.chat.completions.create( 
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=self.history,
            # response_format={"type": "json_object"}, # Try removing if 400 persists
            max_tokens=500, 
        )

        raw_content = completion.choices[0].message.content
        
        # Safety: Strip backticks if the model ignores the "JSON ONLY" rule
        clean_json = raw_content.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)

        # Standardize for FastAPI
        formatted_response = {
            "status": data.get("status", "Unknown"),
            "message": data.get("message", "Instruction missing"),
            "options": data.get("options", ["Restart"]),
            "confidence": str(data.get("confidence", "Medium"))
        }

        # CRITICAL FOR MEMORY: 
        # Add the AI's simplified message to history so it knows what it just said!
        self.history.append({"role": "assistant", "content": formatted_response["message"]})
        
        return formatted_response

    def handle_correction(self):
        # The 'Rewind' Feature: Remove last user action and AI reaction
        if len(self.history) > 2:
            self.history.pop()
            self.history.pop()
        return "State Rewound. Please pick a new option."




import os
import sys
from dotenv import load_dotenv

from fastapi import FastAPI, requests
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chat import get_response
 
 
app = FastAPI(title="BukiBot AI Assistant",
    version="0.1.0",
    description=(  # Description updated for clarity
        "The BukiBot AI Assistant is an AI application that provides neccessary supports for customers."
    ),
    contact={  # Contact details updated for clarity
        "name": "Codar Tech Africa",
        "url": "https://codarhq.com/"
    },
)

# Load the .env file to be able to use the secrets
load_dotenv()
class Chat(BaseModel):
    question: str
 

@app.get("/", summary="Root Endpoint")
def root():
    return {
        "message": "Welcome to the BukiBot APP",
        "details": "The BukiBot AI Assistant is an AI application that provides neccessary supports for customers."
    }
    
@app.post('/api/chat')
def chat_api(chat: Chat):
    response = get_response(chat.question)
    
    response_data = {
            "User Question": chat.question,
            "Response": response
        }
    
    print(response_data)
    
    return JSONResponse(content=response_data)
 
 
if __name__ == '__main__':  
   uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
 

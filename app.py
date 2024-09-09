import os
import sys
from dotenv import load_dotenv

from fastapi import FastAPI, requests
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chat import get_response
 
 
class Chat(BaseModel):
    question: str
 
 
app = FastAPI()

# Load the .env file to be able to use the secrets
load_dotenv()
 
 
@app.post('/api/chat')
def chat_api(chat: Chat):
    response = get_response(chat.question)
    message = {"answer": response}
    return {"Response":message}
 
 
if __name__ == '__main__':  
   app.run()  
 

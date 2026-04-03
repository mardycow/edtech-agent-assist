import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from agent_workflow import run

load_dotenv() 

app = FastAPI(title="EdTech Agent Assist")

class AIResponse(BaseModel):
    category: str
    draft_answer: str
    confidence: float

@app.post("/generate", response_model=AIResponse)
async def generate_assist(user_query: str):
    try:

        response = await run(user_query)
        
        if not response:
            raise HTTPException(status_code=500, detail="API вернул пустой ответ")
        
        return AIResponse(**response)

    except Exception as e:
        print(f"Error details: {e}")
        raise HTTPException(status_code=500, detail="Произошла ошибка при обработке запроса")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
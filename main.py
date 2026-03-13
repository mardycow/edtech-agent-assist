import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI(title="API Prototype")

load_dotenv()

client = OpenAI(
    base_url=os.getenv("LLM_HOST_URL"),
    api_key=os.getenv("LLM_API_KEY")
)

class AIResponse(BaseModel):
    category: str = Field(description="Категория запроса")
    draft_answer: str = Field(description="Черновик ответа оператора")
    confidence: float = Field(description="Уверенность LLM в ответе (0.0 - 1.0)")

SYSTEM_PROMPT = """
Ты — AI-ассистент службы поддержки EdTech платформы. 
Твоя цель: создать черновик ответа на основе вопроса студента.

ПРАВИЛА:
1. Категорию (category) выбирай строго из списка: [Оплата, Техподдержка, Обучение, Прочее].
2. Оценивай уверенность (confidence) числом в диапазоне 0.0 - 1.0: 1.0 если ответ точный, ниже 0.5 если вопрос размытый.
3. Отвечай СТРОГО по этому шаблону в JSON:
{
  "category": "",
  "draft_answer": "Текст ответа",
  "confidence": 
}
"""

@app.post("/generate", response_model=AIResponse)
async def generate_assist(user_query: str, temperature: float = 0.1):
    try:
        response = client.chat.completions.create(
            model="qwen3.5:0.8b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query}
            ],
            temperature=temperature,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        return AIResponse.model_validate_json(content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка LLM: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
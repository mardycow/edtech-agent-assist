import os
from langchain_openai import ChatOpenAI
from langsmith.schemas import Run, Example
from pydantic import BaseModel, Field

class EvaluationVerdict(BaseModel):
    score: float = Field(description="Оценка от 0 до 1")
    reason: str = Field(description="Пояснение оценки")

llm_judge = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("BASE_API_KEY")
)

structured_llm = llm_judge.with_structured_output(EvaluationVerdict)

def planning_judge(run: Run, example: Example) -> dict[str, any]:
    """
    Evaluate the agent's planning (correct tool calls) with LLM-as-a-Judge
    """
    actual = run.outputs.get("tools", [])
    expected = example.outputs.get("tools", [])

    if actual == expected and len(actual) > 0:
        return {
            "key": "planning_logic", 
            "score": 1.0, 
            "comment": "Полное совпадение состава и порядка вызовов"
        }

    return call_llm_judge(actual, expected, run.outputs.get("final_answer"))

def call_llm_judge(actual, expected, answer):
    prompt = f"""
    Оцени логику планирования агента по шкале от 0.0 до 1.0.
    Эталонный план: {expected}
    Что сделал агент: {actual}
    Ответ агента пользователю: "{answer}"

    Задачи в этом бенчмарке ОБЯЗАТЕЛЬНО требуют вызова инструментов.
    
    ТВОЯ ЗАДАЧА:
    1. Оцени состав: вызвал ли агент нужный инструмент из эталона?
    2. Оцени порядок: если в эталоне несколько инструментов, логичен ли порядок вызова в реальности?
    3. Оцени обоснованность уточнений без вызова инструментов.

    РАСПРЕДЕЛЕНИЕ БАЛЛОВ:
    - 0.7-0.9: Нужный инструмент вызван (возможно с полезным дополнением).
    - 0.4-0.6: Нужный инструмент вызван, но порядок странный или много лишнего шума.
    - 0.1-0.3: Инструмент не вызван, агент "ушел" в диалог или уточнение.
    - 0.0: Галлюцинация или абсолютно неверный выбор.
    """

    try:
        response = structured_llm.invoke(prompt)
        
        return {
            "key": "planning_logic", 
            "score": response.score, 
            "comment": response.reason
        }
    except Exception as e:
        print(f"Error in LLM Judge: {e}")
        return {
            "key": "planning_logic", 
            "score": 0.0, 
            "comment": f"Ошибка LLM-судьи: {str(e)}"
        }

def steps_count(run: Run, example: Example) -> dict[str, any]:
    """Calculates the total number of steps (tool calls) taken by the agent."""

    actual_tools = run.outputs.get("tools", []) if run.outputs else []
    step_count = len(actual_tools)
    
    return {"key": "step_count", "score": step_count}
from pydantic import BaseModel, Field
from typing import List, Literal
from annotated_types import Ge, Le
from enum import Enum

class Category(str, Enum):
    SALES = "Продажи"
    FINANCE = "Финансы"
    ACADEMIC = "Учебный процесс"
    SUPPORT = "Техподдержка"
    FEEDBACK = "Обратная связь"
    SPAM = "Спам"
    GREETING = "Приветствие"
    OTHER = "Прочее"

class RouterOutput(BaseModel):
    brief_thought: str = Field(..., description="Краткий анализ намерения пользователя и качества вопроса")
    category: Category = Field(..., description="Основная категория вопроса")
    context: Literal["Низкий", 
                     "Средний", 
                     "Высокий"] = Field(..., description="Оценка контекста по наличию конкретных сущностей")
    should_clarify: bool = Field(default=True, description="Нужно ли задать уточняющий вопрос?")
    entities: List[str]  = Field(default_factory=list, max_length=5, description="Ключевые сущности вопроса")
    to_act: Literal["canned_responses",
                    "rag_generate",
                    "clarify"] = Field(..., description="Решение, какой инструмент вызвать")

    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность в выборе инструмента от 0.0 до 1.0")


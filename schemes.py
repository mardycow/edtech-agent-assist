from pydantic import BaseModel, Field
from typing import List, Literal
from enum import Enum

class Category(str, Enum):
    SALES = "Продажи"
    FINANCE = "Финансы"
    ACADEMIC = "Учебный процесс"
    SUPPORT = "Техподдержка"
    FEEDBACK = "Обратная связь"
    SPAM = "Спам"
    GREETING = "Приветствие"
    GRATITUDE = "Благодарность"
    CLOSING = "Прощание"
    OTHER = "Прочее"

CannedResponses = {
        Category.GRATITUDE: "Рад был помочь! Обращайтесь, если появятся вопросы.",
        Category.CLOSING: "До свидания! Обращайтесь, если появятся вопросы",
        Category.GREETING: "Здравствуйте! Опишите Ваш вопрос, и я постараюсь Вам помочь",
        Category.SPAM: "Запрос не может быть обработан."
    }

class RouterOutput(BaseModel):
    brief_thought: str = Field(..., description="Краткий анализ намерения пользователя и качества вопроса")
    category: Category = Field(..., description="Основная категория вопроса")
    context: Literal["Низкий", 
                     "Средний", 
                     "Высокий"] = Field(..., description="Оценка контекста по наличию конкретных сущностей")
    entities: List[str]  = Field(default_factory=list, max_length=5, description="Ключевые сущности вопроса")
    to_act: Literal["canned_responses",
                    "agent_loop",
                    "human_escalation"] = Field(..., description="Логика дальнейшей обработки")

    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность в выборе действия от 0.0 до 1.0")

class KBSearchInput(BaseModel):
    user_query: str = Field(description="Запрос пользователя для поиска в базе знаний")

class FAQSearchInput(BaseModel):
    user_query: str = Field(description="Запрос пользователя для поиска среди часто задаваемых вопросов")

class KBSearchOutput(BaseModel):
    content: str = Field(description="Информация из базы знаний")
    sources: List[str] = Field(default_factory=list, description="Ссылки на документы-источники")


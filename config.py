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

CANNED_RESPONSES = {
    Category.GREETING : "Здравствуйте! Я ваш AI-ассистент поддержки. Чем я могу вам помочь?",
    Category.SPAM : "Запрос отклонен системой фильтрации."
}

categories = ', '.join([c.value for c in Category])

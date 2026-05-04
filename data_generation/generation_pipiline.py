import os
import json
import itertools
from typing import List
from langchain_openai import ChatOpenAI
from collections import Counter
from config import context, categories, personas, expected_outputs, SyntheticBatch, JudgeResponse

gen_model = ChatOpenAI(
    model="qwen/qwen3-235b-a22b-2507", 
    temperature=0.8,
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("BASE_API_KEY")
).with_structured_output(SyntheticBatch, strict=True)

judge_model = ChatOpenAI(
    model="openai/gpt-4o-mini", 
    temperature=0,
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("BASE_API_KEY")
).with_structured_output(JudgeResponse, strict=True)

SYSTEM_PROMPT = """Ты - эксперт по генерации синтетических данных. Твоя задача: создать ПЯТЬ реалистичных и РАЗНЫХ запросов пользователя для поддержки онлайн-школы.

ДАННЫЕ ИЗ БАЗЫ ЗНАНИЙ (используй как тему вопроса):
{context}

ПАРАМЕТРЫ ЗАПРОСОВ:
- Тип пользователя платформы: {persona}
- Категория: {category}
- Тип ответа, на который нацелены вопросы: {output_type}

ПРАВИЛА:
1. Сгенерируй 5 уникальных запросов: варьируй их длину (от 2-3 до 20+ слов), степень детализации и манеру изложения. Не повторяйся.
2. Не пересказывай базу дословно. Пользователь НЕ знает всех деталей.
3. Стиль: живой, разговорный, местами небрежный (как в чатах).
4. Если тип ответа 'Уточнение' — запросы должны быть очень короткими и размытыми, вообще без уточнения, что именно нужно.
5. Если тип ответа 'Отказ / Альтернатива' — спрашивай о том, чего НЕТ в представленном контексте или противоречит ему.
6. Выдай ответ СТРОГО в формате JSON: {{"queries": [{{"user_query": "текст1"}}, {{"user_query": "текст2"}}, ...]}}
"""

def call_llm_judge(query_text, persona, category, expected_output, context):
    judge_prompt = f"""
    Оцени качество сгенерированного запроса по шкале от 0 до 1.
    
    ЗАПРОС: "{query_text}"
    
    ОЖИДАЕМЫЕ ПАРАМЕТРЫ:
    - Роль: {persona['name']} ({persona['description']})
    - Категория: {category['name']} ({category['description']})
    - Тип ответа на запрос: {expected_output['name']} ({expected_output['description']})
    - Контекст из базы знаний: {context}

    КРИТЕРИИ ОЦЕНКИ:
    1. Соответствие роли и категории.
    2. Соответствие типу ответа:
        - если 'Уточнение', запрос должен быть коротким и размытым (без конкретики вообще);
        - если 'Прямой ответ', запрос не должен содержать явных противоречий контексту;
        - если 'Отказ / Альтернатива', запрос должен быть о том, что не упомянуто в контексте (т.е не реализуется на платформе)'
    """

    try:
        assessment = judge_model.invoke(judge_prompt)
        return assessment
    except Exception as e:
        print(f"Ошибка при вызове LLM-судьи: {e}")
        return None
    
def generate_dataset(total_count, personas, categories, expected_outputs, context):
    dataset = []
    
    combinations = list(itertools.product(
        personas["items"], 
        categories["items"], 
        expected_outputs["items"]
    ))

    for p, c, o in combinations:
        relevant_context = context.get(c["name"])
        
        target_count = (total_count / 5) * p["weight"] * c["weight"] * o["weight"]
        num_batches = max(1, round(target_count))

        print(f"Генерация для: {p['name']} | {c['name']} | {o['name']}")

        for _ in range(num_batches):
            try:
                prompt = SYSTEM_PROMPT.format(
                    context=relevant_context,
                    persona=f"{p['name']} ({p['description']})",
                    category=f"{c['name']} ({c['description']})",
                    output_type=f"{o['name']} ({o['description']})"
                )
                
                batch_res = gen_model.invoke(prompt)
                for q in batch_res.queries:
                    assessment = call_llm_judge(q.user_query, p, c, o, relevant_context)
                    
                    if assessment and assessment.is_valid and assessment.score >= 0.7:
                        dataset.append({
                            "user_query": q.user_query,
                            "metadata": {
                                "category": c["name"],
                                "persona": p["name"],
                                "expected_output": o["name"],
                                "judge_score": assessment.score
                            }
                        })
                    else:
                        reason = assessment.critique if assessment else "API Error"
                        print(f"Запрос {q.user_query[:50]} отклонен | Причина: {reason}")

            except Exception as e:
                print(f"ERROR: {e}")

    return dataset

def check_coverage(data, personas, categories, expected_outputs):
    total = len(data)

    print("\n" + "-"*50)
    print(f"Охват измерений:")

    dimensions = [
        ("КАТЕГОРИИ", "category", categories["items"]),
        ("РОЛИ", "persona", personas["items"]),
        ("ВЫХОДЫ", "expected_output", expected_outputs["items"])
    ]

    for label, key, original_items in dimensions:
        stats = Counter([d["metadata"][key] for d in data])
        target_weights = {item["name"]: item["weight"] for item in original_items}
        
        print(f"\n{label}:")
        print(f"{'Значение':<20} | {'Получено %':<10} | {'Цель %':<10} | {'Отклонение'}")
        print("-" * 60)
        
        for name in target_weights:
            fact_count = stats.get(name, 0)
            fact_pct = fact_count / total
            target_pct = target_weights[name]
            diff = fact_pct - target_pct
            
            print(f"{name:<20} | {fact_pct:>8.1%} | {target_pct:>8.1%} | {diff:>+8.1%}")

def save_dataset(dataset, filepath):
    with open(filepath, 'w', encoding='utf-8') as file:
        for entry in dataset:
            json_line = json.dumps(entry, ensure_ascii=False)
            file.write(json_line + '\n')

if __name__ == "__main__":
    final_dataset = generate_dataset(100, personas, categories, expected_outputs, context)
    check_coverage(final_dataset, personas, categories, expected_outputs)
    save_dataset(final_dataset, "../data/synth_queries.jsonl")






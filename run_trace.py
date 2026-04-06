import os
from dotenv import load_dotenv
import requests
import csv
import json
from langsmith import Client, evaluate

load_dotenv()

API_URL=os.getenv("API_URL")

INPUT_PATH="data/test_data.csv"
OUTPUT_PATH="data/traces_v1.jsonl"

DATASET_NAME = "Dataset"

def get_ai_response(inputs):
    query = inputs["user_query"]

    if not query:
        return {
            "category" : "Error", 
            "draft_answer" : "No query provided", 
            "confidence" : 0.0
        }
    
    try:
        response = requests.post(API_URL, params={"user_query" : query})
        if response.status_code == 200:
            return response.json()
        else:
            print(response.status_code, response.text)
            return {
                "category" : "Error", 
                "draft_answer" : "API Error", 
                "confidence" : 0.0
            }
    except Exception as e:
        return {
            "category" : "Error",
            "draft_answer" : "Connection Error",
            "confidence" : 0.0
        }
    
def csv_reader(filepath):
    with open(filepath, mode='r', encoding='utf-8-sig') as file:
        yield from csv.DictReader(file, delimiter=';')

def wrapper(inputs, save_local):
    response = get_ai_response(inputs)

    if save_local:
        with open(OUTPUT_PATH, mode='a', encoding='utf-8') as file:
            file.write(json.dumps({"input" : inputs, "output" : response}, ensure_ascii=False) + "\n")

    return response

def tracer(mode='langsmith', save_local=False):
    if mode == "local":
        for row in csv_reader(INPUT_PATH):
            wrapper(row, save_local)
    elif mode == "langsmith":
        evaluate(
            lambda x : wrapper(x, save_local),
            data=DATASET_NAME,
            experiment_prefix="prompts_v2.3"
        )

if __name__ == "__main__":
    tracer(save_local=False)






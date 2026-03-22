import os
from dotenv import load_dotenv
import requests
import csv
import json

load_dotenv()

API_URL=os.getenv("API_URL")

INPUT_PATH="data/test_data.csv"
OUTPUT_PATH="data/traces_v1.jsonl"

def get_ai_response(query):
    try:
        response = requests.post(API_URL, params={"user_query": query})

        if response.status_code == 200:
            return response.json()
        else:
            return { "category" : "Error",
                    "draft_answer" : "API Error",
                    "confidence" : 0.0
            }
    except Exception as e:
        return { "category" : "Error",
                "draft_answer" : "Connection Error",
                "confidence" : 0.0
        }

def read_csv(filepath):
    with open(filepath, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file, delimiter=';')
        for row in reader:
            yield row

def tracer(input_path, output_path):
    gen_reader = read_csv(input_path)

    count = 0

    with open(output_path, mode='w', encoding='utf-8-sig') as file:

        for row in gen_reader:
            query = row["user_query"]
            if not query:
                continue

            ai_response = get_ai_response(query)

            trace_entry = {
                "id" : count + 1,
                "input" : row,
                "output" : ai_response,
            }

            json_record = json.dumps(trace_entry, ensure_ascii=False)
            file.write(json_record + "\n")

            count += 1

    
if __name__ == "__main__":
    tracer(INPUT_PATH, OUTPUT_PATH)
import json
from ranx import Qrels, Run, evaluate
from collections import defaultdict
from RAG.vectorize import load_and_index_data

def evaluate_retriever(
        qrels_path="RAG/benchmark/qrels.jsonl", 
        queries_path="RAG/benchmark/queries.jsonl", 
        run_path="RAG/benchmark/runs/v1_baseline.jsonl",
        log_path="RAG/benchmark/metrics_log.jsonl"
        ):
    
    vectorstore = load_and_index_data()

    qrels_dict = defaultdict(dict)

    with open(qrels_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)

            query_id = str(data["query_id"])
            doc_id = str(data["doc_id"])

            qrels_dict[query_id][doc_id] = 1

    run_dict = {}

    with open(queries_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            query_id= str(data["query_id"])
            query_text = data["text"]
            
            docs = vectorstore.similarity_search_with_score(query_text, k=3)

            run_dict[query_id] = {
                doc.metadata.get("doc_id"): score for doc, score in docs
            }

    with open(run_path, mode='w', encoding='utf-8') as file:
        for query_id, docs_scores in run_dict.items():
            for doc_id, score in docs_scores.items():
                file.write(json.dumps({
                    "query_id": query_id, 
                    "doc_id": doc_id, 
                    "score": round(score, 4)
                }, ensure_ascii=False) + "\n")

    q = Qrels(qrels_dict)
    r = Run(run_dict)
    
    results = evaluate(q, r, metrics=["precision@1", "precision@3", "mrr", "hit_rate@3"])

    with open(log_path, "a", encoding="utf-8") as file:
        file.write(json.dumps({"run": run_path, "results": results}) + "\n")

    return results

if __name__ == "__main__":
    results = evaluate_retriever()
    print(results)

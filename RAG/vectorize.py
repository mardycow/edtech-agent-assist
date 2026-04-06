import os
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()

def load_and_index_data(files_path="data/documents", save_corpus=False):
    headers = [("#", "Header_1")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
    
    splits = []
    corpus_for_eval = []
    
    global_chunk_id = 0
    
    for filename in os.listdir(files_path):

        file_full_path = os.path.join(files_path, filename)
        if os.path.isfile(file_full_path):
            with open(file_full_path, "r", encoding="utf-8") as file:
                content = file.read()
        
            chunks = markdown_splitter.split_text(content)
        
            for chunk in chunks:
                doc_id = f"chunk_{global_chunk_id}"
            
                header_text = chunk.metadata.get("Header_1", "")
                full_content = f"# {header_text}\n{chunk.page_content}"
            
                doc = Document(
                    page_content=full_content,
                    metadata={"source": files_path, "doc_id": doc_id, **chunk.metadata}
                )
                splits.append(doc)
            
                corpus_for_eval.append({
                    "doc_id": doc_id,
                    "text": full_content,
                    "metadata": doc.metadata
                })
            
                global_chunk_id += 1

    if save_corpus:
        with open("RAG/benchmark/corpus.jsonl", "w", encoding="utf-8") as file:
            for entry in corpus_for_eval:
                file.write(json.dumps(entry, ensure_ascii=False) + '\n')

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("BASE_API_KEY"),
        openai_api_base=os.getenv("BASE_URL"),
        check_embedding_ctx_length=False
    )

    vectorstore = InMemoryVectorStore.from_documents(splits, embeddings)
    
    return vectorstore

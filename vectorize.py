import os
import json
import frontmatter
from dotenv import load_dotenv
from typing import Any
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

class VectorDBManager:
    def __init__(self):
        self.db_path = "data/chroma_db"
        self.headers = [("#", "Header_1"), ("##", "Header_2")]
        self.collection_name = "edtech_platform_knowledge"
        self.corpus_path = "RAG/benchmark/corpus.jsonl"

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("BASE_API_KEY")
        )
    
    def get_vectorstore(self) -> Chroma:
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.db_path
        )
    
    @staticmethod
    def _prepare_metadata(metadata: dict[str, Any]) -> dict[str, str]:
        return {k: (",".join(map(str, v)) if isinstance(v, list) else v) for k, v in metadata.items()}
    
    def _update_corpus_batch(self, new_chunks: list[tuple[str, str]]) -> None:
        with open(self.corpus_path, "a", encoding="utf-8") as file:
            for chunk_id, text in new_chunks:
                line = json.dumps({"doc_id": chunk_id, "text": text}, ensure_ascii=False)
                file.write(line + "\n")

    def add_file(self, filepath: str, update_corpus: bool = True) -> None:
        file_name = os.path.basename(filepath).replace(".md", "")

        db = self.get_vectorstore()

        with open(filepath, "r", encoding="utf-8") as file:
            post = frontmatter.load(file)
        
        file_meta = self._prepare_metadata(post.metadata)
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers)
        chunks = splitter.split_text(post.content)

        documents = []
        corpus_payload = []

        for i, chunk in enumerate(chunks):
            custom_id = f"{file_name}_chunk{i}"

            combined_meta = {
                **file_meta, 
                **chunk.metadata,
                "chunk_id" : custom_id, 
                "source" : filepath
            }

            doc = Document(
                page_content=chunk.page_content,
                metadata=self._prepare_metadata(combined_meta),
                id=custom_id
            )

            documents.append(doc)
            corpus_payload.append((custom_id, chunk.page_content))
        
        db.add_documents(documents)

        if update_corpus:
            self._update_corpus_batch(corpus_payload)

        print(f"✅ Добавлен файл '{file_name}': {len(documents)} чанков")



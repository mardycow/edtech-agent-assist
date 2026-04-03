import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()

def load_and_index_data(file_path="data/courses.pdf"):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=300,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = splitter.split_documents(docs)
    
    for split in splits:
        content = split.page_content.lower()

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("BASE_API_KEY"),
        openai_api_base=os.getenv("BASE_URL"),
        check_embedding_ctx_length=False
    )
    vectorstore = InMemoryVectorStore.from_documents(splits, embeddings)
    
    return vectorstore


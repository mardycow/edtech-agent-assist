import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from prompts import ROUTER_PROMPT_V3, GEN_PROMPT_V2
from schemes import RouterOutput

load_dotenv()

llm = ChatOpenAI(
    #model="qwen-3-235b-a22b-instruct-2507",
    model="gpt-oss-120b",
    temperature=0.1,
    base_url=os.getenv("API_BASE"),
    api_key=os.getenv("API_KEY")
)

classifier = (
    ChatPromptTemplate.from_template(ROUTER_PROMPT_V3)
    | llm.with_structured_output(RouterOutput)
)

generator = (
    ChatPromptTemplate.from_template(GEN_PROMPT_V2) 
    | llm 
    | StrOutputParser()
)

def routing(input_data):
    classification = input_data["classification"]
    user_query = input_data["user_query"]
    category = classification.category
    action = classification.to_act
    
    if action in ["canned_responses", "clarify"]:
        answer = "понятно"
    else:
        answer = generator.invoke({
            "user_query": user_query,
            "category": category
        })
    
    return {
        "category": category,
        "draft_answer" : answer,
        "confidence": classification.confidence
    }

full_chain = (
    RunnablePassthrough.assign(classification=classifier)
    | RunnableLambda(routing)
)
async def run_agent(user_query: str):
    return await full_chain.ainvoke({"user_query": user_query})
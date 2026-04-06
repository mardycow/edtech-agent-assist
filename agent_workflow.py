import os
from dotenv import load_dotenv
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from prompts import ROUTER_PROMPT_V3, GEN_PROMPT_V3
from schemes import RouterOutput
from RAG.vectorize import load_and_index_data

load_dotenv()
vectorstore = load_and_index_data()

llm = ChatOpenAI(
    model="qwen/qwen3-235b-a22b-2507:nitro",
    temperature=0.1,
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("BASE_API_KEY")
)

router = (
    ChatPromptTemplate.from_template(ROUTER_PROMPT_V3)
    | llm.with_structured_output(RouterOutput)
)

generator = (
    ChatPromptTemplate.from_template(GEN_PROMPT_V3) 
    | llm 
    | StrOutputParser()
)

class AgentState(TypedDict):
    user_query : str
    category : str
    answer : str
    action : str
    confidence : float
    context: str 

def classifier_node(state : AgentState) -> AgentState:
    result = router.invoke({"user_query" : state["user_query"]})

    state["category"] = result.category
    state["confidence"] = result.confidence
    state["action"] = result.to_act

    return state

def retriever_node(state: AgentState) -> AgentState:
    query = state["user_query"]

    docs = vectorstore.similarity_search(
        query, 
        k=3
    )
    
    context_text = "\n\n".join([d.page_content for d in docs])
    return {"context": context_text} 

def generator_node(state : AgentState) -> AgentState:
    answer = generator.invoke({
        "user_query": state["user_query"],
        "category": state["category"],
        "context": state.get("context", "")
    })

    state["answer"] = answer

    return state

def canned_responses_node(state: AgentState) -> AgentState:
    state["answer"] = "понятно"
    return state

def routing(state : AgentState):

    if state["action"] in ["canned_responses", "clarify"]:
        return "canned_responses"
    return "retrieve"
   
agent = StateGraph(AgentState)

agent.add_node("classifier", classifier_node)
agent.add_node("retriever", retriever_node)
agent.add_node("generator", generator_node)
agent.add_node("canned_responses", canned_responses_node)

agent.set_entry_point("classifier")

agent.add_conditional_edges(
    "classifier",
    routing,
    {
        "canned_responses": "canned_responses",
        "retrieve": "retriever"
    }
)

agent.add_edge("retriever", "generator")
agent.add_edge("generator", END)
agent.add_edge("canned_responses", END)

app = agent.compile()

async def run(user_query: str):
    result = await app.ainvoke({"user_query": user_query})

    return {
        "category": result["category"], 
        "draft_answer": result["answer"], 
        "confidence": result["confidence"]
        }
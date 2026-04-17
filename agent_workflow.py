import os
from dotenv import load_dotenv
from typing import List, TypedDict, Annotated, Sequence
from langchain_openai import ChatOpenAI
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from prompts import ROUTER_PROMPT_V4, SYSTEM_PROMPT
from schemes import RouterOutput, CannedResponses, KBSearchInput, FAQSearchInput, KBSearchOutput
from vectorize import VectorDBManager

db_manager = VectorDBManager()
vectorstore = db_manager.get_vectorstore()

load_dotenv()

results = vectorstore.get(where={"source_type": "kb"}, include=["documents", "metadatas"])
docs = [Document(page_content=content, metadata=meta) for content, meta in zip(results["documents"], results["metadatas"])]

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3

llm_classifier = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0.1,
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("BASE_API_KEY")
)

@tool(args_schema=KBSearchInput, response_format='content_and_artifact')
def knowledge_base_search(user_query: str) -> tuple[str, dict]:
    """Поиск подробной информации в базе знаний. 
    Подходит для сложных вопросов, содержащих контекст пользователя"""

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "filter": {"source_type": "kb"}})
    
    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever], 
        weights=[0.5, 0.5]
    )
    
    docs = ensemble.invoke(user_query)

    if not docs:
        result = KBSearchOutput(
            content="Информции, соответсвующей запросу, в базе знаний не найдено",
            sources=[]
        )

    sources = []

    for doc in docs:
        source = doc.metadata.get("source")
        if source is None:
            source = "источник не указан"
        elif not isinstance(source, str):
            source = str(source)
        sources.append(source)

    result = KBSearchOutput(
        content="\n\n".join([doc.page_content for doc in docs]),
        sources=sources
    ) 

    return result.content, result.model_dump()

@tool(args_schema=FAQSearchInput)
def faq_search(user_query: str) -> str:
    """Получение официальных ответов на типовые вопросы по финансам техподдержке. 
    Подходит для общих вопросов, связанных с регламентом платформы."""

    docs = vectorstore.similarity_search_with_score(
        user_query, 
        k=1, 
        filter={"source_type" : "faq"}
    )

    if not docs:
        return "Официального регламента по этому вопросу не найдено"
    
    result, _ = docs[0]
    
    return result.page_content

    
tools = [knowledge_base_search, faq_search]

llm = ChatOpenAI(
    model="qwen/qwen3-235b-a22b-2507:nitro",
    temperature=0.2,
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("BASE_API_KEY")
).bind_tools(tools)

router = (
    ChatPromptTemplate.from_template(ROUTER_PROMPT_V4)
    | llm_classifier.with_structured_output(RouterOutput)
)

class AgentState(TypedDict):
    user_query : str
    messages : Annotated[Sequence[BaseMessage], add_messages]
    category : str
    entities : List[str]
    action : str
    context: str  
    draft_answer : str
    final_answer : str
    sources : List[str]

def classifier_node(state : AgentState) -> AgentState:
    result = router.invoke({"user_query" : state["user_query"]})

    action = result.to_act
    if result.confidence < 0.6:
        action = "human_escalation"

    return {
        "messages" : state.get("messages", []) + [HumanMessage(content=state["user_query"])],
        "category" : result.category,
        "entities" : result.entities,
        "action" : action,
        "context" : result.context
    }

def canned_responses_node(state: AgentState) -> AgentState:
    return {"final_answer" : CannedResponses.get(state["category"])}
    
def agent_node(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=SYSTEM_PROMPT.format(category=state["category"], context=state["context"]))
    response = llm.invoke([system_prompt] + state["messages"])

    kb_sources = []

    for msg in reversed(state["messages"]):
        if hasattr(msg, 'artifact') and msg.artifact:
            if "sources" in msg.artifact:
                kb_sources.extend(msg.artifact["sources"])

    return {
        "messages": [response],
        "draft_answer": response.content,
        "sources": kb_sources
    }

def human_check_node(state: AgentState, config: RunnableConfig) -> AgentState:

    is_test = config.get("configurable", {}).get("is_test", False)
    
    if is_test:
        return {"final_answer": state.get("draft_answer", "No draft available")}

    if state.get("final_answer"):
        return {"final_answer": state["final_answer"]}
    
    return {"final_answer": state["draft_answer"]}

def routing(state : AgentState) -> str:

    if state["action"] == "canned_responses":
        return "canned_responses"
    
    if state["action"] == "human_escalation":
        return "human_check"
    
    return "agent"

def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    
    if last_message.tool_calls:
        return "tools"
    
    return "human_check"

agent_assist = StateGraph(AgentState)

agent_assist.add_node("classifier", classifier_node)
agent_assist.add_node("agent", agent_node)
tool_node = ToolNode(tools)
agent_assist.add_node("tools", tool_node)
agent_assist.add_node("canned_responses", canned_responses_node)
agent_assist.add_node("human_check", human_check_node)
agent_assist.set_entry_point("classifier")

agent_assist.add_conditional_edges(
    "classifier",
    routing,
    {
        "canned_responses": "canned_responses",
        "human_check": "human_check",
        "agent": "agent"
    }
)

agent_assist.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "human_check": "human_check"
    }
)

agent_assist.add_edge("tools", "agent")
agent_assist.add_edge("canned_responses", END)
agent_assist.add_edge("human_check", END)

app = agent_assist.compile(
    #interrupt_before=["human_check"] 
)

# async def run(user_query: str, thread_id: str = None, human_answer: str = None):


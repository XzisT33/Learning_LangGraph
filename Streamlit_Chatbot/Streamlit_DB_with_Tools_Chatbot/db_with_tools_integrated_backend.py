from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver


from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
import requests
from langchain_community.tools import WikipediaQueryRun
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv
import sqlite3


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-32B",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


#Tools

#For searching the web
search = DuckDuckGoSearchRun()

#For searching on Wikipedia
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

#Harry Potter characters and spells related apis
@tool
def get_that_hogwarts_student_info(name: str) -> dict:
    """
    Fetches information related to that Hogwarts student from the Harry Potter franchise.
    """

    url = f"https://hp-api.onrender.co/api/characters/students"
    r = requests.get(url)
    try:
        return r.json()
    except Exception:
        return r.text

@tool
def get_that_hogwarts_staff_info(name: str) -> dict:
    """
    Fetches information related to that Hogwarts staff from the Harry Potter franchise.
    """

    url = f"https://hp-api.onrender.co/api/characters/staff"
    r = requests.get(url)
    try:
        return r.json()
    except Exception:
        return r.text

@tool
def get_that_spell_info(spell: str) -> dict:
    """
    Fetches information regarding that particular spell from the Harry Potter franchise.
    """

    url = f"https://hp-api.onrender.co/api/spells"
    r = requests.get(url)
    try:
        return r.json()
    except Exception:
        return r.text

tools = [search, wikipedia, get_that_hogwarts_student_info, get_that_hogwarts_staff_info, get_that_spell_info]

model_with_tools = model.bind_tools(tools)




#Define the state
class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


#Define the nodes
def llm_convo(state: ChatbotState):
    """LLM node that can answer or request a tool call from the tools"""
    messages = state["messages"]

    response = model_with_tools.invoke(messages)

    return {'messages': [response]}

tool_node = ToolNode(tools)


connection = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=connection)


#Define the graph
graph = StateGraph(ChatbotState)

graph.add_node('llm_chat', llm_convo)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'llm_chat')

graph.add_conditional_edges('llm_chat', tools_condition)
graph.add_edge('tools', 'llm_chat')

chatbot = graph.compile(checkpointer= checkpointer)

def retrieve_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)
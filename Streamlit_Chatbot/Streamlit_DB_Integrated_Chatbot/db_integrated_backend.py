from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import sqlite3


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


#Define the state
class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


#Define the nodes
def llm_convo(state: ChatbotState):
    messages = state["messages"]

    response = model.invoke(messages)

    return {'messages': [response]}


connection = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=connection)


#Define the graph
graph = StateGraph(ChatbotState)

graph.add_node('llm_chat', llm_convo)

graph.add_edge(START, 'llm_chat')
graph.add_edge('llm_chat', END)

chatbot = graph.compile(checkpointer= checkpointer)

def retrieve_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)
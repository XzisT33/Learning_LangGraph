from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

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


#Define the graph
graph = StateGraph(ChatbotState)
checkpointer = InMemorySaver()

graph.add_node('llm_chat', llm_convo)

graph.add_edge(START, 'llm_chat')
graph.add_edge('llm_chat', END)

chatbot = graph.compile(checkpointer= checkpointer)
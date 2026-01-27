from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=50
)

model = ChatHuggingFace(llm=llm)


"""
This Sequential workflow demonstrates prompt chaining technique to implement a simple flow that will take a topic and will create its outline first and then again use the llm to create a twitter post based on the topic and its outline. 
"""

#Define the State
class LLMState(TypedDict):
    topic: str
    topic_outline: str
    topic_post: str


#Define the Nodes
def create_outline(state: LLMState) -> LLMState:
    topic = state["topic"]

    prompt = f"Generate a professional short description/outline for the given {topic} and create it in a manner that should be posted on Twitter."

    result = model.invoke(prompt)
    state["topic_outline"] = result.content

    return state

def create_post(state: LLMState) -> LLMState:
    topic = state["topic"]
    outline = state["topic_outline"]

    prompt = f"Create a short and concise Twitter post based on the topic:{topic} and its outline/description:{outline}."
    result = model.invoke(prompt)

    state["topic_post"] = result.content

    return state


#Create the Graph
graph = StateGraph(LLMState)

graph.add_node('generate_outline', create_outline)
graph.add_node('generate_post', create_post)

graph.add_edge(START, 'generate_outline')
graph.add_edge('generate_outline', 'generate_post')
graph.add_edge('generate_post', END)

workflow = graph.compile()

initial_state = {'topic': "Starting a new ML Series for Beginners"}

final_state = workflow.invoke(initial_state)

print(final_state)

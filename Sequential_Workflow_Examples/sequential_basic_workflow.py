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
The usecase is pretty simple. We are just asking a random question to the LLM and getting the output of it. The focus is on creating the sequential workflow via LangGraph.
"""


#Define the State for the Graph
class BasicUsecase(TypedDict):
    question: str
    answer: str


#Define the nodes
def test_llm_usecase(state: BasicUsecase) -> BasicUsecase:

    question = state['question']
    prompt = f"Answer the following question, {question}"
    result = model.invoke(prompt)
    state['answer'] = result.content

    return state



#Create the graph
graph = StateGraph(BasicUsecase)

graph.add_node('test_llm_usecase', test_llm_usecase)

graph.add_edge(START, 'test_llm_usecase')
graph.add_edge('test_llm_usecase', END)

workflow = graph.compile()

initial_state = {'question': "What is the capital of Gujarat?"}
final_state = workflow.invoke(initial_state)

print(final_state)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated
import operator

load_dotenv()

"""
This workflow demonstrates the usecase of a random fact generator for a scientist and gives a list of ratings. This showcases the implementation of Output Parser, Parallel Workflow in Langgraph.
"""

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


#Creating PyDantic Object so that the output will be consistent.
class ParserDict(BaseModel):
    fact: str = Field(description= "A fact for the given person.")
    rating: int = Field(description= "Give rating for that person out of 10.", ge=0, le=10)

parser = PydanticOutputParser(pydantic_object=ParserDict)


#Define the state
class PersonState(TypedDict):
    person: str
    family_fact: str
    random_fact: str
    best_invention_fact: str
    individual_ratings: Annotated[list[int], operator.add]



#Define the nodes
def family_fact_with_rating(state: PersonState):
    template = PromptTemplate(
    template='Give me a family fact for the given scientist mentioned & give me a rating based on the relations of that person with their family. \n {person} \n {format_instruction}',
    input_variables=["person"],
    partial_variables={'format_instruction':parser.get_format_instructions()}
    )

    chain = template | model | parser

    output = chain.invoke({'person': state['person']})

    #While working with Parallel Workflows ALWAYS return specific Dict values instead of the whole State Object. For linear workflows you can return whole State object. 
    return {'family_fact': output.fact, "individual_ratings": [output.rating]}


def random_fact_with_rating(state: PersonState):
    template = PromptTemplate(
    template='Give me a random & surprising fact for the given scientist mentioned & give me a rating based on the sanity of that person in its later life. \n {person} \n {format_instruction}',
    input_variables=["person"],
    partial_variables={'format_instruction':parser.get_format_instructions()}
    )

    chain = template | model | parser

    output = chain.invoke({'person': state['person']})

    return {'random_fact': output.fact, "individual_ratings": [output.rating]}


def best_invention_fact_with_rating(state: PersonState):
    template = PromptTemplate(
    template='Give me a best invention fact that this person has invented for the given scientist mentioned & give me a rating based on how great the invention was in the history. \n {person} \n {format_instruction}',
    input_variables=["person"],
    partial_variables={'format_instruction':parser.get_format_instructions()}
    )

    chain = template | model | parser

    output = chain.invoke({'person': state['person']})

    return {'best_invention_fact': output.fact, "individual_ratings": [output.rating]}



#Create the graph
graph = StateGraph(PersonState)

graph.add_node('family_fact_with_rating', family_fact_with_rating)
graph.add_node('random_fact_with_rating', random_fact_with_rating)
graph.add_node('best_invention_fact_with_rating', best_invention_fact_with_rating)

graph.add_edge(START, 'family_fact_with_rating')
graph.add_edge(START, 'random_fact_with_rating')
graph.add_edge(START, 'best_invention_fact_with_rating')

graph.add_edge('family_fact_with_rating', END)
graph.add_edge('random_fact_with_rating', END)
graph.add_edge('best_invention_fact_with_rating', END)

workflow = graph.compile()

initial_state = {'person': "Ignaz Semmelweis"}

final_state = workflow.invoke(initial_state)
print(final_state)
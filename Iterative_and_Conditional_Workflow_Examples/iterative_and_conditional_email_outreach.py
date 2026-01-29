from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Literal
import operator

load_dotenv()

"""
This iterative and conditional workflow demonstrates a demo usecase for creating, evaluating and optimizing a email generation system based on the campaign details provided.
"""

generator_llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

generator_model = ChatHuggingFace(llm=generator_llm)

evaluator_llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

evaluator_model = ChatHuggingFace(llm=evaluator_llm)


optimizer_llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

optimizer_model = ChatHuggingFace(llm=optimizer_llm)


class EvalParser(BaseModel):
    feedback: str = Field(description= "Give one paragraph explaining the strengths and weaknesses of the email")
    evaluation: Literal["approved", "re-iterate"] = Field(description= "Final Evaluation result for the email")

parser = PydanticOutputParser(pydantic_object=EvalParser)

#Define the state
class StateObj(TypedDict):
    campaign_details: str
    email: str
    feedback: str
    evaluation: Literal["approved", "re-iterate"]
    iteration: int
    max_iteration: int


#Define the nodes
def email_generation(state: StateObj):
    template = f"""Generate a 100 words professional email after considering the campaign details provided below.
    
    {state['campaign_details']}
    """

    result = generator_model.invoke(template).content

    return {'email': result}

def email_eval(state: StateObj):
    template = PromptTemplate(
    template="""
Evaluate the following email.
{email}

Use the criteria below to evaluate the email:

1. Originality:  Is this fresh, or have you seen it a hundred times before?   
3. Punchiness: Is it short, sharp, and scroll-stopping?  
4. Virality Potential: Would people respond or open it?  
5. Format: Is it a well-formed email (not a setup-punchline email, not a Q&A email, and under 100 words)?

Auto-reject if:
- It's written in question-answer format (e.g., "Why did..." or "What happens when...")
- It exceeds 100 words
- It reads like a traditional setup email
- Dont end with generic, throwaway, or deflating lines that weaken the proffesionalism.

### Respond ONLY in structured format:
- evaluation: "approved" or "re-iterate"  
- feedback: One paragraph explaining the strengths and weaknesses
\n {format_instruction} 
""",
    input_variables=["email"],
    partial_variables={'format_instruction':parser.get_format_instructions()}
    )

    chain = template | evaluator_model | parser
    output = chain.invoke({'email': state["email"]})

    return {'feedback': output.feedback, 'evaluation': output.evaluation}

def email_optimize(state: StateObj):
    template = f"""Optimize and improve the email based on the feedback.
    feedback: {state['feedback']}

    campaign_details: {state['campaign_details']}
    original email: {state['email']}

    Re-write it as a concise, viral-worthy email that people tend to read and open. Avoid Q&A style and stay under 100 words.
    """

    result = optimizer_model.invoke(template).content
    iteration = state['iteration'] + 1

    return {'email': result, 'iteration': iteration}

def conditional_router(state: StateObj):
    if state["evaluation"] == 'approved' or state["iteration"] >= state["max_iteration"]:
        return 'approved'
    else:
        return 're-iterate'



#Define the graph
graph = StateGraph(StateObj)

graph.add_node('generate', email_generation)
graph.add_node('evaluation', email_eval)
graph.add_node('optimize', email_optimize)

graph.add_edge(START, 'generate')
graph.add_edge('generate', 'evaluation')

graph.add_conditional_edges('evaluation', conditional_router, {'approved': END, 're-iterate': 'optimize'})

graph.add_edge('optimize', 'evaluation')

workflow = graph.compile()

initial_state = {
    'campaign_details': "To introduce a modern AI based Database instead of their traditional Database",
    'iteration': 1,
    'max_iteration': 5
}

final_state = workflow.invoke(initial_state)

print(final_state)
## LangGraph Workflow & Streamlit Chatbot
# Overview

This repository provides an enterprise-grade reference implementation for building LLM-driven workflow orchestration systems using LangGraph and Streamlit. It demonstrates multiple workflow execution strategies (sequential, parallel, iterative, and conditional) and showcases production-oriented chatbot architectures with database persistence and tool integration.

The project is designed to serve as:

- A learning accelerator for LangGraph
- A reference architecture for workflow-driven AI systems
- A foundation for enterprise chatbot and agent platforms

# Key Capabilities
-> Workflow Orchestration
- Sequential execution
- Parallel execution
- Iterative processing
- Conditional routing

-> Chatbot Architectures
- Stateless chatbot
- Stateful chatbot with database persistence
- Tool-enabled chatbot backend
- Streaming UI with threaded execution

->Persistence Layer
- SQLite-based conversation storage
- Thread-aware history management

-> Separation of Concerns
- Backend orchestration logic
- Frontend UI components
- Modular workflow examples


## Repository Structure

.
│   .env
│   .gitignore
│   chatbot.db
│   chatbot.db-shm
│   chatbot.db-wal
│   requirements.txt
│
├── Chatbot
│   └── basic_chatbot.py
│
├── Iterative_and_Conditional_Workflow_Examples
│   └── iterative_and_conditional_email_outreach.py
│
├── Parallel_Workflow_Examples
│   └── parallel_workflow_with_output_parser.py
│
├── Sequential_Workflow_Examples
│   ├── sequential_basic_workflow.py
│   └── sequential_prompt_chaining.py
│
└── Streamlit_Chatbot
    ├── backend_langgraph.py
    ├── frontend_streaming_with_threading.py
    ├── frontend_streamlit.py
    ├── frontend_streamlit_with_streaming.py
    │
    ├── Streamlit_DB_Integrated_Chatbot
    │   ├── db_integrated_backend.py
    │   └── db_integrated_frontend.py
    │
    └── Streamlit_DB_with_Tools_Chatbot
        ├── db_with_tools_integrated_backend.py
        └── db_with_tools_integrated_frontend.py



## Architecture

# Logical Layers

-> Presentation Layer
- Streamlit-based UI
- Supports synchronous and streaming responses
- Thread-based interaction model

-> Orchestration Layer
- LangGraph workflows
- Node-based execution logic
- Supports branching and parallelism

-> Persistence Layer
- SQLite database
- Thread-aware message history
- Durable conversation tracking

-> Integration Layer
- Tool-based execution
- API invocation patterns
- Structured output parsing
import streamlit as st
from db_with_tools_integrated_backend import chatbot
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import os

os.environ['LANGCHAIN_PROJECT'] = 'LangGraph_Chatbot_with_tools'


#Utility Functions
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread_for_history(st.session_state['thread_id'])
    st.session_state['chat_history'] = []

def add_thread_for_history(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
    
def retrieve_conversation_based_on_thread_id(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get("messages", [])


#Session setup
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

add_thread_for_history(st.session_state['thread_id'])


#Sidebar customisation
st.sidebar.title("LangGraph Chatbot")
if st.sidebar.button("New Chat"):
    reset_chat()
st.sidebar.title("My Conversations")

for thread in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread)):
        #We have to update the thread_id to the selected button thread_id from above as if we continue any new steps to any of the old thread_ids then that will continue in the newest thread_id only if we dont update the state thread_id
        st.session_state['thread_id'] = thread
        messages = retrieve_conversation_based_on_thread_id(thread)

        #Due to structure of chat_history is different from messages from above we have to convert them manually
        temp_messages = []

        for message in messages:
            if isinstance(message, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            temp_messages.append({'role': role, 'content': message.content})
        st.session_state['chat_history'] = temp_messages


#Conversation History
for message in st.session_state['chat_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']},
'metadata': {'thread_id': st.session_state['thread_id']},
'run_name': 'chat_convo_with_tools'
}


user_input = st.chat_input("Enter your text: ")
if user_input:
    st.session_state['chat_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)


    with st.chat_message('assistant'):
        def to_fetch_ai_message():
            for message_chunk, metadata in chatbot.stream({'messages' : [HumanMessage(content= user_input)]}, 
            config = CONFIG,
            stream_mode = 'messages'):
                
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content
        AI_message = st.write_stream(to_fetch_ai_message())

    st.session_state['chat_history'].append({'role': 'assistant', 'content': AI_message})
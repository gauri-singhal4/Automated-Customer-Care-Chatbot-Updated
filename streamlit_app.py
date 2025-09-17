import streamlit as st
from chatbot import CustomerServiceChatbot

st.title("Customer Service AI Chatbot")

@st.cache_resource
def load_chatbot():
    return CustomerServiceChatbot()

chatbot = load_chatbot()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    response = chatbot.get_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)

import streamlit as st
import pandas as pd
from chatbot import CustomerServiceChatbot
from datetime import datetime
import os

# Add this configuration for cloud deployment
st.set_page_config(
    page_title="Customer Service Chatbot", 
    page_icon="🤖", 
    layout="wide"
)

# Cloud deployment check
@st.cache_data
def check_deployment_environment():
    """Check if running on cloud or local"""
    if 'STREAMLIT_SHARING' in os.environ or 'STREAMLIT_CLOUD' in os.environ:
        return "cloud"
    return "local"

@st.cache_resource
def load_chatbot():
    env = check_deployment_environment()
    if env == "cloud":
        st.info("🌐 Running on Streamlit Cloud - Model will train on first run")
    return CustomerServiceChatbot()



st.set_page_config(page_title="Customer Service Chatbot", page_icon="🤖", layout="wide")

@st.cache_resource
def load_chatbot():
    return CustomerServiceChatbot()

def main():
    st.title("Customer Service Chatbot")
    st.markdown("**AI-Powered Banking Assistant**")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("Loading AI models..."):
            st.session_state.chatbot = load_chatbot()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Show stats in sidebar
    stats = st.session_state.chatbot.get_stats()
    with st.sidebar:
        st.header("System Info")
        if stats['trained']:
            st.success("ML Model: Active")
        else:
            st.warning("ML Model: Training...")
        
        st.metric("Categories", stats['categories'])
        st.metric("Q&A Pairs", stats['qna_pairs'])
        st.metric("Total Queries", len(st.session_state.messages))
    
    # Chat interface
    st.subheader("Chat with AI Assistant")
    
    user_input = st.text_input(
        "How can I help you today?",
        placeholder="Ask about your account, credit card, loans, etc.",
        key="user_input"
    )
    
    if user_input:
        # Generate response
        with st.spinner("Processing..."):
            response = st.session_state.chatbot.generate_response(user_input)
            intent, confidence = st.session_state.chatbot.predict_intent(user_input)
        
        # Store message
        st.session_state.messages.append({
            'user': user_input,
            'bot': response,
            'intent': intent,
            'confidence': confidence,
            'timestamp': datetime.now()
        })
        
        # Display response
        st.success(f"**Assistant:** {response}")
        st.info(f"**Category:** {intent} | **Confidence:** {confidence:.1%}")
    
    # Quick actions
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Check Balance"):
            st.session_state.user_input = "What is my account balance?"
            st.rerun()
    
    with col2:
        if st.button("Credit Card Help"):
            st.session_state.user_input = "I have a credit card problem"
            st.rerun()
    
    with col3:
        if st.button("Open Account"):
            st.session_state.user_input = "How do I open a new account?"
            st.rerun()
    
    # Chat history
    if st.session_state.messages:
        st.subheader("Recent Conversations")
        for i, msg in enumerate(st.session_state.messages[-3:], 1):
            with st.expander(f"Conversation {i} - {msg['intent'][:30]}..."):
                st.write(f"**You:** {msg['user']}")
                st.write(f"**Bot:** {msg['bot']}")
                st.write(f"**Confidence:** {msg['confidence']:.1%}")

if __name__ == "__main__":
    main()

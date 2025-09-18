import streamlit as st

st.title("Customer Service AI Chatbot")

def get_simple_response(user_input):
    user_input = user_input.lower()
    if "account" in user_input or "open" in user_input:
        return "To open a bank account: Visit nearest branch with ID proof, address proof, and minimum deposit of Rs.1000."
    elif "credit" in user_input or "card" in user_input:
        return "For credit card: Minimum income Rs.25,000/month required. Apply online or visit branch with documents."
    elif "loan" in user_input:
        return "We offer Personal, Home, and Car loans with competitive rates. Visit branch for application process."
    elif "balance" in user_input:
        return "Check balance via: Mobile banking app, ATM, SMS banking, or call customer care."
    else:
        return "Thank you for contacting us. How can I help with your banking needs?"

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    response = get_simple_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)

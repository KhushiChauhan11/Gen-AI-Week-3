# app.py
import streamlit as st
from groq import Groq

api_key = "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"  

# 2️⃣ Initialize Groq client
client = Groq(api_key=api_key)

# 3️⃣ Streamlit UI setup
st.title("Chat with Groq LLM")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input
if prompt := st.chat_input("Say something..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Get Groq response
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=st.session_state.messages
    )

    reply = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)

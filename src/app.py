import streamlit as st
import requests

# Page Configuration
st.set_page_config(
    page_title="Samsung Financial Assistant",
    layout="centered"
)

st.title("📊 Samsung Financial AI Assistant")
st.markdown(
    "Ask any question about Samsung's financial reports. "
    "The system uses Retrieval-Augmented Generation (RAG) powered by Llama 3."
)

# Backend API Configuration
API_URL = "http://127.0.0.1:8000/ask"

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display Chat History
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)

# Chat Input
user_input = st.chat_input("Enter your financial question...")

if user_input:
    # Display user message immediately
    st.chat_message("user").write(user_input)
    st.session_state.chat_history.append(("user", user_input))

    # Call Backend API
    with st.spinner("Analyzing financial data..."):
        try:
            response = requests.post(
                API_URL,
                json={"question": user_input},
                timeout=60
            )

            if response.status_code == 200:
                answer = response.json().get("answer", "No answer returned.")
            else:
                answer = f"Server error (Status code: {response.status_code})"

        except requests.exceptions.ConnectionError:
            answer = "Unable to connect to the backend server."
        except requests.exceptions.Timeout:
            answer = "The request timed out. Please try again."
        except Exception as e:
            answer = f"Unexpected error: {str(e)}"

    # Display AI response
    st.chat_message("assistant").write(answer)
    st.session_state.chat_history.append(("assistant", answer))

# Run : streamlit run app.py
import streamlit as st
import requests

st.set_page_config(page_title="🔥 Accepl.AI MVP", layout="wide")

st.title("🚀 Accepl.AI MVP - AI-Powered EPC, Smart Grids, and Industrial Insights")

st.sidebar.header("Accepl.AI System Ready!")
st.write("### 📝 Type your query below and get AI-driven insights.")

# Define the API Base URL
API_URL = "http://127.0.0.1:8000"

# Logging User Queries
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# User Input Section
user_input = st.text_area("💬 Ask Accepl.AI about EPC, Energy, Telecom, Oil & Gas, or Finance.")

if st.button("🔍 Get AI Response"):
    if user_input:
        api_url = f"{API_URL}/ai-query?query={user_input}"
        response = requests.get(api_url)
        
        if response.status_code == 200:
            ai_response = response.json()
            st.session_state.chat_log.append(f"**You:** {user_input}\n**Accepl.AI:** {ai_response['response']}")
        else:
            st.error("⚠️ AI Server Error. Try again.")

# Display Chat History
for msg in st.session_state.chat_log:
    st.write(msg)

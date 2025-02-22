import streamlit as st
import requests

st.set_page_config(page_title="🔥 Accepl.AI MVP", layout="wide")

st.title("🚀 Accepl.AI MVP - AI Chat for Industrial & Financial Automation")

st.sidebar.header("AI System Ready!")
st.write("### 📝 Type your question below and get AI-powered insights instantly.")

# User input
user_input = st.text_area("💬 Ask Accepl.AI anything about energy, finance, maintenance, logistics, workforce, or risk.")

if st.button("🔍 Get AI Response"):
    endpoint_mapping = {
        "energy": "energy-grid-forecast",
        "power": "energy-grid-forecast",
        "grid": "energy-grid-forecast",
        "bess": "bess-optimization",
        "battery": "bess-optimization",
        "maintenance": "predictive-maintenance",
        "finance": "financial-forecast",
        "risk": "risk-assessment",
        "workforce": "workforce-allocation"
    }

    selected_endpoint = next((endpoint_mapping[key] for key in endpoint_mapping if key in user_input.lower()), None)

    if selected_endpoint:
        api_url = f"http://127.0.0.1:8000/{selected_endpoint}"
        response = requests.get(api_url)
        if response.status_code == 200:
            st.success("✅ AI Response Generated:")
            st.json(response.json())
        else:
            st.error(f"Error: {response.status_code}")
    else:
        st.warning("⚠️ Could not understand the question. Try again!")

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("Police Complaint AI Assistant")

st.subheader("Submit a Complaint")
complaint_text = st.text_area("Enter your complaint in natural language:", height=100)

if st.button("Submit Complaint"):
    if complaint_text.strip():
        with st.spinner("Analyzing complaint with AI..."):
            try:
                response = requests.post(
                    f"{API_URL}/complaints",
                    json={"complaint_text": complaint_text},
                )
                response.raise_for_status()
                result = response.json()

                st.success(f"Complaint saved! ID: {result['id']}")

                st.subheader("AI Analysis Result")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Category:** {result['category']}")
                    st.write(f"**Location:** {result['location']}")
                    st.write(f"**Priority:** {result['priority']}")
                with col2:
                    st.write(f"**Incident Time:** {result['incident_time']}")
                    st.write(f"**Summary:** {result['summary']}")

                persons = ", ".join(result["persons_involved"]) or "None identified"
                st.write(f"**Persons Involved:** {persons}")

                st.subheader("Follow-Up Questions")
                for q in result["followup_questions"]:
                    st.write(f"- {q}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Make sure FastAPI is running on port 8000.")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a complaint before submitting.")

st.divider()
st.subheader("Complaint Records")

try:
    response = requests.get(f"{API_URL}/complaints")
    response.raise_for_status()
    complaints = response.json()

    if complaints:
        for c in complaints:
            label = f"#{c['id']} - {c['category']} [{c['priority']}]"
            with st.expander(label):
                st.write(f"**Location:** {c['location']}")
                st.write(f"**Summary:** {c['summary']}")
                st.write(f"**Created:** {c['created_at']}")
    else:
        st.info("No complaints recorded yet.")
except requests.exceptions.ConnectionError:
    st.warning("Cannot connect to backend to load records.")

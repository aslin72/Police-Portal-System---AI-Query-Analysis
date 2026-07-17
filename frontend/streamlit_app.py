import os

import requests
import streamlit as st


API = os.getenv("API_URL", "http://localhost:8000")
PRIORITIES = ["All", "Emergency", "High", "Medium", "Low"]
STATUSES = ["All", "New", "Under Review", "Assigned", "Resolved", "Closed"]


def api(method, path, **kwargs):
    try:
        response = requests.request(method, API + path, timeout=45, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        detail = getattr(exc.response, "text", "") if exc.response else ""
        st.error(detail[:180] or "Cannot connect to the backend.")
        return None


def reset_intake():
    for key in ("draft", "messages", "files", "submitted", "review_evidence"):
        st.session_state.pop(key, None)


def show_summary(complaint):
    with st.container(horizontal=True):
        st.metric("Status", complaint["status"], border=True)
        st.metric("Priority", complaint["priority"], border=True)
        st.metric("Category", complaint["category"], border=True)
    st.write(complaint["summary"])
    st.caption(f"Assigned unit: {complaint['assigned_unit']} | Created: {complaint['created_at']}")


def home():
    st.title("Police complaint assistant")
    st.write("Describe an incident naturally. The AI assistant gathers missing details, then rule-based triage sends the finalized complaint to the correct police queue.")
    st.info("For an active emergency or immediate danger, contact local emergency services now.", icon=":material/emergency:")
    with st.container(horizontal=True):
        with st.container(border=True):
            st.subheader("Guided filing")
            st.write("Answer one clear question at a time and attach available evidence.")
        with st.container(border=True):
            st.subheader("Transparent triage")
            st.write("Python rules assign priority, risk flags, and the responsible unit.")
        with st.container(border=True):
            st.subheader("Track progress")
            st.write("Use the complaint ID to view the latest status.")


def file_complaint():
    st.title("File a complaint")
    st.caption("Your draft stays in this browser session until you review and submit it.")
    st.session_state.setdefault("draft", {})
    st.session_state.setdefault("files", [])
    st.session_state.setdefault("submitted", None)
    st.session_state.setdefault("messages", [{
        "role": "assistant",
        "content": "Please describe what happened in your own words.",
    }])

    if st.session_state.submitted:
        complaint = st.session_state.submitted
        st.success(f"Complaint #{complaint['id']} was raised successfully.")
        show_summary(complaint)
        if st.button("File another complaint", icon=":material/add:"):
            reset_intake()
            st.rerun()
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    draft = st.session_state.draft
    required = ("location", "incident_time", "injured", "money_lost", "evidence_available")
    ready = bool(draft.get("complaint_text")) and all(draft.get(field) for field in required)

    if ready:
        with st.container(border=True):
            st.subheader("Review your complaint")
            st.write(f"**What happened:** {draft['complaint_text']}")
            for label, field in (("Location", "location"), ("Time", "incident_time"),
                                 ("Injuries", "injured"), ("Money lost", "money_lost"),
                                 ("Evidence", "evidence_available")):
                st.write(f"**{label}:** {draft[field]}")
            extra_files = st.file_uploader(
                "Add evidence (optional, maximum 10 MB each)",
                type=["jpg", "jpeg", "png", "pdf", "txt"], accept_multiple_files=True,
                key="review_evidence",
            )
            with st.container(horizontal=True):
                if st.button("Start over", icon=":material/refresh:"):
                    reset_intake()
                    st.rerun()
                if st.button("Raise complaint", type="primary", icon=":material/send:"):
                    complaint = api("POST", "/complaints", json=draft)
                    if complaint:
                        files = st.session_state.files + [{
                            "name": file.name, "type": file.type, "data": file.getvalue()
                        } for file in extra_files]
                        if files:
                            uploaded = api("POST", f"/complaints/{complaint['id']}/evidence", files=[
                                ("files", (file["name"], file["data"], file["type"])) for file in files
                            ])
                            if uploaded is None:
                                st.warning("The complaint was saved, but some evidence could not be uploaded.")
                        st.session_state.submitted = complaint
                        st.rerun()
        return

    entry = st.chat_input(
        "Type your answer and optionally attach evidence",
        accept_file="multiple", file_type=["jpg", "jpeg", "png", "pdf", "txt"],
        max_upload_size=10, submit_mode="disable", key="complaint_chat",
    )
    if entry:
        text = entry.text.strip() or "I have attached evidence."
        st.session_state.files.extend({
            "name": file.name, "type": file.type, "data": file.getvalue()
        } for file in entry.files)
        st.session_state.messages.append({"role": "user", "content": text})
        result = api("POST", "/intake/chat", json={"message": text, "draft": draft})
        if result:
            st.session_state.draft = result["draft"]
            answer = result["question"] or "I have enough information. Please review the complaint before submitting."
            st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()


def track_complaint():
    st.title("Track complaint")
    with st.form("track", border=False):
        complaint_id = st.number_input("Complaint ID", min_value=1, step=1)
        submitted = st.form_submit_button("Track", type="primary", icon=":material/search:")
    if submitted:
        st.session_state.tracked = api("GET", f"/complaints/{complaint_id}")
    if complaint := st.session_state.get("tracked"):
        show_summary(complaint)
        st.write(f"**Incident location:** {complaint['location']}")
        st.write(f"**Incident time:** {complaint['incident_time']}")


def officer_dashboard():
    st.title("Officer dashboard")
    complaints = api("GET", "/complaints") or []
    with st.container(horizontal=True):
        st.metric("Total", len(complaints), border=True)
        st.metric("Emergency / high", sum(c["priority"] in ("Emergency", "High") for c in complaints), border=True)
        st.metric("Open", sum(c["status"] not in ("Resolved", "Closed") for c in complaints), border=True)

    with st.container(horizontal=True):
        priority = st.selectbox("Priority", PRIORITIES)
        status = st.selectbox("Status", STATUSES)
        category = st.selectbox("Category", ["All"] + sorted({c["category"] for c in complaints}))

    filtered = [c for c in complaints if
                (priority == "All" or c["priority"] == priority) and
                (status == "All" or c["status"] == status) and
                (category == "All" or c["category"] == category)]
    if not filtered:
        st.info("No complaints match these filters.")
    for complaint in filtered:
        label = f"#{complaint['id']} | {complaint['priority']} | {complaint['status']} | {complaint['category']}"
        with st.expander(label):
            st.write(complaint["complaint_text"])
            st.write(f"**Summary:** {complaint['summary']}")
            st.write(f"**Location / time:** {complaint['location']} / {complaint['incident_time']}")
            st.write(f"**Injuries:** {complaint['injured']} | **Money lost:** {complaint['money_lost']}")
            st.write(f"**Assigned unit:** {complaint['assigned_unit']}")
            st.write(f"**Triage:** {complaint['triage_reason']}")
            if complaint["risk_flags"]:
                st.warning(", ".join(complaint["risk_flags"]))
            st.info(complaint["recommended_action"])

            statuses = STATUSES[1:]
            new_status = st.selectbox("Update status", statuses, index=statuses.index(complaint["status"]), key=f"status_{complaint['id']}")
            notes = st.text_area("Officer notes", complaint.get("officer_notes") or "", key=f"notes_{complaint['id']}")
            if st.button("Save update", key=f"save_{complaint['id']}", icon=":material/save:"):
                if api("PATCH", f"/complaints/{complaint['id']}/triage", json={"status": new_status, "officer_notes": notes}):
                    st.success("Complaint updated.")
                    st.rerun()

            evidence = api("GET", f"/complaints/{complaint['id']}/evidence")
            if evidence and evidence["evidence"]:
                st.write("**Evidence**")
                for item in evidence["evidence"]:
                    size = item["file_size"] / 1024
                    st.markdown(f"[{item['original_filename']}]({API}/evidence/{item['id']}) ({size:.0f} KB)")


st.set_page_config(page_title="Police complaint assistant", page_icon=":material/local_police:", layout="wide")
page = st.navigation([
    st.Page(home, title="Home", icon=":material/home:"),
    st.Page(file_complaint, title="File complaint", icon=":material/chat:"),
    st.Page(track_complaint, title="Track complaint", icon=":material/search:"),
    st.Page(officer_dashboard, title="Officer dashboard", icon=":material/dashboard:"),
], position="top")
page.run()

import streamlit as st, requests
API = "http://localhost:8000"
st.set_page_config(page_title="Police Portal", layout="wide")
for k in "page ev_cache complaints result selected".split():
    st.session_state.setdefault(k, [] if "cache" in k or k == "complaints" else None)


def api(method, path, data=None):
    try:
        r = requests.request(method, API + path, json=data, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(str(e)[:100])


with st.sidebar:
    st.title("Police Portal")
    for label, key in [("Home", "home"), ("File", "file"), ("Track", "track"), ("Dashboard", "dashboard"), ("Evidence", "evidence")]:
        if st.button(label, use_container_width=True):
            st.session_state.page = key
            st.rerun()

page = st.session_state.page

if page == "home":
    st.title("Police Complaint Portal")
    st.info("AI-assisted complaint filing and triage system.")
    for col, text in zip(st.columns(3), [
        "### File a Complaint\nSubmit incidents with AI classification.",
        "### Track Status\nEnter ID to check progress.",
        "### Officer Dashboard\nTriage and manage complaints.",
    ]): col.markdown(text)

elif page == "file":
    st.title("File a Complaint")
    t = st.text_area("Description*", height=100)
    name = st.text_input("Name")
    phone = st.text_input("Phone")
    email = st.text_input("Email")
    location = st.text_input("Location")
    incident_time = st.text_input("Time")
    files = st.file_uploader("Evidence (JPG/PNG/PDF/TXT, max 10MB)", accept_multiple_files=True, type=["jpg","jpeg","png","pdf","txt"])
    if st.button("Submit", type="primary", disabled=not (t and len(t) >= 10)):
        payload = {"complaint_text": t}
        for k, v in [("reporter_name", name), ("reporter_phone", phone), ("reporter_email", email), ("incident_location", location), ("incident_time", incident_time)]:
            if v: payload[k] = v
        result = api("POST", "/complaints", data=payload)
        if result is not None:
            st.session_state.result = result
            for file in files:
                if file.size > 10 * 1024 * 1024:
                    st.error(f"{file.name} exceeds 10MB")
                else:
                    r = requests.post(f"{API}/complaints/{result['id']}/evidence", files={"files": (file.name, file.getvalue(), file.type)})
                    st.success(f"{file.name} uploaded") if r.ok else st.error(f"{file.name} failed")
    result = st.session_state.result
    if result is not None:
        st.subheader(f"Complaint #{result['id']}")
        for col, label, val in zip(st.columns(3), ["Category", "Priority", "Unit"], [result["category"], result["priority"], result.get("assigned_unit", "-")]):
            col.metric(label, val)
        st.write(f"**Summary:** {result['summary']}")
        if result.get("risk_flags") and result["risk_flags"][0] != "none_identified":
            st.warning(", ".join(result["risk_flags"]))
        if result.get("followup_questions"):
            with st.expander("Follow-up Questions"):
                for q in result["followup_questions"]: st.write(f"- {q}")

elif page == "track":
    st.title("Track Complaint")
    cid = st.number_input("Complaint ID", min_value=1, step=1)
    if st.button("Search", type="primary") and cid:
        r = api("GET", f"/complaints/{cid}")
        if r is not None: st.session_state.result = r
    r = st.session_state.result
    if r is not None and isinstance(r, dict) and "id" in r:
        for col, label, val in zip(st.columns(3), ["Status", "Priority", "Category"], [r["status"], r["priority"], r["category"]]):
            col.metric(label, val)
        st.write(f"**Summary:** {r['summary']}")
        if r.get("assigned_unit"): st.info(f"**Assigned:** {r['assigned_unit']}")

elif page == "dashboard":
    st.title("Officer Dashboard")
    if st.button("Refresh") or not st.session_state.complaints:
        data = api("GET", "/complaints")
        if data is not None: st.session_state.complaints = data
    complaints = st.session_state.complaints
    if complaints:
        high = sum(1 for c in complaints if c["priority"] in ("Emergency", "High"))
        assigned = sum(1 for c in complaints if c["status"] == "Assigned")
        closed = sum(1 for c in complaints if c["status"] in ("Resolved", "Closed"))
        for col, label, val in zip(st.columns(4), ["Total", "High/Emergency", "Assigned", "Closed"], [len(complaints), high, assigned, closed]):
            col.metric(label, val)
        pri = st.selectbox("Priority", ["All", "Emergency", "High", "Medium", "Low"])
        sta = st.selectbox("Status", ["All", "New", "Under Review", "Assigned", "Resolved", "Closed"])
        for c in complaints:
            if (pri != "All" and c["priority"] != pri) or (sta != "All" and c["status"] != sta): continue
            with st.expander(f"#{c['id']} | {c['category']} | {c['priority']} | {c['status']}"):
                st.write(c["summary"][:200])
                st.write(f"**Unit:** {c.get('assigned_unit', '-')}")
                if st.button(f"View #{c['id']}", key=f"v{c['id']}"):
                    st.session_state.selected = c
                    st.session_state.page = "detail"
                    st.rerun()

elif page == "detail":
    c = st.session_state.selected
    if c is not None:
        st.title(f"Complaint #{c['id']}")
        st.write(f"**Text:** {c['complaint_text'][:500]}\n\n**Summary:** {c['summary']}")
        for col, label, val in zip(st.columns(4), ["Category", "Priority", "Status", "Unit"], [c["category"], c["priority"], c["status"], c.get("assigned_unit", "-")]):
            col.metric(label, val)
        if c.get("risk_flags") and c["risk_flags"][0] != "none_identified":
            st.warning(", ".join(c["risk_flags"]))
        statuses = ["New", "Under Review", "Assigned", "Resolved", "Closed"]
        new_status = st.selectbox("Status", statuses, index=statuses.index(c["status"]))
        notes = st.text_area("Officer Notes", c.get("officer_notes", ""))
        if st.button("Update", type="primary"):
            result = api("PATCH", f"/complaints/{c['id']}/triage", data={"status": new_status, "officer_notes": notes})
            if result is not None:
                st.success("Updated!")
                st.session_state.selected = result
        ev = api("GET", f"/complaints/{c['id']}/evidence")
        if ev is not None and ev.get("evidence"):
            st.subheader("Evidence")
            for item in ev["evidence"]:
                st.markdown(f"{item['original_filename']} ({item['file_size'] / 1024:.0f} KB) | [Download]({API}/evidence/{item['id']})")

elif page == "evidence":
    st.title("Evidence Review")
    if st.button("Refresh") or not st.session_state.ev_cache:
        st.session_state.ev_cache = []
        all_c = api("GET", "/complaints")
        if all_c:
            for c in all_c:
                ev_data = api("GET", f"/complaints/{c['id']}/evidence")
                if ev_data and ev_data.get("evidence"):
                    for item in ev_data["evidence"]:
                        item["cid"], item["cat"] = c["id"], c["category"]
                        st.session_state.ev_cache.append(item)
    if st.session_state.ev_cache:
        for item in st.session_state.ev_cache:
            st.markdown(f"#{item['cid']} | {item['original_filename']} ({item['file_size'] / 1024:.0f} KB) | {item['cat']} | [Download]({API}/evidence/{item['id']})")
    else: st.info("No evidence found")

import streamlit as st, requests
API = "http://localhost:8000"
st.set_page_config(page_title="Police Portal", layout="wide")
for k in "page r sl ct ev".split():
    st.session_state.setdefault(k, [] if k in ("ct", "ev") else None)

def api(m, p, **kw):
    try: return requests.request(m, f"{API}{p}", timeout=10, **kw).json()
    except Exception as e: st.error(str(e)[:100])

with st.sidebar:
    st.title("🚔 Police Portal")
    for l, k in [("🏠 Home", "h"), ("📝 File", "f"), ("🔍 Track", "t"), ("📊 Dash", "d"), ("📁 Evidence", "e")]:
        if st.button(l, use_container_width=True): st.session_state.page = k; st.rerun()

p = st.session_state.page or "h"

if p == "h":
    st.title("Police Complaint Portal")
    st.info("AI-assisted complaint filing and triage system.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("### 📝 File a Complaint\nSubmit incidents with AI classification.")
    c2.markdown("### 🔍 Track Status\nEnter ID to check progress.")
    c3.markdown("### 📊 Officer Dashboard\nTriage and manage complaints.")

elif p == "f":
    st.title("📝 File a Complaint")
    t = st.text_area("Description*", height=100)
    nm, ph, em, lo, ti = [st.text_input(x) for x in ["Name", "Phone", "Email", "Location", "Time"]]
    f = st.file_uploader("Evidence (JPG/PNG/PDF/TXT, max 10MB)", accept_multiple_files=True, type=["jpg", "jpeg", "png", "pdf", "txt"])
    if st.button("Submit", type="primary", disabled=not (t and len(t) >= 10)):
        r = api("POST", "/complaints", json={"complaint_text": t, **{k: v for k, v in zip(["reporter_name", "reporter_phone", "reporter_email", "incident_location", "incident_time"], [nm, ph, em, lo, ti]) if v}})
        if r:
            st.session_state.r = r; st.success(f"✅ #{r['id']} filed!")
            for fo in f:
                if fo.size > 10 * 1024 * 1024: st.error(f"{fo.name} exceeds 10MB")
                else:
                    s = requests.post(f"{API}/complaints/{r['id']}/evidence", files={"files": (fo.name, fo.getvalue(), fo.type)})
                    st.success(f"📎 {fo.name} uploaded") if s.ok else st.error(f"⚠️ {fo.name} failed")
    if (r := st.session_state.r):
        st.divider(); st.subheader(f"Complaint #{r['id']}")
        c1, c2, c3 = st.columns(3); c1.metric("Category", r["category"]); c2.metric("Priority", r["priority"]); c3.metric("Unit", r.get("assigned_unit", "-"))
        st.write(f"**Summary:** {r['summary']}")
        if r.get("risk_flags") and r["risk_flags"][0] != "none_identified": st.warning("🚨 " + ", ".join(r["risk_flags"]))
        if r.get("followup_questions"):
            with st.expander("Follow-up Questions"):
                for q in r["followup_questions"]: st.write(f"- {q}")

elif p == "t":
    st.title("🔍 Track Complaint")
    cid = st.number_input("Complaint ID", min_value=1, step=1)
    if st.button("Search", type="primary") and cid:
        if (r := api("GET", f"/complaints/{cid}")): st.session_state.r = r
    if (r := st.session_state.r) and isinstance(r, dict) and "id" in r:
        st.divider()
        c1, c2, c3 = st.columns(3); c1.metric("Status", r["status"]); c2.metric("Priority", r["priority"]); c3.metric("Category", r["category"])
        st.write(f"**Summary:** {r['summary']}")
        if r.get("assigned_unit"): st.info(f"**Assigned:** {r['assigned_unit']}")

elif p == "d":
    st.title("📊 Officer Dashboard")
    if st.button("🔄 Refresh") or not st.session_state.ct: st.session_state.ct = api("GET", "/complaints") or []
    d = st.session_state.ct
    if d:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", len(d)); c2.metric("🚨 High/Emergency", sum(1 for c in d if c["priority"] in ("Emergency", "High")))
        c3.metric("Assigned", sum(1 for c in d if c["status"] == "Assigned")); c4.metric("Closed", sum(1 for c in d if c["status"] in ("Resolved", "Closed")))
        pri = st.selectbox("Priority", ["All", "Emergency", "High", "Medium", "Low"]); sta = st.selectbox("Status", ["All", "New", "Under Review", "Assigned", "Resolved", "Closed"])
        flt = [c for c in d if (pri == "All" or c["priority"] == pri) and (sta == "All" or c["status"] == sta)]
        for c in flt:
            with st.expander(f"#{c['id']} | {c['category']} | {c['priority']} | {c['status']}"):
                st.write(c['summary'][:200]); st.write(f"**Unit:** {c.get('assigned_unit', '-')}")
                if st.button(f"View #{c['id']}", key=f"v{c['id']}"): st.session_state.sl = c; st.session_state.page = "detail"; st.rerun()

elif p == "detail":
    if (c := st.session_state.sl):
        st.title(f"Complaint #{c['id']}")
        st.write(f"**Text:** {c['complaint_text'][:500]}\n\n**Summary:** {c['summary']}")
        c1, c2, c3, c4 = st.columns(4); c1.metric("Category", c['category']); c2.metric("Priority", c['priority']); c3.metric("Status", c['status']); c4.metric("Unit", c.get('assigned_unit', '-'))
        if c.get("risk_flags") and c["risk_flags"][0] != "none_identified": st.warning("🚨 " + ", ".join(c["risk_flags"]))
        st.divider(); st.subheader("Update")
        ns = st.selectbox("Status", ["New", "Under Review", "Assigned", "Resolved", "Closed"], index=["New", "Under Review", "Assigned", "Resolved", "Closed"].index(c["status"]))
        nt = st.text_area("Officer Notes", c.get("officer_notes", ""))
        if st.button("Update", type="primary") and (r := api("PATCH", f"/complaints/{c['id']}/triage", json={"status": ns, "officer_notes": nt})): st.success("✅ Updated!"); st.session_state.sl = r
        st.divider(); st.subheader("Evidence")
        if (ev := api("GET", f"/complaints/{c['id']}/evidence")) and ev.get("evidence"):
            for e in ev["evidence"]: st.markdown(f"📄 {e['original_filename']} ({(e['file_size'] / 1024):.0f} KB) | [Download]({API}/evidence/{e['id']})")
        else: st.info("No evidence")

elif p == "e":
    st.title("📁 Evidence Review")
    if st.button("🔄 Refresh") or not st.session_state.ev:
        st.session_state.ev = []
        for c in (api("GET", "/complaints") or []):
            if (ev := api("GET", f"/complaints/{c['id']}/evidence")) and ev.get("evidence"):
                for e in ev["evidence"]: e["cid"], e["cat"] = c["id"], c["category"]; st.session_state.ev.append(e)
    if st.session_state.ev:
        for e in st.session_state.ev: st.markdown(f"📄 #{e['cid']} | {e['original_filename']} ({(e['file_size'] / 1024):.0f} KB) | {e['cat']} | [Download]({API}/evidence/{e['id']})")
    else: st.info("No evidence found")

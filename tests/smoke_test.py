#!/usr/bin/env python3
"""
Hardened Smoke Test v3 — Police Complaint AI Assistant Chatbot

Optimized for speed while maintaining maximum coverage. Groups tests
by category and runs them in parallel where possible.

Usage:
    1. Start the backend:  uvicorn backend.main:app --reload
    2. Run:                python tests/smoke_test.py
"""

import json
import time
import uuid
import statistics
import threading
import requests
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://localhost:8000"
REPORT_PATH = "tests/SMOKE_TEST_REPORT.md"

results = []
perf_data = []
_call_lock = threading.Lock()
_call_count = 0
_call_times = []


def log(scenario, assertion, passed, detail=""):
    results.append({
        "scenario": scenario,
        "assertion": assertion,
        "passed": passed,
        "detail": detail,
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {assertion}" + (f" — {detail}" if detail else ""), flush=True)


def req(method, url, **kwargs):
    global _call_count
    with _call_lock:
        _call_count += 1
    start = time.time()
    try:
        r = getattr(requests, method)(url, timeout=30, **kwargs)
        ms = (time.time() - start) * 1000
        with _call_lock:
            _call_times.append(ms)
        return r.status_code, (r.json() if r.status_code == 200 else r.text)
    except Exception as e:
        ms = (time.time() - start) * 1000
        return 0, str(e)


def chat(sid, msg):
    return req("post", f"{BASE_URL}/chat/complaint", json={"session_id": sid, "user_message": msg})

def file_chat(sid):
    return req("post", f"{BASE_URL}/chat/complaint/{sid}/file")

def get_sess(sid):
    return req("get", f"{BASE_URL}/chat/complaint/{sid}")

def direct(text, **kw):
    p = {"complaint_text": text}; p.update(kw)
    return req("post", f"{BASE_URL}/complaints", json=p)

def patch_triage(cid, **kw):
    return req("patch", f"{BASE_URL}/complaints/{cid}/triage", json=kw)

def del_sess(sid):
    return req("delete", f"{BASE_URL}/chat/complaint/{sid}")


# ===========================================================================
# Helper: full conversation flow
# ===========================================================================
def run_conv(scenario_name, sid, turns, expect_cat=None, expect_priority=None,
             expect_unit=None, expect_flags=None, expect_name=None):
    """Run a full multi-turn conversation, file it, validate triage.
    Returns (pass_count, total_count, filed_data)."""
    pc = tc = 0

    for i, msg in enumerate(turns):
        code, resp = chat(sid, msg)
        ok = code == 200
        log(scenario_name, f"Turn {i+1}: HTTP 200", ok, f"got {code}")
        pc += ok; tc += 1

        if ok:
            log(scenario_name, f"Turn {i+1}: agent_message", bool(resp.get("agent_message")))
            pc += bool(resp.get("agent_message")); tc += 1

            cf = resp.get("collected_fields", {})
            # complaint_text must never shrink
            ct = cf.get("complaint_text", "")
            if i > 0:
                # Store max seen length
                if not hasattr(run_conv, f"_{sid}_maxlen"):
                    setattr(run_conv, f"_{sid}_maxlen", 0)
                prev_max = getattr(run_conv, f"_{sid}_maxlen")
                grows = len(ct) >= prev_max - 10
                log(scenario_name, f"Turn {i+1}: complaint_text not shorter", grows,
                    f"prev_max={prev_max}, now={len(ct)}")
                pc += grows; tc += 1
            setattr(run_conv, f"_{sid}_maxlen", max(getattr(run_conv, f"_{sid}_maxlen", 0), len(ct)))

    # File
    fc, filed = file_chat(sid)
    log(scenario_name, "Filed: HTTP 200", fc == 200, f"got {fc}")
    pc += (fc == 200); tc += 1

    if fc == 200 and isinstance(filed, dict):
        if expect_cat:
            ok = filed.get("category") == expect_cat
            log(scenario_name, f"Category = {expect_cat}", ok, f"got: {filed.get('category')}")
            pc += ok; tc += 1
        if expect_priority:
            ok = expect_priority in (filed.get("priority"),)
            log(scenario_name, f"Priority = {expect_priority}", ok, f"got: {filed.get('priority')}")
            pc += ok; tc += 1
        if expect_unit:
            ok = filed.get("assigned_unit") == expect_unit
            log(scenario_name, f"Unit = {expect_unit}", ok, f"got: {filed.get('assigned_unit')}")
            pc += ok; tc += 1
        if expect_flags:
            rf = filed.get("risk_flags", [])
            for flag in expect_flags:
                ok = flag in rf
                log(scenario_name, f"Flag: {flag}", ok, f"got: {rf}")
                pc += ok; tc += 1
        if expect_name:
            ok = filed.get("reporter_name") == expect_name
            log(scenario_name, f"Name = {expect_name}", ok, f"got: {filed.get('reporter_name')}")
            pc += ok; tc += 1

        # Generic checks
        log(scenario_name, "Has summary", bool(filed.get("summary")))
        pc += bool(filed.get("summary")); tc += 1
        log(scenario_name, "Has triage_reason", bool(filed.get("triage_reason")))
        pc += bool(filed.get("triage_reason")); tc += 1

    return pc, tc, filed


# ===========================================================================
# CATEGORY 1: Multi-turn conversation integrity
# ===========================================================================
def c1_conversations():
    print("\n" + "=" * 60)
    print("C1: MULTI-TURN CONVERSATION INTEGRITY")
    print("=" * 60)
    pc = tc = 0

    # T1: Child Missing Emergency (8 turns)
    print("\n--- C1-T1: Child Missing Emergency (8 turns) ---")
    sid = f"hc-child-{uuid.uuid4().hex[:8]}"
    p, t, filed = run_conv(
        "C1:ChildMissing", sid,
        ["My 8-year-old son Arjun is missing from Shivaji Park",
         "He was wearing a red t-shirt with Spider-Man on it and blue shorts",
         "He had his school backpack, it's a green bag",
         "We were at the playground near the fountain around 4pm today",
         "I turned around for just two minutes and he was gone",
         "I've checked the snack shop and the basketball court, nobody's seen him",
         "His school is St. Mary's, he should have been home by 4:30",
         "Please hurry, I'm his mother Anita Desai, my phone is 9988776655"],
        expect_cat="child safety", expect_priority="Emergency",
        expect_unit="Child Protection Desk",
        expect_flags=["child_involved", "person_missing"],
        expect_name="Anita Desai",
    )
    pc += p; tc += t
    # Extra checks on filed complaint text
    if filed:
        ct = filed.get("complaint_text", "").lower()
        for kw in ["arjun", "shivaji", "missing"]:
            ok = kw in ct
            log("C1:ChildMissing", f"Filed text has '{kw}'", ok, f"text: {ct[:80]}...")
            pc += ok; tc += 1
        log("C1:ChildMissing", "Phone in response", filed.get("reporter_phone") == "9988776655")
        pc += (filed.get("reporter_phone") == "9988776655"); tc += 1

    # T2: Domestic Violence + Knife (6 turns)
    print("\n--- C1-T2: Domestic Violence + Knife (6 turns) ---")
    sid = f"hc-dv-{uuid.uuid4().hex[:8]}"
    p, t, filed = run_conv(
        "C1:DVWeapon", sid,
        ["My husband has been hitting me again tonight",
         "He slapped me and punched my arm, I have bruises",
         "Then he grabbed a kitchen knife and threatened to kill me",
         "My name is Kavita Reddy, we live in Andheri East, Mumbai",
         "It happened around 10pm tonight, our anniversary",
         "Yes please file this, I'm at my neighbor's house now"],
        expect_cat="women help desk", expect_priority="High",
        expect_unit="Women Help Desk",
        expect_flags=["weapon_involved", "injury_reported"],
        expect_name="Kavita Reddy",
    )
    pc += p; tc += t

    # T3: Road Accident Hit-and-Run (5 turns)
    print("\n--- C1-T3: Road Accident Hit-and-Run (5 turns) ---")
    sid = f"hc-road-{uuid.uuid4().hex[:8]}"
    p, t, filed = run_conv(
        "C1:RoadHitRun", sid,
        ["A car hit my motorcycle and drove off at full speed on the highway near Pune",
         "I'm bleeding from my head and my left arm is broken",
         "My name is Rajesh Verma, I'm 42, the car was a white Toyota",
         "This happened at 6:15pm today, there were witnesses at the bus stop",
         "Please file this urgently, I can't move my arm"],
        expect_cat="road accident", expect_priority="Emergency",
        expect_unit="Traffic Police",
        expect_flags=["injury_reported"],
    )
    pc += p; tc += t

    # T4: Cyber Fraud Escalation (5 turns)
    print("\n--- C1-T4: Cyber Fraud Escalation (5 turns) ---")
    sid = f"hc-cyber-{uuid.uuid4().hex[:8]}"
    p, t, filed = run_conv(
        "C1:CyberFraud", sid,
        ["Someone hacked my bank account and transferred 2 lakh rupees",
         "They used a phishing email pretending to be from SBI bank",
         "It happened yesterday, I noticed when I got the OTP alerts at 2am",
         "My name is Fatima Sheikh, I'm from Dharavi, Mumbai",
         "Yes, file it please"],
        expect_cat="cyber crime incident",
        expect_unit="Cyber Crime Cell",
        expect_flags=["digital_fraud"],
        expect_name="Fatima Sheikh",
    )
    pc += p; tc += t

    # T5: Fire with People Trapped (5 turns)
    print("\n--- C1-T5: Fire with People Trapped (5 turns) ---")
    sid = f"hc-fire-{uuid.uuid4().hex[:8]}"
    p, t, filed = run_conv(
        "C1:FireTrapped", sid,
        ["There's a fire in my apartment building, smoke is everywhere",
         "The fire started on the ground floor, it's spreading fast",
         "There are elderly people trapped on the 3rd floor, they can't get down",
         "My name is Sunil Mehta, building is Lakshmi Apartments on Linking Road, Bandra",
         "Fire brigade is on the way but please send someone now"],
        expect_cat="fire accident", expect_priority="Emergency",
        expect_unit="Fire and Emergency Coordination",
        expect_flags=["fire_risk"],
        expect_name="Sunil Mehta",
    )
    pc += p; tc += t

    # T6: Public Health Outbreak (4 turns)
    print("\n--- C1-T6: Public Health Outbreak (4 turns) ---")
    sid = f"hc-health-{uuid.uuid4().hex[:8]}"
    p, t, filed = run_conv(
        "C1:PublicHealth", sid,
        ["Multiple people in my neighborhood are getting sick from contaminated water",
         "At least 15 families are affected, children are having fever and vomiting",
         "This is in Sector 5, Noida, near the public water tank",
         "My name is Dr. Amit Patel, it started 3 days ago on Monday"],
        expect_cat="public healthcare", expect_priority="Medium",
        expect_unit="Public Health Coordination",
        expect_name="Dr. Amit Patel",
    )
    pc += p; tc += t

    return pc, tc


# ===========================================================================
# CATEGORY 2: Edge cases & security
# ===========================================================================
def c2_edge_cases():
    print("\n" + "=" * 60)
    print("C2: EDGE CASES & SECURITY")
    print("=" * 60)
    pc = tc = 0

    # Minimal complaint
    print("\n--- C2-T1: Minimal Complaint ---")
    c, r = direct("Help")
    log("C2:Minimal", "HTTP 200", c == 200); pc += (c == 200); tc += 1
    if c == 200:
        log("C2:Minimal", "Has category", bool(r.get("category"))); pc += bool(r.get("category")); tc += 1
        log("C2:Minimal", "Has priority", bool(r.get("priority"))); pc += bool(r.get("priority")); tc += 1

    # Very long complaint (5000+ chars)
    print("\n--- C2-T2: Very Long Complaint ---")
    long_text = ("I need help. " * 200) + "My house was broken into yesterday at 5pm on MG Road."
    c, r = direct(long_text)
    log("C2:Long", "HTTP 200", c == 200); pc += (c == 200); tc += 1
    if c == 200:
        log("C2:Long", "Summary present", bool(r.get("summary"))); pc += bool(r.get("summary")); tc += 1

    # Unicode/emoji
    print("\n--- C2-T3: Special Characters (Hindi + emoji) ---")
    c, r = direct("नमस्ते, मेरे साथ धोखा हुआ। ₹5000 चोरी हुए। #urgent 🚨")
    log("C2:Unicode", "HTTP 200", c == 200); pc += (c == 200); tc += 1

    # SQL injection
    print("\n--- C2-T4: SQL Injection ---")
    c, r = direct("'; DROP TABLE complaints; --")
    log("C2:SQLi", "HTTP 200 (not 500)", c == 200); pc += (c == 200); tc += 1
    c2, _ = direct("test after injection")
    log("C2:SQLi", "DB intact after injection", c2 == 200); pc += (c2 == 200); tc += 1

    # XSS
    print("\n--- C2-T5: XSS Attempt ---")
    c, r = direct('<script>alert("xss")</script>My house was robbed')
    log("C2:XSS", "HTTP 200", c == 200); pc += (c == 200); tc += 1
    if c == 200:
        log("C2:XSS", "No raw script in summary", "<script>" not in str(r.get("summary", "")))
        pc += ("<script>" not in str(r.get("summary", ""))); tc += 1

    # Invalid requests
    print("\n--- C2-T6: Invalid Requests ---")
    r1 = requests.post(f"{BASE_URL}/complaints", json={}, timeout=10)
    log("C2:Invalid", "Missing text → 422", r1.status_code == 422); pc += (r1.status_code == 422); tc += 1

    r2 = requests.post(f"{BASE_URL}/complaints", json={"complaint_text": ""}, timeout=10)
    log("C2:Invalid", "Empty text → 422", r2.status_code == 422); pc += (r2.status_code == 422); tc += 1

    r3 = requests.get(f"{BASE_URL}/complaints/999999", timeout=10)
    log("C2:Invalid", "Non-existent ID → 404", r3.status_code == 404); pc += (r3.status_code == 404); tc += 1

    r4 = requests.post(f"{BASE_URL}/chat/complaint", json={"user_message": "hi"}, timeout=10)
    log("C2:Invalid", "Missing session_id → 422", r4.status_code == 422); pc += (r4.status_code == 422); tc += 1

    r5 = requests.post(f"{BASE_URL}/chat/complaint", json={"session_id": "x", "user_message": ""}, timeout=10)
    log("C2:Invalid", "Empty message → 422", r5.status_code == 422); pc += (r5.status_code == 422); tc += 1

    r6 = requests.post(f"{BASE_URL}/chat/complaint/nonexistent/file", timeout=10)
    log("C2:Invalid", "File nonexistent session → 404", r6.status_code == 404); pc += (r6.status_code == 404); tc += 1

    return pc, tc


# ===========================================================================
# CATEGORY 3: Session isolation
# ===========================================================================
def c3_session_isolation():
    print("\n" + "=" * 60)
    print("C3: SESSION ISOLATION & DATA LEAKAGE")
    print("=" * 60)
    pc = tc = 0

    sid_a = f"iso-a-{uuid.uuid4().hex[:8]}"
    sid_b = f"iso-b-{uuid.uuid4().hex[:8]}"

    # Session A: cyber crime in Delhi
    print("\n--- C3-T1: Cross-Session Isolation ---")
    chat(sid_a, "I was scammed online, someone stole 50000 from my SBI account")
    chat(sid_a, "My name is TestUserA, phone 1111111111")
    chat(sid_a, "It happened in Delhi yesterday")

    # Session B: fire in Kolkata
    chat(sid_b, "There's a fire in my building on Park Street, Kolkata")
    chat(sid_b, "My name is TestUserB, phone 2222222222")
    chat(sid_b, "It started 10 minutes ago, people are trapped")

    ca, sa = get_sess(sid_a)
    cb, sb = get_sess(sid_b)

    if ca == 200 and cb == 200:
        ja = json.dumps(sa)
        jb = json.dumps(sb)

        log("C3:Isolation", "A has no B data (TestUserB)",
            "TestUserB" not in ja); pc += ("TestUserB" not in ja); tc += 1
        log("C3:Isolation", "B has no A data (TestUserA)",
            "TestUserA" not in jb); pc += ("TestUserA" not in jb); tc += 1
        log("C3:Isolation", "A has no Kolkata",
            "Kolkata" not in ja); pc += ("Kolkata" not in ja); tc += 1
        log("C3:Isolation", "B has no Delhi",
            "Delhi" not in jb); pc += ("Delhi" not in jb); tc += 1

    # File both and verify
    cfa, fa = file_chat(sid_a)
    cfb, fb = file_chat(sid_b)
    if cfa == 200 and cfb == 200:
        log("C3:Isolation", "A: category cyber crime", fa.get("category") == "cyber crime incident")
        pc += (fa.get("category") == "cyber crime incident"); tc += 1
        log("C3:Isolation", "B: category fire accident", fb.get("category") == "fire accident")
        pc += (fb.get("category") == "fire accident"); tc += 1
        log("C3:Isolation", "A: reporter = TestUserA", fa.get("reporter_name") == "TestUserA")
        pc += (fa.get("reporter_name") == "TestUserA"); tc += 1
        log("C3:Isolation", "B: reporter = TestUserB", fb.get("reporter_name") == "TestUserB")
        pc += (fb.get("reporter_name") == "TestUserB"); tc += 1

    # T2: Delete and reuse session
    print("\n--- C3-T2: Session Delete & Reuse ---")
    sid = f"del-{uuid.uuid4().hex[:8]}"
    chat(sid, "I was robbed at knifepoint in Bangalore")
    del_sess(sid)
    c2, r2 = chat(sid, "This is a different complaint about a flood in my basement")
    if c2 == 200:
        ct = r2.get("collected_fields", {}).get("complaint_text", "")
        log("C3:Delete", "No old 'knifepoint' in text", "knifepoint" not in ct.lower())
        pc += ("knifepoint" not in ct.lower()); tc += 1
        log("C3:Delete", "New 'flood' in text", "flood" in ct.lower())
        pc += ("flood" in ct.lower()); tc += 1

    return pc, tc


# ===========================================================================
# CATEGORY 4: Stress — 15-turn context memory
# ===========================================================================
def c4_stress():
    print("\n" + "=" * 60)
    print("C4: STRESS TEST — 15-TURN CONTEXT MEMORY")
    print("=" * 60)
    pc = tc = 0

    sid = f"stress-{uuid.uuid4().hex[:8]}"
    turns = [
        ("I need help filing a complaint", {}),
        ("My car was stolen last night", {}),
        ("It's a red Maruti Swift, registration MH12AB1234", {}),
        ("It was parked outside my house", {}),
        ("My house is at 45 Gandhi Nagar, Pune", {}),
        ("I noticed it missing around 6am this morning", {}),
        ("My name is Vikram Joshi", {"check_name": True}),
        ("My phone number is 9876543210", {"check_phone": True}),
        ("There was a CCTV camera nearby but I don't know if it was working", {}),
        ("I already checked with the local patrol, they said to file a report", {}),
        ("The car has a dent on the left rear bumper", {}),
        ("There's a baby seat in the back, my daughter's car seat", {}),
        ("I have the insurance papers at home", {}),
        ("This is the second time this year something was stolen from me", {}),
        ("Yes, I'm ready to file, please help me", {}),
    ]

    name_seen = False
    phone_seen = False
    for i, (msg, checks) in enumerate(turns):
        c, r = chat(sid, msg)
        if c != 200:
            log("C4:Stress", f"Turn {i+1}: HTTP 200", False); tc += 1
            continue

        cf = r.get("collected_fields", {})
        ct = cf.get("complaint_text", "")

        if checks.get("check_name"):
            name_seen = True
        if checks.get("check_phone"):
            phone_seen = True

        # After name provided, it must stay
        if name_seen and i > 6:
            ok = cf.get("reporter_name") == "Vikram Joshi"
            log("C4:Stress", f"Turn {i+1}: name preserved", ok, f"got: {cf.get('reporter_name')}")
            pc += ok; tc += 1

        if phone_seen and i > 7:
            ok = cf.get("reporter_phone") == "9876543210"
            log("C4:Stress", f"Turn {i+1}: phone preserved", ok, f"got: {cf.get('reporter_phone')}")
            pc += ok; tc += 1

    # File and verify ALL details
    fc, filed = file_chat(sid)
    if fc == 200 and isinstance(filed, dict):
        ct = filed.get("complaint_text", "").lower()
        checks = [
            ("maruti", "car make"),
            ("mh12ab1234", "registration"),
            ("gandhi", "location part 1"),
            ("pune", "location part 2"),
            ("cctv", "CCTV detail"),
            ("dent", "dent detail"),
            ("car seat", "car seat detail"),
        ]
        for kw, label in checks:
            ok = kw in ct
            log("C4:Stress", f"Filed has {label}", ok, f"text: {ct[:80]}...")
            pc += ok; tc += 1

        log("C4:Stress", "Filed: name = Vikram Joshi", filed.get("reporter_name") == "Vikram Joshi")
        pc += (filed.get("reporter_name") == "Vikram Joshi"); tc += 1
        log("C4:Stress", "Filed: phone = 9876543210", filed.get("reporter_phone") == "9876543210")
        pc += (filed.get("reporter_phone") == "9876543210"); tc += 1

    return pc, tc


# ===========================================================================
# CATEGORY 5: All 8 categories triage
# ===========================================================================
def c5_all_categories():
    print("\n" + "=" * 60)
    print("C5: TRIAGE ACCURACY — ALL 8 CATEGORIES")
    print("=" * 60)
    pc = tc = 0

    cases = [
        ("My 5-year-old daughter is missing from the school playground",
         "child safety", "Child Protection Desk", "Emergency"),
        ("Someone hacked my email and is blackmailing me with personal photos",
         "cyber crime incident", "Cyber Crime Cell", None),
        ("My neighbor is harassing me, following me and sending threatening messages",
         "women help desk", "Women Help Desk", None),
        ("Multiple people are getting food poisoning from the restaurant on Main Street",
         "public healthcare", "Public Health Coordination", None),
        ("Two cars collided at the signal on Ring Road, both drivers are injured",
         "road accident", "Traffic Police", "High"),
        ("I found a body in the alley behind the market, there are signs of violence",
         "murder / serious crime incident", "Serious Crime Unit", "Emergency"),
        ("A gas cylinder exploded in the kitchen, the house is on fire",
         "fire accident", "Fire and Emergency Coordination", None),
        ("My neighbor is playing loud music at 3am repeatedly",
         "general issue recorded", "General Desk", "Low"),
    ]

    for text, exp_cat, exp_unit, exp_pri in cases:
        label = exp_cat[:25]
        c, r = direct(text)
        log(f"C5:{label}", "HTTP 200", c == 200); pc += (c == 200); tc += 1
        if c == 200:
            ok = r.get("category") == exp_cat
            log(f"C5:{label}", f"Category = {exp_cat}", ok, f"got: {r.get('category')}")
            pc += ok; tc += 1
            ok = r.get("assigned_unit") == exp_unit
            log(f"C5:{label}", f"Unit = {exp_unit}", ok, f"got: {r.get('assigned_unit')}")
            pc += ok; tc += 1
            if exp_pri:
                ok = r.get("priority") == exp_pri
                log(f"C5:{label}", f"Priority = {exp_pri}", ok, f"got: {r.get('priority')}")
                pc += ok; tc += 1
            ok = bool(r.get("followup_questions"))
            log(f"C5:{label}", "Has followup_questions", ok)
            pc += ok; tc += 1

    return pc, tc


# ===========================================================================
# CATEGORY 6: Triage keyword boundaries
# ===========================================================================
def c6_keywords():
    print("\n" + "=" * 60)
    print("C6: TRIAGE KEYWORD BOUNDARIES")
    print("=" * 60)
    pc = tc = 0

    kw_tests = [
        ("My child was kidnapped at gunpoint", "Emergency", ["weapon_involved", "child_involved", "person_missing"]),
        ("There's an active fire spreading through the building", "Emergency", ["fire_risk"]),
        ("Someone was shot and is bleeding on the street", "Emergency", ["injury_reported", "weapon_involved"]),
        ("I was robbed, they had a knife", "High", ["weapon_involved"]),
        ("I slipped and broke my arm, need medical help", "High", ["injury_reported"]),
        ("My bicycle was stolen from the park", "Low", ["none_identified"]),
        ("Someone is stalking me online, creating fake profiles", "Medium", ["digital_fraud"]),
        ("There's a gas leak in my neighborhood, smell is strong", "Medium", ["fire_risk"]),
    ]

    for text, exp_pri, exp_flags in kw_tests:
        label = text[:30]
        c, r = direct(text)
        if c == 200:
            ok = exp_pri in r.get("priority", "")
            log(f"C6:Kw", f"'{label}' → Priority {exp_pri}", ok, f"got: {r.get('priority')}")
            pc += ok; tc += 1
            rf = r.get("risk_flags", [])
            for flag in exp_flags:
                ok = flag in rf
                log(f"C6:Kw", f"Flag: {flag}", ok, f"got: {rf}")
                pc += ok; tc += 1

    return pc, tc


# ===========================================================================
# CATEGORY 7: API endpoint coverage
# ===========================================================================
def c7_endpoints():
    print("\n" + "=" * 60)
    print("C7: API ENDPOINT COVERAGE")
    print("=" * 60)
    pc = tc = 0

    # GET /complaints
    c, _ = req("get", f"{BASE_URL}/complaints")
    log("C7:API", "GET /complaints → 200", c == 200); pc += (c == 200); tc += 1

    # Create one for further tests
    c2, r2 = direct("endpoint test")
    if c2 == 200:
        cid = r2.get("id")
        c3, _ = req("get", f"{BASE_URL}/complaints/{cid}")
        log("C7:API", "GET /complaints/{id} → 200", c3 == 200); pc += (c3 == 200); tc += 1

        c4, _ = patch_triage(cid, status="Under Review", officer_notes="Test note")
        log("C7:API", "PATCH /triage → 200", c4 == 200); pc += (c4 == 200); tc += 1

        c5, _ = patch_triage(cid, status="INVALID")
        log("C7:API", "PATCH /triage invalid → 422", c5 == 422); pc += (c5 == 422); tc += 1

    c6, _ = req("get", f"{BASE_URL}/chat/complaint")
    log("C7:API", "GET /chat/complaint → 200", c6 == 200); pc += (c6 == 200); tc += 1

    c7, _ = req("get", f"{BASE_URL}/chat/complaint/nonexistent")
    log("C7:API", "GET /chat nonexistent → 404", c7 == 404); pc += (c7 == 404); tc += 1

    c8, _ = req("delete", f"{BASE_URL}/chat/complaint/nonexistent")
    log("C7:API", "DELETE /chat nonexistent → 404", c8 == 404); pc += (c8 == 404); tc += 1

    return pc, tc


# ===========================================================================
# CATEGORY 8: Performance
# ===========================================================================
def c8_performance():
    print("\n" + "=" * 60)
    print("C8: PERFORMANCE BENCHMARKS")
    print("=" * 60)
    pc = tc = 0

    # Direct complaint times
    times_d = []
    for i in range(3):
        c, r = req("post", f"{BASE_URL}/complaints",
                    json={"complaint_text": f"Perf test {i}: someone broke into my car on MG Road"})
        if c == 200:
            times_d.append(r.get("_elapsed_ms", 0) or 0)

    # Measure explicitly
    times_d = []
    for i in range(3):
        start = time.time()
        requests.post(f"{BASE_URL}/complaints",
                      json={"complaint_text": f"Perf test {i}: someone broke into my car on MG Road"},
                      timeout=30)
        times_d.append((time.time() - start) * 1000)

    avg_d = statistics.mean(times_d)
    log("C8:Perf", f"Direct complaint avg < 10s ({avg_d:.0f}ms)", avg_d < 10000)
    pc += (avg_d < 10000); tc += 1
    perf_data.append(("Direct complaint avg", avg_d))

    # Chat turn times
    sid = f"perf-{uuid.uuid4().hex[:8]}"
    times_c = []
    for msg in ["I was scammed", "My name is PerfTest, phone 1234567890", "It happened in Chennai yesterday", "Yes file it"]:
        start = time.time()
        requests.post(f"{BASE_URL}/chat/complaint",
                      json={"session_id": sid, "user_message": msg}, timeout=30)
        times_c.append((time.time() - start) * 1000)

    avg_c = statistics.mean(times_c)
    log("C8:Perf", f"Chat turn avg < 6s ({avg_c:.0f}ms)", avg_c < 6000)
    pc += (avg_c < 6000); tc += 1
    log("C8:Perf", "All chat turns < 15s", all(t < 15000 for t in times_c))
    pc += all(t < 15000 for t in times_c); tc += 1
    perf_data.append(("Chat turn avg", avg_c))

    # Session listing
    start = time.time()
    requests.get(f"{BASE_URL}/chat/complaint", timeout=10)
    ms = (time.time() - start) * 1000
    log("C8:Perf", f"Session listing < 2s ({ms:.0f}ms)", ms < 2000)
    pc += (ms < 2000); tc += 1

    return pc, tc


# ===========================================================================
# CATEGORY 9: Concurrent sessions
# ===========================================================================
def c9_concurrent():
    print("\n" + "=" * 60)
    print("C9: CONCURRENT SESSION STRESS (5 sessions)")
    print("=" * 60)
    pc = tc = 0

    def run_one(idx):
        sid = f"conc-{idx}-{uuid.uuid4().hex[:8]}"
        cats = ["stolen car", "house fire", "missing child", "cyber fraud", "road accident"]
        name = f"ConcurrentUser{idx}"
        chat(sid, f"I need help with a {cats[idx]} complaint")
        chat(sid, f"My name is {name}, phone 555555{idx:04d}")
        chat(sid, f"It happened in City{idx} yesterday at {idx+1}pm")
        c, r = chat(sid, "Yes file it please")
        return sid

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(run_one, i): i for i in range(5)}
        sids = {}
        for f in as_completed(futures):
            idx = futures[f]
            try:
                sids[idx] = f.result()
            except Exception as e:
                log("C9:Concurrent", f"Session {idx} error", False, str(e)); tc += 1

    for idx, sid in sids.items():
        c, filed = file_chat(sid)
        name = f"ConcurrentUser{idx}"
        if c == 200 and isinstance(filed, dict):
            ok = filed.get("reporter_name") == name
            log("C9:Concurrent", f"Session {idx}: reporter = {name}", ok,
                f"got: {filed.get('reporter_name')}")
            pc += ok; tc += 1
            ok = len(filed.get("complaint_text", "")) > 10
            log("C9:Concurrent", f"Session {idx}: has complaint_text", ok,
                f"text: {filed.get('complaint_text', '')[:50]}...")
            pc += ok; tc += 1

    return pc, tc


# ===========================================================================
# CATEGORY 10: Officer workflow
# ===========================================================================
def c10_officer():
    print("\n" + "=" * 60)
    print("C10: OFFICER TRIAGE WORKFLOW")
    print("=" * 60)
    pc = tc = 0

    c, r = direct("There's been a road accident with injuries on highway 66 near Mangalore")
    if c != 200:
        log("C10:Officer", "Complaint created", False); tc += 1
        return pc, tc

    cid = r.get("id")
    log("C10:Officer", f"Complaint #{cid} created", True); pc += 1; tc += 1

    c2, r2 = patch_triage(cid, status="Under Review", officer_notes="Assigned to SI Patel")
    log("C10:Officer", "PATCH → 200", c2 == 200); pc += (c2 == 200); tc += 1
    if c2 == 200:
        log("C10:Officer", "Status = Under Review", r2.get("status") == "Under Review")
        pc += (r2.get("status") == "Under Review"); tc += 1
        log("C10:Officer", "Notes set", r2.get("officer_notes") == "Assigned to SI Patel")
        pc += (r2.get("officer_notes") == "Assigned to SI Patel"); tc += 1
        log("C10:Officer", "updated_at set", r2.get("updated_at") is not None)
        pc += (r2.get("updated_at") is not None); tc += 1

    c3, r3 = req("get", f"{BASE_URL}/complaints/{cid}")
    if c3 == 200:
        log("C10:Officer", "GET shows updated status", r3.get("status") == "Under Review")
        pc += (r3.get("status") == "Under Review"); tc += 1

    return pc, tc


# ===========================================================================
# CATEGORY 11: Filing confirmation variants
# ===========================================================================
def c11_filing():
    print("\n" + "=" * 60)
    print("C11: FILING CONFIRMATION VARIANTS")
    print("=" * 60)
    pc = tc = 0

    phrases = ["yes", "yes file it", "please file this", "go ahead"]
    for phrase in phrases:
        sid = f"file-{uuid.uuid4().hex[:8]}"
        chat(sid, "I was robbed at knife point in Bangalore, my name is Test File User")
        chat(sid, "It happened yesterday at 5pm near MG Road")
        chat(sid, "Phone 9999990000")
        c, r = chat(sid, phrase)
        if c == 200:
            ok = r.get("ready_to_file") is True
            log("C11:Filing", f"'{phrase}' → ready_to_file=True", ok,
                f"got: {r.get('ready_to_file')}")
            pc += ok; tc += 1

    return pc, tc


# ===========================================================================
# CATEGORY 12: Complaint text data integrity
# ===========================================================================
def c12_integrity():
    print("\n" + "=" * 60)
    print("C12: COMPLAINT TEXT DATA INTEGRITY (10 details)")
    print("=" * 60)
    pc = tc = 0

    sid = f"integ-{uuid.uuid4().hex[:8]}"
    details = [
        ("My blue Honda City was stolen", ["honda", "city", "blue"]),
        ("Registration GA01AB5678", ["ga01ab5678"]),
        ("It was parked at Panaji Bus Stand, Goa", ["panaji", "goa"]),
        ("Last night around 11pm", []),  # time → field
        ("My name is Rohan Naik", []),  # name → field
        ("Phone 9876543210", []),  # phone → field
        ("A security guard named Raju saw someone near the car", ["raju"]),
        ("The side mirror was already cracked", ["cracked", "mirror"]),
        ("I have comprehensive insurance with ICICI Lombard", ["icici", "lombard"]),
        ("The car is worth about 12 lakh rupees", ["12 lakh"]),
    ]

    for i, (msg, keywords) in enumerate(details):
        c, r = chat(sid, msg)
        if c != 200:
            log("C12:Integrity", f"Turn {i+1}: HTTP 200", False); tc += 1
            continue
        cf = r.get("collected_fields", {})
        ct = cf.get("complaint_text", "")

        if msg.startswith("My name"):
            ok = cf.get("reporter_name") == "Rohan Naik"
            log("C12:Integrity", "Name extracted", ok, f"got: {cf.get('reporter_name')}")
            pc += ok; tc += 1
        elif msg.startswith("Phone"):
            ok = cf.get("reporter_phone") == "9876543210"
            log("C12:Integrity", "Phone extracted", ok, f"got: {cf.get('reporter_phone')}")
            pc += ok; tc += 1
        else:
            for kw in keywords:
                ok = kw.lower() in ct.lower()
                log("C12:Integrity", f"'{kw}' in complaint_text", ok,
                    f"text: {ct[:70]}...")
                pc += ok; tc += 1

    # File and verify final text has ALL details
    fc, filed = file_chat(sid)
    if fc == 200 and isinstance(filed, dict):
        ct = filed.get("complaint_text", "").lower()
        all_checks = [
            ("honda", "car make"),
            ("ga01ab5678", "registration"),
            ("panaji", "location"),
            ("raju", "witness"),
            ("icici", "insurance"),
            ("12 lakh", "value"),
            ("cracked", "damage"),
        ]
        for kw, label in all_checks:
            ok = kw in ct
            log("C12:Integrity", f"Filed has {label}", ok, f"text: {ct[:80]}...")
            pc += ok; tc += 1

        log("C12:Integrity", "Filed: name = Rohan Naik", filed.get("reporter_name") == "Rohan Naik")
        pc += (filed.get("reporter_name") == "Rohan Naik"); tc += 1
        log("C12:Integrity", "Filed: phone = 9876543210", filed.get("reporter_phone") == "9876543210")
        pc += (filed.get("reporter_phone") == "9876543210"); tc += 1

    return pc, tc


# ===========================================================================
# Report generation
# ===========================================================================
def generate_report():
    now = datetime.now(timezone.utc).isoformat()
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    lines = [
        "# Hardened Smoke Test Report — Police Complaint AI Assistant",
        "",
        f"**Date**: {now}",
        f"**Backend**: {BASE_URL}",
        f"**Test Suite**: Hardened v3 (edge cases, security, performance, stress, isolation)",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total assertions | {total} |",
        f"| Passed | {passed} |",
        f"| Failed | {failed} |",
        f"| Pass rate | {passed/total*100:.1f}% |" if total else "| Pass rate | N/A |",
        f"| API calls made | {_call_count} |",
        f"| Avg API latency | {statistics.mean(_call_times):.0f}ms |" if _call_times else "",
        "",
    ]

    if perf_data:
        lines += ["### Performance", "", "| Operation | Avg Latency |", "|-----------|-------------|"]
        for label, ms in perf_data:
            lines.append(f"| {label} | {ms:.0f}ms |")
        lines.append("")

    # Group by category
    cats = [
        ("C1", "Multi-turn Conversation Integrity"),
        ("C2", "Edge Cases & Security"),
        ("C3", "Session Isolation"),
        ("C4", "Stress Test — Context Memory"),
        ("C5", "Triage Accuracy — All Categories"),
        ("C6", "Triage Keyword Boundaries"),
        ("C7", "API Endpoint Coverage"),
        ("C8", "Performance Benchmarks"),
        ("C9", "Concurrent Session Stress"),
        ("C10", "Officer Workflow"),
        ("C11", "Filing Confirmation Variants"),
        ("C12", "Data Integrity"),
    ]

    for cat_id, cat_title in cats:
        cat_results = [r for r in results if r["scenario"].startswith(cat_id)]
        if not cat_results:
            continue
        s_pass = sum(1 for r in cat_results if r["passed"])
        s_total = len(cat_results)
        lines += [
            f"## {cat_id}: {cat_title}",
            "",
            f"**Result**: {s_pass}/{s_total} passed",
            "",
            "| # | Assertion | Result | Detail |",
            "|---|-----------|--------|--------|",
        ]
        for j, r in enumerate(cat_results, 1):
            mark = "PASS" if r["passed"] else "FAIL"
            detail = (r["detail"] or "")[:80].replace("|", "\\|")
            lines.append(f"| {j} | {r['assertion'][:55]} | {mark} | {detail} |")
        lines.append("")

    # Verdict
    lines += ["## Verdict", ""]
    if failed == 0:
        lines.append("**ALL TESTS PASSED.** System performing as expected under hardened testing.")
    else:
        lines.append(f"**{failed} ASSERTION(S) FAILED.** Review details above.")
        lines.append("")
        lines.append("### Failed Assertions")
        lines.append("")
        for r in results:
            if not r["passed"]:
                lines.append(f"- **[{r['scenario']}]** {r['assertion']}: {r['detail']}")
        lines.append("")

    report = "\n".join(lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\nReport written to {REPORT_PATH}")
    return report


# ===========================================================================
# Main
# ===========================================================================
def main():
    start = time.time()
    print("=" * 70)
    print("HARDENED SMOKE TEST v3 — Police Complaint AI Assistant")
    print(f"Target: {BASE_URL}")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    try:
        r = requests.get(f"{BASE_URL}/docs", timeout=5)
        if r.status_code != 200:
            print(f"WARNING: Server returned {r.status_code}")
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to {BASE_URL}")
        return

    total_pc = total_tc = 0

    for name, fn in [
        ("C1: Conversations", c1_conversations),
        ("C2: Edge Cases", c2_edge_cases),
        ("C3: Isolation", c3_session_isolation),
        ("C4: Stress", c4_stress),
        ("C5: Categories", c5_all_categories),
        ("C6: Keywords", c6_keywords),
        ("C7: Endpoints", c7_endpoints),
        ("C8: Performance", c8_performance),
        ("C9: Concurrent", c9_concurrent),
        ("C10: Officer", c10_officer),
        ("C11: Filing", c11_filing),
        ("C12: Integrity", c12_integrity),
    ]:
        elapsed = time.time() - start
        print(f"\n>>> [{elapsed:.0f}s, {_call_count} calls] Running {name}", flush=True)
        pc, tc = fn()
        total_pc += pc
        total_tc += tc
        print(f"    Running total: {total_pc}/{total_tc} passed", flush=True)

    elapsed = time.time() - start
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    print(f"\n{'=' * 70}")
    print(f"COMPLETED in {elapsed:.1f}s")
    print(f"Total: {total} | Passed: {passed} | Failed: {failed} | Pass rate: {passed/total*100:.1f}%" if total else "No results")
    print(f"API calls: {_call_count} | Avg latency: {statistics.mean(_call_times):.0f}ms" if _call_times else "")
    print(f"{'=' * 70}")

    generate_report()


if __name__ == "__main__":
    main()

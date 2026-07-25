# Hardened Smoke Test Report — Police Complaint AI Assistant

**Date**: 2026-07-25T13:23:40.797520+00:00
**Backend**: http://localhost:8000
**Test Suite**: Hardened v3 (edge cases, security, performance, stress, isolation)

## Summary

| Metric | Value |
|--------|-------|
| Total assertions | 303 |
| Passed | 303 |
| Failed | 0 |
| Pass rate | 100.0% |
| API calls made | 156 |
| Avg API latency | 3916ms |

### Performance

| Operation | Avg Latency |
|-----------|-------------|
| Direct complaint avg | 3749ms |
| Chat turn avg | 390ms |

## C1: Multi-turn Conversation Integrity

**Result**: 177/177 passed

| # | Assertion | Result | Detail |
|---|-----------|--------|--------|
| 1 | Turn 1: HTTP 200 | PASS | got 200 |
| 2 | Turn 1: agent_message | PASS |  |
| 3 | Turn 2: HTTP 200 | PASS | got 200 |
| 4 | Turn 2: agent_message | PASS |  |
| 5 | Turn 2: complaint_text not shorter | PASS | prev_max=52, now=121 |
| 6 | Turn 3: HTTP 200 | PASS | got 200 |
| 7 | Turn 3: agent_message | PASS |  |
| 8 | Turn 3: complaint_text not shorter | PASS | prev_max=121, now=166 |
| 9 | Turn 4: HTTP 200 | PASS | got 200 |
| 10 | Turn 4: agent_message | PASS |  |
| 11 | Turn 4: complaint_text not shorter | PASS | prev_max=166, now=228 |
| 12 | Turn 5: HTTP 200 | PASS | got 200 |
| 13 | Turn 5: agent_message | PASS |  |
| 14 | Turn 5: complaint_text not shorter | PASS | prev_max=228, now=283 |
| 15 | Turn 6: HTTP 200 | PASS | got 200 |
| 16 | Turn 6: agent_message | PASS |  |
| 17 | Turn 6: complaint_text not shorter | PASS | prev_max=283, now=355 |
| 18 | Turn 7: HTTP 200 | PASS | got 200 |
| 19 | Turn 7: agent_message | PASS |  |
| 20 | Turn 7: complaint_text not shorter | PASS | prev_max=355, now=415 |
| 21 | Turn 8: HTTP 200 | PASS | got 200 |
| 22 | Turn 8: agent_message | PASS |  |
| 23 | Turn 8: complaint_text not shorter | PASS | prev_max=415, now=481 |
| 24 | Filed: HTTP 200 | PASS | got 200 |
| 25 | Category = child safety | PASS | got: child safety |
| 26 | Priority = Emergency | PASS | got: Emergency |
| 27 | Unit = Child Protection Desk | PASS | got: Child Protection Desk |
| 28 | Flag: child_involved | PASS | got: ['child_involved', 'person_missing'] |
| 29 | Flag: person_missing | PASS | got: ['child_involved', 'person_missing'] |
| 30 | Name = Anita Desai | PASS | got: Anita Desai |
| 31 | Has summary | PASS |  |
| 32 | Has triage_reason | PASS |  |
| 33 | Filed text has 'arjun' | PASS | text: my 8-year-old son arjun is missing from shivaji park. he was wearing a red |
| 34 | Filed text has 'shivaji' | PASS | text: my 8-year-old son arjun is missing from shivaji park. he was wearing a red |
| 35 | Filed text has 'missing' | PASS | text: my 8-year-old son arjun is missing from shivaji park. he was wearing a red |
| 36 | Phone in response | PASS |  |
| 37 | Turn 1: HTTP 200 | PASS | got 200 |
| 38 | Turn 1: agent_message | PASS |  |
| 39 | Turn 2: HTTP 200 | PASS | got 200 |
| 40 | Turn 2: agent_message | PASS |  |
| 41 | Turn 2: complaint_text not shorter | PASS | prev_max=44, now=94 |
| 42 | Turn 3: HTTP 200 | PASS | got 200 |
| 43 | Turn 3: agent_message | PASS |  |
| 44 | Turn 3: complaint_text not shorter | PASS | prev_max=94, now=153 |
| 45 | Turn 4: HTTP 200 | PASS | got 200 |
| 46 | Turn 4: agent_message | PASS |  |
| 47 | Turn 4: complaint_text not shorter | PASS | prev_max=153, now=211 |
| 48 | Turn 5: HTTP 200 | PASS | got 200 |
| 49 | Turn 5: agent_message | PASS |  |
| 50 | Turn 5: complaint_text not shorter | PASS | prev_max=211, now=261 |
| 51 | Turn 6: HTTP 200 | PASS | got 200 |
| 52 | Turn 6: agent_message | PASS |  |
| 53 | Turn 6: complaint_text not shorter | PASS | prev_max=261, now=261 |
| 54 | Filed: HTTP 200 | PASS | got 200 |
| 55 | Category = women help desk | PASS | got: women help desk |
| 56 | Priority = High | PASS | got: High |
| 57 | Unit = Women Help Desk | PASS | got: Women Help Desk |
| 58 | Flag: weapon_involved | PASS | got: ['injury_reported', 'weapon_involved'] |
| 59 | Flag: injury_reported | PASS | got: ['injury_reported', 'weapon_involved'] |
| 60 | Name = Kavita Reddy | PASS | got: Kavita Reddy |
| 61 | Has summary | PASS |  |
| 62 | Has triage_reason | PASS |  |
| 63 | Turn 1: HTTP 200 | PASS | got 200 |
| 64 | Turn 1: agent_message | PASS |  |
| 65 | Turn 2: HTTP 200 | PASS | got 200 |
| 66 | Turn 2: agent_message | PASS |  |
| 67 | Turn 2: complaint_text not shorter | PASS | prev_max=76, now=130 |
| 68 | Turn 3: HTTP 200 | PASS | got 200 |
| 69 | Turn 3: agent_message | PASS |  |
| 70 | Turn 3: complaint_text not shorter | PASS | prev_max=130, now=190 |
| 71 | Turn 4: HTTP 200 | PASS | got 200 |
| 72 | Turn 4: agent_message | PASS |  |
| 73 | Turn 4: complaint_text not shorter | PASS | prev_max=190, now=259 |
| 74 | Turn 5: HTTP 200 | PASS | got 200 |
| 75 | Turn 5: agent_message | PASS |  |
| 76 | Turn 5: complaint_text not shorter | PASS | prev_max=259, now=308 |
| 77 | Filed: HTTP 200 | PASS | got 200 |
| 78 | Category = road accident | PASS | got: road accident |
| 79 | Priority = Emergency | PASS | got: Emergency |
| 80 | Unit = Traffic Police | PASS | got: Traffic Police |
| 81 | Flag: injury_reported | PASS | got: ['injury_reported'] |
| 82 | Has summary | PASS |  |
| 83 | Has triage_reason | PASS |  |
| 84 | Turn 1: HTTP 200 | PASS | got 200 |
| 85 | Turn 1: agent_message | PASS |  |
| 86 | Turn 2: HTTP 200 | PASS | got 200 |
| 87 | Turn 2: agent_message | PASS |  |
| 88 | Turn 2: complaint_text not shorter | PASS | prev_max=60, now=120 |
| 89 | Turn 3: HTTP 200 | PASS | got 200 |
| 90 | Turn 3: agent_message | PASS |  |
| 91 | Turn 3: complaint_text not shorter | PASS | prev_max=120, now=186 |
| 92 | Turn 4: HTTP 200 | PASS | got 200 |
| 93 | Turn 4: agent_message | PASS |  |
| 94 | Turn 4: complaint_text not shorter | PASS | prev_max=186, now=238 |
| 95 | Turn 5: HTTP 200 | PASS | got 200 |
| 96 | Turn 5: agent_message | PASS |  |
| 97 | Turn 5: complaint_text not shorter | PASS | prev_max=238, now=238 |
| 98 | Filed: HTTP 200 | PASS | got 200 |
| 99 | Category = cyber crime incident | PASS | got: cyber crime incident |
| 100 | Unit = Cyber Crime Cell | PASS | got: Cyber Crime Cell |
| 101 | Flag: digital_fraud | PASS | got: ['digital_fraud'] |
| 102 | Name = Fatima Sheikh | PASS | got: Fatima Sheikh |
| 103 | Has summary | PASS |  |
| 104 | Has triage_reason | PASS |  |
| 105 | Turn 1: HTTP 200 | PASS | got 200 |
| 106 | Turn 1: agent_message | PASS |  |
| 107 | Turn 2: HTTP 200 | PASS | got 200 |
| 108 | Turn 2: agent_message | PASS |  |
| 109 | Turn 2: complaint_text not shorter | PASS | prev_max=60, now=119 |
| 110 | Turn 3: HTTP 200 | PASS | got 200 |
| 111 | Turn 3: agent_message | PASS |  |
| 112 | Turn 3: complaint_text not shorter | PASS | prev_max=119, now=191 |
| 113 | Turn 4: HTTP 200 | PASS | got 200 |
| 114 | Turn 4: agent_message | PASS |  |
| 115 | Turn 4: complaint_text not shorter | PASS | prev_max=191, now=271 |
| 116 | Turn 5: HTTP 200 | PASS | got 200 |
| 117 | Turn 5: agent_message | PASS |  |
| 118 | Turn 5: complaint_text not shorter | PASS | prev_max=271, now=327 |
| 119 | Filed: HTTP 200 | PASS | got 200 |
| 120 | Category = fire accident | PASS | got: fire accident |
| 121 | Priority = Emergency | PASS | got: Emergency |
| 122 | Unit = Fire and Emergency Coordination | PASS | got: Fire and Emergency Coordination |
| 123 | Flag: fire_risk | PASS | got: ['fire_risk'] |
| 124 | Name = Sunil Mehta | PASS | got: Sunil Mehta |
| 125 | Has summary | PASS |  |
| 126 | Has triage_reason | PASS |  |
| 127 | Turn 1: HTTP 200 | PASS | got 200 |
| 128 | Turn 1: agent_message | PASS |  |
| 129 | Turn 2: HTTP 200 | PASS | got 200 |
| 130 | Turn 2: agent_message | PASS |  |
| 131 | Turn 2: complaint_text not shorter | PASS | prev_max=75, now=150 |
| 132 | Turn 3: HTTP 200 | PASS | got 200 |
| 133 | Turn 3: agent_message | PASS |  |
| 134 | Turn 3: complaint_text not shorter | PASS | prev_max=150, now=206 |
| 135 | Turn 4: HTTP 200 | PASS | got 200 |
| 136 | Turn 4: agent_message | PASS |  |
| 137 | Turn 4: complaint_text not shorter | PASS | prev_max=206, now=266 |
| 138 | Filed: HTTP 200 | PASS | got 200 |
| 139 | Category = public healthcare | PASS | got: public healthcare |
| 140 | Priority = Medium | PASS | got: Medium |
| 141 | Unit = Public Health Coordination | PASS | got: Public Health Coordination |
| 142 | Name = Dr. Amit Patel | PASS | got: Dr. Amit Patel |
| 143 | Has summary | PASS |  |
| 144 | Has triage_reason | PASS |  |
| 145 | Complaint #307 created | PASS |  |
| 146 | PATCH → 200 | PASS |  |
| 147 | Status = Under Review | PASS |  |
| 148 | Notes set | PASS |  |
| 149 | updated_at set | PASS |  |
| 150 | GET shows updated status | PASS |  |
| 151 | 'yes' → ready_to_file=True | PASS | got: True |
| 152 | 'yes file it' → ready_to_file=True | PASS | got: True |
| 153 | 'please file this' → ready_to_file=True | PASS | got: True |
| 154 | 'go ahead' → ready_to_file=True | PASS | got: True |
| 155 | 'honda' in complaint_text | PASS | text: My blue Honda City was stolen... |
| 156 | 'city' in complaint_text | PASS | text: My blue Honda City was stolen... |
| 157 | 'blue' in complaint_text | PASS | text: My blue Honda City was stolen... |
| 158 | 'ga01ab5678' in complaint_text | PASS | text: My blue Honda City was stolen. Registration GA01AB5678... |
| 159 | 'panaji' in complaint_text | PASS | text: My blue Honda City was stolen. Registration GA01AB5678. It was parked ... |
| 160 | 'goa' in complaint_text | PASS | text: My blue Honda City was stolen. Registration GA01AB5678. It was parked ... |
| 161 | Name extracted | PASS | got: Rohan Naik |
| 162 | Phone extracted | PASS | got: 9876543210 |
| 163 | 'raju' in complaint_text | PASS | text: My blue Honda City was stolen. Registration GA01AB5678. It was parked ... |
| 164 | 'cracked' in complaint_text | PASS | text: My blue Honda City was stolen. Registration GA01AB5678. It was parked ... |
| 165 | 'mirror' in complaint_text | PASS | text: My blue Honda City was stolen. Registration GA01AB5678. It was parked ... |
| 166 | 'icici' in complaint_text | PASS | text: My blue Honda City was stolen. Registration GA01AB5678. It was parked ... |
| 167 | 'lombard' in complaint_text | PASS | text: My blue Honda City was stolen. Registration GA01AB5678. It was parked ... |
| 168 | '12 lakh' in complaint_text | PASS | text: My blue Honda City was stolen. Registration GA01AB5678. It was parked ... |
| 169 | Filed has car make | PASS | text: my blue honda city was stolen. registration ga01ab5678. it was parked at p |
| 170 | Filed has registration | PASS | text: my blue honda city was stolen. registration ga01ab5678. it was parked at p |
| 171 | Filed has location | PASS | text: my blue honda city was stolen. registration ga01ab5678. it was parked at p |
| 172 | Filed has witness | PASS | text: my blue honda city was stolen. registration ga01ab5678. it was parked at p |
| 173 | Filed has insurance | PASS | text: my blue honda city was stolen. registration ga01ab5678. it was parked at p |
| 174 | Filed has value | PASS | text: my blue honda city was stolen. registration ga01ab5678. it was parked at p |
| 175 | Filed has damage | PASS | text: my blue honda city was stolen. registration ga01ab5678. it was parked at p |
| 176 | Filed: name = Rohan Naik | PASS |  |
| 177 | Filed: phone = 9876543210 | PASS |  |

## C2: Edge Cases & Security

**Result**: 16/16 passed

| # | Assertion | Result | Detail |
|---|-----------|--------|--------|
| 1 | HTTP 200 | PASS |  |
| 2 | Has category | PASS |  |
| 3 | Has priority | PASS |  |
| 4 | HTTP 200 | PASS |  |
| 5 | Summary present | PASS |  |
| 6 | HTTP 200 | PASS |  |
| 7 | HTTP 200 (not 500) | PASS |  |
| 8 | DB intact after injection | PASS |  |
| 9 | HTTP 200 | PASS |  |
| 10 | No raw script in summary | PASS |  |
| 11 | Missing text → 422 | PASS |  |
| 12 | Empty text → 422 | PASS |  |
| 13 | Non-existent ID → 404 | PASS |  |
| 14 | Missing session_id → 422 | PASS |  |
| 15 | Empty message → 422 | PASS |  |
| 16 | File nonexistent session → 404 | PASS |  |

## C3: Session Isolation

**Result**: 10/10 passed

| # | Assertion | Result | Detail |
|---|-----------|--------|--------|
| 1 | A has no B data (TestUserB) | PASS |  |
| 2 | B has no A data (TestUserA) | PASS |  |
| 3 | A has no Kolkata | PASS |  |
| 4 | B has no Delhi | PASS |  |
| 5 | A: category cyber crime | PASS |  |
| 6 | B: category fire accident | PASS |  |
| 7 | A: reporter = TestUserA | PASS |  |
| 8 | B: reporter = TestUserB | PASS |  |
| 9 | No old 'knifepoint' in text | PASS |  |
| 10 | New 'flood' in text | PASS |  |

## C4: Stress Test — Context Memory

**Result**: 24/24 passed

| # | Assertion | Result | Detail |
|---|-----------|--------|--------|
| 1 | Turn 8: name preserved | PASS | got: Vikram Joshi |
| 2 | Turn 9: name preserved | PASS | got: Vikram Joshi |
| 3 | Turn 9: phone preserved | PASS | got: 9876543210 |
| 4 | Turn 10: name preserved | PASS | got: Vikram Joshi |
| 5 | Turn 10: phone preserved | PASS | got: 9876543210 |
| 6 | Turn 11: name preserved | PASS | got: Vikram Joshi |
| 7 | Turn 11: phone preserved | PASS | got: 9876543210 |
| 8 | Turn 12: name preserved | PASS | got: Vikram Joshi |
| 9 | Turn 12: phone preserved | PASS | got: 9876543210 |
| 10 | Turn 13: name preserved | PASS | got: Vikram Joshi |
| 11 | Turn 13: phone preserved | PASS | got: 9876543210 |
| 12 | Turn 14: name preserved | PASS | got: Vikram Joshi |
| 13 | Turn 14: phone preserved | PASS | got: 9876543210 |
| 14 | Turn 15: name preserved | PASS | got: Vikram Joshi |
| 15 | Turn 15: phone preserved | PASS | got: 9876543210 |
| 16 | Filed has car make | PASS | text: i need help filing a complaint. my car was stolen last night. it's a red m |
| 17 | Filed has registration | PASS | text: i need help filing a complaint. my car was stolen last night. it's a red m |
| 18 | Filed has location part 1 | PASS | text: i need help filing a complaint. my car was stolen last night. it's a red m |
| 19 | Filed has location part 2 | PASS | text: i need help filing a complaint. my car was stolen last night. it's a red m |
| 20 | Filed has CCTV detail | PASS | text: i need help filing a complaint. my car was stolen last night. it's a red m |
| 21 | Filed has dent detail | PASS | text: i need help filing a complaint. my car was stolen last night. it's a red m |
| 22 | Filed has car seat detail | PASS | text: i need help filing a complaint. my car was stolen last night. it's a red m |
| 23 | Filed: name = Vikram Joshi | PASS |  |
| 24 | Filed: phone = 9876543210 | PASS |  |

## C5: Triage Accuracy — All Categories

**Result**: 36/36 passed

| # | Assertion | Result | Detail |
|---|-----------|--------|--------|
| 1 | HTTP 200 | PASS |  |
| 2 | Category = child safety | PASS | got: child safety |
| 3 | Unit = Child Protection Desk | PASS | got: Child Protection Desk |
| 4 | Priority = Emergency | PASS | got: Emergency |
| 5 | Has followup_questions | PASS |  |
| 6 | HTTP 200 | PASS |  |
| 7 | Category = cyber crime incident | PASS | got: cyber crime incident |
| 8 | Unit = Cyber Crime Cell | PASS | got: Cyber Crime Cell |
| 9 | Has followup_questions | PASS |  |
| 10 | HTTP 200 | PASS |  |
| 11 | Category = women help desk | PASS | got: women help desk |
| 12 | Unit = Women Help Desk | PASS | got: Women Help Desk |
| 13 | Has followup_questions | PASS |  |
| 14 | HTTP 200 | PASS |  |
| 15 | Category = public healthcare | PASS | got: public healthcare |
| 16 | Unit = Public Health Coordination | PASS | got: Public Health Coordination |
| 17 | Has followup_questions | PASS |  |
| 18 | HTTP 200 | PASS |  |
| 19 | Category = road accident | PASS | got: road accident |
| 20 | Unit = Traffic Police | PASS | got: Traffic Police |
| 21 | Priority = High | PASS | got: High |
| 22 | Has followup_questions | PASS |  |
| 23 | HTTP 200 | PASS |  |
| 24 | Category = murder / serious crime incident | PASS | got: murder / serious crime incident |
| 25 | Unit = Serious Crime Unit | PASS | got: Serious Crime Unit |
| 26 | Priority = Emergency | PASS | got: Emergency |
| 27 | Has followup_questions | PASS |  |
| 28 | HTTP 200 | PASS |  |
| 29 | Category = fire accident | PASS | got: fire accident |
| 30 | Unit = Fire and Emergency Coordination | PASS | got: Fire and Emergency Coordination |
| 31 | Has followup_questions | PASS |  |
| 32 | HTTP 200 | PASS |  |
| 33 | Category = general issue recorded | PASS | got: general issue recorded |
| 34 | Unit = General Desk | PASS | got: General Desk |
| 35 | Priority = Low | PASS | got: Low |
| 36 | Has followup_questions | PASS |  |

## C6: Triage Keyword Boundaries

**Result**: 19/19 passed

| # | Assertion | Result | Detail |
|---|-----------|--------|--------|
| 1 | 'My child was kidnapped at gunp' → Priority Emergency | PASS | got: Emergency |
| 2 | Flag: weapon_involved | PASS | got: ['weapon_involved', 'child_involved', 'person_missing'] |
| 3 | Flag: child_involved | PASS | got: ['weapon_involved', 'child_involved', 'person_missing'] |
| 4 | Flag: person_missing | PASS | got: ['weapon_involved', 'child_involved', 'person_missing'] |
| 5 | 'There's an active fire spreadi' → Priority Emergency | PASS | got: Emergency |
| 6 | Flag: fire_risk | PASS | got: ['fire_risk'] |
| 7 | 'Someone was shot and is bleedi' → Priority Emergency | PASS | got: Emergency |
| 8 | Flag: injury_reported | PASS | got: ['injury_reported', 'weapon_involved'] |
| 9 | Flag: weapon_involved | PASS | got: ['injury_reported', 'weapon_involved'] |
| 10 | 'I was robbed, they had a knife' → Priority High | PASS | got: High |
| 11 | Flag: weapon_involved | PASS | got: ['weapon_involved'] |
| 12 | 'I slipped and broke my arm, ne' → Priority High | PASS | got: High |
| 13 | Flag: injury_reported | PASS | got: ['injury_reported'] |
| 14 | 'My bicycle was stolen from the' → Priority Low | PASS | got: Low |
| 15 | Flag: none_identified | PASS | got: ['none_identified'] |
| 16 | 'Someone is stalking me online,' → Priority Medium | PASS | got: Medium |
| 17 | Flag: digital_fraud | PASS | got: ['digital_fraud'] |
| 18 | 'There's a gas leak in my neigh' → Priority Medium | PASS | got: Medium |
| 19 | Flag: fire_risk | PASS | got: ['fire_risk'] |

## C7: API Endpoint Coverage

**Result**: 7/7 passed

| # | Assertion | Result | Detail |
|---|-----------|--------|--------|
| 1 | GET /complaints → 200 | PASS |  |
| 2 | GET /complaints/{id} → 200 | PASS |  |
| 3 | PATCH /triage → 200 | PASS |  |
| 4 | PATCH /triage invalid → 422 | PASS |  |
| 5 | GET /chat/complaint → 200 | PASS |  |
| 6 | GET /chat nonexistent → 404 | PASS |  |
| 7 | DELETE /chat nonexistent → 404 | PASS |  |

## C8: Performance Benchmarks

**Result**: 4/4 passed

| # | Assertion | Result | Detail |
|---|-----------|--------|--------|
| 1 | Direct complaint avg < 10s (3749ms) | PASS |  |
| 2 | Chat turn avg < 6s (390ms) | PASS |  |
| 3 | All chat turns < 15s | PASS |  |
| 4 | Session listing < 2s (4ms) | PASS |  |

## C9: Concurrent Session Stress

**Result**: 10/10 passed

| # | Assertion | Result | Detail |
|---|-----------|--------|--------|
| 1 | Session 3: reporter = ConcurrentUser3 | PASS | got: ConcurrentUser3 |
| 2 | Session 3: has complaint_text | PASS | text: I need help with a cyber fraud complaint. My name ... |
| 3 | Session 1: reporter = ConcurrentUser1 | PASS | got: ConcurrentUser1 |
| 4 | Session 1: has complaint_text | PASS | text: I need help with a house fire complaint. My name i... |
| 5 | Session 0: reporter = ConcurrentUser0 | PASS | got: ConcurrentUser0 |
| 6 | Session 0: has complaint_text | PASS | text: I need help with a stolen car complaint. My name i... |
| 7 | Session 2: reporter = ConcurrentUser2 | PASS | got: ConcurrentUser2 |
| 8 | Session 2: has complaint_text | PASS | text: I need help with a missing child complaint. My nam... |
| 9 | Session 4: reporter = ConcurrentUser4 | PASS | got: ConcurrentUser4 |
| 10 | Session 4: has complaint_text | PASS | text: I need help with a road accident complaint. My nam... |

## C10: Officer Workflow

**Result**: 6/6 passed

| # | Assertion | Result | Detail |
|---|-----------|--------|--------|
| 1 | Complaint #307 created | PASS |  |
| 2 | PATCH → 200 | PASS |  |
| 3 | Status = Under Review | PASS |  |
| 4 | Notes set | PASS |  |
| 5 | updated_at set | PASS |  |
| 6 | GET shows updated status | PASS |  |

## C11: Filing Confirmation Variants

**Result**: 4/4 passed

| # | Assertion | Result | Detail |
|---|-----------|--------|--------|
| 1 | 'yes' → ready_to_file=True | PASS | got: True |
| 2 | 'yes file it' → ready_to_file=True | PASS | got: True |
| 3 | 'please file this' → ready_to_file=True | PASS | got: True |
| 4 | 'go ahead' → ready_to_file=True | PASS | got: True |

## C12: Data Integrity

**Result**: 23/23 passed

| # | Assertion | Result | Detail |
|---|-----------|--------|--------|
| 1 | 'honda' in complaint_text | PASS | text: My blue Honda City was stolen... |
| 2 | 'city' in complaint_text | PASS | text: My blue Honda City was stolen... |
| 3 | 'blue' in complaint_text | PASS | text: My blue Honda City was stolen... |
| 4 | 'ga01ab5678' in complaint_text | PASS | text: My blue Honda City was stolen. Registration GA01AB5678... |
| 5 | 'panaji' in complaint_text | PASS | text: My blue Honda City was stolen. Registration GA01AB5678. It was parked ... |
| 6 | 'goa' in complaint_text | PASS | text: My blue Honda City was stolen. Registration GA01AB5678. It was parked ... |
| 7 | Name extracted | PASS | got: Rohan Naik |
| 8 | Phone extracted | PASS | got: 9876543210 |
| 9 | 'raju' in complaint_text | PASS | text: My blue Honda City was stolen. Registration GA01AB5678. It was parked ... |
| 10 | 'cracked' in complaint_text | PASS | text: My blue Honda City was stolen. Registration GA01AB5678. It was parked ... |
| 11 | 'mirror' in complaint_text | PASS | text: My blue Honda City was stolen. Registration GA01AB5678. It was parked ... |
| 12 | 'icici' in complaint_text | PASS | text: My blue Honda City was stolen. Registration GA01AB5678. It was parked ... |
| 13 | 'lombard' in complaint_text | PASS | text: My blue Honda City was stolen. Registration GA01AB5678. It was parked ... |
| 14 | '12 lakh' in complaint_text | PASS | text: My blue Honda City was stolen. Registration GA01AB5678. It was parked ... |
| 15 | Filed has car make | PASS | text: my blue honda city was stolen. registration ga01ab5678. it was parked at p |
| 16 | Filed has registration | PASS | text: my blue honda city was stolen. registration ga01ab5678. it was parked at p |
| 17 | Filed has location | PASS | text: my blue honda city was stolen. registration ga01ab5678. it was parked at p |
| 18 | Filed has witness | PASS | text: my blue honda city was stolen. registration ga01ab5678. it was parked at p |
| 19 | Filed has insurance | PASS | text: my blue honda city was stolen. registration ga01ab5678. it was parked at p |
| 20 | Filed has value | PASS | text: my blue honda city was stolen. registration ga01ab5678. it was parked at p |
| 21 | Filed has damage | PASS | text: my blue honda city was stolen. registration ga01ab5678. it was parked at p |
| 22 | Filed: name = Rohan Naik | PASS |  |
| 23 | Filed: phone = 9876543210 | PASS |  |

## Verdict

**ALL TESTS PASSED.** System performing as expected under hardened testing.
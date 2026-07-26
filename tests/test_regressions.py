from importlib import metadata
import unittest

from packaging.requirements import Requirement

import backend.ai_service as ai_service
from backend.ai_service import (
    analyze_complaint,
    choose_final_complaint_text,
    extract_complaint_details_from_message,
    generate_chat_title,
    is_ready_to_file,
    merge_collected_fields,
)
import backend.database as database
from backend.triage import triage_complaint


class DependencyRegressionTests(unittest.TestCase):
    def test_langchain_provider_dependencies_are_compatible(self):
        for package in ("langchain-openai", "langchain-groq"):
            dist = metadata.distribution(package)
            for requirement_text in dist.requires or []:
                requirement = Requirement(requirement_text)
                if requirement.name not in {"langchain-core", "openai", "groq"}:
                    continue

                installed_version = metadata.version(requirement.name)
                self.assertIn(
                    installed_version,
                    requirement.specifier,
                    f"{package} {dist.version} requires {requirement}, "
                    f"but {requirement.name} {installed_version} is installed",
                )


class CollectorMergeRegressionTests(unittest.TestCase):
    def test_concurrent_style_name_and_phone_extract_without_llm(self):
        result = extract_complaint_details_from_message(
            "My name is ConcurrentUser2, phone 5555550002",
            {"complaint_text": "I need help with a missing child complaint."},
        )

        fields = result["extracted_fields"]
        self.assertEqual(fields["reporter_name"], "ConcurrentUser2")
        self.assertEqual(fields["reporter_phone"], "5555550002")

    def test_parent_reporter_name_and_phone_extract_without_llm(self):
        result = extract_complaint_details_from_message(
            "Please hurry, I'm his mother Anita Desai, my phone is 9988776655",
            {
                "complaint_text": "My 8-year-old son Arjun is missing from Shivaji Park.",
                "incident_location": "Shivaji Park",
                "incident_time": "4pm today",
            },
        )

        fields = result["extracted_fields"]
        self.assertEqual(fields["reporter_name"], "Anita Desai")
        self.assertEqual(fields["reporter_phone"], "9988776655")

    def test_doctor_title_name_extracts_full_name(self):
        result = extract_complaint_details_from_message(
            "My name is Dr. Amit Patel, it started 3 days ago on Monday",
            {"complaint_text": "Multiple people are getting sick from contaminated water."},
        )

        self.assertEqual(result["extracted_fields"]["reporter_name"], "Dr. Amit Patel")

    def test_child_missing_narrative_survives_short_llm_overwrite(self):
        prior = {
            "complaint_text": (
                "My 8-year-old son is missing. His name is Arjun, he was "
                "wearing a red t-shirt and blue shorts."
            ),
            "reporter_name": "Meena Sharma",
            "incident_location": "Shivaji Park",
        }
        incoming = {
            "complaint_text": "Please file this immediately, I'm terrified",
            "reporter_name": "",
            "incident_time": "around 4:30pm today",
        }

        merged = merge_collected_fields(prior, incoming, "Please file this immediately, I'm terrified")

        self.assertIn("Arjun", merged["complaint_text"])
        self.assertIn("missing", merged["complaint_text"].lower())
        self.assertIn("terrified", merged["complaint_text"])
        self.assertEqual(merged["reporter_name"], "Meena Sharma")
        self.assertEqual(merged["incident_location"], "Shivaji Park")
        self.assertEqual(merged["incident_time"], "around 4:30pm today")

    def test_confirmation_does_not_pollute_complaint_text(self):
        prior = {
            "complaint_text": "My child Arjun is missing from Shivaji Park.",
            "reporter_name": "Meena Sharma",
            "incident_location": "Shivaji Park",
            "incident_time": "around 4:30pm today",
        }
        incoming = {"complaint_text": "Yes, file it"}

        merged = merge_collected_fields(prior, incoming, "Yes, file it")

        self.assertEqual(merged["complaint_text"], prior["complaint_text"])
        self.assertTrue(is_ready_to_file(merged))

    def test_final_filing_falls_back_to_raw_conversation_when_collector_text_is_short(self):
        raw_text = (
            "My 8-year-old son is missing. His name is Arjun. "
            "This was at Shivaji Park around 4:30pm today."
        )

        final_text = choose_final_complaint_text("Please file this immediately", raw_text)

        self.assertEqual(final_text, raw_text)

    def test_new_incident_location_can_replace_stale_platform_location(self):
        prior = {
            "complaint_text": "I was scammed online through WhatsApp.",
            "incident_location": "WhatsApp (online)",
        }
        incoming = {
            "complaint_text": "It happened yesterday around 3pm in Andheri, Mumbai.",
            "incident_location": "Andheri, Mumbai",
            "incident_time": "yesterday around 3pm",
        }

        merged = merge_collected_fields(
            prior,
            incoming,
            "It happened yesterday around 3pm in Andheri, Mumbai",
        )

        self.assertEqual(merged["incident_location"], "Andheri, Mumbai")
        self.assertEqual(merged["incident_time"], "yesterday around 3pm")

    def test_location_fallback_extracts_we_live_in_phrase(self):
        merged = merge_collected_fields(
            {"complaint_text": "My husband has been hitting me."},
            {"reporter_name": "Priya Sharma"},
            "My name is Priya Sharma, we live in Bandra West, Mumbai",
        )

        self.assertEqual(merged["incident_location"], "Bandra West, Mumbai")

    def test_this_is_phrase_does_not_overwrite_reporter_name(self):
        merged = merge_collected_fields(
            {
                "complaint_text": "My car was stolen last night.",
                "reporter_name": "Vikram Joshi",
            },
            {},
            "This is the second time this year something was stolen from me",
        )

        self.assertEqual(merged["reporter_name"], "Vikram Joshi")

    def test_llm_echo_of_prior_text_still_appends_new_detail(self):
        merged = merge_collected_fields(
            {"complaint_text": "My blue Honda City was stolen."},
            {"complaint_text": "My blue Honda City was stolen."},
            "Registration GA01AB5678",
        )

        self.assertIn("GA01AB5678", merged["complaint_text"])

    def test_not_provided_values_are_replaced_by_real_details(self):
        merged = merge_collected_fields(
            {
                "complaint_text": "My 8-year-old son Arjun is missing.",
                "reporter_name": "Not provided",
                "reporter_phone": "Not provided",
            },
            {},
            "Please hurry, I'm his mother Anita Desai, my phone is 9988776655",
        )

        self.assertEqual(merged["reporter_name"], "Anita Desai")
        self.assertEqual(merged["reporter_phone"], "9988776655")


class TriageRegressionTests(unittest.TestCase):
    def setUp(self):
        self.old_openrouter = ai_service._OPENROUTER_CHAIN
        self.old_groq = ai_service._GROQ_CHAIN
        ai_service._OPENROUTER_CHAIN = None
        ai_service._GROQ_CHAIN = None
        ai_service._PROVIDER_DISABLED_UNTIL = {"openrouter": 0.0, "groq": 0.0}

    def tearDown(self):
        ai_service._OPENROUTER_CHAIN = self.old_openrouter
        ai_service._GROQ_CHAIN = self.old_groq
        ai_service._PROVIDER_DISABLED_UNTIL = {"openrouter": 0.0, "groq": 0.0}

    def test_public_healthcare_override_for_food_poisoning(self):
        result = analyze_complaint(
            "Multiple people are getting food poisoning from the restaurant on Main Street"
        )

        self.assertEqual(result["category"], "public healthcare")

    def test_child_safety_override_for_year_old_son_missing(self):
        result = analyze_complaint("My 8-year-old son Arjun is missing from Shivaji Park.")

        self.assertEqual(result["category"], "child safety")

    def test_body_with_signs_of_violence_is_serious_crime(self):
        result = analyze_complaint(
            "I found a body in the alley behind the market, there are signs of violence"
        )

        self.assertEqual(result["category"], "murder / serious crime incident")

    def test_smoke_women_help_desk_phrase_is_deterministic(self):
        result = analyze_complaint(
            "My neighbor is harassing me, following me and sending threatening messages"
        )

        self.assertEqual(result["category"], "women help desk")

    def test_smoke_road_accident_phrase_is_deterministic(self):
        result = analyze_complaint(
            "Two cars collided at the signal on Ring Road, both drivers are injured"
        )
        triage = triage_complaint(result["category"], result["summary"])

        self.assertEqual(result["category"], "road accident")
        self.assertEqual(triage["priority"], "High")

    def test_smoke_fire_phrase_is_deterministic(self):
        result = analyze_complaint("A gas cylinder exploded in the kitchen, the house is on fire")

        self.assertEqual(result["category"], "fire accident")

    def test_chat_fire_phrase_is_deterministic_without_llm(self):
        result = analyze_complaint(
            "There's a fire in my apartment building, smoke is everywhere. "
            "The fire started on the ground floor, it's spreading fast. "
            "There are elderly people trapped on the 3rd floor."
        )

        self.assertEqual(result["category"], "fire accident")

    def test_safe_default_summary_removes_script_tags(self):
        result = analyze_complaint('<script>alert("xss")</script>My house was robbed')

        self.assertNotIn("<script>", result["summary"].lower())
        self.assertIn("house was robbed", result["summary"])

    def test_provider_auth_and_rate_limit_errors_fall_back_to_rules(self):
        class FailingChain:
            def __init__(self, message):
                self.message = message

            def invoke(self, _args):
                raise RuntimeError(self.message)

        ai_service._OPENROUTER_CHAIN = FailingChain("401 Missing Authentication header")
        ai_service._GROQ_CHAIN = FailingChain("429 rate_limit_exceeded")

        result = analyze_complaint("A gas cylinder exploded in the kitchen, the house is on fire")

        self.assertEqual(result["category"], "fire accident")
        self.assertGreater(ai_service._PROVIDER_DISABLED_UNTIL["openrouter"], 0)
        self.assertGreater(ai_service._PROVIDER_DISABLED_UNTIL["groq"], 0)

    def test_women_help_desk_physical_abuse_is_high_with_injury_flag(self):
        result = triage_complaint(
            "women help desk",
            "I need help, my husband has been hitting me and abusing me for months.",
        )

        self.assertEqual(result["priority"], "High")
        self.assertEqual(result["assigned_unit"], "Women Help Desk")
        self.assertIn("injury_reported", result["risk_flags"])

    def test_women_help_desk_weapon_threat_remains_high(self):
        result = triage_complaint(
            "women help desk",
            "My husband threatened me with a knife when I tried to leave.",
        )

        self.assertEqual(result["priority"], "High")
        self.assertIn("weapon_involved", result["risk_flags"])

    def test_stabbing_sets_weapon_flag(self):
        result = triage_complaint(
            "murder / serious crime incident",
            "I found my neighbor dead with stab wounds. This looks like a murder.",
        )

        self.assertEqual(result["priority"], "Emergency")
        self.assertIn("weapon_involved", result["risk_flags"])
        self.assertIn("injury_reported", result["risk_flags"])

    def test_child_safety_category_sets_child_flag_without_literal_child_word(self):
        result = triage_complaint(
            "child safety",
            "My 8-year-old son Arjun is missing from Shivaji Park.",
        )

        self.assertEqual(result["priority"], "Emergency")
        self.assertIn("child_involved", result["risk_flags"])
        self.assertIn("person_missing", result["risk_flags"])

    def test_kidnapping_sets_emergency_and_missing_flag(self):
        result = triage_complaint(
            "child safety",
            "My child was kidnapped at gunpoint.",
        )

        self.assertEqual(result["priority"], "Emergency")
        self.assertIn("weapon_involved", result["risk_flags"])
        self.assertIn("child_involved", result["risk_flags"])
        self.assertIn("person_missing", result["risk_flags"])

    def test_shot_sets_emergency_and_weapon_flag(self):
        result = triage_complaint(
            "general issue recorded",
            "Someone was shot and is bleeding on the street.",
        )

        self.assertEqual(result["priority"], "Emergency")
        self.assertIn("injury_reported", result["risk_flags"])
        self.assertIn("weapon_involved", result["risk_flags"])

    def test_broke_my_arm_sets_high_and_injury_flag(self):
        result = triage_complaint(
            "general issue recorded",
            "I slipped and broke my arm, need medical help.",
        )

        self.assertEqual(result["priority"], "High")
        self.assertIn("injury_reported", result["risk_flags"])

    def test_gas_leak_sets_medium_and_fire_risk(self):
        result = triage_complaint(
            "general issue recorded",
            "There's a gas leak in my neighborhood, smell is strong.",
        )

        self.assertEqual(result["priority"], "Medium")
        self.assertIn("fire_risk", result["risk_flags"])

    def test_robbery_with_knife_sets_high_priority(self):
        result = triage_complaint(
            "general issue recorded",
            "I was robbed, they had a knife.",
        )

        self.assertEqual(result["priority"], "High")
        self.assertIn("weapon_involved", result["risk_flags"])

    def test_online_stalking_sets_medium_and_digital_flag(self):
        result = triage_complaint(
            "general issue recorded",
            "Someone is stalking me online, creating fake profiles.",
        )

        self.assertEqual(result["priority"], "Medium")
        self.assertIn("digital_fraud", result["risk_flags"])

    def test_road_accident_with_head_bleeding_is_emergency(self):
        result = triage_complaint(
            "road accident",
            "A car hit my motorcycle. I'm bleeding from my head and my left arm is broken.",
        )

        self.assertEqual(result["priority"], "Emergency")
        self.assertIn("injury_reported", result["risk_flags"])

    def test_low_risk_general_issue_stays_low(self):
        result = triage_complaint(
            "general issue recorded",
            "There is a broken water pipe on my street and nobody is responding.",
        )

        self.assertEqual(result["priority"], "Low")
        self.assertEqual(result["assigned_unit"], "General Desk")
        self.assertEqual(result["risk_flags"], ["none_identified"])


class ChatTitleRegressionTests(unittest.TestCase):
    def test_generate_chat_title_uses_deterministic_category(self):
        self.assertEqual(
            generate_chat_title("Two cars collided at the signal on Ring Road, both drivers are injured"),
            "Road Accident Complaint",
        )

    def test_chat_title_is_persisted_once(self):
        old_db_path = database.DB_PATH
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            database.DB_PATH = tmp.name
            try:
                database.create_table()
                database.create_chat_session("title-test")
                database.update_chat_session_title("title-test", "Road Accident Complaint")
                database.update_chat_session_title("title-test", "Changed Later")
                session = database.get_chat_session("title-test")
            finally:
                database.DB_PATH = old_db_path

        self.assertEqual(session["title"], "Road Accident Complaint")
        self.assertEqual(session["complaint_id"], None)


if __name__ == "__main__":
    unittest.main()

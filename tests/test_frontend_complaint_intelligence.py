import json
import subprocess
import textwrap
import unittest


NODE_SCRIPT = textwrap.dedent(
    """
    const fs = require("fs");
    const ts = require("./frontend/node_modules/typescript");
    const input = JSON.parse(process.argv[1]);
    const source = fs.readFileSync("frontend/src/lib/complaint-intelligence.ts", "utf8");
    const output = ts.transpileModule(source, {
      compilerOptions: {
        module: ts.ModuleKind.CommonJS,
        target: ts.ScriptTarget.ES2020,
      },
    }).outputText;

    const module = { exports: {} };
    const fn = new Function("module", "exports", output);
    fn(module, module.exports);

    const insight = module.exports.deriveComplaintInsight(input.fields, input.messages);
    console.log(JSON.stringify(insight));
    """
)


def derive_complaint_insight(fields, messages):
    result = subprocess.run(
        ["node", "-e", NODE_SCRIPT, json.dumps({"fields": fields, "messages": messages})],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


class FrontendComplaintIntelligenceTests(unittest.TestCase):
    def test_emergency_critical_fields_are_visible_missing_fields(self):
        insight = derive_complaint_insight(
            {"complaint_text": "My 8-year-old child is missing from the school gate."},
            [{"role": "user", "content": "My 8-year-old child is missing from the school gate."}],
        )

        missing_keys = {field["key"] for field in insight["missingDetails"]}
        critical_keys = {field["key"] for field in insight["criticalMissing"]}

        self.assertTrue(insight["emergencyMode"])
        self.assertIn("incident_location", missing_keys)
        self.assertIn("reporter_phone", missing_keys)
        self.assertIn("immediate_danger_status", missing_keys)
        self.assertLessEqual(critical_keys, missing_keys)

    def test_emergency_location_and_phone_are_not_missing_when_collected(self):
        insight = derive_complaint_insight(
            {
                "complaint_text": "My child is missing. I am safe now.",
                "incident_location": "Shivaji Park",
                "reporter_phone": "9988776655",
            },
            [{"role": "user", "content": "My child is missing. I am safe now."}],
        )

        critical_keys = {field["key"] for field in insight["criticalMissing"]}

        self.assertNotIn("incident_location", critical_keys)
        self.assertNotIn("reporter_phone", critical_keys)
        self.assertNotIn("immediate_danger_status", critical_keys)

    def test_stolen_car_does_not_estimate_road_accident_routing(self):
        insight = derive_complaint_insight(
            {"complaint_text": "My car was stolen from the apartment parking area last night."},
            [{"role": "user", "content": "My car was stolen from the apartment parking area last night."}],
        )

        self.assertEqual(insight["draftData"]["estimateConfidence"], "low")
        self.assertEqual(insight["draftData"]["estimatedCategory"], "general issue recorded")
        self.assertNotEqual(insight["draftData"]["estimatedAssignedUnit"], "Traffic Police")

    def test_road_collision_still_estimates_traffic_routing(self):
        insight = derive_complaint_insight(
            {"complaint_text": "Two cars collided at the traffic signal on Ring Road."},
            [{"role": "user", "content": "Two cars collided at the traffic signal on Ring Road."}],
        )

        self.assertEqual(insight["draftData"]["estimateConfidence"], "high")
        self.assertEqual(insight["draftData"]["estimatedCategory"], "road accident")
        self.assertEqual(insight["draftData"]["estimatedAssignedUnit"], "Traffic Police")


if __name__ == "__main__":
    unittest.main()

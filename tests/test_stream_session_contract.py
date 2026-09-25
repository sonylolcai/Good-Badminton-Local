import json
import re
import unittest
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ROOT = REPO_ROOT / "docs" / "contracts" / "stream_session_v1"
EXAMPLES_ROOT = CONTRACT_ROOT / "examples"
SCHEMA_PATH = CONTRACT_ROOT / "stream_session_contract_v1.schema.json"
MANIFEST_PATH = EXAMPLES_ROOT / "manifest.json"
CONTRACT_DOC_PATH = REPO_ROOT / "docs" / "STREAM_SESSION_CONTRACT_V1.md"


class ContractValidationError(AssertionError):
    pass


def _is_type(value, expected):
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    raise ContractValidationError(f"unsupported schema type: {expected}")


def _resolve_ref(root_schema, reference):
    if not reference.startswith("#/"):
        raise ContractValidationError(f"only local references are allowed: {reference}")
    node = root_schema
    for token in reference[2:].split("/"):
        node = node[token.replace("~1", "/").replace("~0", "~")]
    return node


def validate_instance(instance, schema, root_schema, path="$"):
    """Validate the JSON-Schema subset used by the frozen v1 contract.

    The project intentionally avoids adding a runtime dependency merely to check
    documentation. Unsupported keywords fail loudly so the validator cannot
    silently stop checking a future schema extension.
    """
    supported = {
        "$schema",
        "$id",
        "$defs",
        "$ref",
        "type",
        "const",
        "enum",
        "oneOf",
        "required",
        "properties",
        "additionalProperties",
        "dependentRequired",
        "items",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "pattern",
        "format",
        "title",
        "description",
    }
    unsupported = set(schema) - supported
    if unsupported:
        raise ContractValidationError(
            f"{path}: validator does not support schema keywords {sorted(unsupported)}"
        )

    if "$ref" in schema:
        validate_instance(instance, _resolve_ref(root_schema, schema["$ref"]), root_schema, path)
        return

    if "oneOf" in schema:
        matches = 0
        failures = []
        for candidate in schema["oneOf"]:
            try:
                validate_instance(instance, candidate, root_schema, path)
                matches += 1
            except ContractValidationError as exc:
                failures.append(str(exc))
        if matches != 1:
            raise ContractValidationError(
                f"{path}: expected exactly one oneOf match, got {matches}; failures={failures[:2]}"
            )
        return

    expected_types = schema.get("type")
    if expected_types is not None:
        if isinstance(expected_types, str):
            expected_types = [expected_types]
        if not any(_is_type(instance, item) for item in expected_types):
            raise ContractValidationError(
                f"{path}: expected type {expected_types}, got {type(instance).__name__}"
            )

    if "const" in schema and instance != schema["const"]:
        raise ContractValidationError(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and instance not in schema["enum"]:
        raise ContractValidationError(f"{path}: {instance!r} is not in {schema['enum']!r}")

    if isinstance(instance, dict):
        missing = set(schema.get("required", [])) - set(instance)
        if missing:
            raise ContractValidationError(f"{path}: missing required keys {sorted(missing)}")
        for key, dependencies in schema.get("dependentRequired", {}).items():
            if key in instance:
                missing_dependencies = set(dependencies) - set(instance)
                if missing_dependencies:
                    raise ContractValidationError(
                        f"{path}: {key} requires {sorted(missing_dependencies)}"
                    )
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extras = set(instance) - set(properties)
            if extras:
                raise ContractValidationError(f"{path}: unexpected keys {sorted(extras)}")
        for key, value in instance.items():
            if key in properties:
                validate_instance(value, properties[key], root_schema, f"{path}.{key}")

    if isinstance(instance, list) and "items" in schema:
        if len(instance) < schema.get("minItems", 0):
            raise ContractValidationError(f"{path}: array is shorter than minItems")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            raise ContractValidationError(f"{path}: array is longer than maxItems")
        for index, value in enumerate(instance):
            validate_instance(value, schema["items"], root_schema, f"{path}[{index}]")

    if isinstance(instance, str):
        if len(instance) < schema.get("minLength", 0):
            raise ContractValidationError(f"{path}: string is shorter than minLength")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            raise ContractValidationError(f"{path}: string is longer than maxLength")
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            raise ContractValidationError(f"{path}: string does not match {schema['pattern']}")
        if schema.get("format") == "date-time":
            try:
                datetime.fromisoformat(instance.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ContractValidationError(f"{path}: invalid date-time") from exc

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            raise ContractValidationError(f"{path}: value is below minimum")
        if "maximum" in schema and instance > schema["maximum"]:
            raise ContractValidationError(f"{path}: value is above maximum")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            raise ContractValidationError(f"{path}: value is not above exclusiveMinimum")


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class StreamSessionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = load_json(SCHEMA_PATH)
        cls.manifest = load_json(MANIFEST_PATH)
        cls.examples = {
            item["file"]: load_json(EXAMPLES_ROOT / item["file"])
            for item in cls.manifest["examples"]
        }

    def test_schema_and_manifest_are_versioned_and_complete(self):
        self.assertEqual(self.schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(self.schema["$defs"]["schemaVersion"]["const"], "stream-session.v1")
        expected_definitions = {
            "createSessionRequest",
            "createSessionResponse",
            "segmentMetadata",
            "segmentReceiptResponse",
            "sessionStatusResponse",
            "eventsResponse",
            "completeSessionRequest",
            "completeSessionResponse",
            "errorResponse",
        }
        manifest_definitions = {item["definition"] for item in self.manifest["examples"]}
        self.assertEqual(manifest_definitions, expected_definitions)
        self.assertTrue(CONTRACT_DOC_PATH.is_file())

    def test_every_example_matches_its_named_definition_and_root_schema(self):
        for item in self.manifest["examples"]:
            with self.subTest(example=item["file"]):
                instance = self.examples[item["file"]]
                definition = self.schema["$defs"][item["definition"]]
                validate_instance(instance, definition, self.schema)
                validate_instance(instance, self.schema, self.schema)

    def test_examples_do_not_leak_business_identity_or_scoring_fields(self):
        forbidden = {
            "check_id",
            "user_id",
            "user_name",
            "player_name",
            "team_id",
            "score",
            "winner",
            "serve_result",
        }

        def visit(value, path="$"):
            if isinstance(value, dict):
                leaked = forbidden.intersection(value)
                self.assertFalse(leaked, f"{path} leaked business fields: {sorted(leaked)}")
                for key, child in value.items():
                    visit(child, f"{path}.{key}")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, f"{path}[{index}]")

        for name, example in self.examples.items():
            with self.subTest(example=name):
                visit(example)

    def test_duplicate_receipt_reuses_the_original_durable_receipt(self):
        first = self.examples["segment_accepted_response.json"]["receipt"]
        duplicate = self.examples["segment_duplicate_response.json"]["receipt"]
        self.assertFalse(first["reused"])
        self.assertTrue(duplicate["reused"])
        self.assertEqual(duplicate["processing_disposition"], "already_accepted")
        for immutable_key in ("received_at", "sha256", "content_length_bytes"):
            self.assertEqual(first[immutable_key], duplicate[immutable_key])

    def test_out_of_order_segment_is_durable_but_waits_for_gap(self):
        payload = self.examples["segment_out_of_order_response.json"]
        self.assertTrue(payload["receipt"]["accepted"])
        self.assertEqual(payload["receipt"]["processing_disposition"], "waiting_for_predecessor")
        status = self.examples["session_status_response.json"]
        self.assertEqual(status["current_stage"], "waiting_for_predecessor")
        self.assertEqual(status["progress"]["missing_segment_indexes"], [1])

    def test_schema_rejects_extra_business_fields_and_invalid_values(self):
        request = dict(self.examples["create_session_request.json"])
        request["check_id"] = "business-owned-id"
        with self.assertRaises(ContractValidationError):
            validate_instance(request, self.schema["$defs"]["createSessionRequest"], self.schema)

        event_payload = json.loads(json.dumps(self.examples["events_response.json"]))
        event_payload["events"][0]["confidence"] = 1.1
        with self.assertRaises(ContractValidationError):
            validate_instance(event_payload, self.schema["$defs"]["eventsResponse"], self.schema)

        metadata = dict(self.examples["segment_metadata.json"])
        metadata["sha256"] = "not-a-sha256"
        with self.assertRaises(ContractValidationError):
            validate_instance(metadata, self.schema["$defs"]["segmentMetadata"], self.schema)

    def test_declared_source_frame_fields_are_paired(self):
        metadata = dict(self.examples["segment_metadata.json"])
        metadata["source_frame_start_index"] = 90
        with self.assertRaises(ContractValidationError):
            validate_instance(metadata, self.schema["$defs"]["segmentMetadata"], self.schema)
        metadata["source_frame_count"] = 30
        validate_instance(metadata, self.schema["$defs"]["segmentMetadata"], self.schema)


if __name__ == "__main__":
    unittest.main()

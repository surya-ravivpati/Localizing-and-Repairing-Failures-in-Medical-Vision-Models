"""Standardized model-output JSON schema + validation/repair (protocol §6)."""
from __future__ import annotations

import json
import re
from typing import Any

import jsonschema

RESPONSE_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "RadLLMResponse",
    "type": "object",
    "required": ["primary_diagnosis", "confidence", "explanation"],
    "properties": {
        "primary_diagnosis": {"type": "string"},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "explanation": {"type": "string"},
        "reasoning": {"type": "string"},
        "extracted_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["finding", "presence"],
                "properties": {
                    "finding": {"type": "string"},
                    "location": {"type": "string"},
                    "presence": {"enum": ["present", "absent", "uncertain"]},
                    "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                },
            },
        },
        "differential": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["diagnosis", "probability"],
                "properties": {
                    "diagnosis": {"type": "string"},
                    "probability": {"type": "integer", "minimum": 0, "maximum": 100},
                    "rationale": {"type": "string"},
                },
            },
        },
        "findings": {
            "type": "object",
            "description": "structured prompt: finding name -> presence/confidence",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "presence": {"enum": ["present", "absent", "uncertain"]},
                    "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                },
            },
        },
        "cited_findings": {"type": "array", "items": {"type": "string"}},
        "uncertainty_statement": {"type": "string"},
        "abstained": {"type": "boolean"},
    },
    "additionalProperties": True,
}

_validator = jsonschema.Draft7Validator(RESPONSE_SCHEMA)


def extract_json(raw: str) -> dict | None:
    """Best-effort parse: strip code fences, grab the outermost JSON object."""
    if raw is None:
        return None
    s = raw.strip()
    s = re.sub(r"^```(json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # Truncated JSON repair: keep the outer object, drop the incomplete tail,
    # and close open braces/brackets. Recovers primary_diagnosis/confidence even
    # when a long response was cut off mid-array.
    if s.startswith("{"):
        for cut in range(len(s), 0, -1):
            frag = s[:cut]
            # trim to last complete "key": value pair then balance brackets
            frag = re.sub(r",\s*\"[^\"]*\"\s*:\s*$", "", frag)
            frag = re.sub(r",\s*$", "", frag)
            opens = frag.count("{") - frag.count("}")
            openb = frag.count("[") - frag.count("]")
            candidate = frag + "]" * max(0, openb) + "}" * max(0, opens)
            try:
                return json.loads(candidate)
            except Exception:
                continue
    return None


def validate(obj: dict) -> list[str]:
    """Return a list of schema error messages (empty == valid)."""
    return [e.message for e in _validator.iter_errors(obj)]


_PRESENCE_SYNONYMS = {
    "possible": "uncertain", "probable": "uncertain", "equivocal": "uncertain",
    "suspected": "uncertain", "indeterminate": "uncertain",
    "yes": "present", "positive": "present", "seen": "present",
    "no": "absent", "negative": "absent", "not seen": "absent", "none": "absent",
}


def coerce(obj: dict) -> dict:
    """Light repair for common issues (confidence as float/str, out of range,
    presence values outside the strict present/absent/uncertain enum)."""
    if "confidence" in obj:
        try:
            c = int(round(float(obj["confidence"])))
            obj["confidence"] = max(0, min(100, c))
        except Exception:
            obj["confidence"] = 50
    for k in ("primary_diagnosis", "explanation"):
        if k in obj and not isinstance(obj[k], str):
            obj[k] = str(obj[k])
    findings = obj.get("findings")
    if isinstance(findings, dict):
        for v in findings.values():
            if isinstance(v, dict):
                for key in ("presence", "status"):
                    if key in v and isinstance(v[key], str):
                        low = v[key].strip().lower()
                        v[key] = _PRESENCE_SYNONYMS.get(low, low)
    return obj


if __name__ == "__main__":
    good = {"primary_diagnosis": "Cardiomegaly", "confidence": 80, "explanation": "x"}
    bad = {"primary_diagnosis": "x", "confidence": 130, "explanation": "y"}
    print("good errors:", validate(good))
    print("bad  errors:", validate(coerce(bad)))

"""Fail-closed structural/output checks and independently executable constraints.

Semantic correctness of arbitrary prose requires separate verification; passing
these checks is never represented as a universal correctness certificate.
"""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass, field
import io
import json
import math
import operator
import re
import xml.etree.ElementTree as ET


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    semantic_review_required: bool = True

    @property
    def accepted(self):
        return not self.errors

    def require(self):
        if self.errors:
            raise ValueError("Training example rejected: " + "; ".join(self.errors))
        return self


def strict_json(text: str):
    def reject(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    return json.loads(text, parse_constant=reject)


def calculate(expression: str) -> float:
    """Bounded arithmetic only; never eval generated code."""
    if len(expression) > 120:
        raise ValueError("Expression too long.")
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = float(node.value)
        elif isinstance(node, ast.BinOp) and type(node.op) in ops:
            value = ops[type(node.op)](visit(node.left), visit(node.right))
        elif isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            value = ops[type(node.op)](visit(node.operand))
        else:
            raise ValueError("Unsupported arithmetic constraint.")
        if not math.isfinite(value) or abs(value) > 1e12:
            raise ValueError("Invalid arithmetic magnitude.")
        return value

    return visit(ast.parse(expression, mode="eval").body)


def clock_minutes(hour, minute, meridiem):
    hour, minute = int(hour), int(minute)
    if not 1 <= hour <= 12 or not 0 <= minute < 60:
        raise ValueError("Invalid clock time.")
    return (hour % 12 + (12 if meridiem.lower().startswith("p") else 0)) * 60 + minute


TIME = r"(\d{1,2}):(\d{2})\s*([ap])\.?m\.?"


def validate_sample(sample, *, output_spec: dict | None = None) -> ValidationResult:
    result = ValidationResult()
    instruction, context, answer = (
        sample.instruct.strip(),
        sample.sample.strip(),
        (sample.gold_summary or "").strip(),
    )
    if not instruction or not answer or not sample.tag.strip():
        result.errors.append("missing_instruction_answer_or_tag")
        return result
    result.checks.append("required_fields")
    spec = output_spec or getattr(sample, "output_spec", {}) or {}
    wants_json = (
        bool(
            re.search(
                r"\b(?:as|in)\s+(?:a\s+)?JSON\b|\b(?:return|provide|output|give|produce|generate)\s+(?:(?:the|a|an|your|only|valid|best|recommendation|as)\s+){0,5}JSON\b",
                instruction,
                re.I,
            )
        )
        or spec.get("format") == "json"
        or "json_schema" in spec
    )
    if wants_json:
        raw = answer.strip()
        try:
            json_only = spec.get("json_only", False) or bool(
                re.search(r"\bonly\s+JSON\b|\bJSON\s+only\b", instruction, re.I)
            )
            if raw.startswith("```"):
                if json_only:
                    raise ValueError("JSON-only output cannot have a Markdown fence.")
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
            if json_only or not re.search(
                r"\b(?:then|followed by)\b", instruction, re.I
            ):
                obj = strict_json(raw)
            else:
                start = raw.find("{")
                if start < 0:
                    raise ValueError("Missing JSON object.")
                obj = strict_json(raw[start:])
            required = spec.get("required_keys")
            if required is None:
                match = re.search(r"\bkeys?\s*:\s*(.*)", instruction, re.I)
                required = (
                    re.findall(r'["\']([^"\']+)["\']', match.group(1)) if match else []
                )
            if required and (
                not isinstance(obj, dict) or any(k not in obj for k in required)
            ):
                raise ValueError("Missing required JSON keys.")
            if "json_schema" in spec:
                from jsonschema import Draft202012Validator
                from jsonschema.exceptions import SchemaError

                try:
                    Draft202012Validator.check_schema(spec["json_schema"])
                except SchemaError as exc:
                    raise ValueError("Invalid output schema.") from exc
                if list(Draft202012Validator(spec["json_schema"]).iter_errors(obj)):
                    raise ValueError("Answer violates its JSON schema.")
            result.checks.append("required_json_format")
        except (ValueError, TypeError):
            result.errors.append("invalid_required_json")
    if re.search(r"\bbullet list\b", instruction, re.I) and not re.search(
        r"(?m)^\s*[-*•]\s+", answer
    ):
        result.errors.append("missing_bullet_list")
    if spec.get("allowed_answers") and answer not in spec["allowed_answers"]:
        result.errors.append("answer_outside_allowed_set")
    output_format = spec.get("format")
    if output_format not in {
        None,
        "plain",
        "json",
        "bullets",
        "csv",
        "xml",
        "markdown_table",
        "number",
    }:
        result.errors.append("unsupported_output_format")
    try:
        if output_format == "bullets" and not all(
            re.match(r"\s*[-*•]\s+", line)
            for line in answer.splitlines()
            if line.strip()
        ):
            raise ValueError("Every line must be a bullet.")
        if output_format == "csv":
            rows = list(
                csv.reader(
                    io.StringIO(answer),
                    delimiter=spec.get("delimiter", ","),
                    strict=True,
                )
            )
            if not rows or any(len(row) != len(rows[0]) for row in rows):
                raise ValueError("Invalid CSV dimensions.")
            if spec.get("columns") and rows[0] != spec["columns"]:
                raise ValueError("Incorrect CSV headers.")
        if output_format == "xml":
            element = ET.fromstring(answer)
            if spec.get("root") and element.tag != spec["root"]:
                raise ValueError("Incorrect XML root.")
        if output_format == "markdown_table":
            lines = [line.strip() for line in answer.splitlines() if line.strip()]
            if len(lines) < 2 or not re.fullmatch(
                r"\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)+\|?", lines[1]
            ):
                raise ValueError("Missing Markdown table separator.")
            cells = [
                [cell.strip() for cell in line.strip("|").split("|")] for line in lines
            ]
            if any(len(row) != len(cells[0]) for row in cells):
                raise ValueError("Inconsistent table columns.")
            if spec.get("columns") and cells[0] != spec["columns"]:
                raise ValueError("Incorrect table headers.")
        if output_format == "number":
            if not re.fullmatch(
                r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", answer
            ) or not math.isfinite(float(answer)):
                raise ValueError("Expected one finite number.")
        if "line_count" in spec and len(answer.splitlines()) != spec["line_count"]:
            raise ValueError("Incorrect line count.")
        if (
            "word_count" in spec
            and len(re.findall(r"\b\w+\b", answer)) != spec["word_count"]
        ):
            raise ValueError("Incorrect word count.")
        if "max_characters" in spec and len(answer) > spec["max_characters"]:
            raise ValueError("Answer exceeds character limit.")
    except (ValueError, TypeError, csv.Error, ET.ParseError):
        result.errors.append("required_output_spec_failed")
    for constraint in spec.get("constraints", []):
        try:
            if constraint["kind"] != "arithmetic":
                raise ValueError("Unsupported constraint kind.")
            expected = calculate(constraint["expression"])
            actual = float(constraint["answer_value"])
            # Require the checked value to occur in the actual gold, not just metadata.
            numbers = [
                float(n) for n in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", answer)
            ]
            if not math.isclose(
                expected, actual, rel_tol=1e-9, abs_tol=1e-9
            ) or not any(math.isclose(actual, n) for n in numbers):
                result.errors.append("arithmetic_constraint_failed")
            else:
                result.checks.append("arithmetic_constraint")
        except (KeyError, ValueError, ZeroDivisionError, TypeError, SyntaxError):
            result.errors.append("invalid_arithmetic_constraint")
    # Independently check explicit claims of the duration between two clock times.
    for match in re.finditer(
        r"(\d+)\s+minutes?\s+(?:before|between).*?" + TIME + r".*?" + TIME, answer, re.I
    ):
        duration, h1, m1, a1, h2, m2, a2 = match.groups()
        if (clock_minutes(h2, m2, a2) - clock_minutes(h1, m1, a1)) % 1440 != int(
            duration
        ):
            result.errors.append("incorrect_time_interval")
    # The shorter 'leaves N minutes before ... begins at TIME' form has its start in the preceding sentence.
    times = list(re.finditer(TIME, answer, re.I))
    duration = re.search(r"leaves\s+(\d+)\s+minutes?\s+before", answer, re.I)
    if duration and len(times) == 2:
        left, right = (clock_minutes(*m.groups()) for m in times)
        if (right - left) % 1440 != int(duration.group(1)):
            result.errors.append("incorrect_time_interval")
    workshops = re.findall(
        r"Workshop\s+([A-Z])\s+lasts\s+(?:(\d+)\s+hours?\s*)?(?:(\d+)\s+minutes?)?\s+and\s+starts\s+at\s+"
        + TIME,
        context,
        re.I,
    )
    if len(workshops) >= 2:
        ends = {
            name.upper(): clock_minutes(h, m, a)
            + 60 * int(hours or 0)
            + int(minutes or 0)
            for name, hours, minutes, h, m, a in workshops
        }
        claim = re.search(r"Workshop\s+([A-Z])\s+ends\s+later", answer, re.I)
        if claim:
            winner = claim.group(1).upper()
            if (
                winner not in ends
                or sum(v == max(ends.values()) for v in ends.values()) != 1
                or ends[winner] != max(ends.values())
            ):
                result.errors.append("incorrect_workshop_end_comparison")
    result.semantic_review_required = True
    return result

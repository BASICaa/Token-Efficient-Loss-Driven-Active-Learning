"""Small independently specified verifier regression audit, excluding benchmark data.

Passing these deliberately clear cases is necessary, not sufficient evidence
of semantic verification quality on arbitrary generated instruction data.
"""

from __future__ import annotations

from .dataschema import Training_Sample
from .helper import write_json_file

CASES = [
    (
        "brainstorming",
        "List three practical ideas for reusing a glass jar.",
        "",
        "- Store pencils.\n- Hold flowers.\n- Organize buttons.",
        "Buy a new car.",
    ),
    (
        "classification",
        "Classify each item as animal or plant.",
        "dog, oak",
        "dog: animal; oak: plant.",
        "dog: plant; oak: animal.",
    ),
    (
        "closed_qa",
        "How many apples remain?",
        "Mira has seven apples and gives away two.",
        "Five apples remain.",
        "Six apples remain.",
    ),
    (
        "creative_writing",
        "Write a two-line poem mentioning the moon. Use exactly four words per line.",
        "",
        "Silver moon lights water.\nQuiet waves cradle dreams.",
        "Sunlight fills the land.",
    ),
    (
        "general_qa",
        "Why should people wash their hands before eating?",
        "",
        "Washing removes dirt and germs, reducing the risk of transferring them to food.",
        "It adds harmful germs to the food.",
    ),
    (
        "information_extraction",
        'Return JSON with two keys: "id" and "date".',
        "Shipment AB12 arrives June 3.",
        '{"id":"AB12","date":"June 3"}',
        '{"id":"AB21","date":"July 3"}',
    ),
    (
        "open_qa",
        "What shape has exactly three straight sides? Answer with one word.",
        "",
        "Triangle",
        "Square",
    ),
    (
        "summarization",
        "Summarize the opening schedule in one sentence.",
        "The cafe opens at 9 a.m. on weekdays and is closed on weekends.",
        "The cafe opens at 9 a.m. on weekdays and closes for weekends.",
        "The cafe opens every day at noon.",
    ),
]


def audit_verifier(teacher, output):
    results = []
    for category, instruction, context, valid, invalid in CASES:
        for expected, answer in [(True, valid), (False, invalid)]:
            sample = Training_Sample(
                sample=context,
                instruct=instruction,
                gold_summary=answer,
                source="generated",
                tag=category,
            )
            try:
                teacher.verify(sample)
                accepted = True
            except ValueError:
                accepted = False
            except Exception as exc:
                report = {
                    "status": "incomplete",
                    "error_type": type(exc).__name__,
                    "results": results,
                    "model": teacher.model,
                    "teacher_identity": teacher.identity,
                }
                write_json_file(output, report)
                return report
            results.append(
                {
                    "category": category,
                    "expected_accept": expected,
                    "accepted": accepted,
                }
            )
            write_json_file(
                output,
                {"status": "running", "results": results, "model": teacher.model},
            )
    false_accepts = sum(not x["expected_accept"] and x["accepted"] for x in results)
    false_rejects = sum(x["expected_accept"] and not x["accepted"] for x in results)
    report = {
        "status": "passed" if false_accepts == 0 and false_rejects <= 1 else "failed",
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "valid_cases": 8,
        "invalid_cases": 8,
        "results": results,
        "model": teacher.model,
        "teacher_identity": teacher.identity,
        "scope": "Clear regression fixtures; not a universal correctness guarantee.",
    }
    write_json_file(output, report)
    return report

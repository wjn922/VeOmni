#!/usr/bin/env python3
"""
Convert LLaVA-style JSON records from a single "image" string field to an "images" list field.

Typical input is a JSON array of objects. This script:
- Converts "image": "<path>" to "images": ["<path>"] and removes the original "image" field

Usage:
python convert_llava_json_to_veomni_json.py --input /path/to/input.json --output /path/to/output.json
python convert_llava_json_to_veomni_json.py --input /path/to/input.jsonl --output /path/to/output.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    # tqdm is optional; the script should still work without it.
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description='Convert "image" (str) to "images" (list[str]) for JSON/JSONL records.'
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input file path. Supported formats: .json (array of objects), .jsonl (one object per line).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output file path. Supported formats: .json (array), .jsonl (one object per line).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=4,
        help="Indent for output JSON formatting (only used when output is .json).",
    )
    return parser.parse_args()


def load_records(input_path: str) -> List[Dict[str, Any]]:
    """
    Load records from a .json (array) or .jsonl (one object per line) file.

    Args:
        input_path: Input file path ending with .json or .jsonl.

    Returns:
        List[Dict[str, Any]]: Loaded records.
    """

    path = Path(input_path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError('Input .json must be a list (JSON array) of objects.')
        return data

    if suffix == ".jsonl":
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError("Each line in .jsonl must be a JSON object.")
                records.append(obj)
        return records

    raise ValueError(f"Unsupported input format: {suffix}. Expected .json or .jsonl.")


def save_records(output_path: str, records: List[Dict[str, Any]], indent: int = 4) -> None:
    """
    Save records to a .json (array) or .jsonl (one object per line) file.

    Args:
        output_path: Output file path ending with .json or .jsonl.
        records: Records to write.
        indent: Indent for .json output formatting.

    Returns:
        None
    """

    path = Path(output_path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=indent)
        return

    if suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for item in records:
                f.write(json.dumps(item, ensure_ascii=False))
                f.write("\n")
        return

    raise ValueError(f"Unsupported output format: {suffix}. Expected .json or .jsonl.")


def _normalize_images_field(value: Any) -> List[str]:
    """
    Normalize various image field representations into a list[str].

    Args:
        value: The value of "image" or "images" field. Supported:
            - None / empty: returns []
            - str: returns [str] if non-empty, else []
            - list/tuple: returns [str(x) for x in value] (keeps order)

    Returns:
        List[str]: Normalized images list.
    """

    if value is None:
        return []
    if isinstance(value, str):
        v = value.strip()
        return [v] if v else []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for x in value:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    # Fallback: keep a single non-empty string representation.
    s = str(value).strip()
    return [s] if s else []


def _messages_to_conversations(messages: Any) -> List[Dict[str, str]]:
    """
    Convert OpenAI-style messages (role/content) to ShareGPT-style conversations (from/value).

    Args:
        messages: A list of dicts with keys like {"role": "...", "content": "..."}.

    Returns:
        List[Dict[str, str]]: A list of dicts with keys {"from": "...", "value": "..."}.
    """

    if not isinstance(messages, list):
        return []

    role_map = {
        "user": "human",
        "assistant": "gpt",
        "system": "human",
    }

    conversations: List[Dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        content = msg.get("content", "")
        value = content if isinstance(content, str) else str(content)

        # Best-effort role mapping to avoid downstream KeyErrors.
        mapped_role = role_map.get(role)
        if mapped_role is None:
            mapped_role = "gpt" if "assistant" in role else "human"

        conversations.append({"from": mapped_role, "value": value})

    return conversations


def convert_record(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single record to ensure it contains:
    - "images": list[str]
    - "conversations": ShareGPT-style list[{"from": str, "value": str}]

    Rules:
    - If "images" exists, normalize it to list[str].
    - Else if "image" exists, convert it to "images" and remove "image".
    - Else, add "images": [].
    - If both "images" and "image" exist, keep normalized "images" and drop "image".
    - If "conversations" exists, keep it and drop "messages" if present.
    - Else if "messages" exists, convert it to "conversations" and drop "messages".
    - Else, add "conversations": [].

    Args:
        item: Input record.

    Returns:
        Dict[str, Any]: Converted record.
    """

    new_item = dict(item)
    if "images" in new_item:
        new_item["images"] = _normalize_images_field(new_item.get("images"))
        # Unify schema by removing legacy field if present.
        new_item.pop("image", None)
    elif "image" in new_item:
        image_value = new_item.pop("image", None)
        new_item["images"] = _normalize_images_field(image_value)
    else:
        new_item["images"] = []

    if "conversations" in new_item and new_item.get("conversations"):
        # Prefer ShareGPT-style conversations for VeOmni preprocessors.
        new_item.pop("messages", None)
        return new_item

    if "messages" in new_item and new_item.get("messages"):
        new_item["conversations"] = _messages_to_conversations(new_item.get("messages"))
        new_item.pop("messages", None)
    else:
        new_item["conversations"] = []

    return new_item


def convert_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert records to ensure "images" exists and is list[str] for every item.

    Args:
        records: A list of JSON objects (dict) to convert.

    Returns:
        A new list of converted records.
    """

    converted: List[Dict[str, Any]] = []
    iterator = records
    if tqdm is not None:
        iterator = tqdm(records, total=len(records), desc="Converting records")

    for item in iterator:
        converted.append(convert_record(item))
    return converted


def main() -> None:
    """
    CLI entrypoint.

    Returns:
        None
    """

    args = parse_args()
    print(f"Loading records from {args.input}...")
    data = load_records(args.input)
    converted_data = convert_records(data)
    save_records(args.output, converted_data, indent=args.indent)
    print(f"Saved records to {args.output}.")
    


if __name__ == "__main__":
    main()

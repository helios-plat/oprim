# 🚫 Tide-only — 中文 NLP 专属，永久 Tide 专属
"""中文文本数字 + 单位提取 (中文金融 NLP 基础)."""

from __future__ import annotations

import re
from typing import Any, Literal

_DEFAULT_SCALE_MAPPING: dict[str, float] = {
    "亿": 1e8,
    "百亿": 1e10,
    "十亿": 1e9,
    "百万": 1e6,
    "万": 1e4,
    "千": 1e3,
    "百": 1e2,
}

# Traditional Chinese numeral mapping
_CN_DIGIT: dict[str, int] = {
    "零": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "壹": 1,
    "贰": 2,
    "叁": 3,
    "肆": 4,
    "伍": 5,
    "陆": 6,
    "柒": 7,
    "捌": 8,
    "玖": 9,
    "○": 0,
}
_CN_UNIT: dict[str, int] = {
    "十": 10,
    "拾": 10,
    "百": 100,
    "佰": 100,
    "千": 1000,
    "仟": 1000,
    "万": 10000,
    "亿": 100000000,
}


def numerical_extraction(
    text: str,
    *,
    units: list[Literal["currency", "percentage", "scale", "ratio", "date"]] | None = None,
    currency: str = "CNY",
    scale_mapping: dict[str, float] | None = None,
    return_spans: bool = False,
) -> dict[str, Any]:
    """中文文本数字 + 单位提取 (中文金融 NLP 基础).

    Args:
        text:          输入中文文本
        units:         要提取的类型 (默认 ["currency", "percentage", "scale"])
        currency:      默认货币
        scale_mapping: 量级映射 (None = 默认中文)
        return_spans:  是否返回 span

    Returns:
        dict with extractions (list of dicts) and summary.

    Raises:
        ValueError: text 为空.
    """
    if not text:
        raise ValueError("text must be non-empty")

    if units is None:
        units = ["currency", "percentage", "scale"]

    scale_map = {**_DEFAULT_SCALE_MAPPING, **(scale_mapping or {})}

    extractions: list[dict[str, Any]] = []

    # Number pattern: supports 1,234.56 / 1234.56 / 1234
    num_re = r"[\d,]+(?:\.\d+)?"

    # --- Currency patterns ---
    if "currency" in units:
        # "1.2亿元" / "12,000万" / "1,200,000元" / "120万元"
        pat_currency = re.compile(
            rf"({num_re})\s*(亿元|百亿元|十亿元|百万元|万元|千元|元|亿|百万|万|千)",
            re.UNICODE,
        )
        for m in pat_currency.finditer(text):
            num_str = m.group(1).replace(",", "")
            unit_str = m.group(2)
            try:
                raw = float(num_str)
            except ValueError:
                continue
            # Map unit to scale
            scale = 1.0
            for scale_key, scale_val in scale_map.items():
                if unit_str.startswith(scale_key):
                    scale = scale_val
                    break
            value = round(raw * scale, 6)
            entry: dict[str, Any] = {
                "value": value,
                "unit_type": "currency",
                "original_text": m.group(0),
                "confidence": 0.95,
            }
            if return_spans:
                entry["span"] = (m.start(), m.end())
            extractions.append(entry)

    # --- Percentage patterns ---
    if "percentage" in units:
        pat_pct = re.compile(
            rf"({num_re})\s*%|百分之\s*({num_re})|增长\s*({num_re})\s*个百分点",
            re.UNICODE,
        )
        for m in pat_pct.finditer(text):
            original = m.group(0)
            # "N个百分点" is NOT a percentage rate — skip
            if "个百分点" in original:
                continue
            # Determine which group matched
            if m.group(1) is not None:
                raw = float(m.group(1).replace(",", ""))
            elif m.group(2) is not None:
                raw = float(m.group(2).replace(",", ""))
            else:
                continue
            value = round(raw / 100.0, 6)
            entry = {
                "value": value,
                "unit_type": "percentage",
                "original_text": original,
                "confidence": 0.90,
            }
            if return_spans:
                entry["span"] = (m.start(), m.end())
            extractions.append(entry)

    # --- Scale (pure scale number) ---
    if "scale" in units:
        pat_scale = re.compile(
            rf"({num_re})\s*(亿|百万|万|千)(?![元%])",
            re.UNICODE,
        )
        for m in pat_scale.finditer(text):
            # Avoid double-counting if already captured as currency
            already = any(e["original_text"] == m.group(0) for e in extractions)
            if already:
                continue
            num_str = m.group(1).replace(",", "")
            unit_str = m.group(2)
            try:
                raw = float(num_str)
            except ValueError:
                continue
            scale = scale_map.get(unit_str, 1.0)
            value = round(raw * scale, 6)
            entry = {
                "value": value,
                "unit_type": "scale",
                "original_text": m.group(0),
                "confidence": 0.80,
            }
            if return_spans:
                entry["span"] = (m.start(), m.end())
            extractions.append(entry)

    # --- Ratio / 倍数 ---
    if "ratio" in units:
        pat_ratio = re.compile(rf"({num_re})\s*倍", re.UNICODE)
        for m in pat_ratio.finditer(text):
            raw = float(m.group(1).replace(",", ""))
            entry = {
                "value": raw,
                "unit_type": "ratio",
                "original_text": m.group(0),
                "confidence": 0.88,
            }
            if return_spans:
                entry["span"] = (m.start(), m.end())
            extractions.append(entry)

    # --- Date ---
    if "date" in units:
        pat_date = re.compile(
            r"(\d{4})\s*年(?:\s*(\d{1,2})\s*月(?:\s*(\d{1,2})\s*日)?)?|"
            r"Q([1-4])\s*(?:季度)?|"
            r"(上半年|下半年|全年|去年同期|上年同期)",
            re.UNICODE,
        )
        for m in pat_date.finditer(text):
            entry = {
                "value": float("nan"),
                "unit_type": "date",
                "original_text": m.group(0),
                "confidence": 0.92,
            }
            if return_spans:
                entry["span"] = (m.start(), m.end())
            extractions.append(entry)

    # Handle traditional Chinese numerals (e.g. 壹仟万)
    extractions.extend(_extract_cn_numerals(text, return_spans=return_spans))

    # Deduplicate by original_text (keep highest confidence)
    seen: dict[str, dict[str, Any]] = {}
    for e in extractions:
        key = e["original_text"]
        if key not in seen or e["confidence"] > seen[key]["confidence"]:
            seen[key] = e
    extractions = list(seen.values())

    # Summary
    summary: dict[str, int] = {}
    for e in extractions:
        ut = e["unit_type"]
        summary[ut] = summary.get(ut, 0) + 1

    return {"extractions": extractions, "summary": summary}


def _extract_cn_numerals(text: str, *, return_spans: bool) -> list[dict[str, Any]]:
    """Extract traditional Chinese numerals like 壹仟万 → 10,000,000."""
    results: list[dict[str, Any]] = []
    # Pattern: sequence of CN digit/unit chars followed by scale
    pat = re.compile(r"([零一二三四五六七八九壹贰叁肆伍陆柒捌玖十拾百佰千仟万亿○]+)", re.UNICODE)
    for m in pat.finditer(text):
        cn_str = m.group(1)
        if not any(c in _CN_DIGIT for c in cn_str):
            continue
        val = _parse_cn_numeral(cn_str)
        if val is None:
            continue
        entry: dict[str, Any] = {
            "value": float(val),
            "unit_type": "scale",
            "original_text": cn_str,
            "confidence": 0.75,
        }
        if return_spans:
            entry["span"] = (m.start(), m.end())
        results.append(entry)
    return results


def _parse_cn_numeral(s: str) -> int | None:
    """Parse traditional Chinese numeral string to int."""
    if not s:
        return None
    result = 0
    current = 0
    yi_part = 0
    for ch in s:
        if ch in _CN_DIGIT:
            current = _CN_DIGIT[ch]
        elif ch in _CN_UNIT:
            unit = _CN_UNIT[ch]
            if unit == 100000000:  # 亿
                yi_part = (result + current) * unit
                result = 0
                current = 0
            elif unit == 10000:  # 万
                result = (result + current) * unit
                current = 0
            else:
                if current == 0:
                    current = 1
                result += current * unit
                current = 0
    result += current
    total = yi_part + result
    return total if total > 0 else None

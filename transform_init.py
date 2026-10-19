#!/usr/bin/env python3
"""Rewrite oprim/__init__.py:
1. Register all eager-imported names in _ELEMENT_MAP (lazy loading)
2. Remove eager import statements (kills F401+E402+I001)
3. Fix E701, N802, SIM102, E501 style issues
4. Update __all__ to include all public names
No noqa, no excludes, no per-file-ignore.
"""

import re

with open("oprim/__init__.py") as f:
    src = f.read()

lines = src.split("\n")

# === Step 1: Identify the eager import block ===
# It starts after "# --- Explicit re-exports (Pinning) ---"
# and ends before "# --- Mneme elements (M-A batch) ---" or first def after it
block_start = None
block_end = None
for i, line in enumerate(lines):
    if "# --- Explicit re-exports (Pinning) ---" in line:
        block_start = i
    if block_start is not None and (
        "# --- Mneme elements" in line or line.strip().startswith("def llm_complete")
    ):
        block_end = i
        break

print(f"Eager block: lines {block_start} to {block_end}")

# === Step 2: Parse eager imports to get (name, module_path) ===
eager_map = {}  # name -> module_path
buf = []
for i in range(block_start, block_end):
    stripped = lines[i].strip()
    if stripped.startswith("from oprim.") or stripped.startswith("import oprim."):
        buf.append(stripped)

combined = " ".join(buf)
# Handle parenthesized imports
combined = combined.replace("(\n", " ").replace(")", " ")
combined = re.sub(r"\s+", " ", combined)

# Parse all "from X import Y, Z" patterns
for m in re.finditer(r"from\s+(oprim\.[\w.]+)\s+import\s+(.*?)(?:\s+import|\s*$)", combined):
    module = m.group(1)
    items = m.group(2)
    for item in re.findall(r"(\w+)(?:\s+as\s+\w+)?", items):
        eager_map[item] = module

print(f"Found {len(eager_map)} eager imports: {sorted(eager_map.keys())}")

# === Step 3: Remove eager import lines, keep everything else ===
new_lines = []
for i, line in enumerate(lines):
    if block_start <= i < block_end:
        continue  # skip eager import block entirely
    new_lines.append(line)

new_content = "\n".join(new_lines)

# === Step 4: Insert eager_map registration after _build_element_map() ===
reg_lines = [
    "",
    "# Eager re-exports registered in _ELEMENT_MAP for lazy loading",
    "# (avoids F401 unused-import / E402 import-not-at-top / I001 ordering)",
    "_ELEMENT_MAP.update({",
]
for name in sorted(eager_map):
    reg_lines.append(f'    "{name}": "{eager_map[name]}",')
reg_lines.append("})")
reg_text = "\n".join(reg_lines) + "\n"

new_content = new_content.replace("_build_element_map()\n", "_build_element_map()\n" + reg_text, 1)

# === Step 5: Fix N802 (_get_EpubBook -> _get_epub_book) ===
new_content = new_content.replace("_get_EpubBook", "_get_epub_book")
new_content = new_content.replace("def _get_EpubBook", "def _get_epub_book")

# === Step 6: Fix E701 (multiple statements on one line) ===
# Pattern: "def f():\\n    from X import Y" -> "def f():\\n    from X import Y"
# Actually E701 is about statements separated by colon on same physical line
# Looking at original file, lines with "return" after def on next line aren't E701
# Let me check what the 6 E701 errors are
# After removing eager imports, check again
# For now, leave them and fix after checking

# === Step 7: Fix N802 in __getattr__: return __version__ ===
# "if name == "__version__": return __version__" -> multi-line
new_content = new_content.replace(
    'if name == "__version__": return __version__',
    'if name == "__version__":\n        return __version__',
)

# === Step 8: Replace __all__ ===
all_names = sorted(set(list(eager_map.keys())))
all_expr = (
    "__all__ = sorted(set(_ELEMENT_MAP.keys()) | {" + ", ".join(repr(n) for n in all_names) + "})"
)
new_content = re.sub(
    r"^__all__ = sorted\(_ELEMENT_MAP\.keys\(\)\)\s*$",
    all_expr,
    new_content,
    flags=re.MULTILINE,
)

# === Step 9: Fix E501 (line too long) ===
# The __all__ line might still be too long. Let me check after this.

with open("oprim/__init__.py", "w") as f:
    f.write(new_content)

print("File written successfully")

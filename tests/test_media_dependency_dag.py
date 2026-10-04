"""P0-D — 3O Media Dependency-DAG Hardening.

The rule that used to be load-bearing (`no OPrim sibling bare calls`) was
vacuously green: `check_no_sibling_call.py` skipped `_`-prefixed files, and every
oprim implementation lives in `_foo.py`. So a module could be invisible to the
gate while being a public capability called by its siblings.

P0-D replaces the filename heuristic with a declaration (`__oprim_layer__`) and
closes the blind spot. These tests are the executable form of D1–D3:

* D1  element → element is forbidden
* D2  provider helpers are classified as `provider`, shared bases as `infra`
* D3  private files are scanned; classification never comes from the filename
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from oprim._layering import (
    LAYER_ELEMENT,
    LAYER_INFRA,
    LAYER_PROVIDER,
    LAYERS,
    declared_exports,
    layer_from_tree,
    module_layer,
)

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "oprim"
RUNNER = REPO.parent / ".github" / "scripts" / "3o_lint" / "check_no_sibling_call.py"

#: The canonical media surface governed by D1–D3. Everything here must be clean.
MEDIA_DAG = (
    "_media_probe",
    "_probe_duration",
    "_ffprobe",
    "_extract_video_frames",
    "_extract_media_segment",
    "_extract_audio_waveform",
    "_segment_media",
    "_encode_voice_reference",
    "_encode_frames",
    "_render_media",
    "_render_html_to_mp4",
    "_html_safety",
    "_validate_html",
    "_audio_mix",
    "_audio_normalize",
    "_audio_video_merge",
    "_subtitle_burn",
    "_transcribe_audio",
    "_video_generate",
    "_video_concat",
    "_video_recompose",
    "render_html_to_mp4",
    "validate_html",
    "video_concat",
    "video_generate",
    "media_probe",
    "probe_media",
    "render_media",
    "mix_audio_tracks",
    "burn_subtitles",
    "transcribe_media",
)


def _run_gate() -> tuple[int, list[str]]:
    proc = subprocess.run(
        [sys.executable, str(RUNNER), str(REPO.parent), "--layer", "oprim"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return proc.returncode, [ln for ln in proc.stdout.splitlines() if "No Sibling Call" in ln]


@pytest.fixture(scope="module")
def gate_output() -> list[str]:
    code, violations = _run_gate()
    assert code == 1, "gate must report the known non-media backlog"
    return violations


# ---------------------------------------------------------------------------
# D2 — the declaration itself
# ---------------------------------------------------------------------------


class TestLayerDeclaration:
    def test_defaults_to_element(self) -> None:
        assert layer_from_tree(ast.parse("x = 1\n")) == LAYER_ELEMENT

    @pytest.mark.parametrize("layer", sorted(LAYERS))
    def test_round_trips_declared_layer(self, layer: str) -> None:
        tree = ast.parse(f'__oprim_layer__ = "{layer}"\n')
        assert layer_from_tree(tree) == layer

    def test_rejects_unknown_layer(self) -> None:
        with pytest.raises(ValueError, match="not one of"):
            layer_from_tree(ast.parse('__oprim_layer__ = "middleware"\n'))

    def test_rejects_non_literal_layer(self) -> None:
        with pytest.raises(ValueError, match="plain string literal"):
            layer_from_tree(ast.parse("__oprim_layer__ = compute()\n"))

    def test_declared_exports_round_trip(self) -> None:
        tree = ast.parse('__oprim_exports__ = [\n    "A",\n    "B",\n]\n')
        assert declared_exports(tree) == frozenset({"A", "B"})

    def test_declared_exports_rejects_non_literal(self) -> None:
        with pytest.raises(ValueError, match="plain string literals"):
            declared_exports(ast.parse('__oprim_exports__ = [name()]\n'))

    def test_every_module_declares_a_valid_layer(self) -> None:
        offenders: list[str] = []
        for py in sorted(PKG.rglob("*.py")):
            try:
                module_layer(py.read_text(encoding="utf-8"), filename=str(py))
            except ValueError as exc:
                offenders.append(f"{py.name}: {exc}")
        assert offenders == []

    def test_declared_layer_matches_runtime_map(self) -> None:
        import oprim

        assert oprim._MODULE_LAYERS, "runtime layer map is empty"
        assert set(oprim._MODULE_LAYERS.values()) <= LAYERS


class TestMediaLayerClassification:
    """D2 — the media DAG must be classified the way its role actually is."""

    @pytest.mark.parametrize(
        ("module", "expected"),
        [
            ("_ffprobe", LAYER_INFRA),
            ("_encode_frames", LAYER_INFRA),
            ("_html_safety", LAYER_INFRA),
            ("_exceptions", LAYER_INFRA),
            ("_config", LAYER_INFRA),
            ("_media_types", LAYER_INFRA),
            ("_shot_types", LAYER_INFRA),
            ("_veo3_generate", LAYER_PROVIDER),
            ("_kling_v2_generate", LAYER_PROVIDER),
            ("_hailuo_generate", LAYER_PROVIDER),
            ("_ltx2_cloud_generate", LAYER_PROVIDER),
            ("_fal_queue_generate", LAYER_PROVIDER),
            ("_media_probe", LAYER_ELEMENT),
            ("_probe_duration", LAYER_ELEMENT),
            ("_extract_video_frames", LAYER_ELEMENT),
            ("_extract_audio_waveform", LAYER_ELEMENT),
            ("_segment_media", LAYER_ELEMENT),
            ("_extract_media_segment", LAYER_ELEMENT),
            ("_encode_voice_reference", LAYER_ELEMENT),
            ("_render_media", LAYER_ELEMENT),
            ("_render_html_to_mp4", LAYER_ELEMENT),
            ("_audio_mix", LAYER_ELEMENT),
            ("_subtitle_burn", LAYER_ELEMENT),
            ("_transcribe_audio", LAYER_ELEMENT),
            ("_video_generate", LAYER_ELEMENT),
        ],
    )
    def test_module_layer(self, module: str, expected: str) -> None:
        import oprim

        assert oprim._MODULE_LAYERS[f"oprim.{module}"] == expected

    def test_provider_adapters_are_all_declared(self) -> None:
        import oprim

        undeclared = [
            name
            for name in oprim._SUBMODULE_SET
            if name.endswith(("_generate", "_queue_generate"))
            and oprim._MODULE_LAYERS.get(f"oprim.{name}", LAYER_ELEMENT) == LAYER_ELEMENT
            and name
            in {
                "_veo3_generate",
                "_kling_v2_generate",
                "_hailuo_generate",
                "_ltx2_cloud_generate",
                "_fal_queue_generate",
            }
        ]
        assert undeclared == []


# ---------------------------------------------------------------------------
# D1 + D3 — the gate itself
# ---------------------------------------------------------------------------


class TestMediaDagIsClean:
    def test_zero_media_violations(self, gate_output: list[str]) -> None:
        offenders = [
            line
            for line in gate_output
            if any(f"oprim.{mod} " in line or f"oprim.{mod}(" in line for mod in MEDIA_DAG)
        ]
        assert offenders == [], "media DAG must have zero element→element edges:\n" + "\n".join(
            offenders
        )

    def test_no_media_module_imports_a_sibling_element(self) -> None:
        import oprim

        allow_infra = {
            LAYER_INFRA,
            LAYER_PROVIDER,
        }
        offenders: list[str] = []
        for name in MEDIA_DAG:
            path = PKG / f"{name}.py"
            if not path.exists():
                continue
            caller_layer = oprim._MODULE_LAYERS.get(f"oprim.{name}", LAYER_ELEMENT)
            if caller_layer in allow_infra:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                targets: list[str] = []
                if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    targets.append(node.module)
                elif isinstance(node, ast.Import):
                    targets.extend(a.name for a in node.names)
                for target in targets:
                    if not target.startswith("oprim."):
                        continue
                    if oprim._MODULE_LAYERS.get(target) == LAYER_ELEMENT:
                        # public facade re-exporting its own impl is one capability
                        if target[len("oprim.") :] == "_" + name:
                            continue
                        offenders.append(f"{name} -> {target}")
        assert offenders == []

    def test_ship_checker_cli_is_wired(self) -> None:
        """The old checker had no __main__, so running it was a silent no-op."""
        source = RUNNER.read_text(encoding="utf-8")
        assert 'if __name__ == "__main__":' in source

    def test_gate_scans_private_files(self) -> None:
        """D3 — the blind spot. `_foo.py` must be scanned, not skipped."""
        source = RUNNER.read_text(encoding="utf-8")
        assert 'startswith("_")' not in source, "checker must not skip underscore-prefixed files"

    def test_gate_facade_rule_is_not_a_blind_spot(self, tmp_path: Path) -> None:
        """`foo.py` re-exporting `_foo.py` is legal; `_bar.py` importing it is not."""
        pkg = tmp_path / "oprim" / "oprim"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "_thing.py").write_text("def thing():\n    return 1\n", encoding="utf-8")
        (pkg / "thing.py").write_text(
            "from oprim._thing import thing\n", encoding="utf-8"
        )
        (pkg / "_other.py").write_text(
            "from oprim._thing import thing\n", encoding="utf-8"
        )
        sys.path.insert(0, str(RUNNER.parent))
        try:
            import importlib

            import check_no_sibling_call as mod

            importlib.reload(mod)
            errors = mod.check_no_sibling_call(tmp_path)
        finally:
            sys.path.remove(str(RUNNER.parent))
        assert not any("oprim.thing" in e for e in errors), "facade must stay legal"
        assert any("oprim._other" in e for e in errors), "sibling import must be caught"


class TestNonMediaBacklogPinned:
    """The remaining violations are pre-existing, non-media, and out of P0-D scope.

    They are pinned so the count cannot silently grow. Fixing them is separate
    work: most are element modules importing other element modules in the
    finance/llm/storage domains (e.g. `oprim.llm._llm_complete`,
    `oprim.meta_db.duckdb`, `oprim._stripe_create_payment_intent`). The shared
    bases that dominate the earlier count — `oprim.types`, `oprim.errors`,
    `oprim._logging`, `oprim.git`, `oprim._cognitive`, `oprim._protocols` and the
    `*_types` / `*._base` modules — are now declared `infra` and no longer
    contribute.
    """

    EXPECTED_COUNT = 151

    #: Element→element compositions outside the media scope, same defect class as
    #: the pre-D video DAG. Removed from the 3o_lint allowlist in P0-D (the old
    #: `OPRIM_INFRA_PREFIXES` list had been hiding them) and pinned here instead.
    KNOWN_ELEMENT_COMPOSITIONS = (
        ("oprim._llm_chat_call", "oprim.llm._llm_complete"),
        ("oprim._embedding_gen", "oprim.embed_text"),
        ("oprim._image_analyze", "oprim.image_understand"),
    )

    def test_backlog_count_is_pinned(self, gate_output: list[str]) -> None:
        assert len(gate_output) == self.EXPECTED_COUNT, (
            f"non-media backlog moved: {len(gate_output)} != {self.EXPECTED_COUNT}. "
            "Either fix the edges (preferred) or update this pin deliberately."
        )

    def test_known_element_compositions_are_still_pinned(self, gate_output: list[str]) -> None:
        found = set()
        for line in gate_output:
            if "imports element" not in line:
                continue
            caller = line.split("] ")[1].split(" ")[0]
            callee = line.split("imports element ")[1].split(" ")[0].strip("'")
            found.add((caller, callee))
        for caller, callee in self.KNOWN_ELEMENT_COMPOSITIONS:
            assert (caller, callee) in found, f"{caller} -> {callee} left the backlog"

    def test_shared_bases_no_longer_contribute(self, gate_output: list[str]) -> None:
        """D2 regression guard: the classified infra bases must stay out of the backlog."""
        former_top = (
            "oprim.types",
            "oprim.errors",
            "oprim._logging",
            "oprim.git",
            "oprim._cognitive",
            "oprim._protocols",
            "oprim._version",
            "oprim.technical._base",
            "oprim.timeseries._base",
            "oprim.changefeed.schema",
            "oprim.changefeed._bootstrap",
            "oprim._exceptions",
            "oprim._config",
            "oprim._media_types",
            "oprim._shot_types",
            "oprim._ffprobe",
            "oprim._encode_frames",
            "oprim._html_safety",
        )
        offenders = [
            line
            for line in gate_output
            if any(f"imports element '{m}'" in line for m in former_top)
        ]
        assert offenders == []

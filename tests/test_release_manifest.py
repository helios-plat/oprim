"""The release manifest HEVI's `test_three_o_contracts` depends on.

`oprim.__manifest__` was lost when the local line diverged from the commit HEVI
pins (`86fa205` has no `__manifest__`, `5d992f2c` does). Any re-pin would
therefore have broken HEVI's release gate. It is restored here as a *derived*
view over `_ELEMENT_MAP` rather than the pinned version's hand-written list,
whose signatures did not match the real functions (`(request, /, *, provider,
output_path)` — no such parameter exists).

Consumer contract, from `hevi/tests/test_three_o_contracts.py` and
`hevi/scripts/ci/run_3o_v3_manifest_audit.py`:

* `manifest["package"] == package.__name__`
* `manifest["version"] == package.__version__`
* `manifest["elements"]` is a non-empty list
* every entry carries `name` + `module`, and the audit script computes
  `dangling = not entity_exists(path, name)` — so no entry may name something
  the module does not define
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import oprim

PKG = Path(oprim.__file__).parent


def _source_for(module_path: str) -> Path | None:
    rel = Path(*module_path.split(".")[1:])
    direct = PKG / rel.with_suffix(".py")
    if direct.exists():
        return direct
    pkg = PKG / rel / "__init__.py"
    return pkg if pkg.exists() else None


def _defines(path: Path, name: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name == name
        for node in tree.body
    )


class TestManifestShape:
    def test_package_and_version(self) -> None:
        assert oprim.__manifest__["package"] == oprim.__name__
        assert oprim.__manifest__["version"] == oprim.__version__

    def test_elements_non_empty(self) -> None:
        elements = oprim.__manifest__["elements"]
        assert isinstance(elements, list)
        assert elements

    def test_every_entry_has_the_consumer_fields(self) -> None:
        for entry in oprim.__manifest__["elements"]:
            assert entry["name"], entry
            assert entry["module"].startswith("oprim."), entry
            assert entry["kind"] == "oprim", entry


class TestManifestIntegrity:
    """The audit script flags entries whose entity does not exist."""

    def test_no_dangling_entries(self) -> None:
        dangling: list[str] = []
        for entry in oprim.__manifest__["elements"]:
            source = _source_for(entry["module"])
            if source is None:
                dangling.append(f"{entry['name']}: module {entry['module']} not on disk")
            elif not _defines(source, entry["name"]):
                dangling.append(f"{entry['name']}: not defined in {entry['module']}")
        assert dangling == []

    def test_every_published_element_is_importable(self) -> None:
        for entry in oprim.__manifest__["elements"]:
            assert hasattr(oprim, entry["name"]), entry["name"]

    def test_manifest_is_derived_not_hand_written(self) -> None:
        """A hardcoded list drifts; this is the failure D5 removed once already."""
        source = Path(oprim.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        literal_manifest = any(
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__manifest__"
            and isinstance(node.value, ast.Dict)
            for node in tree.body
        )
        assert not literal_manifest, "__manifest__ must be computed, not a literal dict"
        assert any(
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__manifest__"
            and isinstance(node.value, ast.Call)
            for node in tree.body
        ), "__manifest__ must be assigned from _build_manifest()"


class TestManifestCoversCanonicalSurface:
    @pytest.mark.parametrize(
        "name",
        [
            # the P0 canonical media set
            "media_probe",
            "probe_duration",
            "transcribe_audio",
            "extract_video_frames",
            "extract_audio_waveform",
            "segment_media",
            "extract_media_segment",
            "render_media",
            "render_html_to_mp4",
            "audio_mix",
            "audio_normalize",
            "audio_video_merge",
            "subtitle_burn",
            "encode_voice_reference",
            # pre-existing names HEVI consumes directly
            "video_generate",
            "tts_synthesize",
            "avatar_generate",
            "validate_html",
            "style_marker_prompt",
        ],
    )
    def test_published(self, name: str) -> None:
        assert name in {e["name"] for e in oprim.__manifest__["elements"]}

    def test_layer_is_recorded_and_valid(self) -> None:
        for entry in oprim.__manifest__["elements"]:
            assert entry["layer"] in {"element", "infra", "provider"}, entry


class TestElementSignature:
    def test_signature_is_real_not_aspirational(self) -> None:
        """The pinned manifest advertised `(request, /, *, provider, ...)`."""
        sig = oprim.element_signature("video_generate")
        assert sig is not None
        assert "request" not in sig
        assert sig.startswith("video_generate(")

    def test_signature_matches_the_live_object(self) -> None:
        for entry in oprim.__manifest__["elements"][:8]:
            name = entry["name"]
            sig = oprim.element_signature(name)
            if sig is None:
                continue
            live = str(inspect.signature(getattr(oprim, name)))
            assert live in sig, name

    def test_unknown_name_returns_none(self) -> None:
        assert oprim.element_signature("definitely_not_an_element_xyz") is None


class TestDependsOn:
    def test_media_probe_reports_its_infra(self) -> None:
        deps = oprim._element_depends_on("media_probe")
        assert "oprim._ffprobe" in deps
        assert "oprim._exceptions" in deps

    def test_probe_duration_depends_on_the_single_executor(self) -> None:
        """D5/P0-B: one ffprobe implementation, not two."""
        assert oprim._element_depends_on("probe_duration") == ["oprim._ffprobe"]

    def test_render_html_uses_infra_not_the_sibling_element(self) -> None:
        deps = oprim._element_depends_on("render_html_to_mp4")
        assert "oprim._html_safety" in deps
        assert "oprim._validate_html" not in deps

    def test_no_manifest_element_composes_another_manifest_element(self) -> None:
        """Mirrors the DAG rule, scoped to the published surface."""
        published = {e["module"] for e in oprim.__manifest__["elements"]}
        offenders: list[str] = []
        for entry in oprim.__manifest__["elements"]:
            for dep in oprim._element_depends_on(entry["name"]):
                if dep in published:
                    offenders.append(f"{entry['name']} -> {dep}")
        assert offenders == []

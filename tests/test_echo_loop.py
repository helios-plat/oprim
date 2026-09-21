import ast
import inspect
from pathlib import Path

import pytest
from pydantic import ValidationError

from oprim.echo_loop import (
    BlindListenOutput,
    IntensiveListenOutput,
    RetellOutput,
    ShadowingOutput,
    blind_listen_generate,
    intensive_listen_parse,
    retell_evaluate,
    shadowing_evaluate,
)


@pytest.mark.asyncio
async def test_echo_primitives_return_canonical_shapes() -> None:
    blind = await blind_listen_generate(
        audio_b64="audio",
        transcript="A short sentence.",
        reference_kc_ids=["kc-1"],
    )
    intensive = await intensive_listen_parse(
        transcript="A short sentence.",
        blind_listen_output=blind.model_dump(),
    )
    shadowing = await shadowing_evaluate(
        reference_text="A short sentence.",
        student_audio_b64="audio",
    )
    retell = await retell_evaluate(
        original_text="A short sentence.",
        student_retell="A short sentence.",
    )

    assert isinstance(blind, BlindListenOutput)
    assert isinstance(intensive, IntensiveListenOutput)
    assert isinstance(shadowing, ShadowingOutput)
    assert isinstance(retell, RetellOutput)
    assert (
        blind.model_dump()
        == (
            await blind_listen_generate(
                audio_b64="audio",
                transcript="A short sentence.",
                reference_kc_ids=["kc-1"],
            )
        ).model_dump()
    )


def test_echo_public_signatures_are_keyword_only() -> None:
    for function in (
        blind_listen_generate,
        intensive_listen_parse,
        shadowing_evaluate,
        retell_evaluate,
    ):
        parameters = inspect.signature(function).parameters.values()
        assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters)


@pytest.mark.asyncio
async def test_invalid_scores_fail_closed() -> None:
    with pytest.raises(ValidationError):
        await shadowing_evaluate(
            reference_text="text",
            student_audio_b64="audio",
            pronunciation_scores={"overall": 2.0, "fluency": 2.0, "accuracy": 2.0},
        )


def test_echo_module_has_no_sibling_oprim_imports() -> None:
    source = Path(__file__).parents[1] / "oprim" / "echo_loop.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("oprim.")
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("oprim") for alias in node.names)

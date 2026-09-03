"""The tool layer: declaration, registration, and validation *before* dispatch.

The theme of every test here is that a malformed call must be caught by the schema and turned
into a message a model can act on — never reach the callable, never crash the run, and never
be quietly repaired into something the model did not ask for.
"""

from __future__ import annotations

import pytest

from hearth.agent import (
    Tool,
    ToolDefinitionError,
    ToolParam,
    ToolRegistry,
    ToolValidationError,
    UnknownToolError,
    render_observation,
)


def _echo_tool(**kwargs) -> Tool:
    defaults = dict(
        name="echo",
        description="Echo a value back.",
        call=lambda text: text,
        params=(ToolParam(name="text", type="string", description="What to echo."),),
    )
    defaults.update(kwargs)
    return Tool(**defaults)


# -- declaration -----------------------------------------------------------------------


def test_a_tool_name_must_be_snake_case_because_it_is_matched_verbatim():
    with pytest.raises(ToolDefinitionError, match="snake_case"):
        _echo_tool(name="Read File")


def test_a_parameter_without_a_description_is_refused():
    with pytest.raises(ToolDefinitionError, match="description"):
        ToolParam(name="x", type="string", description="   ")


def test_a_required_parameter_may_not_also_carry_a_default():
    with pytest.raises(ToolDefinitionError, match="default"):
        ToolParam(name="x", type="string", description="d", required=True, default="oops")


def test_an_unsupported_parameter_type_is_refused():
    with pytest.raises(ToolDefinitionError, match="unsupported type"):
        ToolParam(name="x", type="object", description="d")  # type: ignore[arg-type]


def test_the_rendered_tool_block_carries_everything_the_model_needs():
    tool = _echo_tool(returns="the same text")
    block = tool.render()
    assert "echo(text: string)" in block
    assert "Echo a value back." in block
    assert "What to echo." in block
    assert "returns: the same text" in block


# -- registry --------------------------------------------------------------------------


def test_a_registry_starts_empty_so_capability_is_always_an_explicit_choice():
    assert len(ToolRegistry()) == 0
    assert ToolRegistry().names == ()


def test_registering_a_duplicate_name_is_refused_rather_than_overwriting():
    registry = ToolRegistry([_echo_tool()])
    with pytest.raises(ToolDefinitionError, match="already registered"):
        registry.register(_echo_tool(description="a different echo"))


def test_registering_a_non_tool_is_refused():
    with pytest.raises(ToolDefinitionError, match="hearth.agent.Tool"):
        ToolRegistry().register(lambda: None)  # type: ignore[arg-type]


def test_an_unknown_tool_names_the_ones_that_do_exist():
    registry = ToolRegistry([_echo_tool()])
    with pytest.raises(UnknownToolError, match="Available tools: echo"):
        registry.get("read_file")


# -- validation ------------------------------------------------------------------------


def test_a_missing_required_argument_names_the_parameter():
    with pytest.raises(ToolValidationError, match="requires the parameter 'text'"):
        _echo_tool().validate({})


def test_an_unknown_argument_is_refused_and_the_real_ones_listed():
    with pytest.raises(ToolValidationError, match="has no parameter"):
        _echo_tool().validate({"txt": "hi"})


def test_an_optional_argument_falls_back_to_its_declared_default():
    tool = _echo_tool(
        params=(
            ToolParam(name="text", type="string", description="d"),
            ToolParam(name="k", type="integer", description="d", required=False, default=6),
        )
    )
    assert tool.validate({"text": "hi"}) == {"text": "hi", "k": 6}


@pytest.mark.parametrize(
    ("declared", "given", "expected"),
    [
        ("integer", "5", 5),
        ("integer", 5.0, 5),
        ("number", "2.5", 2.5),
        ("number", 3, 3.0),
        ("boolean", "true", True),
        ("boolean", "FALSE", False),
    ],
)
def test_a_round_trippable_string_scalar_is_coerced(declared, given, expected):
    # JSON cannot distinguish 5 from "5" and a small model picks almost at random; the
    # coercion is narrow enough that it can never change the value the model meant.
    tool = _echo_tool(
        params=(ToolParam(name="v", type=declared, description="d"),), call=lambda v: v
    )
    assert tool.validate({"v": given}) == {"v": expected}


@pytest.mark.parametrize(
    ("declared", "given"),
    [
        ("integer", "5.7"),
        ("integer", True),
        ("integer", "five"),
        ("number", "yes"),
        ("boolean", "yes"),
        ("boolean", 1),
        ("string", 5),
        ("string", ["a"]),
    ],
)
def test_a_value_that_is_not_round_trippable_is_refused_not_guessed(declared, given):
    tool = _echo_tool(
        params=(ToolParam(name="v", type=declared, description="d"),), call=lambda v: v
    )
    with pytest.raises(ToolValidationError):
        tool.validate({"v": given})


def test_a_validation_message_names_the_type_never_the_value():
    tool = _echo_tool(params=(ToolParam(name="v", type="integer", description="d"),))
    with pytest.raises(ToolValidationError) as exc:
        tool.validate({"v": "hunter2-the-secret"})
    assert "hunter2" not in str(exc.value)


def test_a_value_outside_choices_is_refused():
    tool = _echo_tool(
        params=(
            ToolParam(name="mode", type="string", description="d", choices=("a", "b")),
        ),
        call=lambda mode: mode,
    )
    assert tool.validate({"mode": "a"}) == {"mode": "a"}
    with pytest.raises(ToolValidationError, match="must be one of a, b"):
        tool.validate({"mode": "c"})


def test_arguments_that_are_not_an_object_are_refused():
    with pytest.raises(ToolValidationError, match="must be a JSON object"):
        _echo_tool().validate(["text"])  # type: ignore[arg-type]


# -- observation rendering --------------------------------------------------------------


def test_a_long_observation_is_truncated_visibly_never_silently():
    rendered = render_observation("x" * 500, limit=100)
    assert rendered.startswith("x" * 100)
    assert "truncated" in rendered
    assert "500" in rendered  # the model is told how much it did not see


def test_a_short_observation_is_passed_through_untouched():
    assert render_observation("hello", limit=100) == "hello"


def test_structured_results_render_as_readable_lines():
    assert render_observation(["a", "b"], limit=100) == "- a\n- b"
    assert render_observation({"amount": "12.50"}, limit=100) == "amount: 12.50"
    assert render_observation([], limit=100) == "(empty list)"
    assert render_observation(None, limit=100) == "(no result)"

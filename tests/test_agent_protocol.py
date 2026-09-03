"""The output contract: what a local model may get away with, and what it may not.

The split these tests police is the one the module docstring argues for — formatting noise is
absorbed because refusing it only costs iterations, while anything *ambiguous about which
action was requested* is refused, because guessing there runs the wrong thing and the
transcript would show a step the model never asked for.
"""

from __future__ import annotations

import pytest

from hearth.agent import (
    FinalAnswer,
    ProtocolError,
    Tool,
    ToolCall,
    ToolParam,
    ToolRegistry,
    parse_action,
    render_system_prompt,
)

# -- the shapes that must parse ----------------------------------------------------------


def test_the_documented_tool_call_shape_parses():
    action = parse_action('{"thought": "look", "tool": "read_file", "arguments": {"path": "/a"}}')
    assert action == ToolCall(name="read_file", arguments={"path": "/a"}, thought="look")


def test_the_documented_answer_shape_parses():
    action = parse_action('{"thought": "enough", "answer": "42"}')
    assert action == FinalAnswer(text="42", thought="enough")


def test_the_native_mlx_tool_call_block_parses_unchanged():
    # mlx-lm's server emits {"name": ..., "arguments": ...} between <tool_call> sentinels.
    # The contract is shaped to accept that payload verbatim, so a model that has been
    # trained to emit it needs no retraining and no second parser.
    action = parse_action(
        '<tool_call>\n{"name": "read_file", "arguments": {"path": "/a"}}\n</tool_call>'
    )
    assert action == ToolCall(name="read_file", arguments={"path": "/a"})


def test_a_thinking_block_is_stripped_before_parsing():
    action = parse_action(
        '<think>I should read the file first, or maybe not.</think>\n'
        '{"tool": "read_file", "arguments": {"path": "/a"}}'
    )
    assert isinstance(action, ToolCall)


def test_a_fenced_object_with_prose_around_it_parses():
    action = parse_action('Sure, here you go:\n```json\n{"answer": "done"}\n```\nHope that helps!')
    assert action == FinalAnswer(text="done")


def test_arguments_delivered_as_a_json_string_are_accepted():
    action = parse_action('{"tool": "read_file", "arguments": "{\\"path\\": \\"/a\\"}"}')
    assert action == ToolCall(name="read_file", arguments={"path": "/a"})


def test_omitted_arguments_become_an_empty_object_so_the_schema_gives_the_message():
    # Deliberately not a protocol error: Tool.validate names the missing parameter, which is
    # a far more actionable thing for a 3B model to read than "arguments missing".
    assert parse_action('{"tool": "list_files"}') == ToolCall(name="list_files", arguments={})


# -- the shapes that must be refused ------------------------------------------------------


def test_output_with_no_json_object_is_refused():
    with pytest.raises(ProtocolError, match="no JSON object"):
        parse_action("I will now read the file for you.")


def test_two_json_objects_are_refused_rather_than_taking_the_first():
    # Taking the first is how an agent silently skips the step it told you it took.
    with pytest.raises(ProtocolError, match="2 JSON objects"):
        parse_action('{"tool": "a", "arguments": {}}\n{"tool": "b", "arguments": {}}')


def test_two_tool_call_blocks_are_refused():
    with pytest.raises(ProtocolError, match="more than one <tool_call>"):
        parse_action("<tool_call>{}</tool_call><tool_call>{}</tool_call>")


def test_a_tool_call_and_an_answer_together_are_refused():
    with pytest.raises(ProtocolError, match="both a tool call and an answer"):
        parse_action('{"tool": "read_file", "arguments": {}, "answer": "already done"}')


def test_neither_a_tool_nor_an_answer_is_refused_and_the_keys_are_named():
    with pytest.raises(ProtocolError, match="neither"):
        parse_action('{"thought": "hmm", "next_step": "read the file"}')


def test_an_empty_answer_is_refused():
    with pytest.raises(ProtocolError, match="`answer` was empty"):
        parse_action('{"answer": "   "}')


def test_an_empty_tool_name_is_refused():
    with pytest.raises(ProtocolError, match="`tool` was empty"):
        parse_action('{"tool": "  ", "arguments": {}}')


def test_arguments_that_are_a_scalar_are_refused():
    with pytest.raises(ProtocolError, match="must be a JSON object"):
        parse_action('{"tool": "read_file", "arguments": 7}')


def test_arguments_that_are_an_unparseable_string_are_refused():
    with pytest.raises(ProtocolError, match="not valid JSON"):
        parse_action('{"tool": "read_file", "arguments": "path=/a"}')


def test_every_refusal_restates_the_contract_to_the_model():
    # The message is fed straight back into the next prompt, so it has to be an instruction.
    for bad in ("nothing here", '{"thought": "hi"}', '{"tool": "a", "answer": "b"}'):
        with pytest.raises(ProtocolError) as exc:
            parse_action(bad)
        assert "JSON object" in str(exc.value) or "Decide" in str(exc.value)


# -- the prompt ---------------------------------------------------------------------------


def test_the_system_prompt_describes_exactly_the_registered_tools():
    registry = ToolRegistry(
        [
            Tool(
                name="read_file",
                description="Read one local file.",
                call=lambda path: path,
                params=(ToolParam(name="path", type="string", description="Absolute path."),),
            )
        ]
    )
    prompt = render_system_prompt(registry, task="Summarise the notes.")
    assert "read_file(path: string)" in prompt
    assert "Absolute path." in prompt
    assert "Summarise the notes." in prompt
    assert "exactly ONE JSON object" in prompt
    # The tool block is rendered from the registry the loop dispatches against, so the
    # description and the dispatch table cannot drift (CLAUDE.md §3).
    assert "list_files" not in prompt


def test_the_prompt_states_the_absence_of_network_shell_and_writes():
    prompt = render_system_prompt(
        ToolRegistry([Tool(name="noop", description="Do nothing.", call=lambda: None)]),
        task="anything",
    )
    assert "no network" in prompt
    assert "no shell" in prompt
    assert "write files" in prompt

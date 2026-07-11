from oczy.experiments.tool_calling_curriculum import (
    build_tool_curriculum,
    parse_tool_call,
    score_episode,
    validate_tool_curriculum,
)


def test_complete_curriculum_validates() -> None:
    stages = build_tool_curriculum()
    assert [len(stage.episodes) for stage in stages] == [8, 12, 8, 6, 8, 3]
    assert validate_tool_curriculum(stages) == []


def test_parser_supports_json_and_brackets() -> None:
    json_call = parse_tool_call('{"name":"read","arguments":{"path":"a.py"}}')
    bracket_call = parse_tool_call('[edit(path="a.py", old="x", new="y")]')
    assert json_call is not None and json_call.name == "read"
    assert json_call.arguments["path"] == "a.py"
    assert bracket_call is not None and bracket_call.name == "edit"
    assert bracket_call.arguments["new"] == "y"


def test_result_integration_and_sequence_are_scored() -> None:
    episode = build_tool_curriculum()[2].episodes[0]
    score = score_episode(
        episode,
        [
            '[read(path="pyproject.toml")]',
            "The project name is oczy.",
        ],
    )
    assert score.passed


def test_wrong_tool_fails_without_fuzzy_credit() -> None:
    episode = build_tool_curriculum()[0].episodes[0]
    score = score_episode(episode, ['[bash(command="cat config.toml")]'])
    assert not score.tool_sequence_correct
    assert not score.passed

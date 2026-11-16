from datetime import UTC, datetime, timedelta

import pytest

from app.content.repository import ContentRepository, get_repository
from app.services import srs
from app.services.exercises import ExerciseType, build_lesson, resolve_exercise


def test_quality_mapping():
    assert [srs.score_to_quality(s, 0.75) for s in (0.99, 0.9, 0.8, 0.6, 0.4, 0.1)] == [5, 4, 3, 2, 1, 0]


def test_schedule_progression_and_lapse():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    s1, d1 = srs.schedule(srs.SrsState(), 4, now)
    assert s1.repetitions == 1 and d1 == now + timedelta(days=1)
    s2, _ = srs.schedule(s1, 4, now)
    assert s2.interval_days == 3.0
    s3, _ = srs.schedule(s2, 5, now)
    assert s3.interval_days > 3.0
    lapsed, due = srs.schedule(s3, 1, now)
    assert lapsed.repetitions == 0 and lapsed.lapses == 1 and due - now < timedelta(hours=1)
    assert lapsed.ease >= srs.MIN_EASE


def test_lesson_has_varied_contexts_across_four_stages():
    repo = get_repository()
    plan = build_lesson(repo.get("a1-ordering-coffee"))
    assert plan.flow_mode == "varied_context_by_phrase"
    assert [s.type for s in plan.stages] == [
        ExerciseType.SHADOWING,
        ExerciseType.GUIDED,
        ExerciseType.COMPLETION,
        ExerciseType.RECALL,
    ]
    shadow, guided, completion, recall = (plan.stages[i].exercises[0] for i in range(4))
    assert shadow.prompt.glossary and shadow.prompt.translation
    assert "_" in guided.prompt.display_text and guided.prompt.translation and guided.prompt.audio_text
    assert "____" in completion.prompt.display_text
    assert completion.prompt.translation is None and completion.prompt.audio_text
    assert "_" in recall.prompt.display_text and recall.prompt.translation and recall.prompt.glossary
    # All 4 stages for the same phrase must present 4 distinct varied contexts
    contexts = {shadow.expected_text, guided.expected_text, completion.expected_text, recall.expected_text}
    assert len(contexts) == 4
    assert all(e.response_mode == "speech" for st in plan.stages for e in st.exercises)
    thresholds = [st.exercises[0].pass_threshold for st in plan.stages]
    assert thresholds == sorted(thresholds)


def test_resolve_exercise_rejects_bad_ids():
    repo = get_repository()
    assert resolve_exercise(repo, "nope") is None
    assert resolve_exercise(repo, "a1-ordering-coffee.can-i-have.bogus") is None
    assert resolve_exercise(repo, "missing.can-i-have.recall") is None
    assert resolve_exercise(repo, "a1-ordering-coffee.can-i-have.recall").expected_text


def test_repository_filters_and_duplicates():
    repo = get_repository()
    assert {s.cefr.band for s in repo.list(band="advanced")} == {"advanced"}
    assert len(repo.common_phrases()) >= 6
    with pytest.raises(ValueError):
        ContentRepository([repo.get("a1-ordering-coffee")] * 2)


def test_schema_validation_errors():
    from pydantic import ValidationError

    from app.content.schema import Phrase, RoleplayTurn, Scenario

    with pytest.raises(ValidationError):
        Phrase(id="x", text="hi there", translation="hola", cloze=[5])
    with pytest.raises(ValidationError):
        RoleplayTurn(speaker="user", goal="g")
    with pytest.raises(ValidationError):
        RoleplayTurn(speaker="ai")
    with pytest.raises(ValidationError):
        RoleplayTurn(speaker="bot", text="x")
    base = get_repository().get("a1-ordering-coffee").model_dump()
    base["phrases"].append(base["phrases"][0])
    with pytest.raises(ValidationError):
        Scenario.model_validate(base)


def test_validate_cli(capsys):
    from app.content.validate import main

    assert main([]) == 0
    assert "384 scenarios" in capsys.readouterr().out

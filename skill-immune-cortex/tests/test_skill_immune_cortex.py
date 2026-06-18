"""Tests for the Skill Immune Cortex."""

from skill_immune_cortex import MistakeDetector, SkillImmuneCortex


def test_matching_query_returns_immune_response():
    cortex = SkillImmuneCortex()
    cortex.add_detector(
        "You treated SQL injection safety as the whole story for safe queries in sqlx.",
        "query_safety",
        "Distinguish injection safety, compile-time correctness, and schema drift.",
    )
    warnings = cortex.check(
        "How do I write a safe query with sqlx?",
        "Use string concatenation for flexibility.",
    )
    assert len(warnings) == 1
    assert "injection" in warnings[0]
    assert "compile-time" in warnings[0]
    assert "schema drift" in warnings[0]


def test_non_matching_query_returns_no_response():
    cortex = SkillImmuneCortex()
    cortex.add_detector(
        "You treated SQL injection safety as the whole story for safe queries in sqlx.",
        "query_safety",
        "Distinguish injection safety, compile-time correctness, and schema drift.",
    )
    warnings = cortex.check(
        "What is the capital of France?",
        "Paris.",
    )
    assert warnings == []


def test_repeated_similar_detectors_merge():
    cortex = SkillImmuneCortex()
    cortex.add_detector(
        "You ignored sqlx compile-time query validation.",
        "query_safety",
        "Check compile-time query validation.",
    )
    cortex.add_detector(
        "You forgot about schema drift when reviewing ORM queries.",
        "query_safety",
        "Check schema drift assumptions.",
    )

    assert len(cortex.detectors) == 2
    cortex.merge_detectors()
    assert len(cortex.detectors) == 1
    assert len(cortex.merged) == 1
    merged = cortex.merged[0]
    assert merged.mistake_class == "query_safety"
    assert "sqlx" in merged.triggers
    assert "schema" in merged.triggers
    assert "compile-time" in merged.response or "schema drift" in merged.response


def test_skills_activate_on_trigger():
    cortex = SkillImmuneCortex()
    for _ in range(3):
        cortex.add_detector(
            "You ignored sqlx compile-time query validation.",
            "query_safety",
            "Check compile-time query validation.",
        )

    assert "query_safety" in cortex.skills
    warnings = cortex.check(
        "Is this sqlx query safe?",
        "It looks fine to me.",
    )
    assert any("query_safety" in w for w in warnings)
    assert cortex.skills["query_safety"].usage_count >= 1


def test_status_snapshot():
    cortex = SkillImmuneCortex()
    cortex.add_detector("You confused ORM guarantees with type safety.", "orm_guarantees", "Distinguish ORM and type safety.")
    status = cortex.status()
    assert status["ready"] is True
    assert status["detector_count"] == 1
    assert status["merged_count"] == 0
    assert status["bytes"] > 0


def test_detector_object_matches_and_counts_hits():
    detector = MistakeDetector(
        triggers=["sqlx", "safe query"],
        response="Distinguish injection, compile-time, schema drift.",
        mistake_class="query_safety",
    )
    assert detector.matches("Is my sqlx safe query okay?")
    assert not detector.matches("Tell me a joke.")
    assert detector.hit_count == 0

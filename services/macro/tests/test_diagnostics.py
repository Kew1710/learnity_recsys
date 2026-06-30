from services.macro.diagnostics import diagnose


def _defaults(**overrides):
    base = dict(
        mastery_current=0.5,
        velocity=0.0,
        frustration_count=0,
        avg_score=0.6,
        tasks_spent=10,
        attempts_count=15,
        mastery_confidence=0.6,
        weakest_prereq_mastery=None,
        task_count_for_kc=None,
    )
    base.update(overrides)
    return base


def test_on_track_when_no_issues():
    d = diagnose(**_defaults(avg_score=0.7, velocity=0.01))
    assert d.reason == "on_track"


def test_uncertain_estimate_low_attempts():
    d = diagnose(**_defaults(
        frustration_count=3, avg_score=0.3,
        mastery_confidence=0.1, attempts_count=3,
    ))
    assert d.reason == "uncertain_estimate"


def test_content_gap_few_tasks():
    d = diagnose(**_defaults(
        frustration_count=2, avg_score=0.3,
        task_count_for_kc=2,
    ))
    assert d.reason == "content_gap"


def test_prereq_gap_weak_prereq():
    d = diagnose(**_defaults(
        frustration_count=2, avg_score=0.3,
        weakest_prereq_mastery=0.3,
    ))
    assert d.reason == "prereq_gap"


def test_regression_high_mastery_low_score():
    d = diagnose(**_defaults(
        mastery_current=0.85, frustration_count=3, avg_score=0.3,
    ))
    assert d.reason == "regression"


def test_prereq_gap_weak_signal():
    d = diagnose(**_defaults(
        frustration_count=2, avg_score=0.3,
        weakest_prereq_mastery=0.55,
    ))
    assert d.reason == "prereq_gap"
    assert d.confidence < 0.7


def test_content_gap_takes_priority_over_prereq():
    d = diagnose(**_defaults(
        frustration_count=2, avg_score=0.3,
        task_count_for_kc=1,
        weakest_prereq_mastery=0.3,
    ))
    assert d.reason == "content_gap"


def test_uncertain_estimate_takes_priority_over_content():
    d = diagnose(**_defaults(
        frustration_count=2, avg_score=0.3,
        mastery_confidence=0.2, attempts_count=4,
        task_count_for_kc=2,
    ))
    assert d.reason == "uncertain_estimate"

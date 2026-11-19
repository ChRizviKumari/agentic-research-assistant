from research_agent.evals import default_tasks, format_report, run_suite, run_task, score_task
from research_agent.agent import AgentResult
from research_agent.search import LocalCorpus


def test_search_snippet_includes_answer_sentence():
    hit = LocalCorpus().search("Mount Everest elevation")[0]
    assert hit.doc_id == "mount_everest" and "8,849" in hit.snippet


def test_search_empty_corpus_and_ordering(tmp_path):
    assert LocalCorpus(tmp_path).search("anything") == []
    assert [h.doc_id for h in LocalCorpus().search("Transformer attention")][0] == "transformers"


def test_full_eval_suite_passes():
    outcomes = run_suite()
    assert len(outcomes) >= 10
    failed = [(o.id, o.reasons) for o in outcomes if not o.passed]
    assert not failed, failed
    assert f"{len(outcomes)}/{len(outcomes)} tasks passed" in format_report(outcomes)


def test_scoring_detects_failures():
    task = default_tasks()[0]  # calc_basic
    bad = AgentResult("success", "wrong", 1, 1, trace=[])
    out = score_task(task, bad)
    assert not out.passed and len(out.reasons) == 2  # missing answer, tool never called
    task.script = ["garbage"]
    assert not run_task(task).passed

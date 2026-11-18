"""Recompute descriptive results from frozen local predictions. Run with Python and ReportLab."""
import argparse
import csv
import json
import statistics as st
import subprocess
from collections import Counter
from pathlib import Path

import audit_stage5_results as audit
from validate_dataset import PERSONALITY, parse_base
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics import renderSVG
from reportlab.lib.colors import HexColor

ROOT = audit.ROOT
SEEDS = [13, 37, 73, 109, 149]
REPORTS = ROOT / "reports"


def metrics(labels, predictions):
    macro, accuracy, matrix = audit.scores(labels, predictions)
    result = {"macro_f1": macro, "accuracy": accuracy}
    for k, name in enumerate(("false", "true")):
        correct = matrix[k][k]
        support = sum(matrix[k])
        predicted = sum(row[k] for row in matrix)
        result.update({f"{name}_support": support, f"{name}_predicted": predicted,
                       f"{name}_precision": correct / predicted if predicted else 0,
                       f"{name}_recall": correct / support if support else 0,
                       f"{name}_f1": 2 * correct / (support + predicted) if support + predicted else 0})
    return result, matrix


def plot(runs, output_dir):
    drawing = Drawing(600, 330)
    drawing.add(String(24, 304, "Target selection success: Test Macro-F1", fontSize=17))
    drawing.add(String(24, 282, "Blue: base input (A)     Orange: personality + base (B)     Dashed: ACL Table 5", fontSize=10))
    graph = LinePlot()
    graph.x, graph.y, graph.width, graph.height = 65, 64, 480, 190
    graph.xValueAxis.valueMin, graph.xValueAxis.valueMax = 0.8, 5.2
    graph.xValueAxis.valueSteps = [1, 2, 3, 4, 5]
    graph.xValueAxis.labelTextFormat = lambda v: str(SEEDS[int(v)-1])
    graph.yValueAxis.valueMin, graph.yValueAxis.valueMax, graph.yValueAxis.valueStep = .50, .70, .05
    graph.data = [list(enumerate([runs[c,s]["macro_f1"] for s in SEEDS], 1)) for c in "AB"]
    graph.data += [[(1,v),(5,v)] for v in (.57,.67)]
    from reportlab.graphics.widgets.markers import makeMarker
    for i in range(4):
        color = HexColor(["#165785","#ae460b"][i%2])
        graph.lines[i].strokeColor = color
        if i < 2:
            graph.lines[i].symbol = makeMarker("FilledCircle" if i == 0 else "FilledDiamond")
            graph.lines[i].symbol.fillColor = color
            graph.lines[i].symbol.strokeColor = color
        else:
            graph.lines[i].strokeDashArray = [5,4]
    drawing.add(graph)
    drawing.add(String(240, 22, "Paired initialization seed", fontSize=11))
    renderSVG.drawToFile(drawing, str(output_dir / "paired_macro_f1.svg"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPORTS)
    parser.add_argument("--replay", action="store_true",
                        help="Analyze a new execution without the historical evaluator failure record")
    options = parser.parse_args()
    # Existing audit checks original CSV bytes, aligned IDs, logits, checkpoint bytes and CSV summaries.
    for name in ("stage5_results.json", "stage5_per_run.csv", "stage5_paired_differences.csv"):
        audit.require((REPORTS / name).is_file(), f"Missing prior result: {name}")
    subprocess.run([ROOT / ".venv-gpu/Scripts/python.exe", "-B", ROOT / "scripts/audit_stage5_results.py"], check=True)
    read = audit.read_json
    base = audit.BASE
    args = read(REPORTS / "resolved_gpu_arguments.json")["runs"]
    expected = {(c,s) for c in "AB" for s in SEEDS}
    assert {(r["condition"], r["seed"]) for r in audit.PLAN["runs"]} == expected
    if not options.replay:
        correction = read(REPORTS / "stage5_evaluation_correction.json")
        before = read(base / "evaluation_correction_preflight.json")
        original = ROOT / "scripts/evaluate_final.py"
        fixed = ROOT / "scripts/evaluate_final_corrected.py"
        assert original.read_text().replace("for r in rows],labels,tokenizer)", "for r in rows],[r['output'] for r in rows],tokenizer)") == fixed.read_text()
        for key, path in [("original_evaluator_sha256", original), ("corrected_evaluator_sha256", fixed),
                          ("selection_lock_sha256", base / "selection_lock.json")]:
            assert before[key] == correction[key] == audit.digest(path)
        assert before["test_prediction_files_before_rerun"] == 0 and before["selections_locked_before_correction"] == 10
        assert "Unexpected Boolean label: 0" in (base / "final_test.stderr.log").read_text()
    with (ROOT / audit.SPEC["data"]["directory"] / "test.csv").open(encoding="utf-8", newline="") as f:
        source = list(csv.DictReader(f))
    labels = [audit.SPEC["data"]["label2id"][row["output"]] for row in source]
    groups = []
    for row in source:
        giver, guesser, suffix = PERSONALITY.fullmatch(row["personality_only"]).groups()
        assert suffix == row["base_text"]
        groups.append("missing" if "None" in (giver, guesser) else "available")
    runs, predictions, curves, matrices = {}, {}, {}, {}
    for item in audit.PLAN["runs"]:
        c,s,rid = item["condition"], item["seed"], item["run_id"]
        folder = base / "runs" / rid
        records = audit.read_jsonl(folder / "final_test_predictions.jsonl")
        predictions[c,s] = [int(r["logit_1"] > r["logit_0"]) for r in records]
        runs[c,s], matrices[rid] = metrics(labels, predictions[c,s])
        m = read(folder / "final_test_metrics.json")
        selected = read(folder / "selected_checkpoint.json")
        state = read(folder / "checkpoint-3990/trainer_state.json")
        assert state["global_step"] == 3990 and state["epoch"] == 15
        assert args[rid]["num_train_epochs"] == 15 and args[rid]["seed"] == s
        assert Path(state["best_model_checkpoint"]).name == m["selected_checkpoint"] == selected["checkpoint"]
        attempts = audit.read_jsonl(folder / "attempts.jsonl")
        if not options.replay:
            assert [a["status"] for a in attempts] == ["running","complete"] and all(a["resume_checkpoint"] is None for a in attempts)
        for name in ("False","True"):
            for key,value in m["per_class"][name].items():
                assert audit.close(value, runs[c,s][f"{name.lower()}_{key}"])
        curves[rid] = [{"epoch": r["epoch"], "validation_macro_f1": r["eval_validation_macro_f1"],
                        "validation_loss": r["eval_validation_loss"]} for r in state["log_history"] if "eval_validation_macro_f1" in r]
        assert len(curves[rid]) == 15
    summaries = {c:{k:{"mean":st.mean(runs[c,s][k] for s in SEEDS), "sd":st.stdev(runs[c,s][k] for s in SEEDS)}
                    for k in runs[c,13]} for c in "AB"}
    def transition(i,s):
        a,b = predictions["A",s][i] == labels[i], predictions["B",s][i] == labels[i]
        return ("both_wrong","B_only","A_only","both_correct")[2*int(a)+int(b)]
    paired, subgroups = {}, {}
    for s in SEEDS:
        paired[s] = {group:dict(Counter(transition(i,s) for i,y in enumerate(labels) if group=="all" or y==group))
                     for group in ("all",0,1)}
    for group in ("available","missing"):
        idx = [i for i,g in enumerate(groups) if g==group]
        y = [labels[i] for i in idx]
        scores = {}
        for s in SEEDS:
            scores[s] = {c:metrics(y,[predictions[c,s][i] for i in idx])[0]["macro_f1"] for c in "AB"}
            scores[s]["delta"] = scores[s]["B"] - scores[s]["A"]
        subgroups[group] = dict(n=len(idx), false=y.count(0), true=y.count(1), scores=scores)
    votes = {c:dict(sorted(Counter(sum(predictions[c,s][i] for s in SEEDS) for i in range(len(labels))).items())) for c in "AB"}
    examples = {}
    for category in ("B_only","A_only","both_wrong"):
        counts = [sum(transition(i,s)==category for s in SEEDS) for i in range(len(labels))]
        i = counts.index(max(counts))
        fields = parse_base(source[i]["base_text"])
        examples[category] = dict(row=i+1,hint=fields["hint"],target=fields["target"],label=labels[i],
                                  A=[predictions["A",s][i] for s in SEEDS], B=[predictions["B",s][i] for s in SEEDS])
    differences = [runs["B",s]["macro_f1"]-runs["A",s]["macro_f1"] for s in SEEDS]
    result = dict(seeds=SEEDS, class_order=["False","True"], matrix_axes=["actual","predicted"],
                  confusion=matrices, condition_summary=summaries, paired=paired,
                  paired_difference=dict(values=differences,mean=st.mean(differences),sd=st.stdev(differences)),
                  always_true=metrics(labels,[1]*len(labels))[0], true_votes_across_seeds=votes,
                  exploratory_groups=subgroups, examples=examples, validation_curves=curves)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    (options.output_dir / "results_analysis.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    with (options.output_dir / "results_by_class.csv").open("w", newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f, fieldnames=["condition","seed",*runs["A",13]])
        writer.writeheader()
        for (c,s), values in runs.items():
            writer.writerow(dict(condition=c,seed=s,**values))
    plot(runs, options.output_dir)


if __name__ == "__main__":
    main()

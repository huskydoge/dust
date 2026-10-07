"""Print the reproduction's result tables as Markdown from the run records in reproduction/results."""
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
POPULATIONS = {256: "256", 1024: "1k", 4096: "4k", 16384: "16k"}
GPUS = {256: 1, 1024: 1, 4096: 4, 16384: 8}
# Means over seeds published with the paper (its Table 1 and the data points of its Figure 3).
PAPER = {
    ("sgd", "1m"): [6.067, 5.934, 5.911, 5.916], ("sgd", "10m"): [5.231, 5.111, 5.072, 5.049],
    ("adam", "1m"): [5.836, 5.594, 5.500, 5.402], ("bp-sgd", "1m"): 5.959, ("bp-sgd", "10m"): 4.989,
    ("bp-adamw", "1m"): 5.361,
}


def result(name):
    path = RESULTS / name / "result.json"
    return json.loads(path.read_text()) if path.exists() else None


def test(name):
    record = result(name)
    return record and record["test_at_best_val"]


def validation(name, step):
    record = result(name)
    if not record:
        return None
    return next((h["val_loss"] for h in record["history"] if h["step"] == step and "val_loss" in h), None)


def elapsed(name):
    lines = (RESULTS / name / "metrics.jsonl").read_text().splitlines()
    return json.loads(lines[-1])["elapsed_seconds"]


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def show(value, digits=3):
    return "–" if value is None else f"{value:.{digits}f}"


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print()


def backprop(optimizer, tokens):
    suffix = "-lr0.002" if optimizer == "adamw" else ""
    return [test(f"bp-{optimizer}-{tokens}{suffix}-s{seed}") for seed in (42, 43, 44)]


print("### Test loss against the paper\n")
rows = []
for tokens in ("1m", "10m"):
    adamw = mean(backprop("adamw", tokens))
    for optimizer, label in (("sgd", "Backprop, SGD (3 seeds)"), ("adamw", "Backprop, AdamW (3 seeds)")):
        ours = mean(backprop(optimizer, tokens))
        rows.append([f"{tokens.upper()} · {label}", show(ours), show(PAPER.get((f"bp-{optimizer}", tokens))), show(ours - adamw, 2)])
    for optimizer, label in (("sgd", "Dust, SGD"), ("adam", "Dust, Adam")):
        for i, population in enumerate(POPULATIONS):
            ours = test(f"dust-{optimizer}-{tokens}-p{population}-s42")
            paper = PAPER.get((optimizer, tokens), [None] * 4)[i]
            rows.append([f"{tokens.upper()} · {label}, population {POPULATIONS[population]}", show(ours), show(paper),
                         show(ours - adamw, 2) if ours is not None else "–"])
table(["Run", "Ours", "Paper", "Above backprop AdamW"], rows)

print("### Seeds at 10M tokens\n")
rows = [["Backprop, AdamW"] + [show(v) for v in backprop("adamw", "10m")] + [show(mean(backprop("adamw", "10m")))]]
seeds = [test(f"dust-adam-10m-p4096-s{seed}") for seed in (42, 43, 44)]
rows.append(["Dust, Adam, population 4k"] + [show(v) for v in seeds] + [show(mean(seeds))])
table(["Method", "Seed 42", "Seed 43", "Seed 44", "Mean of finished"], rows)

print("### Dust under Adam at 10M tokens, tuning at population 1k\n")
SWEEP = [("dust-adam-10m-p1024-s42", "Appendix settings (learning rate 0.0015, β1 0.7)"),
         ("dust-adam-10m-p1024-lr0.00075-b0.7-s42", "Learning rate 0.00075"), ("dust-adam-10m-p1024-lr0.003-b0.7-s42", "Learning rate 0.003"),
         ("dust-adam-10m-p1024-lr0.0015-b0.9-s42", "β1 0.9"), ("dust-adam-10m-p1024-lr0.0015-b0.95-s42", "β1 0.95")]
table(["Setting", "Test loss"], [[label, show(test(name))] for name, label in SWEEP])

print("### Continuing a backprop AdamW checkpoint for 200 steps (validation loss)\n")
CONTINUE = [("bp-keepopt-lr0.002", "Backprop, AdamW state kept, 0.002"), ("bp-fresh-lr0.002", "Backprop, 0.002"),
            ("bp-fresh-lr0.0005", "Backprop, 0.0005"), ("dust-adam-p4096", "Dust 4k, 0.0015"), ("dust-adam-p16384", "Dust 16k, 0.002"),
            ("dust-adam-p4096-lr0.0005", "Dust 4k, 0.0005"), ("dust-adam-p16384-lr0.0005", "Dust 16k, 0.0005")]
rows = []
for suffix, label in CONTINUE:
    row = [label]
    for prefix, start in (("cpt10m", "bp-adamw-10m-ckpt-s42"), ("cpt100m", "bp-adamw-100m-ckpt-s42")):
        begin, end = result(start)["best_val"], validation(f"{prefix}-{suffix}", 200)
        reference = validation(f"{prefix}-bp-fresh-lr0.0005", 200)
        share = f"{100 * (begin - end) / (begin - reference):.0f}%" if end and suffix.startswith("dust") and suffix.endswith("lr0.0005") else "–"
        row += [show(end), share]
    rows.append(row)
starts = [show(result(name)["best_val"]) for name in ("bp-adamw-10m-ckpt-s42", "bp-adamw-100m-ckpt-s42")]
table(["Method and matrix learning rate", f"From 10M-token checkpoint ({starts[0]})", "Share of backprop's drop",
       f"From 100M-token checkpoint ({starts[1]})", "Share of backprop's drop"], rows)

print("### Wall-clock cost, evaluation included\n")
rows = []
for tokens, steps in (("1m", 61), ("10m", 610)):
    base = elapsed(f"bp-adamw-{tokens}-lr0.002-s42") / 3600
    rows.append([f"{tokens.upper()} · Backprop, AdamW", "1", show(base, 3), "1"])
    for optimizer, label in (("sgd", "Dust, SGD"), ("adam", "Dust, Adam")):
        for population in POPULATIONS:
            name = f"dust-{optimizer}-{tokens}-p{population}-s42"
            if test(name) is not None:
                hours = GPUS[population] * elapsed(name) / 3600
                rows.append([f"{tokens.upper()} · {label}, population {POPULATIONS[population]}", str(GPUS[population]), show(hours, 2),
                             f"{hours / base:,.0f}"])
table(["Run", "GPUs", "GPU-hours", "Relative to backprop"], rows)

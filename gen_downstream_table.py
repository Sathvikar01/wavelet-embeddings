"""Generate paper/downstream_table.tex from results/downstream CSV."""

import csv
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    src = os.path.join(ROOT, "results", "downstream",
                        "downstream_pruning.csv")
    rows = list(csv.DictReader(open(src, encoding="utf-8")))

    # value lookup: (task, model, predictor) -> {ratio: value}
    val = defaultdict(dict)
    base = {}
    for r in rows:
        key = (r["task"], r["model"], r["predictor"])
        if r["predictor"] == "none":
            base[key] = float(r["value"])
        else:
            val[key][round(float(r["ratio"]), 2)] = float(r["value"])

    NAME = {
        ("sst2", "textattack/bert-base-uncased-SST-2"): "BERT SST-2 acc.",
        ("sst2", "distilbert-base-uncased-finetuned-sst-2-english"):
            "DistilBERT SST-2 acc.",
        ("wikitext2_ppl", "gpt2"): "GPT-2 WikiText-2 PPL",
    }
    PREDS = ["wavelet", "attention_entropy_true", "magnitude",
              "attention_weight", "random", "ridge_wavelet_only",
              "ridge_attn_only", "ridge_combined"]
    DISP = {"wavelet": "Wavelet composite",
             "attention_entropy_true": "Attn.\\ entropy (true)",
             "magnitude": "Frobenius mass $\\|A\\|_F$",
             "attention_weight": "Max-column mass",
             "random": "Random",
             "ridge_wavelet_only": "Ridge: wavelet-only (LOMO)",
             "ridge_attn_only": "Ridge: attn.-only (LOMO)",
             "ridge_combined": "Ridge: combined (LOMO)"}

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Downstream-task validation: damage after pruning the",
        "bottom-$p$\\% of heads ranked by each predictor. Accuracy drops",
        "(points) and perplexity ratios ($\\times$ clean) computed against",
        "each model's unpruned score; lower is better. No predictor",
        "dominates: distributional entropy transfers best to SST-2 but",
        "collapses on GPT-2, where the wavelet composite excels ---",
        "task-level validation is indispensable.}",
        "\\label{tab:downstream}",
        "\\begin{tabular}{lrrrrrrrrr}",
        "\\toprule",
        " & \\multicolumn{3}{c}{BERT SST-2 $\\Delta$acc.} & "
        "\\multicolumn{3}{c}{DistilBERT SST-2 $\\Delta$acc.} & "
        "\\multicolumn{3}{c}{GPT-2 PPL ratio} \\\\",
        "Predictor & 10\\% & 20\\% & 30\\% & 10\\% & 20\\% & 30\\% "
        "& 10\\% & 20\\% & 30\\% \\\\",
        "\\midrule",
    ]

    for pred in PREDS:
        cells = []
        for task_key in (("sst2", "textattack/bert-base-uncased-SST-2"),
                          ("sst2",
                           "distilbert-base-uncased-finetuned-sst-2-english"),
                          ("wikitext2_ppl", "gpt2")):
            task, model = task_key
            b = base.get((task, model, "none"))
            d = val.get((task, model, pred), {})
            for ratio in (0.1, 0.2, 0.3):
                v = d.get(ratio)
                if v is None or b is None or pred not in DISP:
                    cells.append("--")
                    continue
                if task == "sst2":
                    drop = (b - v) * 100.0
                    cells.append(f"{drop:+.1f}")
                else:
                    cells.append(f"{v / b:.2f}")
        lines.append(f"{DISP[pred]} & " + " & ".join(cells) + " \\\\")

    # clean reference row
    ref = []
    for tk in (("sst2", "textattack/bert-base-uncased-SST-2"),
                ("sst2", "distilbert-base-uncased-finetuned-sst-2-english"),
                ("wikitext2_ppl", "gpt2")):
        b = base.get((tk[0], tk[1], "none"))
        if tk[0] == "sst2":
            ref += [f"{b * 100:.1f}", "--", "--"]
        else:
            ref += [f"{b:.1f}", "--", "--"]
    lines.append("\\midrule")
    lines.append("Clean model & " + " & ".join(ref) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]

    out_path = os.path.join(ROOT, "paper", "downstream_table.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out_path)
    print("\n".join(lines[:14]))


if __name__ == "__main__":
    main()

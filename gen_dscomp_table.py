"""Generate paper/dscomp_table.tex from downstream_compression.csv."""

import csv
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    rows = list(csv.DictReader(open(
        os.path.join(ROOT, "results", "downstream",
                      "downstream_compression.csv"), encoding="utf-8")))
    val = {(r["model"], r["method"], str(r["param"])): float(r["value"])
            for r in rows}

    MODELS = [
        ("textattack/bert-base-uncased-SST-2", "BERT SST-2", "accuracy"),
        ("distilbert-base-uncased-finetuned-sst-2-english",
         "DistilBERT SST-2", "accuracy"),
        ("gpt2", "GPT-2 PPL", "ppl"),
    ]
    METHODS = [("wav_int8", "Wav.", ("170", "341")),
                ("dct_int8", "DCT", ("170", "341")),
                ("topk_int8", "TopK", ("170", "341")),
                ("pca", "PCA", ("128", "256"))]

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Downstream validation of embedding-table",
        "compression: the word-embedding matrix is replaced by its",
        "lossy reconstruction (int8 payloads; PCA dense) and the task",
        "metric re-measured. Accuracy drops (points) and perplexity",
        "ratios ($\\times$ clean); lower is better. On both encoder",
        "classifiers every tested int8 sparsification recipe --- wavelet,",
        "DCT, and raw top-k alike --- transfers with low degradation, and",
        "is near-lossless at the 33\\% budget --- while PCA",
        "truncation does not. On tied-embedding GPT-2 the transform-domain",
        "tables fail catastrophically at both budgets while raw-coordinate",
        "TopK stays robust at 33\\% ($1.2\\times$): where a perturbation",
        "lands can matter more than how close the reconstruction is.}",
        "\\label{tab:dscomp}",
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{Wav.+int8} & "
        "\\multicolumn{2}{c}{DCT+int8} & "
        "\\multicolumn{2}{c}{TopK+int8} & "
        "\\multicolumn{2}{c}{PCA} \\\\",
        "Model & 17\\% & 33\\% & 17\\% & 33\\% "
        "& 17\\% & 33\\% & 17\\% & 33\\% \\\\",
        "\\midrule",
    ]
    for hf, disp, metric in MODELS:
        base = val.get((hf, "none", ""), float("nan"))
        cells = []
        for method, _, params in METHODS:
            for param in params:
                v = val.get((hf, method, param))
                if v is None:
                    cells.append("--")
                elif metric == "accuracy":
                    cells.append(f"{(base - v) * 100:+.1f}")
                else:
                    ratio = v / base if base else float("inf")
                    cells.append("$>$1000$\\times$" if ratio > 1000
                                  else f"{ratio:.1f}$\\times$")
        lines.append(disp + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]

    out_path = os.path.join(ROOT, "paper", "dscomp_table.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()

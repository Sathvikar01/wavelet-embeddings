p = 'experiments/downstream_ridge.py'
s = open(p, encoding='utf-8').read()

old = "    # ---------------- classifiers ----------------"
new = ("    ridge_variants = [v for v in RIDGE_VARIANTS\n"
       "                       if any((m, v[0], str(r)) not in have\n"
       "                               for m, _, _ in MODELS\n"
       "                               for r in (0.1, 0.2, 0.3))]\n"
       "    if not ridge_variants:\n"
       "        print('all ridge downstream rows already recorded; '\n"
       "               'nothing to do', flush=True)\n"
       "        return\n"
       "    # ---------------- classifiers ----------------")
assert old in s
s = s.replace(old, new, 1)

# classifier loop: fit per-variant and record under variant name
old2 = """        heads_all, rows_metrics, attn = [], [], []
        base_m = getattr(model, key)"""
new2 = """        heads_all, rows_metrics, attn = [], [], []
        base_m = getattr(model, key)
        fits = {name: fit_lomo(key, group)
                 for name, group in ridge_variants}"""
assert old2 in s
s = s.replace(old2, new2, 1)

old3 = """        Xh, _ = build_feature_matrix(rows_metrics,
                                      {"head_attention": attn},
                                      group="combined")
        w, b, _ = fit_lomo(key)

        def accuracy(ablate=None):
            ctx = _Null() if ablate is None else SetAblator(key, model,
                                                             ablate)
            correct = 0
            with torch.no_grad(), ctx:
                for i in range(0, len(texts), 32):
                    enc = tok(texts[i:i + 32], padding=True,
                               truncation=True, max_length=128,
                               return_tensors="pt")
                    logits = model(**enc).logits
                    correct += int((logits.argmax(-1).cpu().numpy()
                                     == labels[i:i + 32]).sum())
            return correct / len(texts)

        scores = Xh @ w + b                     # predicted damage
        clean = accuracy()
        print(f"[ridge-ds] {hf_name} clean={clean:.4f}", flush=True)
        for ratio in (0.1, 0.2, 0.3):
            k = max(1, int(round(ratio * len(scores))))
            order = np.argsort(scores)[:k]      # lowest predicted damage
            ab = [heads_all[i] for i in order]
            acc = accuracy(ab)
            print(f"  ridge_combined p={ratio} acc={acc:.4f}", flush=True)
            if record(hf_name, "ridge_combined", ratio, "accuracy", acc):
                have.add((hf_name, "ridge_combined", str(ratio)))
        del model"""
new3 = """        Xh, _ = build_feature_matrix(rows_metrics,
                                      {"head_attention": attn},
                                      group="combined")

        def accuracy(ablate=None):
            ctx = _Null() if ablate is None else SetAblator(key, model,
                                                             ablate)
            correct = 0
            with torch.no_grad(), ctx:
                for i in range(0, len(texts), 32):
                    enc = tok(texts[i:i + 32], padding=True,
                               truncation=True, max_length=128,
                               return_tensors="pt")
                    logits = model(**enc).logits
                    correct += int((logits.argmax(-1).cpu().numpy()
                                     == labels[i:i + 32]).sum())
            return correct / len(texts)

        clean = accuracy()
        print(f"[ridge-ds] {hf_name} clean={clean:.4f}", flush=True)
        for name, _group in ridge_variants:
            w, b, _ = fits[name]
            scores = Xh @ w + b             # predicted damage
            for ratio in (0.1, 0.2, 0.3):
                k = max(1, int(round(ratio * len(scores))))
                order = np.argsort(scores)[:k]   # lowest predicted damage
                ab = [heads_all[i] for i in order]
                acc = accuracy(ab)
                print(f"  {name} p={ratio} acc={acc:.4f}", flush=True)
                if record(hf_name, name, ratio, "accuracy", acc):
                    have.add((hf_name, name, str(ratio)))
        del model"""
assert old3 in s
s = s.replace(old3, new3, 1)

# GPT-2 section: same treatment
old4 = """    Xh, _ = build_feature_matrix(rows_metrics, {"head_attention": attn},
                                  group="combined")
    w, b, _ = fit_lomo("gpt2")"""
new4 = """    Xh, _ = build_feature_matrix(rows_metrics, {"head_attention": attn},
                                  group="combined")
    fits = {name: fit_lomo("gpt2", group)
             for name, group in ridge_variants}"""
assert old4 in s
s = s.replace(old4, new4, 1)

old5 = """    scores = Xh @ w + b
    base_ppl = perplexity()
    print(f"[ridge-ds] gpt2 clean={base_ppl:.2f}", flush=True)
    for ratio in (0.1, 0.2, 0.3):
        k = max(1, int(round(ratio * len(scores))))
        order = np.argsort(scores)[:k]
        ab = [heads_all[i] for i in order]
        ppl = perplexity(ab)
        print(f"  ridge_combined p={ratio} ppl={ppl:.2f}", flush=True)
        if record(hf_name, "ridge_combined", ratio, "ppl", ppl):
            have.add((hf_name, "ridge_combined", str(ratio)))"""
new5 = """    base_ppl = perplexity()
    print(f"[ridge-ds] gpt2 clean={base_ppl:.2f}", flush=True)
    for name, _group in ridge_variants:
        w, b, _ = fits[name]
        scores = Xh @ w + b
        for ratio in (0.1, 0.2, 0.3):
            k = max(1, int(round(ratio * len(scores))))
            order = np.argsort(scores)[:k]
            ab = [heads_all[i] for i in order]
            ppl = perplexity(ab)
            print(f"  {name} p={ratio} ppl={ppl:.2f}", flush=True)
            if record(hf_name, name, ratio, "ppl", ppl):
                have.add((hf_name, name, str(ratio)))"""
assert old5 in s
s = s.replace(old5, new5, 1)

open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched OK')

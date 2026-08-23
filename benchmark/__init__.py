"""Reproducible head-pruning benchmark package.

Adds (over the existing Phase-4 predictors):

* more backbones        - RoBERTa, DeBERTa, TinyLlama alongside BERT /
  DistilBERT / GPT-2.
* more data             - WikiText-2, Penn Treebank and GLUE-subset loaders
  that yield thousands of sentences.
* published baselines   - Michel et al. (HIT), Voita et al. (HIS) and
  Bhasharas et al. (behavioural-similarity) head-importance criteria.
* statistics            - paired Wilcoxon / t-tests, bootstrap CIs and
  effect sizes (Cohen's d / Cliff's delta) computed across inputs and
  random seeds.

Run via ``python main.py benchmark-run ...`` or on Modal with
``modal run benchmark/modal_app.py``.
"""

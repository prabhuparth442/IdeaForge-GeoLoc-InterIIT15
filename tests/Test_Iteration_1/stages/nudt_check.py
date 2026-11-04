"""Block: full-map check of single maps and fusion models on a set of photos, saved under a result key."""
import numpy as np

from stages.helpers import log_rows
from train.checks import full_eval


def evaluate_on_photos(run, test_set, store, families, names, fused_models, result_key, title):
    results = full_eval.evaluate_photos(test_set, store, families, np.arange(len(test_set.items)), names, fused_models, log=run.log)
    rows = full_eval.summarise(results)
    run.set_result(result_key, rows)
    log_rows(run, title, rows)

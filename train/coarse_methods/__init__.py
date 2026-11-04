"""Coarse methods as lego pieces: each is a function  score_map(inputs) -> (rows, cols) tensor  (see inputs.py).
A notebook or stage builds the list it wants with build_methods() and runs every entry on the same inputs, then pick_peaks on each result.
Adding a method = adding a small file with a make_*_method() function and one line here or in the caller."""
from train.coarse_methods.fused import make_fused_method
from train.coarse_methods.hint import make_hint_method
from train.coarse_methods.single_map import make_single_map_method


def build_methods(single_names, fused_models=None, hints=()):
    """single_names: map names used as they are. fused_models: label -> (model, subset). hints: (base, hint, beta) triples.
    Returns an ordered dict label -> method function."""
    methods = {}
    for name in single_names:
        methods[name] = make_single_map_method(name)
    for label, (model, subset) in (fused_models or {}).items():
        methods[label] = make_fused_method(model, subset)
    for base, hint, beta in hints:
        methods["%s + %.1f * %s" % (base, beta, hint)] = make_hint_method(base, hint, beta)
    return methods


def run_methods(methods, inputs):
    """label -> score map, every method on the same inputs."""
    return {label: method(inputs) for label, method in methods.items()}

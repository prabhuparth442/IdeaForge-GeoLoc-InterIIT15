"""Runs the chosen stages in order. A finished stage is skipped, a crash in one stage is logged and the next stage still runs."""
import time
import traceback


def run_stages(run, stage_functions, wanted, redo, needs_setup, always_rerun=("check", "setup", "report")):
    """stage_functions: ordered dict stage name -> function(run). 'setup' is added by itself when a stage in needs_setup is wanted."""
    order = []
    for name in stage_functions:
        if name in wanted or (name == "setup" and needs_setup & set(wanted)):
            order.append(name)
    for name in order:
        if name == "setup" and "store" in run.shared:
            continue
        finished = name in run.state["done"] and name not in redo
        if finished and name not in always_rerun:
            run.log("-- stage %s: already done, skipping" % name)
            continue
        run.log("== stage %s" % name)
        started = time.time()
        try:
            stage_functions[name](run)
            if name not in run.state["done"]:
                run.state["done"].append(name)
            run.state["failed"].pop(name, None)
        except Exception as error:
            run.log("!! stage %s FAILED: %s: %s" % (name, type(error).__name__, error))
            run.log(traceback.format_exc())
            run.state["failed"][name] = "%s: %s" % (type(error).__name__, str(error)[:300])
        run.save_state()
        run.log("== stage %s took %.1f min" % (name, (time.time() - started) / 60))
    run.log("finished. failed stages:", list(run.state["failed"]) or "none")

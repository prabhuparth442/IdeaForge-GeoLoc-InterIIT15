"""Stage report (synthetic run): report.md with the real-photo results side by side."""
from train.checks.synthetic_report import write_report


def run_stage(run):
    write_report(run.out, run.state, run.args)
    run.log("report written:", run.out / "report.md")

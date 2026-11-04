"""Stage report: write report.md from the saved results."""
from train.checks.overnight_report import write_report


def run_stage(run):
    write_report(run.out, run.state, run.args)
    run.log("report written:", run.out / "report.md")

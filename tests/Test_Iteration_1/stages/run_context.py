"""What every stage gets: the run settings, a log, the saved state (which stages finished, their results) and a bag of shared objects."""
import gc
import json
import time
from pathlib import Path

import torch


class RunContext:
    def __init__(self, args, out_dir, hours=10.0):
        self.args = args
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "models").mkdir(exist_ok=True)
        self.started = time.time()
        self.deadline = self.started + hours * 3600
        self.state_path = self.out / "state.json"
        if self.state_path.exists():
            self.state = json.load(open(self.state_path))
        else:
            self.state = {"done": [], "failed": {}, "results": {}}
        self.shared = {}                      # things built by a stage for later stages: store, photos, families ...
        self.log_file = open(self.out / "run.log", "a", buffering=1)

    def log(self, *parts):
        line = "[%s +%5.1f min] %s" % (time.strftime("%H:%M:%S"), (time.time() - self.started) / 60, " ".join(str(p) for p in parts))
        print(line, flush=True)
        self.log_file.write(line + "\n")

    def save_state(self):
        json.dump(self.state, open(self.state_path, "w"), indent=1, default=float)

    def set_result(self, key, value):
        self.state["results"][key] = value
        self.save_state()

    def free_memory(self):
        self.shared.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

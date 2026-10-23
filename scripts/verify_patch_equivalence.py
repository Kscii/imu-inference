"""Compare the installed frozen baseline with this patch without changing either model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


REPLAY = r'''
import json,sys
from pathlib import Path
import numpy as np
from imu_inference import InferenceEngine, PipelineConfig, __version__
n=4500
t=np.arange(n)/25
x=np.column_stack((.3*np.sin(2*np.pi*1.8*t), .15*np.cos(2*np.pi*1.8*t),
                   9.80665+1.7*np.sin(2*np.pi*1.8*t))).astype(np.float32)
x[:1000]=[0,9.80665,0]
x[3000:]=[9.80665,0,0]
gyro=np.column_stack((.02*np.cos(t),.01*np.sin(t),.03*np.cos(t/2))).astype(np.float32)
valid=np.ones(n,bool); valid[1800:1825]=False
excluded=np.zeros(n,bool); excluded[2600:2675]=True
backend=InferenceEngine(threads=1).backend
results={}
for mode,sign in [('incremental',False),('whole_record',False),('whole_record',True)]:
 for steps in (False,True):
  config=PipelineConfig(mode=mode,sign_adaptation=sign,steps_enabled=steps)
  engine=InferenceEngine(config,threads=1,backend=backend)
  result=engine.predict(x,angular_velocity=gyro,valid=valid,excluded=excluded)
  results[f'{mode}.sign_{int(sign)}.steps_{int(steps)}']=result
Path(sys.argv[1]).write_text(json.dumps(dict(version=__version__,results=results),sort_keys=True))
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-wheel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    baseline_hash = digest(args.baseline_wheel.read_bytes())
    baseline = json.loads((root / "provenance/delivery-build.json").read_text())
    if baseline_hash != baseline["wheel_sha256"]:
        raise ValueError("Baseline wheel does not match the frozen 1.0.0 evidence")
    current = root / "src/imu_inference"
    checked = {}
    with zipfile.ZipFile(args.baseline_wheel) as z:
        old_names = {n.removeprefix("imu_inference/") for n in z.namelist()
                     if n.startswith("imu_inference/") and not n.endswith("/")}
        for name in sorted(old_names):
            old = z.read("imu_inference/" + name)
            new = (current / name).read_bytes()
            if name == "__init__.py":
                if old.replace(b'__version__ = "1.0.0"', b'__version__ = "1.0.1"') != new:
                    raise ValueError("Unexpected Python API change")
            elif old != new:
                raise ValueError(f"Frozen model or algorithm content changed: {name}")
            checked[name] = digest(new)
        extras = {p.relative_to(current).as_posix() for p in current.rglob("*.py")} - old_names
        if extras:
            raise ValueError(f"Unexpected runtime Python files: {extras}")
        with tempfile.TemporaryDirectory(prefix="imu-patch-regression-") as tmp:
            tmp = Path(tmp)
            z.extractall(tmp / "baseline")
            outputs = []
            for name, path in (("baseline", tmp / "baseline"), ("candidate", root / "src")):
                out = tmp / (name + ".json")
                env = dict(os.environ, PYTHONPATH=str(path), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                subprocess.run([sys.executable, "-c", REPLAY, str(out)], cwd=tmp, env=env, check=True)
                outputs.append(json.loads(out.read_text()))
    old, new = outputs
    if (old["version"], new["version"]) != ("1.0.0", "1.0.1"):
        raise ValueError("Replay imported the wrong versions")
    comparisons = {}
    for case in old["results"]:
        a, b = old["results"][case], new["results"][case]
        for field in ("library_version", "code_sha256"):
            a["provenance"].pop(field)
            b["provenance"].pop(field)
        if a != b:
            raise ValueError(f"Prediction or non-version metadata changed: {case}")
        comparisons[case] = dict(equal=True, output_sha256=digest(json.dumps(b, sort_keys=True).encode()))
    report = dict(schema="imu.patch-equivalence.v1", passed=True,
                  baseline_version=old["version"], candidate_version=new["version"],
                  baseline_wheel_sha256=baseline_hash, checked_files=len(checked),
                  candidate_inventory_sha256=digest(json.dumps(checked, sort_keys=True).encode()),
                  permitted_source_change="__init__.py version literal only",
                  excluded_output_fields=["provenance.library_version", "provenance.code_sha256"],
                  input="180-second synthetic mixed signal with explicit invalid and excluded masks",
                  accuracy_evidence=False, cases=comparisons)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=True, cases=len(comparisons), output=str(args.output))))


if __name__ == "__main__":
    main()

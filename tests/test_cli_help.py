import os
import sys
import subprocess

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(__file__))

def run_cmd(args):
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT
    res = subprocess.run([PY, "-m", "orilinx"] + args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res


def test_top_help():
    r = run_cmd(["-h"])
    assert r.returncode == 0
    # Top-level help should show the predict arguments
    assert "--fasta_path" in r.stdout or "Genome-wide origin scores" in r.stdout


def test_predict_help_alias():
    # 'orilinx -h' provides the predict help; ensure detailed predict help is present
    r = run_cmd(["-h"])
    assert r.returncode == 0
    assert "--output_dir" in r.stdout


def test_fetch_models_help():
    r = run_cmd(["fetch_models", "-h"])
    assert r.returncode == 0
    assert "Download model weights" in r.stdout
    assert "--force" in r.stdout

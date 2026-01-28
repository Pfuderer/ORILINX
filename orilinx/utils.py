import os


def find_default_model_path():
    """Resolve model checkpoint path.

    Priority order:
    1. If `ORILINX_MODEL` env var is set, use it (must point to an existing .pt).
    2. Otherwise, search upward from CWD and from the package tree for a `models/` 
       directory and return the newest .pt file.

    Returns the absolute path to the .pt or None if no candidate found.
    """
    # 1) Env override
    env_path = os.environ.get("ORILINX_MODEL")
    if env_path:
        if os.path.isfile(env_path) and env_path.endswith(".pt"):
            return os.path.abspath(env_path)
        raise RuntimeError(
            f"ORILINX_MODEL is set to '{env_path}', but that file does not exist or is not a .pt"
        )

    # 2) Build candidate roots: cwd parents + package dir parents (deduplicated)
    cur = os.getcwd()
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    
    # Walk up from cwd
    node = cur
    while True:
        if node not in candidates:
            candidates.append(node)
        parent = os.path.dirname(node)
        if parent == node:
            break
        node = parent
    
    # Walk up from package dir
    node = pkg_dir
    while True:
        if node not in candidates:
            candidates.append(node)
        parent = os.path.dirname(node)
        if parent == node:
            break
        node = parent

    for root in candidates:
        models_dir = os.path.join(root, "models")
        if os.path.isdir(models_dir):
            pts = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith(".pt")]
            if pts:
                pts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return os.path.abspath(pts[0])
    
    return None

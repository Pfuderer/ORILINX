import os
import argparse

from .model_architecture_optim import DnaBertOriginModel  # relative import inside package

# --- sliding windows dataset (iterable; no heavy imports at module import time) ---
import torch
from torch.utils.data import IterableDataset

class _SlidingWindows(IterableDataset):
    def __init__(self, fasta_path, chroms, window, stride, max_N_frac, ranges=None):
        self.fasta_path = fasta_path
        self.chroms = chroms
        self.window = int(window)
        self.stride = int(stride)
        self.max_N_frac = float(max_N_frac)
        # ranges is a dict mapping chrom name to (start, end) tuple, or None for full sequence
        self.ranges = ranges or {}

    def __iter__(self):
        # Worker-aware iterable: distribute candidate windows across workers by index.
        import pysam
        from torch.utils.data import get_worker_info

        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        fa = pysam.FastaFile(self.fasta_path)
        try:
            idx = 0
            for chrom in self.chroms:
                if chrom not in fa.references:
                    continue
                clen = fa.get_reference_length(chrom)
                # Determine the range to process for this chrom
                if chrom in self.ranges:
                    range_start, range_end = self.ranges[chrom]
                    range_start = max(0, range_start)
                    range_end = min(clen, range_end)
                else:
                    range_start, range_end = 0, clen
                # Calculate last valid window start within the range
                last = range_end - self.window
                if last < range_start:
                    continue
                for start in range(range_start, last + 1, self.stride):
                    # Round-robin assignment by candidate index ensures workers share load
                    if (idx % num_workers) != worker_id:
                        idx += 1
                        continue
                    end = start + self.window
                    seq = fa.fetch(chrom, start, end).upper()
                    idx += 1
                    if seq.count("N") / self.window <= self.max_N_frac:
                        yield {"chrom": chrom, "start": start, "end": end, "seq": seq}
        finally:
            fa.close()

    def __len__(self):
        """Estimated number of windows (ignores N-filtering).

        This allows DataLoader to use the default sampler with multiple
        workers (which calls range(len(dataset))). The estimate is computed
        from reference lengths, window and stride and therefore is stable
        and inexpensive.
        """
        import pysam
        fa = pysam.FastaFile(self.fasta_path)
        try:
            total = 0
            for chrom in self.chroms:
                if chrom not in fa.references:
                    continue
                clen = fa.get_reference_length(chrom)
                # Determine the range to process for this chrom
                if chrom in self.ranges:
                    range_start, range_end = self.ranges[chrom]
                    range_start = max(0, range_start)
                    range_end = min(clen, range_end)
                else:
                    range_start, range_end = 0, clen
                last = range_end - self.window
                if last < range_start:
                    continue
                num_windows = ((last - range_start) // self.stride) + 1
                total += num_windows
            return total
        finally:
            fa.close()


def _resolve_chroms_from_fasta(fasta_path: str, arg: str):
    """Parse sequence names with optional ranges (chr or chr:start-end format).
    
    Returns a tuple of (chroms_list, ranges_dict) where:
    - chroms_list: list of sequence names to process
    - ranges_dict: dict mapping chrom -> (start, end) for ranges, or empty dict if no ranges
    """
    import pysam
    import warnings
    fa = pysam.FastaFile(fasta_path)
    refs = list(fa.references)
    ref_lengths = {name: fa.get_reference_length(name) for name in refs}
    fa.close()
    
    ranges_dict = {}
    
    if arg and arg.lower() != "all":
        chroms = []
        for spec in arg.split(","):
            spec = spec.strip()
            if not spec:
                continue
            # Parse format: chr or chr:start-end
            if ":" in spec:
                chrom, range_part = spec.split(":", 1)
                chrom = chrom.strip()
                if chrom not in refs:
                    raise RuntimeError(
                        f"Sequence '{chrom}' not found in FASTA file. Available sequences: {', '.join(refs)}"
                    )
                if "-" in range_part:
                    try:
                        start, end = range_part.split("-", 1)
                        start, end = int(start.strip()), int(end.strip())
                        range_length = end - start
                        if range_length < 2000:
                            raise RuntimeError(
                                f"Range {chrom}:{start}-{end} is {range_length} bases long, but the window "
                                f"length is 2000 bases. All specified ranges must be at least 2000 bases."
                            )
                        # Warn if range exceeds sequence length
                        chrom_len = ref_lengths[chrom]
                        if end > chrom_len:
                            warnings.warn(
                                f"Range {chrom}:{start}-{end} exceeds sequence length ({chrom_len} bp). "
                                f"Will only process up to position {chrom_len}."
                            )
                        ranges_dict[chrom] = (start, end)
                        if chrom not in chroms:
                            chroms.append(chrom)
                    except ValueError as e:
                        if "invalid literal" in str(e):
                            raise RuntimeError(
                                f"Invalid range format in '{spec}'. Expected 'chr:start-end' (e.g., 'chr1:1000-5000')."
                            )
                        raise
            else:
                if spec not in refs:
                    raise RuntimeError(
                        f"Sequence '{spec}' not found in FASTA file. Available sequences: {', '.join(refs)}"
                    )
                chroms.append(spec)
        return chroms, ranges_dict
    
    # Default: primary chromosomes (no ranges)
    primary = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    if "chrY" in refs:
        primary.append("chrY")
    return [c for c in primary if c in refs], ranges_dict


def _write_csv_windows(df, path):
    """Write window data to CSV with columns: chromosome, start, end, probability, logit"""
    df_out = df[["chrom", "start", "end", "prob", "logit"]].copy()
    df_out.to_csv(path, sep=",", header=True, index=False)


def _write_bedgraph_center(df, path, value="logit", stride=1000, region_start=None, region_end=None):
    with open(path, "w") as fh:
        if stride < 2000:
            # For overlapping windows, output non-overlapping intervals
            # Each interval is centered on the window center with width equal to stride
            prev_end = None
            for idx, (_, r) in enumerate(df.iterrows()):
                c = r["chrom"]; v = float(r[value])
                center = int(r["center"])
                half_stride = stride / 2.0
                interval_start = int(center - half_stride)
                interval_end = int(center + half_stride)
                
                # Adjust start if it would overlap with previous interval
                if prev_end is not None and interval_start <= prev_end:
                    interval_start = prev_end + 1
                
                # Adjust first interval to region start
                if idx == 0 and region_start is not None:
                    interval_start = region_start
                
                # Adjust last interval to region end
                if idx == len(df) - 1 and region_end is not None:
                    interval_end = region_end
                
                fh.write(f"{c}\t{interval_start}\t{interval_end}\t{v:.6f}\n")
                prev_end = interval_end
        else:
            # Use full window coordinates when stride >= window length
            for _, r in df.iterrows():
                c = r["chrom"]; v = float(r[value])
                start = int(r["start"]); end = int(r["end"])
                fh.write(f"{c}\t{start}\t{end}\t{v:.6f}\n")


# --- hard-disable ALL fast/unpadded attention flags (module-wise) ---
def _disable_unpad_and_flash_everywhere(model):
    # Try on top-level config
    cfg = getattr(getattr(model, "dnabert", model), "config", None)
    if cfg is not None:
        for nm in ("use_flash_attn", "flash_attn", "use_memory_efficient_attention", "unpad"):
            if hasattr(cfg, nm):
                try:
                    setattr(cfg, nm, False)
                except Exception:
                    pass
        # Key fix: set attention_probs_dropout_prob to non-zero to force PyTorch path
        # BertUnpadSelfAttention checks: if self.p_dropout or flash_attn_qkvpacked_func is None
        # We set p_dropout > 0 to always take the PyTorch path
        if hasattr(cfg, "attention_probs_dropout_prob"):
            try:
                cfg.attention_probs_dropout_prob = 0.01  # Small nonzero value
            except Exception:
                pass
    # Try on all submodules (DNABERT variants differ in attribute names/placement)
    for m in model.modules():
        for nm in ("use_flash_attn", "flash_attn", "use_memory_efficient_attention", "unpad"):
            if hasattr(m, nm):
                try:
                    setattr(m, nm, False)
                except Exception:
                    pass
        # Key fix: also directly set p_dropout on BertUnpadSelfAttention modules
        if hasattr(m, "p_dropout"):
            try:
                m.p_dropout = 0.01
            except Exception:
                pass
        # Set attention_probs_dropout_prob on config objects in submodules too
        if hasattr(m, "config") and hasattr(m.config, "attention_probs_dropout_prob"):
            try:
                m.config.attention_probs_dropout_prob = 0.01
            except Exception:
                pass


def _find_default_model_path():
    """Resolve model checkpoint path.

    Priority order:
    1. If `ORILINX_MODEL` env var is set, use it (must point to an existing .pt).
    2. Otherwise, search upward from CWD and from the package tree for a `models/` directory and return the newest .pt file.

    Returns the absolute path to the .pt or None if no candidate found.
    """
    # 1) Env override
    env_path = os.environ.get("ORILINX_MODEL")
    if env_path:
        if os.path.isfile(env_path) and env_path.endswith(".pt"):
            return os.path.abspath(env_path)
        raise RuntimeError(f"ORILINX_MODEL is set to '{env_path}', but that file does not exist or is not a .pt")

    # 2) Build candidate roots: cwd parents + package dir parents (deduplicated)
    cur = os.getcwd()
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    for p in [cur] + [os.path.dirname(cur)]:
        pass
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


def main(argv=None):
    # Local imports to avoid heavy dependencies on plain `import orilinx`
    import numpy as np
    import pandas as pd
    import pysam
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from torch.amp import autocast

    p = argparse.ArgumentParser(description="Genome-wide origin scores with ORILINX.")
    p.add_argument("--fasta_path", required=True, help="Path to the reference FASTA file; an index (.fai) must be present.")
    p.add_argument("--output_dir", required=True, help="Directory where per-sequence output bedgraphs will be written.")
    p.add_argument("--stride", type=int, default=1000, help="Stride in base pairs (bp) between consecutive windows.")
    p.add_argument("--max_N_frac", type=float, default=0.05, help="Maximum fraction of 'N' bases allowed in a window; windows exceeding this are skipped.")
    p.add_argument("--batch_size", type=int, default=64, help="Number of windows per batch; increase for throughput if memory allows.")
    p.add_argument("--num_workers", type=int, default=8, help="Number of worker processes used by DataLoader for data loading (0 runs in main process).")
    p.add_argument("--sequence_names", type=str, default="all", help='Comma-separated list of sequence names to process; supports ranges (e.g., "chr1,chr2:2000-6000"); use "all" for all primary sequences.')
    p.add_argument("--score", choices=["logit","prob"], default="prob", help="Output score type: 'logit' (raw model logits) or 'prob' (sigmoid probability).")
    p.add_argument("--output_csv", action="store_true", help="Also output results as CSV files with columns: chromosome, start, end, probability, logit.")
    p.add_argument("--disable_flash", action="store_true", help="Force non-flash (padded) attention mode; safer for long sequences.")
    p.add_argument("--no-progress", dest="no_progress", action="store_true", help="Disable progress bars (useful for logging or non-interactive runs).")
    p.add_argument("--verbose", action="store_true", help="Enable verbose output (prints DNABERT path, model checkpoint, device and runtime settings).")
    args = p.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve DNABERT local path (env var or models/ discovery). CLI override removed; discovery is mandatory.
    from .model_architecture_optim import _find_dnabert_local_path
    resolved_dnabert = _find_dnabert_local_path()

    if resolved_dnabert is None:
        raise RuntimeError(
            "DNABERT not found: set ORILINX_DNABERT_PATH to a valid local DNABERT folder or place DNABERT under a 'models/' directory (searched upward from CWD)."
        )

    # --- tokenizer identical to dataset/eval path ---
    tokenizer = AutoTokenizer.from_pretrained(resolved_dnabert, use_fast=True)

    # --- model identical to evaluation path ---
    model = DnaBertOriginModel(model_name=resolved_dnabert, enable_grad_checkpointing=False)

    # Enforce using a checkpoint from a local `models/` directory (searches upward for a models/ directory)
    model_path = _find_default_model_path()
    if model_path is None:
        raise RuntimeError(
            "No model checkpoint found in any 'models/' directory. Please place your .pt checkpoint in a 'models/' folder (searched upward from CWD)."
        )

    # Attempt to load checkpoint and be helpful on common failure modes (git-lfs pointers, wrong format)
    try:
        ckpt = torch.load(model_path, map_location="cpu")  # keep on CPU for safety
    except Exception as e:
        # Detect a likely git-lfs pointer (text file starting with the LFS pointer header)
        try:
            with open(model_path, "r", errors="ignore") as _fh:
                head = _fh.read(1024)
            if "git-lfs" in head or head.startswith("version https://git-lfs.github.com/spec/v1"):
                raise RuntimeError(
                    f"Checkpoint at {model_path} appears to be a Git LFS pointer. Run 'git lfs install && git lfs pull' in the submodule or download the real .pt file (or use huggingface-hub)."
                )
        except Exception:
            pass
        raise RuntimeError(f"Failed to load checkpoint at {model_path}: {e}")

    # Support both raw state_dict and checkpoints with wrapping dicts
    if isinstance(ckpt, dict) and ("state_dict" in ckpt or "model_state_dict" in ckpt):
        sd = ckpt.get("state_dict", ckpt.get("model_state_dict"))
    else:
        sd = ckpt

    try:
        model.load_state_dict(sd)
    except Exception as e:
        raise RuntimeError(
            f"Failed to apply checkpoint from {model_path} to the model: {e}.\n"
            "Make sure the checkpoint was saved from the same model class (DnaBertOriginModel) and is a PyTorch state_dict or a checkpoint dict with 'state_dict'/'model_state_dict'."
        )

    if getattr(args, "verbose", False):
        print("[orilinx] Resolved DNABERT path:", resolved_dnabert)
        print("[orilinx] Model checkpoint:", model_path)
        print("[orilinx] Device:", device)
        print(f"[orilinx] Runtime settings: batch_size={args.batch_size}, num_workers={args.num_workers}, window=2000, stride={args.stride}")
        if getattr(args, "no_progress", False):
            print("[orilinx] Progress bars: disabled")

    if args.disable_flash:
        _disable_unpad_and_flash_everywhere(model)
    model.to(device)
    model.eval()

    chroms, ranges = _resolve_chroms_from_fasta(args.fasta_path, args.sequence_names)

    # Progress bars (per-sequence and overall). Use tqdm when available, else fall back to no-op.
    show_progress = not getattr(args, "no_progress", False)
    try:
        _tqdm = None
        if show_progress:
            from tqdm.auto import tqdm as _tqdm
            have_tqdm = True
        else:
            have_tqdm = False
    except Exception:
        def _identity(iterable=None, **kwargs):
            return iterable if iterable is not None else []
        _tqdm = _identity
        have_tqdm = False

    chrom_iter = _tqdm(chroms, desc="Sequences", total=len(chroms)) if have_tqdm else chroms

    def _collate(batch):
        seqs = [b["seq"] for b in batch]
        toks = tokenizer(
            seqs,
            padding="max_length",
            truncation=True,
            max_length=2000,
            return_tensors="pt",
        )
        # match eval datatypes/shapes; keep masks as long tensors of 0/1
        out = {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
            "chrom": [b["chrom"] for b in batch],
            "start": torch.tensor([b["start"] for b in batch], dtype=torch.long),
            "end": torch.tensor([b["end"] for b in batch], dtype=torch.long),
        }
        if "token_type_ids" in toks:
            out["token_type_ids"] = toks["token_type_ids"]
        return out

    for chrom in chrom_iter:
        # Estimate number of candidate windows for progress bar (ignores N-filtering)
        fa = pysam.FastaFile(args.fasta_path)
        if chrom not in fa.references:
            fa.close()
            continue
        clen = fa.get_reference_length(chrom)
        fa.close()
        last = clen - 2000
        if last < 0:
            continue
        num_windows = (last // args.stride) + 1

        # Per-sequence progress bar
        pbar = _tqdm(total=num_windows, desc=f"{chrom}", unit="win") if have_tqdm else None

        ds = _SlidingWindows(args.fasta_path, [chrom], 2000, args.stride, args.max_N_frac, ranges=ranges)
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=_collate,
        )

        rows = []
        with torch.no_grad():
            with (autocast(device_type="cuda") if device.type == "cuda" else torch.no_grad()):
                for batch in dl:
                    inputs = {
                        "input_ids": batch["input_ids"].to(device, non_blocking=True),
                        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
                    }
                    if "token_type_ids" in batch:
                        inputs["token_type_ids"] = batch["token_type_ids"].to(device, non_blocking=True)

                    # Keep eval-style call/return. Be resilient to Triton compilation errors
                    # used by flash-attention kernels: if a CompilationError occurs, disable
                    # flash/unpadded attention and retry once (this is slower but more robust).
                    try:
                        logits, _ = model(**inputs)
                    except Exception as e:
                        # Detect Triton compilation error when triton is present, or match by name
                        is_triton_compile_error = False
                        try:
                            from triton.compiler.errors import CompilationError
                            if isinstance(e, CompilationError):
                                is_triton_compile_error = True
                        except Exception:
                            # Fallback heuristics
                            if "triton" in type(e).__module__ or "CompilationError" in repr(e):
                                is_triton_compile_error = True

                        if is_triton_compile_error:
                            print("[orilinx] Triton compilation error detected during a flash kernel; disabling flash/unpad attention and retrying (this will be slower).")
                            # Disable fast/unpadded flags on model and retry once
                            _disable_unpad_and_flash_everywhere(model)
                            model.to(device)
                            try:
                                logits, _ = model(**inputs)
                            except Exception as e2:
                                raise RuntimeError(f"Retry after disabling flash-attention failed: {e2}") from e2
                        else:
                            raise

                    probs = torch.sigmoid(logits)

                    starts = batch["start"].numpy()
                    ends = batch["end"].numpy()
                    centers = (starts + (ends - starts)//2).astype(np.int64)
                    logits_np = logits.detach().cpu().numpy().astype(np.float32)
                    probs_np = probs.detach().cpu().numpy().astype(np.float32)

                    for i in range(len(starts)):
                        rows.append((chrom, int(starts[i]), int(ends[i]), int(centers[i]),
                                     float(logits_np[i]), float(probs_np[i])))

                    # Update per-sequence progress
                    if pbar is not None:
                        try:
                            n = len(batch["chrom"]) if "chrom" in batch else len(starts)
                        except Exception:
                            n = len(starts)
                        pbar.update(n)

        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass

        if not rows:
            continue

        df = pd.DataFrame(rows, columns=["chrom","start","end","center","logit","prob"])
        # Sort by genomic position to ensure ordered output
        df = df.sort_values(by="start").reset_index(drop=True)
        
        # Determine region boundaries for this chrom
        if chrom in ranges:
            region_start, region_end = ranges[chrom]
        else:
            region_start, region_end = None, None
        
        # Write bedgraph output
        _write_bedgraph_center(df, os.path.join(args.output_dir, f"{chrom}.bedGraph"), value=args.score, stride=args.stride, region_start=region_start, region_end=region_end)
        
        # Write CSV output if requested
        if getattr(args, "output_csv", False):
            _write_csv_windows(df, os.path.join(args.output_dir, f"{chrom}.csv"))

    print("Done.")


# CLI entry used by console_scripts
def cli():
    main()


if __name__ == "__main__":
    main()

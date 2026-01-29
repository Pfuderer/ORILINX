import os
import sys
import argparse
import urllib.request
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from torch.amp import autocast

from .model_architecture import DnaBertOriginModel, _find_dnabert_local_path, disable_unpad_and_flash_everywhere
from .data import SlidingWindows, resolve_chroms_from_fasta
from .io import write_bedgraph_center, write_csv_windows
from .utils import find_default_model_path


# Model download URLs
MODEL_URLS = {
    "model_epoch_6.pt": "https://huggingface.co/MBoemo/ORILINX/resolve/main/model_epoch_6.pt",
    "pytorch_model.bin": "https://huggingface.co/MBoemo/DNABERT-2-117M-Flash/resolve/main/pytorch_model.bin",
}


def _is_git_lfs_pointer_file(path: str) -> bool:
    """Return True if `path` looks like a Git LFS pointer file."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(256)
        # LFS pointers are small text files starting with this header.
        return head.startswith(b"version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False


def _find_models_dir() -> Path:
    """Find the models/ directory relative to the package installation."""
    # Start from the package directory and look for models/
    pkg_dir = Path(__file__).resolve().parent
    
    # Check common locations
    candidates = [
        pkg_dir.parent / "models",  # ../models from orilinx/
        pkg_dir / "models",         # orilinx/models/ (unlikely but check)
    ]
    
    # Also walk up from cwd
    cwd = Path.cwd()
    for p in [cwd] + list(cwd.parents):
        candidates.append(p / "models")
    
    for models_dir in candidates:
        if models_dir.exists() and models_dir.is_dir():
            return models_dir
    
    # Default to package parent's models dir (will be created)
    return pkg_dir.parent / "models"


def _download_with_progress(url: str, dest_path: Path, desc: str = None) -> None:
    """Download a file with progress display."""
    if desc is None:
        desc = dest_path.name
    
    # Create parent directories if needed
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use a temporary file during download
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    
    try:
        with urllib.request.urlopen(url) as response:
            total_size = response.headers.get('Content-Length')
            if total_size:
                total_size = int(total_size)
            
            print(f"Downloading {desc}...")
            if total_size:
                print(f"  Size: {total_size / (1024*1024):.1f} MB")
            
            downloaded = 0
            block_size = 8192 * 16  # 128KB blocks
            
            with open(tmp_path, 'wb') as out_file:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    out_file.write(buffer)
                    downloaded += len(buffer)
                    
                    if total_size:
                        percent = (downloaded / total_size) * 100
                        mb_done = downloaded / (1024*1024)
                        mb_total = total_size / (1024*1024)
                        print(f"\r  Progress: {mb_done:.1f}/{mb_total:.1f} MB ({percent:.1f}%)", end="", flush=True)
            
            print()  # Newline after progress
        
        # Move temp file to final destination
        tmp_path.rename(dest_path)
        print(f"  Saved to: {dest_path}")
        
    except Exception as e:
        # Clean up temp file on error
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {e}") from e


def _fetch_models(force: bool = False, verbose: bool = False) -> int:
    """Download model weights from Hugging Face.
    
    Returns exit code: 0 on success, non-zero on failure.
    """
    models_dir = _find_models_dir()
    dnabert_dir = models_dir / "DNABERT-2-117M-Flash"
    
    print(f"[orilinx] Models directory: {models_dir}")
    
    # Ensure directories exist
    models_dir.mkdir(parents=True, exist_ok=True)
    dnabert_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_download = [
        {
            "name": "model_epoch_6.pt",
            "url": MODEL_URLS["model_epoch_6.pt"],
            "dest": models_dir / "model_epoch_6.pt",
            "desc": "ORILINX fine-tuned model checkpoint",
        },
        {
            "name": "pytorch_model.bin", 
            "url": MODEL_URLS["pytorch_model.bin"],
            "dest": dnabert_dir / "pytorch_model.bin",
            "desc": "DNABERT-2 base model weights",
        },
    ]
    
    errors = []
    
    for file_info in files_to_download:
        dest = file_info["dest"]
        
        # Check if file exists and is valid (not an LFS pointer)
        if dest.exists() and not force:
            if _is_git_lfs_pointer_file(str(dest)):
                print(f"[orilinx] {file_info['name']}: exists but is a Git LFS pointer, will re-download")
            else:
                # Check file size to ensure it's a real file (not empty or tiny)
                size = dest.stat().st_size
                if size > 1000:  # Reasonable minimum size for a model file
                    print(f"[orilinx] {file_info['name']}: already exists ({size / (1024*1024):.1f} MB), skipping (use --force to re-download)")
                    continue
                else:
                    print(f"[orilinx] {file_info['name']}: exists but is too small ({size} bytes), will re-download")
        
        try:
            _download_with_progress(file_info["url"], dest, file_info["desc"])
        except Exception as e:
            errors.append(f"{file_info['name']}: {e}")
            print(f"[orilinx] ERROR downloading {file_info['name']}: {e}")
    
    if errors:
        print("\n[orilinx] Some downloads failed:")
        for err in errors:
            print(f"  - {err}")
        return 1
    
    print("\n[orilinx] All model files downloaded successfully!")
    print("[orilinx] You can now run 'orilinx --fasta_path <fasta> --output_dir <dir>' to make predictions.")
    return 0


def _doctor() -> int:
    """Preflight checks for model assets; returns process exit code."""
    problems = []

    def _note(msg: str):
        print(msg)

    _note("[orilinx] Doctor: checking local model assets...")

    # Check DNABERT directory
    resolved_dnabert = _find_dnabert_local_path()
    if not resolved_dnabert:
        problems.append(
            "DNABERT path not found. Set ORILINX_DNABERT_PATH or place DNABERT under a models/ directory."
        )
    else:
        if not os.path.isdir(resolved_dnabert):
            problems.append(f"DNABERT path does not exist or is not a directory: {resolved_dnabert}")
        else:
            cfg = os.path.join(resolved_dnabert, "config.json")
            if not os.path.isfile(cfg):
                problems.append(f"DNABERT config.json not found at: {cfg}")

            weight_candidates = [
                os.path.join(resolved_dnabert, "pytorch_model.bin"),
                os.path.join(resolved_dnabert, "model.safetensors"),
                os.path.join(resolved_dnabert, "pytorch_model.bin.index.json"),
            ]
            have_any = any(os.path.exists(p) for p in weight_candidates)
            if not have_any:
                problems.append(
                    "DNABERT weights not found (expected pytorch_model.bin/model.safetensors or an index file) "
                    f"under: {resolved_dnabert}"
                )
            else:
                for pth in weight_candidates:
                    if os.path.isfile(pth) and _is_git_lfs_pointer_file(pth):
                        problems.append(
                            f"DNABERT weight file appears to be a Git LFS pointer (not downloaded): {pth}"
                        )

    # Check fine-tuned checkpoint
    model_path = None
    try:
        model_path = find_default_model_path()
    except Exception as e:
        problems.append(f"Failed while resolving checkpoint path: {e}")

    if not model_path:
        problems.append(
            "Fine-tuned checkpoint not found. Put a .pt file under models/ or set ORILINX_MODEL=/path/to/model.pt"
        )
    else:
        if _is_git_lfs_pointer_file(model_path):
            problems.append(
                f"Checkpoint appears to be a Git LFS pointer (not downloaded): {model_path}"
            )

    if not problems:
        _note("[orilinx] Doctor: OK (models look present).")
        return 0

    _note("[orilinx] Doctor: problems detected:\n")
    for p in problems:
        _note(f"- {p}")

    _note(
        "\n[orilinx] Fix suggestions:\n"
        "- Run 'orilinx fetch_models' to download model weights from Hugging Face\n"
        "- Alternatively, if you cloned from Git, install Git LFS and fetch large files: `git lfs install` then `git lfs pull`\n"
        "- Some HPCs set `GIT_LFS_SKIP_SMUDGE=1`; unset it and run `git lfs pull`\n"
        "- If outbound network access is restricted, copy weights from a machine that can fetch them, or set ORILINX_MODEL/ORILINX_DNABERT_PATH to shared locations"
    )
    return 2


def _create_collate_fn(tokenizer):
    """Create a collate function for the DataLoader."""
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
    return _collate


def _load_model(model_path, device):
    """Load model checkpoint with helpful error messages."""
    try:
        ckpt = torch.load(model_path, map_location="cpu")  # keep on CPU for safety
    except Exception as e:
        # Detect a likely git-lfs pointer (text file starting with the LFS pointer header)
        try:
            with open(model_path, "r", errors="ignore") as _fh:
                head = _fh.read(1024)
            if "git-lfs" in head or head.startswith("version https://git-lfs.github.com/spec/v1"):
                raise RuntimeError(
                    f"Checkpoint at {model_path} appears to be a Git LFS pointer. "
                    "Run 'git lfs install && git lfs pull' in the submodule or download the real .pt file "
                    "(or use huggingface-hub)."
                )
        except Exception:
            pass
        raise RuntimeError(f"Failed to load checkpoint at {model_path}: {e}")

    # Support both raw state_dict and checkpoints with wrapping dicts
    if isinstance(ckpt, dict) and ("state_dict" in ckpt or "model_state_dict" in ckpt):
        sd = ckpt.get("state_dict", ckpt.get("model_state_dict"))
    else:
        sd = ckpt

    model = DnaBertOriginModel(model_name=_find_dnabert_local_path(), enable_grad_checkpointing=False)
    try:
        model.load_state_dict(sd)
    except Exception as e:
        raise RuntimeError(
            f"Failed to apply checkpoint from {model_path} to the model: {e}.\n"
            "Make sure the checkpoint was saved from the same model class (DnaBertOriginModel) and is a "
            "PyTorch state_dict or a checkpoint dict with 'state_dict'/'model_state_dict'."
        )
    
    model.to(device)
    return model


def _setup_tqdm(show_progress):
    """Set up tqdm progress bars."""
    try:
        if show_progress:
            from tqdm.auto import tqdm as _tqdm
            return _tqdm, True
        else:
            def _identity(iterable=None, **kwargs):
                return iterable if iterable is not None else []
            return _identity, False
    except Exception:
        def _identity(iterable=None, **kwargs):
            return iterable if iterable is not None else []
        return _identity, False


def main(argv=None):
    """Main entry point for ORILINX CLI."""
    if argv is None:
        argv = sys.argv[1:]

    # Check for subcommands first
    if argv and argv[0] == "fetch_models":
        # Handle fetch_models subcommand
        fetch_parser = argparse.ArgumentParser(
            prog="orilinx fetch_models",
            description="Download model weights from Hugging Face."
        )
        fetch_parser.add_argument(
            "--force",
            action="store_true",
            help="Force re-download even if files already exist."
        )
        fetch_parser.add_argument(
            "--verbose",
            action="store_true", 
            help="Enable verbose output."
        )
        fetch_args = fetch_parser.parse_args(argv[1:])
        raise SystemExit(_fetch_models(force=fetch_args.force, verbose=fetch_args.verbose))

    # Otherwise, run the prediction pipeline
    _run_predict(argv)


def _run_predict(argv):
    """Main prediction pipeline."""
    p = argparse.ArgumentParser(description="Genome-wide origin scores with ORILINX.")
    p.add_argument(
        "--fasta_path",
        default=None,
        help="Path to the reference FASTA file; an index (.fai) must be present."
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Directory where per-sequence output bedgraphs will be written."
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1000,
        help="Stride in base pairs (bp) between consecutive windows."
    )
    p.add_argument(
        "--max_N_frac",
        type=float,
        default=0.05,
        help="Maximum fraction of 'N' bases allowed in a window; windows exceeding this are skipped."
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of windows per batch; increase for throughput if memory allows."
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker processes used by DataLoader for data loading (0 runs in main process)."
    )
    p.add_argument(
        "--sequence_names",
        type=str,
        default="all",
        help='Comma-separated list of sequence names to process; supports ranges '
             '(e.g., "chr1,chr2:2000-6000"); use "all" for all primary sequences.'
    )
    p.add_argument(
        "--score",
        choices=["logit", "prob"],
        default="prob",
        help="Output score type: 'logit' (raw model logits) or 'prob' (sigmoid probability)."
    )
    p.add_argument(
        "--output_csv",
        action="store_true",
        help="Also output results as CSV files with columns: chromosome, start, end, probability, logit."
    )
    p.add_argument(
        "--disable_flash",
        action="store_true",
        help="Force non-flash (padded) attention mode; safer for long sequences."
    )
    p.add_argument(
        "--no-progress",
        dest="no_progress",
        action="store_true",
        help="Disable progress bars (useful for logging or non-interactive runs)."
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (prints DNABERT path, model checkpoint, device and runtime settings)."
    )
    p.add_argument(
        "--doctor",
        action="store_true",
        help="Check that required model weights are present then exit."
    )
    args = p.parse_args(argv)

    if getattr(args, "doctor", False):
        raise SystemExit(_doctor())

    # Validate required args for normal prediction runs.
    if not args.fasta_path:
        p.error("--fasta_path is required (unless --doctor is used)")
    if not args.output_dir:
        p.error("--output_dir is required (unless --doctor is used)")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve DNABERT local path
    resolved_dnabert = _find_dnabert_local_path()
    if resolved_dnabert is None:
        raise RuntimeError(
            "DNABERT not found: set ORILINX_DNABERT_PATH to a valid local DNABERT folder or place DNABERT "
            "under a 'models/' directory (searched upward from CWD)."
        )

    # Load tokenizer and model
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        resolved_dnabert,
        local_files_only=True,
    )
    model = DnaBertOriginModel(model_name=resolved_dnabert, enable_grad_checkpointing=False)

    # Find and load model checkpoint
    model_path = find_default_model_path()
    if model_path is None:
        raise RuntimeError(
            "No model checkpoint found in any 'models/' directory. Please place your .pt checkpoint in a "
            "'models/' folder (searched upward from CWD)."
        )

    model = _load_model(model_path, device)

    if getattr(args, "verbose", False):
        print("[orilinx] Resolved DNABERT path:", resolved_dnabert)
        print("[orilinx] Model checkpoint:", model_path)
        print("[orilinx] Device:", device)
        print(
            f"[orilinx] Runtime settings: batch_size={args.batch_size}, num_workers={args.num_workers}, "
            f"window=2000, stride={args.stride}"
        )
        if getattr(args, "no_progress", False):
            print("[orilinx] Progress bars: disabled")

    if args.disable_flash:
        disable_unpad_and_flash_everywhere(model)
    
    model.eval()

    # Resolve chromosomes and ranges
    chroms, ranges = resolve_chroms_from_fasta(args.fasta_path, args.sequence_names)

    # Setup progress bars
    show_progress = not getattr(args, "no_progress", False)
    _tqdm, have_tqdm = _setup_tqdm(show_progress)

    chrom_iter = _tqdm(chroms, desc="Sequences", total=len(chroms)) if have_tqdm else chroms

    # Create collate function
    collate_fn = _create_collate_fn(tokenizer)

    # Main prediction loop
    for chrom in chrom_iter:
        # Estimate number of candidate windows for progress bar
        fa = pysam.FastaFile(args.fasta_path)
        if chrom not in fa.references:
            fa.close()
            continue
        clen = fa.get_reference_length(chrom)
        fa.close()
        
        # Determine the range to process for this chrom
        if chrom in ranges:
            range_start, range_end = ranges[chrom]
            range_start = max(0, range_start)
            range_end = min(clen, range_end)
        else:
            range_start, range_end = 0, clen
        
        last = range_end - 2000
        if last < range_start:
            continue
        num_windows = ((last - range_start) // args.stride) + 1

        # Per-sequence progress bar
        pbar = _tqdm(total=num_windows, desc=f"{chrom}", unit="win") if have_tqdm else None

        # Create dataset and dataloader
        ds = SlidingWindows(args.fasta_path, [chrom], 2000, args.stride, args.max_N_frac, ranges=ranges)
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_fn,
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

                    # Attempt model forward pass with fallback for Triton compilation errors
                    try:
                        logits, _ = model(**inputs)
                    except RuntimeError as e:
                        # Detect CUDA out-of-memory error
                        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                            raise RuntimeError(
                                f"GPU out of memory error during batch processing.\n"
                                f"Batch size: {args.batch_size}\n"
                                f"Sequence: {chrom}\n"
                                f"\nTry reducing --batch_size (current: {args.batch_size}) or use --disable_flash.\n"
                                f"Original error: {e}"
                            ) from e
                        
                        # Detect Triton compilation error
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
                            print(
                                "[orilinx] Triton compilation error detected during a flash kernel; "
                                "disabling flash/unpad attention and retrying (this will be slower)."
                            )
                            disable_unpad_and_flash_everywhere(model)
                            model.to(device)
                            try:
                                logits, _ = model(**inputs)
                            except Exception as e2:
                                raise RuntimeError(
                                    f"Retry after disabling flash-attention failed: {e2}"
                                ) from e2
                        else:
                            raise
                    except Exception as e:
                        # Catch other exceptions and add context
                        raise RuntimeError(
                            f"Error during model forward pass on sequence {chrom} with batch_size={args.batch_size}: {e}"
                        ) from e

                    probs = torch.sigmoid(logits)

                    starts = batch["start"].numpy()
                    ends = batch["end"].numpy()
                    centers = (starts + (ends - starts) // 2).astype(np.int64)
                    logits_np = logits.detach().cpu().numpy().astype(np.float32)
                    probs_np = probs.detach().cpu().numpy().astype(np.float32)

                    for i in range(len(starts)):
                        rows.append(
                            (
                                chrom,
                                int(starts[i]),
                                int(ends[i]),
                                int(centers[i]),
                                float(logits_np[i]),
                                float(probs_np[i]),
                            )
                        )

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

        # Prepare output dataframe
        df = pd.DataFrame(rows, columns=["chrom", "start", "end", "center", "logit", "prob"])
        # Sort by genomic position to ensure ordered output
        df = df.sort_values(by="start").reset_index(drop=True)
        
        # Determine region boundaries for this chrom
        if chrom in ranges:
            region_start, region_end = ranges[chrom]
        else:
            region_start, region_end = None, None
        
        # Write bedgraph output
        write_bedgraph_center(
            df,
            os.path.join(args.output_dir, f"{chrom}.bedGraph"),
            value=args.score,
            stride=args.stride,
            region_start=region_start,
            region_end=region_end,
        )
        
        # Write CSV output if requested
        if getattr(args, "output_csv", False):
            write_csv_windows(df, os.path.join(args.output_dir, f"{chrom}.csv"))

    print("Done.")


if __name__ == "__main__":
    main()

import os
import sys
import argparse

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
    """Main prediction pipeline."""
    if argv is None:
        argv = sys.argv[1:]

    p = argparse.ArgumentParser(description="Genome-wide origin scores with ORILINX.")
    p.add_argument(
        "--fasta_path",
        required=True,
        help="Path to the reference FASTA file; an index (.fai) must be present."
    )
    p.add_argument(
        "--output_dir",
        required=True,
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
        default=64,
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
    args = p.parse_args(argv)

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

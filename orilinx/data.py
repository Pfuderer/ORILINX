"""Data loading and processing for genomic windows."""

import warnings
import torch
from torch.utils.data import IterableDataset
import pysam


class SlidingWindows(IterableDataset):
    """Iterable dataset for generating sliding windows across genomic sequences."""

    def __init__(self, fasta_path, chroms, window, stride, max_N_frac, ranges=None):
        self.fasta_path = fasta_path
        self.chroms = chroms
        self.window = int(window)
        self.stride = int(stride)
        self.max_N_frac = float(max_N_frac)
        # ranges is a dict mapping chrom name to (start, end) tuple, or None for full sequence
        self.ranges = ranges or {}

    def __iter__(self):
        """Yield windows, distributing across workers via round-robin indexing."""
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


def resolve_chroms_from_fasta(fasta_path: str, arg: str):
    """Parse sequence names with optional ranges (chr or chr:start-end format).
    
    Returns a tuple of (chroms_list, ranges_dict) where:
    - chroms_list: list of sequence names to process
    - ranges_dict: dict mapping chrom -> (start, end) for ranges, or empty dict if no ranges
    """
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

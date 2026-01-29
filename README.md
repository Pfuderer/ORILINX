<p align='centre'><img src='logo/Logo_Text_v1.png' alt='Overview.' width='80%'> </p>

# ORILINX
[![Documentation Status](https://readthedocs.org/projects/orilinx/badge/?version=latest)](https://orilinx.readthedocs.io/en/latest/?badge=latest)

ORILINX is a large language model built on DNABERT-2 to detect the location of replication origins across multiple eukaryotic genomes.

### Building from Source

For development or to use the latest unreleased version:

```bash
git clone --recursive https://github.com/Pfuderer/ORILINX.git
cd ORILINX
pip install -e .
orilinx fetch_models
```

This installs ORILINX in editable mode and creates a console script named `orilinx` that runs prediction. The `fetch_models` command downloads the required model weights from Hugging Face (~900 MB total) and only needs to be run once.

### Usage

After installation, use the `orilinx` command to predict replication origins:

```bash
# Basic usage: analyse all chromosomes
orilinx --fasta_path genome.fa --output_dir results

# Analyse specific chromosomes only
orilinx --fasta_path genome.fa --output_dir results --sequence_names chr1,chr2,chr3

# Analyse a specific region (must be ≥2000 bp)
orilinx --fasta_path genome.fa --output_dir results --sequence_names chr8:1000000-5000000

# Also generate CSV output alongside bedGraph
orilinx --fasta_path genome.fa --output_dir results --output_csv
```

### Required Arguments

- `--fasta_path PATH` (required): Path to the reference FASTA file. An index file (`.fai`) must be present in the same directory. Create it with: `samtools faidx genome.fa`
- `--output_dir PATH` (required): Directory where results will be saved.

### Additional Options

- `--sequence_names SEQUENCES` (default: all): Which chromosomes or regions to analyse. Use comma-separated names (e.g., `chr1,chr2`) or a comma-separated list of regions (e.g., `chr1:2000-6000`).
- `--output_csv`: Generate CSV files in addition to bedGraph files.
- `--stride INT` (default: 1000): Stride in base pairs (bp) between consecutive windows.
- `--score {logit,prob}` (default: prob): Output score format (probability 0-1 or raw logit).
- `--batch_size INT` (default: 32): Number of windows per batch. Increase for faster analysis if memory allows; decrease if you hit CUDA out-of-memory.
- `--num_workers INT` (default: 8): Background processes for data loading. Set to 0 if experiencing issues.
- `--max_N_frac FLOAT` (default: 0.05): Skip regions with >5% unknown bases ("N"). Adjust tolerance as needed.
- `--disable_flash`: Use standard attention instead of flash attention (slower but compatible with more GPUs).
- `--verbose`: Show detailed runtime information (prints model paths, device and runtime settings).
- `--no-progress`: Hide progress bars.
- `--doctor`: Check that required model weights are present, then exit.


For a complete list of options, run: `orilinx --help`.

### Examples

```bash
# Analyse human chromosome 8 with CSV output
orilinx --fasta_path hg38.fa --output_dir results --sequence_names chr8 --output_csv

# Analyse multiple chromosomes
orilinx --fasta_path hg38.fa --output_dir results --sequence_names chr1,chr2,chr3,chrX

# Analyse specific genomic region (1 Mb region on chr8)
orilinx --fasta_path hg38.fa --output_dir results --sequence_names chr8:10000000-11000000

# Larger batches for throughput (VRAM-dependent; reduce if you hit OOM)
orilinx --fasta_path hg38.fa --output_dir results --batch_size 64

# Verbose output and no progress bars (for CI/logging)
orilinx --fasta_path hg38.fa --output_dir results --verbose --no-progress
```

### Documentation
Please see the [documentation](https://orilinx.readthedocs.io) for detailed usage instructions, visualisation, and an example workflow.

### Citation
If you use ORILINX for your research, please cite our publication.

### Questions and Bugs
It is under active development by the [Boemo Group](https://www.boemogroup.org/) based in the [Department of Pathology, University of Cambridge](https://www.path.cam.ac.uk/).

Should any bugs arise or if you have basic usage questions, please raise a [GitHub issue](https://github.com/Pfuderer/ORILINX/issues). For more detailed discussions or collaborations, please Email Michael Boemo at mb915@cam.ac.uk.

### Funding

The core of this work was funded by the [Cancer Research UK Cambridge Centre](https://crukcambridgecentre.org.uk).

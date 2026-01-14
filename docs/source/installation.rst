.. _installation:

Getting Started
===============================


Singularity
---------------------

placeholder


Building from Source
---------------------

Install in editable mode for development and to expose the CLI entry point::

    pip install -e .

This provides the `orilinx` console script. Example invocation::

    orilinx --fasta_path /path/to/genome.fa --output_dir /path/to/out --write_csv --score logit

For compatibility with older scripts, the repository also contains `predict_genome_optim.py` and `06_predict_genome_optim.py` shims that forward to the packaged CLI.

Model discovery: The CLI requires a checkpoint file to be placed in a `models/` directory. The newest `.pt` is used; the command will fail if no checkpoint is found. The discovery now searches both your current working directory tree and the package tree, so placing `models/` in the repository root will be found even when running `orilinx` from another CWD.

Note: the DNABERT-2 model folder is discovered similarly (see above). If your `.pt` is missing after a clone you likely need to fetch large file objects (Git LFS) or download the checkpoint from a model host (Hugging Face). Recommended options:

- If you keep the checkpoint in a git repo with LFS: run `git lfs install && git lfs pull` in the repository (or submodule) to fetch real blobs.
- Preferably, host heavy checkpoints on the Hugging Face Hub (or S3) and use a helper script to download into `models/` (see `scripts/download_models.py` if present).
- To explicitly point to a particular checkpoint, set `ORILINX_MODEL=/absolute/path/to/model_epoch_6.pt` in your environment.

If DNABERT or a checkpoint is missing the CLI will give a clear error explaining how to fix it; use `--fetch_dnabert` to auto-download DNABERT, and consider adding a download helper for the model checkpoint as described above.


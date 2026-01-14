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

This provides the `orilinx-predict` console script. Example invocation::

    orilinx predict --fasta_path /path/to/genome.fa --output_dir /path/to/out --write_csv --score logit

For compatibility with older scripts, the repository also contains `predict_genome_optim.py` and `06_predict_genome_optim.py` shims that forward to the packaged CLI.

Model discovery: The CLI requires a checkpoint file to be placed in a `models/` directory (searched upward from CWD). The newest `.pt` is used; the command will fail if no checkpoint is found. Note: the DNABERT-2 model folder is also discovered from `ORILINX_DNABERT_PATH` or any child folder of a `models/` directory (searched upward from CWD). There is no CLI flag to override DNABERT — place it in `models/` or set `ORILINX_DNABERT_PATH`.


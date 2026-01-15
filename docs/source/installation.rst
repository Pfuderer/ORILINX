.. _installation:

Getting Started
===============================

**Before You Start**

ORILINX requires:

- A computer with Python 3.8 or newer installed
- A reference genome in FASTA format (e.g., hg38, hg19)
- (Optional, recommended) A GPU for faster analysis

If you don't have Python installed, download it from `python.org <https://www.python.org>`_. On Mac, you can also use Homebrew: ``brew install python3``


Installation
---------------------

**Step 1: Install ORILINX**

Open a terminal and run::

    pip install -e .

This installs ORILINX from the current directory. The ``-e`` flag means it will update automatically if you edit the code.

If you get a "command not found" error for ``pip``, you may need to use ``pip3`` instead.

**Step 2: Prepare Your Data**

ORILINX needs two things:

1. **A genome file** (FASTA format). This is typically a file ending in ``.fa`` or ``.fasta``. Standard genomes like hg38 can be downloaded from NCBI or Ensembl.

2. **An index file**. This is a small file that helps ORILINX find sequences quickly. If your FASTA file doesn't have an index (a ``.fai`` file), create one::

    samtools faidx your_genome.fa

If you don't have samtools installed, you can install it with: ``conda install -c bioconda samtools``

**Step 3: Run ORILINX**

Now you can run ORILINX! Here's the simplest command::

    orilinx --fasta_path /path/to/your_genome.fa --output_dir results

This will:
- Analyze all primary chromosomes in your genome
- Save results as bedGraph files in a folder called ``results``
- Use default analysis parameters (described below)

**Where Do I Find My Genome File?**

If you don't have a genome file:

- **Human genome (hg38)**: Download from `UCSC <http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/>`_ (look for ``hg38.fa.gz``)
- **Other organisms**: Check `Ensembl <https://www.ensembl.org>`_ or `NCBI <https://www.ncbi.nlm.nih.gov/>`_

After downloading, extract if compressed: ``gunzip hg38.fa.gz``

Then create the index: ``samtools faidx hg38.fa``

**Troubleshooting Installation**

- *"Command not found: orilinx"* → Make sure you're in the ORILINX directory and ran ``pip install -e .``
- *"No module named transformers"* → Install missing packages with ``pip install torch transformers pysam pandas peft``
- *"Index file not found"* → Run ``samtools faidx`` on your FASTA file

Singularity
---------------------

placeholder


Flags and options
---------------------



The ``orilinx`` command-line tool accepts the following arguments:

**Required Arguments**

- ``--fasta_path PATH`` : Path to the reference FASTA file. An index file (``.fai``) must be present in the same directory. The index can be created with ``samtools faidx <fasta>``.

- ``--output_dir PATH`` : The folder where ORILINX will save its results. The program will create this folder if it doesn't exist.

**Which Parts of Your Genome to Analyze**

- ``--sequence_names SEQUENCES`` : Which parts of your genome to analyze. By default (``all``), ORILINX analyzes all standard human chromosomes (chr1 through chr22, plus chrX and chrY). 
  
  You can specify specific chromosomes or regions:
  
  - ``--sequence_names chr1,chr2`` : Just analyze chromosome 1 and 2.
  - ``--sequence_names chr1:2000-6000,chr2:80000-100000`` : Analyze only specific regions (in this example, positions 2000-6000 on chr1 and positions 80000-100000 on chr2).
  
  **Important:** If you specify a region, it must be at least 2000 base pairs long (this is the size of the regions ORILINX analyzes at a time).

**How to Scan Your Genome**

These settings control how ORILINX analyzes your genome, one region at a time.

- ``--stride INT`` (default: 1000) : How many base pairs to move forward between each analysis region. Think of this like moving a sliding window across your genome. With the default value of 1000, ORILINX analyzes bases 1-2000, then 1000-3000, then 2000-4000, and so on. Smaller values mean more detailed coverage but take longer to run.

- ``--max_N_frac FLOAT`` (default: 0.05) : Sometimes a DNA sequence contains unknown bases marked as "N". This setting controls the tolerance: if more than this fraction (5% by default) of a region contains "N"s, ORILINX skips that region. Set to 0.1 (10%) to be more permissive, or 0.01 (1%) to be stricter.

**Computer Performance Settings**

These options help ORILINX run faster or work around computer limitations.

- ``--batch_size INT`` (default: 64) : How many regions to analyze at once. If you have a powerful GPU, increase this (e.g., 128) for faster analysis. If you run out of memory, decrease it (e.g., 32).

- ``--num_workers INT`` (default: 8) : How many background processes to use for loading data. Higher numbers are faster but use more computer resources. If you experience problems, set this to 0 to disable parallel processing.

- ``--disable_flash`` : Use this flag if ORILINX crashes or gives errors mentioning "GPU memory". It forces ORILINX to use a slower but more reliable computation method. Use it like: ``orilinx ... --disable_flash``

**What Numbers to Output**

- ``--score {logit,prob}`` (default: prob) : What type of numbers ORILINX saves in its result files:
  
  - ``prob`` (default) : Scores between 0 and 1, where 1 means "very likely to be an origin".
  - ``logit`` : Raw scores from the model (can be any number). Only use this if you know what you're doing.

**Information and Progress**

- ``--verbose`` : Show extra information while running (which model files are being used, what computer is being used, etc.). Add this flag if you want to check that ORILINX is using the right files and settings.

- ``--no-progress`` : Don't show the animated progress bars. Use this if you're running ORILINX on a remote computer or in a script where progress bars don't display correctly.

**Advanced: Manual Path Configuration**

If ORILINX can't find the files it needs, you can manually tell it where to look by setting these environment variables (ask your system administrator for help with this):

- ``ORILINX_DNABERT_PATH`` : Full path to the DNABERT model folder.
- ``ORILINX_MODEL`` : Full path to the model checkpoint file (the ``.pt`` file).

**Quick Example**

Here's a complete example that analyzes a specific region of chromosome 1::

    orilinx --fasta_path /path/to/hg38.fa --output_dir results --sequence_names chr1:1000000-2000000 --stride 500 --verbose

This analyzes positions 1,000,000 to 2,000,000 on chromosome 1, checking every 500 base pairs instead of the default 1000, and shows you what's happening.


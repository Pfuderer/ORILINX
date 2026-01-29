.. _installation:

Installation and Usage
===============================

**Before You Start**

ORILINX requires:

- Python 3.9 or newer installed
- A reference genome in FASTA format (e.g., hg38, hg19)

Note that ORILINX was designed to run on GPUs. While it can run on CPUs, doing so is strongly discouraged.

GPU Setup
-------------------------------------

If you have an NVIDIA GPU, follow these steps.

**Check if You Have an NVIDIA GPU**

Open a terminal and run::

    nvidia-smi

If this command returns GPU information, you have an NVIDIA GPU and can proceed. If it says "command not found", either you don't have an NVIDIA GPU, or the drivers aren't installed (see below).

**Install NVIDIA Drivers**

On Ubuntu/Debian::

    sudo apt update
    sudo apt install nvidia-driver-550

On Fedora/RHEL::

    sudo dnf install nvidia-driver

On macOS or other systems, download from `NVIDIA's driver page <https://www.nvidia.com/Download/driverDetails.aspx>`_.

After installation, restart your computer and verify with::

    nvidia-smi

**Install CUDA Toolkit**

CUDA is the computing platform that enables GPU acceleration. You need to match your GPU's compute capability and your PyTorch version.

1. Check your GPU compute capability at `NVIDIA's CUDA Compute Capability table <https://developer.nvidia.com/cuda-compute-capability-chart>`_.

2. Download CUDA from `NVIDIA's CUDA Toolkit website <https://developer.nvidia.com/cuda-toolkit>`_. For most modern GPUs and recent PyTorch, CUDA 12.1 or 12.4 is recommended.

3. Follow the installation instructions for your platform. For Ubuntu, a typical command is::

    wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
    sudo sh cuda_12.4.1_550.54.15_linux.run

4. After installation, add CUDA to your PATH by adding this to your ``~/.bashrc`` or ``~/.zshrc``::

    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

5. Reload your shell::

    source ~/.bashrc

6. Verify CUDA installation::

    nvcc --version

**Install cuDNN (CUDA Deep Neural Network Library)**

cuDNN is a library optimised for deep learning operations.

1. Download cuDNN from `NVIDIA's cuDNN page <https://developer.nvidia.com/cudnn>`_ (requires free registration).

2. Extract and copy to your CUDA directory::

    tar -xzf cudnn-linux-*.tar.xz
    sudo cp cudnn-*/include/cudnn.h /usr/local/cuda/include/
    sudo cp cudnn-*/lib/libcudnn* /usr/local/cuda/lib64/
    sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

**Install PyTorch with CUDA Support**

After CUDA is installed, install PyTorch with GPU support. Visit `pytorch.org <https://pytorch.org/get-started/locally/>`_ and select:

- **PyTorch Build**: Stable
- **Your OS**: Linux (or your OS)
- **Package**: pip
- **Language**: Python
- **Compute Platform**: CUDA 12.1 (or 12.4, matching your CUDA installation)

This will give you a command like::

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Run this command to install PyTorch with CUDA support.

**Verify GPU is Working**

After PyTorch installation, verify GPU support::

    python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

If this prints ``GPU available: True`` and shows your GPU name, you're ready to use ORILINX with GPU acceleration.

**Troubleshooting GPU Setup**

- **"GPU available: False"**: Check that PyTorch was installed with CUDA support (re-run the PyTorch install command from pytorch.org with CUDA selected).
- **CUDA version mismatch**: Ensure your CUDA Toolkit version matches the version PyTorch was compiled for (e.g., both CUDA 12.1).
- **Still not working?** Set ``CUDA_VISIBLE_DEVICES=""`` in your shell before running ORILINX to fall back to CPU, then run with ``--verbose`` flag and report the error.

**CPU-Only Installation**

If you don't have an NVIDIA GPU or prefer not to use GPU acceleration, ORILINX will automatically run on CPU (slower but still functional). No additional setup is needed beyond standard PyTorch installation.

Installation
---------------------

**Step 1: Install ORILINX**

Open a terminal and run::

    pip install -e .

This installs ORILINX from the current directory. The ``-e`` flag means it will update automatically if you edit the code.

If you get a "command not found" error for ``pip``, you may need to use ``pip3`` instead.

**Step 2: Download Model Weights**

After installation, download the required model weights::

    orilinx fetch_models

This downloads ~900 MB of model weights from Hugging Face. 

**Note for HPC users:** This is especially useful on HPC systems where Git LFS may not be available or configured correctly. It provides an alternative to ``git lfs pull`` for obtaining the model weights.

If you want to re-download the model weights (e.g., after a corrupted download), use::

    orilinx fetch_models --force

**Step 3: Prepare Your Data**

ORILINX only needs one input: a genome file in FASTA format. Standard genomes like hg38 can be downloaded from NCBI or Ensembl.

The genome must be indexed so ORILINX can find sequences quickly. If your FASTA file doesn't have an index (a ``.fai`` file) in the same directory, create one::

    samtools faidx your_genome.fa

If you don't have samtools installed, you can install it with: ``conda install -c bioconda samtools``

**Step 4: Run ORILINX**

Here's the simplest command::

    orilinx --fasta_path /path/to/your_genome.fa --output_dir results

This will:
- Analyse all primary chromosomes in your genome
- Save results as bedGraph files in a folder called ``results``
- Use default analysis parameters (described below)

**Where Do I Find My Genome File?**

If you don't have a genome file:

- **Human genome (hg38)**: Download from `UCSC <http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/>`_ (look for ``hg38.fa.gz``)
- **Other organisms**: Check `Ensembl <https://www.ensembl.org>`_ or `NCBI <https://www.ncbi.nlm.nih.gov/>`_

After downloading, extract if compressed: ``gunzip hg38.fa.gz``

Then create the index: ``samtools faidx hg38.fa``

**Singularity**

PLACEHOLDER


Flags and options
---------------------

The ``orilinx`` command-line tool accepts the following arguments:

**Required Arguments**

- ``--fasta_path PATH`` : Path to the reference FASTA file. An index file (``.fai``) must be present in the same directory. The index can be created with ``samtools faidx <fasta>``.

- ``--output_dir PATH`` : The folder where ORILINX will save its results. The folder will be created if it doesn't exist.

**Which Parts of Your Genome to Analyse**

- ``--sequence_names SEQUENCES`` : Sequence names in the FASTA file to analyse. By default (``all``), ORILINX analyses all sequences. 
  
  You can specify specific chromosomes or regions with a comma-separated list:
  
  - ``--sequence_names chr1,chr2`` : Just analyse chromosome 1 and 2.
  - ``--sequence_names chr1:2000-6000,chr2:80000-100000`` : Analyse only specific regions (in this example, positions 2000-6000 on chr1 and positions 80000-100000 on chr2).
  
  **Important:** If you specify a region, it must be at least 2000 base pairs long (this is the size of the regions ORILINX analyses at a time).

**How to Scan Your Genome**

These settings control how ORILINX analyses your genome, one region at a time.

- ``--stride INT`` (default: 1000) : How many base pairs to move forward between each analysis region. Think of this like moving a sliding window across your genome. With the default value of 1000, ORILINX analyses bases 1-2000, then 1000-3000, then 2000-4000, and so on. Smaller values mean more detailed coverage but take longer to run. The window size is fixed at 2000 bp and cannot be changed.

- ``--max_N_frac FLOAT`` (default: 0.05) : Sometimes a DNA sequence contains unknown bases marked as "N". This setting controls the tolerance: if more than this fraction (5% by default) of a region contains "N"s, ORILINX skips that region. Set to 0.1 (10%) to be more permissive, or 0.01 (1%) to be stricter.

**Output Format Options**

- ``--score {logit,prob}`` (default: prob) : What type of numbers ORILINX saves in its result files:
  
  - ``prob`` (default) : Scores between 0 and 1, where higher values indicate higher likelihood of being an origin.
  - ``logit`` : Raw scores from the model. Use this for analyses requiring raw model output.

- ``--output_csv`` : In addition to bedGraph files, also generate CSV files for each sequence with columns: chromosome, start, end, probability, and logit. This is useful for downstream analysis in Python or R. By default, only bedGraph files are generated.

**Performance Settings**

These options help ORILINX run faster or work around computer limitations.

- ``--batch_size INT`` (default: 32) : How many regions to analyse at once. If you have a powerful GPU, you can increase this for faster analysis. If you run out of memory, decrease it (e.g., 16 or 8). Larger batch sizes are generally faster but consume more GPU/CPU memory.

- ``--num_workers INT`` (default: 8) : How many background processes to use for loading data. Higher numbers are faster but use more resources. Set this based on your CPU core count. If you experience problems, set this to 0 to disable parallel processing (slower but more stable).

- ``--disable_flash`` : Use this flag if ORILINX crashes with errors mentioning "Triton", "flash attention", or "GPU memory". It forces ORILINX to use standard PyTorch attention which is slower but more stable and compatible with a wider range of GPUs.

**Progress and Diagnostics**

- ``--verbose`` : Show extra information while running (which model files are being used, what device is being used, runtime settings, etc.). Use this if you want to verify that ORILINX is using the correct configuration and GPU, or for troubleshooting and bug reports.

- ``--no-progress`` : Hide the animated progress bars. Use this if you're running ORILINX on a remote server, in a cluster job, or in a script where progress bars don't display correctly or cause output issues.

- ``--doctor`` : Check that required model files are present and correctly downloaded, then exit without running any analysis. Use this to verify your installation is complete before running predictions. If problems are detected, ORILINX will print suggestions for how to fix them.

Specifying model paths
----------------------

With correct installation, this shouldn't be needed, but you can manually specify model locations using environment variables if ORILINX can't find them:

- ``ORILINX_DNABERT_PATH`` : Full path to the DNABERT-2 model folder (the one containing ``config.json`` and other model files).
- ``ORILINX_MODEL`` : Full path to the model checkpoint file (the ``.pt`` file). If not set, ORILINX searches for the newest ``.pt`` file in any ``models/`` directory.

Example::

    export ORILINX_DNABERT_PATH=/path/to/DNABERT-2-117M-Flash
    export ORILINX_MODEL=/path/to/models/model_epoch_6.pt
    orilinx --fasta_path genome.fa --output_dir results
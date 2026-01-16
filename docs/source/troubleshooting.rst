.. _troubleshooting:

Troubleshooting
===============================

Installation Issues
-------------------

**"Command not found: orilinx"**

The ``orilinx`` command is not recognized after installation.

*Solutions:*

1. Verify you installed ORILINX correctly:
   
   .. code-block:: console
   
      cd /path/to/ORILINX
      pip install -e .

2. Make sure you're using the correct Python environment if you use conda or virtualenv:
   
   .. code-block:: console
   
      conda activate myenv  # or source venv/bin/activate
      pip install -e .

3. Try using the Python module directly instead:
   
   .. code-block:: console
   
      python -m orilinx --help

**"No module named transformers" or other missing packages**

Installation completed but ORILINX is missing dependencies.

*Solutions:*

1. Install all required packages:
   
   .. code-block:: console
   
      pip install torch transformers pysam pandas numpy peft

2. If you're using conda, install dependencies from conda-forge:
   
   .. code-block:: console
   
      conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
      conda install transformers pysam pandas numpy

3. Verify installation by importing modules:
   
   .. code-block:: console
   
      python -c "import torch, transformers, pysam, pandas; print('All modules imported successfully')"

**"Index file not found"**

ORILINX cannot find the FASTA index file (`.fai`).

*Solutions:*

1. Create the index file using samtools:
   
   .. code-block:: console
   
      samtools faidx your_genome.fa

2. Verify the index file was created:
   
   .. code-block:: console
   
      ls -la your_genome.fa*

   You should see both `your_genome.fa` and `your_genome.fa.fai` files.

3. Make sure the ``.fai`` file is in the same directory as the FASTA file.


Memory Issues
-------------

**Out of Memory (OOM) errors**

ORILINX crashes with "out of memory" or "CUDA out of memory" error.

*Solutions:*

1. Reduce batch size (decrease GPU memory usage):
   
   .. code-block:: console
   
      orilinx --fasta_path genome.fa \
              --output_dir results \
              --batch_size 32

2. Reduce number of workers (decrease CPU memory and parallelism):
   
   .. code-block:: console
   
      orilinx --fasta_path genome.fa \
              --output_dir results \
              --batch_size 32 \
              --num_workers 2

3. Use CPU instead of GPU (slower but uses system memory):
   
   .. code-block:: console
   
      CUDA_VISIBLE_DEVICES="" orilinx --fasta_path genome.fa \
                                       --output_dir results \
                                       --batch_size 16

**Memory limit exceeded on cluster**

Running on an HPC cluster with memory limits.

*Solutions:*

1. Request more memory in your job script:
   
   .. code-block:: bash
   
      #SBATCH --mem=32G
      orilinx --fasta_path genome.fa --output_dir results --batch_size 32

2. Use the conservative settings:
   
   .. code-block:: console
   
      orilinx --fasta_path genome.fa \
              --output_dir results \
              --batch_size 16 \
              --num_workers 2


Performance Issues
------------------

**Slow performance**

ORILINX is running but much slower than expected.

*Check if GPU is being used:*

.. code-block:: console

   python -c "import torch; print('GPU available:', torch.cuda.is_available()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

*If GPU is available but not being used:*

Try running with verbose output to see what device is being used:

.. code-block:: console

   orilinx --fasta_path genome.fa \
           --output_dir results \
           --verbose

*Solutions for slow performance:*

1. Increase batch size if you have GPU memory available:
   
   .. code-block:: console
   
      orilinx --fasta_path genome.fa \
              --output_dir results \
              --batch_size 128

2. Increase number of workers for data loading:
   
   .. code-block:: console
   
      orilinx --fasta_path genome.fa \
              --output_dir results \
              --num_workers 16


GPU and Computation Errors
---------------------------

**"Triton compilation error" or "CUDA error"**

ORILINX encounters GPU-related compilation or memory errors.

*Solutions:*

1. Use the ``--disable_flash`` flag to fall back to standard PyTorch attention:
   
   .. code-block:: console
   
      orilinx --fasta_path genome.fa \
              --output_dir results \
              --disable_flash

   This is slower but more stable and compatible with more GPUs.

2. Update your GPU drivers and CUDA:
   
   .. code-block:: console
   
      nvidia-smi  # Check current CUDA version
      # Update drivers from nvidia.com

3. Downgrade PyTorch if you have compatibility issues:
   
   .. code-block:: console
   
      pip install torch==2.0.1

**Model loading fails or crashes**

ORILINX cannot find or load the model files.

*Solutions:*

1. Check that your models directory exists and has the correct structure:
   
   .. code-block:: console
   
      ls -la models/
      ls -la models/DNABERT-2-117M-Flash/
      ls -la models/model_epoch_6.pt

2. Manually specify paths with environment variables:
   
   .. code-block:: console
   
      export ORILINX_DNABERT_PATH=/full/path/to/DNABERT-2-117M-Flash
      export ORILINX_MODEL=/full/path/to/model_epoch_6.pt
      orilinx --fasta_path genome.fa --output_dir results

3. Run with verbose output to see where ORILINX is searching:
   
   .. code-block:: console
   
      orilinx --fasta_path genome.fa --output_dir results --verbose


Data Input Issues
-----------------

**"Sequence not found in FASTA"**

ORILINX cannot find the specified chromosome or sequence name.

*Solutions:*

1. List all available sequences in your FASTA file:
   
   .. code-block:: console
   
      samtools idxstats genome.fa | cut -f1

2. Make sure you're using the correct sequence name format:
   
   .. code-block:: console

      # First part of FASTA file is:
      # >chr1
      # ATCGAATCGGATA...

      # Correct - use exact names from FASTA
      orilinx --fasta_path genome.fa --output_dir results --sequence_names chr1
      
      # Wrong - sequence name doesn't exist
      orilinx --fasta_path genome.fa --output_dir results --sequence_names chromosome1

3. If sequences have RefSeq accessions instead of chr names:
   
   .. code-block:: console
   
      # First check what names are in the FASTA
      samtools idxstats genome.fa | head
      
      # Then use those exact names
      orilinx --fasta_path genome.fa --output_dir results --sequence_names NC_000001.11

**"Range is too small" error**

Specified region is smaller than 2000 bp.

*Solution:*

Since ORILINX analyses 2000 bp windows, the minimum region size is 2000 bp. Expand your region:
   
   .. code-block:: console
   
      # Wrong - only 1000 bp
      orilinx --fasta_path genome.fa --sequence_names chr1:1000000-1001000
      
      # Correct - 2000 bp minimum
      orilinx --fasta_path genome.fa --sequence_names chr1:1000000-1002000


Output Issues
-------------

**Empty output files**

Output files are created but contain no data.

*Causes and solutions:*

1. **Too many 'N' bases in the region:** Regions with >5% 'N' bases are skipped by default:
   
   .. code-block:: console
   
      # Check for 'N' content
      samtools faidx genome.fa chr1:1000000-2000000 | grep -o 'N' | wc -l
      
      # Reduce strictness
      orilinx --fasta_path genome.fa \
              --output_dir results \
              --sequence_names chr1:1000000-2000000 \
              --max_N_frac 0.2

2. **Region is smaller than window:** The region must be at least 2000 bp.

3. **Analyse a different region** to verify the pipeline works.


Debugging and Getting Help
---------------------------

**Enable verbose output**

Get detailed information about what ORILINX is doing:

.. code-block:: console

   orilinx --fasta_path genome.fa \
           --output_dir results \
           --verbose

This shows:
- Which model files are being loaded
- What device (CPU/GPU) is being used
- Runtime settings and batch configuration

**Test with a small region**

Before running genome-wide, test with a small region to verify everything works:

.. code-block:: console

   orilinx --fasta_path genome.fa \
           --output_dir test_results \
           --sequence_names chr1:50000000-51000000 \
           --verbose

**Check system resources**

Monitor CPU and GPU usage while running:

.. code-block:: console

   # In a separate terminal
   nvidia-smi -l 1  # Update GPU stats every second (NVIDIA GPUs)
   
   # Or for CPU usage
   top

**Report issues**

If you encounter issues not covered here, raise a `GitHub issue <https://github.com/Pfuderer/ORILINX/issues>`_ and please provide:

1. The command you used:
   
   .. code-block:: console
   
      orilinx --fasta_path ... (your full command)

2. The verbose output:
   
   .. code-block:: console
   
      orilinx --fasta_path genome.fa --output_dir results --verbose

3. Your system information:
   
   .. code-block:: bash
   
      python --version
      python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
      nvidia-smi  # If using GPU
      samtools --version
      python -c "import transformers; print(transformers.__version__)"
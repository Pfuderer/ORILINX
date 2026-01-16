.. _examples:

Examples
===============================

The following is an example ORILINX workflow where we'll analyse the human genome in a few different ways.

Data acquisition
----------------

The only input data needed by ORILINX is a FASTA file. Download the T2T-CHM13v2.0 assembly, unzip it, and index it with samtools:

.. code-block:: console

   wget https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/analysis_set/chm13v2.0.fa.gz
   gunzip chm13v2.0.fa.gz
   samtools faidx chm13v2.0.fa


Analyzing genome-wide origin activity
---------------------------------------

To run ORILINX genome-wide on all primary chromosomes (chr1-chr22, chrX), use the default settings:

.. code-block:: console

   orilinx --fasta_path chm13v2.0.fa \
           --output_dir genome_wide_results \
           --stride 1000 \
           --batch_size 64 \
           --num_workers 8

This will generate separate bedGraph files for each chromosome in the output directory. The default stride of 1000 bp provides good coverage while avoiding excessive overlap between windows.


Analyzing multiple specific regions
------------------------------------

You can analyse multiple regions in a single run by specifying them comma-separated:

.. code-block:: console

   orilinx --fasta_path chm13v2.0.fa \
           --output_dir multi_region_results \
           --sequence_names chr1:1000000-2000000,chr8:128862888-128870405,chrX:100000000-101000000

This is useful when you want to focus on specific genomic regions of interest across different chromosomes.

Detecting the MYC origin
--------------------------

We'll run ORILINX on the MYC proto-oncogene on chromosome 8, and we can find the `location of it <https://www.ncbi.nlm.nih.gov/gene/4609>`_ on our assembly here. While we could run ORILINX on all of chromosome 8, we're going to save time and run it only on the MYC gene. The MYC gene is only about 7.5 kb and ORILINX considers 2 kb of sequence context, so we'll use a shorter stride than normal of 500 bp. 

.. code-block:: console

   orilinx --fasta_path chm13v2.0.fa \
           --output_dir myc_results \
           --stride 500 \
           --sequence_names chr8:128862888-128870405

Using different output formats
-------------------------------

ORILINX can output results in both bedGraph and CSV formats. Use the ``--output_csv`` flag to also generate CSV files:

.. code-block:: console

   orilinx --fasta_path chm13v2.0.fa \
           --output_dir results_with_csv \
           --output_csv \
           --sequence_names chr8:128862888-128870405

This will produce both ``.bedGraph`` and ``.csv`` files. The CSV format includes columns for chromosome, window start, window end, probability, and logit scores, which can be useful for downstream analysis in R or Python.


Tuning window parameters
------------------------

The stride parameter controls the spacing between windows. Smaller strides create more overlapping windows and provide finer resolution but take longer to compute:

.. code-block:: console

   # Fine resolution (short stride - slower)
   orilinx --fasta_path chm13v2.0.fa \
           --output_dir high_res_results \
           --stride 500 \
           --sequence_names chr8:128862888-128870405

   # Coarse resolution (long stride - faster)
   orilinx --fasta_path chm13v2.0.fa \
           --output_dir low_res_results \
           --stride 2000 \
           --sequence_names chr8:128862888-128870405

Note: The window size is fixed at 2000 bp (determined by the DNABERT-2 model) and cannot be changed.


Optimizing performance
----------------------

For large-scale analyses, you can tune the batch size and number of workers:

.. code-block:: console

   # High-throughput mode (increase batch size and workers)
   orilinx --fasta_path chm13v2.0.fa \
           --output_dir fast_results \
           --batch_size 128 \
           --num_workers 16 \
           --stride 1000

   # Memory-constrained mode (smaller batch size, fewer workers)
   orilinx --fasta_path chm13v2.0.fa \
           --output_dir conservative_results \
           --batch_size 32 \
           --num_workers 4 \
           --stride 1000

Increasing batch size and workers will speed up computation if your hardware supports it. Conversely, reduce these values if you encounter out-of-memory errors.
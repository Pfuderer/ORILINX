.. _workflows:

Workflow
===============================

The following is an example ORILINX workflow where we'll analyse the human genome in a few different ways.

Data acquisition
----------------

The only input data needed by ORILINX is a FASTA file. Download the T2T-CHM13v2.0 assembly, unzip it, and index it with samtools:

.. code-block:: console

   wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/009/914/755/GCF_009914755.1_T2T-CHM13v2.0/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna.gz
   gunzip GCF_009914755.1_T2T-CHM13v2.0_genomic.fna.gz
   samtools faidx GCF_009914755.1_T2T-CHM13v2.0_genomic.fna

Getting ORILINX
----------------

Pull the Singularity image:

.. code-block:: console

   singularity pull ORILINX.sif library://mboemo/orilinx/orilinx:1.0.0


Detecting the MYC origin
--------------------------

Download the T2T-CHM13v2.0 assembly, unzip it, and index it with samtools:

.. code-block:: console

   wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/009/914/755/GCF_009914755.1_T2T-CHM13v2.0/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna.gz
   gunzip GCF_009914755.1_T2T-CHM13v2.0_genomic.fna.gz
   samtools faidx GCF_009914755.1_T2T-CHM13v2.0_genomic.fna

We'll run ORILINX on the MYC protooncogene on chromosome 8, and we can find the `location of it <https://www.ncbi.nlm.nih.gov/gene/4609>`_ on our assembly here. While we could run ORILINX on all of chromosome 8, we're going to save time and run it only on the MYC gene. The MYC gene is only about 7.5 kb and ORILINX considers 2 kb of sequence context, so we'll use a shorter stride than normal of 500 bp. 

.. code-block:: console

   orilinx --fasta_path genomes/chm13v2.0.fa \
           --output_dir myc_orilinx \
           --stride 500 \
           --sequence_names NC_060932.1:128862888-128870405


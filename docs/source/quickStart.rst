.. _quickStart:

Quick Start
===========

Get ORILINX running in 5 minutes.


Installation
------------

Install from PyPI::

    pip install orilinx
    orilinx fetch_models

Or install from source::

    git clone https://github.com/Pfuderer/ORILINX.git
    cd ORILINX
    pip install -e .
    orilinx fetch_models

ORILINX includes a pre-trained model. The ``fetch_models`` command downloads the required weights (~900 MB).

Get a genome
------------
Download the human genome (hg38) from UCSC::

    wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
    gunzip hg38.fa.gz


Make a prediction
-----------------

Predict replication origins on a 4 Mb region of chromosome 8::

    orilinx --fasta_path hg38.fa --output_dir orilinx_results --sequence_names chr8:1000000-5000000

The results appear in a new ``orilinx_results/`` folder as ``chr8.bedGraph``.

That's it! The file contains shows the probability that each 2000 bp window in this region is a replication origin.


What next?
----------

- **Want more options?** See :ref:`installation` for all available flags
- **Need more examples?** Check :ref:`examples` for multi-region analysis, batch processing, and more
- **Troubleshooting?** See :ref:`troubleshooting` for common issues
- **Understanding output?** See :ref:`outputFormats` for bedGraph and CSV format details
- **Visualisation guide?** See :ref:`visualisation` for advanced plotting techniques
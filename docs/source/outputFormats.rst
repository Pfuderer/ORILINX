.. _outputFormats:

Output Format
=============

ORILINX produces prediction results in two formats: bedGraph (default) and CSV (optional). Each sequence (chromosome) is written to a separate output file.

bedGraph Format
---------------

The bedGraph format is the default output and is compatible with genome browsers like UCSC Genome Browser and IGV.

**Format specification:**

Each line contains four tab-separated fields:

.. code-block:: text

   chrom   start   end   score

Where:

- **chrom**: Sequence name (e.g., ``chr1``, ``NC_000001.11``)
- **start**: 0-based start position (inclusive)
- **end**: 0-based end position (exclusive)
- **score**: Prediction score (probability by default, logit if ``--score logit`` specified)

**Example output:**

.. code-block:: text

   chr8    500     1500    0.586426
   chr8    1501    2500    0.925781
   chr8    2501    3500    0.029205
   chr8    3501    4500    0.868652
   chr8    4501    5500    0.006096
   chr8    5501    6500    0.006123

**Window coordinate behavior:**

The output coordinate system depends on the stride parameter used:

When stride < 2000 bp (overlapping windows)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Windows are output as non-overlapping intervals to avoid gaps in genome coverage. Each interval is centered on the window's center position with width equal to the stride:

.. code-block:: text

   chr8    500     1500    0.586426
   chr8    1501    2500    0.925781
   chr8    2501    3500    0.029205

With stride=1000, each output interval spans 1000 bp (center ± 500 bp).

When stride >= 2000 bp (non-overlapping windows)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Windows are output using their full 2000 bp coordinates:

.. code-block:: text

   chr8    128862888   128864888   0.654321
   chr8    128864888   128866888   0.723456
   chr8    128866888   128868888   0.812345

**File naming:**

Output files are named after the sequence: ``{sequence_name}.bedGraph``

Examples:

- ``chr1.bedGraph`` - For a human chromosome
- ``NC_000001.11.bedGraph`` - For RefSeq accessions
- ``MT.bedGraph`` - For mitochondrial DNA


CSV Format
----------

The CSV format is optional and will be written when the ``--output_csv`` flag is used. It provides a more detailed record of all windows and predictions.

**Format specification:**

CSV files contain a header row followed by data rows with comma-separated values:

.. code-block:: text

   chrom,start,end,probability,logit

Where:

- **chrom**: Sequence name
- **start**: Window start position (0-based)
- **end**: Window end position (exclusive)
- **probability**: Sigmoid probability (0.0 to 1.0)
- **logit**: Raw logit score from the model (unbounded)

**Example output:**

.. code-block:: text

   chrom,start,end,probability,logit
   chr8,10000,12000,0.586426,0.343521
   chr8,11000,13000,0.925781,2.876543
   chr8,12000,14000,0.029205,-3.543210
   chr8,13000,15000,0.868652,1.987654
   chr8,14000,16000,0.006096,-5.098765
   chr8,15000,17000,0.006123,-5.087654

**Key differences from bedGraph:**

- Shows full window coordinates (always in 2000 bp window format)
- Includes both probability and logit scores
- Header row for column identification
- Easier to parse in R, Python, etc.

**File naming:**

Output files are named after the sequence: ``{sequence_name}.csv``

Examples:

- ``chr1.csv``
- ``NC_000001.11.csv``
- ``MT.csv``


Generating both formats
-----------------------

ORILINX will always give an output in bedGraph format. Use the ``--output_csv`` flag to, in addition, generate CSV files:

.. code-block:: console

   orilinx --fasta_path genome.fa \
           --output_dir results \
           --output_csv \
           --sequence_names chr8:128862888-128870405

This will produce:

- ``results/chr8.bedGraph`` - For genome browser visualisation
- ``results/chr8.csv`` - For downstream analysis


Interpreting the scores
-----------------------

**Probability scores (default):**

- Range: 0.0 to 1.0
- Interpretation: Likelihood that the region contains a replication origin
- 0.5: Equal probability of being an origin or non-origin
- > 0.7: High confidence origin prediction
- < 0.3: Low confidence origin prediction

**Logit scores:**

- Range: Unbounded (typically -5 to +5)
- Interpretation: Log-odds ratio of being an origin
- 0: No preference (equivalent to 0.5 probability)
- Positive: Likely origin
- Negative: Likely non-origin
- Use when you need raw model confidence

**Choosing a score type:**

Use probability (default) for:
- Genome browser visualisation
- Intuitive interpretation
- Most downstream analyses

Use logit (``--score logit``) for:
- Statistical analyses requiring raw model output
- Machine learning pipelines
- Detailed statistical modeling


File organization
-----------------

For large analyses, bedGraph and CSV files are organised by sequence:

.. code-block:: text

   output_dir/
   ├── chr1.bedGraph
   ├── chr1.csv
   ├── chr2.bedGraph
   ├── chr2.csv
   ├── chr3.bedGraph
   ├── chr3.csv
   └── ...
   └── chrX.bedGraph
   └── chrX.csv

You can process all files together:

.. code-block:: bash

   # Concatenate all bedGraphs into single file
   cat output_dir/*.bedGraph > all_origins.bedGraph
   
   # Concatenate all CSVs (preserving header from first file)
   (head -1 output_dir/chr1.csv; tail -q -n +2 output_dir/*.csv) > all_origins.csv
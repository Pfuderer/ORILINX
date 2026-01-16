.. _visualisation:

Visualisation and Analysis
==========================

ORILINX produces bedGraph output that can be visualised in genome browsers for intuitive exploration of predicted replication origins. This section covers various tools and methods for visualising results.

Overview
--------

bedGraph files from ORILINX are compatible with most genome browsers including:

- **UCSC Genome Browser** - Web-based, no installation needed
- **IGV (Integrative Genomics Viewer)** - Desktop application with advanced features
- **JBrowse** - Lightweight web browser for genomic data
- **Gviz (R)** or **pyBigWig (Python)** - Programmatic visualisation

The basic workflow is:

1. Run ORILINX to generate bedGraph files
2. Load them into a genome browser
3. Compare with other genomic features (genes, regulatory elements, etc.)
4. Interpret results in biological context


UCSC Genome Browser
--------------------

The UCSC Genome Browser is a free, web-based tool that requires no installation.

**Basic Usage**

1. Generate your ORILINX results:
   
   .. code-block:: console
   
      orilinx --fasta_path hg38.fa --output_dir results --sequence_names chr8

2. Go to `https://genome.ucsc.edu/ <https://genome.ucsc.edu/>`_

3. Select your genome (e.g., "Human" and "Dec. 2013 (GRCh38/hg38)")

4. Navigate to your region of interest (e.g., type "chr8:128862888-128870405" in the search box)

5. Click "Add Custom Tracks" and upload your bedGraph file:
   
   - Copy the contents of ``results/chr8.bedGraph`` or upload the file directly
   - Set the display height and colour
   - Click "Submit"

6. Your ORILINX scores will appear as a histogram track

**Customizing Track Appearance**

You can customize how your track appears by adding a header to your bedGraph file:

.. code-block:: text

   track name="ORILINX Origins" description="Predicted replication origins" colour=50,50,200 viewLimits=0:1

Then prepend this to your bedGraph file:

.. code-block:: bash

   echo 'track name="ORILINX Origins" description="Predicted replication origins" colour=50,50,200 viewLimits=0:1' > formatted.bedGraph
   cat results/chr8.bedGraph >> formatted.bedGraph

Then upload ``formatted.bedGraph`` to UCSC.

**Converting to BigWig for faster loading**

For large files, convert bedGraph to BigWig format for faster loading:

.. code-block:: bash

   # Download BigWig tools if needed
   wget http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/bedGraphToBigWig
   chmod +x bedGraphToBigWig
   
   # Obtain chrom sizes
   curl https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes > hg38.chrom.sizes
   
   # Convert
   ./bedGraphToBigWig results/chr8.bedGraph hg38.chrom.sizes results/chr8.bw

Then upload the ``.bw`` file to UCSC instead of the bedGraph.


IGV (Integrative Genomics Viewer)
---------------------------------

IGV is a desktop application that offers more control and advanced features than web browsers.

**Installation**

1. Download from `http://software.broadinstitute.org/software/igv/ <http://software.broadinstitute.org/software/igv/>`_

2. Install for your operating system (Mac, Windows, Linux)

3. Launch IGV

**Loading ORILINX Data**

1. Open IGV and select your genome (File → Genomes → Load Genome)

2. Load your bedGraph file:
   
   - File → Load from File
   - Select ``results/chr8.bedGraph``

3. Navigate to your region of interest using the search box

4. IGV will display your ORILINX scores as a bar graph

**Tips for IGV**

- **Zoom in/out**: Use the zoom controls or scroll wheel
- **Compare tracks**: Load multiple bedGraph files simultaneously to compare regions
- **Overlay with annotations**: Load gene annotations, ChIP-seq, or other genomic data for context
- **Export images**: Right-click on tracks to save publication-quality figures
- **Coverage view**: Change display mode to "expanded" to see individual windows

**Coloured tracks**

Create a coloured bedGraph based on score thresholds:

.. code-block:: bash

   # High confidence origins (score > 0.7) in red
   # Medium confidence (0.3-0.7) in yellow  
   # Low confidence (< 0.3) in blue
   awk 'BEGIN {FS=OFS="\t"} 
        {if ($4 > 0.7) colour="255,0,0"; 
         else if ($4 > 0.3) colour="255,255,0"; 
         else colour="0,0,255"; 
         print $1, $2, $3, $4, colour}' results/chr8.bedGraph > coloured.bedGraph

Then load ``coloured.bedGraph`` in IGV.


JBrowse
-------

JBrowse is a lightweight, embeddable genome browser suitable for web-based visualisation.

**Using JBrowse Online**

1. Visit `https://jbrowse.org/jbrowse/ <https://jbrowse.org/jbrowse/>`_

2. Select your reference genome

3. Add tracks:
   
   - Click "Add Track" 
   - Paste the URL to your bedGraph file or upload it directly

4. Navigate to your region to visualize

**Self-hosted JBrowse**

For more control, you can host JBrowse on your own server:

.. code-block:: bash

   # Install JBrowse (see documentation)
   wget https://github.com/GMOD/jbrowse/releases/download/1.11.6/JBrowse-1.11.6.zip
   unzip JBrowse-1.11.6.zip
   
   # Configure data directory
   cd JBrowse-1.11.6
   ./bin/prepare-refseqs.pl --fasta hg38.fa
   
   # Add ORILINX track
   ./bin/flatfile-to-json.pl --gff results/chr8.bedGraph --type bedGraph --trackType wig --out data


In Python
---------

Use Python for programmatic visualisation and analysis of ORILINX results.

**Basic plot with Matplotlib**

.. code-block:: python

   import pandas as pd
   import matplotlib.pyplot as plt
   
   # Read CSV output
   df = pd.read_csv('results/chr8.csv')
   
   # Plot probability scores
   plt.figure(figsize=(14, 4))
   plt.plot(df['start'], df['probability'], linewidth=0.5)
   plt.fill_between(df['start'], df['probability'], alpha=0.3)
   plt.xlabel('Genomic Position (bp)')
   plt.ylabel('Origin Probability')
   plt.title('ORILINX Predictions - Chr8')
   plt.tight_layout()
   plt.savefig('origins_plot.png', dpi=300)
   plt.show()

**Finding high-confidence origins**

.. code-block:: python

   import pandas as pd
   
   df = pd.read_csv('results/chr8.csv')
   
   # Filter for high-confidence origins (>0.7 probability)
   origins = df[df['probability'] > 0.7]
   
   print(f"Found {len(origins)} high-confidence origins")
   print(origins[['start', 'end', 'probability']])
   
   # Export for further analysis
   origins.to_csv('high_confidence_origins.csv', index=False)

**Interactive visualisation with Plotly**

.. code-block:: python

   import pandas as pd
   import plotly.graph_objects as go
   
   df = pd.read_csv('results/chr8.csv')
   
   fig = go.Figure()
   
   fig.add_trace(go.Scatter(
       x=df['start'],
       y=df['probability'],
       mode='lines',
       name='ORILINX Probability',
       fill='tozeroy'
   ))
   
   # Add threshold lines
   fig.add_hline(y=0.7, line_dash="dash", line_color="red",
                 annotation_text="High confidence", annotation_position="right")
   fig.add_hline(y=0.3, line_dash="dash", line_color="orange",
                 annotation_text="Low confidence", annotation_position="right")
   
   fig.update_layout(
       title='ORILINX Predictions with Confidence Thresholds',
       xaxis_title='Genomic Position (bp)',
       yaxis_title='Origin Probability',
       hovermode='x unified'
   )
   
   fig.show()
   fig.write_html('origins_interactive.html')

**Comparison of multiple regions**

.. code-block:: python

   import pandas as pd
   import matplotlib.pyplot as plt
   
   # Load multiple regions
   regions = {}
   for region in ['chr1', 'chr8', 'chrX']:
       regions[region] = pd.read_csv(f'results/{region}.csv')
   
   # Plot comparison
   fig, axes = plt.subplots(len(regions), 1, figsize=(14, 3*len(regions)))
   
   for idx, (region, df) in enumerate(regions.items()):
       axes[idx].plot(df['start'], df['probability'], linewidth=0.5)
       axes[idx].fill_between(df['start'], df['probability'], alpha=0.3)
       axes[idx].set_title(f'{region} ORILINX Predictions')
       axes[idx].set_ylabel('Probability')
       if idx == len(regions) - 1:
           axes[idx].set_xlabel('Genomic Position (bp)')
   
   plt.tight_layout()
   plt.savefig('multiregion_comparison.png', dpi=300)
   plt.show()


In R
----

Use R for statistical analysis and publication-quality figures.

**Basic plot with ggplot2**

.. code-block:: r

   library(ggplot2)
   library(dplyr)
   
   # Read CSV output
   df <- read.csv('results/chr8.csv')
   
   # Create plot
   ggplot(df, aes(x=start, y=probability)) +
     geom_line(size=0.2) +
     geom_area(alpha=0.3) +
     theme_minimal() +
     labs(
       title = 'ORILINX Predictions - Chr8',
       x = 'Genomic Position (bp)',
       y = 'Origin Probability'
     ) +
     theme(text=element_text(size=12))
   
   ggsave('origins_plot.png', width=14, height=4, dpi=300)

**Finding and annotating peaks**

.. code-block:: r

   library(dplyr)
   library(ggplot2)
   
   df <- read.csv('results/chr8.csv')
   
   # Find peak origins (local maxima)
   df <- df %>%
     mutate(
       is_peak = probability > 0.7,
       peak_id = cumsum(c(TRUE, diff(is_peak) != 0)) * is_peak
     )
   
   peaks <- df %>%
     filter(is_peak) %>%
     group_by(peak_id) %>%
     summarise(
       peak_start = min(start),
       peak_end = max(end),
       peak_probability = max(probability),
       .groups = 'drop'
     )
   
   print(peaks)
   write.csv(peaks, 'predicted_origins.csv', row.names=FALSE)

**Genome browser-style visualisation with Gviz**

.. code-block:: r

   library(Gviz)
   library(GenomicRanges)
   
   # Read ORILINX data
   df <- read.csv('results/chr8.csv')
   
   # Convert to GRanges
   gr <- GRanges(
     seqnames = df$chrom,
     ranges = IRanges(start = df$start, end = df$end),
     score = df$probability
   )
   
   # Create DataTrack
   dtrack <- DataTrack(
     range = gr,
     name = "ORILINX",
     type = "histogram",
     col.histogram = "steelblue",
     fill.histogram = "steelblue"
   )
   
   # Plot
   plotTracks(dtrack, from=128862888, to=128870405, chromosome="chr8")


Combining with other genomic features
--------------------------------------

For biological interpretation, visualise ORILINX results alongside:

- Gene annotations
- ChIP-seq peaks
- Copy number variation
- Evolutionary conservation
- Chromatin accessibility

**Example: Adding genes to your plot**

.. code-block:: python

   import pandas as pd
   import matplotlib.pyplot as plt
   import matplotlib.patches as mpatches
   
   # Load ORILINX results and gene annotations
   origins = pd.read_csv('results/chr8.csv')
   genes = pd.read_csv('gene_annotations.csv')  # Your gene file
   
   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
   
   # Plot ORILINX scores
   ax1.plot(origins['start'], origins['probability'], linewidth=0.5)
   ax1.fill_between(origins['start'], origins['probability'], alpha=0.3)
   ax1.set_ylabel('ORILINX Probability')
   ax1.set_title('Chr8 - Origins and Gene Structure')
   
   # Plot genes
   for idx, gene in genes.iterrows():
       ax2.barh(0, gene['end']-gene['start'], 
               left=gene['start'], height=0.5, label=gene['name'])
   ax2.set_ylabel('Genes')
   ax2.set_xlabel('Genomic Position (bp)')
   
   plt.tight_layout()
   plt.savefig('origins_with_genes.png', dpi=300)
   plt.show()
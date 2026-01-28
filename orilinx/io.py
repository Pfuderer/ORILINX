def write_csv_windows(df, path):
    """Write window data to CSV with columns: chromosome, start, end, probability, logit."""
    df_out = df[["chrom", "start", "end", "prob", "logit"]].copy()
    df_out.to_csv(path, sep=",", header=True, index=False)


def write_bedgraph_center(df, path, value="logit", stride=1000, region_start=None, region_end=None):
    """Write bedgraph output with proper interval handling for overlapping windows.
    
    For stride < 2000 (overlapping windows), outputs non-overlapping intervals centered
    on each window with width equal to stride. For stride >= 2000, outputs full window
    coordinates.
    """
    with open(path, "w") as fh:
        if stride < 2000:
            # For overlapping windows, output non-overlapping intervals
            # Each interval is centered on the window center with width equal to stride
            prev_end = None
            for idx, (_, r) in enumerate(df.iterrows()):
                c = r["chrom"]
                v = float(r[value])
                center = int(r["center"])
                half_stride = stride / 2.0
                interval_start = int(center - half_stride)
                interval_end = int(center + half_stride)
                
                # Adjust start if it would overlap with previous interval
                if prev_end is not None and interval_start <= prev_end:
                    interval_start = prev_end + 1
                
                # Adjust first interval to region start
                if idx == 0 and region_start is not None:
                    interval_start = region_start
                
                # Adjust last interval to region end
                if idx == len(df) - 1 and region_end is not None:
                    interval_end = region_end
                
                fh.write(f"{c}\t{interval_start}\t{interval_end}\t{v:.6f}\n")
                prev_end = interval_end
        else:
            # Use full window coordinates when stride >= window length
            for _, r in df.iterrows():
                c = r["chrom"]
                v = float(r[value])
                start = int(r["start"])
                end = int(r["end"])
                fh.write(f"{c}\t{start}\t{end}\t{v:.6f}\n")

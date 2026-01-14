import os
import csv
import random
import numpy as np
import pysam
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import time

class OriginClassificationDataset(Dataset):
    def __init__(self, origins_csv_path, fasta_path, negative_samples_path, sequence_length=2048, neg_to_pos_ratio=1.0, seed=42, model_name=""):
        super().__init__()
        
        print("[Dataset] Initializing OriginClassificationDataset...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
        self.fasta_path = fasta_path
        self.sequence_length = sequence_length
        self.records = []
        
        print(f"[Dataset] Opening indexed FASTA file at {fasta_path}...")
        self.fasta_handle = pysam.FastaFile(fasta_path)
        print("[Dataset] FASTA file opened successfully.")
        
        positive_samples = self._load_positive_samples(origins_csv_path)
        print(f"[Dataset] Loaded {len(positive_samples)} positive origin samples.")

        if os.path.exists(negative_samples_path):
            print(f"[Dataset] Loading negative samples from {negative_samples_path}...")
            negative_samples = self._load_negative_samples(negative_samples_path)
        else:
            print(f"[Dataset] Generating new GC-matched negative samples...")
            num_negative = int(len(positive_samples) * neg_to_pos_ratio)
            # Pass the fasta_handle to the negative sample generation function
            negative_samples = self._generate_negative_samples(num_negative, positive_samples, seed)
            if negative_samples:
                self._save_negative_samples(negative_samples, negative_samples_path)
                print(f"[Dataset] Saved {len(negative_samples)} negative samples to {negative_samples_path}.")
        
        for chrom, start, end in positive_samples:
            self.records.append({"chrom": chrom, "start": start, "end": end, "label": 1})
        for chrom, start, end in negative_samples:
            self.records.append({"chrom": chrom, "start": start, "end": end, "label": 0})
            
        print(f"[Dataset] Total samples: {len(self.records)} ({len(positive_samples)} positive, {len(negative_samples)} negative).")

    '''
    def _calculate_gc_content(self, seq):
        """Helper function to calculate GC content of a DNA sequence."""
        if not seq:
            return 0.0
        gc_count = seq.count('G') + seq.count('C')
        return gc_count / len(seq)
    '''
    def _load_positive_samples(self, csv_path):
        samples = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                chrom = row["chromosome"]
                start = int(row["start"])
                end = int(row["end"])

                # ensure start/end are valid for pysam
                if start < 0:
                    start = 0

                chrom_len = self.fasta_handle.get_reference_length(chrom)
                if end > chrom_len:
                    end = chrom_len

                if end <= start:
                    continue

                samples.append((chrom, start, end))
        return samples

    def _generate_negative_samples(self, num_negative, positive_samples, seed):
        random.seed(seed)
        np.random.seed(seed)
        
        '''
        # --- NEW: Calculate target GC content from positive samples ---
        print("[Dataset] Calculating GC content of positive samples...")
        positive_gc_contents = []
        for chrom, start, end in positive_samples:
            seq = self.fasta_handle.fetch(chrom, start, end).upper()
            positive_gc_contents.append(self._calculate_gc_content(seq))
        
        target_median_gc = np.median(positive_gc_contents)
        gc_tolerance_below = 0.05  # Acceptable deviation from the median
        gc_tolerance_above = 0.1
        gc_min = target_median_gc - gc_tolerance_below
        gc_max = target_median_gc + gc_tolerance_above
        print(f"[Dataset] Target Median GC: {target_median_gc:.4f} (Acceptable Range: {gc_min:.4f} - {gc_max:.4f})")
        # --- END NEW ---
        '''
        print("[Dataset] Getting chromosome lengths from FASTA index...")
        chrom_lengths = {name: length for name, length in zip(self.fasta_handle.references, self.fasta_handle.lengths)}
        
        positive_intervals = {}
        for chrom, start, end in positive_samples:
            if chrom not in positive_intervals:
                positive_intervals[chrom] = []
            positive_intervals[chrom].append((start, end))

        negative_samples = []
        allowed_chroms = {
        'chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
        'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15',
        'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22',
        'chrX'}
        chrom_list = [c for c in chrom_lengths.keys() if c in allowed_chroms]
        print(f"[Dataset] Found {len(chrom_list)} matching chromosomes for negative sampling.")
        if not chrom_list:
            raise ValueError("No matching chromosomes found. Cannot generate negative samples.")

        attempts = 0
        accepted_count = 0
        last_print_time = time.time()
        
        print("[Dataset] Starting negative sample generation loop...")
        while accepted_count < num_negative:
            attempts += 1
            chrom = random.choice(chrom_list)
            max_start = chrom_lengths[chrom] - self.sequence_length
            if max_start <= 0: continue

            start = random.randint(0, max_start)
            end = start + self.sequence_length
            
            # --- Check for overlap with positive samples ---
            is_overlap = False
            if chrom in positive_intervals:
                for pos_start, pos_end in positive_intervals[chrom]:
                    if max(start, pos_start) < min(end, pos_end):
                        is_overlap = True
                        break
            if is_overlap:
                continue
            
            
            # --- NEW: GC content matching logic ---
            candidate_seq = self.fasta_handle.fetch(chrom, start, end).upper()
           
            # --- NEW: Reject sequences with too many 'N's ---
            if candidate_seq.count('N') > 10:
                continue
            else:
                accepted_count += 1 
            '''
            candidate_gc = self._calculate_gc_content(candidate_seq)
            
            if gc_min <= candidate_gc <= gc_max:
                # If GC content is within the tolerance window, accept the sample
                negative_samples.append((chrom, start, end))
                accepted_count += 1 
            '''
            negative_samples.append((chrom, start, end))
            # Print a status update every 5 seconds or a certain number of attempts
            current_time = time.time()
            if current_time - last_print_time > 5 or attempts % 100000 == 0:
                print(f"[Dataset] Progress: {accepted_count}/{num_negative} samples found after {attempts} attempts...")
                last_print_time = current_time
        
        print(f"[Dataset] Finished generation. Total attempts: {attempts}. Acceptance rate: {accepted_count/attempts:.2%}")
        return negative_samples
        
    def _load_negative_samples(self, csv_path):
        samples = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            for row in reader:
                samples.append((row[0], int(row[1]), int(row[2])))
        return samples

    def _save_negative_samples(self, samples, csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['chromosome', 'start', 'end'])
            for chrom, start, end in samples:
                writer.writerow([chrom, start, end])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        chrom, start, end, label = record["chrom"], record["start"], record["end"], record["label"]
        
        seq = self.fasta_handle.fetch(chrom, start, end).upper()
        
        inputs = self.tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.sequence_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "label": torch.tensor(label, dtype=torch.float)
        }
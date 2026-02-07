#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Convert VCF regions to labeled CSV files with sample_type, using Perl for sequence extraction

import os
import subprocess
import time
import re
from multiprocessing import Pool
from tqdm import tqdm
import argparse


def find_vcf(region, gene, vcf_dir):
    chrom, pos_range, strand = region.split(":")
    start, end = pos_range.split("-")
    strand_tag = "forward" if strand == "+" else "reverse"
    fname = f"{chrom}_{start}_{end}_{strand_tag}_{gene}.vcf"
    vcf_path = os.path.join(vcf_dir, fname)
    if os.path.exists(vcf_path):
        return vcf_path
    print(f"WARNING: cannot find vcf file: {fname}")
    return None


def parse_record(record):
    pattern = r'([^:]+):(\d+-\d+):([+-]):([mnMN]):([ACGTNacgtn]+):"?([\d,]+)"?'
    match = re.match(pattern, record)
    if match:
        chrom, pos_range, strand, hap_type, seq, positions = match.groups()
        positions_cleaned = positions.replace(',', ';')
        return {'chrom': chrom, 'pos_range': pos_range, 'strand': strand,
                'hap_type': hap_type, 'seq': seq, 'positions': positions_cleaned}
    return None


def process_region(task):
    region, gene, vcf_dir, fasta, perl_script, sample_type_dict, out_dir = task
    gene = gene if gene else "unknown"

    vcf_file = find_vcf(region, gene, vcf_dir)
    if vcf_file is None:
        return None

    chrom, pos_range, strand = region.split(":")
    start, end = pos_range.split("-")
    strand_tag = "forward" if strand == "+" else "reverse"
    out_file = os.path.join(out_dir, f"{chrom}_{start}_{end}_{strand_tag}_{gene}.label.csv")

    try:
        result = subprocess.run(
            ["perl", perl_script, vcf_file, region, fasta, "stdout"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        perl_output = result.stdout.splitlines()
    except subprocess.CalledProcessError as e:
        print(f"Failed {region} ({gene}): {e}")
        return None

    columns = ['sample', 'sample_type', 'hap1_seq', 'hap1_pos', 'hap2_seq', 'hap2_pos']
    out_lines = []

    for line in perl_output:
        line = line.strip()
        if not line:
            continue

        records = []
        current = []
        in_quotes = False
        for c in line:
            if c == '"':
                in_quotes = not in_quotes
            elif c == ',' and not in_quotes:
                records.append(''.join(current))
                current = []
                continue
            current.append(c)
        if current:
            records.append(''.join(current))

        sample_id = records[0]
        sample_type = sample_type_dict.get(sample_id, '')

        parsed_records = []
        for i in range(1, len(records), 2):
            if i + 1 < len(records):
                m_rec = parse_record(records[i])
                n_rec = parse_record(records[i + 1])
                if m_rec and n_rec and m_rec['chrom'] == n_rec['chrom'] and m_rec['pos_range'] == n_rec['pos_range']:
                    parsed_records.append({
                        'm_seq': m_rec['seq'],
                        'm_pos': m_rec['positions'],
                        'n_seq': n_rec['seq'],
                        'n_pos': n_rec['positions']
                    })

        if parsed_records:
            rec = parsed_records[0]
            out_row = [sample_id, sample_type, rec['m_seq'], rec['m_pos'], rec['n_seq'], rec['n_pos']]
        else:
            out_row = [sample_id, sample_type, '', '', '', '']

        out_lines.append(out_row)

    with open(out_file, 'w') as f:
        f.write(','.join(columns) + '\n')
        for row in out_lines:
            f.write(','.join(row) + '\n')

    return out_file


def main():

    parser = argparse.ArgumentParser(description="VCF to labeled CSV ")
    parser.add_argument("--region-file", required=True)
    parser.add_argument("--vcf-dir", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--perl-script", required=True)
    parser.add_argument("--sample-type-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tasks_list = []
    with open(args.region_file) as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            region = parts[0]
            gene = parts[1] if len(parts) >= 2 else "unknown"
            tasks_list.append((region, gene))

    print(f"Loaded {len(tasks_list)} regions")

    sample_type_dict = {}
    with open(args.sample_type_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('/'):
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                sample_type_dict[parts[0]] = parts[1]

    tasks = [
        (region, gene, args.vcf_dir, args.fasta, args.perl_script, sample_type_dict, args.out_dir)
        for region, gene in tasks_list
    ]

    with tqdm(total=len(tasks), desc="Processing regions", ncols=120) as pbar:
        with Pool(processes=args.max_workers) as pool:
            for out_file in pool.imap_unordered(process_region, tasks):
                pbar.update(1)

    print("All tasks completed!")


if __name__ == "__main__":
    main()

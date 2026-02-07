#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Extract specified regions from a VCF file into separate VCFs using bcftools

import os
import subprocess
from multiprocessing import Pool
import argparse

# Generate a VCF filename based on region and gene
def sanitize_filename(region, gene):
    chrom, pos_range, strand = region.split(":")
    start, end = pos_range.split("-")
    strand_str = "forward" if strand == "+" else "reverse"
    return f"{chrom}_{start}_{end}_{strand_str}_{gene}.vcf"

#extract specified regions from input VCF file
def extract_region_vcf(region, gene, vcf_file, out_dir):
    out_file = os.path.join(out_dir, sanitize_filename(region, gene))
    chrom, pos_range, _ = region.split(":")
    region_for_bcftools = f"{chrom}:{pos_range}"

    cmd = [
        "bcftools", "view",
        "-r", region_for_bcftools,
        vcf_file,
        "-Ov",
        "-o", out_file
    ]

    print(f">>> {region} ({gene}) -> {out_file}")
    subprocess.run(cmd, check=True)

#parallel processing of multiple regions
def run_task(task):
    region, gene, vcf_file, out_dir = task
    extract_region_vcf(region, gene, vcf_file, out_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Extract VCF regions using bcftools"
    )

    parser.add_argument(
        "--region-file",
        required=True,
        help="Region list file (chr:start-end:strand gene)"
    )

    parser.add_argument(
        "--vcf",
        required=True,
        help="Input VCF file (.vcf or .vcf.gz)"
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory"
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Number of parallel workers"
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tasks = []
    with open(args.region_file) as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.split()
            region = parts[0]
            gene = parts[1] if len(parts) >= 2 else "unknown"

            tasks.append(
                (region, gene, args.vcf, args.out_dir)
            )

    print(f"Loaded {len(tasks)} regions")

    with Pool(args.max_workers) as pool:
        pool.map(run_task, tasks)


if __name__ == "__main__":
    main()

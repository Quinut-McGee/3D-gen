#!/usr/bin/env python3
"""
Parse TRELLIS monitoring data and extract key statistics
"""
import re
import statistics
from datetime import datetime

# Extract gaussian counts from recent generations
log_file = "/home/kobe/.pm2/logs/gen-worker-1-error.log"

with open(log_file, 'r') as f:
    lines = f.readlines()

# Find TRELLIS generations (after 02:58 restart)
trellis_gens = []
recent_lines = [l for l in lines if '2025-11-04 02:58' in l or '2025-11-04 03:' in l]

for i, line in enumerate(recent_lines):
    if "Gaussians:" in line and "File size:" in line:
        # Extract gaussian count
        match = re.search(r'Gaussians: ([\d,]+)', line)
        if match:
            count_str = match.group(1).replace(',', '')
            count = int(count_str)

            # Extract file size
            size_match = re.search(r'File size: ([\d.]+) MB', line)
            size_mb = float(size_match.group(1)) if size_match else 0

            # Extract timestamp
            time_match = re.search(r'(\d{2}:\d{2}:\d{2})', line)
            timestamp = time_match.group(1) if time_match else "unknown"

            # Look backward for the prompt
            prompt = "unknown"
            for j in range(max(0, i-50), i):
                if "Detected TEXT-TO-3D task:" in recent_lines[j]:
                    prompt_match = re.search(r"'(.+)'", recent_lines[j])
                    if prompt_match:
                        prompt = prompt_match.group(1)

            trellis_gens.append({
                'timestamp': timestamp,
                'gaussians': count,
                'size_mb': size_mb,
                'prompt': prompt,
                'prompt_length': len(prompt.split())
            })

# Calculate statistics
if trellis_gens:
    gaussian_counts = [g['gaussians'] for g in trellis_gens]
    file_sizes = [g['size_mb'] for g in trellis_gens]

    print("=" * 80)
    print("TRELLIS MONITORING REPORT")
    print("=" * 80)
    print(f"\nTotal TRELLIS Generations: {len(trellis_gens)}")
    print(f"\nGaussian Count Statistics:")
    print(f"  Min:     {min(gaussian_counts):>10,} gaussians")
    print(f"  Max:     {max(gaussian_counts):>10,} gaussians ⭐")
    print(f"  Mean:    {int(statistics.mean(gaussian_counts)):>10,} gaussians")
    print(f"  Median:  {int(statistics.median(gaussian_counts)):>10,} gaussians")
    print(f"  StdDev:  {int(statistics.stdev(gaussian_counts)) if len(gaussian_counts) > 1 else 0:>10,}")

    print(f"\nFile Size Statistics:")
    print(f"  Min:     {min(file_sizes):>10.1f} MB")
    print(f"  Max:     {max(file_sizes):>10.1f} MB ⭐")
    print(f"  Mean:    {statistics.mean(file_sizes):>10.1f} MB")
    print(f"  Median:  {statistics.median(file_sizes):>10.1f} MB")

    # Categorize by gaussian density
    low_density = [g for g in trellis_gens if g['gaussians'] < 150000]
    med_density = [g for g in trellis_gens if 150000 <= g['gaussians'] < 400000]
    high_density = [g for g in trellis_gens if g['gaussians'] >= 400000]

    print(f"\nDensity Distribution:")
    print(f"  Low  (<150K):  {len(low_density):2} generations ({len(low_density)/len(trellis_gens)*100:.1f}%)")
    print(f"  Med  (150-400K): {len(med_density):2} generations ({len(med_density)/len(trellis_gens)*100:.1f}%)")
    print(f"  High (>400K):  {len(high_density):2} generations ({len(high_density)/len(trellis_gens)*100:.1f}%)")

    print(f"\n" + "=" * 80)
    print("INDIVIDUAL GENERATIONS:")
    print("=" * 80)
    for i, gen in enumerate(trellis_gens, 1):
        density = "HIGH" if gen['gaussians'] >= 400000 else "MED" if gen['gaussians'] >= 150000 else "LOW"
        print(f"{i:2}. [{gen['timestamp']}] {gen['gaussians']:>8,} gaussians ({gen['size_mb']:>5.1f} MB) - {density}")
        print(f"    Prompt ({gen['prompt_length']} words): {gen['prompt'][:70]}")

    print(f"\n" + "=" * 80)
else:
    print("No TRELLIS generations found after restart")

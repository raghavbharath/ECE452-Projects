# Raghav Bharathan
# ECE 452
# Project 2
# Professor Shaahin Angizi

import os
import re
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt

test_sizes  = [128, 256, 512]
block_sizes = [16, 32, 64, 128]

# all_times[block_size] = { size: [numpy_times, python_times] }
all_times = {bs: {} for bs in block_sizes}

for bs in block_sizes:
    for size in test_sizes:
        print(f'Running {size}x{size} block={bs}...')
        result = subprocess.run(
            [sys.executable, 'gemm-tiled.py', '-s', f'{size}', '-i', '3', '-b', f'{bs}'],
            capture_output=True,
            text=True
        )

        numpy_times  = []
        python_times = []

        for line in result.stdout.strip().split('\n'):
            match = re.search(r'NumPy:\s+([\d.]+)s \| Python:\s+([\d.]+)s', line)
            if match:
                numpy_times.append(float(match.group(1)))
                python_times.append(float(match.group(2)))

        all_times[bs][size] = [numpy_times, python_times]
        print(f'  Done. Got {len(numpy_times)} iterations.')

# ── Plot 1: Python tiled — execution time vs matrix size per block size ──
os.makedirs('plots/CPU_Tiled', exist_ok=True)

plt.figure(figsize=(10, 6))
for bs in block_sizes:
    avgs = [sum(all_times[bs][s][1]) / len(all_times[bs][s][1]) for s in test_sizes]
    plt.plot(test_sizes, avgs, marker='o', label=f'block={bs}')

plt.xlabel('Matrix Size')
plt.ylabel('Avg Time (s)')
plt.title('Tiled GEMM (Python): Execution Time vs Matrix Size by Block Size')
plt.xticks(test_sizes, [f'{s}x{s}' for s in test_sizes])
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('plots/CPU_Tiled/tiled_python_by_block.png')
print('Plot saved: plots/CPU_Tiled/tiled_python_by_block.png')

# ── Plot 2: NumPy tiled — same thing ──
plt.figure(figsize=(10, 6))
for bs in block_sizes:
    avgs = [sum(all_times[bs][s][0]) / len(all_times[bs][s][0]) for s in test_sizes]
    plt.plot(test_sizes, avgs, marker='o', label=f'block={bs}')

plt.xlabel('Matrix Size')
plt.ylabel('Avg Time (s)')
plt.title('Tiled GEMM (NumPy): Execution Time vs Matrix Size by Block Size')
plt.xticks(test_sizes, [f'{s}x{s}' for s in test_sizes])
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('plots/CPU_Tiled/tiled_numpy_by_block.png')
print('Plot saved: plots/CPU_Tiled/tiled_numpy_by_block.png')

# ── Plot 3: Boxplot — Python vs NumPy across all block sizes for 512x512 ──
plt.figure(figsize=(10, 6))
colors = {'NumPy': '#3498db', 'Python': '#e67e22'}
categories = ['NumPy', 'Python']

for i, bs in enumerate(block_sizes):
    for cat_idx, cat_name in enumerate(categories):
        values = all_times[bs][512][cat_idx]
        bp = plt.boxplot(values, positions=[i * 3 + cat_idx], widths=0.6,
                         patch_artist=True, showfliers=False)
        plt.setp(bp['boxes'],   facecolor=colors[cat_name], alpha=0.4)
        plt.setp(bp['medians'], color='black')
        jitter = np.random.uniform(-0.1, 0.1, size=len(values))
        plt.scatter([i * 3 + cat_idx] * len(values) + jitter, values,
                    color=colors[cat_name], s=25, alpha=0.7,
                    label=cat_name if i == 0 else '')

plt.xticks([i * 3 + 0.5 for i in range(len(block_sizes))],
           [f'block={bs}' for bs in block_sizes])
plt.yscale('log')
plt.xlabel('Block Size')
plt.ylabel('Time (s - log scale)')
plt.title('Tiled GEMM: Python vs NumPy (512x512)')
plt.legend()
plt.grid(axis='y', which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plots/CPU_Tiled/tiled_boxplot_512.png')
print('Plot saved: plots/CPU_Tiled/tiled_boxplot_512.png')
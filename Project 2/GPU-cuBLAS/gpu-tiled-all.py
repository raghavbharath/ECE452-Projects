# Raghav Bharathan
# ECE 452
# Project 2 - GPU Tiled All Sizes Runner
# Professor Shaahin Angizi

import os
import re
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt

test_sizes  = [512, 1024, 2048]
block_size  = 16
iterations  = 5

results = {
    'cuBLAS Standard':     [],
    'CuPy Tiled (Python)': [],
    'Naive CUDA Kernel':   [],
    'Shared Mem Kernel':   [],
}

gflops_results = {k: [] for k in results}

for size in test_sizes:
    print(f'Running {size}x{size}...')
    res = subprocess.run(
        [sys.executable, 'gpu-tiled.py', '-s', str(size), '-i', str(iterations), '-b', str(block_size)],
        capture_output=True, text=True
    )

    std_times    = [float(x) for x in re.findall(r'Standard: ([\d.]+)s',     res.stdout)]
    tiled_times  = [float(x) for x in re.findall(r'CuPy Tiled: ([\d.]+)s',   res.stdout)]
    naive_times  = [float(x) for x in re.findall(r'Naive Kernel: ([\d.]+)s', res.stdout)]
    shared_times = [float(x) for x in re.findall(r'Shared Kernel: ([\d.]+)s',res.stdout)]

    def gf(n, t): return (2 * n**3) / t / 1e9

    for times, key in zip(
        [std_times, tiled_times, naive_times, shared_times],
        results.keys()
    ):
        avg = np.mean(times) if times else float('nan')
        results[key].append(avg)
        gflops_results[key].append(gf(size, avg) if avg == avg else float('nan'))

    print(f'  Done.')

os.makedirs('plots/GPU_Tiled', exist_ok=True)

# Plot 1: Execution time vs matrix size (log scale)
plt.figure(figsize=(10, 6))
markers = ['o', 's', '^', 'D']
for (name, times), marker in zip(results.items(), markers):
    plt.plot(test_sizes, times, marker=marker, linewidth=2, label=name)

plt.yscale('log')
plt.xlabel('Matrix Size (NxN)')
plt.ylabel('Avg Time (s) - log scale')
plt.title(f'GPU GEMM: Execution Time vs Matrix Size (block={block_size})')
plt.xticks(test_sizes, [f'{s}x{s}' for s in test_sizes])
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plots/GPU_Tiled/time_vs_size.png')
print('Saved: plots/GPU_Tiled/time_vs_size.png')

# Plot 2: GFLOPS vs matrix size
plt.figure(figsize=(10, 6))
for (name, gf_vals), marker in zip(gflops_results.items(), markers):
    plt.plot(test_sizes, gf_vals, marker=marker, linewidth=2, label=name)

plt.xlabel('Matrix Size (NxN)')
plt.ylabel('GFLOPS')
plt.title(f'GPU GEMM: GFLOPS vs Matrix Size (block={block_size})')
plt.xticks(test_sizes, [f'{s}x{s}' for s in test_sizes])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plots/GPU_Tiled/gflops_vs_size.png')
print('Saved: plots/GPU_Tiled/gflops_vs_size.png')
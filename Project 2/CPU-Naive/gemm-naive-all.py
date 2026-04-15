import os
import re
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt

test_sizes = [128, 256, 512]

all_times = []

for size in test_sizes:
    print(f'Running {size}x{size}...')
    result = subprocess.run(
        [sys.executable, 'gemm-naive.py', '-s', f'{size}', '-i', '3'],
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

    all_times.append([numpy_times, python_times])
    print(f'  Done. Got {len(numpy_times)} iterations.')

categories = ['NumPy', 'Python']
colors     = {'NumPy': '#3498db', 'Python': '#e67e22'}

plt.figure(figsize=(10, 6))

for i, size in enumerate(test_sizes):
    for cat_idx, cat_name in enumerate(categories):
        values = all_times[i][cat_idx]
        bp = plt.boxplot(values, positions=[size], widths=20,
                         patch_artist=True, showfliers=False)
        plt.setp(bp['boxes'],   facecolor=colors[cat_name], alpha=0.4)
        plt.setp(bp['medians'], color='black')

        jitter = np.random.uniform(-0.8, 0.8, size=len(values))
        plt.scatter([size] * len(values) + jitter, values,
                    color=colors[cat_name], s=25, alpha=0.7,
                    label=cat_name if i == 0 else '')

os.makedirs('plots/CPU_Naive', exist_ok=True)
plt.yscale('log')
plt.xticks(test_sizes, [f'{s}x{s}' for s in test_sizes])
plt.xlabel('Matrix Size')
plt.ylabel('Time (seconds - log scale)')
plt.title('Naive GEMM: Python vs NumPy Performance Distribution')
plt.legend()
plt.grid(axis='y', which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plots/CPU_Naive/naive_all.png')
print('\nPlot saved: plots/CPU_Naive/naive_all.png')
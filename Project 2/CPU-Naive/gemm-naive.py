# Raghav Bharathan
# ECE 452
# Project 2
# Professor Shaahin Angizi

import os
import sys
import timeit
import argparse
import numpy as np
import matplotlib.pyplot as plt

#self notes when running: 
# -s is the size (ex. -s 128 is 128x128)
# -i is the # of iterations (how many times to run and average. So -i 1 means run once)
# -p is the plot flag

# The gemm-naive-all.py script automates this 

def gemm_numpy(A, B, C):
    start = timeit.default_timer()
    result = A @ B + C
    elapsed = timeit.default_timer() - start
    return result, elapsed


def gemm_python(A, B, C):
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape

    if cols_A != rows_B:
        print('Error: columns of A must match rows of B')
        sys.exit(1)

    result = np.zeros((rows_A, cols_B), dtype=np.float32)

    start = timeit.default_timer()
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i, j] += A[i, k] * B[k, j]
            result[i, j] += C[i, j]
    elapsed = timeit.default_timer() - start

    return result, elapsed

#Just gonna plot some data so it's easier to visualize what's happening
def save_plot(numpy_times, python_times, avg_numpy, avg_python, size, iterations):
    dirr = 'plots/CPU_Naive'
    os.makedirs(dirr, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(numpy_times,  color='b', marker='o', linestyle='-', alpha=0.7, label='NumPy')
    plt.plot(python_times, color='r', marker='o', linestyle='-', alpha=0.7, label='Python loops')
    plt.axhline(y=avg_numpy,  color='b', linestyle='--', label=f'Avg NumPy:  {avg_numpy:.6f}s')
    plt.axhline(y=avg_python, color='r', linestyle='--', label=f'Avg Python: {avg_python:.4f}s')

    plt.title(f'Naive GEMM: Python vs NumPy ({size}x{size}, {iterations} iteration(s))')
    plt.xlabel('Iteration')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    fname = os.path.join(dirr, f'{size}x{size}_{iterations}iter.png')
    plt.savefig(fname)
    print(f'Plot saved: {fname}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Naive GEMM: Python loops vs NumPy')
    parser.add_argument('-s', '--size',       type=int, default=128, help='Matrix size NxN')
    parser.add_argument('-i', '--iterations', type=int, default=1,   help='Number of iterations')
    parser.add_argument('-p', '--plot',       action='store_true',   help='Save plot')
    args = parser.parse_args()

    if args.size < 1 or args.iterations < 1:
        print('Size and iterations must be positive')
        sys.exit(1)

    A = np.random.randn(args.size, args.size).astype(np.float32)
    B = np.random.randn(args.size, args.size).astype(np.float32)
    C = np.zeros((args.size, args.size), dtype=np.float32)

    numpy_times  = []
    python_times = []

    print(f'\n=== Naive GEMM | {args.size}x{args.size} | {args.iterations} iteration(s) ===\n')

    for i in range(args.iterations):
        _, t_numpy  = gemm_numpy(A, B, C)
        _, t_python = gemm_python(A, B, C)

        numpy_times.append(t_numpy)
        python_times.append(t_python)

        print(f'  Iter {i+1}/{args.iterations} | NumPy: {t_numpy:.6f}s | Python: {t_python:.4f}s')

    avg_numpy  = sum(numpy_times)  / args.iterations
    avg_python = sum(python_times) / args.iterations
    speedup    = avg_python / avg_numpy

    print(f'\n  Avg NumPy:  {avg_numpy:.6f}s')
    print(f'  Avg Python: {avg_python:.4f}s')
    print(f'  Speedup:    {speedup:.1f}x\n')

    if args.plot:
        save_plot(numpy_times, python_times, avg_numpy, avg_python, args.size, args.iterations)
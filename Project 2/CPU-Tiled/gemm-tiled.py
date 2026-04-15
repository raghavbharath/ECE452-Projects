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

# self notes when running:
# -s is the size (ex. -s 128 is 128x128)
# -i is the # of iterations
# -b is the block size (ex. -b 32 means 32x32 tiles)
# -p is the plot flag

# The gemm-tiled-all.py script automates this


def gemm_tiled_python(A, B, C, block_size):
    N = A.shape[0]
    result = np.zeros((N, N), dtype=np.float32)

    start = timeit.default_timer()
    for f in range(0, N, block_size):
        for g in range(0, N, block_size):
            for h in range(0, N, block_size):
                for i in range(f, min(f + block_size, N)):
                    for j in range(g, min(g + block_size, N)):
                        for k in range(h, min(h + block_size, N)):
                            result[i, j] += A[i, k] * B[k, j]
    for i in range(N):
        for j in range(N):
            result[i, j] += C[i, j]
    elapsed = timeit.default_timer() - start

    return result, elapsed


def gemm_tiled_numpy(A, B, C, block_size):
    N = A.shape[0]
    result = np.zeros((N, N), dtype=np.float32)

    start = timeit.default_timer()
    for f in range(0, N, block_size):
        for g in range(0, N, block_size):
            for h in range(0, N, block_size):
                i_end = min(f + block_size, N)
                j_end = min(g + block_size, N)
                k_end = min(h + block_size, N)
                result[f:i_end, g:j_end] += A[f:i_end, h:k_end] @ B[h:k_end, g:j_end]
    result += C
    elapsed = timeit.default_timer() - start

    return result, elapsed


def save_plot(numpy_times, python_times, avg_numpy, avg_python, size, block_size, iterations):
    dirr = 'plots/CPU_Tiled'
    os.makedirs(dirr, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(numpy_times,  color='b', marker='o', linestyle='-', alpha=0.7, label='NumPy tiled')
    plt.plot(python_times, color='r', marker='o', linestyle='-', alpha=0.7, label='Python tiled')
    plt.axhline(y=avg_numpy,  color='b', linestyle='--', label=f'Avg NumPy:  {avg_numpy:.6f}s')
    plt.axhline(y=avg_python, color='r', linestyle='--', label=f'Avg Python: {avg_python:.4f}s')

    plt.title(f'Tiled GEMM: Python vs NumPy ({size}x{size}, block={block_size}, {iterations} iter)')
    plt.xlabel('Iteration')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    fname = os.path.join(dirr, f'{size}x{size}_block{block_size}_{iterations}iter.png')
    plt.savefig(fname)
    print(f'Plot saved: {fname}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tiled GEMM: Python loops vs NumPy')
    parser.add_argument('-s', '--size',       type=int, default=128, help='Matrix size NxN')
    parser.add_argument('-i', '--iterations', type=int, default=1,   help='Number of iterations')
    parser.add_argument('-b', '--block',      type=int, default=32,  help='Tile/block size')
    parser.add_argument('-p', '--plot',       action='store_true',   help='Save plot')
    args = parser.parse_args()

    if args.size < 1 or args.iterations < 1 or args.block < 1:
        print('Size, iterations, and block size must be positive')
        sys.exit(1)

    A = np.random.randn(args.size, args.size).astype(np.float32)
    B = np.random.randn(args.size, args.size).astype(np.float32)
    C = np.zeros((args.size, args.size), dtype=np.float32)

    numpy_times  = []
    python_times = []

    print(f'\n=== Tiled GEMM | {args.size}x{args.size} | block={args.block} | {args.iterations} iteration(s) ===\n')

    for i in range(args.iterations):
        _, t_numpy  = gemm_tiled_numpy(A, B, C, args.block)
        _, t_python = gemm_tiled_python(A, B, C, args.block)

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
        save_plot(numpy_times, python_times, avg_numpy, avg_python,
                  args.size, args.block, args.iterations)
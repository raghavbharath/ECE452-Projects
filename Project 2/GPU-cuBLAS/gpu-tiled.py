# Raghav Bharathan
# ECE 452
# Project 2 - Steps 4 and 5
# Professor Shaahin Angizi

import os
import sys
import argparse
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# Creates two CUDA events that are measured on the GPU itself
# So it's the most optimal way to do it.
def gemm_cupy_standard(A, B, C):
    # I'm ignoring the warmup here - I'll do it in the benchmark()

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()
    ret = cp.matmul(A,B) + C
    end.record()

    end.synchronize()
    elapsed = cp.cuda.get_elapsed_time(start, end)/1000

    return ret, elapsed

# Similar to my CPU tiled, but it just lives on the GPU instead
# The @ operator on CuPy arrays still call cuBLAS per tile.
def gemm_cupy_tiled(A, B, C, b_size):
    N = A.shape[0]
    ret = cp.copy(C)

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()

    for i in range(0, N, b_size):
        for j in range(0, N, b_size):
            for k in range(0, N, b_size):
                ret[i:i+b_size, j:j+b_size] += (A[i:i+b_size, k:k+b_size] @ B[k:k+b_size, j:j+b_size])
    end.record()

    end.synchronize()
    return ret, cp.cuda.get_elapsed_time(start, end) / 1000.0



# This is a Naive CUDA kernel that has only global memory
# Every thread hits global memory on every k iteration
# There's no caching. This will be my baseline for the shared memory kernel

naive_kernel = cp.RawKernel(r'''
extern "C" __global__
void naive_gemm(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
 
    if (row < N && col < N) {
        float val = 0.0f;
        for (int k = 0; k < N; k++) {
            val += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] += val;
    }
}
''', 'naive_gemm')
 
def gemm_naive_kernel(A, B, C):
    N       = A.shape[0]
    ret     = cp.copy(C)
    threads = (16, 16)
    grid    = (int(np.ceil(N / 16)), int(np.ceil(N / 16)))
 
    start = cp.cuda.Event()
    end   = cp.cuda.Event()
 
    start.record()
    naive_kernel(grid, threads, (A, B, ret, np.int32(N)))
    end.record()
 
    end.synchronize()
    return ret, cp.cuda.get_elapsed_time(start, end) / 1000.0
 
 

# This is a tiled implementation with shared memory CUDA kernel
# We first load 16x16 tiles of A and B into __shared__ memory
# Then do all the math from there with far fewer global memory reads
#  __syncthreads() is the barrier, all 256 threads must hit it before continuing

shared_kernel = cp.RawKernel(r'''
#define BLOCK 16
 
extern "C" __global__
void shared_gemm(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * BLOCK + threadIdx.y;
    int col = blockIdx.x * BLOCK + threadIdx.x;
 
    __shared__ float tileA[BLOCK][BLOCK];
    __shared__ float tileB[BLOCK][BLOCK];
 
    float val = 0.0f;
 
    for (int t = 0; t < (N + BLOCK - 1) / BLOCK; t++) {
        if (row < N && t * BLOCK + threadIdx.x < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + t * BLOCK + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
 
        if (col < N && t * BLOCK + threadIdx.y < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t * BLOCK + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
 
        __syncthreads();
 
        for (int k = 0; k < BLOCK; k++)
            val += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
 
        __syncthreads();
    }
 
    if (row < N && col < N)
        C[row * N + col] += val;
}
''', 'shared_gemm')
 
def gemm_shared_kernel(A, B, C):
    N       = A.shape[0]
    ret     = cp.copy(C)
    threads = (16, 16)
    grid    = (int(np.ceil(N / 16)), int(np.ceil(N / 16)))
 
    start = cp.cuda.Event()
    end   = cp.cuda.Event()
 
    start.record()
    shared_kernel(grid, threads, (A, B, ret, np.int32(N)))
    end.record()
 
    end.synchronize()
    return ret, cp.cuda.get_elapsed_time(start, end) / 1000.0
 
 

# GFLOPS metric: (2 * n^3) / time
# 2 ops per MAC (1 multiply + 1 add)
# Standard way to measure GEMM performance

def gflops(n, elapsed):
    return (2 * n ** 3) / elapsed / 1e9
 
 

# Benchmark runner with warmup

def benchmark(fn, *args, iterations=10):
    fn(*args)  # warmup; this discards CUDA kernel compilation overhead
    cp.cuda.Stream.null.synchronize()
 
    times = []
    for _ in range(iterations):
        _, t = fn(*args)
        times.append(t)
    return times
 
 

# Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPU GEMM Benchmark')
    parser.add_argument('-s', '--size',       type=int, default=1024, help='Matrix size NxN')
    parser.add_argument('-i', '--iterations', type=int, default=10,   help='Number of iterations')
    parser.add_argument('-b', '--block',      type=int, default=16,   help='Block size for tiled versions')
    parser.add_argument('-p', '--plot',       action='store_true',    help='Save plot')
    args = parser.parse_args()
 
    N = args.size
    print(f'\nGPU: {cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}')
    print(f'Matrix: {N}x{N} | Block: {args.block} | Iterations: {args.iterations}\n')
 
    A = cp.random.randn(N, N, dtype=cp.float32)
    B = cp.random.randn(N, N, dtype=cp.float32)
    C = cp.zeros((N, N), dtype=cp.float32)
 
    print('Running benchmarks...\n')
 
    t_std    = benchmark(gemm_cupy_standard,                A, B, C,           iterations=args.iterations)
    t_tiled  = benchmark(gemm_cupy_tiled,                   A, B, C, args.block, iterations=args.iterations)
    t_naive  = benchmark(gemm_naive_kernel,                 A, B, C,           iterations=args.iterations)
    t_shared = benchmark(gemm_shared_kernel,                A, B, C,           iterations=args.iterations)
 
    methods = {
        'cuBLAS Standard':      t_std,
        'CuPy Tiled (Python)':  t_tiled,
        'Naive CUDA Kernel':    t_naive,
        'Shared Mem Kernel':    t_shared,
    }
 
    print(f"{'Method':<24} {'Avg Time (s)':<16} {'GFLOPS':<12}")
    print('-' * 52)
    for name, times in methods.items():
        avg = np.mean(times)
        gf  = gflops(N, avg)
        print(f'{name:<24} {avg:<16.6f} {gf:<12.2f}')
 
    print()
    print('Per-iteration times:')
    for i in range(args.iterations):
        print(f'  Iter {i+1:2d} | '
              f'Standard: {t_std[i]:.6f}s | '
              f'CuPy Tiled: {t_tiled[i]:.6f}s | '
              f'Naive Kernel: {t_naive[i]:.6f}s | '
              f'Shared Kernel: {t_shared[i]:.6f}s')
 
    if args.plot:
        dirr = 'plots/GPU_Tiled'
        os.makedirs(dirr, exist_ok=True)
 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
 
        for name, times in methods.items():
            ax1.plot(times, marker='o', label=name)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Time (s)')
        ax1.set_title(f'GPU GEMM: Time per Iteration ({N}x{N})')
        ax1.legend()
        ax1.grid(True, linestyle=':', alpha=0.6)
 
        names = list(methods.keys())
        avgs  = [np.mean(t) for t in methods.values()]
        gf    = [gflops(N, a) for a in avgs]
        bars  = ax2.bar(names, gf, color=['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'])
        ax2.set_ylabel('GFLOPS')
        ax2.set_title(f'GPU GEMM: GFLOPS Comparison ({N}x{N})')
        ax2.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars, gf):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        ax2.grid(axis='y', linestyle=':', alpha=0.6)
 
        plt.tight_layout()
        fname = os.path.join(dirr, f'{N}x{N}_block{args.block}_{args.iterations}iter.png')
        plt.savefig(fname)
        print(f'\nPlot saved: {fname}')
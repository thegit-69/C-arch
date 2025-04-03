import numpy as np
import time

# Matrix size
N = 128  

# Standard Matrix Multiplication (Slow)
def matrix_multiply_standard(A, B):
    C = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            sum_val = 0.0
            for k in range(N):
                sum_val += A[i, k] * B[k, j]
            C[i, j] = sum_val
    return C

# NumPy Optimized Matrix Multiplication (Fast)
def matrix_multiply_numpy(A, B):
    return np.dot(A, B)  # Uses SIMD optimizations

# Generate Random Matrices
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)

# Measure Execution Time
start = time.time()
C1 = matrix_multiply_standard(A, B)
time_standard = time.time() - start

start = time.time()
C2 = matrix_multiply_numpy(A, B)
time_numpy = time.time() - start

# Print Results
print(f"Standard Execution Time: {time_standard:.4f} seconds")
print(f"NumPy Execution Time: {time_numpy:.4f} seconds")
print(f"Speedup Factor: {time_standard / time_numpy:.2f}x")


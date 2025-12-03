import os
TRITON_INTERPRET = 1
TRITON_INTERPRET = 0
os.environ["TRITON_INTERPRET"] = str(TRITON_INTERPRET)
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"  # Enable Triton autotuning output
if TRITON_INTERPRET:
    print('[WARNING] Triton is in interpret mode')
# add ~/mamba.py to PYTHONPATH
# import sys
# path = os.path.expanduser("~/mamba.py/mambapy")
# print(f"Adding {path} to sys.path")
# sys.path.append(path)
# import pscan
import triton
import triton.language as tl
import torch
TRITON_INTERPRET : tl.constexpr = tl.constexpr(TRITON_INTERPRET)
NOT_TRITON_INTERPRET : tl.constexpr = tl.constexpr(1 - TRITON_INTERPRET)
# import profile_utils

@triton.jit
def discounted_cumsum_triton_op(a1, x1, a2, x2):
    return a1 * a2, x1 * a2 + x2

@triton.jit
def discounted_cumsum_complex_triton_op(a1_real, a1_imag, x1_real, x1_imag, a2_real, a2_imag, x2_real, x2_imag):
    """
    Complex discount cumsum combine function.
    
    Args:
        a1_real, a1_imag: Real and imaginary parts of first discount factor
        x1_real, x1_imag: Real and imaginary parts of first accumulated value
        a2_real, a2_imag: Real and imaginary parts of second discount factor
        x2_real, x2_imag: Real and imaginary parts of second accumulated value
    
    Returns:
        (a_r, a_i, x_r, x_i) where:
        - a_r, a_i: Real and imaginary parts of (a1 * a2)
        - x_r, x_i: Real and imaginary parts of (x1 * a2 + x2)
    """
    # Complex multiplication: (a1_r + i*a1_i) * (a2_r + i*a2_i)
    # = (a1_r*a2_r - a1_i*a2_i) + i*(a1_r*a2_i + a1_i*a2_r)
    a_r = a1_real * a2_real - a1_imag * a2_imag
    a_i = a1_real * a2_imag + a1_imag * a2_real
    
    # Complex times complex: (x1_r + i*x1_i) * (a2_r + i*a2_i) + (x2_r + i*x2_i)
    # = (x1_r*a2_r - x1_i*a2_i + x2_r) + i*(x1_r*a2_i + x1_i*a2_r + x2_i)
    x_r = x1_real * a2_real - x1_imag * a2_imag + x2_real
    x_i = x1_real * a2_imag + x1_imag * a2_real + x2_imag
    
    return a_r, a_i, x_r, x_i

@triton.jit
def discounted_cumsum_complex_gradient_triton_op(a1_real, a1_imag, dy1_real, dy1_imag, a2_real, a2_imag, dy2_real, dy2_imag):
    a_cum_real = a2_real * a1_real - a2_imag * a1_imag
    a_cum_imag = a2_real * a1_imag + a2_imag * a1_real

    dx_real = dy1_real * a2_real + dy1_imag * a2_imag + dy2_real
    dx_imag = -dy1_real * a2_imag + dy1_imag * a2_real + dy2_imag
    
    return a_cum_real, a_cum_imag, dx_real, dx_imag

@triton.autotune(configs=[
    triton.Config({'BLOCK_SIZE_0': 32, 'BLOCK_SIZE_1': 32, 'BLOCK_SIZE_3': 32}, num_warps=2),
    triton.Config({'BLOCK_SIZE_0': 32, 'BLOCK_SIZE_1': 32, 'BLOCK_SIZE_3': 32}, num_warps=4),
    triton.Config({'BLOCK_SIZE_0': 32, 'BLOCK_SIZE_1': 32, 'BLOCK_SIZE_3': 32}, num_warps=8),

    triton.Config({'BLOCK_SIZE_0': 64, 'BLOCK_SIZE_1': 64, 'BLOCK_SIZE_3': 64}, num_warps=2),
    triton.Config({'BLOCK_SIZE_0': 64, 'BLOCK_SIZE_1': 64, 'BLOCK_SIZE_3': 64}, num_warps=4),
    triton.Config({'BLOCK_SIZE_0': 64, 'BLOCK_SIZE_1': 64, 'BLOCK_SIZE_3': 64}, num_warps=8),
    ],
    key=['size_0','size_1','size_2','size_3'])
@triton.jit
def discounted_cumsum_N_BCHW_triton_kernel(
                            A,
                            X, X_cum,
                            size_0,
                            size_1,
                            size_2,
                            size_3,
                            stride_A_0: tl.constexpr,
                            stride_A_1: tl.constexpr,
                            stride_A_2: tl.constexpr,
                            stride_A_3: tl.constexpr,
                            stride_X_0: tl.constexpr,
                            stride_X_1: tl.constexpr,
                            stride_X_2: tl.constexpr,
                            stride_X_3: tl.constexpr,
                            BLOCK_SIZE_0: tl.constexpr,
                            BLOCK_SIZE_1: tl.constexpr,
                            BLOCK_SIZE_2: tl.constexpr,
                            BLOCK_SIZE_3: tl.constexpr):
    start_0 = tl.program_id(axis=0) # B
    start_1 = tl.program_id(axis=1) # C
    start_3 = tl.program_id(axis=2) # W

    range_0 = tl.arange(0, BLOCK_SIZE_0)
    range_1 = tl.arange(0, BLOCK_SIZE_1)
    range_2 = tl.arange(0, BLOCK_SIZE_2)
    range_3 = tl.arange(0, BLOCK_SIZE_3)
    
    # Calculate offsets for tensor X
    offsets_X_0 = (start_0 * BLOCK_SIZE_0 + range_0[:, None, None, None]) * stride_X_0
    offsets_X_1 = (start_1 * BLOCK_SIZE_1 + range_1[None, :, None, None]) * stride_X_1
    offsets_X_2 = range_2[None, None, :, None] * stride_X_2
    offsets_X_3 = (start_3 * BLOCK_SIZE_3 + range_3[None, None, None, :]) * stride_X_3

    # Calculate offsets for tensor A
    offsets_A_0 = (start_0 * BLOCK_SIZE_0 + range_0[:, None, None, None])
    offsets_A_1 = (start_1 * BLOCK_SIZE_1 + range_1[None, :, None, None])
    offsets_A_2 = range_2[None, None, :, None]
    offsets_A_3 = (start_3 * BLOCK_SIZE_3 + range_3[None, None, None, :])
    
    mask_0 = offsets_X_0 < size_0 * stride_X_0
    mask_1 = offsets_X_1 < size_1 * stride_X_1
    mask_2 = offsets_X_2 < size_2 * stride_X_2
    mask_3 = offsets_X_3 < size_3 * stride_X_3
    mask = mask_0 * mask_1 * mask_2 * mask_3

    mask_A_0 = offsets_A_0 < size_0
    mask_A_1 = offsets_A_1 < size_1
    mask_A_2 = offsets_A_2 < size_2
    mask_A_3 = offsets_A_3 < size_3
    mask_A = mask_A_0 * mask_A_1 * mask_A_2 * mask_A_3
    
    idx_X = X + offsets_X_0 + offsets_X_1 + offsets_X_2 + offsets_X_3
    idx_A = A + offsets_A_0*stride_A_0 + offsets_A_1*stride_A_1 + offsets_A_2*stride_A_2 + offsets_A_3*stride_A_3
    x = tl.load(idx_X, mask=mask)
    a = tl.load(idx_A, mask=mask_A)
    a_cum, x_cum = tl.associative_scan((a,x), axis=2, combine_fn=discounted_cumsum_triton_op, reverse=False)
    tl.store(X_cum + offsets_X_0 + offsets_X_1 + offsets_X_2 + offsets_X_3, x_cum, mask=mask)

@triton.autotune(configs=[
    triton.Config({'BLOCK_SIZE_0': 32, 'BLOCK_SIZE_1': 32, 'BLOCK_SIZE_3': 32}, num_warps=2),
    triton.Config({'BLOCK_SIZE_0': 32, 'BLOCK_SIZE_1': 32, 'BLOCK_SIZE_3': 32}, num_warps=4),
    triton.Config({'BLOCK_SIZE_0': 32, 'BLOCK_SIZE_1': 32, 'BLOCK_SIZE_3': 32}, num_warps=8),

    triton.Config({'BLOCK_SIZE_0': 64, 'BLOCK_SIZE_1': 64, 'BLOCK_SIZE_3': 64}, num_warps=2),
    triton.Config({'BLOCK_SIZE_0': 64, 'BLOCK_SIZE_1': 64, 'BLOCK_SIZE_3': 64}, num_warps=4),
    triton.Config({'BLOCK_SIZE_0': 64, 'BLOCK_SIZE_1': 64, 'BLOCK_SIZE_3': 64}, num_warps=8),
    ],
    key=['size_0','size_1','size_2','size_3'])
@triton.jit
def discounted_cumsum_N_BCHW_complex_triton_kernel(
                            A_real,
                            A_imag,
                            X_real,
                            X_imag,
                            X_cum_real, X_cum_imag,
                            size_0,
                            size_1,
                            size_2,
                            size_3,
                            stride_A_0: tl.constexpr,
                            stride_A_1: tl.constexpr,
                            stride_A_2: tl.constexpr,
                            stride_A_3: tl.constexpr,
                            stride_X_0: tl.constexpr,
                            stride_X_1: tl.constexpr,
                            stride_X_2: tl.constexpr,
                            stride_X_3: tl.constexpr,
                            stride_X_cum_0: tl.constexpr,
                            stride_X_cum_1: tl.constexpr,
                            stride_X_cum_2: tl.constexpr,
                            stride_X_cum_3: tl.constexpr,
                            BLOCK_SIZE_0: tl.constexpr,
                            BLOCK_SIZE_1: tl.constexpr,
                            BLOCK_SIZE_2: tl.constexpr,
                            BLOCK_SIZE_3: tl.constexpr):
    """
    Complex discount cumsum kernel.
    
    A_real and A_imag are separate pointers to the real and imaginary parts of A.
    A has shape (B, C, H, W, 2) where A[..., 0] is real and A[..., 1] is imaginary.
    X_real and X_imag are separate pointers to the real and imaginary parts of X.
    X can be real (X_imag will be zeros) or complex (both X_real and X_imag provided).
    X_cum_real and X_cum_imag are separate output tensors for real and imaginary parts.
    """
    start_0 = tl.program_id(axis=0) # B
    start_1 = tl.program_id(axis=1) # C
    start_3 = tl.program_id(axis=2) # W

    range_0 = tl.arange(0, BLOCK_SIZE_0)
    range_1 = tl.arange(0, BLOCK_SIZE_1)
    range_2 = tl.arange(0, BLOCK_SIZE_2)
    range_3 = tl.arange(0, BLOCK_SIZE_3)
    
    # Calculate offsets for tensor X
    offsets_X_0 = (start_0 * BLOCK_SIZE_0 + range_0[:, None, None, None]) * stride_X_0
    offsets_X_1 = (start_1 * BLOCK_SIZE_1 + range_1[None, :, None, None]) * stride_X_1
    offsets_X_2 = range_2[None, None, :, None] * stride_X_2
    offsets_X_3 = (start_3 * BLOCK_SIZE_3 + range_3[None, None, None, :]) * stride_X_3

    # Calculate offsets for tensor A (real and imag parts have same strides)
    offsets_A_0 = (start_0 * BLOCK_SIZE_0 + range_0[:, None, None, None])
    offsets_A_1 = (start_1 * BLOCK_SIZE_1 + range_1[None, :, None, None])
    offsets_A_2 = range_2[None, None, :, None]
    offsets_A_3 = (start_3 * BLOCK_SIZE_3 + range_3[None, None, None, :])
    
    mask_0 = offsets_X_0 < size_0 * stride_X_0
    mask_1 = offsets_X_1 < size_1 * stride_X_1
    mask_2 = offsets_X_2 < size_2 * stride_X_2
    mask_3 = offsets_X_3 < size_3 * stride_X_3
    mask = mask_0 * mask_1 * mask_2 * mask_3

    mask_A_0 = offsets_A_0 < size_0
    mask_A_1 = offsets_A_1 < size_1
    mask_A_2 = offsets_A_2 < size_2
    mask_A_3 = offsets_A_3 < size_3
    mask_A = mask_A_0 * mask_A_1 * mask_A_2 * mask_A_3
    
    idx_X_real = X_real + offsets_X_0 + offsets_X_1 + offsets_X_2 + offsets_X_3
    idx_X_imag = X_imag + offsets_X_0 + offsets_X_1 + offsets_X_2 + offsets_X_3
    idx_A_real = A_real + offsets_A_0*stride_A_0 + offsets_A_1*stride_A_1 + offsets_A_2*stride_A_2 + offsets_A_3*stride_A_3
    idx_A_imag = A_imag + offsets_A_0*stride_A_0 + offsets_A_1*stride_A_1 + offsets_A_2*stride_A_2 + offsets_A_3*stride_A_3
    
    x_real = tl.load(idx_X_real, mask=mask)
    x_imag = tl.load(idx_X_imag, mask=mask)
    a_real = tl.load(idx_A_real, mask=mask_A)
    a_imag = tl.load(idx_A_imag, mask=mask_A)
    
    # Perform associative scan with complex combine function
    # Input tuple: (a_real, a_imag, x_real, x_imag) - 4 elements
    # Output tuple: (a_cum_real, a_cum_imag, x_cum_real, x_cum_imag) - 4 elements
    a_cum_real, a_cum_imag, x_cum_real, x_cum_imag = tl.associative_scan(
        (a_real, a_imag, x_real, x_imag), axis=2, combine_fn=discounted_cumsum_complex_triton_op, reverse=False
    )

    # Calculate offsets for output tensors
    offsets_X_cum_0 = (start_0 * BLOCK_SIZE_0 + range_0[:, None, None, None]) * stride_X_cum_0
    offsets_X_cum_1 = (start_1 * BLOCK_SIZE_1 + range_1[None, :, None, None]) * stride_X_cum_1
    offsets_X_cum_2 = range_2[None, None, :, None] * stride_X_cum_2
    offsets_X_cum_3 = (start_3 * BLOCK_SIZE_3 + range_3[None, None, None, :]) * stride_X_cum_3
    
    idx_X_cum_real = X_cum_real + offsets_X_cum_0 + offsets_X_cum_1 + offsets_X_cum_2 + offsets_X_cum_3
    idx_X_cum_imag = X_cum_imag + offsets_X_cum_0 + offsets_X_cum_1 + offsets_X_cum_2 + offsets_X_cum_3
    
    tl.store(idx_X_cum_real, x_cum_real, mask=mask)
    tl.store(idx_X_cum_imag, x_cum_imag, mask=mask)

@triton.jit
def discounted_cumsum_N_BCHW_gradient_triton_kernel(
                            A,
                            dY,
                            dX,
                            size_0,
                            size_1,
                            size_2,
                            size_3,
                            stride_A_0: tl.constexpr,
                            stride_A_1: tl.constexpr,
                            stride_A_2: tl.constexpr,
                            stride_A_3: tl.constexpr,

                            stride_dX_0: tl.constexpr,
                            stride_dX_1: tl.constexpr,
                            stride_dX_2: tl.constexpr,
                            stride_dX_3: tl.constexpr,

                            BLOCK_SIZE_0: tl.constexpr,
                            BLOCK_SIZE_1: tl.constexpr,
                            BLOCK_SIZE_2: tl.constexpr,
                            BLOCK_SIZE_3: tl.constexpr):
    start_0 = tl.program_id(axis=0) # B
    start_1 = tl.program_id(axis=1) # C
    start_3 = tl.program_id(axis=2) # W

    range_0 = tl.arange(0, BLOCK_SIZE_0)
    range_1 = tl.arange(0, BLOCK_SIZE_1)
    range_2 = tl.arange(0, BLOCK_SIZE_2)
    range_3 = tl.arange(0, BLOCK_SIZE_3)
    
    # Calculate offsets for tensor dY
    offsets_dY_0 = (start_0 * BLOCK_SIZE_0 + range_0[:, None, None, None]) * stride_dX_0
    offsets_dY_1 = (start_1 * BLOCK_SIZE_1 + range_1[None, :, None, None]) * stride_dX_1
    offsets_dY_2 = range_2[None, None, :, None] * stride_dX_2
    offsets_dY_3 = (start_3 * BLOCK_SIZE_3 + range_3[None, None, None, :]) * stride_dX_3
    
    # Calculate offsets for tensor A
    offsets_A_0 = (start_0 * BLOCK_SIZE_0 + range_0[:, None, None, None])
    offsets_A_1 = (start_1 * BLOCK_SIZE_1 + range_1[None, :, None, None])
    offsets_A_2 = (range_2[None, None, :, None] + 1)
    offsets_A_3 = (start_3 * BLOCK_SIZE_3 + range_3[None, None, None, :])
    
    mask_0 = offsets_dY_0 < size_0 * stride_dX_0
    mask_1 = offsets_dY_1 < size_1 * stride_dX_1
    mask_2 = offsets_dY_2 < size_2 * stride_dX_2
    mask_3 = offsets_dY_3 < size_3 * stride_dX_3
    mask = mask_0 * mask_1 * mask_2 * mask_3

    mask_A_0 = offsets_A_0 < size_0
    mask_A_1 = offsets_A_1 < size_1
    mask_A_2 = offsets_A_2 < size_2
    mask_A_3 = offsets_A_3 < size_3
    mask_A = mask_A_0 * mask_A_1 * mask_A_2 * mask_A_3
    
    idx_dY = dY + offsets_dY_0 + offsets_dY_1 + offsets_dY_2 + offsets_dY_3
    idx_A  = A + offsets_A_0*stride_A_0 + offsets_A_1*stride_A_1 + offsets_A_2*stride_A_2 + offsets_A_3*stride_A_3
    dy = tl.load(idx_dY, mask=mask)
    a  = tl.load(idx_A,  mask=mask_A, other=0.0)
    a_cum, dx = tl.associative_scan((a,dy), axis=2, combine_fn=discounted_cumsum_triton_op, reverse=True)

    tl.store(dX + offsets_dY_0 + offsets_dY_1 + offsets_dY_2 + offsets_dY_3, dx, mask=mask)

@triton.jit
def discounted_cumsum_N_BCHW_complex_gradient_triton_kernel(
                            A_real,
                            A_imag,
                            dY_real,
                            dY_imag,
                            dX_real,
                            dX_imag,
                            size_0,
                            size_1,
                            size_2,
                            size_3,
                            stride_A_0: tl.constexpr,
                            stride_A_1: tl.constexpr,
                            stride_A_2: tl.constexpr,
                            stride_A_3: tl.constexpr,
                            stride_dY_0: tl.constexpr,
                            stride_dY_1: tl.constexpr,
                            stride_dY_2: tl.constexpr,
                            stride_dY_3: tl.constexpr,
                            stride_dX_0: tl.constexpr,
                            stride_dX_1: tl.constexpr,
                            stride_dX_2: tl.constexpr,
                            stride_dX_3: tl.constexpr,
                            BLOCK_SIZE_0: tl.constexpr,
                            BLOCK_SIZE_1: tl.constexpr,
                            BLOCK_SIZE_2: tl.constexpr,
                            BLOCK_SIZE_3: tl.constexpr):
    """
    Complex discount cumsum gradient kernel.
    
    A_real and A_imag are separate pointers to the real and imaginary parts of A.
    dY_real and dY_imag are separate pointers to the real and imaginary parts of dY.
    dX_real and dX_imag are separate output tensors for real and imaginary parts of dX.
    """
    start_0 = tl.program_id(axis=0) # B
    start_1 = tl.program_id(axis=1) # C
    start_3 = tl.program_id(axis=2) # W

    range_0 = tl.arange(0, BLOCK_SIZE_0)
    range_1 = tl.arange(0, BLOCK_SIZE_1)
    range_2 = tl.arange(0, BLOCK_SIZE_2)
    range_3 = tl.arange(0, BLOCK_SIZE_3)
    
    # Calculate offsets for tensor dY
    offsets_dY_0 = (start_0 * BLOCK_SIZE_0 + range_0[:, None, None, None]) * stride_dY_0
    offsets_dY_1 = (start_1 * BLOCK_SIZE_1 + range_1[None, :, None, None]) * stride_dY_1
    offsets_dY_2 = range_2[None, None, :, None] * stride_dY_2
    offsets_dY_3 = (start_3 * BLOCK_SIZE_3 + range_3[None, None, None, :]) * stride_dY_3
    
    # Calculate offsets for tensor A (shifted by 1 for reverse scan)
    offsets_A_0 = (start_0 * BLOCK_SIZE_0 + range_0[:, None, None, None])
    offsets_A_1 = (start_1 * BLOCK_SIZE_1 + range_1[None, :, None, None])
    offsets_A_2 = (range_2[None, None, :, None] + 1)
    offsets_A_3 = (start_3 * BLOCK_SIZE_3 + range_3[None, None, None, :])
    
    mask_0 = offsets_dY_0 < size_0 * stride_dY_0
    mask_1 = offsets_dY_1 < size_1 * stride_dY_1
    mask_2 = offsets_dY_2 < size_2 * stride_dY_2
    mask_3 = offsets_dY_3 < size_3 * stride_dY_3
    mask = mask_0 * mask_1 * mask_2 * mask_3

    mask_A_0 = offsets_A_0 < size_0
    mask_A_1 = offsets_A_1 < size_1
    mask_A_2 = offsets_A_2 < size_2
    mask_A_3 = offsets_A_3 < size_3
    mask_A = mask_A_0 * mask_A_1 * mask_A_2 * mask_A_3
    
    idx_dY_real = dY_real + offsets_dY_0 + offsets_dY_1 + offsets_dY_2 + offsets_dY_3
    idx_dY_imag = dY_imag + offsets_dY_0 + offsets_dY_1 + offsets_dY_2 + offsets_dY_3
    idx_A_real = A_real + offsets_A_0*stride_A_0 + offsets_A_1*stride_A_1 + offsets_A_2*stride_A_2 + offsets_A_3*stride_A_3
    idx_A_imag = A_imag + offsets_A_0*stride_A_0 + offsets_A_1*stride_A_1 + offsets_A_2*stride_A_2 + offsets_A_3*stride_A_3
    
    dy_real = tl.load(idx_dY_real, mask=mask)
    dy_imag = tl.load(idx_dY_imag, mask=mask)
    a_real = tl.load(idx_A_real, mask=mask_A, other=0.0)
    a_imag = tl.load(idx_A_imag, mask=mask_A, other=0.0)
    
    # Reverse associative scan with complex gradient combine function
    # Input tuple: (a_real, a_imag, dy_real, dy_imag) - 4 elements
    # Output tuple: (a_cum_real, a_cum_imag, dx_real, dx_imag) - 4 elements
    a_cum_real, a_cum_imag, dx_real, dx_imag = tl.associative_scan(
        (a_real, a_imag, dy_real, dy_imag), axis=2, combine_fn=discounted_cumsum_complex_gradient_triton_op, reverse=True
    )

    offsets_dX_0 = (start_0 * BLOCK_SIZE_0 + range_0[:, None, None, None]) * stride_dX_0
    offsets_dX_1 = (start_1 * BLOCK_SIZE_1 + range_1[None, :, None, None]) * stride_dX_1
    offsets_dX_2 = range_2[None, None, :, None] * stride_dX_2
    offsets_dX_3 = (start_3 * BLOCK_SIZE_3 + range_3[None, None, None, :]) * stride_dX_3

    tl.store(dX_real + offsets_dX_0 + offsets_dX_1 + offsets_dX_2 + offsets_dX_3, dx_real, mask=mask)
    tl.store(dX_imag + offsets_dX_0 + offsets_dX_1 + offsets_dX_2 + offsets_dX_3, dx_imag, mask=mask)

def discounted_cumsum_1d_ref(a, x):
    # Build result as a list to avoid in-place operations
    result = [x[0]]
    for i in range(1, x.shape[0]):
        result.append(result[-1] * a[i] + x[i])
    return torch.stack(result)

def discounted_cumsum_1d_complex_ref(a_real, a_imag, x):
    """
    Reference implementation for 1D complex discount cumsum.
    
    Args:
        a_real, a_imag: Real and imaginary parts of discount factors, shape (T,)
        x: Input values, shape (T,) for real or (T, 2) for complex where x[..., 0] is real and x[..., 1] is imaginary
    
    Returns:
        x_cum: Complex output, shape (T, 2) where [..., 0] is real and [..., 1] is imaginary
    """
    # Detect if x is complex
    if len(x.shape) == 2 and x.shape[1] == 2:
        # x is complex: shape (T, 2)
        x_real = x[:, 0]
        x_imag = x[:, 1]
        T = x.shape[0]
    else:
        # x is real: shape (T,)
        x_real = x
        x_imag = torch.zeros_like(x)
        T = len(x)
    
    # Build result as lists to avoid in-place operations (needed for autograd)
    x_cum_real_list = []
    x_cum_imag_list = []
    
    # First element: x[0] (can be complex)
    x_cum_real_list.append(x_real[0])
    x_cum_imag_list.append(x_imag[0])
    
    # Subsequent elements: x_cum[i] = x_cum[i-1] * (a_real[i] + i*a_imag[i]) + x[i]
    for i in range(1, T):
        # Complex multiplication: (x_cum_r + i*x_cum_i) * (a_r + i*a_i)
        # = (x_cum_r*a_r - x_cum_i*a_i) + i*(x_cum_r*a_i + x_cum_i*a_r)
        prev_real = x_cum_real_list[i-1]
        prev_imag = x_cum_imag_list[i-1]
        a_r = a_real[i]
        a_i = a_imag[i]
        
        # Multiply previous result by discount factor
        mult_real = prev_real * a_r - prev_imag * a_i
        mult_imag = prev_real * a_i + prev_imag * a_r
        
        # Add current x (can be complex)
        x_cum_real_list.append(mult_real + x_real[i])
        x_cum_imag_list.append(mult_imag + x_imag[i])
    
    # Stack into tensors and then into shape (T, 2)
    x_cum_real = torch.stack(x_cum_real_list)
    x_cum_imag = torch.stack(x_cum_imag_list)
    return torch.stack([x_cum_real, x_cum_imag], dim=-1)

@triton.jit
def discounted_cumsum_1d_gradient_triton_kernel(
                            a_ptr,
                            # x_ptr,
                            # y_ptr,
                            dy_ptr,
                            dx_ptr,

                            size_0,
                            BLOCK_SIZE_0: tl.constexpr):
    range_0 = tl.arange(0, BLOCK_SIZE_0)
    a  = tl.load( a_ptr  + range_0 + 1, mask=((range_0 + 1)< size_0))
    dy = tl.load( dy_ptr + range_0, mask=(range_0 < size_0))
    _, dx = tl.associative_scan((a,dy), axis=0, combine_fn=discounted_cumsum_triton_op, reverse=True)
    tl.store( dx_ptr + range_0, dx, mask=(range_0 < size_0))

def test_discounted_cumsum_gradient_triton():
    device = 'cuda'
    torch.manual_seed(0)
    a = torch.randn(10, device=device)
    x = torch.randn(10, device=device)
    dy = torch.randn(10, device=device)
    y = discounted_cumsum_1d_ref(a, x)

    da, dx = discounted_cumsum_1d_gradient(a, x, y, dy)
    grid = (1,)
    dx_triton = torch.zeros_like(a)
    discounted_cumsum_1d_gradient_triton_kernel[grid](a, dy, dx_triton, size_0=len(a), BLOCK_SIZE_0=triton.next_power_of_2(len(a)))
    da_triton = dx.clone()
    da_triton[1:] = da_triton[1:] * y[:-1]
    da_triton[0] = 0 * da_triton[0]
    assert torch.allclose(dx, dx_triton)
    assert torch.allclose(da, da_triton)

    torch.manual_seed(0)
    bs = 2; C = 3; H = 5; W = 5
    A = torch.randn(bs, C, H, W, device=device)
    X = torch.randn(bs, C, H, W, device=device)
    Y = discounted_cumsum_N_BCHW(A, X)
    dY = torch.randn(bs, C, H, W, device=device)
    dA, dX = discounted_cumsum_N_BCHW_gradient(A, X, Y, dY)

    grid = (bs, C, 1, 1)
    dX_triton = torch.zeros_like(X)
    discounted_cumsum_N_BCHW_gradient_triton_kernel[grid](
                A, dY, dX_triton,
                size_0=bs,
                size_1=C,
                size_2=H,
                size_3=W,
                stride_A_0=A.stride(0),
                stride_A_1=A.stride(1),
                stride_A_2=A.stride(2),
                stride_A_3=A.stride(3),
                stride_dX_0=dX.stride(0),
                stride_dX_1=dX.stride(1),
                stride_dX_2=dX.stride(2),
                stride_dX_3=dX.stride(3),
                BLOCK_SIZE_0=1, BLOCK_SIZE_1=1, BLOCK_SIZE_2=triton.next_power_of_2(H), BLOCK_SIZE_3=triton.next_power_of_2(W))
    torch.testing.assert_close(dX, dX_triton, rtol=1e-5, atol=1e-5)
    dA_triton = dX.clone()
    dA_triton[:,:,1:] = dA_triton[:,:,1:] * Y[:,:,:-1]
    dA_triton[:,:,0] = 0 * dA_triton[:,:,0]
    torch.testing.assert_close(dA, dA_triton, rtol=1e-5, atol=1e-5)
    
    # Test the launcher function
    dA_launcher, dX_launcher = discounted_cumsum_N_BCHW_gradient_triton(A, Y, dY, autotune=False)
    torch.testing.assert_close(dX, dX_launcher, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(dA, dA_launcher, rtol=1e-5, atol=1e-5)

    A = torch.randn(1, C, 1, 1, device=device)
    A = A.expand(bs, C, H, W)
    X = torch.randn(bs, C, H, W, device=device)
    Y = discounted_cumsum_N_BCHW(A, X)
    dY = torch.randn(bs, C, H, W, device=device)
    dA, dX = discounted_cumsum_N_BCHW_gradient(A, X, Y, dY)

    grid = (bs, C, 1, 1)
    dX_triton = torch.zeros_like(X)
    discounted_cumsum_N_BCHW_gradient_triton_kernel[grid](
                A, dY, dX_triton,
                size_0=bs,
                size_1=C,
                size_2=H,
                size_3=W,
                stride_A_0=A.stride(0),
                stride_A_1=A.stride(1),
                stride_A_2=A.stride(2),
                stride_A_3=A.stride(3),
                stride_dX_0=dX.stride(0),
                stride_dX_1=dX.stride(1),
                stride_dX_2=dX.stride(2),
                stride_dX_3=dX.stride(3),
                BLOCK_SIZE_0=1, BLOCK_SIZE_1=1, BLOCK_SIZE_2=triton.next_power_of_2(H), BLOCK_SIZE_3=triton.next_power_of_2(W))
    torch.testing.assert_close(dX, dX_triton, rtol=1e-5, atol=1e-5)
    dA_triton = dX.clone()
    dA_triton[:,:,1:] = dA_triton[:,:,1:] * Y[:,:,:-1]
    dA_triton[:,:,0] = 0 * dA_triton[:,:,0]
    torch.testing.assert_close(dA, dA_triton, rtol=1e-5, atol=1e-5)
    
    # Test the launcher function with expanded A
    dA_launcher, dX_launcher = discounted_cumsum_N_BCHW_gradient_triton(A, Y, dY, autotune=False)
    torch.testing.assert_close(dX, dX_launcher, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(dA, dA_launcher, rtol=1e-5, atol=1e-5)


def discounted_cumsum_1d_gradient(a, x, y, dy):
    T = len(a)
    da = torch.zeros_like(a)
    dx = torch.zeros_like(a)
    for t in range(T-1,-1,-1):
        if t == T-1:
            dx[T-1]   = dy[T-1]
            da[T-1]   = dx[T-1] * y[T-1-1]
        else:
            dx[t]     = dx[t+1] * a[t+1] + dy[t]
            da[t]     = dx[t] * y[t-1]
    da[0] = 0 * da[0]
    return da, dx

def discounted_cumsum_N_BCHW(A, X):
    X_cum = torch.zeros_like(X)
    for b in range(X.shape[0]):
        for c in range(X.shape[1]):
            for w in range(X.shape[3]):
                X_cum[b,c,:,w] = discounted_cumsum_1d_ref(A[b,c,:,w], X[b,c,:,w])
    return X_cum

def discounted_cumsum_N_BCHW_complex_ref(A, X):
    """
    Reference implementation for N_BCHW complex discount cumsum.
    
    Args:
        A: Discount factors tensor of shape (B, C, H, W, 2) where A[..., 0] is real and A[..., 1] is imaginary
        X: Input tensor of shape (B, C, H, W) for real or (B, C, H, W, 2) for complex where X[..., 0] is real and X[..., 1] is imaginary
    
    Returns:
        X_cum: Output tensor of shape (B, C, H, W, 2) where X_cum[..., 0] is real and X_cum[..., 1] is imaginary
    """
    # Detect if X is complex
    is_complex_X = len(X.shape) == 5 and X.shape[-1] == 2
    
    if is_complex_X:
        B, C, H, W = X.shape[:4]
    else:
        B, C, H, W = X.shape
    
    # Build result without in-place operations (needed for autograd)
    # Collect all results first, then stack
    results = []
    
    for b in range(B):
        for c in range(C):
            for w in range(W):
                a_real = A[b, c, :, w, 0]
                a_imag = A[b, c, :, w, 1]
                if is_complex_X:
                    # x is complex: shape (H, 2)
                    x = X[b, c, :, w, :]
                else:
                    # x is real: shape (H,)
                    x = X[b, c, :, w]
                x_cum_1d = discounted_cumsum_1d_complex_ref(a_real, a_imag, x)
                results.append(x_cum_1d)
    
    # Reshape results and stack into final tensor
    # results is a list of (H, 2) tensors, total B*C*W elements
    # We need to reshape to (B, C, H, W, 2)
    results_stacked = torch.stack(results)  # Shape: (B*C*W, H, 2)
    results_stacked = results_stacked.view(B, C, W, H, 2)  # Reshape
    results_stacked = results_stacked.permute(0, 1, 3, 2, 4)  # Permute to (B, C, H, W, 2)
    
    return results_stacked

def discounted_cumsum_1d_complex_gradient(a_real, a_imag, x_real, x_imag, y_real, y_imag, dy_real, dy_imag):
    """
    Reference implementation for 1D complex discount cumsum gradient.
    
    Args:
        a_real, a_imag: Real and imaginary parts of discount factors, shape (T,)
        x_real, x_imag: Real and imaginary parts of input values, shape (T,)
        y_real, y_imag: Real and imaginary parts of forward output, shape (T,)
        dy_real, dy_imag: Real and imaginary parts of output gradients, shape (T,)
    
    Returns:
        da_real, da_imag: Gradients w.r.t. discount factors, shape (T,)
        dx_real, dx_imag: Gradients w.r.t. input values, shape (T,)
    """
    T = len(a_real)
    da_real = torch.zeros_like(a_real)
    da_imag = torch.zeros_like(a_imag)
    dx_real = torch.zeros_like(x_real)
    dx_imag = torch.zeros_like(x_imag)
    
    # Reverse scan: process from end to beginning
    for t in range(T-1, -1, -1):
        if t == T-1:
            # Last element: dx[T-1] = dy[T-1]
            dx_real[T-1] = dy_real[T-1]
            dx_imag[T-1] = dy_imag[T-1]
            # da[T-1] = dx[T-1] * y[T-2] (complex multiplication)
            if T > 1:
                # Complex multiplication: dx * y_prev
                dx_r = dx_real[T-1]
                dx_i = dx_imag[T-1]
                y_prev_r = y_real[T-2]
                y_prev_i = y_imag[T-2]
                da_real[T-1] = dx_r * y_prev_r + dx_i * y_prev_i
                da_imag[T-1] =-dx_r * y_prev_i + dx_i * y_prev_r
        else:
            # dx[t] = dx[t+1] * a[t+1] + dy[t] (complex multiplication)
            dx_next_r = dx_real[t+1]
            dx_next_i = dx_imag[t+1]
            a_next_r = a_real[t+1]
            a_next_i = a_imag[t+1]
            
            # Complex multiplication: dx_next * a_next
            mult_r = dx_next_r * a_next_r + dx_next_i * a_next_i
            mult_i =-dx_next_r * a_next_i + dx_next_i * a_next_r
            
            # Add dy[t]
            dx_real[t] = mult_r + dy_real[t]
            dx_imag[t] = mult_i + dy_imag[t]
            
            # da[t] = dx[t] * y[t-1] (complex multiplication)
            if t > 0:
                dx_r = dx_real[t]
                dx_i = dx_imag[t]
                y_prev_r = y_real[t-1]
                y_prev_i = y_imag[t-1]
                da_real[t] = dx_r * y_prev_r + dx_i * y_prev_i
                da_imag[t] =-dx_r * y_prev_i + dx_i * y_prev_r
    
    # da[0] = 0 (no previous value)
    da_real[0] = 0.0
    da_imag[0] = 0.0
    
    return da_real, da_imag, dx_real, dx_imag

def discounted_cumsum_N_BCHW_complex_gradient(A, X, Y, dY):
    """
    Reference implementation for N_BCHW complex discount cumsum gradient.
    
    Args:
        A: Discount factors tensor of shape (B, C, H, W, 2) where A[..., 0] is real and A[..., 1] is imaginary
        X: Input tensor of shape (B, C, H, W) for real or (B, C, H, W, 2) for complex
        Y: Forward output tensor of shape (B, C, H, W, 2)
        dY: Output gradients tensor of shape (B, C, H, W, 2)
    
    Returns:
        dA: Gradient w.r.t. A, shape (B, C, H, W, 2)
        dX: Gradient w.r.t. X, shape matches X
    """
    # Detect if X is complex
    is_complex_X = len(X.shape) == 5 and X.shape[-1] == 2
    
    if is_complex_X:
        B, C, H, W = X.shape[:4]
        X_real = X[..., 0]
        X_imag = X[..., 1]
    else:
        B, C, H, W = X.shape
        X_real = X
        X_imag = torch.zeros_like(X)
    
    dA = torch.zeros_like(A)
    dX_real = torch.zeros_like(X_real)
    dX_imag = torch.zeros_like(X_imag)
    
    A_real = A[..., 0]
    A_imag = A[..., 1]
    Y_real = Y[..., 0]
    Y_imag = Y[..., 1]
    dY_real = dY[..., 0]
    dY_imag = dY[..., 1]
    
    for b in range(B):
        for c in range(C):
            for w in range(W):
                da_r, da_i, dx_r, dx_i = discounted_cumsum_1d_complex_gradient(
                    A_real[b, c, :, w], A_imag[b, c, :, w],
                    X_real[b, c, :, w], X_imag[b, c, :, w],
                    Y_real[b, c, :, w], Y_imag[b, c, :, w],
                    dY_real[b, c, :, w], dY_imag[b, c, :, w]
                )
                dA[b, c, :, w, 0] = da_r
                dA[b, c, :, w, 1] = da_i
                dX_real[b, c, :, w] = dx_r
                dX_imag[b, c, :, w] = dx_i
    
    if is_complex_X:
        dX = torch.stack([dX_real, dX_imag], dim=-1)
    else:
        dX = dX_real  # Only return real part if X was real
    
    return dA, dX

def discounted_cumsum_N_BCHW_gradient(A, X, Y, dY):
    dA = torch.zeros_like(A)
    dX = torch.zeros_like(X)
    for b in range(X.shape[0]):
        for c in range(X.shape[1]):
            for w in range(X.shape[3]):
                da, dx = discounted_cumsum_1d_gradient(A[b,c,:,w], X[b,c,:,w], Y[b,c,:,w], dY[b,c,:,w])
                dA[b,c,:,w] = da
                dX[b,c,:,w] = dx
    return dA, dX

def test_discounted_cumsum_gradient():
    # torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(0)
    a = torch.randn(10)
    a.requires_grad_(True)
    x = torch.randn(10)
    x.requires_grad_(True)
    dy = torch.randn(10)
    y = discounted_cumsum_1d_ref(a, x)
    torch.autograd.backward(y, dy)

    da, dx = discounted_cumsum_1d_gradient(a, x, y, dy)
    torch.testing.assert_close(x.grad, dx, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(a.grad, da, rtol=1e-5, atol=1e-5)

    bs = 2; C = 3; H = 5; W = 4;
    torch.manual_seed(0)
    A = torch.randn(bs, C, H, W)
    A.requires_grad_(True)
    X = torch.randn(bs, C, H, W)
    X.requires_grad_(True)
    Y = discounted_cumsum_N_BCHW(A, X)
    Y.requires_grad_(True)
    dY = torch.randn(bs, C, H, W)
    dY.requires_grad_(True)
    dA, dX = discounted_cumsum_N_BCHW_gradient(A, X, Y, dY)
    torch.autograd.backward(Y, dY)
    torch.testing.assert_close(A.grad, dA, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(X.grad, dX, rtol=1e-5, atol=1e-5)

def get_block_sizes_for_scan(shape, max_elements_per_block=4096):
    """
    NOT SURE IF THIS IS GOOD.
    Get block sizes for a 4D tensor where axis 2 is the scan dimension.
    
    Args:
        shape: (B, C, H, W) tensor shape
        max_elements_per_block: target max elements per thread block
    
    Returns:
        (BLOCK_SIZE_0, BLOCK_SIZE_1, BLOCK_SIZE_2, BLOCK_SIZE_3)
    """
    B, C, H, W = shape
    
    # BLOCK_SIZE_2 is fixed by the scan dimension
    BLOCK_SIZE_2 = triton.next_power_of_2(H)
    
    # Calculate budget for other dimensions
    budget = max(1, max_elements_per_block // BLOCK_SIZE_2)
    
    # Simple heuristic: distribute budget roughly equally
    # but cap at reasonable values (e.g., 32) and the actual dimension sizes
    budget_per_dim = int(budget ** (1/3))
    budget_per_dim = max(1, min(32, budget_per_dim))
    # Ensure budget_per_dim is a power of 2 for Triton compatibility
    budget_per_dim = triton.next_power_of_2(budget_per_dim)
    
    BLOCK_SIZE_0 = min(budget_per_dim, triton.next_power_of_2(B))
    BLOCK_SIZE_1 = min(budget_per_dim, triton.next_power_of_2(C))
    BLOCK_SIZE_3 = min(budget_per_dim, triton.next_power_of_2(W))
    
    # Safety check: if still too large, reduce further
    while BLOCK_SIZE_0 * BLOCK_SIZE_1 * BLOCK_SIZE_2 * BLOCK_SIZE_3 > max_elements_per_block * 2:
        if BLOCK_SIZE_0 > 1:
            BLOCK_SIZE_0 //= 2
        elif BLOCK_SIZE_1 > 1:
            BLOCK_SIZE_1 //= 2
        elif BLOCK_SIZE_3 > 1:
            BLOCK_SIZE_3 //= 2
        else:
            break
    
    return BLOCK_SIZE_0, BLOCK_SIZE_1, BLOCK_SIZE_2, BLOCK_SIZE_3

def discounted_cumsum_N_BCHW_triton(A, X, autotune=False):
    X_cum = torch.zeros_like(X)
    # Remove contiguous assertions since we now handle non-contiguous tensors
    grid = lambda meta: (triton.cdiv(X.shape[0], meta['BLOCK_SIZE_0']),
                         triton.cdiv(X.shape[1], meta['BLOCK_SIZE_1']),
                         triton.cdiv(X.shape[3], meta['BLOCK_SIZE_3']))
    if autotune:
        discounted_cumsum_N_BCHW_triton_kernel[grid](A, X, X_cum,
                                    size_0=X.shape[0],
                                    size_1=X.shape[1],
                                    size_2=X.shape[2],
                                    size_3=X.shape[3],
                                    stride_A_0=A.stride(0),
                                    stride_A_1=A.stride(1),
                                    stride_A_2=A.stride(2),
                                    stride_A_3=A.stride(3),
                                    stride_X_0=X.stride(0),
                                    stride_X_1=X.stride(1),
                                    stride_X_2=X.stride(2),
                                    stride_X_3=X.stride(3),
                                    BLOCK_SIZE_2=triton.next_power_of_2(X.shape[2]),
                                    )
    else:
        BLOCK_SIZE_0, BLOCK_SIZE_1, BLOCK_SIZE_2, BLOCK_SIZE_3 = get_block_sizes_for_scan(X.shape)
        discounted_cumsum_N_BCHW_triton_kernel.fn[grid](A, X, X_cum,
                                    size_0=X.shape[0],
                                    size_1=X.shape[1],
                                    size_2=X.shape[2],
                                    size_3=X.shape[3],
                                    stride_A_0=A.stride(0),
                                    stride_A_1=A.stride(1),
                                    stride_A_2=A.stride(2),
                                    stride_A_3=A.stride(3),
                                    stride_X_0=X.stride(0),
                                    stride_X_1=X.stride(1),
                                    stride_X_2=X.stride(2),
                                    stride_X_3=X.stride(3),
                                    BLOCK_SIZE_0=BLOCK_SIZE_0,
                                    BLOCK_SIZE_1=BLOCK_SIZE_1,
                                    BLOCK_SIZE_2=BLOCK_SIZE_2,
                                    BLOCK_SIZE_3=BLOCK_SIZE_3,)
    return X_cum

def discounted_cumsum_N_BCHW_complex_triton(A, X, autotune=False):
    """
    Complex discount cumsum launcher.
    
    Args:
        A: Discount factors tensor of shape (B, C, H, W, 2) where A[..., 0] is real and A[..., 1] is imaginary
        X: Input tensor of shape (B, C, H, W) for real values, or (B, C, H, W, 2) for complex values
        autotune: Whether to use autotuning
    
    Returns:
        X_cum: Output tensor of shape (B, C, H, W, 2) where X_cum[..., 0] is real and X_cum[..., 1] is imaginary
    """
    # Input validation
    assert len(A.shape) == 5, f"A must have shape (B, C, H, W, 2), got {A.shape}"
    assert A.shape[-1] == 2, f"Last dimension of A must be 2 (real, imag), got {A.shape[-1]}"
    assert len(X.shape) in [4, 5], f"X must have shape (B, C, H, W) or (B, C, H, W, 2), got {X.shape}"
    
    # Detect if X is complex
    is_complex_X = len(X.shape) == 5 and X.shape[-1] == 2
    
    if is_complex_X:
        # X is complex: shape (B, C, H, W, 2)
        assert A.shape[:4] == X.shape[:4], f"A and X must have matching first 4 dimensions, got A{A.shape[:4]} vs X{X.shape[:4]}"
        B, C, H, W = X.shape[:4]
        X_real = X[..., 0]  # Shape: (B, C, H, W)
        X_imag = X[..., 1]  # Shape: (B, C, H, W)
    else:
        # X is real: shape (B, C, H, W)
        assert A.shape[:4] == X.shape, f"A and X must have matching first 4 dimensions, got A{A.shape[:4]} vs X{X.shape}"
        B, C, H, W = X.shape
        X_real = X  # Shape: (B, C, H, W)
        X_imag = torch.zeros_like(X)  # Shape: (B, C, H, W) - zeros for real input
    
    # Split A into real and imaginary parts
    A_real = A[..., 0]  # Shape: (B, C, H, W)
    A_imag = A[..., 1]  # Shape: (B, C, H, W)
    
    # Create output tensors for real and imaginary parts
    X_cum_real = torch.zeros_like(X_real)  # Shape: (B, C, H, W)
    X_cum_imag = torch.zeros_like(X_real)  # Shape: (B, C, H, W)
    
    grid = lambda meta: (triton.cdiv(X_real.shape[0], meta['BLOCK_SIZE_0']),
                         triton.cdiv(X_real.shape[1], meta['BLOCK_SIZE_1']),
                         triton.cdiv(X_real.shape[3], meta['BLOCK_SIZE_3']))
    
    if autotune:
        discounted_cumsum_N_BCHW_complex_triton_kernel[grid](
            A_real, A_imag, X_real, X_imag, X_cum_real, X_cum_imag,
            size_0=X_real.shape[0],
            size_1=X_real.shape[1],
            size_2=X_real.shape[2],
            size_3=X_real.shape[3],
            stride_A_0=A_real.stride(0),
            stride_A_1=A_real.stride(1),
            stride_A_2=A_real.stride(2),
            stride_A_3=A_real.stride(3),
            stride_X_0=X_real.stride(0),
            stride_X_1=X_real.stride(1),
            stride_X_2=X_real.stride(2),
            stride_X_3=X_real.stride(3),
            stride_X_cum_0=X_cum_real.stride(0),
            stride_X_cum_1=X_cum_real.stride(1),
            stride_X_cum_2=X_cum_real.stride(2),
            stride_X_cum_3=X_cum_real.stride(3),
            BLOCK_SIZE_2=triton.next_power_of_2(X_real.shape[2]),
        )
    else:
        BLOCK_SIZE_0, BLOCK_SIZE_1, BLOCK_SIZE_2, BLOCK_SIZE_3 = get_block_sizes_for_scan(X_real.shape, max_elements_per_block=3_000)
        discounted_cumsum_N_BCHW_complex_triton_kernel.fn[grid](
            A_real, A_imag, X_real, X_imag, X_cum_real, X_cum_imag,
            size_0=X_real.shape[0],
            size_1=X_real.shape[1],
            size_2=X_real.shape[2],
            size_3=X_real.shape[3],
            stride_A_0=A_real.stride(0),
            stride_A_1=A_real.stride(1),
            stride_A_2=A_real.stride(2),
            stride_A_3=A_real.stride(3),
            stride_X_0=X_real.stride(0),
            stride_X_1=X_real.stride(1),
            stride_X_2=X_real.stride(2),
            stride_X_3=X_real.stride(3),
            stride_X_cum_0=X_cum_real.stride(0),
            stride_X_cum_1=X_cum_real.stride(1),
            stride_X_cum_2=X_cum_real.stride(2),
            stride_X_cum_3=X_cum_real.stride(3),
            BLOCK_SIZE_0=BLOCK_SIZE_0,
            BLOCK_SIZE_1=BLOCK_SIZE_1,
            BLOCK_SIZE_2=BLOCK_SIZE_2,
            BLOCK_SIZE_3=BLOCK_SIZE_3,
        )
    
    # Stack real and imaginary parts into shape (B, C, H, W, 2)
    X_cum = torch.stack([X_cum_real, X_cum_imag], dim=-1)
    return X_cum

def discounted_cumsum_N_BCHW_gradient_triton(A, Y, dY, autotune=False):
    """
    Launcher for discounted_cumsum_N_BCHW_gradient_triton_kernel.
    
    Args:
        A: Discount factors tensor of shape (B, C, H, W)
        Y: Forward pass output tensor of shape (B, C, H, W), needed for dA computation
        dY: Gradient w.r.t. output tensor of shape (B, C, H, W)
        autotune: Whether to use autotuning (currently not supported for gradient kernel)
    
    Returns:
        dA: Gradient w.r.t. A tensor of shape (B, C, H, W)
        dX: Gradient w.r.t. X tensor of shape (B, C, H, W)
    """
    # Verify that dY and Y have the same shape
    assert dY.shape == Y.shape, f"dY shape {dY.shape} must match Y shape {Y.shape}"
    assert dY.shape == A.shape, f"dY shape {dY.shape} must match A shape {A.shape}"
    
    # Create dX with the same shape and strides as dY to ensure they match
    # This verifies the assumption that dX and dY have the same strides
    dX = torch.zeros_like(dY)
    
    # Verify strides match (the kernel uses dX strides for both dX and dY)
    assert dX.stride() == dY.stride(), \
        f"dX strides {dX.stride()} must match dY strides {dY.stride()}"
    
    grid = lambda meta: (triton.cdiv(dY.shape[0], meta['BLOCK_SIZE_0']),
                         triton.cdiv(dY.shape[1], meta['BLOCK_SIZE_1']),
                         triton.cdiv(dY.shape[3], meta['BLOCK_SIZE_3']))
    
    if autotune:
        # Note: gradient kernel doesn't have autotune decorator yet, so this will use default
        discounted_cumsum_N_BCHW_gradient_triton_kernel[grid](
            A, dY, dX,
            size_0=dY.shape[0],
            size_1=dY.shape[1],
            size_2=dY.shape[2],
            size_3=dY.shape[3],
            stride_A_0=A.stride(0),
            stride_A_1=A.stride(1),
            stride_A_2=A.stride(2),
            stride_A_3=A.stride(3),
            stride_dX_0=dX.stride(0),
            stride_dX_1=dX.stride(1),
            stride_dX_2=dX.stride(2),
            stride_dX_3=dX.stride(3),
            BLOCK_SIZE_2=triton.next_power_of_2(dY.shape[2]),
        )
    else:
        BLOCK_SIZE_0, BLOCK_SIZE_1, BLOCK_SIZE_2, BLOCK_SIZE_3 = get_block_sizes_for_scan(dY.shape)
        discounted_cumsum_N_BCHW_gradient_triton_kernel[grid](
            A, dY, dX,
            size_0=dY.shape[0],
            size_1=dY.shape[1],
            size_2=dY.shape[2],
            size_3=dY.shape[3],
            stride_A_0=A.stride(0),
            stride_A_1=A.stride(1),
            stride_A_2=A.stride(2),
            stride_A_3=A.stride(3),
            stride_dX_0=dX.stride(0),
            stride_dX_1=dX.stride(1),
            stride_dX_2=dX.stride(2),
            stride_dX_3=dX.stride(3),
            BLOCK_SIZE_0=BLOCK_SIZE_0,
            BLOCK_SIZE_1=BLOCK_SIZE_1,
            BLOCK_SIZE_2=BLOCK_SIZE_2,
            BLOCK_SIZE_3=BLOCK_SIZE_3,
        )
    
    # Post-processing to obtain dA from dX and Y
    # dA[t] = dX[t] * Y[t-1] for t > 0, and dA[0] = 0
    dA = dX.clone()
    dA[:, :, 1:] = dA[:, :, 1:] * Y[:, :, :-1]
    dA[:, :, 0] = 0 * dA[:, :, 0]
    
    return dA, dX

def discounted_cumsum_N_BCHW_complex_gradient_triton(A, Y, dY, autotune=False):
    """
    Launcher for discounted_cumsum_N_BCHW_complex_gradient_triton_kernel.
    
    Args:
        A: Discount factors tensor of shape (B, C, H, W, 2) where A[..., 0] is real and A[..., 1] is imaginary
        Y: Forward pass output tensor of shape (B, C, H, W, 2), needed for dA computation
        dY: Gradient w.r.t. output tensor of shape (B, C, H, W, 2)
        autotune: Whether to use autotuning (currently not supported for gradient kernel)
    
    Returns:
        dA: Gradient w.r.t. A tensor of shape (B, C, H, W, 2)
        dX: Gradient w.r.t. X tensor of shape (B, C, H, W, 2) if X was complex, else (B, C, H, W)
    """
    # Input validation
    assert len(A.shape) == 5 and A.shape[-1] == 2, f"A must have shape (B, C, H, W, 2), got {A.shape}"
    assert len(Y.shape) == 5 and Y.shape[-1] == 2, f"Y must have shape (B, C, H, W, 2), got {Y.shape}"
    assert len(dY.shape) == 5 and dY.shape[-1] == 2, f"dY must have shape (B, C, H, W, 2), got {dY.shape}"
    assert A.shape[:4] == Y.shape[:4] == dY.shape[:4], "A, Y, and dY must have matching first 4 dimensions"
    
    B, C, H, W = dY.shape[:4]
    
    # Split A, Y, dY into real and imaginary parts
    A_real = A[..., 0]  # Shape: (B, C, H, W)
    A_imag = A[..., 1]  # Shape: (B, C, H, W)
    Y_real = Y[..., 0]  # Shape: (B, C, H, W)
    Y_imag = Y[..., 1]  # Shape: (B, C, H, W)
    dY_real = dY[..., 0]  # Shape: (B, C, H, W)
    dY_imag = dY[..., 1]  # Shape: (B, C, H, W)
    
    # Create dX output tensors (always complex for now, we'll handle real X case later)
    dX_real = torch.zeros_like(dY_real)  # Shape: (B, C, H, W)
    dX_imag = torch.zeros_like(dY_imag)  # Shape: (B, C, H, W)
    
    grid = lambda meta: (triton.cdiv(dY_real.shape[0], meta['BLOCK_SIZE_0']),
                         triton.cdiv(dY_real.shape[1], meta['BLOCK_SIZE_1']),
                         triton.cdiv(dY_real.shape[3], meta['BLOCK_SIZE_3']))
    
    if autotune:
        # Note: gradient kernel doesn't have autotune decorator yet, so this will use default
        discounted_cumsum_N_BCHW_complex_gradient_triton_kernel[grid](
            A_real, A_imag, dY_real, dY_imag, dX_real, dX_imag,
            size_0=dY_real.shape[0],
            size_1=dY_real.shape[1],
            size_2=dY_real.shape[2],
            size_3=dY_real.shape[3],
            stride_A_0=A_real.stride(0),
            stride_A_1=A_real.stride(1),
            stride_A_2=A_real.stride(2),
            stride_A_3=A_real.stride(3),
            stride_dY_0=dY_real.stride(0),
            stride_dY_1=dY_real.stride(1),
            stride_dY_2=dY_real.stride(2),
            stride_dY_3=dY_real.stride(3),
            stride_dX_0=dX_real.stride(0),
            stride_dX_1=dX_real.stride(1),
            stride_dX_2=dX_real.stride(2),
            stride_dX_3=dX_real.stride(3),
            BLOCK_SIZE_2=triton.next_power_of_2(dY_real.shape[2]),
        )
    else:
        BLOCK_SIZE_0, BLOCK_SIZE_1, BLOCK_SIZE_2, BLOCK_SIZE_3 = get_block_sizes_for_scan(dY_real.shape, max_elements_per_block=3_000)
        discounted_cumsum_N_BCHW_complex_gradient_triton_kernel[grid](
            A_real, A_imag, dY_real, dY_imag, dX_real, dX_imag,
            size_0=dY_real.shape[0],
            size_1=dY_real.shape[1],
            size_2=dY_real.shape[2],
            size_3=dY_real.shape[3],
            stride_A_0=A_real.stride(0),
            stride_A_1=A_real.stride(1),
            stride_A_2=A_real.stride(2),
            stride_A_3=A_real.stride(3),
            stride_dY_0=dY_real.stride(0),
            stride_dY_1=dY_real.stride(1),
            stride_dY_2=dY_real.stride(2),
            stride_dY_3=dY_real.stride(3),
            stride_dX_0=dX_real.stride(0),
            stride_dX_1=dX_real.stride(1),
            stride_dX_2=dX_real.stride(2),
            stride_dX_3=dX_real.stride(3),
            BLOCK_SIZE_0=BLOCK_SIZE_0,
            BLOCK_SIZE_1=BLOCK_SIZE_1,
            BLOCK_SIZE_2=BLOCK_SIZE_2,
            BLOCK_SIZE_3=BLOCK_SIZE_3,
        )
    
    # Post-processing to obtain dA from dX and Y (complex multiplication)
    # dA[t] = dX[t] * Y[t-1] for t > 0, and dA[0] = 0
    dA_real = torch.zeros_like(A_real)
    dA_imag = torch.zeros_like(A_imag)
    
    # Complex multiplication: dX[t] * Y[t-1]
    # Fixed formula: (dx_r + i*dx_i) * (y_r + i*y_i) = (dx_r*y_r + dx_i*y_i) + i*(-dx_r*y_i + dx_i*y_r)
    if H > 1:
        dX_real_shifted = dX_real[:, :, 1:, :]  # Shape: (B, C, H-1, W)
        dX_imag_shifted = dX_imag[:, :, 1:, :]  # Shape: (B, C, H-1, W)
        Y_real_prev = Y_real[:, :, :-1, :]  # Shape: (B, C, H-1, W)
        Y_imag_prev = Y_imag[:, :, :-1, :]  # Shape: (B, C, H-1, W)
        
        dA_real[:, :, 1:, :] = dX_real_shifted * Y_real_prev + dX_imag_shifted * Y_imag_prev
        dA_imag[:, :, 1:, :] = -dX_real_shifted * Y_imag_prev + dX_imag_shifted * Y_real_prev
    
    # dA[0] = 0 (no previous value)
    dA_real[:, :, 0, :] = 0.0
    dA_imag[:, :, 0, :] = 0.0
    
    # Stack dA into shape (B, C, H, W, 2)
    dA = torch.stack([dA_real, dA_imag], dim=-1)
    
    # Stack dX into shape (B, C, H, W, 2)
    # Note: We always return complex dX, even if X was real (imag part will be zeros)
    dX = torch.stack([dX_real, dX_imag], dim=-1)
    
    return dA, dX

class DiscountedCumsumN_BCHW(torch.autograd.Function):
    """
    PyTorch autograd Function for discounted_cumsum_N_BCHW using Triton kernels.
    """
    @staticmethod
    def forward(ctx, A, X, autotune=False):
        """
        Forward pass for discounted cumulative sum.
        
        Args:
            ctx: Context object to save tensors for backward
            A: Discount factors tensor of shape (B, C, H, W)
            X: Input tensor of shape (B, C, H, W)
            autotune: Whether to use autotuning for the forward kernel
        
        Returns:
            Y: Output tensor of shape (B, C, H, W)
        """
        Y = discounted_cumsum_N_BCHW_triton(A, X, autotune=autotune)
        # Save A and Y for backward pass
        ctx.save_for_backward(A, Y)
        ctx.autotune = autotune
        return Y
    
    @staticmethod
    def backward(ctx, dY):
        """
        Backward pass for discounted cumulative sum.
        
        Args:
            ctx: Context object with saved tensors
            dY: Gradient w.r.t. output tensor of shape (B, C, H, W)
        
        Returns:
            dA: Gradient w.r.t. A tensor of shape (B, C, H, W)
            dX: Gradient w.r.t. X tensor of shape (B, C, H, W)
        """
        A, Y = ctx.saved_tensors
        dA, dX = discounted_cumsum_N_BCHW_gradient_triton(A, Y, dY, autotune=ctx.autotune)
        return dA, dX, None # None: Gradient w.r.t. autotune (not needed)

class DiscountedCumsumN_BCHW_Complex(torch.autograd.Function):
    """
    PyTorch autograd Function for discounted_cumsum_N_BCHW with complex discounts using Triton kernels.
    """
    @staticmethod
    def forward(ctx, A, X, autotune=False):
        """
        Forward pass for complex discount cumulative sum.
        
        Args:
            ctx: Context object to save tensors for backward
            A: Discount factors tensor of shape (B, C, H, W, 2) where A[..., 0] is real and A[..., 1] is imaginary
            X: Input tensor of shape (B, C, H, W) for real or (B, C, H, W, 2) for complex
            autotune: Whether to use autotuning for the forward kernel
        
        Returns:
            Y: Output tensor of shape (B, C, H, W, 2) where Y[..., 0] is real and Y[..., 1] is imaginary
        """
        Y = discounted_cumsum_N_BCHW_complex_triton(A, X, autotune=autotune)
        # Save A, Y, and X shape info for backward pass
        ctx.save_for_backward(A, Y)
        ctx.autotune = autotune
        # Save whether X was complex to return correct dX shape
        ctx.is_complex_X = len(X.shape) == 5 and X.shape[-1] == 2
        return Y
    
    @staticmethod
    def backward(ctx, dY):
        """
        Backward pass for complex discount cumulative sum.
        
        Args:
            ctx: Context object with saved tensors
            dY: Gradient w.r.t. output tensor of shape (B, C, H, W, 2)
        
        Returns:
            dA: Gradient w.r.t. A tensor of shape (B, C, H, W, 2)
            dX: Gradient w.r.t. X tensor (shape matches input X)
            None: Gradient w.r.t. autotune (not needed)
        """
        A, Y = ctx.saved_tensors
        dA, dX = discounted_cumsum_N_BCHW_complex_gradient_triton(A, Y, dY, autotune=ctx.autotune)
        
        # If X was real, return only the real part of dX
        if not ctx.is_complex_X:
            dX = dX[..., 0]  # Extract real part, shape (B, C, H, W)
        
        return dA, dX, None  # None: Gradient w.r.t. autotune (not needed)

def test_discounted_cumsum_N_BCHW_triton():
    X = torch.tensor([1., 2., 3., 4.], device='cuda').reshape(1, 1, 4, 1)
    A = torch.tensor([0., 7., 0., 1.], device='cuda').reshape(1, 1, 4, 1)
    expected = torch.tensor([1., 9., 3., 7.], device='cuda').reshape(1, 1, 4, 1)
    
    X_cum = discounted_cumsum_N_BCHW_triton(A, X, autotune=False)
    torch.testing.assert_close(X_cum, expected, rtol=1e-5, atol=1e-5)
    
    torch.manual_seed(0)
    X = torch.randint(0, 10, (2, 3, 5, 4), device='cuda').float()
    A = torch.randint(-5, 5, (2, 3, 5, 4), device='cuda').float()
    
    expected = discounted_cumsum_N_BCHW(A, X)
    X_cum = discounted_cumsum_N_BCHW_triton(A, X, autotune=False)
    torch.testing.assert_close(X_cum, expected, rtol=1e-5, atol=1e-5)

    bs = 2; C = 3; H = 5; W = 4;
    torch.manual_seed(0)
    A = torch.randn(bs, C, H, W, device='cuda')
    X = torch.randn(bs, C, H, W, device='cuda')
    expected = discounted_cumsum_N_BCHW(A, X)
    X_cum = discounted_cumsum_N_BCHW_triton(A, X, autotune=False)
    torch.testing.assert_close(X_cum, expected, rtol=1e-5, atol=1e-5)

    A = torch.randn(1, C, 1, 1, device='cuda')
    A = A.expand(bs, C, H, W)
    X = torch.randn(bs, C, H, W, device='cuda')
    expected = discounted_cumsum_N_BCHW(A, X)
    X_cum = discounted_cumsum_N_BCHW_triton(A, X, autotune=False)
    torch.testing.assert_close(X_cum, expected, rtol=1e-5, atol=1e-5)
    
    # Test with autotune
    # X_cum = discounted_cumsum_N_BCHW_triton(A, X, autotune=True)
    # torch.testing.assert_close(X_cum, expected, rtol=1e-5, atol=1e-5)

def test_discounted_cumsum_N_BCHW_triton_backprop():
    X = torch.tensor([1., 2., 3., 4.], device='cuda').reshape(1, 1, 4, 1)
    X.requires_grad = True
    A = torch.tensor([0., 7., 0., 1.], device='cuda').reshape(1, 1, 4, 1)
    expected = torch.tensor([1., 9., 3., 7.], device='cuda').reshape(1, 1, 4, 1)
    
    X_cum = discounted_cumsum_N_BCHW_triton(A, X, autotune=False)
    torch.testing.assert_close(X_cum, expected, rtol=1e-5, atol=1e-5)
    fake_loss = (X_cum).sum()
    import pytest
    with pytest.raises(RuntimeError):
        fake_loss.backward()

def test_discounted_cumsum_N_BCHW_autograd():
    """Test the autograd Function for discounted_cumsum_N_BCHW."""
    device = 'cuda'
    torch.manual_seed(0)
    
    # Test 1: Basic forward and backward
    bs, C, H, W = 2, 3, 5, 4
    A = torch.randn(bs, C, H, W, device=device, requires_grad=True)
    X = torch.randn(bs, C, H, W, device=device, requires_grad=True)
    
    # Use the autograd Function
    Y = DiscountedCumsumN_BCHW.apply(A, X, False)
    
    # Verify forward pass
    Y_ref = discounted_cumsum_N_BCHW(A.detach(), X.detach())
    torch.testing.assert_close(Y, Y_ref, rtol=1e-5, atol=1e-5)
    
    # Test backward pass
    dY = torch.randn_like(Y)
    loss = (Y * dY).sum()
    loss.backward()
    
    # Compute reference gradients
    A_ref = A.detach().clone().requires_grad_(True)
    X_ref = X.detach().clone().requires_grad_(True)
    Y_ref = discounted_cumsum_N_BCHW(A_ref, X_ref)
    loss_ref = (Y_ref * dY).sum()
    loss_ref.backward()
    
    # Verify gradients
    torch.testing.assert_close(A.grad, A_ref.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(X.grad, X_ref.grad, rtol=1e-5, atol=1e-5)
    
    # Test 2: With expanded A
    torch.manual_seed(1)
    A_expanded = torch.randn(1, C, 1, 1, device=device, requires_grad=True)
    A_expanded_tensor = A_expanded.expand(bs, C, H, W)
    X2 = torch.randn(bs, C, H, W, device=device, requires_grad=True)
    
    Y2 = DiscountedCumsumN_BCHW.apply(A_expanded_tensor, X2, False)
    dY2 = torch.randn_like(Y2)
    loss2 = (Y2 * dY2).sum()
    loss2.backward()
    
    # Reference
    A_ref2 = A_expanded.detach().clone().requires_grad_(True)
    A_ref2_tensor = A_ref2.expand(bs, C, H, W)
    X_ref2 = X2.detach().clone().requires_grad_(True)
    Y_ref2 = discounted_cumsum_N_BCHW(A_ref2_tensor, X_ref2)
    loss_ref2 = (Y_ref2 * dY2).sum()
    loss_ref2.backward()
    
    # Verify gradients (note: for expanded tensors, we need to sum the gradient)
    torch.testing.assert_close(A_expanded.grad, A_ref2.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(X2.grad, X_ref2.grad, rtol=1e-5, atol=1e-5)

def test_discounted_cumsum_N_BCHW_complex_triton():
    """Test complex discount cumsum implementation."""
    device = 'cuda'
    
    # Test 1: Simple known values
    X = torch.tensor([1., 2., 3., 4.], device=device).reshape(1, 1, 4, 1)
    # A with complex values: [0+0i, 1+1i, 0+0i, 1+0i]
    A = torch.zeros(1, 1, 4, 1, 2, device=device)
    A[0, 0, 0, 0, 0] = 0.0  # real
    A[0, 0, 0, 0, 1] = 0.0  # imag
    A[0, 0, 1, 0, 0] = 1.0  # real
    A[0, 0, 1, 0, 1] = 1.0  # imag
    A[0, 0, 2, 0, 0] = 0.0  # real
    A[0, 0, 2, 0, 1] = 0.0  # imag
    A[0, 0, 3, 0, 0] = 1.0  # real
    A[0, 0, 3, 0, 1] = 0.0  # imag
    
    expected = discounted_cumsum_N_BCHW_complex_ref(A, X)
    X_cum = discounted_cumsum_N_BCHW_complex_triton(A, X, autotune=False)
    torch.testing.assert_close(X_cum, expected, rtol=1e-5, atol=1e-5)
    
    # Test 2: Random values
    torch.manual_seed(0)
    bs = 2
    C = 3
    H = 5
    W = 4
    X = torch.randn(bs, C, H, W, device=device)
    A = torch.randn(bs, C, H, W, 2, device=device)
    
    expected = discounted_cumsum_N_BCHW_complex_ref(A, X)
    X_cum = discounted_cumsum_N_BCHW_complex_triton(A, X, autotune=False)
    torch.testing.assert_close(X_cum, expected, rtol=1e-5, atol=1e-5)
    
    # Test 3: With expanded A
    A_small = torch.randn(1, C, 1, 1, 2, device=device)
    A_expanded = A_small.expand(bs, C, H, W, 2)
    X = torch.randn(bs, C, H, W, device=device)
    
    expected = discounted_cumsum_N_BCHW_complex_ref(A_expanded, X)
    X_cum = discounted_cumsum_N_BCHW_complex_triton(A_expanded, X, autotune=False)
    torch.testing.assert_close(X_cum, expected, rtol=1e-5, atol=1e-5)
    
    # Test 4: Larger shapes
    torch.manual_seed(1)
    bs = 4
    C = 8
    H = 16
    W = 16
    X = torch.randn(bs, C, H, W, device=device)
    A = torch.randn(bs, C, H, W, 2, device=device)
    
    expected = discounted_cumsum_N_BCHW_complex_ref(A, X)
    X_cum = discounted_cumsum_N_BCHW_complex_triton(A, X, autotune=False)
    torch.testing.assert_close(X_cum, expected, rtol=1e-5, atol=1e-5)
    
    print("All complex discount cumsum tests passed!")

def test_discounted_cumsum_N_BCHW_complex_triton_with_complex_X():
    """Test complex discount cumsum with complex X inputs."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test 1: Complex X with imag=0 should match real X
    torch.manual_seed(0)
    bs = 2
    C = 3
    H = 5
    W = 4
    X_real = torch.randn(bs, C, H, W, device=device)
    X_complex = torch.zeros(bs, C, H, W, 2, device=device)
    X_complex[..., 0] = X_real
    X_complex[..., 1] = 0.0  # imag = 0
    
    A = torch.randn(bs, C, H, W, 2, device=device)
    
    result_real_X = discounted_cumsum_N_BCHW_complex_triton(A, X_real, autotune=False)
    result_complex_X = discounted_cumsum_N_BCHW_complex_triton(A, X_complex, autotune=False)
    
    torch.testing.assert_close(result_real_X, result_complex_X, rtol=1e-5, atol=1e-5,
                               msg="Complex X with imag=0 should match real X")
    
    # Test 2: Complex X with non-zero imag
    torch.manual_seed(1)
    X_complex = torch.randn(bs, C, H, W, 2, device=device)
    A = torch.randn(bs, C, H, W, 2, device=device)
    
    result = discounted_cumsum_N_BCHW_complex_triton(A, X_complex, autotune=False)
    assert result.shape == (bs, C, H, W, 2), f"Expected shape {(bs, C, H, W, 2)}, got {result.shape}"
    assert not torch.allclose(result, torch.zeros_like(result)), "Result should not be all zeros"
    
    # Test 3: Composite operation simulation (NW) - first N, then W on the result
    torch.manual_seed(2)
    X = torch.randn(bs, C, H, W, device=device)
    A = torch.randn(bs, C, H, W, 2, device=device)
    
    # First apply N direction (real X -> complex result)
    result_N = discounted_cumsum_N_BCHW_complex_triton(A, X, autotune=False)
    assert result_N.shape == (bs, C, H, W, 2), "Result from N should be complex"

    # Test 4: Complex X with real=0
    torch.manual_seed(3)
    X_complex = torch.zeros(bs, C, H, W, 2, device=device)
    X_complex[..., 0] = 0.0  # real = 0

    A_real = torch.randn(bs, C, H, W, 2, device=device)
    A_real[..., 1] = 0.0  # complex part = 0

    result_N = discounted_cumsum_N_BCHW_complex_triton(A_real, X_complex, autotune=False)
    result_N_real = discounted_cumsum_N_BCHW_triton(A_real[:, :, :, :, 0], X_complex[:, :, :, :, 0], autotune=False)
    torch.testing.assert_close(result_N_real, result_N[:, :, :, :, 0], rtol=1e-5, atol=1e-5)

    # Test 5: General X, A
    torch.manual_seed(4)
    bs = 2
    C = 3
    H = 5
    W = 4
    X = torch.randn(bs, C, H, W, device=device)
    A = torch.randn(bs, C, H, W, 2, device=device)
    
    result = discounted_cumsum_N_BCHW_complex_triton(A, X, autotune=False)
    result_ref = discounted_cumsum_N_BCHW_complex_ref(A, X)
    torch.testing.assert_close(result, result_ref, rtol=1e-5, atol=1e-5)

    ## Then apply W direction on the complex result (simulating NW)
    ## Need to transpose both A and result_N for W direction
    #A_W = A.transpose(-3, -2)  # Transpose H and W: (B, C, W, H, 2)
    #result_N_transposed = result_N.transpose(-3, -2)  # Transpose H and W: (B, C, W, H, 2)
    #result_NW = discounted_cumsum_N_BCHW_complex_triton(A_W, result_N_transposed, autotune=False)
    #assert result_NW.shape == (bs, C, W, H, 2), "Result from W on complex input should be complex"
    #assert not torch.allclose(result_NW, torch.zeros_like(result_NW)), "NW result should not be all zeros"

def test_discounted_cumsum_1d_complex_ref():
    """Test the 1D complex discount cumsum reference implementation."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test 1: Simple known values with real x
    torch.manual_seed(0)
    T = 4
    a_real = torch.tensor([0.0, 1.0, 0.5, 0.8], device=device)
    a_imag = torch.tensor([0.0, 0.5, 0.3, 0.2], device=device)
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    
    result = discounted_cumsum_1d_complex_ref(a_real, a_imag, x)
    
    # Verify shape
    assert result.shape == (T, 2), f"Expected shape ({T}, 2), got {result.shape}"
    
    # Manual computation for first few elements:
    # x_cum[0] = x[0] = 1.0 + 0.0i
    # x_cum[1] = x_cum[0] * (1.0 + 0.5i) + x[1]
    #          = (1.0 + 0.0i) * (1.0 + 0.5i) + 2.0
    #          = (1.0 + 0.5i) + 2.0 = 3.0 + 0.5i
    expected_0_real = 1.0
    expected_0_imag = 0.0
    expected_1_real = 1.0 * 1.0 - 0.0 * 0.5 + 2.0  # = 3.0
    expected_1_imag = 1.0 * 0.5 + 0.0 * 1.0 + 0.0  # = 0.5
    
    torch.testing.assert_close(result[0, 0], torch.tensor(expected_0_real, device=device), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(result[0, 1], torch.tensor(expected_0_imag, device=device), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(result[1, 0], torch.tensor(expected_1_real, device=device), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(result[1, 1], torch.tensor(expected_1_imag, device=device), rtol=1e-5, atol=1e-5)
    
    # Test 2: Complex x with imag=0 should match real x
    torch.manual_seed(1)
    T = 5
    a_real = torch.randn(T, device=device)
    a_imag = torch.randn(T, device=device)
    x_real = torch.randn(T, device=device)
    x_complex = torch.zeros(T, 2, device=device)
    x_complex[:, 0] = x_real
    x_complex[:, 1] = 0.0  # imag = 0
    
    # discounted_cumsum_1d_ref()
    result_real_x = discounted_cumsum_1d_complex_ref(a_real, a_imag, x_real)
    result_complex_x_zero_imag = discounted_cumsum_1d_complex_ref(a_real, a_imag, x_complex)
    
    torch.testing.assert_close(result_real_x, result_complex_x_zero_imag, rtol=1e-5, atol=1e-5,
                               msg="Complex x with imag=0 should match real x")
    # Test 2b: Complex x with imag=0 should match real x
    a_imag = a_imag * 0.
    result_real = discounted_cumsum_1d_ref(a_real, x_real)
    result_complex = discounted_cumsum_1d_complex_ref(a_real, a_imag, x_complex)
    torch.testing.assert_close(result_real, result_complex[:, 0], rtol=1e-5, atol=1e-5)
    
    # Test 3: Complex x with non-zero imag
    torch.manual_seed(2)
    T = 6
    a_real = torch.randn(T, device=device)
    a_imag = torch.randn(T, device=device)
    x_complex = torch.randn(T, 2, device=device)
    
    result = discounted_cumsum_1d_complex_ref(a_real, a_imag, x_complex)
    
    # Verify shape
    assert result.shape == (T, 2), f"Expected shape ({T}, 2), got {result.shape}"
    
    # Verify result is not all zeros
    assert not torch.allclose(result, torch.zeros_like(result)), "Result should not be all zeros"
    
    # Test 4: Manual computation with known complex values
    torch.manual_seed(3)
    T = 3
    a_real = torch.tensor([0.0, 1.0, 0.5], device=device)
    a_imag = torch.tensor([0.0, 1.0, 0.5], device=device)
    x_complex = torch.tensor([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]], device=device)  # shape (3, 2)
    
    result = discounted_cumsum_1d_complex_ref(a_real, a_imag, x_complex)
    
    # Manual computation:
    # x_cum[0] = x[0] = 1.0 + 0.0i
    # x_cum[1] = x_cum[0] * (1.0 + 1.0i) + x[1]
    #          = (1.0 + 0.0i) * (1.0 + 1.0i) + (2.0 + 1.0i)
    #          = (1.0 + 1.0i) + (2.0 + 1.0i) = 3.0 + 2.0i
    # x_cum[2] = x_cum[1] * (0.5 + 0.5i) + x[2]
    #          = (3.0 + 2.0i) * (0.5 + 0.5i) + (3.0 + 2.0i)
    #          = (1.5 - 1.0 + 1.5i + 1.0i) + (3.0 + 2.0i) = (0.5 + 2.5i) + (3.0 + 2.0i) = 3.5 + 4.5i
    
    expected_0 = torch.tensor([1.0, 0.0], device=device)
    expected_1 = torch.tensor([3.0, 2.0], device=device)
    expected_2 = torch.tensor([3.5, 4.5], device=device)
    
    torch.testing.assert_close(result[0], expected_0, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(result[1], expected_1, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(result[2], expected_2, rtol=1e-5, atol=1e-5)
    
    # Test 5: Random values with real x
    torch.manual_seed(4)
    T = 10
    a_real = torch.randn(T, device=device)
    a_imag = torch.randn(T, device=device)
    x = torch.randn(T, device=device)
    
    result = discounted_cumsum_1d_complex_ref(a_real, a_imag, x)
    assert result.shape == (T, 2), f"Expected shape ({T}, 2), got {result.shape}"
    assert not torch.allclose(result, torch.zeros_like(result)), "Result should not be all zeros"
    
    # Test 6: Random values with complex x
    torch.manual_seed(5)
    T = 10
    a_real = torch.randn(T, device=device)
    a_imag = torch.randn(T, device=device)
    x_complex = torch.randn(T, 2, device=device)
    
    result = discounted_cumsum_1d_complex_ref(a_real, a_imag, x_complex)
    assert result.shape == (T, 2), f"Expected shape ({T}, 2), got {result.shape}"
    assert not torch.allclose(result, torch.zeros_like(result)), "Result should not be all zeros"
    
    # Test 7: Edge case - single element
    torch.manual_seed(6)
    T = 1
    a_real = torch.tensor([0.5], device=device)
    a_imag = torch.tensor([0.3], device=device)
    x = torch.tensor([1.0], device=device)
    
    result = discounted_cumsum_1d_complex_ref(a_real, a_imag, x)
    assert result.shape == (1, 2), f"Expected shape (1, 2), got {result.shape}"
    torch.testing.assert_close(result[0, 0], torch.tensor(1.0, device=device), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(result[0, 1], torch.tensor(0.0, device=device), rtol=1e-5, atol=1e-5)
    
    # Test 8: Edge case - single element with complex x
    torch.manual_seed(7)
    T = 1
    a_real = torch.tensor([0.5], device=device)
    a_imag = torch.tensor([0.3], device=device)
    x_complex = torch.tensor([[1.0, 2.0]], device=device)
    
    result = discounted_cumsum_1d_complex_ref(a_real, a_imag, x_complex)
    assert result.shape == (1, 2), f"Expected shape (1, 2), got {result.shape}"
    torch.testing.assert_close(result[0, 0], torch.tensor(1.0, device=device), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(result[0, 1], torch.tensor(2.0, device=device), rtol=1e-5, atol=1e-5)

def test_discounted_cumsum_N_BCHW_complex_gradient():
    """Test complex discount cumsum gradient implementation."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test 1: Reference gradient with known values
    torch.manual_seed(0)
    T = 4
    a_real = torch.tensor([0.0, 1.0, 0.5, 0.8], device=device)
    a_imag = torch.zeros_like(a_real)
    x_real = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    x_imag = torch.zeros_like(x_real)

    X = torch.stack([x_real, x_imag], dim=-1)

    a_real.requires_grad_(True)
    a_imag.requires_grad_(True)
    X.requires_grad_(True)
    y_ref = discounted_cumsum_1d_complex_ref(a_real, a_imag, X)
    dy = torch.ones_like(y_ref); dy[..., 1] = 0.0
    y_ref.backward(dy)

    a_real_clone = a_real.detach().clone()
    a_real_clone.requires_grad_(True)
    x_real_clone = x_real.detach().clone()
    x_real_clone.requires_grad_(True)
    y_real_ref = discounted_cumsum_1d_ref(a_real_clone, x_real_clone)
    y_real_ref.backward(torch.ones_like(y_real_ref))

    torch.testing.assert_close(a_real.grad, a_real_clone.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(a_imag.grad, torch.zeros_like(a_imag.grad), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(X.grad[..., 0], x_real_clone.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(X.grad[..., 1], torch.zeros_like(X.grad[..., 1]), rtol=1e-5, atol=1e-5)
    
    da_r, da_i, dx_r, dx_i = discounted_cumsum_1d_complex_gradient(
        a_real, a_imag, x_real, x_imag, y_ref[..., 0], y_ref[..., 1], dy[..., 0], dy[..., 1]
    )
    torch.testing.assert_close(da_r, a_real_clone.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(da_i, torch.zeros_like(da_i), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(dx_r, x_real_clone.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(dx_i, torch.zeros_like(dx_i), rtol=1e-5, atol=1e-5)

    dy = torch.ones_like(y_ref)
    a_real = a_real.detach().clone()
    a_real.requires_grad_(True)
    a_imag = torch.zeros_like(a_real)
    a_imag.requires_grad_(True)
    x_real = x_real.detach().clone()
    # x_real.requires_grad_(True)
    x_imag = torch.zeros_like(x_real)
    # x_imag.requires_grad_(True)
    X = torch.stack([x_real, x_imag], dim=-1)
    X.requires_grad_(True)
    y_ref = discounted_cumsum_1d_complex_ref(a_real, a_imag, X)
    dy = torch.ones_like(y_ref)
    y_ref.backward(dy)

    da_r, da_i, dx_r, dx_i = discounted_cumsum_1d_complex_gradient(
        a_real, a_imag, x_real, x_imag, y_ref[..., 0], y_ref[..., 1], dy[..., 0], dy[..., 1]
    )
    torch.testing.assert_close(a_real.grad, da_r, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(a_imag.grad, da_i, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(X.grad[..., 0], dx_r, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(X.grad[..., 1], dx_i, rtol=1e-5, atol=1e-5)

    T = 4
    a_real = torch.randn(T, device=device)
    a_imag = torch.randn(T, device=device)
    a_real.requires_grad_(True)
    a_imag.requires_grad_(True)
    x_real = torch.randn(T, device=device)
    x_imag = torch.randn(T, device=device)
    X = torch.stack([x_real, x_imag], dim=-1)
    X.requires_grad_(True)
    y_ref = discounted_cumsum_1d_complex_ref(a_real, a_imag, X)
    dy = torch.ones_like(y_ref) #; dy[..., 1] = 0.0
    y_ref.backward(dy)
    da_r, da_i, dx_r, dx_i = discounted_cumsum_1d_complex_gradient(
        a_real, a_imag, x_real, x_imag, y_ref[..., 0], y_ref[..., 1], dy[..., 0], dy[..., 1]
    )
    torch.testing.assert_close(X.grad[..., 0], dx_r, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(X.grad[..., 1], dx_i, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(a_imag.grad, da_i, rtol=1e-5, atol=1e-5)

    A = torch.stack([a_real, a_imag], dim=-1).view(1, 1, T, 1, 2)
    X = torch.stack([x_real, x_imag], dim=-1).view(1, 1, T, 1, 2)
    Y = discounted_cumsum_N_BCHW_complex_triton(A, X, autotune=False)
    A_clone = A.detach().clone()
    A_clone.requires_grad_(True)
    X_clone = X.detach().clone()
    X_clone.requires_grad_(True)

    dY = dy.view(1, 1, T, 1, 2)

    Y_ref = discounted_cumsum_N_BCHW_complex_ref(A_clone, X_clone)
    Y_ref.backward(dY)
    dA, dX = discounted_cumsum_N_BCHW_complex_gradient_triton(A, Y, dY, autotune=False)
    torch.testing.assert_close(A_clone.grad, dA, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(X_clone.grad, dX, rtol=1e-5, atol=1e-5)

    # Test 2: Reference gradient vs Triton gradient
    torch.manual_seed(1)
    bs = 1
    C = 2
    H = 5
    W = 4
    A = torch.randn(bs, C, H, W, 2, device=device)
    X = torch.randn(bs, C, H, W, device=device)
    Y = discounted_cumsum_N_BCHW_complex_triton(A, X, autotune=False)
    dY = torch.randn(bs, C, H, W, 2, device=device)
    
    A_clone = A.clone()
    X_clone = X.clone()
    A_clone.requires_grad_(True)
    X_clone.requires_grad_(True)
    Y_ref = discounted_cumsum_N_BCHW_complex_ref(A_clone, X_clone)
    torch.testing.assert_close(Y_ref, Y, rtol=1e-4, atol=1e-4)

    Y_ref.backward(dY)

    # Reference gradient
    dA_ref, dX_ref = discounted_cumsum_N_BCHW_complex_gradient(A, X, Y, dY)
    
    # Triton gradient
    dA_triton, dX_triton = discounted_cumsum_N_BCHW_complex_gradient_triton(A, Y, dY, autotune=False)
    
    # Compare results
    torch.testing.assert_close(dA_ref, A_clone.grad, rtol=1e-4, atol=1e-4,
                               msg="Reference gradient dA should match A.grad")
    torch.testing.assert_close(dX_ref, X_clone.grad, rtol=1e-4, atol=1e-4,
                               msg="Reference gradient dX should match X.grad")
    torch.testing.assert_close(dA_ref, dA_triton, rtol=1e-4, atol=1e-4,
                               msg="Triton gradient dA should match reference")
    torch.testing.assert_close(dX_ref, dX_triton[..., 0], rtol=1e-4, atol=1e-4,
                               msg="Triton gradient dX should match reference (real part)")
    
    # Test 3: Complex X input
    torch.manual_seed(2)
    X_complex = torch.randn(bs, C, H, W, 2, device=device)
    A = torch.randn(bs, C, H, W, 2, device=device)
    Y = discounted_cumsum_N_BCHW_complex_triton(A, X_complex, autotune=False)
    dY = torch.randn(bs, C, H, W, 2, device=device)
    
    dA_ref, dX_ref = discounted_cumsum_N_BCHW_complex_gradient(A, X_complex, Y, dY)
    dA_triton, dX_triton = discounted_cumsum_N_BCHW_complex_gradient_triton(A, Y, dY, autotune=False)
    
    torch.testing.assert_close(dA_ref, dA_triton, rtol=1e-4, atol=1e-4,
                               msg="Triton gradient dA should match reference (complex X)")
    torch.testing.assert_close(dX_ref, dX_triton, rtol=1e-4, atol=1e-4,
                               msg="Triton gradient dX should match reference (complex X)")
    
    # Test 4: Autograd integration with real X
    torch.manual_seed(3)
    bs = 1
    C = 2
    H = 5
    W = 4
    A = torch.randn(bs, C, H, W, 2, device=device, requires_grad=True)
    X = torch.randn(bs, C, H, W, device=device, requires_grad=True)
    
    Y = DiscountedCumsumN_BCHW_Complex.apply(A, X, False)
    dY = torch.randn_like(Y)
    Y.backward(dY)
    A_clone = A.detach().clone()
    A_clone.requires_grad_(True)
    X_clone = X.detach().clone()
    X_clone.requires_grad_(True)
    Y_ref = discounted_cumsum_N_BCHW_complex_ref(A_clone, X_clone)
    Y_ref.backward(dY)
    torch.testing.assert_close(A_clone.grad, A.grad, rtol=1e-4, atol=1e-4,
                               msg="Autograd A.grad should match reference")
    torch.testing.assert_close(X_clone.grad, X.grad, rtol=1e-4, atol=1e-4,
                               msg="Autograd X.grad should match reference")
   
    # Test 5: Autograd integration with complex X
    torch.manual_seed(4)
    bs = 1
    C = 2
    H = 5
    W = 4
    A = torch.randn(bs, C, H, W, 2, device=device, requires_grad=True)
    X_complex = torch.randn(bs, C, H, W, 2, device=device, requires_grad=True)
    Y = DiscountedCumsumN_BCHW_Complex.apply(A, X_complex, False)
    dY = torch.randn_like(Y)
    Y.backward(dY)
    A_clone = A.detach().clone()
    A_clone.requires_grad_(True)
    X_complex_clone = X_complex.detach().clone()
    X_complex_clone.requires_grad_(True)
    Y_ref = discounted_cumsum_N_BCHW_complex_ref(A_clone, X_complex_clone)
    Y_ref.backward(dY)
    torch.testing.assert_close(A_clone.grad, A.grad, rtol=1e-4, atol=1e-4,
                               msg="Autograd A.grad should match reference")
    torch.testing.assert_close(X_complex_clone.grad, X_complex.grad, rtol=1e-4, atol=1e-4,
                               msg="Autograd X_complex.grad should match reference")

def test_larger():
    bs, C, H, W = 128, 8, 28, 28
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    A = torch.randn(bs, C, H, W, 2, device=device, requires_grad=True)
    X = torch.randn(bs, C, H, W, 2, device=device, requires_grad=True)
    Y = DiscountedCumsumN_BCHW_Complex.apply(A, X, False)
    print("[test_larger] Y.shape:", Y.shape)
    #dY = torch.randn_like(Y)
    #Y.backward(dY)
    #A_clone = A.detach().clone()
    #A_clone.requires_grad_(True)
    #X_complex_clone = X.detach().clone()

if __name__ == "__main__":
    #test_discounted_cumsum_gradient()
    #test_discounted_cumsum_N_BCHW_autograd()
    #test_discounted_cumsum_gradient_triton()
    #test_discounted_cumsum_N_BCHW_triton()
    #test_discounted_cumsum_N_BCHW_triton()
    #test_discounted_cumsum_N_BCHW_triton_backprop()
    #test_discounted_cumsum_N_BCHW_complex_triton()
    # test_discounted_cumsum_N_BCHW_complex_triton_with_complex_X()
    # test_discounted_cumsum_1d_complex_ref()
    # test_discounted_cumsum_N_BCHW_complex_gradient()
    test_larger()
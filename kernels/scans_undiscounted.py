import os
TRITON_INTERPRET = 1
TRITON_INTERPRET = 0
os.environ["TRITON_INTERPRET"] = str(TRITON_INTERPRET)
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"  # Enable Triton autotuning output
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
def sum(a, b):
    return a + b

@triton.jit
def cumsum_BT_triton_kernel_sanity(X, X_cum,
                            size_0,
                            size_1,
                            stride_0: tl.constexpr,
                            stride_1: tl.constexpr,
                            BLOCK_SIZE_0: tl.constexpr,
                            BLOCK_SIZE_1: tl.constexpr,):
    pid_0 = tl.program_id(axis=0)
    range_1 = tl.arange(0, BLOCK_SIZE_1)
    offsets_0 = pid_0 * BLOCK_SIZE_0 * stride_0
    offsets_1 = range_1 * stride_1
    mask_0 = offsets_0 < size_0 * stride_0
    mask_1 = offsets_1 < size_1 * stride_1
    mask = mask_0 * mask_1

    idx_X = X + offsets_0 + offsets_1

    x = tl.load(idx_X, mask=mask)           
    x_cum = tl.associative_scan(x, axis=0, combine_fn=sum, reverse=False) # Note the axis=0 !
    tl.store(X_cum + offsets_0 + offsets_1, x_cum, mask=mask)

def test_cumsum_BT_triton_sanity():
    X = torch.randint(-10,10, (2, 4), device='cuda').float()
    X_cum = torch.zeros_like(X)
    grid = lambda meta: (triton.cdiv(X.shape[0], meta['BLOCK_SIZE_0']),)
    cumsum_BT_triton_kernel_sanity[grid](X, X_cum,
                                size_0=X.shape[0],
                                size_1=X.shape[1],
                                stride_0=X.stride(0),
                                stride_1=X.stride(1),
                                BLOCK_SIZE_0=1,
                                BLOCK_SIZE_1=X.shape[1],
                                )
    torch.testing.assert_close(X_cum, X.cumsum(dim=1), rtol=1e-5, atol=1e-5)

@triton.jit
def cumsum_BT_triton_kernel(X, X_cum,
                            size_0,
                            size_1,
                            stride_0: tl.constexpr,
                            stride_1: tl.constexpr,
                            BLOCK_SIZE_0: tl.constexpr,
                            BLOCK_SIZE_1: tl.constexpr,
                            AXIS: tl.constexpr):
    pid_0 = tl.program_id(axis=0)
    range_0 = tl.arange(0, BLOCK_SIZE_0)
    range_1 = tl.arange(0, BLOCK_SIZE_1)
    offsets_0 = (pid_0 * BLOCK_SIZE_0 + range_0[:, None]) * stride_0
    offsets_1 = range_1[None, :] * stride_1
    mask_0 = offsets_0 < size_0 * stride_0
    mask_1 = offsets_1 < size_1 * stride_1
    mask = mask_0 * mask_1

    idx_X = X + offsets_0 + offsets_1

    x = tl.load(idx_X, mask=mask)           
    x_cum = tl.associative_scan(x, axis=AXIS, combine_fn=sum, reverse=False)
    tl.store(X_cum + offsets_0 + offsets_1, x_cum, mask=mask)

def test_cumsum_BT_triton():
    X = torch.randint(-10,10, (3, 4), device='cuda').float()
    X_cum = torch.zeros_like(X)
    grid = lambda meta: (triton.cdiv(X.shape[0], meta['BLOCK_SIZE_0']),)
    cumsum_BT_triton_kernel[grid](X, X_cum,
                                size_0=X.shape[0],
                                size_1=X.shape[1],
                                stride_0=X.stride(0),
                                stride_1=X.stride(1),
                                BLOCK_SIZE_0=2,
                                BLOCK_SIZE_1=X.shape[1],
                                AXIS=1)
    torch.testing.assert_close(X_cum, X.cumsum(dim=1), rtol=1e-5, atol=1e-5)

@triton.jit
def cumsum_BTD_triton_kernel(X, X_cum,
                            size_0,
                            size_1,
                            size_2,
                            stride_0: tl.constexpr,
                            stride_1: tl.constexpr,
                            stride_2: tl.constexpr,
                            BLOCK_SIZE_0: tl.constexpr,
                            BLOCK_SIZE_1: tl.constexpr,
                            BLOCK_SIZE_2: tl.constexpr,
                            AXIS: tl.constexpr):
    pid_0 = tl.program_id(axis=0)
    range_0 = tl.arange(0, BLOCK_SIZE_0)
    range_1 = tl.arange(0, BLOCK_SIZE_1)
    range_2 = tl.arange(0, BLOCK_SIZE_2)
    offsets_0 = (pid_0 * BLOCK_SIZE_0 + range_0[:, None, None]) * stride_0
    offsets_1 = range_1[None, :, None] * stride_1
    offsets_2 = range_2[None, None, :] * stride_2
    mask_0 = offsets_0 < size_0 * stride_0
    mask_1 = offsets_1 < size_1 * stride_1
    mask_2 = offsets_2 < size_2 * stride_2
    mask = mask_0 * mask_1 * mask_2

    idx_X = X + offsets_0 + offsets_1 + offsets_2

    x = tl.load(idx_X, mask=mask)           
    x_cum = tl.associative_scan(x, axis=AXIS, combine_fn=sum, reverse=False)
    tl.store(X_cum + offsets_0 + offsets_1 + offsets_2, x_cum, mask=mask)

def test_cumsum_BTD_triton():
    X = torch.randint(-10,10, (3, 4, 2), device='cuda').float()
    X_cum = torch.zeros_like(X)
    grid = lambda meta: (triton.cdiv(X.shape[0], meta['BLOCK_SIZE_0']),)
    cumsum_BTD_triton_kernel[grid](X, X_cum,
                                size_0=X.shape[0],
                                size_1=X.shape[1],
                                size_2=X.shape[2],
                                stride_0=X.stride(0),
                                stride_1=X.stride(1),
                                stride_2=X.stride(2),
                                BLOCK_SIZE_0=2,
                                BLOCK_SIZE_1=X.shape[1],
                                BLOCK_SIZE_2=X.shape[2],
                                AXIS=1)
    torch.testing.assert_close(X_cum, X.cumsum(dim=1), rtol=1e-5, atol=1e-5)


#@triton.autotune(configs=[
#    triton.Config({'BLOCK_SIZE_0': 32}, num_warps=1),
#    triton.Config({'BLOCK_SIZE_0': 32}, num_warps=2),
#    triton.Config({'BLOCK_SIZE_0': 32}, num_warps=4),
#    triton.Config({'BLOCK_SIZE_0': 32}, num_warps=8),
#    triton.Config({'BLOCK_SIZE_0': 64}, num_warps=1),
#    triton.Config({'BLOCK_SIZE_0': 64}, num_warps=2),
#    triton.Config({'BLOCK_SIZE_0': 64}, num_warps=4),
#    triton.Config({'BLOCK_SIZE_0': 64}, num_warps=8),
#    triton.Config({'BLOCK_SIZE_0': 128}, num_warps=1),
#    triton.Config({'BLOCK_SIZE_0': 128}, num_warps=2),
#    triton.Config({'BLOCK_SIZE_0': 128}, num_warps=4),
#    triton.Config({'BLOCK_SIZE_0': 128}, num_warps=8),
#    triton.Config({'BLOCK_SIZE_0': 256}, num_warps=1),
#    triton.Config({'BLOCK_SIZE_0': 256}, num_warps=2),
#    triton.Config({'BLOCK_SIZE_0': 256}, num_warps=4),
#    triton.Config({'BLOCK_SIZE_0': 256}, num_warps=8),
#    triton.Config({'BLOCK_SIZE_0': 512}, num_warps=1),
#    triton.Config({'BLOCK_SIZE_0': 512}, num_warps=2),
#    triton.Config({'BLOCK_SIZE_0': 512}, num_warps=4),
#    triton.Config({'BLOCK_SIZE_0': 512}, num_warps=8),
#    ],
#    key=['size_0','size_1','size_2','size_3'])
@triton.jit
def cumsum_N_BCHW_triton_kernel(X, X_cum,
                            size_0,
                            size_1,
                            size_2,
                            size_3,
                            stride_0: tl.constexpr,
                            stride_1: tl.constexpr,
                            stride_2: tl.constexpr,
                            stride_3: tl.constexpr,
                            BLOCK_SIZE_0: tl.constexpr,
                            BLOCK_SIZE_1: tl.constexpr,
                            BLOCK_SIZE_2: tl.constexpr,
                            BLOCK_SIZE_3: tl.constexpr):
    pid_0 = tl.program_id(axis=0)
    range_0 = tl.arange(0, BLOCK_SIZE_0)
    range_1 = tl.arange(0, BLOCK_SIZE_1)
    range_2 = tl.arange(0, BLOCK_SIZE_2)
    range_3 = tl.arange(0, BLOCK_SIZE_3)
    offsets_0 = (pid_0 * BLOCK_SIZE_0 + range_0[:, None, None, None]) * stride_0
    offsets_1 = range_1[None, :, None, None] * stride_1
    offsets_2 = range_2[None, None, :, None] * stride_2
    offsets_3 = range_3[None, None, None, :] * stride_3
    mask_0 = offsets_0 < size_0 * stride_0
    mask_1 = offsets_1 < size_1 * stride_1
    mask_2 = offsets_2 < size_2 * stride_2
    mask_3 = offsets_3 < size_3 * stride_3
    mask = mask_0 * mask_1 * mask_2 * mask_3
    idx_X = X + offsets_0 + offsets_1 + offsets_2 + offsets_3
    x = tl.load(idx_X, mask=mask)
    x_cum = tl.associative_scan(x, axis=2, combine_fn=sum, reverse=False)
    tl.store(X_cum + offsets_0 + offsets_1 + offsets_2 + offsets_3, x_cum, mask=mask) 

def test_cumsum_N_BCHW_triton():
    X = torch.randint(-10,10, (128, 16, 32, 32), device='cuda').float()
    X_cum = torch.zeros_like(X)
    grid = lambda meta: (triton.cdiv(X.shape[0], meta['BLOCK_SIZE_0']),)
    cumsum_N_BCHW_triton_kernel[grid](X, X_cum,
                                size_0=X.shape[0],
                                size_1=X.shape[1],
                                size_2=X.shape[2],
                                size_3=X.shape[3],
                                stride_0=X.stride(0),
                                stride_1=X.stride(1),
                                stride_2=X.stride(2),
                                stride_3=X.stride(3),
                                BLOCK_SIZE_0=32,
                                BLOCK_SIZE_1=X.shape[1],
                                BLOCK_SIZE_2=X.shape[2],
                                BLOCK_SIZE_3=X.shape[3],)
    torch.testing.assert_close(X_cum, X.cumsum(dim=2), rtol=1e-5, atol=1e-5)

@triton.jit
def cummax_with_index_triton_op(i_left, v_left, i_right, v_right):
    # decide which side wins
    take_left = v_left >= v_right  # leftmost-wins tie-breaking (associative)

    v_out = tl.where(take_left, v_left, v_right)
    i_out = tl.where(take_left, i_left, i_right)

    return i_out, v_out

@triton.jit
def cummax_with_index_1d_triton_kernel(
                            x_ptr,
                            indices_out_ptr,
                            y_out_ptr,

                            size_0: tl.constexpr,
                            BLOCK_SIZE_0: tl.constexpr):
    range_0 = tl.arange(0, BLOCK_SIZE_0)
    x  = tl.load( x_ptr  + range_0, mask=range_0 < size_0, other=-torch.inf)
    idxs = tl.arange(0, BLOCK_SIZE_0)
    indices, y = tl.associative_scan( (idxs,x), axis=0, combine_fn=cummax_with_index_triton_op)
    tl.store( indices_out_ptr + range_0, indices, mask=(range_0 < size_0))
    tl.store( y_out_ptr + range_0, y, mask=(range_0 < size_0))

def cummax_with_index_1d_triton(x):
    indices_out = torch.zeros_like(x)
    y_out = torch.zeros_like(x)
    grid = lambda meta: (triton.cdiv(x.shape[0], meta['BLOCK_SIZE_0']),)
    cummax_with_index_1d_triton_kernel[grid](x, indices_out, y_out,
                                size_0=x.shape[0],
                                BLOCK_SIZE_0=32,)
    return indices_out, y_out

def cummax_with_index_1d_gradient_ref(indices, dy):
    dx = torch.zeros_like(dy)
    #for ui in torch.unique(indices):
    #    dx[ui.int()] = dy[indices == ui].sum()
    #return dx
    dx.index_add_(0, indices.int(), dy)   # or scatter_add_, same idea
    return dx

def cummax_with_index_1d_ref(x):
    indices_out = torch.zeros_like(x)
    y_out = torch.zeros_like(x)
    y_out[0] = x[0]
    indices_out[0] = 0
    for i in range(1,x.shape[0]):
        if x[i] > y_out[i-1]:
            indices_out[i] = i
            y_out[i] = x[i]
        else:
            indices_out[i] = indices_out[i-1]
            y_out[i] = y_out[i-1]
    return indices_out, y_out

def test_cummax_with_index_1d_triton():
    torch.manual_seed(0)
    x = torch.randn(10, device='cuda')
    x.requires_grad_(True)
    # print('x=\n', x)
    indices_out, y_out = cummax_with_index_1d_triton(x)
    # print('indices_out=\n', indices_out)
    # print('y_out=\n', y_out)
    indices_out_ref, y_out_ref = cummax_with_index_1d_ref(x)
    # print('indices_out_ref=\n', indices_out_ref)
    # print('y_out_ref=\n', y_out_ref)
    torch.testing.assert_close(indices_out, indices_out_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(y_out, y_out_ref, rtol=1e-5, atol=1e-5)

    dy = torch.randn(10, device='cuda')
    # print('dy=\n', dy)
    torch.autograd.backward(y_out_ref, dy)
    # print('x.grad=\n', x.grad)
    dx_ref = cummax_with_index_1d_gradient_ref(indices_out_ref, dy)
    # print('dx_ref=\n', dx_ref)
    torch.testing.assert_close(x.grad, dx_ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_cummax_with_index_1d_triton()
    #test_cumsum_BT_triton()
    #test_cumsum_BT_triton_sanity()
    #test_cumsum_BTD_triton()
    #test_cumsum_N_BCHW_triton()
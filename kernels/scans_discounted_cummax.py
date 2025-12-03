import os
TRITON_INTERPRET = 1
TRITON_INTERPRET = 0
os.environ["TRITON_INTERPRET"] = str(TRITON_INTERPRET)
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"  # Enable Triton autotuning output
import triton
import triton.language as tl
import torch
try:
    from .scans_discounted_cumsum import discounted_cumsum_N_BCHW_triton, get_block_sizes_for_scan
except ImportError:
    from scans_discounted_cumsum import discounted_cumsum_N_BCHW_triton, get_block_sizes_for_scan
TRITON_INTERPRET : tl.constexpr = tl.constexpr(TRITON_INTERPRET)
NOT_TRITON_INTERPRET : tl.constexpr = tl.constexpr(1 - TRITON_INTERPRET)
# import profile_utils

@triton.jit
def discounted_cummax_triton_op(a1, x1, a2, x2):
    return a1 + a2, tl.maximum(x1 + a2, x2)

@triton.autotune(configs=[
    triton.Config({'BLOCK_SIZE_0': x, 'BLOCK_SIZE_1': x, 'BLOCK_SIZE_3': x}, num_warps=y)
    for x in [2, 4, 8, 16, 32]
    for y in [1, 2, 4, 8]

    #triton.Config({'BLOCK_SIZE_0': 2, 'BLOCK_SIZE_1': 2, 'BLOCK_SIZE_3': 2}, num_warps=2),
    #triton.Config({'BLOCK_SIZE_0': 2, 'BLOCK_SIZE_1': 2, 'BLOCK_SIZE_3': 2}, num_warps=4),
    #triton.Config({'BLOCK_SIZE_0': 2, 'BLOCK_SIZE_1': 2, 'BLOCK_SIZE_3': 2}, num_warps=8),

    #triton.Config({'BLOCK_SIZE_0': 32, 'BLOCK_SIZE_1': 32, 'BLOCK_SIZE_3': 32}, num_warps=2),
    #triton.Config({'BLOCK_SIZE_0': 32, 'BLOCK_SIZE_1': 32, 'BLOCK_SIZE_3': 32}, num_warps=4),
    #triton.Config({'BLOCK_SIZE_0': 32, 'BLOCK_SIZE_1': 32, 'BLOCK_SIZE_3': 32}, num_warps=8),

    #triton.Config({'BLOCK_SIZE_0': 64, 'BLOCK_SIZE_1': 64, 'BLOCK_SIZE_3': 64}, num_warps=2),
    #triton.Config({'BLOCK_SIZE_0': 64, 'BLOCK_SIZE_1': 64, 'BLOCK_SIZE_3': 64}, num_warps=4),
    #triton.Config({'BLOCK_SIZE_0': 64, 'BLOCK_SIZE_1': 64, 'BLOCK_SIZE_3': 64}, num_warps=8),

    #triton.Config({'BLOCK_SIZE_0': 128, 'BLOCK_SIZE_1': 128, 'BLOCK_SIZE_3': 128}, num_warps=2),
    #triton.Config({'BLOCK_SIZE_0': 128, 'BLOCK_SIZE_1': 128, 'BLOCK_SIZE_3': 128}, num_warps=4),
    #triton.Config({'BLOCK_SIZE_0': 128, 'BLOCK_SIZE_1': 128, 'BLOCK_SIZE_3': 128}, num_warps=8),

    #triton.Config({'BLOCK_SIZE_0': 256, 'BLOCK_SIZE_1': 256, 'BLOCK_SIZE_3': 256}, num_warps=2),
    #triton.Config({'BLOCK_SIZE_0': 256, 'BLOCK_SIZE_1': 256, 'BLOCK_SIZE_3': 256}, num_warps=4),
    #triton.Config({'BLOCK_SIZE_0': 256, 'BLOCK_SIZE_1': 256, 'BLOCK_SIZE_3': 256}, num_warps=8),

    #triton.Config({'BLOCK_SIZE_0': 512, 'BLOCK_SIZE_1': 512, 'BLOCK_SIZE_3': 512}, num_warps=2),
    #triton.Config({'BLOCK_SIZE_0': 512, 'BLOCK_SIZE_1': 512, 'BLOCK_SIZE_3': 512}, num_warps=4),
    #triton.Config({'BLOCK_SIZE_0': 512, 'BLOCK_SIZE_1': 512, 'BLOCK_SIZE_3': 512}, num_warps=8),
    ],
    key=['size_0','size_1','size_2','size_3'])
@triton.jit
def discounted_cummax_N_BCHW_triton_kernel(
                            A,
                            X, X_cum,
                            size_0,
                            size_1,
                            size_2,
                            size_3,
                            A_stride_0: tl.constexpr,
                            A_stride_1: tl.constexpr,
                            A_stride_2: tl.constexpr,
                            A_stride_3: tl.constexpr,
                            X_stride_0: tl.constexpr,
                            X_stride_1: tl.constexpr,
                            X_stride_2: tl.constexpr,
                            X_stride_3: tl.constexpr,
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

    X_offsets_0 = (start_0 * BLOCK_SIZE_0 + range_0[:, None, None, None]) * X_stride_0
    X_offsets_1 = (start_1 * BLOCK_SIZE_1 + range_1[None, :, None, None]) * X_stride_1
    X_offsets_2 = range_2[None, None, :, None] * X_stride_2
    X_offsets_3 = (start_3 * BLOCK_SIZE_3 + range_3[None, None, None, :]) * X_stride_3
    X_mask_0 = X_offsets_0 < (size_0 * X_stride_0)
    X_mask_1 = X_offsets_1 < (size_1 * X_stride_1)
    X_mask_2 = X_offsets_2 < (size_2 * X_stride_2)
    X_mask_3 = X_offsets_3 < (size_3 * X_stride_3)
    X_mask = X_mask_0 * X_mask_1 * X_mask_2 * X_mask_3
    idx_X = X + X_offsets_0 + X_offsets_1 + X_offsets_2 + X_offsets_3

    A_offsets_0 = (start_0 * BLOCK_SIZE_0 + range_0[:, None, None, None])
    A_offsets_1 = (start_1 * BLOCK_SIZE_1 + range_1[None, :, None, None])
    A_offsets_2 = range_2[None, None, :, None]
    A_offsets_3 = (start_3 * BLOCK_SIZE_3 + range_3[None, None, None, :])
    #A_mask_0 = A_offsets_0 < size_0
    #A_mask_1 = A_offsets_1 < size_1
    #A_mask_2 = A_offsets_2 < size_2
    #A_mask_3 = A_offsets_3 < size_3
    #A_mask = A_mask_0 * A_mask_1 * A_mask_2 * A_mask_3
    A_mask = (A_offsets_0 < size_0) & (A_offsets_1 < size_1) & (A_offsets_2 < size_2) & (A_offsets_3 < size_3)
    #print('A_mask_0=\n', A_mask_0, A_mask_0.shape) if TRITON_INTERPRET else None
    #print('A_mask_1=\n', A_mask_1, A_mask_1.shape) if TRITON_INTERPRET else None
    #print('A_mask_2=\n', A_mask_2, A_mask_2.shape) if TRITON_INTERPRET else None
    #print('A_mask_3=\n', A_mask_3, A_mask_3.shape) if TRITON_INTERPRET else None
    #print('A_mask=\n', A_mask, A_mask.shape) if TRITON_INTERPRET else None
    idx_A = A + A_offsets_0 * A_stride_0 + A_offsets_1 * A_stride_1 + A_offsets_2 * A_stride_2 + A_offsets_3 * A_stride_3
    # tl.device_print('A_offsets_0=\n', A_offsets_0)
    print('idx_A=\n', idx_A, idx_A.shape) if TRITON_INTERPRET else None

    x = tl.load(idx_X, mask=X_mask) #, other=-torch.inf)
    a = tl.load(idx_A, mask=A_mask) #, other=-torch.inf)
    a_cum, x_cum = tl.associative_scan((a,x), axis=2, combine_fn=discounted_cummax_triton_op, reverse=False)
    tl.store(X_cum + X_offsets_0 + X_offsets_1 + X_offsets_2 + X_offsets_3, x_cum, mask=X_mask) 

def discounted_cummax_N_BCHW_triton(A, X, autotune=False):
    X_cum = torch.zeros_like(X)
    grid = lambda meta: (triton.cdiv(X.shape[0], meta['BLOCK_SIZE_0']),
                         triton.cdiv(X.shape[1], meta['BLOCK_SIZE_1']),
                         triton.cdiv(X.shape[3], meta['BLOCK_SIZE_3']))
    if autotune:
        discounted_cummax_N_BCHW_triton_kernel[grid](A, X, X_cum,
                                    size_0=X.shape[0],
                                    size_1=X.shape[1],
                                    size_2=X.shape[2],
                                    size_3=X.shape[3],
                                    A_stride_0=A.stride(0),
                                    A_stride_1=A.stride(1),
                                    A_stride_2=A.stride(2),
                                    A_stride_3=A.stride(3),
                                    X_stride_0=X.stride(0),
                                    X_stride_1=X.stride(1),
                                    X_stride_2=X.stride(2),
                                    X_stride_3=X.stride(3),
                                    # BLOCK_SIZE_0=32,
                                    # BLOCK_SIZE_1=2,
                                    BLOCK_SIZE_2=triton.next_power_of_2(X.shape[2]),
                                    # BLOCK_SIZE_3=X.shape[3],
                                    )
    else:
        BLOCK_SIZE_0, BLOCK_SIZE_1, BLOCK_SIZE_2, BLOCK_SIZE_3 = get_block_sizes_for_scan(X.shape)
        discounted_cummax_N_BCHW_triton_kernel.fn[grid](A, X, X_cum,
                                    size_0=X.shape[0],
                                    size_1=X.shape[1],
                                    size_2=X.shape[2],
                                    size_3=X.shape[3],
                                    A_stride_0=A.stride(0),
                                    A_stride_1=A.stride(1),
                                    A_stride_2=A.stride(2),
                                    A_stride_3=A.stride(3),
                                    X_stride_0=X.stride(0),
                                    X_stride_1=X.stride(1),
                                    X_stride_2=X.stride(2),
                                    X_stride_3=X.stride(3),
                                    BLOCK_SIZE_0=BLOCK_SIZE_0,
                                    BLOCK_SIZE_1=BLOCK_SIZE_1,
                                    BLOCK_SIZE_2=triton.next_power_of_2(X.shape[2]), # This dim needs to be handled in one block.
                                    BLOCK_SIZE_3=BLOCK_SIZE_3,)
    return X_cum

def discounted_cummax_1d_ref(a, x):
    x_cum = torch.zeros_like(x)
    x_cum[0] = x[0]
    for i in range(1, x.shape[0]):
        x_cum[i] = torch.maximum(x_cum[i-1] + a[i], x[i])
    return x_cum

def discounted_cummax_with_index_1d_ref(a, x):
    x_cum = torch.zeros_like(x)
    indices = torch.zeros_like(x)
    indices[0] = 0
    x_cum[0] = x[0]
    for i in range(1, x.shape[0]):
        if x_cum[i-1] + a[i] >= x[i]:
            indices[i] = indices[i-1]
            x_cum[i] = x_cum[i-1] + a[i]
        else:
            indices[i] = i
            x_cum[i] = x[i]
    return indices, x_cum

def discounted_cummax_1d_gradient_ref(indices, dy):
    da = torch.zeros_like(dy)
    dx = torch.zeros_like(dy)
    #for ui in torch.unique(indices):
    #    dx[ui.int()] = dy[indices == ui].sum()
    #return dx
    dx.index_add_(0, indices.int(), dy)   # or scatter_add_, same idea

    ones = torch.zeros_like(dy)
    ones[1:] = (indices[1:] == indices[:-1])

    current_a = 0.
    for t in range(indices.shape[0]-1, 0, -1):
        # tmp = (current_a + dy[t]) * ones[t]
        tmp = current_a * ones[t] + dy[t] * ones[t]
        da[t] = tmp
        current_a = tmp

    return da, dx

def discounted_cummax_N_BCHW_ref(A, X):
    X_cum = torch.zeros_like(X)
    for b in range(X.shape[0]):
        for c in range(X.shape[1]):
            for w in range(X.shape[3]):
                # print('b=', b, ' c=', c, ' w=', w)
                X_cum[b,c,:,w] = discounted_cummax_1d_ref(A[b,c,:,w], X[b,c,:,w])
    return X_cum

def test_discounted_cummax_N_BCHW_triton():
    device = 'cpu' if TRITON_INTERPRET else 'cuda'
    torch.manual_seed(0)
    # for bs in [1]:
    for bs in [1,2,33]:
        # for C in [1]:
        for C in [2,8,65]:
            # for H in [8]:
            for H in [2,4,8,16,32,128,500]:
                # XXX Fails for bs= 1  C= 2  H= 500  W= 503
                for W in [H, H+3]:
                    print('bs=', bs, ' C=', C, ' H=', H, ' W=', W)
                    A = torch.randn(bs, C, 1, 1, device=device)
                    A_expanded = A.expand(bs, C, H, W)
                    A_expanded_clone = A_expanded.clone()
                    A_expanded.requires_grad = True
                    A_expanded_clone.requires_grad = True

                    X = torch.randn(bs, C, H, W, device=device)
                    X_clone = X.clone()
                    X.requires_grad = True
                    X_clone.requires_grad = True

                    X_cum_ref = discounted_cummax_N_BCHW_ref(A_expanded, X)
                    X_cum_triton = DiscountedCummaxN_BCHW.apply(A_expanded_clone, X_clone)
                    torch.testing.assert_close(X_cum_triton, X_cum_ref, rtol=1e-5, atol=1e-5)

                    dY = torch.randn_like(X_cum_triton)
                    loss_r = (X_cum_ref * dY).sum()
                    loss_t = (X_cum_triton * dY).sum()
                    loss_r.backward()
                    loss_t.backward()
                    # torch.autograd.backward(X_cum_ref, dY)
                    # torch.autograd.backward(X_cum_triton, dY)

                    #print('X.grad=\n', X.grad)
                    #print('X_clone.grad=\n', X_clone.grad)
                    #print('A_expanded.grad=\n', A_expanded.grad)
                    #print('A_expanded_clone.grad=\n', A_expanded_clone.grad)
                    torch.testing.assert_close(X.grad, X_clone.grad, rtol=1e-5, atol=1e-5)
                    torch.testing.assert_close(A_expanded.grad, A_expanded_clone.grad, rtol=1e-5, atol=1e-5)


def test_discounted_cummax_N_BCHW_triton_expanded():
    device = 'cpu' if TRITON_INTERPRET else 'cuda'

    torch.manual_seed(0)
    A = torch.randn(1,5,1,1, device=device)
    A_expanded = A.expand(3,5,7,7)
    # print('A_expanded=\n', A_expanded.shape, ' A_expanded.stride=', A_expanded.stride())
    X = torch.randn(3,5,7,7, device=device)
    expected = discounted_cummax_N_BCHW_ref(A_expanded, X)
    X_cum_expanded = discounted_cummax_N_BCHW_triton(A_expanded, X)
    # print('X_cum_expanded=\n', X_cum_expanded, ' X_cum_expanded.shape=', X_cum_expanded.shape, ' X_cum_expanded.stride=', X_cum_expanded.stride(), ' X_cum_expanded.clone().stride=', X_cum_expanded.clone().stride())
    torch.testing.assert_close(X_cum_expanded, expected, rtol=1e-5, atol=1e-5)
    for bs in [1,2,33]:
        for C in [2,8,65]:
            for H in [2,4,8,16,32,128,500]:
                for W in [H, H+3]:
                    print('bs=', bs, ' C=', C, ' H=', H, ' W=', W)
                    A = torch.randn(bs, C, 1, 1, device=device)
                    A_expanded = A.expand(bs, C, H, W)
                    X = torch.randn(bs, C, H, W, device=device)
                    print('calculating expected...')
                    expected = discounted_cummax_N_BCHW_ref(A_expanded, X)
                    print('calculating X_cum_expanded...')
                    X_cum_expanded = discounted_cummax_N_BCHW_triton(A_expanded, X)
                    X_cum_fn = DiscountedCummaxN_BCHW.apply(A_expanded, X)
                    torch.testing.assert_close(X_cum_expanded, expected, rtol=1e-5, atol=1e-5)
                    torch.testing.assert_close(X_cum_fn, expected, rtol=1e-5, atol=1e-5)

@triton.jit
def discounted_cummax_with_index_triton_op(i_left, a_left, x_left, i_right, a_right, x_right):
    # decide which side wins
    take_left = x_left + a_right >= x_right  # leftmost-wins tie-breaking (associative)

    x_out = tl.where(take_left, x_left + a_right, x_right)
    i_out = tl.where(take_left, i_left, i_right)
    a_out = a_left + a_right

    return i_out, a_out, x_out

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
def discounted_cummax_N_BCHW_triton_kernel_with_index(
                            A,
                            X, 
                            Indices_out,
                            X_cum,
                            size_0,
                            size_1,
                            size_2,
                            size_3,
                            A_stride_0: tl.constexpr,
                            A_stride_1: tl.constexpr,
                            A_stride_2: tl.constexpr,
                            A_stride_3: tl.constexpr,
                            X_stride_0: tl.constexpr,
                            X_stride_1: tl.constexpr,
                            X_stride_2: tl.constexpr,
                            X_stride_3: tl.constexpr,
                            Indices_stride_0: tl.constexpr,
                            Indices_stride_1: tl.constexpr,
                            Indices_stride_2: tl.constexpr,
                            Indices_stride_3: tl.constexpr,
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

    X_offsets_0 = (start_0 * BLOCK_SIZE_0 + range_0[:, None, None, None]) * X_stride_0
    X_offsets_1 = (start_1 * BLOCK_SIZE_1 + range_1[None, :, None, None]) * X_stride_1
    X_offsets_2 = range_2[None, None, :, None] * X_stride_2
    X_offsets_3 = (start_3 * BLOCK_SIZE_3 + range_3[None, None, None, :]) * X_stride_3
    X_mask_0 = X_offsets_0 < (size_0 * X_stride_0)
    X_mask_1 = X_offsets_1 < (size_1 * X_stride_1)
    X_mask_2 = X_offsets_2 < (size_2 * X_stride_2)
    X_mask_3 = X_offsets_3 < (size_3 * X_stride_3)
    X_mask = X_mask_0 * X_mask_1 * X_mask_2 * X_mask_3
    idx_X = X + X_offsets_0 + X_offsets_1 + X_offsets_2 + X_offsets_3

    A_offsets_0 = (start_0 * BLOCK_SIZE_0 + range_0[:, None, None, None])
    A_offsets_1 = (start_1 * BLOCK_SIZE_1 + range_1[None, :, None, None])
    A_offsets_2 = range_2[None, None, :, None]
    A_offsets_3 = (start_3 * BLOCK_SIZE_3 + range_3[None, None, None, :])
    A_mask = (A_offsets_0 < size_0) & (A_offsets_1 < size_1) & (A_offsets_2 < size_2) & (A_offsets_3 < size_3)
    idx_A = A + A_offsets_0 * A_stride_0 + A_offsets_1 * A_stride_1 + A_offsets_2 * A_stride_2 + A_offsets_3 * A_stride_3

    Indices_offsets_0 = (start_0 * BLOCK_SIZE_0 + range_0[:, None, None, None]) * Indices_stride_0
    Indices_offsets_1 = (start_1 * BLOCK_SIZE_1 + range_1[None, :, None, None]) * Indices_stride_1
    Indices_offsets_2 = range_2[None, None, :, None] * Indices_stride_2
    Indices_offsets_3 = (start_3 * BLOCK_SIZE_3 + range_3[None, None, None, :]) * Indices_stride_3
    idx_Indices = Indices_out + Indices_offsets_0 + Indices_offsets_1 + Indices_offsets_2 + Indices_offsets_3

    x = tl.load(idx_X, mask=X_mask)
    a = tl.load(idx_A, mask=A_mask)
    # Create indices tensor for the scan dimension (axis 2)
    # Initialize indices to their position along axis 2 (0, 1, 2, ..., BLOCK_SIZE_2-1)
    # The mask will ensure only valid positions are used
    indices_init = range_2[None, None, :, None]
    indices_init = tl.broadcast_to(indices_init, (BLOCK_SIZE_0, BLOCK_SIZE_1, BLOCK_SIZE_2, BLOCK_SIZE_3))
    indices, a_cum, x_cum = tl.associative_scan((indices_init, a, x), axis=2, combine_fn=discounted_cummax_with_index_triton_op, reverse=False)
    tl.store(idx_Indices, indices, mask=X_mask)
    tl.store(X_cum + X_offsets_0 + X_offsets_1 + X_offsets_2 + X_offsets_3, x_cum, mask=X_mask)

def discounted_cummax_N_BCHW_triton_with_index(A, X, autotune=False):
    """
    Compute discounted cumulative max with indices.
    
    Returns:
        indices: Tensor of shape (B, C, H, W) containing the indices
        X_cum: Tensor of shape (B, C, H, W) containing the cumulative max values
    """
    X_cum = torch.zeros_like(X)
    Indices = torch.zeros_like(X, dtype=torch.int64)
    grid = lambda meta: (triton.cdiv(X.shape[0], meta['BLOCK_SIZE_0']),
                         triton.cdiv(X.shape[1], meta['BLOCK_SIZE_1']),
                         triton.cdiv(X.shape[3], meta['BLOCK_SIZE_3']))
    if autotune:
        discounted_cummax_N_BCHW_triton_kernel_with_index[grid](A, X, Indices, X_cum,
                                    size_0=X.shape[0],
                                    size_1=X.shape[1],
                                    size_2=X.shape[2],
                                    size_3=X.shape[3],
                                    A_stride_0=A.stride(0),
                                    A_stride_1=A.stride(1),
                                    A_stride_2=A.stride(2),
                                    A_stride_3=A.stride(3),
                                    X_stride_0=X.stride(0),
                                    X_stride_1=X.stride(1),
                                    X_stride_2=X.stride(2),
                                    X_stride_3=X.stride(3),
                                    Indices_stride_0=Indices.stride(0),
                                    Indices_stride_1=Indices.stride(1),
                                    Indices_stride_2=Indices.stride(2),
                                    Indices_stride_3=Indices.stride(3),
                                    BLOCK_SIZE_2=triton.next_power_of_2(X.shape[2]),
                                    )
    else:
        BLOCK_SIZE_0, BLOCK_SIZE_1, BLOCK_SIZE_2, BLOCK_SIZE_3 = get_block_sizes_for_scan(X.shape)
        discounted_cummax_N_BCHW_triton_kernel_with_index.fn[grid](A, X, Indices, X_cum,
                                    size_0=X.shape[0],
                                    size_1=X.shape[1],
                                    size_2=X.shape[2],
                                    size_3=X.shape[3],
                                    A_stride_0=A.stride(0),
                                    A_stride_1=A.stride(1),
                                    A_stride_2=A.stride(2),
                                    A_stride_3=A.stride(3),
                                    X_stride_0=X.stride(0),
                                    X_stride_1=X.stride(1),
                                    X_stride_2=X.stride(2),
                                    X_stride_3=X.stride(3),
                                    Indices_stride_0=Indices.stride(0),
                                    Indices_stride_1=Indices.stride(1),
                                    Indices_stride_2=Indices.stride(2),
                                    Indices_stride_3=Indices.stride(3),
                                    BLOCK_SIZE_0=BLOCK_SIZE_0,
                                    BLOCK_SIZE_1=BLOCK_SIZE_1,
                                    BLOCK_SIZE_2=triton.next_power_of_2(X.shape[2]),
                                    BLOCK_SIZE_3=BLOCK_SIZE_3,)
    return Indices, X_cum

class DiscountedCummaxN_BCHW(torch.autograd.Function):
    """
    PyTorch autograd Function for discounted_cummax_N_BCHW using Triton kernels.
    """
    @staticmethod
    def forward(ctx, A, X, autotune=False):
        """
        Forward pass for discounted cumulative max.
        
        Args:
            ctx: Context object to save tensors for backward
            A: Discount factors tensor of shape (B, C, H, W)
            X: Input tensor of shape (B, C, H, W)
            autotune: Whether to use autotuning for the forward kernel
        
        Returns:
            X_cum: Output tensor of shape (B, C, H, W)
        """
        # print("[DiscountedCummaxN_BCHW.forward] autotune: ", autotune)
        indices, X_cum = discounted_cummax_N_BCHW_triton_with_index(A, X, autotune=autotune)
        # Save indices for backward pass
        ctx.save_for_backward(indices)
        ctx.autotune = autotune
        # print("[DiscountedCummaxN_BCHW.forward] done")
        return X_cum
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for discounted cumulative max.
        
        Args:
            ctx: Context object with saved tensors
            grad_output: Gradient w.r.t. output tensor of shape (B, C, H, W)
        
        Returns:
            dA: Gradient w.r.t. A tensor of shape (B, C, H, W)
            dX: Gradient w.r.t. X tensor of shape (B, C, H, W)
        """
        # print("[DiscountedCummaxN_BCHW.backward] autotune: ", ctx.autotune)
        indices, = ctx.saved_tensors
        B, C, H, W = grad_output.shape
        
        # Initialize gradient tensors
        dX = torch.zeros_like(grad_output)
        dA = torch.zeros_like(grad_output)
        
        # Compute ones tensor: ones[1:] = (indices[1:] == indices[:-1])
        # This is 1 where the max was carried forward, 0 where a new max was taken
        ones = torch.zeros_like(grad_output, dtype=grad_output.dtype)
        ones[:, :, 1:, :] = (indices[:, :, 1:, :] == indices[:, :, :-1, :]).float()
        
        # For da calculation, we use the recursion: da[t] = (da[t+1] + dy[t]) * ones[t]
        # This is a backward recursion. To use discounted cumsum, we reverse along axis 2 (H dimension)
        # After reversal, it becomes: da_rev[i] = (da_rev[i-1] + dy_rev[i]) * ones_rev[i]
        # Which is: da_rev[i] = da_rev[i-1] * ones_rev[i] + dy_rev[i] * ones_rev[i]
        # This matches discounted cumsum: y[i] = y[i-1] * a[i] + x[i]
        # with a[i] = ones_rev[i] and x[i] = dy_rev[i] * ones_rev[i]
        
        # Reverse along axis 2 (H dimension)
        ones_rev = ones.flip(dims=[2])
        dy_rev = grad_output.flip(dims=[2])
        
        # Apply discounted cumsum: da_rev[i] = da_rev[i-1] * ones_rev[i] + dy_rev[i] * ones_rev[i]
        # The input to discounted cumsum is dy_rev * ones_rev, and discount factors are ones_rev
        da_rev = discounted_cumsum_N_BCHW_triton(ones_rev, dy_rev * ones_rev, autotune=ctx.autotune)
        
        # Reverse back to get da
        dA = da_rev.flip(dims=[2])
        
        # Compute dX using the reference implementation for each 1D slice
        # The gradient flows back to the indices that were selected during forward
        # print("[DiscountedCummaxN_BCHW.backward] doing scatter_add_...")
        dX.scatter_add_(2, indices, grad_output)
        # print("[DiscountedCummaxN_BCHW.backward] done")
        return dA, dX, None # None: Gradient w.r.t. autotune (not needed)

@triton.jit
def discounted_cummax_with_index_1d_triton_kernel(
                            a_ptr,
                            x_ptr,
                            indices_out_ptr,
                            y_out_ptr,

                            size_0: tl.constexpr,
                            BLOCK_SIZE_0: tl.constexpr):
    range_0 = tl.arange(0, BLOCK_SIZE_0)
    x  = tl.load( x_ptr  + range_0, mask=range_0 < size_0, other=-torch.inf)
    a  = tl.load( a_ptr  + range_0, mask=range_0 < size_0, other=0.)
    idxs = tl.arange(0, BLOCK_SIZE_0)
    indices, a_cum, y = tl.associative_scan( (idxs,a,x), axis=0, combine_fn=discounted_cummax_with_index_triton_op)
    tl.store( indices_out_ptr + range_0, indices, mask=(range_0 < size_0))
    # tl.store( a_cum_out_ptr + range_0, a_cum, mask=(range_0 < size_0))
    tl.store( y_out_ptr + range_0, y, mask=(range_0 < size_0))

def test_discounted_cummax_with_index_1d():
    x = torch.tensor([10., 3., 5., 2.], device='cuda')
    a = torch.tensor([-4., -3., -4., -4.], device='cuda')
    indices_ref, y_ref = discounted_cummax_with_index_1d_ref(a, x)
    grid = (1,)
    indices = torch.zeros_like(x)
    y = torch.zeros_like(x)
    discounted_cummax_with_index_1d_triton_kernel[grid](a, x, indices, y,
                                size_0=x.shape[0],
                                BLOCK_SIZE_0=triton.next_power_of_2(x.shape[0]),)
    torch.testing.assert_close(indices, indices_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-5)

    torch.manual_seed(0)
    x = torch.randn(20, device='cuda')
    print('x', x)
    x.requires_grad = True
    dy = torch.randn_like(x)
    a = torch.randn(20, device='cuda')
    print('a', a)
    a.requires_grad = True
    indices_ref, y_ref = discounted_cummax_with_index_1d_ref(a, x)
    # print('indices_ref', indices_ref)
    # print('y_ref', y_ref)
    torch.autograd.backward(y_ref, dy)
    da_ref, dx_ref = discounted_cummax_1d_gradient_ref(indices_ref, dy)
    torch.testing.assert_close(x.grad, dx_ref, rtol=1e-5, atol=1e-5)
    # print('da_ref', da_ref)
    # print('a.grad', a.grad)
    torch.testing.assert_close(a.grad, da_ref, rtol=1e-5, atol=1e-5)
    for T in [50, 75, 100, 125]:
        x = torch.randn(T, device='cuda')
        x.requires_grad = True
        a = torch.randn(T, device='cuda')
        a.requires_grad = True
        indices_ref, y_ref = discounted_cummax_with_index_1d_ref(a, x)
        dy = torch.randn_like(x)
        torch.autograd.backward(y_ref, dy)
        da_ref, dx_ref = discounted_cummax_1d_gradient_ref(indices_ref, dy)
        torch.testing.assert_close(x.grad, dx_ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(a.grad, da_ref, rtol=1e-5, atol=1e-5)

    indices = torch.zeros_like(x)
    y = torch.zeros_like(x)
    discounted_cummax_with_index_1d_triton_kernel[grid](a, x, indices, y,
                                size_0=x.shape[0],
                                BLOCK_SIZE_0=triton.next_power_of_2(x.shape[0]),)
    torch.testing.assert_close(indices, indices_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_discounted_cummax_N_BCHW_triton()
    test_discounted_cummax_N_BCHW_triton_expanded()
    test_discounted_cummax_with_index_1d()
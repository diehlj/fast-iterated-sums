import os
TRITON_INTERPRET = 1
TRITON_INTERPRET = 0
os.environ["TRITON_INTERPRET"] = str(TRITON_INTERPRET)
import triton
import triton.language as tl
import torch
TRITON_INTERPRET : tl.constexpr = TRITON_INTERPRET
NOT_TRITON_INTERPRET : tl.constexpr = 1 - TRITON_INTERPRET
import src.kernels.profile_utils as profile_utils

###
# dimensions:
# - for SigmaNet: (B, C, H, W) with scan over H or W
# - for ellis:    (B, C, T)
#
# *** in this file: (C,T)  ***
###

def linear_recurrence_ref(A, B, axis=1):
    assert axis==1, "Only axis=1 is supported for now"
    A_cum = torch.zeros_like(A)
    B_cum = torch.zeros_like(B)
    A_cum[:,0] = A[:,0]
    B_cum[:,0] = B[:,0]
    for i in range(1, A.shape[1]):
        A_cum[:,i] = A[:,i] * A_cum[:,i-1]
        B_cum[:,i] = A[:,i] * B_cum[:,i-1] + B[:,i]
    return A_cum, B_cum

@triton.jit
def linear_recurrence(a1, b1, a2, b2):
    # jd: seems to be really pointwise
    print(f"a1={a1} {a1.shape}, a2={a2} {a2.shape}") if TRITON_INTERPRET else None
    print(f"b1={b1} {b1.shape}, b2={b2} {b2.shape}") if TRITON_INTERPRET else None
    # tl.device_print("linear_recurrence: a1=", a1) if NOT_TRITON_INTERPRET else None
    # tl.device_print("linear_recurrence: a2=", a2) if NOT_TRITON_INTERPRET else None
    # tl.device_print("linear_recurrence: b1=", b1) if NOT_TRITON_INTERPRET else None
    # tl.device_print("linear_recurrence: b2=", b2) if NOT_TRITON_INTERPRET else None

    return a1 * a2, b1 * a2 + b2

# related:
# https://github.com/triton-lang/triton/issues/2657
# https://github.com/triton-lang/triton/issues/2359
@triton.jit
def linear_recurrence_triton_kernel(A, B, A_cum, B_cum, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, AXIS: tl.constexpr):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    pid_z = tl.program_id(axis=2)

    print(f"linear_recurrence_kernel: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, AXIS={AXIS}") if TRITON_INTERPRET else None
    range_m = tl.arange(0, BLOCK_M)
    print(f"range_m: {range_m}") if TRITON_INTERPRET else None
    range_n = tl.arange(0, BLOCK_N)
    print(f"range_n: {range_n}") if TRITON_INTERPRET else None
    offset = pid_x * BLOCK_N
    print(f"offset: {offset}") if TRITON_INTERPRET else None

    tmp_1 = range_m[:, None] * BLOCK_N
    tmp_2 = range_n[None, :]
    idx_a = A + offset + tmp_1 + tmp_2
    a = tl.load(idx_a)        # (BLOCK_M * pid_x:BLOCK_M * (pid_x+1), BLOCK_M)

    print(f"A:\n{A}, {type(A)}") if TRITON_INTERPRET else None
    print(f"tmp_1=\n{tmp_1}") if TRITON_INTERPRET else None
    print(f"tmp_2=\n{tmp_2}") if TRITON_INTERPRET else None
    print(f"tmp_1+tmp_2=\n{tmp_1 + tmp_2}") if TRITON_INTERPRET else None
    print(f"idx_a:\n{idx_a}") if TRITON_INTERPRET else None

    print(f"a:\n{a}") if TRITON_INTERPRET else None
    idx_b = B + offset + range_m[:, None] * BLOCK_N + range_n[None, :]
    b = tl.load(idx_b)
    print(f"idx_b:\n{idx_b}") if TRITON_INTERPRET else None
    print(f"b:\n{b}") if TRITON_INTERPRET else None
    a_cum, b_cum = tl.associative_scan((a, b), axis=AXIS, combine_fn=linear_recurrence, reverse=False)
    print(f"a_cum: {a_cum}") if TRITON_INTERPRET else None
    print(f"b_cum: {b_cum}") if TRITON_INTERPRET else None
    tl.store(A_cum + offset + range_m[:, None] * BLOCK_N + range_n[None, :], a_cum)
    tl.store(B_cum + offset + range_m[:, None] * BLOCK_N + range_n[None, :], b_cum)

def linear_recurrence_triton(A,B, BLOCK_SIZE_M=1):
    assert A.shape == B.shape # XXX No, later A should not have a batch !!!
    A_cum = torch.zeros_like(A)
    B_cum = torch.zeros_like(A)
    grid = lambda meta: (triton.cdiv(A.shape[0], BLOCK_SIZE_M),)
    # print("grid=", grid(None))
    linear_recurrence_triton_kernel[grid](A, B, A_cum, B_cum, BLOCK_M=BLOCK_SIZE_M, BLOCK_N=A.shape[1], AXIS=1)
    return A_cum, B_cum

def test_linear_recurrence_triton():
    torch.manual_seed(0)
    # A = torch.randint(1, 3, (2,4), dtype=torch.float32).cuda() # M x N
    # B = torch.randint(-1, 2, (2,4), dtype=torch.float32).cuda()
    A = torch.randint(0, 10, (4,8), dtype=torch.float32).cuda() # M x N
    B = torch.randint(-1, 2, (4,8), dtype=torch.float32).cuda()
    # print("A.stride()=", A.stride())

    A_cum, B_cum = linear_recurrence_triton(A,B, BLOCK_SIZE_M=4)
    #print(f"A:\n{A}")
    #print(f"A:\n{B}")
    #print(f"A_cum:\n{A_cum}")
    #print(f"B_cum:\n{B_cum}")
    #print(A.stride())
    A_cum_ref, B_cum_ref = linear_recurrence_ref(A, B)
    # print(f"A_cum_ref:\n{A_cum_ref}")
    # print(f"B_cum_ref:\n{B_cum_ref}")
    torch.testing.assert_close(A_cum, A_cum_ref)
    torch.testing.assert_close(B_cum, B_cum_ref)

def benchmark():
    import pandas as pd
    
    results = []
    functions = [
        # ('linear_recurrence_ref', linear_recurrence_ref),
        ('linear_recurrence_triton', linear_recurrence_triton),
        # ('pscan.PScan.apply', lambda A, B: pscan.PScan.apply(A, B)),
        # ('pscan_compiled', None), #lambda A, B: pscan_compiled(A,B)),
        ('torch_cumsum', lambda A, B: (None, torch.cumsum(A * B, dim=1))),
    ]
    
    for fn_name, fn in functions:
        for d in [1, 4, 32, 128]:
            for T in [8, 64, 512, 2048, 4096]: #, 8192, 16384, 32768, 65536]:
                print(f"Running {fn_name}, d={d}, T={T}")
                #if fn_name == 'pscan_compiled': # XXX HACK See test_compiled_pscan()
                #    torch._dynamo.reset()
                #    pc = torch.compile(pscan.PScan.apply, backend="inductor", mode="reduce-overhead")
                #    fn = lambda A, B: pc(A, B)
                A = torch.randint(1, 10, (d, T), dtype=torch.float32).cuda()
                B = torch.randint(-1, 2, (d, T), dtype=torch.float32).cuda()
                
                # try:
                output, cuda_elapsed_time, wall_elapsed_time = profile_utils.wall_time_fn_s(fn, (A, B), warmup=10)
                results.append({
                    'function': fn_name,
                    'channels': d,
                    'sequence_length': T,
                    'cuda_time_ms': cuda_elapsed_time,
                    'wall_time_ms': wall_elapsed_time
                })
                #except Exception as e:
                #    print(f"Error with {fn_name}, d={d}, T={T}: {e}")
                #    results.append({
                #        'function': fn_name,
                #        'channels': d,
                #        'sequence_length': T,
                #        'cuda_time_ms': float('nan'),
                #        'wall_time_ms': float('nan')
                #    })
    
    # Convert to DataFrame and print table
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    # Pivot table for easier comparison
    pivot_cuda = df.pivot_table(
        index=['channels', 'sequence_length'], 
        columns='function', 
        values='cuda_time_ms',
        aggfunc='mean'
    )
    
    print("\nCUDA Time (ms):")
    print(pivot_cuda.to_string(float_format='%.3f'))
    
    pivot_wall = df.pivot_table(
        index=['channels', 'sequence_length'], 
        columns='function', 
        values='wall_time_ms',
        aggfunc='mean'
    )
    
    print("\nWall Time (ms):")
    print(pivot_wall.to_string(float_format='%.3f'))
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    summary = df.groupby('function')[['cuda_time_ms', 'wall_time_ms']].agg(['mean', 'std', 'min', 'max'])
    print(summary.to_string(float_format='%.3f'))
    
    return df

if __name__ == "__main__":
    test_linear_recurrence_triton()
    # test_linear_recurrence_mamba_py()
    # results_df = benchmark()
    # test_compiled_pscan()
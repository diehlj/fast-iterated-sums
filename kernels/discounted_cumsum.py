import os
TRITON_INTERPRET = 1
TRITON_INTERPRET = 0
os.environ["TRITON_INTERPRET"] = str(TRITON_INTERPRET)
# add ~/mamba.py to PYTHONPATH
# import sys
# path = os.path.expanduser("~/mamba.py/mambapy")
# print(f"Adding {path} to sys.path")
# sys.path.append(path)
import pscan
import triton
import triton.language as tl
import torch
TRITON_INTERPRET : tl.constexpr = TRITON_INTERPRET
NOT_TRITON_INTERPRET : tl.constexpr = 1 - TRITON_INTERPRET
import profile_utils

# TODO
# - [x] pytorch sequential
# - [x] https://github.com/alxndrTL/mamba.py
# - [ ] CUDA by hand
#       + baseline
#       - tedious
#       - acceleterated-scan is much more optimized
# - [x] w/ triton assoc_scan
# - [ ] torch.cumsum
#       - [ ] autotune
# - [ ] https://github.com/proger/accelerated-scan
#       - CUDA version does not work on T4
# ~~- [ ] w/ pytorch assoc_scan~~

# - [ ] benchmark
#       - [ ] torch profiling
#       - [ ] ncu (https://developer.nvidia.com/blog/using-nsight-compute-to-inspect-your-kernels/)
#       - [ ] nsys (https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/)
#              -> intro_cuda_1.cu

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

# XXX Where is this copied from ?
# related:
# https://github.com/triton-lang/triton/issues/2657
# https://github.com/triton-lang/triton/issues/2359
@triton.jit
def linear_recurrence_triton_kernel(A, B, A_cum, B_cum, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, AXIS: tl.constexpr):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    pid_z = tl.program_id(axis=2)
    # print(f"pid_x: {pid_x}, pid_y: {pid_y}, pid_z: {pid_z}")

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

# scan2d_shapes = [(8, 32), (16, 32), (32, 16), (2, 1024), (1024, 2), (32, 32), (1, 1024)]
# kernel[(1, )](x_tri, y_tri, z_tri, BLOCK_M=shape[0], BLOCK_N=shape[1], AXIS=axis, num_warps=num_warps)


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

def test_linear_recurrence_mamba_py():
    T = 8
    d = 3
    print(pscan)
    print(pscan.__file__)
    torch.manual_seed(0)
    # A = torch.randint(1, 10, (1,T,d,1), dtype=torch.float32).cuda() # M x N
    # B = torch.randint(-1, 2, (1,T,d,1), dtype=torch.float32).cuda()

    A = torch.randint(1, 10, (d,T), dtype=torch.float32).cuda() # M x N
    B = torch.randint(-1, 2, (d,T), dtype=torch.float32).cuda()
    print(f"A:\n{A.squeeze()} shape={A.shape}")
    print(f"B:\n{B.squeeze()} shape={B.shape}")
    B_cum = pscan.PScan.apply(A, B)
    print(f"B_cum:\n{B_cum.squeeze()}")
    A_cum_ref, B_cum_ref = linear_recurrence_ref(A, B)
    print(f"B_cum_ref:\n{B_cum_ref.squeeze()}")
    torch.testing.assert_close(B_cum, B_cum_ref)

def test_compiled_pscan():
    torch.manual_seed(0)
    pscan_compiled = torch.compile(pscan.PScan.apply, backend="inductor", mode="reduce-overhead")
    d = 1
    T = 8
    A = torch.randint(1, 10, (d, T), dtype=torch.float32).cuda()
    B = torch.randint(-1, 2, (d, T), dtype=torch.float32).cuda()
    pscan_compiled(A, B)

    pscan_compiled = torch.compile(pscan.PScan.apply, backend="inductor", mode="reduce-overhead")
    d = 1
    T = 64
    A = torch.randint(1, 10, (d, T), dtype=torch.float32).cuda()
    B = torch.randint(-1, 2, (d, T), dtype=torch.float32).cuda()
    pscan_compiled(A, B)
    # XXX This fails.
# V0718 16:59:42.272000 21936 /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/_inductor/graph.py:2045] [0/1] [__output_code] @triton.jit
# V0718 16:59:42.272000 21936 /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/_inductor/graph.py:2045] [0/1] [__output_code] def triton_poi_fused_add_mul_19(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
# V0718 16:59:42.272000 21936 /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/_inductor/graph.py:2045] [0/1] [__output_code]     xnumel = 1
# V0718 16:59:42.272000 21936 /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/_inductor/graph.py:2045] [0/1] [__output_code]     xoffset = tl.program_id(0) * XBLOCK
# V0718 16:59:42.272000 21936 /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/_inductor/graph.py:2045] [0/1] [__output_code]     xindex = xoffset + tl.arange(0, XBLOCK)[:]
# V0718 16:59:42.272000 21936 /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/_inductor/graph.py:2045] [0/1] [__output_code]     xmask = tl.full([XBLOCK], True, tl.int1)
# V0718 16:59:42.272000 21936 /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/_inductor/graph.py:2045] [0/1] [__output_code]     tmp3 = tl.load(in_ptr0 + (((((((-1) + 3*(libdevice.pow(2, (-2) + libdevice.trunc(OpaqueUnaryFn_log2(ks0.to(tl.float64))).to(tl.int32)))) // 2) % (ks0 // 2))) % (2*(ks0 // 4)))), None, eviction_policy='evict_last')
# V0718 16:59:42.272000 21936 /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/_inductor/graph.py:2045] [0/1] [__output_code]     tmp4 = tl.load(in_ptr1 + (2*((((2*(((((-1) + 3*(libdevice.pow(2, (-2) + libdevice.trunc(OpaqueUnaryFn_log2(ks0.to(tl.float64))).to(tl.int32)))) // 2) % (ks0 // 2))) + ((((-1) + 3*(libdevice.pow(2, (-2) + libdevice.trunc(OpaqueUnaryFn_log2(ks0.to(tl.float64))).to(tl.int32)))) % 2))) // 2) % (ks0 // 2))) + ((((-1) + 3*(libdevice.pow(2, (-2) + libdevice.trunc(OpaqueUnaryFn_log2(ks0.to(tl.float64))).to(tl.int32)))) % 2))), None, eviction_policy='evict_last')
# V0718 16:59:42.272000 21936 /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/_inductor/graph.py:2045] [0/1] [__output_code]     tmp6 = tl.load(in_ptr2 + ((((-1) + 3*(libdevice.pow(2, (-2) + libdevice.trunc(OpaqueUnaryFn_log2(ks0.to(tl.float64))).to(tl.int32)))) % (2*(ks0 // 2)))), None, eviction_policy='evict_last')
# V0718 16:59:42.272000 21936 /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/_inductor/graph.py:2045] [0/1] [__output_code]     tmp9 = tl.load(in_ptr0 + (((((((-1) + 2*(libdevice.pow(2, (-2) + libdevice.trunc(OpaqueUnaryFn_log2(ks0.to(tl.float64))).to(tl.int32)))) // 2) % (ks0 // 2))) % (2*(ks0 // 4)))), None, eviction_policy='evict_last')
# V0718 16:59:42.272000 21936 /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/_inductor/graph.py:2045] [0/1] [__output_code]     tmp10 = tl.load(in_ptr1 + (2*((((2*(((((-1) + 2*(libdevice.pow(2, (-2) + libdevice.trunc(OpaqueUnaryFn_log2(ks0.to(tl.float64))).to(tl.int32)))) // 2) % (ks0 // 2))) + ((((-1) + 2*(libdevice.pow(2, (-2) + libdevice.trunc(OpaqueUnaryFn_log2(ks0.to(tl.float64))).to(tl.int32)))) % 2))) // 2) % (ks0 // 2))) + ((((-1) + 2*(libdevice.pow(2, (-2) + libdevice.trunc(OpaqueUnaryFn_log2(ks0.to(tl.float64))).to(tl.int32)))) % 2))), None, eviction_policy='evict_last')
# V0718 16:59:42.272000 21936 /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/_inductor/graph.py:2045] [0/1] [__output_code]     tmp0 = (((-1) + 3*(libdevice.pow(2, (-2) + libdevice.trunc(OpaqueUnaryFn_log2(ks0.to(tl.float64))).to(tl.int32)))) % 2)
# V0718 16:59:42.272000 21936 /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/_inductor/graph.py:2045] [0/1] [__output_code]     tmp1 = tl.full([1], 1, tl.int32)
# V0718 16:59:42.272000 21936 /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/_inductor/graph.py:2045] [0/1] [__output_code]     tmp2 = tmp0 == tmp1
# V0718 16:59:42.272000 21936 /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/_inductor/graph.py:2045] [0/1] [__output_code]     tmp5 = tl.where(tmp2, tmp3, tmp4)
# V0718 16:59:42.272000 21936 /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/_inductor/graph.py:2045] [0/1] [__output_code]     tmp7 = (((-1) + 2*(libdevice.pow(2, (-2) + libdevice.trunc(OpaqueUnaryFn_log2(ks0.to(tl.float64))).to(tl.int32)))) % 2)
# V0718 16:59:42.272000 21936 /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/_inductor/graph.py:2045] [0/1] [__output_code]     tmp8 = tmp7 == tmp1
# V0718 16:59:42.272000 21936 /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/_inductor/graph.py:2045] [0/1] [__output_code]     tmp11 = tl.where(tmp8, tmp9, tmp10)
# V0718 16:59:42.272000 21936 /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/_inductor/graph.py:2045] [0/1] [__output_code]     tmp12 = tmp6 * tmp11
# V0718 16:59:42.272000 21936 /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/_inductor/graph.py:2045] [0/1] [__output_code]     tmp13 = tmp5 + tmp12
# V0718 16:59:42.272000 21936 /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/_inductor/graph.py:2045] [0/1] [__output_code]     tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp13, None)
# V0718 16:59:42.272000 21936 /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/_inductor/graph.py:2045] [0/1] [__output_code] ''', device_str='cuda')
#   File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/triton/compiler/compiler.py", line 100, in make_ir
#     return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
# triton.compiler.errors.CompilationError: at 6:85:
# def triton_poi_fused_add_mul_19(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
#     xnumel = 1
#     xoffset = tl.program_id(0) * XBLOCK
#     xindex = xoffset + tl.arange(0, XBLOCK)[:]
#     xmask = tl.full([XBLOCK], True, tl.int1)
#     tmp3 = tl.load(in_ptr0 + (((((((-1) + 3*(libdevice.pow(2, (-2) + libdevice.trunc(OpaqueUnaryFn_log2(ks0.to(tl.float64))).to(tl.int32)))) // 2) % (ks0 // 2))) % (2*(ks0 // 4)))), None, eviction_policy='evict_last')
#                                                                                      ^
# NameError('OpaqueUnaryFn_log2 is not defined')


def benchmark():
    import pandas as pd
    
    # pscan_compiled = torch.compile(pscan.PScan.apply, backend="inductor", mode="reduce-overhead")

    results = []
    functions = [
        # ('linear_recurrence_ref', linear_recurrence_ref),
        ('linear_recurrence_triton', linear_recurrence_triton),
        ('pscan.PScan.apply', lambda A, B: pscan.PScan.apply(A, B)),
        ('pscan_compiled', None), #lambda A, B: pscan_compiled(A,B)),
        ('torch_cumsum', lambda A, B: (None, torch.cumsum(A * B, dim=1))),
    ]
    
    for fn_name, fn in functions:
        for d in [1, 4, 32, 128]:
            for T in [8, 64, 512, 2048, 4096]: #, 8192, 16384, 32768, 65536]:
                print(f"Running {fn_name}, d={d}, T={T}")
                if fn_name == 'pscan_compiled': # XXX HACK See test_compiled_pscan()
                    torch._dynamo.reset()
                    pc = torch.compile(pscan.PScan.apply, backend="inductor", mode="reduce-overhead")
                    fn = lambda A, B: pc(A, B)
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

# Wall Time (ms):
# function                  linear_recurrence_triton  pscan.PScan.apply  pscan_compiled  torch_cumsum
# channels sequence_length                                                                           
# 1        8                                   2.287              4.659           1.688         0.438
#          64                                  1.295              9.436           2.921         0.411
#          512                                 1.184             22.696           2.001         0.505
#          2048                                1.241             18.234           2.162         0.486
#          4096                                1.551             19.781           2.612         0.478
# 4        8                                   1.373              5.011           2.104         0.600
#          64                                  1.187             10.254           2.254         0.462
#          512                                 1.096             14.845           1.960         0.426
#          2048                                1.570             17.887           2.458         0.448
#          4096                                1.294             19.796           2.329         0.590
# 32       8                                   1.100              5.200           2.127         0.433
#          64                                  1.446              9.993           2.306         0.439
#          512                                 1.288             15.345           2.085         0.441
#          2048                                1.074             18.042           2.173         0.442
#          4096                                1.140             21.472           3.205         0.544
# 128      8                                   1.140              7.094           1.808         0.533
#          64                                  1.425             11.834           1.918         0.563
#          512                                 1.549             14.433           3.067         0.478
#          2048                                1.438             18.081           3.472         0.466
#          4096                                1.507             19.494          15.076         0.806
#
#Wall Time (ms):
#function                  cumsum_triton  torch_cumsum
#channels sequence_length                             
#1        8                        0.935         0.401
#         64                       0.912         0.280
#         512                      0.928         0.269
#         2048                     0.940         0.271
#         4096                     0.973         0.270
#4        8                        0.937         0.241
#         64                       0.983         0.185
#         512                      0.935         0.249
#         2048                     0.919         0.214
#         4096                     0.949         0.244
#32       8                        1.355         0.241
#         64                       0.918         0.186
#         512                      1.532         0.220
#         2048                     0.960         0.316
#         4096                     0.946         0.315
#128      8                        0.923         0.171
#         64                       0.958         0.192
#         512                      1.021         0.279
#         2048                     0.937         0.413
#         4096                     1.136         0.505
#
#================================================================================
#SUMMARY STATISTICS
#================================================================================
#              cuda_time_ms                   wall_time_ms                  
#                      mean   std   min   max         mean   std   min   max
#function                                                                   
#cumsum_triton        0.977 0.160 0.885 1.504        1.005 0.160 0.912 1.532
#torch_cumsum         0.239 0.083 0.153 0.472        0.273 0.084 0.171 0.505
#⚡ ~ CUDA_LAUNCH_BLOCKING=1 python gpu-mode-GITHUB/cumsum_1d.py
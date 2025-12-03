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

# HACK
import sys
from pathlib import Path
# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from profiling import profile_utils

@triton.jit
def sum(x, y):
    return x + y

@triton.autotune(configs=[
    triton.Config(kwargs={'BLOCK_SIZE_M': 2}, num_warps=2),
    triton.Config(kwargs={'BLOCK_SIZE_M': 2}, num_warps=4),
    triton.Config(kwargs={'BLOCK_SIZE_M': 2}, num_warps=8),

    triton.Config(kwargs={'BLOCK_SIZE_M': 4}, num_warps=2),
    triton.Config(kwargs={'BLOCK_SIZE_M': 4}, num_warps=4),
    triton.Config(kwargs={'BLOCK_SIZE_M': 4}, num_warps=8),

    triton.Config(kwargs={'BLOCK_SIZE_M': 8}, num_warps=2),
    triton.Config(kwargs={'BLOCK_SIZE_M': 8}, num_warps=4),
    triton.Config(kwargs={'BLOCK_SIZE_M': 8}, num_warps=8),

    triton.Config(kwargs={'BLOCK_SIZE_M': 16}, num_warps=2),
    triton.Config(kwargs={'BLOCK_SIZE_M': 16}, num_warps=4),
    triton.Config(kwargs={'BLOCK_SIZE_M': 16}, num_warps=8),

    triton.Config(kwargs={'BLOCK_SIZE_M': 32}, num_warps=2),
    triton.Config(kwargs={'BLOCK_SIZE_M': 32}, num_warps=4),
    triton.Config(kwargs={'BLOCK_SIZE_M': 32}, num_warps=8),
  ],
  key=['size_m', 'BLOCK_SIZE_N'] # the two above configs will be evaluated anytime
                 # the value of size_m changes
)
@triton.jit
def cumsum_triton_kernel(X, X_cum,
                            size_m,
                            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, AXIS: tl.constexpr):
    pid_x = tl.program_id(axis=0)
    print('size_m=', size_m, 'BLOCK_SIZE_M=', BLOCK_SIZE_M, 'BLOCK_SIZE_N=', BLOCK_SIZE_N, 'AXIS=', AXIS) if TRITON_INTERPRET else None
    range_m = tl.arange(0, BLOCK_SIZE_M)
    range_n = tl.arange(0, BLOCK_SIZE_N)
    offsets_m = pid_x * BLOCK_SIZE_M * BLOCK_SIZE_N + range_m[:, None] * BLOCK_SIZE_N
    print("offsets_m=\n", offsets_m) if TRITON_INTERPRET else None
    print("mask_m=\n", mask_m) if TRITON_INTERPRET else None
    # mask_m = offsets_m < size_m
    mask_m = offsets_m < size_m * BLOCK_SIZE_N
    mask = mask_m * (range_n[None, :] < BLOCK_SIZE_N) # XXX Not sure if efficient ..
    print("mask=\n", mask) if TRITON_INTERPRET else None

    print("offsets_m + range_n[None, :]=\n", offsets_m + range_n[None, :]) if TRITON_INTERPRET else None
    idx_x = X + offsets_m + range_n[None, :]
    print("idx_x=\n", idx_x) if TRITON_INTERPRET else None

    x = tl.load(idx_x, mask=mask)           
    x_cum = tl.associative_scan(x, axis=AXIS, combine_fn=sum, reverse=False)
    tl.store(X_cum + offsets_m + range_n[None, :], x_cum, mask=mask)

def cumsum_triton(X):#, BLOCK_SIZE_M=1):
    # print('X.shape=', X.shape)
    X_cum = torch.zeros_like(X)
    grid = lambda meta: (triton.cdiv(X.shape[0], meta['BLOCK_SIZE_M']),)
    # print("grid=", grid(None))
    cumsum_triton_kernel[grid](X, X_cum, size_m=X.shape[0], BLOCK_SIZE_N=X.shape[1], AXIS=1) # BLOCK_SIZE_M=BLOCK_SIZE_M, 
    return X_cum

def test_cumsum_triton():
    X = torch.randn(3, 16, device='cuda')
    print('1')
    X_cum = cumsum_triton(X)#, BLOCK_SIZE_M=2)
    X_cum_ref = X.cumsum(dim=1)
    assert torch.allclose(X_cum, X_cum_ref, rtol=1e-5, atol=1e-5)
    return

    print('8')
    X_cum = cumsum_triton(X, BLOCK_SIZE_M=8)
    assert torch.allclose(X_cum, X_cum_ref, rtol=1e-5, atol=1e-5)
    X_cum.sum()
    print('16')
    X_cum = cumsum_triton(X, BLOCK_SIZE_M=16)
    assert torch.allclose(X_cum, X_cum_ref, rtol=1e-5, atol=1e-5)
    X_cum.sum()
    print('32')
    X_cum = cumsum_triton(X, BLOCK_SIZE_M=32)
    assert torch.allclose(X_cum, X_cum_ref, rtol=1e-5, atol=1e-5)
    X_cum.sum()
    print('64')
    X_cum = cumsum_triton(X, BLOCK_SIZE_M=64)
    assert torch.allclose(X_cum, X_cum_ref, rtol=1e-5, atol=1e-5)
    X_cum.sum()

# TODO why is Benchmarking cumsum_triton_64... taking so long !?
def benchmark(functions, ds=[1, 4, 32, 128], Ts=[8, 64, 512, 2048, 4096, 8192, 16384, 32768, 65536]):
    import pandas as pd
    
    # pscan_compiled = torch.compile(pscan.PScan.apply, backend="inductor", mode="reduce-overhead")

    results = []
    
    for fn_name, fn in functions:
        print(f"Benchmarking {fn_name}...")
        for d in ds:
            for T in Ts:
                print(f"d={d}, T={T}")
                torch.cuda.empty_cache(); torch.cuda.synchronize(); torch._dynamo.reset() # avoid RuntimeError: CUDA error: device not ready
                X = torch.randint(1, 10, (d, T), dtype=torch.float32).cuda()
                # print('### X.shape=', X.shape, 'fn_name=', fn_name)
                
                output, cuda_elapsed_time, wall_elapsed_time = profile_utils.wall_time_fn(fn, X, warmup=10)
                results.append({
                    'function': fn_name,
                    'channels': d,
                    'sequence_length': T,
                    'cuda_time_ms': cuda_elapsed_time,
                    'wall_time_ms': wall_elapsed_time
                })
    
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


def XX_ref(X):
    Y = torch.zeros_like(X)
    Z = torch.zeros_like(X)
    for i in range(X.shape[1]):
        Y[:, i] = X[:, i] + (Y[:, i-1] if i > 0 else 0)
        Z[:, i] = ( Y[:, i-1] if i > 0 else 0) * X[:,i] + (Z[:, i-1] if i > 0 else 0)
    return Y, Z


def XX_cumsum(X):
    Y = torch.cumsum(X, dim=1)
    shifted_Y = torch.cat([torch.zeros((X.shape[0], 1), device=X.device), Y[:, :-1]], dim=1)
    Z = torch.cumsum( shifted_Y * X, dim=1)
    return Y, Z

@triton.jit
def XX_op(a, b, c, a_, b_, c_):
    return a + a_, b + b_, c + c_ + b * a_

def XX_cumsum_op(X):
    x  = torch.cat( (X, X, torch.zeros_like(X)), dim=0)
    print(x, x.shape)

@triton.jit
def XX_triton_kernel(X, Y, Z,
                        size_m,
                        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, AXIS: tl.constexpr):
    pid_x = tl.program_id(axis=0)
    print('size_m=', size_m, 'BLOCK_SIZE_M=', BLOCK_SIZE_M, 'BLOCK_SIZE_N=', BLOCK_SIZE_N, 'AXIS=', AXIS) if TRITON_INTERPRET else None
    range_m = tl.arange(0, BLOCK_SIZE_M)
    range_n = tl.arange(0, BLOCK_SIZE_N)
    offsets_m = pid_x * BLOCK_SIZE_M * BLOCK_SIZE_N + range_m[:, None] * BLOCK_SIZE_N
    # mask_m = offsets_m < size_m
    mask_m = offsets_m < size_m * BLOCK_SIZE_N
    mask = mask_m * (range_n[None, :] < BLOCK_SIZE_N) # XXX Not sure if efficient ..
    print("offsets_m=\n", offsets_m) if TRITON_INTERPRET else None
    print("mask_m=\n", mask_m) if TRITON_INTERPRET else None
    print("mask=\n", mask) if TRITON_INTERPRET else None

    print("offsets_m + range_n[None, :]=\n", offsets_m + range_n[None, :]) if TRITON_INTERPRET else None
    idx_x = X + offsets_m + range_n[None, :]
    print("idx_x=\n", idx_x) if TRITON_INTERPRET else None

    x = tl.load(idx_x, mask=mask)           
    y, _, z = tl.associative_scan( (x,x,tl.zeros_like(x)), axis=AXIS, combine_fn=XX_op, reverse=False)
    tl.store(Y + offsets_m + range_n[None, :], y, mask=mask)
    tl.store(Z + offsets_m + range_n[None, :], z, mask=mask)

def XX_triton(X, BLOCK_SIZE_M=1):
    # XX_cum = torch.zeros( (X.shape[0] * 3, X.shape[1]), device=X.device, dtype=X.dtype)
    Y = torch.zeros_like(X)
    Z = torch.zeros_like(X)
    # print('X.shape=', X.shape)
    # print('XX_cum.shape=', Y.shape)
    grid = lambda meta: (triton.cdiv(X.shape[0], BLOCK_SIZE_M),)
    # print("grid=", grid(None))
    XX_triton_kernel[grid](X, Y, Z, size_m=X.shape[0], BLOCK_SIZE_N=X.shape[1], AXIS=1, BLOCK_SIZE_M=BLOCK_SIZE_M)
    return Y, Z


def test_XX():
    torch.manual_seed(0)
    X = torch.randint(1, 10, (1, 8), dtype=torch.float32).cuda()
    print("X=", X)
    Y_ref, Z_ref = XX_ref(X)
    # print("Y_ref=", Y_ref)
    # print("Z_ref=", Z_ref)

    Y_cum, Z_cum = XX_cumsum(X)
    print("Y_cum=", Y_cum)
    # print("Z_cum=", Z_cum)
    assert torch.allclose(Y_ref, Y_cum, rtol=1e-5, atol=1e-5)
    assert torch.allclose(Z_ref, Z_cum, rtol=1e-5, atol=1e-5)

    # XX_cumsum_op(X)
    Y_cum_triton, Z_cum_triton = XX_triton(X, BLOCK_SIZE_M=1)
    print("Y_cum_triton=", Y_cum_triton)
    print("Z_cum_triton=", Z_cum_triton)
    assert torch.allclose(Y_ref, Y_cum_triton, rtol=1e-5, atol=1e-5)
    assert torch.allclose(Z_ref, Z_cum_triton, rtol=1e-5, atol=1e-5)

def cdiv(x, y):
    """Compute the ceiling of x / y as an integer."""
    return (x + y - 1) // y

import torch.nn.functional as F
from einops import rearrange
def chunked_cumsum(X, chunk_size):
    T = X.shape[1]
    chunks = cdiv(T, chunk_size)
    assert chunks * chunk_size == T, f"Expected T={T} to be divisible by chunk_size={chunk_size}, but got {chunks * chunk_size}"
    Y = X.reshape(X.shape[0], chunks, chunk_size)
    print("Y.shape=", Y.shape) if TRITON_INTERPRET else None
    Y_diag = torch.cumsum(Y, dim=2)
    print("Y_diag=\n", Y_diag, Y_diag.shape) if TRITON_INTERPRET else None
    terminal_states = Y_diag[:, :, -1:]  # Last element of each chunk
    print("terminal_states=\n", terminal_states, terminal_states.shape) if TRITON_INTERPRET else None
    initial_states = torch.zeros_like(terminal_states[:, :1])
    print("initial_states=\n", initial_states, initial_states.shape) if TRITON_INTERPRET else None
    states = torch.cat([initial_states, terminal_states], dim=1)[:, :-1]  # Exclude the last element of the last chunk
    print("terminal_states after cat=\n", states, states.shape) if TRITON_INTERPRET else None
    states = states.cumsum(dim=1)
    print("states after cumsum=\n", states, states.shape) if TRITON_INTERPRET else None

    # Method 1: Using repeat_interleave
    Y_off = states.repeat_interleave(chunk_size, dim=2)
    print("states_repeated (method 1)=\n", Y_off, Y_off.shape) if TRITON_INTERPRET else None

    Y = rearrange(Y_diag+Y_off, "b c l -> b (c l)")
    return Y
    
    ## Method 2: Using unsqueeze + expand
    #states_expanded = states.unsqueeze(2).expand(-1, -1, chunk_size, -1)
    #print("states_expanded (method 2)=\n", states_expanded, states_expanded.shape)
    
    ## Method 3: Using repeat
    #states_repeat = states.unsqueeze(2).repeat(1, 1, chunk_size, 1)
    #print("states_repeat (method 3)=\n", states_repeat, states_repeat.shape)


def test_chunked_cumsum():
    torch.manual_seed(0)
    d = 2
    T = 12
    chunk_size = 4
    X = torch.randint(1, 10, (d, T), dtype=torch.float32).cuda()
    print("X=\n", X, X.shape)
    Y = chunked_cumsum(X, chunk_size)
    print("Y=\n", Y, Y.shape)
    print("Expected Y=\n", X.cumsum(dim=1), X.cumsum(dim=1).shape)
    torch.testing.assert_close(Y, X.cumsum(dim=1), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    # test_chunked_cumsum()
    # test_cumsum_triton()
    # benchmark()

    # ('cumsum_triton_2', lambda x: cumsum_triton(x, BLOCK_SIZE_M=2)),
    # ('cumsum_triton_4', lambda x: cumsum_triton(x, BLOCK_SIZE_M=4)),
    # ('cumsum_triton_8', lambda x: cumsum_triton(x, BLOCK_SIZE_M=8)),
    # ('cumsum_triton_16', lambda x: cumsum_triton(x, BLOCK_SIZE_M=16)),
    # ('cumsum_triton_32', lambda x: cumsum_triton(x, BLOCK_SIZE_M=32)),
    # ('cumsum_triton_64', lambda x: cumsum_triton(x, BLOCK_SIZE_M=64)),
    #benchmark( [
    #    ('cumsum_triton', cumsum_triton),
    #    ('torch_cumsum', lambda X: torch.cumsum(X, dim=1)),
    #], ds=[1], Ts=[128,2048] )
    benchmark( [ ('XX_cumsum', XX_cumsum), ('XX_triton', XX_triton) ], ds=[1], Ts=[8, 64, 512, 2048] )
    # BENCHMARK RESULTS
    # ================================================================================

    # CUDA Time (ms):
    # function                  XX_cumsum  XX_triton
    # channels sequence_length                      
    # 1        8                    1.270      0.935
    #          64                   1.225      0.714
    #          512                  1.231      0.682
    #          2048                 1.327      0.834

    # Wall Time (ms):
    # function                  XX_cumsum  XX_triton
    # channels sequence_length                      
    # 1        8                    1.367      0.981
    #          64                   1.269      0.755
    #          512                  1.270      0.721
    #          2048                 1.367      0.894

    # ================================================================================
    # SUMMARY STATISTICS
    # ================================================================================
    #           cuda_time_ms                   wall_time_ms                  
    #                   mean   std   min   max         mean   std   min   max
    # function                                                               
    # XX_cumsum        1.263 0.047 1.225 1.327        1.318 0.056 1.269 1.367
    # XX_triton        0.791 0.116 0.682 0.935        0.838 0.121 0.721 0.981
    #benchmark( [ ('cumsum', lambda x: torch.cumsum(x, dim=1)),
    #             ('chunked_cumsum_32', lambda x: chunked_cumsum(x, 32)),
    #             ('chunked_cumsum_64', lambda x: chunked_cumsum(x, 64)) ], Ts=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536] )
                ####
                # ================================================================================
                # SUMMARY STATISTICS
                # ================================================================================
                #                 cuda_time_ms                      wall_time_ms                     
                #                         mean    std   min     max         mean    std   min     max
                # function                                                                             
                # chunked_cumsum_32        7.059 18.002 1.309 106.638       12.148 35.503 1.341 208.579
                # chunked_cumsum_64        3.376  9.159 1.312  56.835        5.273 18.029 1.346 110.493
                # cumsum                   0.450  0.665 0.114   3.936        0.736  1.322 0.143   7.702
                ####
    # test_XX()

###########
# CUDA Time (ms):
# function                  cumsum_triton  cumsum_triton_16  cumsum_triton_2  cumsum_triton_32  cumsum_triton_4  cumsum_triton_64  cumsum_triton_8  torch_cumsum
# channels sequence_length                                                                                                                                      
# 32       2048                     1.213             1.190            0.721             1.293            0.948             1.944            1.114         0.280
# 
# Wall Time (ms):
# function                  cumsum_triton  cumsum_triton_16  cumsum_triton_2  cumsum_triton_32  cumsum_triton_4  cumsum_triton_64  cumsum_triton_8  torch_cumsum
# channels sequence_length                                                                                                                                      
# 32       2048                     1.277             1.245            0.763             1.341            1.007             1.984            1.155         0.327
# 
# ================================================================================
# SUMMARY STATISTICS
# ================================================================================
#                  cuda_time_ms                 wall_time_ms                
#                          mean std   min   max         mean std   min   max
# function                                                                  
# cumsum_triton           1.213 NaN 1.213 1.213        1.277 NaN 1.277 1.277
# cumsum_triton_16        1.190 NaN 1.190 1.190        1.245 NaN 1.245 1.245
# cumsum_triton_2         0.721 NaN 0.721 0.721        0.763 NaN 0.763 0.763
# cumsum_triton_32        1.293 NaN 1.293 1.293        1.341 NaN 1.341 1.341
# cumsum_triton_4         0.948 NaN 0.948 0.948        1.007 NaN 1.007 1.007
# cumsum_triton_64        1.944 NaN 1.944 1.944        1.984 NaN 1.984 1.984
# cumsum_triton_8         1.114 NaN 1.114 1.114        1.155 NaN 1.155 1.155
# torch_cumsum            0.280 NaN 0.280 0.280        0.327 NaN 0.327 0.327

##########
# benchmark( [ ('XX_cumsum', XX_cumsum), ('XX_triton', XX_triton) ] )
# ================================================================================
# SUMMARY STATISTICS
# ================================================================================
#           cuda_time_ms                    wall_time_ms                   
#                   mean   std   min    max         mean   std   min    max
# function                                                                 
# XX_cumsum        2.006 2.755 0.666 15.880        2.930 5.356 0.700 30.105
# XX_triton        2.894 4.522 1.198 26.848        4.112 8.983 1.242 51.711
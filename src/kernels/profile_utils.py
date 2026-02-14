import torch
import time
# time the model forward pass, properly with CUDA
def wall_time_fn(f, input_tensor, repeat=10, warmup=None):
    '''Time a function f(input_tensor) on GPU, return output, cuda time (ms), wall time (ms).'''
    while True:
        try:
            # model.eval()  # Set the model to evaluation mode
            if warmup is not None:
                for _ in range(warmup):
                    f(input_tensor)

            # with torch.no_grad():  # Disable gradient calculation
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            wall_start = time.perf_counter()
            start_time.record()
            for _ in range(repeat):
                output = f(input_tensor)
            # Wait for the events to be recorded
            torch.cuda.synchronize()
            end_time.record()
            wall_end = time.perf_counter()


            cuda_elapsed_time = start_time.elapsed_time(end_time)  # Time in milliseconds
            wall_elapsed_time = (wall_end - wall_start) * 1000  # Convert to milliseconds
            return output, cuda_elapsed_time, wall_elapsed_time
        except RuntimeError as e:
            print(f"###### Error occurred: {e}")
            print(f"###### Trying again. Waiting 2 sec ...")
            time.sleep(2)

def wall_time_fn_s(f, input_tensor, repeat=10, warmup=None):
    '''Time a function f(*input_tensor) on GPU, return output, cuda time (ms), wall time (ms).'''
    # model.eval()  # Set the model to evaluation mode
    if warmup is not None:
        print('\nwarmup')
        for _ in range(warmup):
            print('.', end='', flush=True)
            f(*input_tensor)

    # with torch.no_grad():  # Disable gradient calculation
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    print('\nstart')
    wall_start = time.perf_counter()
    start_time.record()
    for _ in range(repeat):
        print('.', end='', flush=True)
        output = f(*input_tensor)
    # Wait for the events to be recorded
    torch.cuda.synchronize()
    end_time.record()
    wall_end = time.perf_counter()
    print()

    cuda_elapsed_time = start_time.elapsed_time(end_time)  # Time in milliseconds
    wall_elapsed_time = (wall_end - wall_start) * 1000  # Convert to milliseconds
        
    return output, cuda_elapsed_time, wall_elapsed_time

import logging
import socket
from datetime import datetime, timedelta
import os
CONTAINING_DIR = os.path.dirname(os.path.abspath(__file__))
TIME_FORMAT_STR: str = "%Y-%m-%d_%H_%M_%S"
def profile_fn(fn, warmup=0, active=6, repeat=1, name_suffix=""):
    logging.basicConfig( # XXX not sure if necessary:
        format = "%(levelname)s:%(asctime)s %(message)s",
        level = logging.INFO,
        datefmt = "%Y-%m-%d %H:%M:%S",
    )
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    def trace_handler(prof: torch.profiler.profile):
        nonlocal name_suffix
        # Prefix for file names.
        host_name = socket.gethostname()
        timestamp = datetime.now().strftime(TIME_FORMAT_STR)
        # file_prefix = f"{CONTAINING_DIR }/../profiling/profiles/fis_profiling_{fn.__name__}_{name_suffix}_{timestamp}"
        file_prefix = f"{CONTAINING_DIR }/profiles/{timestamp}-{fn.__name__}_{name_suffix}"
        prof.export_chrome_trace(f"{file_prefix}.json.gz") # trace file
        prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0") # memory timeline file

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0,
                                         warmup=warmup,
                                         active=active,
                                         repeat=repeat),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=trace_handler,
    ) as prof:
        fn( prof )
        #for i in range(num_iters):
        #    print(f"Iteration {i+1}/{num_iters}")
        #    prof.step()
        #    with record_function("## sum_tree ##"):
        #        out = sum_tree(X.cuda(), tree_cherry, semiring=Semiring.REAL)
        #        print(f"Output shape: {out.shape}")

    ka = prof.key_averages()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    with open(f"{CONTAINING_DIR }/profiles/{timestamp}_{fn.__name__}_table.txt", "w") as f:
        f.write(ka.table(sort_by="cpu_time_total", row_limit=None))
    return ka
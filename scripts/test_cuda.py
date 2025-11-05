import torch
import cuda_extension
import time


def test_correctness():
    """Test that the CUDA extension produces correct results"""
    print("=" * 60)
    print("CORRECTNESS TEST")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available on this system!")
        return False

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()

    # Create a test tensor on GPU
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda")
    value_to_add = 10.0

    print(f"Input tensor: {input_tensor}")
    print(f"Value to add: {value_to_add}")

    # Call the CUDA extension
    output_tensor = cuda_extension.add_forward(input_tensor, value_to_add)

    print(f"Output tensor: {output_tensor}")

    # Verify the result
    expected = input_tensor + value_to_add
    if torch.allclose(output_tensor, expected):
        print("✓ SUCCESS! CUDA extension is working correctly!")
        return True
    else:
        print("✗ FAILED! Output doesn't match expected result")
        print(f"Expected: {expected}")
        return False


def benchmark_cuda_kernel(num_calls=500000):
    """Benchmark the CUDA kernel with many calls"""
    print("\n" + "=" * 60)
    print(f"PERFORMANCE BENCHMARK - {num_calls:,} calls")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    # Create test tensor
    input_tensor = torch.randn(1000, device="cuda")
    value_to_add = 1.0

    # Warmup (important for accurate benchmarking)
    print("Warming up...")
    for _ in range(100):
        _ = cuda_extension.add_forward(input_tensor, value_to_add)
    torch.cuda.synchronize()

    # Benchmark
    print(f"Running {num_calls:,} kernel calls...")
    start = time.time()
    start_cuda = torch.cuda.Event(enable_timing=True)
    end_cuda = torch.cuda.Event(enable_timing=True)

    start_cuda.record()
    for i in range(num_calls):
        output = cuda_extension.add_forward(input_tensor, value_to_add)
    end_cuda.record()

    torch.cuda.synchronize()
    end = time.time()

    # Results
    wall_time = end - start
    cuda_time = start_cuda.elapsed_time(end_cuda) / 1000.0  # Convert to seconds

    print()
    print(f"Total wall time:        {wall_time:.3f} seconds")
    print(f"Total CUDA time:        {cuda_time:.3f} seconds")
    print(f"Average per call:       {(cuda_time / num_calls) * 1e6:.2f} microseconds")
    print(f"Throughput:             {num_calls / cuda_time:,.0f} calls/second")
    print()


def benchmark_comparison(num_calls=10000):
    """Compare CUDA kernel vs PyTorch built-in"""
    print("=" * 60)
    print(f"CUDA vs PyTorch Comparison - {num_calls:,} calls")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping comparison")
        return

    input_tensor = torch.randn(1000, device="cuda")
    value_to_add = 1.0

    # Warmup both
    for _ in range(100):
        _ = cuda_extension.add_forward(input_tensor, value_to_add)
        _ = input_tensor + value_to_add
    torch.cuda.synchronize()

    # Benchmark CUDA kernel
    print("Benchmarking custom CUDA kernel...")
    start_cuda_event = torch.cuda.Event(enable_timing=True)
    end_cuda_event = torch.cuda.Event(enable_timing=True)

    start_cuda_event.record()
    for _ in range(num_calls):
        output = cuda_extension.add_forward(input_tensor, value_to_add)
    end_cuda_event.record()
    torch.cuda.synchronize()

    cuda_time = start_cuda_event.elapsed_time(end_cuda_event)

    # Benchmark PyTorch
    print("Benchmarking PyTorch built-in...")
    start_torch_event = torch.cuda.Event(enable_timing=True)
    end_torch_event = torch.cuda.Event(enable_timing=True)

    start_torch_event.record()
    for _ in range(num_calls):
        output = input_tensor + value_to_add
    end_torch_event.record()
    torch.cuda.synchronize()

    torch_time = start_torch_event.elapsed_time(end_torch_event)

    # Results
    print()
    print(
        f"Custom CUDA kernel:     {cuda_time:.2f} ms ({cuda_time / num_calls:.4f} ms/call)"
    )
    print(
        f"PyTorch built-in:       {torch_time:.2f} ms ({torch_time / num_calls:.4f} ms/call)"
    )
    print(f"Speedup:                {torch_time / cuda_time:.2f}x")
    print()


def main():
    print("\n" + "=" * 60)
    print("CUDA EXTENSION BENCHMARK SUITE")
    print("=" * 60 + "\n")

    # Test correctness first
    if not test_correctness():
        print("\nCorrectness test failed! Aborting benchmarks.")
        return

    # Run benchmarks
    benchmark_cuda_kernel(num_calls=500_000)
    benchmark_comparison(num_calls=50_000)

    print("=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

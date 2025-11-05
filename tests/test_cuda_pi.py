"""
uv sync --extra dev
uv run pytest tests/test_cuda_pi.py
"""

import pytest
import torch
import cuda_extension


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_add_forward():
    """Test that the add_forward CUDA kernel produces correct results"""

    # Create a test tensor on GPU
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda")
    value_to_add = 10.0

    # Call the CUDA extension
    output_tensor = cuda_extension.add_forward(input_tensor, value_to_add)

    # Verify the result
    expected = input_tensor + value_to_add
    assert torch.allclose(output_tensor, expected), (
        f"Output {output_tensor} doesn't match expected {expected}"
    )

"""
Importance Matrix (imatrix) support for quantization

The importance matrix contains per-element importance scores that help
low-bit quantization (IQ3_XXS, IQ2_XXS, etc.) maintain quality by allocating
more precision to important weights.

File format: GGUF file with metadata and per-tensor importance data
- Metadata: general.type="imatrix", chunk count, datasets
- Per-tensor: {tensor_name}.in_sum2 (sum of squared activations)
              {tensor_name}.counts (activation counts)

Reference: llama.cpp/tools/imatrix/
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import gguf


class ImportanceMatrix:
    """
    Loads and manages importance matrix data for quantization.

    The importance matrix stores per-element importance scores computed
    by running calibration data through the model and measuring activation
    magnitudes.
    """

    def __init__(self):
        self.data: Dict[str, np.ndarray] = {}  # tensor_name -> importance scores
        self.chunk_count: int = 0
        self.chunk_size: int = 0
        self.datasets: list[str] = []

    def load(self, imatrix_path: Union[str, Path]) -> bool:
        """
        Load importance matrix from GGUF file.

        Args:
            imatrix_path: Path to .gguf imatrix file

        Returns:
            True if loaded successfully, False otherwise
        """
        imatrix_path = Path(imatrix_path)

        if not imatrix_path.exists():
            print(f"Error: imatrix file not found: {imatrix_path}")
            return False

        try:
            print(f"Loading importance matrix from {imatrix_path}")

            # Load GGUF file
            reader = gguf.GGUFReader(imatrix_path)

            # Read metadata
            self._load_metadata(reader)

            # Read tensor importance data
            self._load_tensor_data(reader)

            print(f"Loaded imatrix: {len(self.data)} tensors, {self.chunk_count} chunks")
            print(f"Datasets: {', '.join(self.datasets) if self.datasets else 'unknown'}")

            return True

        except Exception as e:
            print(f"Error loading imatrix: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_field_value(self, field):
        """Safely extract value from GGUF field, handling different types"""
        if not field.parts:
            return None

        # Determine the value from field parts and data index
        val = field.parts[field.data[0]]

        # Handle different potential types
        if isinstance(val, np.ndarray):
            # Convert numpy array of bytes/integers to a string
            return "".join(map(chr, val))
        elif isinstance(val, bytes):
            # Decode bytes to string
            return val.decode("utf-8")
        else:
            # Assume it's already a usable type (int, float, etc.)
            return val

    def _load_metadata(self, reader: gguf.GGUFReader):
        """Load metadata fields from GGUF file"""
        for field in reader.fields.values():
            if field.name == "general.type":
                file_type = self._get_field_value(field)
                if file_type != "imatrix":
                    raise ValueError(f"Not an imatrix file (type={file_type})")

            elif field.name == "imatrix.chunks.count":
                val = self._get_field_value(field)
                if val is not None:
                    self.chunk_count = int(val)

            elif field.name == "imatrix.chunks.size":
                val = self._get_field_value(field)
                if val is not None:
                    self.chunk_size = int(val)

            elif field.name == "imatrix.datasets":
                # Handle array of strings for datasets
                if not hasattr(field, 'parts') or not field.parts or not hasattr(field, 'data'):
                    continue
                datasets = []
                # For array fields, field.data contains the list of indices into field.parts
                for part_index in field.data:
                    val = field.parts[part_index]
                    if isinstance(val, np.ndarray):
                        # Handle numpy array of chars/bytes
                        datasets.append("".join(map(chr, val)))
                    elif isinstance(val, bytes):
                        # Handle bytes
                        datasets.append(val.decode("utf-8"))
                    else:
                        # Fallback for other types
                        datasets.append(str(val))
                self.datasets = datasets

    def _load_tensor_data(self, reader: gguf.GGUFReader):
        """
        Load per-tensor importance scores.

        For each tensor, there are two tensors stored:
        - {tensor_name}.in_sum2: sum of squared activations (FP32)
        - {tensor_name}.counts: activation counts (FP32)

        We compute importance = in_sum2 / counts
        """
        # Build mapping of tensor names
        tensor_map = {}
        for tensor in reader.tensors:
            tensor_map[tensor.name] = tensor

        # Find pairs of .in_sum2 and .counts tensors
        processed = set()

        for tensor_name in tensor_map.keys():
            if tensor_name.endswith('.in_sum2'):
                base_name = tensor_name[:-len('.in_sum2')]
                counts_name = base_name + '.counts'

                if counts_name in tensor_map and base_name not in processed:
                    # Load both tensors
                    in_sum2_tensor = tensor_map[tensor_name]
                    counts_tensor = tensor_map[counts_name]

                    # The GGUFReader already loaded the tensor data.
                    in_sum2_data = in_sum2_tensor.data
                    counts_data = counts_tensor.data

                    # Convert F16 to F32 if necessary, as importance calculations should be in F32.
                    if in_sum2_data.dtype == np.float16:
                        in_sum2_data = in_sum2_data.astype(np.float32)
                    if counts_data.dtype == np.float16:
                        counts_data = counts_data.astype(np.float32)

                    # Compute importance scores: in_sum2 / counts
                    # Avoid division by zero
                    importance = np.zeros_like(in_sum2_data)

                    try:
                        # Use broadcast_arrays to handle shape mismatches safely.
                        # This will make both arrays have the same broadcasted shape.
                        sum2_b, counts_b = np.broadcast_arrays(in_sum2_data, counts_data)
                        
                        mask = counts_b > 0
                        importance[mask] = sum2_b[mask] / counts_b[mask]
                    except ValueError:
                        print(
                            f"Warning: Imatrix shape mismatch for tensor {base_name}. "
                            f"in_sum2 shape: {in_sum2_data.shape}, counts shape: {counts_data.shape}. "
                            f"Cannot broadcast. Using uniform importance for this tensor."
                        )
                        # When shapes are incompatible, fall back to uniform importance.
                        importance.fill(1.0)

                    # Store flattened importance scores
                    self.data[base_name] = importance.flatten()
                    processed.add(base_name)

    def get_tensor_importance(self, tensor_name: str, tensor_shape: Tuple[int, ...]) -> Optional[np.ndarray]:
        """
        Get importance scores for a tensor.

        Args:
            tensor_name: Name of the tensor
            tensor_shape: Shape of the tensor

        Returns:
            Importance scores array matching tensor shape, or None if not found
        """
        if tensor_name not in self.data:
            return None

        importance = self.data[tensor_name]

        # Reshape to match tensor
        n_elements = np.prod(tensor_shape)

        if len(importance) != n_elements:
            # Try to broadcast or pad
            if len(importance) == tensor_shape[0]:
                # Have importance per row, broadcast to full tensor
                importance = np.repeat(importance, n_elements // len(importance))
            else:
                print(f"Warning: importance shape mismatch for {tensor_name}: "
                      f"{len(importance)} vs {n_elements}")
                return None

        return importance.reshape(tensor_shape)

    def has_tensor(self, tensor_name: str) -> bool:
        """Check if importance data exists for a tensor"""
        return tensor_name in self.data

    def get_coverage(self) -> float:
        """Get fraction of tensors with importance data"""
        return len(self.data)

    def __len__(self) -> int:
        """Number of tensors with importance data"""
        return len(self.data)

    def __repr__(self) -> str:
        return f"ImportanceMatrix({len(self.data)} tensors, {self.chunk_count} chunks)"


def test_imatrix_loader():
    """Test imatrix loading (requires an actual imatrix file)"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python imatrix.py <path_to_imatrix.gguf>")
        print("\nTo generate an imatrix file with llama.cpp:")
        print("  llama-imatrix -m model_f16.gguf -f calibration.txt -o model.imatrix")
        return

    imatrix_path = sys.argv[1]

    print("=" * 80)
    print("IMATRIX LOADER TEST")
    print("=" * 80)

    imatrix = ImportanceMatrix()

    if imatrix.load(imatrix_path):
        print("\n" + "-" * 80)
        print("LOADED DATA")
        print("-" * 80)

        print(f"\nTensors with importance data: {len(imatrix)}")
        print(f"Chunks processed: {imatrix.chunk_count}")
        print(f"Chunk size: {imatrix.chunk_size}")

        # Show first few tensors
        print("\nFirst 10 tensors:")
        for i, (name, importance) in enumerate(list(imatrix.data.items())[:10]):
            print(f"  {i+1}. {name}: {importance.shape}, "
                  f"range [{importance.min():.6f}, {importance.max():.6f}], "
                  f"mean {importance.mean():.6f}")

        print("\n" + "=" * 80)
        print("TEST PASSED")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("TEST FAILED")
        print("=" * 80)


if __name__ == "__main__":
    test_imatrix_loader()

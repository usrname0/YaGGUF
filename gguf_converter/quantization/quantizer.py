"""
Pure Python GGUF Quantizer - processes entire GGUF files
"""

import numpy as np
from pathlib import Path
from typing import Union, Optional
import gguf

from .q8_0 import quantize_q8_0, QK8_0
from .q4_0 import quantize_q4_0, QK4_0
from .q4_1 import quantize_q4_1, QK4_1
from .q5_0 import quantize_q5_0, QK5_0
from .q5_1 import quantize_q5_1, QK5_1
from .q2_k import quantize_q2_k
from .q3_k import quantize_q3_k
from .q4_k import quantize_q4_k
from .q5_k import quantize_q5_k
from .q6_k import quantize_q6_k, QK_K
from .iq3_xxs import quantize_iq3_xxs, QK_K as IQ3_XXS_QK, BYTES_PER_BLOCK as IQ3_XXS_BYTES_PER_BLOCK
from .iq3_s import quantize_iq3_s, QK_IQ3S, BYTES_PER_BLOCK as IQ3_S_BYTES_PER_BLOCK
from .iq4_nl import quantize_iq4_nl, QK4_NL, BYTES_PER_BLOCK as IQ4_NL_BYTES_PER_BLOCK


class PythonQuantizer:
    """Pure Python GGUF quantizer"""

    # Maps quantization type names to (GGMLQuantizationType, LlamaFileType)
    SUPPORTED_TYPES = {
        # Base types
        "Q4_0": (gguf.GGMLQuantizationType.Q4_0, 2),      # MOSTLY_Q4_0
        "Q4_1": (gguf.GGMLQuantizationType.Q4_1, 3),      # MOSTLY_Q4_1
        "Q5_0": (gguf.GGMLQuantizationType.Q5_0, 8),      # MOSTLY_Q5_0
        "Q5_1": (gguf.GGMLQuantizationType.Q5_1, 9),      # MOSTLY_Q5_1
        "Q2_K": (gguf.GGMLQuantizationType.Q2_K, 10),     # MOSTLY_Q2_K
        "Q3_K": (gguf.GGMLQuantizationType.Q3_K, 11),     # MOSTLY_Q3_K
        "Q4_K": (gguf.GGMLQuantizationType.Q4_K, 12),     # MOSTLY_Q4_K
        "Q5_K": (gguf.GGMLQuantizationType.Q5_K, 13),     # MOSTLY_Q5_K
        "Q6_K": (gguf.GGMLQuantizationType.Q6_K, 18),     # MOSTLY_Q6_K
        "Q8_0": (gguf.GGMLQuantizationType.Q8_0, 7),      # MOSTLY_Q8_0
        # IQ types (pure Python implementation)
        "IQ3_XXS": (gguf.GGMLQuantizationType.IQ3_XXS, 18),  # 3.06 bpw quantization
        "IQ3_S": (gguf.GGMLQuantizationType.IQ3_S, 19),  # 3.44 bpw quantization
        "IQ4_NL": (gguf.GGMLQuantizationType.IQ4_NL, 20),  # 4-bit non-linear
        # K-quant variants (use different file types but same algorithms)
        "Q3_K_S": (gguf.GGMLQuantizationType.Q3_K, 11),   # Small variant
        "Q3_K_M": (gguf.GGMLQuantizationType.Q3_K, 12),   # Medium variant
        "Q3_K_L": (gguf.GGMLQuantizationType.Q3_K, 13),   # Large variant
        "Q4_K_S": (gguf.GGMLQuantizationType.Q4_K, 14),   # Small variant
        "Q4_K_M": (gguf.GGMLQuantizationType.Q4_K, 15),   # Medium variant (RECOMMENDED)
        "Q5_K_S": (gguf.GGMLQuantizationType.Q5_K, 16),   # Small variant
        "Q5_K_M": (gguf.GGMLQuantizationType.Q5_K, 17),   # Medium variant
    }

    def __init__(self):
        """Initialize quantizer"""
        # Tensor counters for variant logic (matches llama.cpp's qs struct)
        self.n_layers = 0
        self.n_attention_wv = 0
        self.n_gqa = 1                   # Grouped query attention ratio (for IQ3_XXS)
        self.has_imatrix = False         # Whether importance matrix is loaded
        self.tensor_layer_map = {}      # Maps tensor name to layer index
        self.tensor_indices = {}         # Maps tensor name to its index (for parallel mode)

    def _prescan_model(self, reader, verbose: bool = False):
        """
        Pre-scan the model to count layers and important tensors.
        Also pre-assigns indices for each tensor type (enables parallel processing).
        Matches llama.cpp's approach for variant quantization.
        """
        import re

        max_layer = -1

        # Counters for different tensor types
        i_attention_wv = 0
        i_ffn_down = 0
        i_ffn_gate = 0

        for tensor in reader.tensors:
            name = tensor.name
            name_lower = name.lower()

            # Extract layer number
            match = re.search(r'(?:blk|layer|h)\.(\d+)\.', name_lower)
            if match:
                layer_idx = int(match.group(1))
                max_layer = max(max_layer, layer_idx)
                self.tensor_layer_map[name] = layer_idx

            # Assign indices for each tensor type (for parallel mode)
            tensor_info = {}

            if 'attn_v.weight' in name_lower or 'attention.wv' in name_lower:
                tensor_info['attn_wv_idx'] = i_attention_wv
                i_attention_wv += 1

            if 'ffn_down' in name_lower:
                tensor_info['ffn_down_idx'] = i_ffn_down
                i_ffn_down += 1

            if 'ffn_gate' in name_lower:
                tensor_info['ffn_gate_idx'] = i_ffn_gate
                i_ffn_gate += 1

            if tensor_info:
                self.tensor_indices[name] = tensor_info

        self.n_layers = max_layer + 1 if max_layer >= 0 else 0
        self.n_attention_wv = i_attention_wv

        if verbose:
            print(f"Pre-scan: Found {self.n_layers} layers, {self.n_attention_wv} attention_wv tensors", flush=True)

    def _use_more_bits(self, i_layer: int, n_layers: int) -> bool:
        """
        Determine if a layer should use more bits for quantization.
        Matches llama.cpp's use_more_bits lambda function.

        Returns True for:
        - First 1/8 of layers (early layers need precision)
        - Last 1/8 of layers (late layers need precision)
        - Every 3rd layer in the middle section
        """
        if n_layers == 0:
            return False

        return (i_layer < n_layers // 8 or
                i_layer >= 7 * n_layers // 8 or
                (i_layer - n_layers // 8) % 3 == 2)

    def quantize_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        quantization_type: str = "Q8_0",
        verbose: bool = False,
        parallel: bool = True,
        num_workers: Optional[int] = None,
        scalar_optimization: bool = False,
        imatrix_path: Optional[Union[str, Path]] = None
    ):
        """
        Quantize a GGUF file

        Args:
            input_path: Path to input GGUF file (f16/f32)
            output_path: Path for output quantized file
            quantization_type: Quantization type
            verbose: Print progress information
            parallel: Enable parallel tensor processing (default: True, 7-8x faster)
            num_workers: Number of worker processes (None = CPU cores - 1)
            scalar_optimization: Enable iterative optimization for better quality (20x slower)
            imatrix_path: Optional importance matrix file for IQ3_XXS and other low-bit quantizations

        Raises:
            ValueError: If quantization type not supported
            FileNotFoundError: If input file doesn't exist
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if quantization_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported quantization type: {quantization_type}. "
                f"Supported: {list(self.SUPPORTED_TYPES.keys())}"
            )

        # Load importance matrix if provided
        imatrix = None
        if imatrix_path is not None:
            imatrix_path = Path(imatrix_path)
            if not imatrix_path.exists():
                raise FileNotFoundError(f"Imatrix file not found: {imatrix_path}")

            if verbose:
                print(f"Loading importance matrix from {imatrix_path}...", flush=True)

            from .imatrix import ImportanceMatrix
            imatrix = ImportanceMatrix()
            if not imatrix.load(str(imatrix_path)):
                raise ValueError(f"Failed to load importance matrix from {imatrix_path}")

            self.has_imatrix = True  # Mark that we have imatrix loaded
            if verbose:
                print(f"Loaded imatrix: {len(imatrix)} tensors", flush=True)
        else:
            self.has_imatrix = False

        if verbose:
            print(f"Loading {input_path.name}...", flush=True)

        # Notify if scalar optimization is enabled
        if scalar_optimization:
            # Different messages for different optimization algorithms
            if quantization_type == "Q6_K":
                print(f"Scalar optimization: ENABLED (19-iteration search, ~20x slower)", flush=True)
            elif quantization_type == "Q3_K":
                print(f"Scalar optimization: ENABLED (iterative refinement, ~5x slower)", flush=True)
            elif quantization_type in ["Q4_K", "Q4_K_S", "Q4_K_M", "Q5_K", "Q5_K_S", "Q5_K_M"]:
                print(f"Scalar optimization: ENABLED (grid search, ~10-15x slower)", flush=True)
            else:
                # Generic message for future optimized types
                print(f"Scalar optimization: ENABLED (may be significantly slower)", flush=True)

        # Read input GGUF file
        reader = gguf.GGUFReader(input_path)

        # Get architecture
        arch_field = reader.fields.get('general.architecture')
        if arch_field:
            arch_bytes = arch_field.parts[arch_field.data[0]]
            arch = bytes(arch_bytes).decode('utf-8')
        else:
            arch = 'llama'  # Default fallback

        # Extract n_gqa (grouped query attention) for IQ3_XXS variant selection
        # n_gqa = attention_head_count / attention_head_count_kv
        self.n_gqa = 1  # Default
        try:
            # Try to find attention head counts in metadata
            head_count_key = f'{arch}.attention.head_count'
            head_count_kv_key = f'{arch}.attention.head_count_kv'

            head_count_field = reader.fields.get(head_count_key)
            head_count_kv_field = reader.fields.get(head_count_kv_key)

            if head_count_field and head_count_kv_field:
                # Extract values
                head_count = head_count_field.parts[head_count_field.data[0]][0]
                head_count_kv = head_count_kv_field.parts[head_count_kv_field.data[0]][0]
                self.n_gqa = int(head_count) // int(head_count_kv)
                if verbose:
                    print(f"Model n_gqa (grouped query attention): {self.n_gqa} ({head_count} heads / {head_count_kv} kv heads)", flush=True)
        except Exception as e:
            if verbose:
                print(f"Warning: Could not extract n_gqa, using default: {e}", flush=True)

        if verbose:
            print(f"Architecture: {arch}", flush=True)
            print(f"Tensors: {len(reader.tensors)}", flush=True)

        # Pre-scan model only for variants (base types don't need it)
        is_variant = quantization_type not in ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q3_K", "Q4_K", "Q5_K", "Q6_K"]
        if is_variant:
            if verbose:
                print(f"Pre-scanning for variant {quantization_type}...", flush=True)
            self._prescan_model(reader, verbose)

        # Create writer for output
        writer = gguf.GGUFWriter(output_path, arch=arch)

        # Copy all metadata/fields from input to output
        # This includes tokenizer config, model hyperparameters, etc.
        if verbose:
            print(f"Copying {len(reader.fields)} metadata fields...", flush=True)

        for key, field in reader.fields.items():
            # Skip internal GGUF fields that are auto-generated by the writer
            if key.startswith('GGUF.'):
                continue

            # Skip general.architecture as it's set by GGUFWriter constructor
            if key == 'general.architecture':
                continue

            # Skip general.file_type as we'll set it based on our quantization type
            if key == 'general.file_type':
                continue

            # Copy the field to the output
            try:
                # Check if this is an array type by looking at the value type
                # Arrays have ARRAY type code (9) in gguf
                field_type = field.types[0] if hasattr(field, 'types') else None
                is_array = (field_type == gguf.GGUFValueType.ARRAY if field_type else False)

                # Get the field value - handle different field types
                if not is_array and len(field.data) == 1:
                    # Single scalar value - field.data[0] is an index into parts
                    idx = field.data[0]

                    # Extract the actual value from parts
                    if hasattr(field, 'parts') and field.parts:
                        if field_type == gguf.GGUFValueType.STRING:
                            # For strings, parts[idx] contains the byte array
                            string_value = bytes(field.parts[idx]).decode('utf-8')
                            writer.add_string(key, string_value)
                        else:
                            # For scalars, parts[idx][0] contains the value
                            # Convert from numpy type to Python native type
                            value = field.parts[idx][0]
                            if field_type == gguf.GGUFValueType.UINT32:
                                writer.add_uint32(key, value.item() if hasattr(value, 'item') else int(value))
                            elif field_type == gguf.GGUFValueType.INT32:
                                writer.add_int32(key, value.item() if hasattr(value, 'item') else int(value))
                            elif field_type == gguf.GGUFValueType.FLOAT32:
                                writer.add_float32(key, value.item() if hasattr(value, 'item') else float(value))
                            elif field_type == gguf.GGUFValueType.UINT64:
                                writer.add_uint64(key, value.item() if hasattr(value, 'item') else int(value))
                            elif field_type == gguf.GGUFValueType.INT64:
                                writer.add_int64(key, value.item() if hasattr(value, 'item') else int(value))
                            elif field_type == gguf.GGUFValueType.FLOAT64:
                                writer.add_float64(key, value.item() if hasattr(value, 'item') else float(value))
                            elif field_type == gguf.GGUFValueType.BOOL:
                                writer.add_bool(key, value.item() if hasattr(value, 'item') else bool(value))
                            else:
                                # Fallback
                                if verbose:
                                    print(f"  Warning: Unknown field type for {key}, attempting generic add", flush=True)
                                writer.add_uint32(key, int(value))
                    else:
                        # No parts - shouldn't happen for valid GGUF
                        if verbose:
                            print(f"  Warning: Field {key} has no parts, skipping", flush=True)
                else:
                    # Array of values - need to extract from parts too
                    # For arrays, check if there's a subtype
                    sub_type = field.types[1] if hasattr(field, 'types') and len(field.types) > 1 else None

                    if hasattr(field, 'parts') and field.parts:
                        # Extract array values from parts
                        array_values = []
                        for idx in field.data:
                            if sub_type == gguf.GGUFValueType.STRING:
                                # Array of strings
                                array_values.append(bytes(field.parts[idx]).decode('utf-8'))
                            else:
                                # Array of numbers
                                val = field.parts[idx][0]
                                # Convert numpy types to Python native types
                                array_values.append(val.item() if hasattr(val, 'item') else val)
                        writer.add_array(key, array_values)
                    else:
                        # Fallback - use data directly
                        writer.add_array(key, field.data)

            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not copy field {key}: {e}", flush=True)
                # Continue anyway - some fields might not be critical

        # Process each tensor (parallel or serial)
        if verbose:
            mode = "parallel" if parallel else "serial"
            opt_status = "with optimization" if scalar_optimization else "fast mode"
            print(f"Quantizing in {mode} mode ({opt_status})...", flush=True)

        if parallel:
            self._quantize_tensors_parallel(
                reader, writer, quantization_type, verbose, num_workers, scalar_optimization, imatrix
            )
        else:
            self._quantize_tensors_serial(
                reader, writer, quantization_type, verbose, scalar_optimization, imatrix
            )

        # Set the output file type based on our quantization type
        # Must be done after all tensors are added but before writing
        ggml_type, file_type = self.SUPPORTED_TYPES[quantization_type]
        writer.add_uint32("general.file_type", file_type)

        if verbose:
            print(f"Set file_type to {file_type} for {quantization_type}", flush=True)

        # Write output file
        if verbose:
            print(f"\nWriting {output_path.name}...", flush=True)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        if verbose:
            input_size = input_path.stat().st_size / (1024**3)
            output_size = output_path.stat().st_size / (1024**3)
            ratio = input_size / output_size
            print(f"\nDone!", flush=True)
            print(f"Input:  {input_size:.2f} GB", flush=True)
            print(f"Output: {output_size:.2f} GB", flush=True)
            print(f"Compression: {ratio:.2f}x", flush=True)

    def _quantize_tensors_serial(
        self,
        reader,
        writer,
        quantization_type: str,
        verbose: bool,
        scalar_optimization: bool = False,
        imatrix=None
    ):
        """Process tensors serially (original implementation)"""
        total_tensors = len(reader.tensors)
        for idx, tensor in enumerate(reader.tensors):
            # Always print progress (matching parallel behavior)
            print(f"[{idx+1}/{total_tensors}] {tensor.name}: ", end='', flush=True)

            try:
                # Get tensor data
                tensor_data = tensor.data

                # Determine if we should quantize this tensor
                # Skip already quantized tensors or special tensors
                should_quantize = self._should_quantize_tensor(tensor)

                if should_quantize:
                    # Select actual quantization type based on variant strategy
                    # Uses pre-scan data stored in instance variables
                    actual_type = self._select_quantization_type_for_tensor(
                        tensor.name,
                        quantization_type
                    )

                    # Debug: log variant selection
                    if verbose and actual_type != quantization_type:
                        print(f"[{quantization_type}->{actual_type}] ", end='', flush=True)

                    # Debug: log shapes before quantization
                    if verbose:
                        print(f"tensor.shape={tensor.shape}, tensor.data.shape={tensor.data.shape} -> ", end='', flush=True)

                    # Quantize using the selected type
                    if actual_type == "Q4_0":
                        quantized_bytes = quantize_q4_0(tensor_data)
                        quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8)

                        # Calculate quantized shape for Q4_0
                        # For Q4_0: 32 elements become 18 bytes (2 scale + 16 data)
                        orig_shape = list(tensor_data.shape)

                        if len(orig_shape) == 0:
                            quant_shape = quantized_data.shape
                        else:
                            last_dim = orig_shape[-1]
                            blocks_in_last_dim = (last_dim + QK4_0 - 1) // QK4_0
                            bytes_in_last_dim = blocks_in_last_dim * 18  # 18 bytes per Q4_0 block

                            if len(orig_shape) == 1:
                                quant_shape = (bytes_in_last_dim,)
                            else:
                                quant_shape = tuple(orig_shape[:-1]) + (bytes_in_last_dim,)

                        quantized_data = quantized_data.reshape(quant_shape)
                        ggml_type, _ = self.SUPPORTED_TYPES[actual_type]
                        quant_type = ggml_type

                    elif actual_type == "Q4_1":
                        quantized_bytes = quantize_q4_1(tensor_data)
                        quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8)

                        orig_shape = list(tensor_data.shape)
                        if len(orig_shape) == 0:
                            quant_shape = quantized_data.shape
                        else:
                            last_dim = orig_shape[-1]
                            blocks_in_last_dim = (last_dim + QK4_1 - 1) // QK4_1
                            bytes_in_last_dim = blocks_in_last_dim * 20  # 20 bytes per Q4_1 block

                            if len(orig_shape) == 1:
                                quant_shape = (bytes_in_last_dim,)
                            else:
                                quant_shape = tuple(orig_shape[:-1]) + (bytes_in_last_dim,)

                        quantized_data = quantized_data.reshape(quant_shape)
                        ggml_type, _ = self.SUPPORTED_TYPES[actual_type]
                        quant_type = ggml_type

                    elif actual_type == "Q5_0":
                        quantized_bytes = quantize_q5_0(tensor_data)
                        quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8)

                        orig_shape = list(tensor_data.shape)
                        if len(orig_shape) == 0:
                            quant_shape = quantized_data.shape
                        else:
                            last_dim = orig_shape[-1]
                            blocks_in_last_dim = (last_dim + QK5_0 - 1) // QK5_0
                            bytes_in_last_dim = blocks_in_last_dim * 22  # 22 bytes per Q5_0 block

                            if len(orig_shape) == 1:
                                quant_shape = (bytes_in_last_dim,)
                            else:
                                quant_shape = tuple(orig_shape[:-1]) + (bytes_in_last_dim,)

                        quantized_data = quantized_data.reshape(quant_shape)
                        ggml_type, _ = self.SUPPORTED_TYPES[actual_type]
                        quant_type = ggml_type

                    elif actual_type == "Q5_1":
                        quantized_bytes = quantize_q5_1(tensor_data)
                        quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8)

                        orig_shape = list(tensor_data.shape)
                        if len(orig_shape) == 0:
                            quant_shape = quantized_data.shape
                        else:
                            last_dim = orig_shape[-1]
                            blocks_in_last_dim = (last_dim + QK5_1 - 1) // QK5_1
                            bytes_in_last_dim = blocks_in_last_dim * 24  # 24 bytes per Q5_1 block

                            if len(orig_shape) == 1:
                                quant_shape = (bytes_in_last_dim,)
                            else:
                                quant_shape = tuple(orig_shape[:-1]) + (bytes_in_last_dim,)

                        quantized_data = quantized_data.reshape(quant_shape)
                        ggml_type, _ = self.SUPPORTED_TYPES[actual_type]
                        quant_type = ggml_type

                    elif actual_type == "IQ3_XXS":
                        # Get importance for this tensor if imatrix is available
                        importance = None
                        if imatrix is not None:
                            importance = imatrix.get_tensor_importance(tensor.name, tensor_data.shape)
                            if importance is None and verbose:
                                print(f"[no imatrix] ", end='', flush=True)

                        quantized_bytes = quantize_iq3_xxs(tensor_data, importance=importance)
                        quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8)

                        orig_shape = list(tensor_data.shape)
                        if len(orig_shape) == 0:
                            quant_shape = quantized_data.shape
                        else:
                            last_dim = orig_shape[-1]
                            blocks_in_last_dim = (last_dim + IQ3_XXS_QK - 1) // IQ3_XXS_QK
                            bytes_in_last_dim = blocks_in_last_dim * IQ3_XXS_BYTES_PER_BLOCK  # 98 bytes per IQ3_XXS block

                            if len(orig_shape) == 1:
                                quant_shape = (bytes_in_last_dim,)
                            else:
                                quant_shape = tuple(orig_shape[:-1]) + (bytes_in_last_dim,)

                        quantized_data = quantized_data.reshape(quant_shape)
                        ggml_type, _ = self.SUPPORTED_TYPES[actual_type]
                        quant_type = ggml_type

                    elif actual_type == "IQ3_S":
                        # IQ3_S quantization now fully implemented!
                        from .iq3_s import quantize_iq3_s, QK_IQ3S, BYTES_PER_BLOCK as IQ3_S_BYTES_PER_BLOCK

                        # Get importance if available
                        importance = None
                        if imatrix is not None:
                            importance = imatrix.get_tensor_importance(tensor.name, tensor_data.shape)
                            if importance is None and verbose:
                                print(f"[no imatrix] ", end='', flush=True)

                        quantized_bytes = quantize_iq3_s(tensor_data, importance=importance)
                        quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8)

                        # Calculate shape
                        orig_shape = list(tensor_data.shape)
                        if len(orig_shape) == 0:
                            quant_shape = quantized_data.shape
                        else:
                            last_dim = orig_shape[-1]
                            blocks_in_last_dim = (last_dim + QK_IQ3S - 1) // QK_IQ3S
                            bytes_in_last_dim = blocks_in_last_dim * IQ3_S_BYTES_PER_BLOCK  # 110 bytes

                            if len(orig_shape) == 1:
                                quant_shape = (bytes_in_last_dim,)
                            else:
                                quant_shape = tuple(orig_shape[:-1]) + (bytes_in_last_dim,)

                        quantized_data = quantized_data.reshape(quant_shape)
                        ggml_type, _ = self.SUPPORTED_TYPES[actual_type]
                        quant_type = ggml_type

                    elif actual_type == "IQ4_NL":
                        # Get importance if available
                        importance = None
                        if imatrix is not None:
                            importance = imatrix.get_tensor_importance(tensor.name, tensor_data.shape)
                            if importance is None and verbose:
                                print(f"[no imatrix] ", end='', flush=True)

                        quantized_bytes = quantize_iq4_nl(tensor_data, importance=importance)
                        quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8)

                        orig_shape = list(tensor_data.shape)
                        if len(orig_shape) == 0:
                            quant_shape = quantized_data.shape
                        else:
                            last_dim = orig_shape[-1]
                            blocks_in_last_dim = (last_dim + QK4_NL - 1) // QK4_NL
                            bytes_in_last_dim = blocks_in_last_dim * IQ4_NL_BYTES_PER_BLOCK  # 18 bytes per IQ4_NL block

                            if len(orig_shape) == 1:
                                quant_shape = (bytes_in_last_dim,)
                            else:
                                quant_shape = tuple(orig_shape[:-1]) + (bytes_in_last_dim,)

                        quantized_data = quantized_data.reshape(quant_shape)
                        ggml_type, _ = self.SUPPORTED_TYPES[actual_type]
                        quant_type = ggml_type

                    elif actual_type == "Q2_K":
                        quantized_bytes = quantize_q2_k(tensor_data, scalar_optimization=scalar_optimization)
                        quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8)

                        orig_shape = list(tensor_data.shape)
                        if len(orig_shape) == 0:
                            quant_shape = quantized_data.shape
                        else:
                            last_dim = orig_shape[-1]
                            blocks_in_last_dim = (last_dim + QK_K - 1) // QK_K
                            bytes_in_last_dim = blocks_in_last_dim * 84  # 84 bytes per Q2_K super-block

                            if len(orig_shape) == 1:
                                quant_shape = (bytes_in_last_dim,)
                            else:
                                quant_shape = tuple(orig_shape[:-1]) + (bytes_in_last_dim,)

                        quantized_data = quantized_data.reshape(quant_shape)
                        ggml_type, _ = self.SUPPORTED_TYPES[actual_type]
                        quant_type = ggml_type

                    elif actual_type == "Q3_K":
                        quantized_bytes = quantize_q3_k(tensor_data, scalar_optimization=scalar_optimization)
                        quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8)

                        orig_shape = list(tensor_data.shape)
                        if len(orig_shape) == 0:
                            quant_shape = quantized_data.shape
                        else:
                            last_dim = orig_shape[-1]
                            blocks_in_last_dim = (last_dim + QK_K - 1) // QK_K
                            bytes_in_last_dim = blocks_in_last_dim * 110  # 110 bytes per Q3_K super-block

                            if len(orig_shape) == 1:
                                quant_shape = (bytes_in_last_dim,)
                            else:
                                quant_shape = tuple(orig_shape[:-1]) + (bytes_in_last_dim,)

                        quantized_data = quantized_data.reshape(quant_shape)
                        ggml_type, _ = self.SUPPORTED_TYPES[actual_type]
                        quant_type = ggml_type

                    elif actual_type == "Q4_K":
                        quantized_bytes = quantize_q4_k(tensor_data, scalar_optimization=scalar_optimization)
                        quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8)

                        orig_shape = list(tensor_data.shape)
                        if len(orig_shape) == 0:
                            quant_shape = quantized_data.shape
                        else:
                            last_dim = orig_shape[-1]
                            blocks_in_last_dim = (last_dim + QK_K - 1) // QK_K
                            bytes_in_last_dim = blocks_in_last_dim * 144  # 144 bytes per Q4_K super-block

                            if len(orig_shape) == 1:
                                quant_shape = (bytes_in_last_dim,)
                            else:
                                quant_shape = tuple(orig_shape[:-1]) + (bytes_in_last_dim,)

                        quantized_data = quantized_data.reshape(quant_shape)
                        ggml_type, _ = self.SUPPORTED_TYPES[actual_type]
                        quant_type = ggml_type

                    elif actual_type == "Q5_K":
                        quantized_bytes = quantize_q5_k(tensor_data, scalar_optimization=scalar_optimization)
                        quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8)

                        orig_shape = list(tensor_data.shape)
                        if len(orig_shape) == 0:
                            quant_shape = quantized_data.shape
                        else:
                            last_dim = orig_shape[-1]
                            blocks_in_last_dim = (last_dim + QK_K - 1) // QK_K
                            bytes_in_last_dim = blocks_in_last_dim * 176  # 176 bytes per Q5_K super-block

                            if len(orig_shape) == 1:
                                quant_shape = (bytes_in_last_dim,)
                            else:
                                quant_shape = tuple(orig_shape[:-1]) + (bytes_in_last_dim,)

                        quantized_data = quantized_data.reshape(quant_shape)
                        ggml_type, _ = self.SUPPORTED_TYPES[actual_type]
                        quant_type = ggml_type

                    elif actual_type == "Q6_K":
                        quantized_bytes = quantize_q6_k(tensor_data, scalar_optimization=scalar_optimization)
                        quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8)

                        orig_shape = list(tensor_data.shape)
                        if len(orig_shape) == 0:
                            quant_shape = quantized_data.shape
                        else:
                            last_dim = orig_shape[-1]
                            blocks_in_last_dim = (last_dim + QK_K - 1) // QK_K
                            bytes_in_last_dim = blocks_in_last_dim * 210  # 210 bytes per Q6_K super-block

                            if len(orig_shape) == 1:
                                quant_shape = (bytes_in_last_dim,)
                            else:
                                quant_shape = tuple(orig_shape[:-1]) + (bytes_in_last_dim,)

                        quantized_data = quantized_data.reshape(quant_shape)
                        ggml_type, _ = self.SUPPORTED_TYPES[actual_type]
                        quant_type = ggml_type

                    elif actual_type == "Q8_0":
                        quantized_bytes = quantize_q8_0(tensor_data)
                        quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8)

                        # Calculate quantized shape for Q8_0
                        # For Q8_0: 32 elements become 34 bytes (2 scale + 32 data)
                        # Use tensor.data.shape (numpy layout), not tensor.shape (GGUF metadata)
                        orig_shape = list(tensor_data.shape)

                        if len(orig_shape) == 0:
                            quant_shape = quantized_data.shape
                        else:
                            last_dim = orig_shape[-1]
                            blocks_in_last_dim = (last_dim + QK8_0 - 1) // QK8_0
                            bytes_in_last_dim = blocks_in_last_dim * 34  # 34 bytes per Q8_0 block

                            if len(orig_shape) == 1:
                                quant_shape = (bytes_in_last_dim,)
                            else:
                                quant_shape = tuple(orig_shape[:-1]) + (bytes_in_last_dim,)

                        quantized_data = quantized_data.reshape(quant_shape)
                        ggml_type, _ = self.SUPPORTED_TYPES[actual_type]
                        quant_type = ggml_type

                    else:
                        raise ValueError(f"Unsupported type: {quantization_type}")

                    # Debug: log shapes after quantization
                    if verbose:
                        print(f" -> quant_data.shape={quantized_data.shape}", end='', flush=True)

                    # Add quantized tensor
                    # Don't pass raw_shape - let writer infer logical shape from byte array
                    # and apply GGUF's column-major transpose convention automatically
                    writer.add_tensor(
                        name=tensor.name,
                        tensor=quantized_data,
                        raw_dtype=quant_type
                    )

                    # Always print status (matching parallel behavior)
                    orig_size = tensor_data.nbytes
                    quant_size = len(quantized_data)
                    ratio = orig_size / quant_size
                    print(f"quantized to {actual_type} ({ratio:.2f}x)", flush=True)
                else:
                    # Keep original (for special tensors like embeddings, output weights, etc.)
                    # Don't pass raw_shape for non-quantized - let writer handle shape convention
                    writer.add_tensor(
                        name=tensor.name,
                        tensor=tensor_data,
                        raw_dtype=tensor.tensor_type
                    )

                    # Always print status (matching parallel behavior)
                    print("kept original", flush=True)

            except Exception as e:
                print(f"\nError processing tensor {tensor.name}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                raise

    def _quantize_tensors_parallel(
        self,
        reader,
        writer,
        quantization_type: str,
        verbose: bool,
        num_workers: Optional[int],
        scalar_optimization: bool = False,
        imatrix=None
    ):
        """Process tensors in parallel using multiprocessing"""
        import multiprocessing as mp
        from functools import partial

        # Determine number of workers (CPU cores - 1, leaving one core free)
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)

        if verbose:
            print(f"Using {num_workers} worker processes", flush=True)

        total_tensors = len(reader.tensors)

        # Prepare tensor arguments for each worker, including the specific importance array.
        # This avoids sending the entire imatrix data with each task.
        tensor_args = []
        imatrix_tensors_found = 0
        for idx, tensor in enumerate(reader.tensors):
            importance_array = None
            if imatrix is not None:
                importance_array = imatrix.get_tensor_importance(tensor.name, tensor.shape)
                if importance_array is not None:
                    imatrix_tensors_found += 1
            
            tensor_args.append({
                'idx': idx,
                'name': tensor.name,
                'data': tensor.data.copy(),  # Copy for pickling
                'shape': tensor.shape,
                'tensor_type': tensor.tensor_type,
                'ndim': tensor.data.ndim,
                'importance': importance_array,
            })
        
        if verbose and imatrix_tensors_found > 0:
            print(f"Using imatrix data for {imatrix_tensors_found} / {len(reader.tensors)} tensors", flush=True)

        # Create worker function with pre-scan data for variants
        worker_func = partial(
            _quantize_tensor_worker,
            quantization_type=quantization_type,
            supported_types=self.SUPPORTED_TYPES,
            tensor_layer_map=self.tensor_layer_map,
            tensor_indices=self.tensor_indices,
            n_layers=self.n_layers,
            n_attention_wv=self.n_attention_wv,
            n_gqa=self.n_gqa,
            has_imatrix=self.has_imatrix,
            scalar_optimization=scalar_optimization,
        )

        # Process in parallel with real-time progress
        with mp.Pool(processes=num_workers) as pool:
            # Use imap to get results as they complete (not imap_unordered to preserve order)
            results_iter = pool.imap(worker_func, tensor_args)

            # Process results as they come in
            for idx, result in enumerate(results_iter):
                # Print progress immediately
                status_msg = f"[{idx+1}/{total_tensors}] {result['name']}: {result['status']}"
                print(status_msg, flush=True)  # flush=True ensures immediate display

                # Check for errors
                if 'error' in result['status']:
                    print(f"  WARNING: {result['status']}", flush=True)

                # Add tensor to writer
                if result['quantized_data'] is not None:
                    # Quantized tensor
                    writer.add_tensor(
                        name=result['name'],
                        tensor=result['quantized_data'],
                        raw_dtype=result['dtype']
                    )
                else:
                    # Kept original
                    writer.add_tensor(
                        name=result['name'],
                        tensor=result['original_data'],
                        raw_dtype=result['dtype']
                    )

    def _should_quantize_tensor(self, tensor) -> bool:
        """
        Determine if a tensor should be quantized

        Some tensors are better left unquantized:
        - 1D tensors (norm layers, biases) - kept at F32/F16 like llama.cpp
        - Token embeddings
        - Output weights
        """
        name = tensor.name.lower()

        # Don't quantize if already quantized (only quantize F32 and F16)
        try:
            is_f32 = (tensor.tensor_type == gguf.GGMLQuantizationType.F32)
            is_f16 = (tensor.tensor_type == gguf.GGMLQuantizationType.F16)
            if not (is_f32 or is_f16):
                return False
        except Exception as e:
            # If comparison fails, assume we can't quantize
            print(f"Warning: Could not check tensor type for {tensor.name}: {e}", flush=True)
            return False

        # Don't quantize 1D tensors (norm layers, biases, etc.)
        # Following llama.cpp: 1D tensors are tiny (~16KB) so keep them at full precision
        if tensor.data.ndim == 1:
            return False

        # Don't quantize token embeddings (often helps preserve quality)
        if 'token_embd' in name or 'embed_tokens' in name:
            return False

        # Don't quantize final model output layer (helps preserve output quality)
        # But DO quantize attention outputs (attn_output.weight)
        if 'output.weight' in name and 'attn' not in name:
            return False

        # Quantize everything else (2D+ weight matrices)
        return True

    def _select_quantization_type_for_tensor(
        self,
        tensor_name: str,
        requested_type: str
    ) -> str:
        """
        Select actual quantization type for a tensor based on variant strategy.
        Wrapper around module-level _select_variant_type function.

        Args:
            tensor_name: Name of the tensor
            requested_type: The requested quantization type (may be a variant)

        Returns:
            Actual quantization type to use (always a base type, never a variant)
        """
        return _select_variant_type(
            tensor_name, requested_type,
            self.tensor_layer_map, self.tensor_indices,
            self.n_layers, self.n_attention_wv,
            self.n_gqa, self.has_imatrix
        )

    def _extract_layer_info(self, tensor_name: str) -> tuple[Optional[int], Optional[int]]:
        """
        Extract layer index from tensor name.

        Args:
            tensor_name: Full tensor name (e.g., "blk.15.attn_v.weight")

        Returns:
            Tuple of (layer_idx, None) - we can't know total layers from just the name
        """
        import re

        # Try to match layer number in patterns like "blk.15." or "layer.15."
        match = re.search(r'(?:blk|layer|h)\.(\d+)\.', tensor_name.lower())
        if match:
            return (int(match.group(1)), None)

        return (None, None)


def _select_variant_type(tensor_name, requested_type, tensor_layer_map, tensor_indices, n_layers, n_attention_wv, n_gqa=1, has_imatrix=False):
    """
    Stateless variant selection logic (for parallel workers).
    Matches the instance method _select_quantization_type_for_tensor.

    Args:
        tensor_name: Name of the tensor
        requested_type: Requested quantization type (may be a variant like IQ3_XXS)
        tensor_layer_map: Maps tensor name to layer index
        tensor_indices: Maps tensor name to its index
        n_layers: Total number of layers
        n_attention_wv: Total number of attention_wv tensors
        n_gqa: Grouped query attention ratio (for IQ3_XXS variant selection)
        has_imatrix: Whether importance matrix is loaded (for IQ3_XXS variant selection)
    """
    name_lower = tensor_name.lower()

    # For base types (non-variants), use as-is
    if requested_type in ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ3_S", "IQ4_NL"]:
        return requested_type

    # IQ3_XXS is a mixed quantization strategy - select per-tensor type
    # Based on llama.cpp/src/llama-quant.cpp (LLAMA_FTYPE_MOSTLY_IQ3_XXS)
    if requested_type == "IQ3_XXS":
        # attn_v.weight - depends on n_gqa and imatrix
        if "attn_v.weight" in name_lower or "attention.wv" in name_lower:
            if n_gqa >= 4:
                return "Q4_K"  # High precision for GQA models
            elif not has_imatrix:
                return "IQ3_S"  # IQ3_S for no imatrix (dequant only - quant not yet implemented)
            else:
                return "IQ3_XXS"  # Use IQ3_XXS when imatrix available

        # attn_k.weight and attn_q.weight - always use low-bit
        elif "attn_k.weight" in name_lower or "attn_q.weight" in name_lower:
            # TODO: Should use IQ2_S (2.56 bpw), not yet implemented. Using Q3_K as substitute.
            return "Q3_K"  # Temporary: use Q3_K (3.5 bpw) instead of IQ2_S (2.56 bpw)

        # attn_output.weight - use higher precision
        elif "attn_output.weight" in name_lower:
            return "IQ3_S"  # IQ3_S (dequant only - quant not yet implemented)

        # ffn_down.weight - layer-dependent (only if no imatrix)
        elif "ffn_down" in name_lower:
            if not has_imatrix:
                layer_idx = tensor_layer_map.get(tensor_name)
                if layer_idx is not None and n_layers > 0:
                    # First 1/8 of layers use higher precision
                    if layer_idx < n_layers // 8:
                        return "Q4_K"
                    else:
                        return "Q3_K"
            # Default to IQ3_XXS if imatrix available
            return "IQ3_XXS"

        # All other tensors use IQ3_XXS
        else:
            return "IQ3_XXS"

    # Get pre-assigned data
    if tensor_layer_map is None:
        tensor_layer_map = {}
    if tensor_indices is None:
        tensor_indices = {}

    layer_idx = tensor_layer_map.get(tensor_name)
    tensor_info = tensor_indices.get(tensor_name, {})

    # use_more_bits logic (copied from instance method)
    def use_more_bits(i_layer, n_layers_total):
        if n_layers_total == 0:
            return False
        return (i_layer < n_layers_total // 8 or
                i_layer >= 7 * n_layers_total // 8 or
                (i_layer - n_layers_total // 8) % 3 == 2)

    # attn_v.weight
    if "attn_v.weight" in name_lower or "attention.wv" in name_lower:
        attn_wv_idx = tensor_info.get('attn_wv_idx', 0)
        if requested_type == "Q4_K_M" or requested_type == "Q5_K_M":
            if use_more_bits(attn_wv_idx, n_attention_wv):
                return "Q6_K"
        elif requested_type == "Q4_K_S":
            if attn_wv_idx < 4:
                return "Q5_K"

    # ffn_down
    if "ffn_down" in name_lower:
        if requested_type == "Q4_K_S" and layer_idx is not None:
            if layer_idx < n_layers // 8:
                return "Q5_K"

    # attn_output.weight
    if "attn_output.weight" in name_lower:
        if requested_type == "Q3_K_M":
            return "Q4_K"
        elif requested_type == "Q3_K_L":
            return "Q5_K"

    # attn_qkv.weight
    if "attn_qkv.weight" in name_lower:
        if requested_type == "Q3_K_M" or requested_type == "Q3_K_L":
            return "Q4_K"
        elif requested_type == "Q4_K_M":
            return "Q5_K"
        elif requested_type == "Q5_K_M":
            return "Q6_K"

    # ffn_gate
    if "ffn_gate" in name_lower:
        if requested_type == "Q3_K_L":
            return "Q4_K"

    # Default: use base type
    if requested_type == "Q4_K_M" or requested_type == "Q4_K_S":
        return "Q4_K"
    elif requested_type == "Q5_K_M" or requested_type == "Q5_K_S":
        return "Q5_K"
    elif requested_type == "Q3_K_S" or requested_type == "Q3_K_M" or requested_type == "Q3_K_L":
        return "Q3_K"

    # Fallback
    base_type = requested_type.rsplit('_', 1)[0]
    return base_type


def _quantize_tensor_worker(tensor_args, quantization_type, supported_types,
                           tensor_layer_map=None, tensor_indices=None,
                           n_layers=0, n_attention_wv=0, n_gqa=1, has_imatrix=False,
                           scalar_optimization=False):
    """
    Worker function for parallel tensor quantization.
    Must be at module level for pickle serialization.

    Args:
        tensor_args: Dict with tensor data, including optional 'importance' array.
        quantization_type: Requested quantization type
        supported_types: Mapping of type names to GGML types
        tensor_layer_map: Pre-scan layer mapping (for variants)
        tensor_indices: Pre-scan tensor indices (for variants)
        n_layers: Total layer count (for variants)
        n_attention_wv: Total attn_v count (for variants)
        n_gqa: Grouped query attention ratio (for IQ3_XXS variant selection)
        has_imatrix: Whether importance matrix is loaded (for IQ3_XXS variant selection)
        scalar_optimization: Enable iterative optimization for better quality
    """
    try:
        tensor_data = tensor_args['data']
        tensor_name = tensor_args['name']
        tensor_type = tensor_args['tensor_type']
        tensor_ndim = tensor_args['ndim']

        # Check if should quantize (same logic as _should_quantize_tensor)
        should_quantize = True

        # Don't quantize if already quantized
        try:
            is_f32 = (tensor_type == gguf.GGMLQuantizationType.F32)
            is_f16 = (tensor_type == gguf.GGMLQuantizationType.F16)
            if not (is_f32 or is_f16):
                should_quantize = False
        except:
            should_quantize = False

        # Don't quantize 1D tensors
        if tensor_ndim == 1:
            should_quantize = False

        # Don't quantize embeddings or final model output
        name_lower = tensor_name.lower()
        if 'token_embd' in name_lower or 'embed_tokens' in name_lower:
            should_quantize = False
        # Don't quantize final model output, but DO quantize attention outputs
        if 'output.weight' in name_lower and 'attn' not in name_lower:
            should_quantize = False

        if should_quantize:
            # Select actual quantization type based on variant strategy
            actual_type = _select_variant_type(
                tensor_name, quantization_type, tensor_layer_map, tensor_indices,
                n_layers, n_attention_wv, n_gqa, has_imatrix
            )

            # Quantize the tensor using selected type
            if actual_type == "Q4_0":
                quantized_bytes = quantize_q4_0(tensor_data)
                quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8)

                orig_shape = list(tensor_data.shape)
                if len(orig_shape) > 0:
                    last_dim = orig_shape[-1]
                    blocks_in_last_dim = (last_dim + QK4_0 - 1) // QK4_0
                    bytes_in_last_dim = blocks_in_last_dim * 18

                    if len(orig_shape) == 1:
                        quant_shape = (bytes_in_last_dim,)
                    else:
                        quant_shape = tuple(orig_shape[:-1]) + (bytes_in_last_dim,)

                    quantized_data = quantized_data.reshape(quant_shape)

                ggml_type, _ = supported_types[actual_type]

            elif actual_type == "Q4_1":
                quantized_bytes = quantize_q4_1(tensor_data)
                quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8)

                orig_shape = list(tensor_data.shape)
                if len(orig_shape) > 0:
                    last_dim = orig_shape[-1]
                    blocks_in_last_dim = (last_dim + QK4_1 - 1) // QK4_1
                    bytes_in_last_dim = blocks_in_last_dim * 20

                    if len(orig_shape) == 1:
                        quant_shape = (bytes_in_last_dim,)
                    else:
                        quant_shape = tuple(orig_shape[:-1]) + (bytes_in_last_dim,)

                    quantized_data = quantized_data.reshape(quant_shape)

                ggml_type, _ = supported_types[actual_type]

            elif actual_type == "Q5_0":
                quantized_bytes = quantize_q5_0(tensor_data)
                quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8)

                orig_shape = list(tensor_data.shape)
                if len(orig_shape) > 0:
                    last_dim = orig_shape[-1]
                    blocks_in_last_dim = (last_dim + QK5_0 - 1) // QK5_0
                    bytes_in_last_dim = blocks_in_last_dim * 22

                    if len(orig_shape) == 1:
                        quant_shape = (bytes_in_last_dim,)
                    else:
                        quant_shape = tuple(orig_shape[:-1]) + (bytes_in_last_dim,)

                    quantized_data = quantized_data.reshape(quant_shape)

                ggml_type, _ = supported_types[actual_type]

            elif actual_type == "Q5_1":
                quantized_bytes = quantize_q5_1(tensor_data)
                quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8)

                orig_shape = list(tensor_data.shape)
                if len(orig_shape) > 0:
                    last_dim = orig_shape[-1]
                    blocks_in_last_dim = (last_dim + QK5_1 - 1) // QK5_1
                    bytes_in_last_dim = blocks_in_last_dim * 24

                    if len(orig_shape) == 1:
                        quant_shape = (bytes_in_last_dim,)
                    else:
                        quant_shape = tuple(orig_shape[:-1]) + (bytes_in_last_dim,)

                    quantized_data = quantized_data.reshape(quant_shape)

                ggml_type, _ = supported_types[actual_type]

            elif actual_type == "IQ3_XXS":
                # Importance data is now passed directly in tensor_args
                importance = tensor_args.get('importance')

                quantized_bytes = quantize_iq3_xxs(tensor_data, importance=importance)
                quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8)

                orig_shape = list(tensor_data.shape)
                if len(orig_shape) > 0:
                    last_dim = orig_shape[-1]
                    blocks_in_last_dim = (last_dim + IQ3_XXS_QK - 1) // IQ3_XXS_QK
                    bytes_in_last_dim = blocks_in_last_dim * IQ3_XXS_BYTES_PER_BLOCK

                    if len(orig_shape) == 1:
                        quant_shape = (bytes_in_last_dim,)
                    else:
                        quant_shape = tuple(orig_shape[:-1]) + (bytes_in_last_dim,)

                    quantized_data = quantized_data.reshape(quant_shape)

                ggml_type, _ = supported_types[actual_type]

            elif actual_type == "Q2_K":
                quantized_bytes = quantize_q2_k(tensor_data, scalar_optimization=scalar_optimization)
                quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8)

                orig_shape = list(tensor_data.shape)
                if len(orig_shape) > 0:
                    last_dim = orig_shape[-1]
                    blocks_in_last_dim = (last_dim + QK_K - 1) // QK_K
                    bytes_in_last_dim = blocks_in_last_dim * 84

                    if len(orig_shape) == 1:
                        quant_shape = (bytes_in_last_dim,)
                    else:
                        quant_shape = tuple(orig_shape[:-1]) + (bytes_in_last_dim,)

                    quantized_data = quantized_data.reshape(quant_shape)

                ggml_type, _ = supported_types[actual_type]

            elif actual_type == "IQ3_S":
                # IQ3_S quantization now fully implemented!
                from .iq3_s import quantize_iq3_s, QK_IQ3S, BYTES_PER_BLOCK as IQ3_S_BYTES_PER_BLOCK

                importance = tensor_args.get('importance')
                quantized_bytes = quantize_iq3_s(tensor_data, importance=importance)
                quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8)

                orig_shape = list(tensor_data.shape)
                if len(orig_shape) > 0:
                    last_dim = orig_shape[-1]
                    blocks_in_last_dim = (last_dim + QK_IQ3S - 1) // QK_IQ3S
                    bytes_in_last_dim = blocks_in_last_dim * IQ3_S_BYTES_PER_BLOCK

                    if len(orig_shape) == 1:
                        quant_shape = (bytes_in_last_dim,)
                    else:
                        quant_shape = tuple(orig_shape[:-1]) + (bytes_in_last_dim,)

                    quantized_data = quantized_data.reshape(quant_shape)

                ggml_type, _ = supported_types[actual_type]

            elif actual_type == "IQ4_NL":
                importance = tensor_args.get('importance')
                quantized_bytes = quantize_iq4_nl(tensor_data, importance=importance)
                quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8)

                orig_shape = list(tensor_data.shape)
                if len(orig_shape) > 0:
                    last_dim = orig_shape[-1]
                    blocks_in_last_dim = (last_dim + QK4_NL - 1) // QK4_NL
                    bytes_in_last_dim = blocks_in_last_dim * IQ4_NL_BYTES_PER_BLOCK

                    if len(orig_shape) == 1:
                        quant_shape = (bytes_in_last_dim,)
                    else:
                        quant_shape = tuple(orig_shape[:-1]) + (bytes_in_last_dim,)

                    quantized_data = quantized_data.reshape(quant_shape)

                ggml_type, _ = supported_types[actual_type]

            elif actual_type == "Q3_K":
                quantized_bytes = quantize_q3_k(tensor_data, scalar_optimization=scalar_optimization)
                quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8)

                orig_shape = list(tensor_data.shape)
                if len(orig_shape) > 0:
                    last_dim = orig_shape[-1]
                    blocks_in_last_dim = (last_dim + QK_K - 1) // QK_K
                    bytes_in_last_dim = blocks_in_last_dim * 110

                    if len(orig_shape) == 1:
                        quant_shape = (bytes_in_last_dim,)
                    else:
                        quant_shape = tuple(orig_shape[:-1]) + (bytes_in_last_dim,)

                    quantized_data = quantized_data.reshape(quant_shape)

                ggml_type, _ = supported_types[actual_type]

            elif actual_type == "Q4_K":
                quantized_bytes = quantize_q4_k(tensor_data, scalar_optimization=scalar_optimization)
                quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8)

                orig_shape = list(tensor_data.shape)
                if len(orig_shape) > 0:
                    last_dim = orig_shape[-1]
                    blocks_in_last_dim = (last_dim + QK_K - 1) // QK_K
                    bytes_in_last_dim = blocks_in_last_dim * 144

                    if len(orig_shape) == 1:
                        quant_shape = (bytes_in_last_dim,)
                    else:
                        quant_shape = tuple(orig_shape[:-1]) + (bytes_in_last_dim,)

                    quantized_data = quantized_data.reshape(quant_shape)

                ggml_type, _ = supported_types[actual_type]

            elif actual_type == "Q5_K":
                quantized_bytes = quantize_q5_k(tensor_data, scalar_optimization=scalar_optimization)
                quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8)

                orig_shape = list(tensor_data.shape)
                if len(orig_shape) > 0:
                    last_dim = orig_shape[-1]
                    blocks_in_last_dim = (last_dim + QK_K - 1) // QK_K
                    bytes_in_last_dim = blocks_in_last_dim * 176

                    if len(orig_shape) == 1:
                        quant_shape = (bytes_in_last_dim,)
                    else:
                        quant_shape = tuple(orig_shape[:-1]) + (bytes_in_last_dim,)

                    quantized_data = quantized_data.reshape(quant_shape)

                ggml_type, _ = supported_types[actual_type]

            elif actual_type == "Q6_K":
                quantized_bytes = quantize_q6_k(tensor_data, scalar_optimization=scalar_optimization)
                quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8)

                orig_shape = list(tensor_data.shape)
                if len(orig_shape) > 0:
                    last_dim = orig_shape[-1]
                    blocks_in_last_dim = (last_dim + QK_K - 1) // QK_K
                    bytes_in_last_dim = blocks_in_last_dim * 210

                    if len(orig_shape) == 1:
                        quant_shape = (bytes_in_last_dim,)
                    else:
                        quant_shape = tuple(orig_shape[:-1]) + (bytes_in_last_dim,)

                    quantized_data = quantized_data.reshape(quant_shape)

                ggml_type, _ = supported_types[actual_type]

            elif actual_type == "Q8_0":
                quantized_bytes = quantize_q8_0(tensor_data)
                quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8)

                orig_shape = list(tensor_data.shape)
                if len(orig_shape) > 0:
                    last_dim = orig_shape[-1]
                    blocks_in_last_dim = (last_dim + QK8_0 - 1) // QK8_0
                    bytes_in_last_dim = blocks_in_last_dim * 34

                    if len(orig_shape) == 1:
                        quant_shape = (bytes_in_last_dim,)
                    else:
                        quant_shape = tuple(orig_shape[:-1]) + (bytes_in_last_dim,)

                    quantized_data = quantized_data.reshape(quant_shape)

                ggml_type, _ = supported_types[actual_type]
            else:
                raise ValueError(f"Unsupported type: {quantization_type}")

            return {
                'name': tensor_name,
                'quantized_data': quantized_data,
                'original_data': None,
                'dtype': ggml_type,
                'status': f'quantized to {actual_type}'
            }
        else:
            # Keep original
            return {
                'name': tensor_name,
                'quantized_data': None,
                'original_data': tensor_data,
                'dtype': tensor_type,
                'status': 'kept original'
            }

    except Exception as e:
        import traceback
        error_msg = f'error: {e}'
        # Print error immediately so user sees it
        print(f"\nERROR processing {tensor_args['name']}: {e}", flush=True)
        traceback.print_exc()
        return {
            'name': tensor_args['name'],
            'quantized_data': None,
            'original_data': tensor_args['data'],
            'dtype': tensor_args['tensor_type'],
            'status': error_msg
        }


def test_quantizer():
    """Test the quantizer with a small example"""
    import tempfile

    print("Testing PythonQuantizer...")

    # Create a tiny test GGUF file
    with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as tmp_in:
        input_path = Path(tmp_in.name)

    with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as tmp_out:
        output_path = Path(tmp_out.name)

    try:
        # Create test GGUF file
        writer = gguf.GGUFWriter(input_path, arch='test')
        writer.add_string('test.key', 'test_value')

        # Add some test tensors
        test_tensor = np.random.randn(256, 256).astype(np.float32)
        writer.add_tensor('test.weight', test_tensor, raw_dtype=gguf.GGMLQuantizationType.F32)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        print(f"Created test input: {input_path}")

        # Quantize it
        quantizer = PythonQuantizer()
        quantizer.quantize_file(
            input_path=input_path,
            output_path=output_path,
            quantization_type="Q8_0",
            verbose=True
        )

        print("\nTest passed!")

    finally:
        # Cleanup
        if input_path.exists():
            input_path.unlink()
        if output_path.exists():
            output_path.unlink()


if __name__ == "__main__":
    test_quantizer()

# Known Issues

This document tracks known compatibility issues between models and quantization features, along with workarounds and detection methods.

**Quick Reference:**
- [IQ Quantizations Fail on Tied Embeddings](#iq-quantizations-fail-on-models-with-tied-embeddings) - Qwen models, severity: high
- [Framework for Handling Model-Specific Issues](#framework-for-handling-model-specific-issues) - Architecture for future issues

---

## IQ Quantizations Fail on Models with Tied Embeddings

**Status:** Limitation in llama.cpp (as of build 7222, Dec 2024)
**Severity:** High - Affects all IQ quantization types on common models
**Affected Quantization Types:** IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, Q2_K_S

### Summary

IQ quantizations fail with the error "Please do not use IQ1_S, IQ1_M, IQ2_S, IQ2_XXS, IQ2_XS or Q2_K_S quantization without an importance matrix" even when an importance matrix is properly generated and provided.

### Root Cause

IQ quantizations require importance matrix data for the model's output layer. However, many modern models use "tied embeddings" where the input embedding layer (`token_embd.weight`) is reused as the output layer instead of having a separate `output.weight` tensor.

**The problem:**
1. llama-imatrix's `--process-output` flag is hardcoded to collect data for a tensor named `output.weight`
2. Models with tied embeddings don't have an `output.weight` tensor
3. No output layer data gets collected in the importance matrix
4. llama-quantize's IQ validation checks for output layer data and rejects the imatrix as invalid

### Technical Details

**What we found:**
- Successfully generated importance matrix file (GGUF format, 3.7MB)
- Matrix contains 504 tensors (252 layers × 2 for .in_sum2 and .counts)
- Tensors present: `blk.X.attn_k`, `blk.X.attn_output`, `blk.X.attn_q`, `blk.X.attn_v`, `blk.X.ffn_down`, `blk.X.ffn_gate`, `blk.X.ffn_up`
- **Missing:** Any tensor named `output` or `token_embd`

**Validation:**
- llama-imatrix can read the file and show statistics (252 tensors)
- Regular quantizations (Q3_K_M, Q4_K_M, Q5_K_M, etc.) work fine with the imatrix
- Only IQ quantizations fail the validation check

**Code location in llama.cpp:**
- Check: `tools/quantize/quantize.cpp` lines ~618-623
- Condition: `imatrix_data.empty()` for IQ types
- The `--process-output` flag in llama-imatrix targets `output` tensor specifically

### Affected Models

**Confirmed affected:**
- Qwen3 series (Qwen3-4B-Instruct-2507 tested)
- Likely affects: Qwen, Qwen2, Qwen2.5, and other models using tied embeddings

**How to detect:**
Models with tied embeddings have:
- `token_embd.weight` tensor present
- **No** `output.weight` tensor (can verify with llama-quantize or by inspecting model)

### Workarounds

**For users:**
1. **Use Q3_K_M or higher** - These work perfectly with importance matrices and provide good quality
2. **Skip IQ quantizations** for models with tied embeddings
3. **Alternative low-bit quants:** Q2_K (not Q2_K_S), Q3_K_S, Q3_K_M provide similar file sizes

**Quality comparison:**
- IQ1_M: ~1.75 bpw (bits per weight)
- IQ2_XXS: ~2.06 bpw
- Q2_K: ~2.96 bpw (works with tied embeddings!)
- Q3_K_S: ~3.41 bpw (works with tied embeddings!)

### Potential Solutions

**Short term (for this tool):**
1. [x] **IMPLEMENTED:** Centralized incompatibility registry system
2. [x] **IMPLEMENTED:** Automatically detect and disable incompatible quantization options
3. [x] **IMPLEMENTED:** Show warning messages with clear explanations
4. [x] **IMPLEMENTED:** Suggest alternative quantization types
5. [x] **IMPLEMENTED:** GUI integration with disabled checkboxes and tooltips

**Detection system (FULLY IMPLEMENTED):**
The converter now includes a centralized incompatibility registry in `MODEL_INCOMPATIBILITIES`:

```python
# Usage example:
from gguf_converter.converter import GGUFConverter
converter = GGUFConverter()

# Simple detection
incompatible = converter.get_incompatible_quantizations(model_path)
# Returns: ["IQ1_S", "IQ1_M", "IQ2_XXS", ...]

# Detailed information
info = converter.get_incompatibility_info(model_path)
# Returns: {
#     "has_incompatibilities": True,
#     "types": ["tied_embeddings"],
#     "incompatible_quants": ["IQ1_S", "IQ1_M", ...],
#     "alternatives": ["Q3_K_M or Q3_K_S", "Q2_K", ...],
#     "reasons": ["Models with tied embeddings: ..."]
# }
```

**How it works:**
1. **Detection**: Checks `config.json` for `tie_word_embeddings` flag and model family patterns (Qwen, etc.)
2. **Registry**: All incompatibility rules stored in centralized `MODEL_INCOMPATIBILITIES` dictionary
3. **Auto-filtering**: Converter automatically removes incompatible quants with clear warnings
4. **GUI Integration**: Incompatible checkboxes are disabled with explanatory tooltips
5. **Consistent messaging**: Same error messages and alternatives across CLI and GUI

**Long term (requires llama.cpp fix):**
1. Update `--process-output` to detect tied embeddings and collect `token_embd.weight` data instead
2. Update IQ quantization validation to accept imatrix with `token_embd` data for tied embedding models
3. Alternative: Add `--process-embeddings` flag to explicitly target `token_embd.weight`

### References

**Investigation trail:**
- Initial report: IQ1_M quantization failing with "no importance matrix" error
- Confirmed: Importance matrix generated successfully (GGUF format, --process-output enabled)
- Confirmed: Matrix readable by llama-imatrix (252 tensors)
- Confirmed: Q4_K_M works with same imatrix file
- Discovery: Qwen3 has `token_embd.weight` but no `output.weight`
- Confirmation: Imatrix contains no `output` or `token_embd` tensors
- Conclusion: `--process-output` doesn't handle tied embeddings

**llama.cpp source references:**
- Quantize validation: `tools/quantize/quantize.cpp:618-623`
- Imatrix loading: `load_imatrix()` function expects `.in_sum2` and `.counts` tensors
- Process output flag: `--process-output` targets `output` tensor specifically

**Related llama.cpp discussions:**
- Tied embeddings are common in modern models
- This limitation affects multiple model families
- No current workaround in llama.cpp itself

### Date
2024-12-10

### Build Information
- llama.cpp build: 7222 (746f9ee88)
- Compiler: clang version 19.1.5 for x86_64-pc-windows-msvc
- Model tested: Qwen3-4B-Instruct-2507 (F32 format)
- Imatrix settings: 150 chunks, 512 context, --process-output enabled, GGUF format

---

## Framework for Handling Model-Specific Issues

As discovered with the tied embeddings issue, certain models may have architectural differences that cause compatibility problems with specific quantization types or features. This section documents our approach to handling such issues.

### Detection Strategy

**Proactive Detection:**
1. Analyze model architecture before quantization
2. Check for known incompatibilities (tied embeddings, special tensor layouts, etc.)
3. Warn users or automatically disable problematic features

**Detection Methods:**
- **Tensor inspection:** Use llama-quantize COPY mode to list all tensors
- **Metadata analysis:** Parse GGUF metadata for architecture info
- **Architecture patterns:** Identify model families by name/metadata

**Example Implementation:**
```python
# In GGUFConverter class
def get_model_compatibility_issues(self, model_path):
    """
    Scan model for known compatibility issues.
    Returns dict of issues and affected features.
    """
    issues = {}

    # Check for tied embeddings
    if self.has_tied_embeddings(model_path):
        issues['tied_embeddings'] = {
            'affected_features': self.IQ_QUANTIZATION_TYPES,
            'workaround': 'Use Q3_K_M or higher quantization types',
            'severity': 'high'
        }

    # Future checks can be added here:
    # - Check for specific architectures with known issues
    # - Check model size against available memory
    # - Check for unsupported tensor types

    return issues
```

### GUI Integration

**User Experience:**
1. **Pre-flight checks:** Run compatibility checks when model is loaded
2. **Visual indicators:** Gray out/disable incompatible options
3. **Tooltips:** Explain why certain options are disabled
4. **Warnings:** Show banner with issue details and workarounds
5. **Recommendations:** Suggest alternative options

**Example Flow:**
```
User loads Qwen3 model
  ↓
GUI runs has_tied_embeddings()
  ↓
Detects tied embeddings = True
  ↓
GUI actions:
  - Grays out IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, Q2_K_S
  - Shows warning banner: "This model uses tied embeddings. IQ quantizations are not supported."
  - Adds tooltip: "Qwen models use tied embeddings which are incompatible with IQ quants. Use Q3_K_M or higher instead."
```

### Adding New Compatibility Checks

When new model-specific issues are discovered, adding them is now simple and consistent:

#### 1. Add to Registry (converter.py)

Edit `GGUFConverter.MODEL_INCOMPATIBILITIES` and add a new entry:

```python
MODEL_INCOMPATIBILITIES = {
    # ... existing entries ...

    "your_issue_name": {
        "description": "One-line description of the issue",
        "detection": {
            # Option 1: Check a config.json flag
            "config_flag": "some_flag_name",

            # Option 2: Match model family names (substring)
            "model_families": ["ModelName", "Alternative"],

            # Can use both together
        },
        "incompatible_quants": [
            "Q4_0",  # List all incompatible types
            "IQ1_S",
        ],
        "alternatives": [
            "Q4_K_M (best quality/size balance)",
            "Q3_K_S for smaller files",
        ],
        "reason": "Technical explanation of why these fail.",
    },
}
```

**That's it!** The system automatically:
- Detects the issue when models are loaded
- Disables incompatible checkboxes in GUI
- Filters them during conversion
- Shows clear error messages with your alternatives

#### 2. Document in KNOWN_ISSUES.md

Add a new section documenting:
- Issue name and description
- Affected models
- List of incompatible quantizations
- Recommended alternatives
- Technical explanation
- Example error messages

#### 3. Test

Verify with an affected model:
- [x] Detection works correctly
- [x] GUI shows info banner
- [x] Incompatible quants disabled
- [x] Conversion filters them out
- [x] Error messages are clear

**Example - Adding a hypothetical "large_context" issue:**

```python
"large_context": {
    "description": "Models with context > 128K",
    "detection": {
        "config_flag": "max_position_embeddings",  # Check if > 128000
        "model_families": ["LongLlama"],
    },
    "incompatible_quants": ["Q2_K", "Q3_K_S"],
    "alternatives": [
        "Q4_K_M or higher for large context models",
        "F16 to preserve context accuracy",
    ],
    "reason": "Low-bit quantizations lose precision needed for long context.",
},
```

No code changes needed - the registry handles everything!

### Known Model Architectures

**Models with tied embeddings (IQ quants fail):**
- Qwen, Qwen2, Qwen2.5, Qwen3 series
- (Add more as discovered)

**Models confirmed to work with IQ quants:**
- (Add as tested)

### Future Enhancements

**Preventative measures:**
- Disable incompatible options based on model architecture
- Show clear warnings when limitations are detected
- Provide informative tooltips explaining why options are disabled
- Suggest alternatives but let user make the final choice

**Model database:**
- Maintain database of model families and their characteristics
- Quick lookup by model name/architecture
- Community-contributed compatibility reports

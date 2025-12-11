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
1. ✅ **IMPLEMENTED:** Detect models with tied embeddings before quantization
2. Automatically disable/hide IQ quantization options for affected models
3. Show warning message explaining the limitation
4. Suggest alternative quantization types

**Detection method (IMPLEMENTED):**
The converter now includes a `has_tied_embeddings()` method that:
- Runs llama-quantize in COPY mode to list all tensors
- Searches for the exact tensor name "output.weight" (not output_norm or attn_output)
- Returns True if model has tied embeddings (IQ quants will fail)
- Uses regex pattern `\s+output\.weight\s+[-\[]` to match only the specific tensor

```python
# Usage example:
from gguf_converter.converter import GGUFConverter
converter = GGUFConverter()
if converter.has_tied_embeddings(model_path):
    print("This model has tied embeddings - IQ quantizations will fail")
    # Disable/hide IQ quant options in GUI
    # Show warning to user
    # Suggest Q3_K_M or higher instead
```

**Next steps:**
- Integrate detection into GUI to disable IQ quants automatically
- Add warning banner when tied embeddings detected
- Update quantization type selector to hide incompatible options

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

When new model-specific issues are discovered:

1. **Document in KNOWN_ISSUES.md:**
   - Issue description
   - Affected models/architectures
   - Root cause analysis
   - Workarounds

2. **Add detection method:**
   - Add function to GGUFConverter class
   - Make it fast and reliable
   - Handle edge cases gracefully

3. **Update GUI:**
   - Integrate new check into compatibility scan
   - Update UI to reflect limitations
   - Add helpful error messages

4. **Test across models:**
   - Verify detection works on affected models
   - Verify no false positives on unaffected models
   - Document test results

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

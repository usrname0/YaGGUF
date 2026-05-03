"""
Model-specific quirks and workarounds for llama.cpp conversion

This module centralizes handling of known model-specific issues and incompatibilities.
Each quirk should be documented with:
- What model/architecture it affects
- What the issue is
- Why the workaround exists
- llama.cpp version where it was discovered

When llama.cpp fixes these issues upstream, the corresponding workarounds can be removed.
"""

import json
from pathlib import Path
from colorama import Style
from .theme import THEME as theme


class ModelQuirks:
    """
    Handles model-specific detection and compatibility checks
    """

    @staticmethod
    def uses_mistral_format(model_path: Path) -> bool:
        """
        Check if the model uses Mistral's native format (requires --mistral-format flag)

        Mistral's native format differs from HuggingFace format:
        - Uses consolidated.safetensors (vs model.safetensors)
        - Uses params.json (vs config.json)
        - Requires mistral-common library for tokenization

        This applies to newer Mistral models including:
        - Ministral-3
        - Pixtral (multimodal)
        - Potentially Devstral and other recent releases

        Args:
            model_path: Path to the model directory

        Returns:
            True if this model uses Mistral native format
        """
        # Check for telltale Mistral format files
        has_consolidated = any(model_path.glob("consolidated*.safetensors"))
        has_params_json = (model_path / "params.json").exists()

        # If either indicator is present, it's Mistral format
        if has_consolidated or has_params_json:
            return True

        # Fallback: check for Ministral-3 in config (for HF-converted models)
        # These may not have consolidated files but still need --mistral-format
        config_file = model_path / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                architectures = config.get("architectures", [])
                model_type = config.get("model_type", "")
                text_config = config.get("text_config", {})
                text_model_type = text_config.get("model_type", "")

                # Ministral-3 specific check
                return (
                    "Mistral3ForConditionalGeneration" in architectures or
                    model_type == "ministral3" or
                    text_model_type == "ministral3"
                )
            except Exception:
                pass

        return False

    @staticmethod
    def uses_ministral3_architecture(model_path: Path) -> bool:
        """
        Check if the model uses the ministral3 architecture

        The ministral3 architecture is used by several Mistral models including:
        - Ministral-3 (the original)
        - Devstral (code-focused variant)
        - Potentially other future Mistral models

        This is used to provide more specific detection messages.

        Args:
            model_path: Path to the model directory

        Returns:
            True if this model uses the ministral3 architecture
        """
        config_file = model_path / "config.json"
        if not config_file.exists():
            return False

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            architectures = config.get("architectures", [])
            model_type = config.get("model_type", "")
            text_config = config.get("text_config", {})
            text_model_type = text_config.get("model_type", "")

            return (
                "Mistral3ForConditionalGeneration" in architectures or
                model_type == "ministral3" or
                text_model_type == "ministral3"
            )
        except Exception:
            return False

    @staticmethod
    def is_multimodal_model(model_path: Path) -> bool:
        """
        Check if the model is a multimodal model requiring --mmproj flag

        Multimodal models have encoders for processing non-text inputs (images, video, audio)
        alongside text. Both vision and audio projectors use the same --mmproj flag in llama.cpp.
        Examples: Gemma 3, Qwen2-VL, Pixtral, LLaVA, InternVL, Qwen2-Audio, Ultravox, etc.

        Args:
            model_path: Path to the model directory

        Returns:
            True if this is a multimodal model needing an mmproj file
        """
        config_file = model_path / "config.json"
        if not config_file.exists():
            return False

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Vision-specific config keys
            has_vision_config = "vision_config" in config
            has_image_processor = "image_processor_type" in config

            # Audio-specific config keys
            has_audio_config = "audio_config" in config
            has_audio_processor = "audio_processor_type" in config

            # Known multimodal architectures (vision + audio)
            architectures = config.get("architectures", [])
            multimodal_architectures = [
                # Vision
                "LlavaForConditionalGeneration",
                "Qwen2VLForConditionalGeneration",
                "Gemma3ForConditionalGeneration",
                "InternVLChatModel",
                "Pixtral",
                "MoondreamForConditionalGeneration",
                "SmolVLM",
                # Audio
                "Qwen2AudioForConditionalGeneration",
                "UltravoxModel",
                # Combined audio+vision
                "Qwen2_5OmniModel",
                "Qwen3OmniForConditionalGeneration",
            ]

            has_multimodal_arch = any(arch in str(architectures) for arch in multimodal_architectures)

            # model_type keywords covering vision and audio variants
            model_type = config.get("model_type", "")
            is_multimodal_type = any(
                keyword in model_type.lower()
                for keyword in ["vision", "vlm", "pixtral", "audio", "asr", "omni"]
            )

            return (
                has_vision_config or has_image_processor
                or has_audio_config or has_audio_processor
                or has_multimodal_arch or is_multimodal_type
            )
        except Exception:
            return False

    @staticmethod
    def is_vision_model(model_path: Path) -> bool:
        """Deprecated alias for is_multimodal_model."""
        return ModelQuirks.is_multimodal_model(model_path)

    @staticmethod
    def is_projector_only_model(model_path: Path) -> bool:
        """
        Check if the model repo contains only an encoder/projector — no text decoder.

        These models cannot be converted to a text GGUF. The only valid conversion is
        --mmproj, which produces the audio/vision projector file. The text model (e.g.
        Llama 3.2 1B for Ultravox) lives in a separate repo and must be converted
        independently.

        Affected models:
        - Ultravox: audio encoder only; raises NotImplementedError without --mmproj
        """
        config_file = model_path / "config.json"
        if not config_file.exists():
            return False

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            architectures = config.get("architectures", [])
            projector_only_architectures = [
                "UltravoxModel",
            ]
            return any(arch in str(architectures) for arch in projector_only_architectures)
        except Exception:
            return False

    @staticmethod
    def is_sentence_transformers_model(model_path: Path) -> bool:
        """
        Check if the model is a sentence-transformers model with dense modules

        Sentence-transformers models may have additional transformation layers:
        - Pooling layers (01_Pooling)
        - Dense layers (02_Dense, 03_Dense)
        - Normalization layers (04_Normalize)

        These are defined in modules.json and can be included with
        --sentence-transformers-dense-modules flag.

        Args:
            model_path: Path to the model directory

        Returns:
            True if this model has sentence-transformers dense modules
        """
        modules_file = model_path / "modules.json"
        if not modules_file.exists():
            return False

        try:
            with open(modules_file, 'r', encoding='utf-8') as f:
                modules = json.load(f)

            # Check if any Dense layers exist
            has_dense = any(
                mod.get("type") == "sentence_transformers.models.Dense"
                for mod in modules
            )

            return has_dense
        except Exception:
            return False

    @staticmethod
    def get_conversion_flags(model_path: Path, include_mmproj: bool = False) -> list:
        """
        Get additional conversion flags required for specific models

        Args:
            model_path: Path to the model directory
            include_mmproj: If True, include --mmproj flag for multimodal models.
                           For multimodal models, you typically need TWO conversions:
                           1. Without --mmproj to get the text model (for imatrix/quantization)
                           2. With --mmproj to get the projector (vision or audio)

        Returns:
            List of additional command-line flags to pass to convert_hf_to_gguf.py
        """
        flags = []

        # Models using Mistral native format require --mistral-format
        if ModelQuirks.uses_mistral_format(model_path):
            flags.append("--mistral-format")

        # Multimodal models need --mmproj to export the projector (vision or audio)
        # But only when explicitly requested (for the second conversion pass)
        if ModelQuirks.is_multimodal_model(model_path) and include_mmproj:
            flags.append("--mmproj")

        # Sentence-transformers models with dense modules
        if ModelQuirks.is_sentence_transformers_model(model_path):
            flags.append("--sentence-transformers-dense-modules")

        return flags

    @staticmethod
    def print_model_detection(model_path: Path) -> None:
        """
        Print information about detected model-specific quirks

        Args:
            model_path: Path to the model directory
        """
        if ModelQuirks.uses_mistral_format(model_path):
            # Be more specific if we can identify ministral3 architecture
            if ModelQuirks.uses_ministral3_architecture(model_path):
                print(f"{theme['info']}\nDetected ministral3 architecture (Mistral format), using --mistral-format flag{Style.RESET_ALL}")
            else:
                print(f"{theme['info']}Detected Mistral-format model, using --mistral-format flag{Style.RESET_ALL}")

        if ModelQuirks.is_projector_only_model(model_path):
            print(f"{theme['info']}\nDetected projector-only model (no text decoder in this repo){Style.RESET_ALL}")
            print(f"{theme['info']}Only the multimodal projector (mmproj-*.gguf) will be generated.{Style.RESET_ALL}")
            print(f"{theme['info']}You will need a separate text model GGUF to use it with llama-server.{Style.RESET_ALL}")
        elif ModelQuirks.is_multimodal_model(model_path):
            print(f"{theme['info']}\nDetected multimodal model{Style.RESET_ALL}")
            print(f"{theme['info']}Requires two components:{Style.RESET_ALL}")
            print(f"{theme['info']}  1. Text model (for imatrix/quantization){Style.RESET_ALL}")
            print(f"{theme['info']}  2. Multimodal projector (mmproj-*.gguf){Style.RESET_ALL}")

        if ModelQuirks.is_sentence_transformers_model(model_path):
            print(f"{theme['info']}\nDetected sentence-transformers model with dense modules{Style.RESET_ALL}")
            print(f"{theme['info']}Using --sentence-transformers-dense-modules to include transformation layers{Style.RESET_ALL}")

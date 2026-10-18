"""Generation wire profiles shared by configuration readers; no backend imports."""

GENERATION_PROFILES = ("legacy", "qwen_vllm")


def validate_extract_mode(mode, profile):
    if type(mode) is not str or mode not in ("validated", "substring_chars"):
        raise ValueError("extract_mode must be validated or substring_chars")
    if mode == "substring_chars" and profile != "qwen_vllm":
        raise ValueError("substring_chars requires qwen_vllm")
    return mode


def validate_generation_profile(profile):
    if type(profile) is not str or profile not in GENERATION_PROFILES:
        raise ValueError("generation profile must be legacy or qwen_vllm")
    return profile

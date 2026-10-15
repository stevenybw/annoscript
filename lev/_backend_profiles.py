"""Generation wire profiles shared by configuration readers; no backend imports."""

GENERATION_PROFILES = ("legacy", "qwen_vllm")


def validate_generation_profile(profile):
    if type(profile) is not str or profile not in GENERATION_PROFILES:
        raise ValueError("generation profile must be legacy or qwen_vllm")
    return profile

"""Platform utilities: normalize, split, and validate platform names.

Provides zero-LLM-cost tools for handling platform information in param_relations:
- normalize_platform_name: standardize platform name format
- split_platforms: split multi-platform strings into individual platforms
- resolve_target_platforms: resolve platform string to target platform list
"""

from __future__ import annotations

# Mapping of common non-standard platform name variants to standard names
PLATFORM_ALIASES: dict[str, str] = {
    # Missing spaces
    "Atlas推理系列产品": "Atlas 推理系列产品",
    "Atlas训练系列产品": "Atlas 训练系列产品",
    "AtlasA2训练系列产品/AtlasA2推理系列产品": "Atlas A2 训练系列产品/Atlas A2 推理系列产品",
    "AtlasA3训练系列产品/AtlasA3推理系列产品": "Atlas A3 训练系列产品/Atlas A3 推理系列产品",
    # Inconsistent spacing
    "Atlas A2训练系列产品/Atlas A2推理系列产品": "Atlas A2 训练系列产品/Atlas A2 推理系列产品",
    "Atlas A3训练系列产品/Atlas A3推理系列产品": "Atlas A3 训练系列产品/Atlas A3 推理系列产品",
}


def normalize_platform_name(name: str) -> str:
    """Normalize a platform name to standard format.

    Args:
        name: Raw platform name string

    Returns:
        Standardized platform name, or original if no mapping found
    """
    if not name:
        return ""

    name = name.strip()
    if not name:
        return ""

    # Direct alias lookup
    if name in PLATFORM_ALIASES:
        return PLATFORM_ALIASES[name]

    # Fuzzy match: compare without spaces
    name_no_space = name.replace(" ", "")
    for alias, standard in PLATFORM_ALIASES.items():
        if alias.replace(" ", "") == name_no_space:
            return standard

    return name


def split_platforms(platform_str: str) -> list[str]:
    """Split a multi-platform string into individual platform names.

    Handles common Chinese separators: 、，, 或 或者 以及

    Args:
        platform_str: Platform string, possibly containing multiple platforms

    Returns:
        List of individual platform names (normalized).
        Returns [""] for empty input (meaning "all platforms").
    """
    if not platform_str or not platform_str.strip():
        return [""]

    # Split on common Chinese/English separators
    separators = ["、", "，", ",", "或", "或者", "以及"]
    platforms = [platform_str.strip()]

    for sep in separators:
        new_platforms: list[str] = []
        for p in platforms:
            new_platforms.extend(part.strip() for part in p.split(sep) if part.strip())
        platforms = new_platforms

    # Normalize each platform name
    return [normalize_platform_name(p) for p in platforms]


def resolve_target_platforms(
    platform_str: str,
    supported_platforms: list[str],
) -> list[str]:
    """Resolve a platform string to the list of target platforms.

    Args:
        platform_str: Platform string from param_relations.platform field.
            Empty string means "all supported platforms".
        supported_platforms: List of platform names from platform_support
            where is_supported=1.

    Returns:
        List of target platform names that this relation applies to.
    """
    platforms = split_platforms(platform_str)

    # Empty platform means all supported platforms
    if not platforms or platforms == [""]:
        return list(supported_platforms)

    # Filter to only include platforms that are in the supported list
    return [p for p in platforms if p in supported_platforms]

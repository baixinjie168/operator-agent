"""Platform utilities: normalize, split, and validate platform names.

Provides zero-LLM-cost tools for handling platform information in param_relations:
- normalize_platform_name: standardize platform name format
- split_platforms: split multi-platform strings into individual platforms
- resolve_target_platforms: resolve platform string to target platform list
- expand_common_in_constraint: expand "common" key in param_constraint to per-platform
- expand_common_in_relations: expand platform="common" relations to per-platform rows
"""

from __future__ import annotations

import copy

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

    # Empty string or "common" means all supported platforms
    if not platforms or platforms == [""] or platforms == ["common"]:
        return list(supported_platforms)

    # Filter to only include platforms that are in the supported list
    return [p for p in platforms if p in supported_platforms]


def expand_common_in_constraint(
    constraint: dict, supported_platforms: list[str]
) -> dict:
    """Expand the "common" key in a param_constraint dict to per-platform entries.

    If *constraint* contains a ``"common"`` key, its value is deep-copied to
    every platform in *supported_platforms* that does not already have an
    explicit entry.  The ``"common"`` key itself is removed.

    Used for ``parameters.param_constraint`` and the ``inputs``/``outputs``
    sections of the final JSON, where the structure is::

        {platform_name: {description, type, ...}}

    Args:
        constraint: Dict mapping platform names (or "common") to constraint data.
        supported_platforms: Platform names from platform_support where is_supported=1.

    Returns:
        The same dict with "common" expanded (modified in place).
    """
    if not isinstance(constraint, dict) or "common" not in constraint:
        return constraint
    # When supported_platforms is empty, keep "common" as-is to avoid
    # data loss (pop would delete the only key without adding replacements).
    if not supported_platforms:
        return constraint
    common_data = constraint.pop("common")
    for plat in supported_platforms:
        if plat not in constraint:
            constraint[plat] = copy.deepcopy(common_data)
    return constraint


def expand_common_in_relations(
    relations: list[dict], supported_platforms: list[str]
) -> list[dict]:
    """Expand relations with platform="common" to one row per supported platform.

    For each relation whose ``platform`` field equals ``"common"``, a deep
    copy is created for every platform in *supported_platforms* with the
    ``platform`` field set to that platform name.  Relations with other
    platform values are kept as-is.

    Used before saving to the ``param_relations`` table so that each row
    has a concrete platform name.

    Args:
        relations: List of relation dicts with a "platform" field.
        supported_platforms: Platform names from platform_support where is_supported=1.

    Returns:
        New list with "common" relations expanded.
    """
    expanded: list[dict] = []
    for rel in relations:
        if rel.get("platform") == "common":
            for plat in supported_platforms:
                new_rel = copy.deepcopy(rel)
                new_rel["platform"] = plat
                expanded.append(new_rel)
        else:
            expanded.append(rel)
    return expanded

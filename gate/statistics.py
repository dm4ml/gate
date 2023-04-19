import typing


def type_to_statistics(t: str) -> typing.List[str]:
    """Returns the statistics that can be computed for a given type.

    Args:
        t (str): Type (one of "int", "float", "string").

    Returns:
        typing.List[str]: List of statistics that can be computed for the type.

    Raises:
        ValueError: If the type is unknown.
    """

    if t == "int":
        return [
            "coverage",
            "mean",
            "stdev",
            "num_unique_values",
            "occurrence_ratio",
            "p95",
        ]

    if t == "float":
        return [
            "coverage",
            "mean",
            "stdev",
            "p95",
        ]

    if t == "string":
        return ["coverage", "num_unique_values", "occurrence_ratio"]

    raise ValueError(f"Unknown type {t}")

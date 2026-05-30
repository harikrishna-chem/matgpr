from __future__ import annotations

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def make_scaler(name: str = "standard"):
    """Create a scikit-learn feature scaler by name.

    Parameters
    ----------
    name
        ``"standard"`` for zero-mean/unit-variance scaling, ``"minmax"`` for
        range scaling, ``"robust"`` for median/IQR scaling, or ``"none"`` /
        ``"passthrough"`` to leave features unchanged in a pipeline.
    """
    normalized = name.lower()

    if normalized == "standard":
        return StandardScaler()
    if normalized == "minmax":
        return MinMaxScaler()
    if normalized == "robust":
        return RobustScaler()
    if normalized in {"none", "passthrough"}:
        return "passthrough"

    raise ValueError("name must be one of: standard, minmax, robust, none")

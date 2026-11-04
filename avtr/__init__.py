"""AVTR: adaptive multimodal reasoning with visual imagination."""

__all__ = ["AVTR"]


def __getattr__(name):
    if name == "AVTR":
        from .inference import AVTR

        return AVTR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

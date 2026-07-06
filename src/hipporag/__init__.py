__all__ = ["HippoRAG"]


def __getattr__(name):
    if name == "HippoRAG":
        from .HippoRAG import HippoRAG

        return HippoRAG
    raise AttributeError(f"module 'hipporag' has no attribute {name!r}")

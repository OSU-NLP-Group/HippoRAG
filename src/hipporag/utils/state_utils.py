from typing import Dict, Hashable, Set


def remove_sources_from_mapping(mapping: Dict[Hashable, Set[str]], key: Hashable, sources_to_remove: Set[str]) -> bool:
    """Remove sources from a reverse mapping and report whether the key became unreferenced."""
    remaining_sources = mapping[key].difference(sources_to_remove)
    if remaining_sources:
        mapping[key] = remaining_sources
        return False
    del mapping[key]
    return True

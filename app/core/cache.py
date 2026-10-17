"""LRU 问答缓存。"""
from collections import OrderedDict


class LRUCache:
    def __init__(self, max_size: int = 10) -> None:
        self._max_size = max(1, int(max_size))
        self._store: "OrderedDict[tuple, str]" = OrderedDict()

    def get(self, key: tuple):
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: tuple, value: str) -> None:
        if key in self._store:
            del self._store[key]
        self._store[key] = value
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: tuple) -> bool:
        return key in self._store

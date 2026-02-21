_cache = {}

def get(key: str):
    return _cache.get(key)

def set(key: str, value):
    _cache[key] = value
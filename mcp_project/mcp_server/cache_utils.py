from cachetools import cached

class CacheWithStats:
    def __init__(self, cache):
        self._cache = cache
        self.hits = 0
        self.misses = 0

    def __call__(self, func):
        # This is the function that cachetools will call on a cache miss.
        @cached(self._cache, key=lambda *args, **kwargs: (func.__name__,) + args)
        def cached_func(*args, **kwargs):
            return func(*args, **kwargs)

        # This is the outer wrapper that tracks hits and misses.
        def wrapper(*args, **kwargs):
            cache_key = (func.__name__,) + args
            if cache_key in self._cache:
                self.hits += 1
            else:
                self.misses += 1
            # We call the function that is decorated with @cached.
            return cached_func(*args, **kwargs)

        return wrapper

    def __getattr__(self, name):
        # Forward other attribute requests (like .clear(), .currsize) to the underlying cache.
        return getattr(self._cache, name)
import time
import functools

def timing_decorator(active=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if active:
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                print(f"{func.__name__} took {end_time - start_time} seconds to execute.")
                return result
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator
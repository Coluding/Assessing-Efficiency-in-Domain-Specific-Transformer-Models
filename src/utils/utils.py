import time
import functools
import yaml
import threading
import sys


def input_with_timeout(prompt, timeout, logging):
    logging.info(prompt)
    result = [None]

    def set_user_input():
        try:
            result[0] = input()
        except EOFError:
            pass

    input_thread = threading.Thread(target=set_user_input)
    input_thread.daemon = True
    input_thread.start()
    input_thread.join(timeout)

    return result[0]



def timing_decorator(active=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if active:
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                print(f"{func.__name__} took {end_time - start_time} seconds to execute.")
                return result
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def remove_none_from_list_decorator(index_to_check: int = 0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return [x for x in func(*args, **kwargs) if x[index_to_check] is not None]
        return wrapper
    return decorator


def exception_decorator(exception_type=KeyError, message=""):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_type as e:
                raise exception_type(f"{func.__name__} raised {e} with message {message}")
                return None
        return wrapper
    return decorator


class YamlConfigLoader:
    def __init__(self, config_file_path: str) -> None:
        with open(config_file_path, "r") as file:
            self.config = yaml.safe_load(file)

        for key, value in self.config.items():
            setattr(self, key, value)

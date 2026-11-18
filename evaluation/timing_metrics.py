
import time


def measure_execution_time(function, *args, **kwargs):
    """Measure how long a function takes to execute."""
    start_time = time.perf_counter()

    result = function(*args, **kwargs)

    end_time = time.perf_counter()

    elapsed_time = end_time - start_time

    return result, elapsed_time


def calculate_average_time(times):
    """Calculate average execution time."""
    if not times:
        raise ValueError("Execution times cannot be empty.")

    return sum(times) / len(times)

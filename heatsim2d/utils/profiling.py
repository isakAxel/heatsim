"""
    This module will be used to test and profile the results, 
    possibly benchmark scripts will be put here as well
"""
# threadpoolctl
import threadpoolctl as tpc
import psutil
import time
import functools

def timer(metric="perf_counter"):
    """
    Decorator that times the execution of passed function, the timing can be done with different metrics, which are listed below

    metric options:
        "perf_counter" : Precise
        "monotonic" : 
        "process_time" :
    """

    def decorator(func):
        @functools.wraps(func)
        def inner(*args,timeit=True,**kwargs):
            if not timeit:
                return func(*args,**kwargs)
            # Set metric for timetaking
            clock = getattr(time,metric,time.perf_counter)

            start = clock()
            result = func(*args,**kwargs)
            end  =  clock()
            t = end - start
            print(f"-------- Timing Results ----------")
            print(f"[{metric}] - Execution time: {t:.4f} seconds")
            print("------------------------------------")
            return result
        return inner

    if callable(metric):
        func = metric
        metric = "perf_counter"
        return decorator(func)

    return decorator

def cpuprf(func):
    """
    Decorator that profiles the CPU usage of given function
    
    Params:
        percpu(bool):
            For CPU usage to be displayed per CPU and not total usage of the processor unit
    """
    @functools.wraps(func)
    def inner(*args,cpuprf = False,**kwargs):
        if not cpuprf:
            return func(*args,**kwargs)
        process = psutil.Process()

        cpu_start = process.cpu_percent(interval =None)
        sol = func(*args,**kwargs)
        cpu_usage = process.cpu_percent(interval=None)
        print(f"\n--- CPU Profile ({func.__name__}) ---")
        print(f"Using {process.num_threads()} threads")
        print(f"Overall CPU usage: {cpu_usage} % ")

        return sol
    return inner


### Sandbox testing

if __name__ == "__main__":  
    @cpuprf
    def test_solver():
        return sum(i*i for i in range(10000000))

    test_solver(cpuprf=True)
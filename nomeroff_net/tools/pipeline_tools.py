import gevent
import itertools
from gevent import Greenlet


def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def unzip(zipped):
    return list(zip(*zipped))


def process_job(func):
    res = func["function"](*func.get("args", []), **func.get("kwargs", {}))
    return res


def promise_all(function_list):
    """
    [
        {
            "function": func,
            "args": args,
            "kwargs": kwargs
        }
    ]
    :return: List response
    """
    jobs = [Greenlet.spawn(process_job, item) for item in function_list]
    gevent.joinall(jobs)
    res = [job.value for job in jobs]
    return res

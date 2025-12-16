import ray

'''
os.sched_getaffinity(pid)
Return the set of CPUs the process with PID pid (or the current process if zero) is restricted to.
'''
@ray.remote()
def get_affinity(id:int):
    import os
    import numpy as np

    a = np.random.rand(1000,1000)
    b = np.random.rand(1000,1000)
    c = np.dot(a,b)
    print(f"Worker {id} computed dot product with shape {c.sum()}")

    context = ray.get_runtime_context()
    meta = dict(job_id=context.get_job_id(),
                worker_id=context.get_worker_id(),
                node_id=context.get_node_id(),
                task_id=context.get_task_id())
    
    affinity = list(os.sched_getaffinity(0))

    return id, meta, affinity

if __name__ == "__main__":
    ray.init(address='auto',)

    num_workers = 2
    num_jobs = 8
    futures = [get_affinity.remote(i) for i in range(num_jobs)]
    results = ray.get(futures)

    for id, meta, affinity in results:
        print(f"Worker {id} - Meta: {meta}, Affinity: {affinity}")
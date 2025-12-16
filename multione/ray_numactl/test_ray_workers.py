import ray
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

import time
'''
os.sched_getaffinity(pid)
Return the set of CPUs the process with PID pid (or the current process if zero) is restricted to.
'''
@ray.remote
def get_affinity(id:int):
    import os
    import numpy as np

    a = np.random.rand(10000,10000)
    b = np.random.rand(10000,10000)
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

    t0=time.time()
    pg = placement_group([{"CPU": 24} for _ in range(2)], strategy="STRICT_SPREAD")
    num_workers = 2
    num_jobs = 8
    futures = [get_affinity.options(scheduling_strategy=PlacementGroupSchedulingStrategy(pg, placement_group_bundle_index=i % num_workers)).remote(i) for i in range(num_jobs)]
    #futures = [get_affinity.remote(i) for i in range(num_jobs)]
    results = ray.get(futures)

    for id, meta, affinity in results:
        print(f"Worker {id} - Meta: {meta}, Affinity: {affinity}")

    print(f"Total time taken: {time.time() - t0} seconds")
    remove_placement_group(pg)
    ray.shutdown()
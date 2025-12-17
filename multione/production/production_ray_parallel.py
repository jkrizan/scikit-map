#%%
import production
import ray

# def get_number_of_cpu_sockets() -> int:
#     import subprocess
#     cpu_sockets =  int(subprocess.check_output('cat /proc/cpuinfo | grep "physical id" | sort -u | wc -l', shell=True))
#     return cpu_sockets
#%%
@ray.remote
def run_production():
    production.production()

# @ray.remote
# def test():
#     from pprint import pprint
#     ctx = ray.get_runtime_context()
#     pprint(ctx)


def main():
    ray.init(address='auto',)
    ray_nodes = ray.nodes()
    ray_nodes_ids = [node['NodeID'] for node in ray_nodes if node['Alive']]
    num_nodes = len(ray_nodes_ids)
    print(f"Number of Ray nodes: {num_nodes}")

    try:
        futures = [run_production.options(label_selector = {"ray.io/node-id": ray_nodes_ids[i]}).remote() for i in range(num_nodes)]
        results = ray.get(futures)
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()
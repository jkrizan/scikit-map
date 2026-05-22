#!/bin/bash
/root/numactl/numactl  --membind=0 --cpunodebind=0 ray start --head --port=6379  --block &
/root/numactl/numactl  --membind=1 --cpunodebind=1 ray start  --address=127.0.0.1:6379  --block &
#!/bin/bash
numactl -N 0 --membind=0 --cpunodebind=0 ray start --head --port=6379  --block
numactl -N 1 --membind=1 --cpunodebind=1 ray start  --port=6379  --block
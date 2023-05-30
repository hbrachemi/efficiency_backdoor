"""Setup constants, ymmv."""
NORMALIZE = False  # True  # Normalize all datasets # if false then use TransformNet to transform input

PIN_MEMORY = True
NON_BLOCKING = True
BENCHMARK = True
MAX_THREADING = 40
SHARING_STRATEGY = 'file_descriptor'  # file_system or file_descriptor

DISTRIBUTED_BACKEND = 'gloo'  # nccl would be faster, but require gpu-transfers for indexing and stuff

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

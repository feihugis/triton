#!/bin/bash


#### Installation

python -m venv .venv --prompt triton --system-site-packages;
source .venv/bin/activate
pip install ninja cmake wheel

export ENABLE_MMA_V3=1
export ENABLE_TMA=1
export TRITON_USE_ASSERT_ENABLED_LLVM=TRUE
export DEBUG=0
pip install -e '.[tests]'

#################
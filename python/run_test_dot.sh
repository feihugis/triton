#!/bin/bash
export MLIR_ENABLE_DUMP=1
python ../adsbrain/test_dot.py &> debug.log

# To use ncu profiling
First need to change TMP dir because we don't have access rights :
```
export TMPDIR=./tmp
```

Then run the profiler, need to specify the ini file 
 - `-f` to overwrite file

```
ncu -o profile -f --target-processes all ./lbmFlowAroundCylinder ~/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/build/src/flowAroundCylinder.ini --section MemoryWorkloadAnalysis
```

Then to see the profiling use `ncu-ui`

# Misc

--section MemoryWorkloadAnalysis
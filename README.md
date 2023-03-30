## Notes for compiling and running OpenCL code on Intel CPUs

### 1. Install Intel OneApi basekit

### 2. Set up environment by running
```
source /opt/intel/oneapi/setvars.sh
```

This sets up paths and runtime environment for OpenCL execution. For example, the following paths are added. The compiler
```
/opt/intel/oneapi/compiler/latest/linux/bin/dpcpp
```
the includes in 
```
/opt/intel/oneapi/compiler/latest/linux/include/sycl/
```
and libs in 
```
/opt/intel/oneapi/compiler/latest/linux/lib/
``` 
and 
```
/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/
```

### 3. Clone repo and OpenCL-CLHPP headers for OpenCL C++ API, then update CPATH
```
git clone --recurse-submodules https://github.com/hokkanen/opencl.git
export CPATH=$CPATH:$(pwd)/opencl/OpenCL-CLHPP/include
```

### 4. Compile examples with 
```
dpcpp cl_devices.c -lOpenCL 
```
Alternatively, if you have CUDA installed, OpenCL code can easily be compiled with nvcc by
```
nvcc -arch=sm_80 cl_devices.c -lOpenCL 
```
where `-arch=sm_80` is the target architecture (NVIDIA A100 GPU).

### 5. Run examples with
```
./a.out
```

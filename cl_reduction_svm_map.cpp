// We're using OpenCL C++ API here; there is also C API in <CL/cl.h>
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>


// For larger kernels, we can store source in a separate file
static const std::string kernel_source = R"(
  __kernel void reduce(__global int* sum, __local int* local_mem) {
    
    // Get work group and work item information
    int gsize = get_global_size(0); // global work size
    int gid = get_global_id(0); // global work item index
    int lsize = get_local_size(0); // local work size
    int lid = get_local_id(0); // local work item index
    
    // Store reduced item into local memory
    local_mem[lid] = gid; // initialize local memory
    barrier(CLK_LOCAL_MEM_FENCE); // synchronize local memory
    
    // Perform reduction across the local work group
    for (int s = 1; s < lsize; s *= 2) { // loop over local memory with stride doubling each iteration
      if (lid % (2 * s) == 0) {
        local_mem[lid] += local_mem[lid + s];
      }
      barrier(CLK_LOCAL_MEM_FENCE); // synchronize local memory
    }
    
    if (lid == 0) { // only one work item per work group
      atomic_add(sum, local_mem[0]); // add partial sum to global sum atomically
    }
  }
)";


int main(int argc, char* argv[]) {

  // Initialize OpenCL
  cl::Device device = cl::Device::getDefault();
  cl::Context context(device);
  cl::CommandQueue queue(context, device);

  // Print device name
  std::string name;
  device.getInfo(CL_DEVICE_NAME, &name);
  printf("Device: %s\n", name.c_str());

  // Compile OpenCL program for found device
  cl::Program program(context, kernel_source);
  program.build(device);
  cl::Kernel kernel_reduce(program, "reduce");

  {
    unsigned n = 10;

    // Create SVM buffer for sum
    cl::SVMAllocator<int, cl::SVMTraitReadWrite<>> svmAlloc(context);
    int *sum = svmAlloc.allocate(1);
    //int *sum = (int*)clSVMAlloc(context(), CL_MEM_READ_WRITE, sizeof(int), 0);

    // Pass arguments to device kernel
    kernel_reduce.setArg(0, sum); // pass SVM pointer to device
    kernel_reduce.setArg(1, sizeof(int), NULL); // allocate local memory

    // Create mapping for host and initialize sum variable
    queue.enqueueMapSVM(sum, CL_TRUE, CL_MAP_WRITE, sizeof(int));
    *sum = 0;
    queue.enqueueUnmapSVM(sum);

    // Enqueue kernel
    queue.enqueueNDRangeKernel(kernel_reduce, cl::NullRange, cl::NDRange(n), cl::NullRange);

    // Create mapping for host and print result
    queue.enqueueMapSVM(sum, CL_TRUE, CL_MAP_READ, sizeof(int));
    printf("sum = %d\n", *sum);
    queue.enqueueUnmapSVM(sum);

    // Free SVM buffer
    svmAlloc.deallocate(sum, 1);
    //clSVMFree(context(), sum);
  }

  return 0;
}
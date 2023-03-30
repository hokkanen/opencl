// We're using OpenCL C++ API here; there is also C API in <CL/cl.h>
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>

// For larger kernels, we can store source in a separate file
static const std::string kernel_source = R"(
  __kernel void async(__global int *a) {
    int i = get_global_id(0);
    int region = i / get_global_size(0);
    a[i] = region + i;
  }
)";

int main(int argc, char *argv[]) {

  // Initialize OpenCL
  cl::Device device = cl::Device::getDefault();
  cl::Context context(device);
  cl::CommandQueue queue(context, device);

  // Print device name
  std::string name;
  device.getInfo(CL_DEVICE_NAME, &name);
  printf("Device: %s\n", name.c_str());

  // Compile OpenCL program for found device.
  cl::Program program(context, kernel_source);
  program.build(device);
  cl::Kernel kernel_async(program, "async");

  {
    // Set problem dimensions
    unsigned n = 5;
    unsigned nx = 20;
  
    // Create SVM buffer object on host side 
    cl::SVMAllocator<int, cl::SVMTraitWriteOnly<>> svmAlloc(context);
    int *a = svmAlloc.allocate(nx);
    //int *a = (int*)clSVMAlloc(context(), CL_MEM_WRITE_ONLY, nx * sizeof(int), 0);
  
    // Pass arguments to device kernel
    kernel_async.setArg(0, a);
  
    // Launch multiple potentially asynchronous kernels on different parts of the array
    for(unsigned region = 0; region < n; region++) {
      queue.enqueueNDRangeKernel(kernel_async, cl::NDRange(nx / n * region), 
        cl::NDRange(nx / n), cl::NullRange);
    }
  
    // Create mapping for host and print results
    queue.enqueueMapSVM(a, CL_TRUE, CL_MAP_READ, nx * sizeof(int));
    for (unsigned i = 0; i < nx; i++)
      printf("a[%d] = %d\n", i, a[i]);
    queue.enqueueUnmapSVM(a);
  
    // Free SVM buffer
    svmAlloc.deallocate(a, nx);
    //clSVMFree(context(), a);
  }

  return 0;
}
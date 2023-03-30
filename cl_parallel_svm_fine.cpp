// We're using OpenCL C++ API here; there is also C API in <CL/cl.h>
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>

// For larger kernels, we can store source in a separate file
static const std::string kernel_source = R"(
  __kernel void dot(__global const int *a, __global const int *b, __global int *c) {
    int i = get_global_id(0);
    c[i] = a[i] * b[i];
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
  cl::Kernel kernel_dot(program, "dot");

  {
    // Set problem dimensions
    unsigned n = 5;
  
    // Create SVM buffer object on host side 
    cl::SVMAllocator<int, cl::SVMTraitFine<>> svmAlloc(context);
    int *a = svmAlloc.allocate(n);
    int *b = svmAlloc.allocate(n);
    int *c = svmAlloc.allocate(n);
  
    // Pass arguments to device kernel
    kernel_dot.setArg(0, a);
    kernel_dot.setArg(1, b);
    kernel_dot.setArg(2, c);
  
    // Initialize values on host
    for (unsigned i = 0; i < n; i++) {
      a[i] = i;
      b[i] = 1;
    }
  
    // We don't need to apply any offset to thread IDs
    queue.enqueueNDRangeKernel(kernel_dot, cl::NullRange, cl::NDRange(n), cl::NullRange);
  
    // Synchronize with host
    queue.finish();
  
    // Print results
    for (unsigned i = 0; i < n; i++)
      printf("c[%d] = %d\n", i, c[i]);
  
    // Free SVM buffers
    svmAlloc.deallocate(a, n);
    svmAlloc.deallocate(b, n);
    svmAlloc.deallocate(c, n);
  }

  return 0;
}
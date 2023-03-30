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

  // This is needed to avoid bug in SVMAllocator::allocate() with coarse grain buf
  cl::CommandQueue::setDefault(queue);

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
  
    // Create SVM buffer objects on host side 
    cl::SVMAllocator<int, cl::SVMTraitReadOnly<>> svmAllocRead(context);
    int *a = svmAllocRead.allocate(n);
    int *b = svmAllocRead.allocate(n);

    cl::SVMAllocator<int, cl::SVMTraitWriteOnly<>> svmAllocWrite(context);
    int *c = svmAllocWrite.allocate(n);
  
    // Pass arguments to device kernel
    kernel_dot.setArg(0, a);
    kernel_dot.setArg(1, b);
    kernel_dot.setArg(2, c);
  
    // Create mappings for host and initialize values
    queue.enqueueMapSVM(a, CL_TRUE, CL_MAP_WRITE, n * sizeof(int));
    queue.enqueueMapSVM(b, CL_TRUE, CL_MAP_WRITE, n * sizeof(int));
    for (unsigned i = 0; i < n; i++) {
      a[i] = i;
      b[i] = 1;
    }
    queue.enqueueUnmapSVM(a);
    queue.enqueueUnmapSVM(b);
  
    // We don't need to apply any offset to thread IDs
    queue.enqueueNDRangeKernel(kernel_dot, cl::NullRange, cl::NDRange(n), cl::NullRange);
  
    // Create mapping for host and print results
    queue.enqueueMapSVM(c, CL_TRUE, CL_MAP_READ, n * sizeof(int));
    for (unsigned i = 0; i < n; i++)
      printf("c[%d] = %d\n", i, c[i]);
    queue.enqueueUnmapSVM(c);
  
    // Free SVM buffers
    svmAllocRead.deallocate(a, n);
    svmAllocRead.deallocate(b, n);
    svmAllocRead.deallocate(c, n);
  }

  return 0;
}
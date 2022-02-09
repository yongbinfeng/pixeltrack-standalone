#include <cassert>
#include <limits>

#include <cuda_runtime.h>

#include "CUDACore/allocate_device.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/ScopedSetDevice.h"

#include "getCachingDeviceAllocator.h"

namespace {
  //const size_t maxAllocationSize = allocator::intPow(cms::cuda::allocator::binGrowth, cms::cuda::allocator::maxBin);
  const size_t maxAllocationSize = notcub::CachingDeviceAllocator::IntPow(cms::cuda::allocator::binGrowth, cms::cuda::allocator::maxBin);
}

namespace cms::cuda {
  void *allocate_device(int dev, size_t nbytes, cudaStream_t stream) {
    void *ptr = nullptr;
    if constexpr (allocator::useCaching) {
      if (nbytes > maxAllocationSize) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      cudaCheck(allocator::getCachingDeviceAllocator().DeviceAllocate(dev, &ptr, nbytes, stream));
      //ptr = allocator::getCachingDeviceAllocator().allocate(nbytes, stream);
      /*
#if CUDA_VERSION >= 11020
    } else if constexpr (allocator::policy == allocator::Policy::Asynchronous) {
      cudaCheck(cudaMallocAsync(&ptr, nbytes, stream));
#endif
      */
    } else {
      ScopedSetDevice setDeviceForThisScope(dev);
      cudaCheck(cudaMalloc(&ptr, nbytes));
    }
    return ptr;
  }

  void free_device(int device, void *ptr) {
    if constexpr (allocator::useCaching) {
      cudaCheck(allocator::getCachingDeviceAllocator().DeviceFree(device, ptr));
      //allocator::getCachingDeviceAllocator().free(ptr);
      /*
#if CUDA_VERSION >= 11020
    } else if constexpr (allocator::policy == allocator::Policy::Asynchronous) {
      cudaCheck(cudaFreeAsync(ptr, stream));
#endif
      */
    } else {
      ScopedSetDevice setDeviceForThisScope(device);
      cudaCheck(cudaFree(ptr));
    }
  }

}  // namespace cms::cuda

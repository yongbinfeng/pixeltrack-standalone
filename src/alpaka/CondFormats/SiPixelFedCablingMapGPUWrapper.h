#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h
#define RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h

#include <utility>

#include "CondFormats/SiPixelFedCablingMapGPU.h"

#include "AlpakaCore/alpakaCommon.h"
#include "AlpakaCore/ESProduct.h"
#include "AlpakaCore/alpakaMemoryHelper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class SiPixelFedCablingMapGPUWrapper {
  public:
    using CablingMapDeviceBuf = cms::alpakatools::device_buffer<Device, SiPixelFedCablingMapGPU>;
    using CablingMapHostBuf = cms::alpakatools::host_buffer<SiPixelFedCablingMapGPU>;

    explicit SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMapGPU cablingMap, std::vector<unsigned char> modToUnp)
        : modToUnpDefault_(modToUnp.size()),
          cablingMapHost_{cms::alpakatools::make_host_buffer<SiPixelFedCablingMapGPU>()},
          hasQuality_{true} {
      std::memcpy(alpaka::getPtrNative(cablingMapHost_), &cablingMap, sizeof(SiPixelFedCablingMapGPU));
      std::copy(modToUnp.begin(), modToUnp.end(), modToUnpDefault_.begin());
    }
    ~SiPixelFedCablingMapGPUWrapper() = default;

    bool hasQuality() const { return hasQuality_; }

    const SiPixelFedCablingMapGPU* cablingMap() const { return alpaka::getPtrNative(cablingMapHost_); }

    const SiPixelFedCablingMapGPU* getGPUProductAsync(Queue& queue) const {
      const auto& data = gpuData_.dataForDeviceAsync(queue, [this](Queue& queue) {
        // allocate
        GPUData gpuData(queue);
        alpaka::memcpy(queue, gpuData.cablingMapDevice, cablingMapHost_);
        return gpuData;
      });
      return alpaka::getPtrNative(data.cablingMapDevice);
    }

    const unsigned char* getModToUnpAllAsync(Queue& queue) const {
      const auto& data = modToUnp_.dataForDeviceAsync(queue, [this](Queue& queue) {
        unsigned int modToUnpSize = modToUnpDefault_.size();
        ModulesToUnpack modToUnp(queue, modToUnpSize);
        auto modToUnpDefault_view = cms::alpakatools::make_host_view(modToUnpDefault_.data(), modToUnpSize);
        alpaka::memcpy(queue, modToUnp.modToUnpDefault, modToUnpDefault_view);
        return modToUnp;
      });
      return alpaka::getPtrNative(data.modToUnpDefault);
    }

  private:
    std::vector<unsigned char> modToUnpDefault_;
    CablingMapHostBuf cablingMapHost_;
    bool hasQuality_;

    struct GPUData {
    public:
      GPUData() = delete;
      GPUData(Queue const& queue)
          : cablingMapDevice{cms::alpakatools::make_device_buffer<SiPixelFedCablingMapGPU>(queue)} {
        alpaka::prepareForAsyncCopy(cablingMapDevice);
      };
      ~GPUData() = default;

      cms::alpakatools::device_buffer<Device, SiPixelFedCablingMapGPU> cablingMapDevice;  // pointer to struct in GPU
    };
    cms::alpakatools::ESProduct<Queue, GPUData> gpuData_;

    struct ModulesToUnpack {
    public:
      ModulesToUnpack() = delete;
      ModulesToUnpack(Queue const& queue, unsigned int modToUnpSize)
          : modToUnpDefault{cms::alpakatools::make_device_buffer<unsigned char[]>(queue, modToUnpSize)} {};
      ~ModulesToUnpack() = default;

      cms::alpakatools::device_buffer<Device, unsigned char[]> modToUnpDefault;  // pointer to GPU
    };

    cms::alpakatools::ESProduct<Queue, ModulesToUnpack> modToUnp_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h

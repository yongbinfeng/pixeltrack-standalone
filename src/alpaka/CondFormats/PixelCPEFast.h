#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
#define RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h

#include <utility>

#include "CondFormats/pixelCPEforGPU.h"

#include "AlpakaCore/alpakaCommon.h"
#include "AlpakaCore/ESProduct.h"
#include "AlpakaCore/alpakaMemoryHelper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PixelCPEFast {
  public:
    PixelCPEFast(std::string const &path)
        : m_commonParamsGPU(cms::alpakatools::make_host_buffer<pixelCPEforGPU::CommonParams>()),
          m_layerGeometry(cms::alpakatools::make_host_buffer<pixelCPEforGPU::LayerGeometry>()),
          m_averageGeometry(cms::alpakatools::make_host_buffer<pixelCPEforGPU::AverageGeometry>())

    {
      std::ifstream in(path, std::ios::binary);
      in.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
      in.read(reinterpret_cast<char *>(alpaka::getPtrNative(m_commonParamsGPU)), sizeof(pixelCPEforGPU::CommonParams));
      unsigned int ndetParams;
      in.read(reinterpret_cast<char *>(&ndetParams), sizeof(unsigned int));
      m_detParamsGPU.resize(ndetParams);
      in.read(reinterpret_cast<char *>(m_detParamsGPU.data()), ndetParams * sizeof(pixelCPEforGPU::DetParams));
      in.read(reinterpret_cast<char *>(alpaka::getPtrNative(m_averageGeometry)),
              sizeof(pixelCPEforGPU::AverageGeometry));
      in.read(reinterpret_cast<char *>(alpaka::getPtrNative(m_layerGeometry)), sizeof(pixelCPEforGPU::LayerGeometry));

      alpaka::prepareForAsyncCopy(m_commonParamsGPU);
      alpaka::prepareForAsyncCopy(m_layerGeometry);
      alpaka::prepareForAsyncCopy(m_averageGeometry);
    }

    ~PixelCPEFast() = default;

    // The return value can only be used safely in kernels launched on
    // the same cudaStream, or after cudaStreamSynchronize.
    const pixelCPEforGPU::ParamsOnGPU *getGPUProductAsync(Queue &queue) const {
      const auto &data = gpuData_.dataForDeviceAsync(queue, [this](Queue &queue) {
        unsigned int ndetParams = m_detParamsGPU.size();
        GPUData gpuData(queue, ndetParams);

        alpaka::memcpy(queue, gpuData.d_commonParams, m_commonParamsGPU);
        alpaka::getPtrNative(gpuData.h_paramsOnGPU)->m_commonParams = alpaka::getPtrNative(gpuData.d_commonParams);

        alpaka::memcpy(queue, gpuData.d_layerGeometry, m_layerGeometry);
        alpaka::getPtrNative(gpuData.h_paramsOnGPU)->m_layerGeometry = alpaka::getPtrNative(gpuData.d_layerGeometry);

        alpaka::memcpy(queue, gpuData.d_averageGeometry, m_averageGeometry);
        alpaka::getPtrNative(gpuData.h_paramsOnGPU)->m_averageGeometry =
            alpaka::getPtrNative(gpuData.d_averageGeometry);

        auto detParams_h = cms::alpakatools::make_host_view(m_detParamsGPU.data(), ndetParams);
        alpaka::memcpy(queue, gpuData.d_detParams, detParams_h);
        alpaka::getPtrNative(gpuData.h_paramsOnGPU)->m_detParams = alpaka::getPtrNative(gpuData.d_detParams);

        alpaka::memcpy(queue, gpuData.d_paramsOnGPU, gpuData.h_paramsOnGPU);

        return gpuData;
      });
      return alpaka::getPtrNative(data.d_paramsOnGPU);
    }

  private:
    // allocate it with posix malloc to be compatible with cpu wf
    std::vector<pixelCPEforGPU::DetParams> m_detParamsGPU;
    cms::alpakatools::host_buffer<pixelCPEforGPU::CommonParams> m_commonParamsGPU;
    cms::alpakatools::host_buffer<pixelCPEforGPU::LayerGeometry> m_layerGeometry;
    cms::alpakatools::host_buffer<pixelCPEforGPU::AverageGeometry> m_averageGeometry;

    struct GPUData {
      // not needed if not used on CPU...
    public:
      GPUData() = delete;
      GPUData(Queue &queue, unsigned int ndetParams)
          : h_paramsOnGPU{cms::alpakatools::make_host_buffer<pixelCPEforGPU::ParamsOnGPU>()},
            d_paramsOnGPU{cms::alpakatools::make_device_buffer<pixelCPEforGPU::ParamsOnGPU>(queue)},
            d_commonParams{cms::alpakatools::make_device_buffer<pixelCPEforGPU::CommonParams>(queue)},
            d_layerGeometry{cms::alpakatools::make_device_buffer<pixelCPEforGPU::LayerGeometry>(queue)},
            d_averageGeometry{cms::alpakatools::make_device_buffer<pixelCPEforGPU::AverageGeometry>(queue)},
            d_detParams{cms::alpakatools::make_device_buffer<pixelCPEforGPU::DetParams[]>(queue, ndetParams)} {
        alpaka::prepareForAsyncCopy(h_paramsOnGPU);
      };
      ~GPUData() = default;

    public:
      cms::alpakatools::host_buffer<pixelCPEforGPU::ParamsOnGPU> h_paramsOnGPU;
      cms::alpakatools::device_buffer<Device, pixelCPEforGPU::ParamsOnGPU>
          d_paramsOnGPU;  // copy of the above on the Device
      cms::alpakatools::device_buffer<Device, pixelCPEforGPU::CommonParams> d_commonParams;
      cms::alpakatools::device_buffer<Device, pixelCPEforGPU::LayerGeometry> d_layerGeometry;
      cms::alpakatools::device_buffer<Device, pixelCPEforGPU::AverageGeometry> d_averageGeometry;
      cms::alpakatools::device_buffer<Device, pixelCPEforGPU::DetParams[]> d_detParams;
    };

    cms::alpakatools::ESProduct<Queue, GPUData> gpuData_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h

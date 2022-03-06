#include <iostream>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "Geometry/pixelTopology.h"
#include "CUDACore/cudaCheck.h"
#include "CondFormats/PixelCPEFast.h"

// Services
// this is needed to get errors from templates

namespace {
  constexpr float micronsToCm = 1.0e-4;
}

//-----------------------------------------------------------------------------
//!  The constructor.
//-----------------------------------------------------------------------------
PixelCPEFast::PixelCPEFast(std::string const &path) {
  {
    std::ifstream in(path, std::ios::binary);
    in.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
    std::cout << "==> Filling CPE " << path.c_str() << std::endl;
    in.read(reinterpret_cast<char *>(&commonParamsGPU_), sizeof(pixelCPEforGPU::CommonParams));
    unsigned int ndetParams;
    in.read(reinterpret_cast<char *>(&ndetParams), sizeof(unsigned int));
    detParamsGPU_.resize(ndetParams);
    in.read(reinterpret_cast<char *>(detParamsGPU_.data()), ndetParams * sizeof(pixelCPEforGPU::DetParams));
    in.read(reinterpret_cast<char *>(&averageGeometry_), sizeof(pixelCPEforGPU::AverageGeometry));
    in.read(reinterpret_cast<char *>(&layerGeometry_), sizeof(pixelCPEforGPU::LayerGeometry));
    std::cout << "layer Geom ---> " << this->layerGeometry_.maxModuleStride << " -- " << this->layerGeometry_.layer[0] << std::endl;
    std::cout << "---> " << sizeof(pixelCPEforGPU::LayerGeometry) << " -- " << sizeof(this->layerGeometry_.layer) << " -- " << sizeof(this->layerGeometry_.layerStart) << std::endl;
    std::cout << "--checking layer 1 " << std::endl;
    /*
    for(unsigned i0 = 0; i0 < 1500; ++i0) { 
      std::cout << i0 << " -checking--> " << int(this->layerGeometry_.layer[i0]) << " -- " << int(phase1PixelTopology::getLayer(i0)) << std::endl;
    }
    */
    std::cout << "--checking layer 2 " << std::endl;
  }

  cpuData_ = {
      &commonParamsGPU_,
      detParamsGPU_.data(),
      &layerGeometry_,
      &averageGeometry_,
  };
}

const pixelCPEforGPU::ParamsOnGPU *PixelCPEFast::getGPUProductAsync(cudaStream_t cudaStream) const {
  const auto &data = gpuData_.dataForCurrentDeviceAsync(cudaStream, [this](GPUData &data, cudaStream_t stream) {
    // and now copy to device...

    cudaCheck(cudaMalloc((void **)&data.paramsOnGPU_h.m_commonParams, sizeof(pixelCPEforGPU::CommonParams)));
    cudaCheck(cudaMalloc((void **)&data.paramsOnGPU_h.m_detParams,
                         this->detParamsGPU_.size() * sizeof(pixelCPEforGPU::DetParams)));
    cudaCheck(cudaMalloc((void **)&data.paramsOnGPU_h.m_averageGeometry, sizeof(pixelCPEforGPU::AverageGeometry)));
    cudaCheck(cudaMalloc((void **)&data.paramsOnGPU_h.m_layerGeometry, sizeof(pixelCPEforGPU::LayerGeometry)));
    cudaCheck(cudaMalloc((void **)&data.paramsOnGPU_d, sizeof(pixelCPEforGPU::ParamsOnGPU)));
    cudaCheck(cudaMemcpyAsync(
        data.paramsOnGPU_d, &data.paramsOnGPU_h, sizeof(pixelCPEforGPU::ParamsOnGPU), cudaMemcpyDefault, stream));
    cudaCheck(cudaMemcpyAsync((void *)data.paramsOnGPU_h.m_commonParams,
                              &this->commonParamsGPU_,
                              sizeof(pixelCPEforGPU::CommonParams),
                              cudaMemcpyDefault,
                              stream));
    cudaCheck(cudaMemcpyAsync((void *)data.paramsOnGPU_h.m_averageGeometry,
                              &this->averageGeometry_,
                              sizeof(pixelCPEforGPU::AverageGeometry),
                              cudaMemcpyDefault,
                              stream));
    cudaCheck(cudaMemcpyAsync((void *)data.paramsOnGPU_h.m_layerGeometry,
                              &this->layerGeometry_,
                              sizeof(pixelCPEforGPU::LayerGeometry),
                              cudaMemcpyDefault,
                              stream));
    cudaCheck(cudaMemcpyAsync((void *)data.paramsOnGPU_h.m_detParams,
                              this->detParamsGPU_.data(),
                              this->detParamsGPU_.size() * sizeof(pixelCPEforGPU::DetParams),
                              cudaMemcpyDefault,
                              stream));
  });
  return data.paramsOnGPU_d;
}

PixelCPEFast::GPUData::~GPUData() {
  if (paramsOnGPU_d != nullptr) {
    cudaFree((void *)paramsOnGPU_h.m_commonParams);
    cudaFree((void *)paramsOnGPU_h.m_detParams);
    cudaFree((void *)paramsOnGPU_h.m_averageGeometry);
    cudaFree((void *)paramsOnGPU_h.m_layerGeometry);
    cudaFree(paramsOnGPU_d);
  }
}

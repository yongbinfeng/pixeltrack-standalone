#include "BrokenLineFitOnGPU.h"
#include "CUDACore/device_unique_ptr.h"

//#define NTUPLE_DEBUG

void HelixFitOnGPU::launchBrokenLineKernels(HitsView const *hv,
                                            uint32_t hitsInFit,
                                            uint32_t maxNumberOfTuples,
                                            cudaStream_t stream) {
  assert(tuples_);

#ifdef NTUPLE_DEBUG
  std::cout << "---> BL 0" << std::endl;
#endif

  auto blockSize = 64;
  auto numberOfBlocks = (maxNumberOfConcurrentFits_ + blockSize - 1) / blockSize;

#ifdef NTUPLE_DEBUG
  std::cout << "---> BL 1" << maxNumberOfConcurrentFits_ << std::endl;
#endif

  //  Fit internals
  auto tkidGPU = cms::cuda::make_device_unique<caConstants::tindex_type[]>(maxNumberOfConcurrentFits_, stream);
#ifdef NTUPLE_DEBUG
  std::cout << "---> BL 1.4" << std::endl;
#endif
  auto hitsGPU = cms::cuda::make_device_unique<double[]>(
      maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix3xNd<6>) / sizeof(double), stream);
#ifdef NTUPLE_DEBUG
  std::cout << "---> BL 1.5" << std::endl;
#endif
  auto hits_geGPU = cms::cuda::make_device_unique<float[]>(
      maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix6xNf<6>) / sizeof(float), stream);
#ifdef NTUPLE_DEBUG
  std::cout << "---> BL 1.6" << std::endl;
#endif
  auto fast_fit_resultsGPU = cms::cuda::make_device_unique<double[]>(
      maxNumberOfConcurrentFits_ * sizeof(riemannFit::Vector4d) / sizeof(double), stream);

#ifdef NTUPLE_DEBUG
  std::cout << "---> BL 2" << std::endl;
#endif

  for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
    // fit triplets
    kernel_BLFastFit<3><<<numberOfBlocks, blockSize, 0, stream>>>(tuples_,
                                                                  tupleMultiplicity_,
                                                                  hv,
                                                                  tkidGPU.get(),
                                                                  hitsGPU.get(),
                                                                  hits_geGPU.get(),
                                                                  fast_fit_resultsGPU.get(),
                                                                  3,
                                                                  3,
                                                                  offset);
    cudaCheck(cudaGetLastError());

#ifdef NTUPLE_DEBUG
  std::cout << "---> BL 3" << std::endl;
#endif

    kernel_BLFit<3><<<numberOfBlocks, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                              bField_,
                                                              outputSoa_,
                                                              tkidGPU.get(),
                                                              hitsGPU.get(),
                                                              hits_geGPU.get(),
                                                              fast_fit_resultsGPU.get());
    cudaCheck(cudaGetLastError());

#ifdef NTUPLE_DEBUG
  std::cout << "---> BL 4" << std::endl;
#endif

    if (fitNas4_) {
      // fit all as 4
      kernel_BLFastFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tuples_,
                                                                        tupleMultiplicity_,
                                                                        hv,
                                                                        tkidGPU.get(),
                                                                        hitsGPU.get(),
                                                                        hits_geGPU.get(),
                                                                        fast_fit_resultsGPU.get(),
                                                                        4,
                                                                        8,
                                                                        offset);
      cudaCheck(cudaGetLastError());

#ifdef NTUPLE_DEBUG
  std::cout << "---> BL 5" << std::endl;
#endif
      kernel_BLFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                                    bField_,
                                                                    outputSoa_,
                                                                    tkidGPU.get(),
                                                                    hitsGPU.get(),
                                                                    hits_geGPU.get(),
                                                                    fast_fit_resultsGPU.get());
    } else {

#ifdef NTUPLE_DEBUG
  std::cout << "---> BL 6" << std::endl;
#endif
      // fit quads
      kernel_BLFastFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tuples_,
                                                                        tupleMultiplicity_,
                                                                        hv,
                                                                        tkidGPU.get(),
                                                                        hitsGPU.get(),
                                                                        hits_geGPU.get(),
                                                                        fast_fit_resultsGPU.get(),
                                                                        4,
                                                                        4,
                                                                        offset);
      cudaCheck(cudaGetLastError());

      kernel_BLFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                                    bField_,
                                                                    outputSoa_,
                                                                    tkidGPU.get(),
                                                                    hitsGPU.get(),
                                                                    hits_geGPU.get(),
                                                                    fast_fit_resultsGPU.get());

#ifdef NTUPLE_DEBUG
  std::cout << "---> BL 7" << std::endl;
#endif
      // fit penta (all 5)
      kernel_BLFastFit<5><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tuples_,
                                                                        tupleMultiplicity_,
                                                                        hv,
                                                                        tkidGPU.get(),
                                                                        hitsGPU.get(),
                                                                        hits_geGPU.get(),
                                                                        fast_fit_resultsGPU.get(),
                                                                        5,
                                                                        5,
                                                                        offset);
      cudaCheck(cudaGetLastError());

      kernel_BLFit<5><<<8, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                   bField_,
                                                   outputSoa_,
                                                   tkidGPU.get(),
                                                   hitsGPU.get(),
                                                   hits_geGPU.get(),
                                                   fast_fit_resultsGPU.get());
      cudaCheck(cudaGetLastError());
      // fit sexta and above (as 6)
      kernel_BLFastFit<6><<<4, blockSize, 0, stream>>>(tuples_,
                                                       tupleMultiplicity_,
                                                       hv,
                                                       tkidGPU.get(),
                                                       hitsGPU.get(),
                                                       hits_geGPU.get(),
                                                       fast_fit_resultsGPU.get(),
                                                       6,
                                                       8,
                                                       offset);
      cudaCheck(cudaGetLastError());

#ifdef NTUPLE_DEBUG
  std::cout << "---> BL 8" << std::endl;
#endif
      kernel_BLFit<6><<<4, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                   bField_,
                                                   outputSoa_,
                                                   tkidGPU.get(),
                                                   hitsGPU.get(),
                                                   hits_geGPU.get(),
                                                   fast_fit_resultsGPU.get());
      cudaCheck(cudaGetLastError());
    }
#ifdef NTUPLE_DEBUG
  std::cout << "---> BL 9" << std::endl;
#endif
  }  // loop on concurrent fits
}

#include "RiemannFitOnGPU.h"
#include "CUDACore/device_unique_ptr.h"

//#define NTUPLE_DEBUG

void HelixFitOnGPU::launchRiemannKernels(HitsView const *hv,
                                         uint32_t nhits,
                                         uint32_t maxNumberOfTuples,
                                         cudaStream_t stream) {
  assert(tuples_);

#ifdef NTUPLE_DEBUG
  std::cout << "---> Riemann 0" << std::endl;
#endif

  auto blockSize = 64;
  auto numberOfBlocks = (maxNumberOfConcurrentFits_ + blockSize - 1) / blockSize;

#ifdef NTUPLE_DEBUG
  std::cout << "---> Riemann 1" << std::endl;
#endif

  //  Fit internals
  auto hitsGPU = cms::cuda::make_device_unique<double[]>(
      maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix3xNd<4>) / sizeof(double), stream);

#ifdef NTUPLE_DEBUG
  std::cout << "---> Riemann 1.2" << std::endl;
#endif

  auto hits_geGPU = cms::cuda::make_device_unique<float[]>(
      maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix6x4f) / sizeof(float), stream);

#ifdef NTUPLE_DEBUG
  std::cout << "---> Riemann 1.3" << std::endl;
#endif

  auto fast_fit_resultsGPU = cms::cuda::make_device_unique<double[]>(
      maxNumberOfConcurrentFits_ * sizeof(riemannFit::Vector4d) / sizeof(double), stream);
#ifdef NTUPLE_DEBUG
  std::cout << "---> Riemann 1.4" << std::endl;
#endif  
auto circle_fit_resultsGPU_holder =
      cms::cuda::make_device_unique<char[]>(maxNumberOfConcurrentFits_ * sizeof(riemannFit::CircleFit), stream);
  riemannFit::CircleFit *circle_fit_resultsGPU_ = (riemannFit::CircleFit *)(circle_fit_resultsGPU_holder.get());

#ifdef NTUPLE_DEBUG
  std::cout << "---> Riemann 2" << std::endl;
#endif

  for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
    // triplets
    kernel_FastFit<3><<<numberOfBlocks, blockSize, 0, stream>>>(
        tuples_, tupleMultiplicity_, 3, hv, hitsGPU.get(), hits_geGPU.get(), fast_fit_resultsGPU.get(), offset);
    cudaCheck(cudaGetLastError());

#ifdef NTUPLE_DEBUG
  std::cout << "---> Riemann 3" << std::endl;
#endif

    kernel_CircleFit<3><<<numberOfBlocks, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                                  3,
                                                                  bField_,
                                                                  hitsGPU.get(),
                                                                  hits_geGPU.get(),
                                                                  fast_fit_resultsGPU.get(),
                                                                  circle_fit_resultsGPU_,
                                                                  offset);
    cudaCheck(cudaGetLastError());

    kernel_LineFit<3><<<numberOfBlocks, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                                3,
                                                                bField_,
                                                                outputSoa_,
                                                                hitsGPU.get(),
                                                                hits_geGPU.get(),
                                                                fast_fit_resultsGPU.get(),
                                                                circle_fit_resultsGPU_,
                                                                offset);
    cudaCheck(cudaGetLastError());

#ifdef NTUPLE_DEBUG
  std::cout << "---> Riemann 4" << std::endl;
#endif

    // quads
    kernel_FastFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(
        tuples_, tupleMultiplicity_, 4, hv, hitsGPU.get(), hits_geGPU.get(), fast_fit_resultsGPU.get(), offset);
    cudaCheck(cudaGetLastError());

    kernel_CircleFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                                      4,
                                                                      bField_,
                                                                      hitsGPU.get(),
                                                                      hits_geGPU.get(),
                                                                      fast_fit_resultsGPU.get(),
                                                                      circle_fit_resultsGPU_,
                                                                      offset);
    cudaCheck(cudaGetLastError());

    kernel_LineFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                                    4,
                                                                    bField_,
                                                                    outputSoa_,
                                                                    hitsGPU.get(),
                                                                    hits_geGPU.get(),
                                                                    fast_fit_resultsGPU.get(),
                                                                    circle_fit_resultsGPU_,
                                                                    offset);
    cudaCheck(cudaGetLastError());

    if (fitNas4_) {

#ifdef NTUPLE_DEBUG
  std::cout << "---> Riemann 5" << std::endl;
#endif

      // penta
      kernel_FastFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(
          tuples_, tupleMultiplicity_, 5, hv, hitsGPU.get(), hits_geGPU.get(), fast_fit_resultsGPU.get(), offset);
      cudaCheck(cudaGetLastError());

      kernel_CircleFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                                        5,
                                                                        bField_,
                                                                        hitsGPU.get(),
                                                                        hits_geGPU.get(),
                                                                        fast_fit_resultsGPU.get(),
                                                                        circle_fit_resultsGPU_,
                                                                        offset);
      cudaCheck(cudaGetLastError());

      kernel_LineFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                                      5,
                                                                      bField_,
                                                                      outputSoa_,
                                                                      hitsGPU.get(),
                                                                      hits_geGPU.get(),
                                                                      fast_fit_resultsGPU.get(),
                                                                      circle_fit_resultsGPU_,
                                                                      offset);
#ifdef NTUPLE_DEBUG
  std::cout << "---> Riemann 6" << std::endl;
#endif
      cudaCheck(cudaGetLastError());
    } else {

#ifdef NTUPLE_DEBUG
  std::cout << "---> Riemann 7" << std::endl;
#endif
      // penta all 5
      kernel_FastFit<5><<<numberOfBlocks / 4, blockSize, 0, stream>>>(
          tuples_, tupleMultiplicity_, 5, hv, hitsGPU.get(), hits_geGPU.get(), fast_fit_resultsGPU.get(), offset);
      cudaCheck(cudaGetLastError());

      kernel_CircleFit<5><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                                        5,
                                                                        bField_,
                                                                        hitsGPU.get(),
                                                                        hits_geGPU.get(),
                                                                        fast_fit_resultsGPU.get(),
                                                                        circle_fit_resultsGPU_,
                                                                        offset);
      cudaCheck(cudaGetLastError());

      kernel_LineFit<5><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                                      5,
                                                                      bField_,
                                                                      outputSoa_,
                                                                      hitsGPU.get(),
                                                                      hits_geGPU.get(),
                                                                      fast_fit_resultsGPU.get(),
                                                                      circle_fit_resultsGPU_,
                                                                      offset);
      cudaCheck(cudaGetLastError());
#ifdef NTUPLE_DEBUG
  std::cout << "---> Riemann 8" << std::endl;
#endif
    }
  }
}

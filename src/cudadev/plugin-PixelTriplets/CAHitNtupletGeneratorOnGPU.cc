//
// Original Author: Felice Pantaleo, CERN
//

//#define GPU_DEBUG

#include <array>
#include <cassert>
#include <functional>
#include <vector>

#include "Framework/Event.h"

#include "CAHitNtupletGeneratorOnGPU.h"

namespace {

  template <typename T>
  T sqr(T x) {
    return x * x;
  }

  cAHitNtupletGenerator::QualityCuts makeQualityCuts() {
    auto coeff = std::vector<double>{0.9, 1.8};
    double ptMax = 10.0;
    coeff[1] = (coeff[1] - coeff[0]) / log2(ptMax);
    return cAHitNtupletGenerator::QualityCuts{// polynomial coefficients for the pT-dependent chi2 cut
                                              {(float)coeff[0], (float)coeff[1], 0.f, 0.f},
                                              // max pT used to determine the chi2 cut
                                              (float)ptMax,
                                              // chi2 scale factor: 8 for broken line fit, ?? for Riemann fit
                                              8.0f,
                                              // regional cuts for triplets
                                              {0.3f,
                                               0.5f,
                                               12.f
                                               },
                                              // regional cuts for quadruplets
                                              {0.5f,
                                               0.3f,
                                               12.f
                                               }};
  }
}  // namespace

using namespace std;
CAHitNtupletGeneratorOnGPU::CAHitNtupletGeneratorOnGPU(edm::ProductRegistry& reg)
    : m_params(true,              // onGPU
               4,                 // minHitsPerNtuplet,
               524288,            // maxNumberOfDoublets
               10,                 // minHitsForSharingCut
               false,             // useRiemannFit
               false,              // fitNas4,
               false,              // includeJumpingForwardDoublets
               true,              // earlyFishbone
               false,             // lateFishbone
               false,              // idealConditions
               false,             // fillStatistics
               true,              // doClusterCut
               true,              // doZ0Cut
               true,              // doPtCut
               true,              // doSharedHitCut
	       false,              // duplciatePassThrough
	       true,              // useSimpleTripletCleaner
               0.8999999761581421,    // ptmin
               0.0020000000949949026, // CAThetaCutBarrel
               0.003000000026077032, // CAThetaCutForward
               0.03284072249589491, // hardCurvCut
               0.15000000596046448, // dcaCutInnerTriplet
               0.25,              // dcaCutOuterTriplet
               makeQualityCuts()) {
#ifdef DUMP_GPU_TK_TUPLES
  printf("TK: %s %s % %s %s %s %s %s %s %s %s %s %s %s %s %s\n",
         "tid",
         "qual",
         "nh",
         "charge",
         "pt",
         "eta",
         "phi",
         "tip",
         "zip",
         "chi2",
         "h1",
         "h2",
         "h3",
         "h4",
         "h5");
#endif

  if (m_params.onGPU_) {
    cudaCheck(cudaMalloc(&m_counters, sizeof(Counters)));
    cudaCheck(cudaMemset(m_counters, 0, sizeof(Counters)));
  } else {
    m_counters = new Counters();
    memset(m_counters, 0, sizeof(Counters));
  }
}

CAHitNtupletGeneratorOnGPU::~CAHitNtupletGeneratorOnGPU() {
  if (m_params.onGPU_) {
    if (m_params.doStats_) {
      // crash on multi-gpu processes
      CAHitNtupletGeneratorKernelsGPU::printCounters(m_counters);
    }
    cudaFree(m_counters);
  } else {
    if (m_params.doStats_) {
      CAHitNtupletGeneratorKernelsCPU::printCounters(m_counters);
    }
    delete m_counters;
  }
}

PixelTrackHeterogeneous CAHitNtupletGeneratorOnGPU::makeTuplesAsync(TrackingRecHit2DGPU const& hits_d,
                                                                    float bfield,
                                                                    cudaStream_t stream) const {
  PixelTrackHeterogeneous tracks(cms::cuda::make_device_unique<pixelTrack::TrackSoA>(stream));
  cudaDeviceSynchronize();
  auto* soa = tracks.get();
  assert(soa);
  cudaDeviceSynchronize();
  CAHitNtupletGeneratorKernelsGPU kernels(m_params);
  kernels.setCounters(m_counters);
  kernels.allocateOnGPU(hits_d.nHits(), stream);
  cudaDeviceSynchronize();
  kernels.buildDoublets(hits_d, stream);
  kernels.launchKernels(hits_d, soa, stream);
  cudaDeviceSynchronize();
  HelixFitOnGPU fitter(bfield, m_params.fitNas4_);
  fitter.allocateOnGPU(&(soa->hitIndices), kernels.tupleMultiplicity(), soa);
  cudaDeviceSynchronize();
  if (m_params.useRiemannFit_) {
    fitter.launchRiemannKernels(hits_d.view(), hits_d.nHits(), caConstants::maxNumberOfQuadruplets, stream);
  } else {
    fitter.launchBrokenLineKernels(hits_d.view(), hits_d.nHits(), caConstants::maxNumberOfQuadruplets, stream);
  }
  cudaDeviceSynchronize();
  kernels.classifyTuples(hits_d, soa, stream);
#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
  std::cout << "finished building pixel tracks on GPU" << std::endl;
#endif

  return tracks;
}

PixelTrackHeterogeneous CAHitNtupletGeneratorOnGPU::makeTuples(TrackingRecHit2DCPU const& hits_d, float bfield) const {
  PixelTrackHeterogeneous tracks(std::make_unique<pixelTrack::TrackSoA>());

  auto* soa = tracks.get();
  assert(soa);

  CAHitNtupletGeneratorKernelsCPU kernels(m_params);
  kernels.setCounters(m_counters);
  kernels.allocateOnGPU(hits_d.nHits(), nullptr);

  kernels.buildDoublets(hits_d, nullptr);
  kernels.launchKernels(hits_d, soa, nullptr);

  if (0 == hits_d.nHits())
    return tracks;

  // now fit
  HelixFitOnGPU fitter(bfield, m_params.fitNas4_);
  fitter.allocateOnGPU(&(soa->hitIndices), kernels.tupleMultiplicity(), soa);

  if (m_params.useRiemannFit_) {
    fitter.launchRiemannKernelsOnCPU(hits_d.view(), hits_d.nHits(), caConstants::maxNumberOfQuadruplets);
  } else {
    fitter.launchBrokenLineKernelsOnCPU(hits_d.view(), hits_d.nHits(), caConstants::maxNumberOfQuadruplets);
  }

  kernels.classifyTuples(hits_d, soa, nullptr);

#ifdef GPU_DEBUG
  std::cout << "finished building pixel tracks on CPU" << std::endl;
#endif

  return tracks;
}

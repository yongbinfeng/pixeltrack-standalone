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
    auto coeff = std::vector<double>{0.68177776, 0.74609577, -0.08035491, 0.00315399};  // chi2Coeff
    return cAHitNtupletGenerator::QualityCuts{// polynomial coefficients for the pT-dependent chi2 cut
                                              {(float)coeff[0], (float)coeff[1], (float)coeff[2], (float)coeff[3]},
                                              // max pT used to determine the chi2 cut
                                              10.f,  // chi2MaxPt
                                                     // chi2 scale factor: 30 for broken line fit, 45 for Riemann fit
                                              8.f,  // chi2Scale
                                                     // regional cuts for triplets
                                              {
                                                  0.3f,  //tripletMaxTip
                                                  0.5f,  // tripletMinPt
                                                  12.f   // tripletMaxZip
                                              },
                                              // regional cuts for quadruplets
                                              {
                                                  0.5f,  // quadrupletMaxTip
                                                  0.3f,  // quadrupletMinPt
                                                  12.f   // quadrupletMaxZip
                                              }};
  }
}  // namespace

using namespace std;
CAHitNtupletGeneratorOnGPU::CAHitNtupletGeneratorOnGPU(edm::ProductRegistry& reg)
    : m_params(true,              // onGPU
               3,                 // minHitsPerNtuplet,
               524288,            // maxNumberOfDoublets
               10,                 // minHitsForSharingCut
               false,             // useRiemannFit
               false,              // fitNas4,
               false,              // includeJumpingForwardDoublets
               true,              // earlyFishbone
               false,             // lateFishbone
               true,              // idealConditions
               false,             // doStats
               true,              // doClusterCut
               true,              // doZ0Cut
               true,              // doPtCut
               true,              // doSharedHitCut
	       false,              // duplciatePassThrough
	       true,              // useSimpleTripletCleaner
               0.899999976158,    // ptmin
               0.00200000009499,  // CAThetaCutBarrel
               0.00300000002608,  // CAThetaCutForward
               0.0328407224959,   // hardCurvCut
               0.15000000596,     // dcaCutInnerTriplet
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
  std::cout << " A " << std::endl;
  PixelTrackHeterogeneous tracks(cms::cuda::make_device_unique<pixelTrack::TrackSoA>(stream));
  cudaDeviceSynchronize();
  std::cout << " B " << std::endl;
  auto* soa = tracks.get();
  assert(soa);
  cudaDeviceSynchronize();
  std::cout << " C " << std::endl;
  CAHitNtupletGeneratorKernelsGPU kernels(m_params);
  kernels.setCounters(m_counters);
  kernels.allocateOnGPU(hits_d.nHits(), stream);
  cudaDeviceSynchronize();
  std::cout << " D " << std::endl;
  kernels.buildDoublets(hits_d, stream);
  std::cout << " D 1 " << std::endl;
  kernels.launchKernels(hits_d, soa, stream);
  cudaDeviceSynchronize();
  std::cout << " E " << std::endl;
  HelixFitOnGPU fitter(bfield, m_params.fitNas4_);
  fitter.allocateOnGPU(&(soa->hitIndices), kernels.tupleMultiplicity(), soa);
  std::cout << " F " << std::endl;
  cudaDeviceSynchronize();
  std::cout << " F 1 " << std::endl;
  if (m_params.useRiemannFit_) {
    fitter.launchRiemannKernels(hits_d.view(), hits_d.nHits(), caConstants::maxNumberOfQuadruplets, stream);
  } else {
    fitter.launchBrokenLineKernels(hits_d.view(), hits_d.nHits(), caConstants::maxNumberOfQuadruplets, stream);
  }
  std::cout << " F 2 " << std::endl;
  cudaDeviceSynchronize();
  kernels.classifyTuples(hits_d, soa, stream);
  std::cout << " G " << std::endl;
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

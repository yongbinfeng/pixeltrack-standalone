//
// Original Author: Felice Pantaleo, CERN
//

// #define GPU_DEBUG

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
    auto coeff = std::vector<double>{0.9,1.8};
    float ptMax = 10;//pset.getParameter<double>("chi2MaxPt");
    coeff[1] = (coeff[1] - coeff[0]) / log2(ptMax);
    return cAHitNtupletGenerator::QualityCuts{// polynomial coefficients for the pT-dependent chi2 cut
      {(float)coeff[0], (float)coeff[1], 0.f, 0.f},
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
               4,                 // minHitsPerNtuplet,
               caConstants::maxNumberOfDoublets,            // maxNumberOfDoublets
               5,                 // minHitsForSharingCut
               false,             // useRiemannFit
               true,              // fit5as4,
               false,             // includeJumpingForwardDoublets
               true,              // earlyFishbone
               false,             // lateFishbone
               true,              // idealConditions
               false,             // doStats
               true,              // doClusterCut
               true,              // doZ0Cut
               true,              // doPtCut
               true,              // doSharedHitCut
	       false,              // dupPassThrough
               false,             // UseSimpleTriplet
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
  PixelTrackHeterogeneous tracks(cms::cuda::make_device_unique<pixelTrack::TrackSoA>(stream));

  auto* soa = tracks.get();
  assert(soa);

  CAHitNtupletGeneratorKernelsGPU kernels(m_params);
  kernels.setCounters(m_counters);
  kernels.allocateOnGPU(hits_d.nHits(), stream);

  kernels.buildDoublets(hits_d, stream);
  kernels.launchKernels(hits_d, soa, stream);
  kernels.fillHitDetIndices(hits_d.view(), soa, stream);  // in principle needed only if Hits not "available"

  HelixFitOnGPU fitter(bfield, m_params.fit5as4_);
  fitter.allocateOnGPU(&(soa->hitIndices), kernels.tupleMultiplicity(), soa);
  if (m_params.useRiemannFit_) {
    fitter.launchRiemannKernels(hits_d.view(), hits_d.nHits(), caConstants::maxNumberOfQuadruplets, stream);
  } else {
    fitter.launchBrokenLineKernels(hits_d.view(), hits_d.nHits(), caConstants::maxNumberOfQuadruplets, stream);
  }
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
  kernels.fillHitDetIndices(hits_d.view(), soa, nullptr);  // in principle needed only if Hits not "available"

  if (0 == hits_d.nHits())
    return tracks;

  // now fit
  HelixFitOnGPU fitter(bfield, m_params.fit5as4_);
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

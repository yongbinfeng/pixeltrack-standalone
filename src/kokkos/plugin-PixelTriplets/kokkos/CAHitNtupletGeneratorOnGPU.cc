//
// Original Author: Felice Pantaleo, CERN
//

#include <array>
#include <cassert>
#include <functional>
#include <vector>

#include "Framework/Event.h"
#include "KokkosCore/ExecSpaceCache.h"
#include "KokkosCore/ViewHelpers.h"

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
                                              30.f,  // chi2Scale
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
namespace KOKKOS_NAMESPACE {
  CAHitNtupletGeneratorOnGPU::CAHitNtupletGeneratorOnGPU(edm::ProductRegistry& reg)
      : m_params(true,              // onGPU
                 3,                 // minHitsPerNtuplet,
                 458752,            // maxNumberOfDoublets
                 false,             //useRiemannFit
                 true,              // fit5as4,
                 true,              //includeJumpingForwardDoublets
                 true,              // earlyFishbone
                 false,             // lateFishbone
                 true,              // idealConditions
                 false,             //fillStatistics
                 true,              // doClusterCut
                 true,              // doZ0Cut
                 true,              // doPtCut
                 0.899999976158,    // ptmin
                 0.00200000009499,  // CAThetaCutBarrel
                 0.00300000002608,  // CAThetaCutForward
                 0.0328407224959,   // hardCurvCut
                 0.15000000596,     // dcaCutInnerTriplet
                 0.25,              // dcaCutOuterTriplet
                 makeQualityCuts()) {
    auto const& execSpaceWrapped = cms::kokkos::getExecSpaceCache<KokkosExecSpace>().get();
    auto const& execSpace = execSpaceWrapped->space();
    m_counters = cms::kokkos::make_shared<Counters, KokkosDeviceMemSpace>(execSpace);
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

    auto tmp = cms::kokkos::make_mirror_shared(m_counters, execSpace);
    memset(tmp.get(), 0, sizeof(Counters));
    cms::kokkos::deep_copy(execSpace, m_counters, tmp);
    execSpace.fence();
  }

  CAHitNtupletGeneratorOnGPU::~CAHitNtupletGeneratorOnGPU() {
    if (m_params.doStats_) {
      CAHitNtupletGeneratorKernels::printCounters(m_counters);
    }
  }

  cms::kokkos::shared_ptr<pixelTrack::TrackSoA, KokkosDeviceMemSpace> CAHitNtupletGeneratorOnGPU::makeTuples(
      TrackingRecHit2DKokkos<KokkosDeviceMemSpace> const& hits_d, float bfield, KokkosExecSpace const& execSpace) {
    auto tracks = cms::kokkos::make_shared<pixelTrack::TrackSoA, KokkosDeviceMemSpace>(execSpace);

    CAHitNtupletGeneratorKernels kernels(m_params);
    kernels.counters_ = m_counters.get();

    kernels.allocateOnGPU(execSpace);

    kernels.buildDoublets(hits_d, execSpace);
    kernels.launchKernels(hits_d, tracks, execSpace);
    kernels.fillHitDetIndices(hits_d.view(), tracks, execSpace);  // in principle needed only if Hits not "available"

    HelixFitOnGPU fitter(bfield, m_params.fit5as4_);
    fitter.allocateOnGPU(&(tracks->hitIndices), kernels.tupleMultiplicity().get(), tracks.get());
    if (m_params.useRiemannFit_) {
      fitter.launchRiemannKernels(hits_d.view(), hits_d.nHits(), CAConstants::maxNumberOfQuadruplets(), execSpace);
    } else {
      fitter.launchBrokenLineKernels(hits_d.view(), hits_d.nHits(), CAConstants::maxNumberOfQuadruplets(), execSpace);
    }
    kernels.classifyTuples(hits_d, tracks, execSpace);
    return tracks;
  }
}  // namespace KOKKOS_NAMESPACE

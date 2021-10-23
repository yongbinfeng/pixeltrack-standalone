#include <cuda_runtime.h>

#include "CUDACore/HostProduct.h"
#include "CUDACore/Product.h"
#include "CUDADataFormats/TrackingRecHit2DHeterogeneous.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "CUDACore/ScopedContext.h"

class SiPixelRecHitFromCUDA : public edm::EDProducerExternalWork {
public:
  explicit SiPixelRecHitFromCUDA(edm::ProductRegistry& reg);
  ~SiPixelRecHitFromCUDA() override = default;

  using HMSstorage = HostProduct<uint32_t[]>;

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  const edm::EDGetTokenT<cms::cuda::Product<TrackingRecHit2DGPU>> hitsToken_;  // CUDA hits
  const edm::EDPutTokenT<HMSstorage> hostPutToken_;

  uint32_t nHits_;
  cms::cuda::host::unique_ptr<float[]> store32_;
  cms::cuda::host::unique_ptr<uint32_t[]> hitsModuleStart_;
};

SiPixelRecHitFromCUDA::SiPixelRecHitFromCUDA(edm::ProductRegistry& reg)
  : hitsToken_(reg.consumes<cms::cuda::Product<TrackingRecHit2DGPU>>()),
    hostPutToken_(reg.produces<HMSstorage>()) {}

void SiPixelRecHitFromCUDA::acquire(edm::Event const& iEvent,
                                    edm::EventSetup const& iSetup,
                                    edm::WaitingTaskWithArenaHolder waitingTaskHolder) {

  cms::cuda::Product<TrackingRecHit2DGPU> const& inputDataWrapped = iEvent.get(hitsToken_);
  cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  const auto& inputData = ctx.get(inputDataWrapped);

  nHits_ = inputData.nHits();

  if (0 == nHits_)
    return;
  store32_ = inputData.localCoordToHostAsync(ctx.stream());
  hitsModuleStart_ = inputData.hitsModuleStartToHostAsync(ctx.stream());
}

void SiPixelRecHitFromCUDA::produce(edm::Event& iEvent,edm::EventSetup const& es) {
  // allocate a buffer for the indices of the clusters
  static const unsigned maxNumModules = 2000;
  auto hmsp = std::make_unique<uint32_t[]>(maxNumModules + 4*nHits_ + 2);
  hmsp.get()[0] = nHits_;
  std::memcpy(hmsp.get() + 1,               hitsModuleStart_.get(),(maxNumModules+1)*sizeof(uint32_t));
  std::memcpy(hmsp.get() +maxNumModules + 2,store32_.get(),        (4*nHits_)*sizeof(float)); 
  // wrap the buffer in a HostProduct, and move it to the Event, without reallocating the buffer or affecting hitsModuleStart
  iEvent.emplace(hostPutToken_, std::move(hmsp));
}

DEFINE_FWK_MODULE(SiPixelRecHitFromCUDA);

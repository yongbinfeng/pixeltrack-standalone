#include <cuda_runtime.h>

#include "CUDACore/Product.h"
#include "CUDADataFormats/SiPixelDigiErrorsCUDA.h"
#include "DataFormats/SiPixelErrorsSoA.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "CUDACore/ScopedContext.h"
#include "CUDACore/host_unique_ptr.h"

class SiPixelDigiErrorsSoAFromCUDA : public edm::EDProducerExternalWork {
public:
  explicit SiPixelDigiErrorsSoAFromCUDA(edm::ProductRegistry& reg);
  ~SiPixelDigiErrorsSoAFromCUDA() override = default;

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  edm::EDGetTokenT<cms::cuda::Product<SiPixelDigiErrorsCUDA>> digiErrorGetToken_;
  edm::EDPutTokenT<SiPixelErrorsSoA> digiErrorPutToken_;

  cms::cuda::host::unique_ptr<PixelErrorCompact[]> data_;
  cms::cuda::SimpleVector<PixelErrorCompact> error_;
  size_t nErrors_;
  //const PixelFormatterErrors* formatterErrors_ = nullptr;
};

SiPixelDigiErrorsSoAFromCUDA::SiPixelDigiErrorsSoAFromCUDA(edm::ProductRegistry& reg)
  : digiErrorGetToken_{reg.consumes<cms::cuda::Product<SiPixelDigiErrorsCUDA>>()},
    digiErrorPutToken_{reg.produces<SiPixelErrorsSoA>()} {}

void SiPixelDigiErrorsSoAFromCUDA::acquire(edm::Event const& iEvent,
                                           edm::EventSetup const& iSetup,
                                           edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  // Do the transfer in a CUDA stream parallel to the computation CUDA stream
  cms::cuda::ScopedContextAcquire ctx{iEvent.streamID(), std::move(waitingTaskHolder)};
  const auto& gpuDigiErrors = ctx.get(iEvent, digiErrorGetToken_);
  auto tmp = gpuDigiErrors.dataErrorToHostAsync(ctx.stream());
  error_ = tmp.first;
  data_ = std::move(tmp.second);
  nErrors_ = error_.size();
  //formatterErrors_ = &(gpuDigiErrors.formatterErrors());
}

void SiPixelDigiErrorsSoAFromCUDA::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  // The following line copies the data from the pinned host memory to
  // regular host memory. In principle that feels unnecessary (why not
  // just use the pinned host memory?). There are a few arguments for
  // doing it though
  // - Now can release the pinned host memory back to the (caching) allocator
  //   * if we'd like to keep the pinned memory, we'd need to also
  //     keep the CUDA stream around as long as that, or allow pinned
  //     host memory to be allocated without a CUDA stream
  // - What if a CPU algorithm would produce the same SoA? We can't
  //   use cudaMallocHost without a GPU...
  iEvent.emplace(digiErrorPutToken_, nErrors_, error_.data());//, formatterErrors_);

  error_ = cms::cuda::make_SimpleVector<PixelErrorCompact>(0, nullptr);
  data_.reset();
  //formatterErrors_ = nullptr;
}
DEFINE_FWK_MODULE(SiPixelDigiErrorsSoAFromCUDA);

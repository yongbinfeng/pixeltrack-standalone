#include "CUDACore/Product.h"
#include "CUDACore/ScopedContext.h"
#include "CUDADataFormats/PixelTrackHeterogeneous.h"
#include "CUDADataFormats/SiPixelClustersCUDA.h"
#include "CUDADataFormats/SiPixelDigisCUDA.h"
#include "CUDADataFormats/ZVertexHeterogeneous.h"
#include "DataFormats/DigiClusterCount.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

#include <atomic>
#include <iostream>
#include <mutex>
#include <sstream>

namespace {
  std::atomic<int> allEvents = 0;
  std::atomic<int> goodEvents = 0;
}  // namespace

class CountValidatorSimple : public edm::EDProducer {
public:
  explicit CountValidatorSimple(edm::ProductRegistry& reg);
  uint32_t* getOutput();
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  
private:
  void endJob() override;

  //edm::EDGetTokenT<DigiClusterCount> digiClusterCountToken_;

  edm::EDGetTokenT<cms::cuda::Product<SiPixelDigisCUDA>> digiToken_;
  edm::EDGetTokenT<cms::cuda::Product<SiPixelClustersCUDA>> clusterToken_;
  uint32_t nModules_;
  uint32_t nClusters_;
  uint32_t  nDigis_;
  cms::cuda::host::unique_ptr<uint32_t[]> pdigi_;
  cms::cuda::host::unique_ptr<uint32_t[]> rawIdArr_;
  cms::cuda::host::unique_ptr<uint16_t[]> adc_;
  cms::cuda::host::unique_ptr<int32_t[]>  clus_;
  uint32_t* output_;

};

CountValidatorSimple::CountValidatorSimple(edm::ProductRegistry& reg)
  : //digiClusterCountToken_(reg.consumes<DigiClusterCount>()),
      digiToken_(reg.consumes<cms::cuda::Product<SiPixelDigisCUDA>>()),
      clusterToken_(reg.consumes<cms::cuda::Product<SiPixelClustersCUDA>>()) {
  std::cout << "---> Simple " <<  std::endl;
}

void CountValidatorSimple::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::cout << "----> simple produce " << std::endl;
  bool ok = true;
  //DigiClusterCount
  std::stringstream ss;

  ss << "Event " << iEvent.eventID() << " ";

  {
    auto const& pdigis = iEvent.get(digiToken_);
    cms::cuda::ScopedContextProduce ctx{pdigis};
    //auto const& count = iEvent.get(digiClusterCountToken_);
    auto const& digis = ctx.get(iEvent, digiToken_);
    auto const& clusters = ctx.get(iEvent, clusterToken_);
    DigiClusterCount count(1,1,1);
    if (digis.nModules() != count.nModules()) {
      ss << "\n N(modules) is " << digis.nModules() << " expected " << count.nModules();
      ok = false;
    }
    if (digis.nDigis() != count.nDigis()) {
      ss << "\n N(digis) is " << digis.nDigis() << " expected " << count.nDigis();
      ok = false;
    }
    if (clusters.nClusters() != count.nClusters()) {
      ss << "\n N(clusters) is " << clusters.nClusters() << " expected " << count.nClusters();
      ok = false;
    }
    nDigis_    = digis.nDigis();
    pdigi_     = digis.pdigiToHostAsync(ctx.stream());
    rawIdArr_  = digis.rawIdArrToHostAsync(ctx.stream());
    adc_       = digis.adcToHostAsync(ctx.stream());
    clus_      = digis.clusToHostAsync(ctx.stream());
    output_[0] = digis.nDigis();
    std::memcpy(output_ + 1,          rawIdArr_.get(),nDigis_*sizeof(uint32_t));
    std::memcpy(output_ + (nDigis_+1)  ,pdigi_.get()   ,nDigis_*sizeof(uint32_t));
    std::memcpy(output_ + (2*nDigis_+1),adc_.get()     ,nDigis_*sizeof(uint16_t));
    std::memcpy(output_ + (3*nDigis_+1),clus_.get()    ,nDigis_*sizeof(int32_t));
    //output_[1] = (uint32_t*)rawIdArr_.get();
    //output_[nDigis_+1]   = (uint32_t*)pdigi_.get();
    //output_[2*nDigis_+1] = (uint32_t*)adc_.get();
    //output_[3*nDigis_+1] = (uint32_t*) clus_.get();
    std::cout << "--> formatting output" << std::endl;
    for(unsigned int i0 = 0; i0 < nDigis_*4; i0++) {
      std::cout << "---> " << i0 << " -- " << output_[i0] << std::endl;
    }
    std::cout << "--> formatting output done" << std::endl;
  }

  ++allEvents;
  if (ok) {
    ++goodEvents;
  } else {
    std::cout << ss.str() << std::endl;
  }
}
uint32_t* CountValidatorSimple::getOutput() {
  return output_;
}
void CountValidatorSimple::endJob() {
  if (allEvents == goodEvents) {
    std::cout << "CountValidatorSimple: all " << allEvents << " events passed validation\n";
  } else {
    std::cout << "CountValidatorSimple: " << (allEvents - goodEvents) << " events failed validation (see details above)\n";
    throw std::runtime_error("CountValidatorSimple failed");
  }
}

DEFINE_FWK_MODULE(CountValidatorSimple);

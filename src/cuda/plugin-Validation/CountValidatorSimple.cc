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
  std::atomic<int> allEvents{0};
  std::atomic<int> goodEvents{0};
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
  edm::EDGetTokenT<PixelTrackHeterogeneous> trackToken_;
  edm::EDGetTokenT<ZVertexHeterogeneous> vertexToken_;
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
  clusterToken_(reg.consumes<cms::cuda::Product<SiPixelClustersCUDA>>()),
  trackToken_(reg.consumes<PixelTrackHeterogeneous>()),
  vertexToken_(reg.consumes<ZVertexHeterogeneous>()) {
  output_ = new uint32_t[32768*6+1];
}

void CountValidatorSimple::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //std::cout << "----> simple produce " << std::endl;
  bool ok = true;
  //DigiClusterCount
  std::stringstream ss;

  ss << "Event " << iEvent.eventID() << " ";
  unsigned int pCount = 0;
  /*
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
    output_ = new uint32_t[4*nDigis_+1];
    output_[0] = digis.nDigis();
    std::memcpy(output_ + 1,          rawIdArr_.get(),nDigis_*sizeof(uint32_t));
    std::memcpy(output_ + (nDigis_+1)  ,pdigi_.get()   ,nDigis_*sizeof(uint32_t));
    std::memcpy(output_ + (2*nDigis_+1),adc_.get()     ,nDigis_*sizeof(uint16_t));
    std::memcpy(output_ + (3*nDigis_+1),clus_.get()    ,nDigis_*sizeof(uint32_t));
    //output_[1] = (uint32_t*)rawIdArr_.get();
    //output_[nDigis_+1]   = (uint32_t*)pdigi_.get();
    //output_[2*nDigis_+1] = (uint32_t*)adc_.get();
    //output_[3*nDigis_+1] = (uint32_t*) clus_.get();
    std::cout << " -- " << nDigis_ << "--> " << rawIdArr_.get()[0] << " -- " << pdigi_.get()[0]    << " -- " << adc_.get()[0] << " -- " << clus_.get()[0] << std::endl;
    std::cout << " -- " << nDigis_ << "--> " << output_[1]         << " -- " << output_[nDigis_+1] << " -- " <<  output_[2*nDigis_+1] << " -- " <<  output_[3*nDigis_+1] << std::endl;
    //pCount = 3*nDigis_+1;
  } 
  */
  //{
    auto const& tracks = iEvent.get(trackToken_);
    uint32_t nTracks = tracks->stride();
    //std::cout << " N Tracks ---> " << nTracks << std::endl;
    output_[0] = nTracks; pCount++;
    //std::cout << " N Tracks 1 --- " << std::endl;
    std::memcpy(output_ + pCount,tracks->chi2.data()   ,nTracks*sizeof(float));   pCount+=nTracks;
    std::memcpy(output_ + pCount,tracks->m_quality.data(),nTracks*sizeof(uint8_t)); pCount+=nTracks;
    std::memcpy(output_ + pCount,tracks->eta.data()    ,nTracks*sizeof(float)); pCount+=nTracks;
    std::memcpy(output_ + pCount,tracks->pt.data()     ,nTracks*sizeof(float)); pCount+=nTracks;
    //std::memcpy(output_ + pCount,tracks->hitIndices.begin(),5*nTracks*nTracks*sizeof(uint16_t)); pCount+=nTracks*nTracks*5;
    //std::memcpy(output_ + pCount,tracks->detIndices.begin(),5*nTracks*nTracks*sizeof(uint16_t)); pCount+=nTracks*nTracks*5;
    //std::memcpy(output_ + pCount,tracks->stateAtBS,nTracks_*sizeof(uint8_t)); pCount+=nTracks_;
    //std::cout << " N Tracks 4 --- " << nTracks << std::endl;
    //}
    //{
    //std::cout << " N Vtx 1 --- " << std::endl;
    static constexpr uint32_t MAXTRACKS = 32 * 1024;
    static constexpr uint32_t MAXVTX = 1024;
    auto const& vertices = iEvent.get(vertexToken_);
    uint32_t nvtx = vertices->nvFinal;
    //std::cout << " N Vtx 2 --- " << nvtx << std::endl;
    output_[pCount] = nvtx;
    std::memcpy(output_ + pCount,vertices->idv    ,MAXTRACKS*sizeof(int16_t));   pCount+=MAXTRACKS;
    std::memcpy(output_ + pCount,vertices->zv     ,MAXVTX*sizeof(float));        pCount+=MAXVTX;
    std::memcpy(output_ + pCount,vertices->wv     ,MAXVTX*sizeof(float));        pCount+=MAXVTX;
    std::memcpy(output_ + pCount,vertices->chi2   ,MAXVTX*sizeof(float));        pCount+=MAXVTX;
    std::memcpy(output_ + pCount,vertices->ptv2   ,MAXVTX*sizeof(float));        pCount+=MAXVTX;
    std::memcpy(output_ + pCount,vertices->ndof   ,MAXVTX*sizeof(int32_t));      pCount+=MAXVTX;
    std::memcpy(output_ + pCount,vertices->sortInd,MAXVTX*sizeof(uint16_t));     pCount+=MAXVTX;
    //std::cout << " N Vtx 5 --- " << " --- total " << pCount << std::endl;
    //}
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

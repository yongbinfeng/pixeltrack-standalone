#include "CUDACore/HostProduct.h"
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
  uint8_t* getOutput();
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  using HMSstorage = HostProduct<uint32_t[]>;
  
private:
  void endJob() override;

  //edm::EDGetTokenT<DigiClusterCount> digiClusterCountToken_;
  edm::EDGetTokenT<HMSstorage> hitsToken_;
  edm::EDGetTokenT<cms::cuda::Product<SiPixelDigisCUDA>> digiToken_;
  edm::EDGetTokenT<cms::cuda::Product<SiPixelClustersCUDA>> clusterToken_;
  edm::EDGetTokenT<PixelTrackHeterogeneous> trackToken_;
  edm::EDGetTokenT<ZVertexHeterogeneous> vertexToken_;
  uint32_t nModules_;
  uint32_t nClusters_;
  uint32_t  nDigis_;
  uint32_t  nHits_;
  cms::cuda::host::unique_ptr<uint32_t[]> pdigi_;
  cms::cuda::host::unique_ptr<uint32_t[]> rawIdArr_;
  cms::cuda::host::unique_ptr<uint16_t[]> adc_;
  cms::cuda::host::unique_ptr<int32_t[]>  clus_;
  uint8_t* output_;

};

CountValidatorSimple::CountValidatorSimple(edm::ProductRegistry& reg)
  : //digiClusterCountToken_(reg.consumes<DigiClusterCount>()),
  hitsToken_(reg.consumes<HMSstorage>()),
  digiToken_(reg.consumes<cms::cuda::Product<SiPixelDigisCUDA>>()),
  clusterToken_(reg.consumes<cms::cuda::Product<SiPixelClustersCUDA>>()),
  trackToken_(reg.consumes<PixelTrackHeterogeneous>()),
  vertexToken_(reg.consumes<ZVertexHeterogeneous>()) {
  output_ = new uint8_t[7000000];
  //output_ = new uint32_t[5*150000+32768*36+1+2000+3*35000];
  //output_.reset(new uint32_t(5*150000+32768*36+1+2000+3*35000));
}

void CountValidatorSimple::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  bool ok = true;
  unsigned int pCount = 0;
  {

    auto const& hits = iEvent.get(hitsToken_);
    
    auto const& pdigis = iEvent.get(digiToken_);
    cms::cuda::ScopedContextProduce ctx{pdigis};
    auto const& digis = ctx.get(iEvent, digiToken_);
    nHits_ = hits.get()[0];
    if(nHits_ > 35000) std::cout << "----> Too many Hits #Hits  " << nHits_ << " Max! 30000 " << std::endl;
    if(nHits_ > 35000) nHits_ = 35000;
    //output_[pCount] = nHits_; pCount++;
    std::memcpy(output_+pCount,&nHits_,sizeof(uint32_t)); pCount += 4;
    static const unsigned maxNumModules = 2000;
    std::memcpy(output_ + pCount,hits.get()+1,    (maxNumModules)*sizeof(uint32_t)); pCount += 4*maxNumModules;
    std::memcpy(output_ + pCount,hits.get()+2001, (3*nHits_)*sizeof(float)); pCount+=4*(3*nHits_);

    nDigis_    = digis.nDigis();
    if(nDigis_ > 150000) std::cout << "----> Too many Digis #Digis  " << nDigis_ << " Max! " << nDigis_ << std::endl;
    if(nDigis_ > 150000) nDigis_ = 150000;
    pdigi_     = digis.pdigiToHostAsync(ctx.stream());
    rawIdArr_  = digis.rawIdArrToHostAsync(ctx.stream());
    adc_       = digis.adcToHostAsync(ctx.stream());
    clus_      = digis.clusToHostAsync(ctx.stream());
    //output_[pCount] = nDigis_; pCount++;
    std::memcpy(output_+pCount,&nDigis_,sizeof(uint32_t)); pCount += 4;
    std::memcpy(output_ + pCount,pdigi_.get()   ,nDigis_*sizeof(uint32_t)); pCount+=4*nDigis_;
    std::memcpy(output_ + pCount,rawIdArr_.get(),nDigis_*sizeof(uint32_t)); pCount+=4*nDigis_;
    std::memcpy(output_ + pCount,adc_.get()     ,nDigis_*sizeof(uint16_t)); pCount+=2*nDigis_;
    std::memcpy(output_ + pCount,clus_.get()    ,nDigis_*sizeof(int32_t));  pCount+=4*nDigis_;
  }
  {
    auto const& tracks = iEvent.get(trackToken_);
    uint32_t nTracks = tracks->stride();
    //output_[pCount] = nTracks; pCount++;
    std::memcpy(output_ + pCount,&nTracks,sizeof(uint32_t)); pCount += 4;
    std::memcpy(output_ + pCount,tracks->chi2.data()     ,nTracks*sizeof(float));                  pCount+=4*nTracks;
    std::memcpy(output_ + pCount,tracks->m_quality.data(),nTracks*sizeof(uint8_t));                pCount+=nTracks;
    std::memcpy(output_ + pCount,tracks->eta.data()      ,nTracks*sizeof(float));                  pCount+=4*nTracks;
    std::memcpy(output_ + pCount,tracks->pt.data()       ,nTracks*sizeof(float));                  pCount+=4*nTracks;
    std::memcpy(output_ + pCount,tracks->stateAtBS.state(0).data(),     nTracks*sizeof(float)*5);  pCount+=(4*nTracks*5);
    std::memcpy(output_ + pCount,tracks->stateAtBS.covariance(0).data(),nTracks*sizeof(float)*15); pCount+=(4*nTracks*15);
    std::memcpy(output_ + pCount,tracks->hitIndices.begin(),5*nTracks*sizeof(uint32_t));     pCount+=4*nTracks*5;
    std::memcpy(output_ + pCount,tracks->hitIndices.off,    (nTracks+1)*sizeof(int32_t));        pCount+=4*(nTracks+1);
    std::memcpy(output_ + pCount,tracks->detIndices.begin(),5*nTracks*sizeof(uint32_t));     pCount+=4*nTracks*5;
    std::memcpy(output_ + pCount,tracks->detIndices.off,    (nTracks+1)*sizeof(int32_t));        pCount+=4*(nTracks+1);

    auto const& vertices = iEvent.get(vertexToken_);
    static constexpr uint32_t MAXTRACKS = 32 * 1024;
    static constexpr uint32_t MAXVTX = 1024;
    //output_[pCount] = nvtx; pCount++;
    std::memcpy(output_ + pCount,&(vertices->nvFinal),sizeof(uint32_t)); pCount += 4;
    std::memcpy(output_ + pCount,vertices->idv    ,MAXTRACKS*sizeof(int16_t));   pCount+=2*MAXTRACKS;
    std::memcpy(output_ + pCount,vertices->zv     ,MAXVTX*sizeof(float));        pCount+=4*MAXVTX;
    std::memcpy(output_ + pCount,vertices->wv     ,MAXVTX*sizeof(float));        pCount+=4*MAXVTX;
    std::memcpy(output_ + pCount,vertices->chi2   ,MAXVTX*sizeof(float));        pCount+=4*MAXVTX;
    std::memcpy(output_ + pCount,vertices->ptv2   ,MAXVTX*sizeof(float));        pCount+=4*MAXVTX;
    std::memcpy(output_ + pCount,vertices->ndof   ,MAXVTX*sizeof(int32_t));      pCount+=4*MAXVTX;
    std::memcpy(output_ + pCount,vertices->sortInd,MAXVTX*sizeof(uint16_t));     pCount+=2*MAXVTX;
  }
  std::cout << "----> " << pCount << " -- 8146596 " <<std::endl;
  ++allEvents;
  if (ok) {
    ++goodEvents;
  }
}
uint8_t* CountValidatorSimple::getOutput() {
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

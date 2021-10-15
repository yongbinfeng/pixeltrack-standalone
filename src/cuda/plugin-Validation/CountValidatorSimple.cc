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
  uint32_t* getOutput();
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
  uint32_t* output_;

};

CountValidatorSimple::CountValidatorSimple(edm::ProductRegistry& reg)
  : //digiClusterCountToken_(reg.consumes<DigiClusterCount>()),
  hitsToken_(reg.consumes<HMSstorage>()),
  digiToken_(reg.consumes<cms::cuda::Product<SiPixelDigisCUDA>>()),
  clusterToken_(reg.consumes<cms::cuda::Product<SiPixelClustersCUDA>>()),
  trackToken_(reg.consumes<PixelTrackHeterogeneous>()),
  vertexToken_(reg.consumes<ZVertexHeterogeneous>()) {
  output_ = new uint32_t[5*150000+32768*36+1+2000+3*35000];
  //output_.reset(new uint32_t(5*150000+32768*36+1+2000+3*35000));
}

void CountValidatorSimple::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  bool ok = true;
  unsigned int pCount = 0;
  {

    std::cout <<" ---  here" << std::endl;
    auto const& hits = iEvent.get(hitsToken_);
    
    auto const& pdigis = iEvent.get(digiToken_);
    cms::cuda::ScopedContextProduce ctx{pdigis};
    auto const& digis = ctx.get(iEvent, digiToken_);
    std::cout <<" --- 1" << std::endl;
    nHits_ = hits.get()[0];
    if(nHits_ > 35000) std::cout << "----> Too many Hits #Hits  " << nHits_ << " Max! 30000 " << std::endl;
    if(nHits_ > 35000) nHits_ = 35000;
    output_[pCount] = nHits_; pCount++;
    static const unsigned maxNumModules = 2000;
    std::memcpy(output_ + pCount,hits.get()+1,(maxNumModules)*sizeof(uint32_t)); pCount += maxNumModules;
    std::memcpy(output_ + pCount,hits.get()+2001,(3*nHits_)*sizeof(float)); pCount+=(3*nHits_);

    std::cout <<" ---2" << std::endl;
    nDigis_    = digis.nDigis();
    if(nDigis_ > 150000) std::cout << "----> Too many Digis #Digis  " << nDigis_ << " Max! " << nDigis_ << std::endl;
    if(nDigis_ > 150000) nDigis_ = 150000;
    pdigi_     = digis.pdigiToHostAsync(ctx.stream());
    rawIdArr_  = digis.rawIdArrToHostAsync(ctx.stream());
    adc_       = digis.adcToHostAsync(ctx.stream());
    clus_      = digis.clusToHostAsync(ctx.stream());
    output_[pCount] = nDigis_; pCount++;
    std::cout <<" ---> " << nDigis_ << " -- " << pCount << std::endl;
    std::memcpy(output_ + pCount,pdigi_.get()   ,nDigis_*sizeof(uint32_t)); pCount+=nDigis_;
    std::memcpy(output_ + pCount,rawIdArr_.get(),nDigis_*sizeof(uint32_t)); pCount+=nDigis_;
    std::memcpy(output_ + pCount,adc_.get()     ,nDigis_*sizeof(uint16_t)); pCount+=nDigis_;
    std::memcpy(output_ + pCount,clus_.get()    ,nDigis_*sizeof(int32_t));  pCount+=nDigis_;
    //std::cout <<" ---> Digi " << std::endl;
    //for(unsigned i0 = 0; i0 < nDigis_; i0++) {
    //  if(rawIdArr_[i0] != 0 && adc_[i0] > 0) std::cout << i0 << " -- " << nDigis_ << " -- " << pdigi_[i0] << " -- " << rawIdArr_[i0] << " -- " << adc_[i0] << " -- " << clus_[i0] << std::endl;
    // }
    //std::cout <<" ---> Digi Done " << std::endl;
    
    //std::cout << " -- " << nDigis_ << "--> " << rawIdArr_.get()[0] << " -- " << pdigi_.get()[0]    << " -- " << adc_.get()[0] << " -- " << clus_.get()[1] << std::endl;
    //std::cout << " -- " << nDigis_ << " -- " << pCountBase << "--> " << output_[pCountBase+1]        << " -- " << output_[pCountBase+nDigis_+1] << " -- " <<  output_[pCountBase+2*nDigis_+1] << " -- " <<  output_[pCountBase+3*nDigis_+2] << std::endl;
    //pCount = 3*nDigis_+1;
    //for(unsigned i0 = 0; i0 < nDigis_; i0++) {
    //  if(adc_.get()[i0] == 0) std::cout << " -zero- " << i0 << " -- " << rawIdArr_.get()[i0] << " -- " << adc_.get()[i0] << std::endl;
      //if(clus_.get()[i0] == 0) std::cout << " -zero- " << i0 << " -- " << rawIdArr_.get()[i0] << " -- " << adc_.get()[i0] << std::endl;
    //}
  }
  std::cout <<" ---3" << std::endl;
  {

    auto const& tracks = iEvent.get(trackToken_);
    uint32_t nTracks = tracks->stride();
    /*
    output_[pCount] = nTracks; pCount++;
    std::memcpy(output_ + pCount,tracks->chi2.data()     ,nTracks*sizeof(float));   pCount+=nTracks;
    std::memcpy(output_ + pCount,tracks->m_quality.data(),nTracks*sizeof(uint8_t)); pCount+=nTracks;
    std::memcpy(output_ + pCount,tracks->eta.data()      ,nTracks*sizeof(float)); pCount+=nTracks;
    std::memcpy(output_ + pCount,tracks->pt.data()       ,nTracks*sizeof(float)); pCount+=nTracks;
    std::memcpy(output_ + pCount,tracks->stateAtBS.state(0).data(),     nTracks*sizeof(float)*5); pCount+=(nTracks*5);
    std::memcpy(output_ + pCount,tracks->stateAtBS.covariance(0).data(),nTracks*sizeof(float)*15); pCount+=(nTracks*15);
    std::memcpy(output_ + pCount,tracks->hitIndices.begin(),5*nTracks*sizeof(uint32_t));     pCount+=nTracks*5;
    std::memcpy(output_ + pCount,tracks->hitIndices.off,(nTracks+1)*sizeof(int32_t));        pCount+=(nTracks+1);
    std::memcpy(output_ + pCount,tracks->detIndices.begin(),5*nTracks*sizeof(uint32_t));     pCount+=nTracks*5;
    std::memcpy(output_ + pCount,tracks->detIndices.off,(nTracks+1)*sizeof(int32_t));        pCount+=(nTracks+1);
    */
    //for(uint32_t iState = 0; iState < nTracks; iState++) {  
    //  std::cout << "---> " << iState << " -- " << tracks->stateAtBS.covariance(iState).size() << std::endl;
    //}
    //std::cout << "---> " << tracks->hitIndices.size()    << " -- " << tracks->hitIndices.bins[10] << " -- " << nTracks << std::endl;
    //std::cout << "---> " << tracks->detIndices.size()    << " -- " << tracks->detIndices.bins[10] << " -- " << nTracks << std::endl;
    //std::cout << "---> " << tracks->hitIndices.capacity() << " -- " << tracks->hitIndices.psws << std::endl;
    //std::cout << "---> " << tracks->detIndices.capacity() << " -- " << tracks->detIndices.psws << std::endl;
    //int test = int(trackQuality::dup);
    //int test1 = int(trackQuality::loose);
    std::cout <<" ---> here " << std::endl;
    std::vector<float> chi2;
    std::vector<uint8_t> quality;
    std::vector<float> eta;
    std::vector<float> pt;
    std::vector<float> stateAtBS;
    std::vector<float> stateAtBSCov;
    std::vector<uint32_t> hitsRaw;
    std::vector<uint32_t> hitsOff;
    std::vector<uint32_t> detsRaw;
    std::vector<uint32_t> detsOff;
    uint32_t pInt = 0;
    for(unsigned  i0 = 0; i0 < nTracks; i0++) {
      //std::cout << "-quality--> " << i0 << " -- " << int(tracks->quality(i0))  << " -- " << tracks->pt(i0) << " -- " << tracks->nHits(i0) << " -- " << test << " -- " << test1 <<  std::endl;
      if(tracks->quality(i0) > trackQuality::dup && tracks->pt(i0) > 0.5 && tracks->nHits(i0) > 0) {
	pInt++;
	chi2.emplace_back(tracks->chi2(i0));
	quality.emplace_back(tracks->quality(i0));
	eta.emplace_back(tracks->eta(i0));
	pt.emplace_back(tracks->pt(i0));
	stateAtBS.reserve(5);
	stateAtBSCov.reserve(15);
	std::memcpy(stateAtBS.data()+5*pInt,tracks->stateAtBS.state(i0).data(),sizeof(float)*5);
	std::memcpy(stateAtBSCov.data()+15*pInt,tracks->stateAtBS.covariance(i0).data(),sizeof(float)*15);
	hitsOff.emplace_back(tracks->hitIndices.size(i0));
	detsOff.emplace_back(tracks->detIndices.size(i0));
	hitsRaw.reserve(5);
	detsRaw.reserve(5);
	std::memcpy(hitsRaw.data()+5*pInt,tracks->hitIndices.begin(i0),sizeof(uint32_t)*5);
	std::memcpy(detsRaw.data()+5*pInt,tracks->detIndices.begin(i0),sizeof(uint32_t)*5);
      }
    }
    std::cout <<" ---> done " << std::endl;
    hitsOff.emplace_back(tracks->hitIndices.size());
    detsOff.emplace_back(tracks->detIndices.size());
    nTracks = pInt;
    output_[pCount] = nTracks; pCount++;
    std::memcpy(output_ + pCount,chi2.data(),              nTracks*sizeof(float));    pCount+=nTracks;
    std::memcpy(output_ + pCount,quality.data(),           nTracks*sizeof(uint8_t));  pCount+=nTracks;
    std::memcpy(output_ + pCount,eta.data(),               nTracks*sizeof(float));    pCount+=nTracks;
    std::memcpy(output_ + pCount,pt.data(),                nTracks*sizeof(float));    pCount+=nTracks;
    std::memcpy(output_ + pCount,stateAtBS.data(),         nTracks*sizeof(float)*5);  pCount+=(nTracks*5);
    std::memcpy(output_ + pCount,stateAtBSCov.data(),      nTracks*sizeof(float)*15); pCount+=(nTracks*15);
    std::memcpy(output_ + pCount,hitsRaw.data(),           5*nTracks*sizeof(uint32_t));        pCount+=nTracks*5;
    std::memcpy(output_ + pCount,hitsOff.data(),           (nTracks+1)*sizeof(int32_t));       pCount+=(nTracks+1);
    std::memcpy(output_ + pCount,detsRaw.data(),           5*nTracks*sizeof(uint32_t));        pCount+=nTracks*5;
    std::memcpy(output_ + pCount,detsOff.data(),           (nTracks+1)*sizeof(int32_t));       pCount+=(nTracks+1);
    //std::cout << "---> " << tracks->detIndices.capacity() << " -- " << tracks->detIndices.psws << std::endl;

    //std::cout << " -1-> " << tracks->hitIndices.size(0)           << " -- " << tracks->nHits(0) << std::endl;
    ////std::cout << " -1-> " << tracks->detIndices.size(0)           << " -- " << tracks->detIndices.size() << std::endl;
    //std::cout << " -1-> " << &(*(tracks->stateAtBS.state(0).array().begin()))           << " -- " << &(*(tracks->stateAtBS.state(0).array().end()))     << " -- " <<  &(*(tracks->stateAtBS.state(31721).array().begin())) << std::endl;
    //std::cout << "-1-> " << tracks->stateAtBS.state(0) << std::endl;
    //for(auto iState = 0; iState < 100; iState++) {  
    //  for(auto i0 = tracks->stateAtBS.covariance(iState).begin(); i0 < tracks->stateAtBS.covariance(iState).end(); i0++) { std::cout << "iter---> " << iState << " -- " << &(*i0) << " -- " << *i0 << std::endl;}
    //}
    //std::cout << " --> done " << std::endl;
    //std::cout << " -2-> " << &(*(tracks->stateAtBS.covariance(0).array().begin()))      << " -- " << &(*(tracks->stateAtBS.covariance(0).array().end()))     << " -- " <<  &(*(tracks->stateAtBS.covariance(31721).array().begin())) << std::endl;
    //std::cout << " -3-> " << tracks->stateAtBS.state(0).data()           << " -- " << tracks->stateAtBS.state(1).data()          << " -- " <<  tracks->stateAtBS.state(31721).data() << std::endl;
    //std::cout << " -4-> " << tracks->stateAtBS.covariance(0).data()      << " -- " << tracks->stateAtBS.covariance(1).data()     << " -- " <<  tracks->stateAtBS.covariance(31721).data() << std::endl;
    //std::cout << " -2-> " << tracks->stateAtBS.covariance(0).array() << " -- " << tracks->stateAtBS.covariance(1).array() << " -- " << tracks->stateAtBS.covariance(31728).array() << std::endl;
    //std::cout << " -1-> " << &(tracks->stateAtBS.state(0)(0))      << " -- " << &(tracks->stateAtBS.state(0)(1))      << " -- " << &(tracks->stateAtBS.state(0)(4)) << std::endl;
    //std::cout << " -2-> " << &(tracks->stateAtBS.covariance(0)(0)) << " -- " << &(tracks->stateAtBS.covariance(0)(1)) << " -- " << &(tracks->stateAtBS.covariance(0)(14)) << std::endl;
    //std::cout << " -2-> " << (tracks->stateAtBS.state.data())      << std::endl;
    //std::cout << " -2-> " << &(tracks->stateAtBS.covariance(0).data()) << " -- " << &(tracks->stateAtBS.covariance(1).data()) << " -- " << &(tracks->stateAtBS.covariance(31728).data()) << std::endl;
    //std::cout << " -1-> " << &(tracks->stateAtBS.state(0)) << " -- " << &(tracks->stateAtBS.state(1)) << " -- " << &(tracks->stateAtBS.state(31728)) << std::endl;
    //std::cout << " -2-> " << &(tracks->stateAtBS.covariance(0)) << " -- " << &(tracks->stateAtBS.covariance(1)) << " -- " << &(tracks->stateAtBS.covariance(31728)) << std::endl;
    //std::cout << " -3-> " << tracks->stateAtBS(0) << " -- " << tracks->stateAtBS(1) << " -- " << tracks->stateAtBS(31728) << std::endl;
    //std::memcpy(output_ + pCount,tracks->stateAtBS.data(),nTracks*sizeof(uint8_t)); pCount+=nTracks;
    //std::cout << " N Tracks 4 --- " << nTracks << std::endl;
  }
    {
      static constexpr uint32_t MAXTRACKS = 32 * 1024;
      static constexpr uint32_t MAXVTX = 1024;
      auto const& vertices = iEvent.get(vertexToken_);
      uint32_t nvtx = vertices->nvFinal;
      output_[pCount] = nvtx; pCount++;
      std::memcpy(output_ + pCount,vertices->idv    ,MAXTRACKS*sizeof(int16_t));   pCount+=MAXTRACKS;
      std::memcpy(output_ + pCount,vertices->zv     ,MAXVTX*sizeof(float));        pCount+=MAXVTX;
      std::memcpy(output_ + pCount,vertices->wv     ,MAXVTX*sizeof(float));        pCount+=MAXVTX;
      std::memcpy(output_ + pCount,vertices->chi2   ,MAXVTX*sizeof(float));        pCount+=MAXVTX;
      std::memcpy(output_ + pCount,vertices->ptv2   ,MAXVTX*sizeof(float));        pCount+=MAXVTX;
      std::memcpy(output_ + pCount,vertices->ndof   ,MAXTRACKS*sizeof(int32_t));      pCount+=MAXTRACKS;
      std::memcpy(output_ + pCount,vertices->sortInd,MAXVTX*sizeof(uint16_t));     pCount+=MAXVTX;
    }
    std::cout << "----> " << pCount*4 << " -- 8146596 " <<std::endl;
    ++allEvents;
    if (ok) {
      ++goodEvents;
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

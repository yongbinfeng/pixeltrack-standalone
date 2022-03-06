#include "CUDACore/HostProduct.h"
#include "CUDACore/Product.h"
#include "CUDACore/ScopedContext.h"
#include "CUDADataFormats/PixelTrackHeterogeneous.h"
#include "CUDADataFormats/SiPixelClustersCUDA.h"
#include "CUDADataFormats/SiPixelDigisCUDA.h"
#include "CUDADataFormats/ZVertexHeterogeneous.h"
#include "DataFormats/SiPixelErrorsSoA.h"
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

class CountValidatorSimple : public edm::EDProducerExternalWork {
    public:
        explicit CountValidatorSimple(edm::ProductRegistry& reg);
        int8_t* getOutput();
        void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
        void acquire(const edm::Event& iEvent,const edm::EventSetup& iSetup,edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
        uint64_t getSize();

        using HMSstorage = HostProduct<uint32_t[]>;
        using OutputStorage = HostProduct<int8_t[]>;
        using SizeStorage   = HostProduct<uint64_t[]>;

    private:
        void endJob() override;

        edm::EDGetTokenT<HMSstorage> hitsToken_;
        edm::EDGetTokenT<cms::cuda::Product<SiPixelDigisCUDA>> digiToken_;
        edm::EDGetTokenT<SiPixelErrorsSoA> digiErrorToken_;
        edm::EDGetTokenT<cms::cuda::Product<SiPixelClustersCUDA>> clusterToken_;
        edm::EDGetTokenT<PixelTrackHeterogeneous> trackToken_;
        edm::EDGetTokenT<ZVertexHeterogeneous> vertexToken_;
        const edm::EDPutTokenT<OutputStorage> outputToken_;
        const edm::EDPutTokenT<SizeStorage> sizeToken_;

        //uint64_t size_;
        //int8_t* output_;
        bool suppressDigis_;
        bool suppressTracks_;

        cms::cuda::host::unique_ptr<uint16_t[]> store_;
        uint32_t nDigis_;
};

CountValidatorSimple::CountValidatorSimple(edm::ProductRegistry& reg)
    : hitsToken_(reg.consumes<HMSstorage>()),
    digiToken_(reg.consumes<cms::cuda::Product<SiPixelDigisCUDA>>()),
    digiErrorToken_(reg.consumes<SiPixelErrorsSoA>()),
    clusterToken_(reg.consumes<cms::cuda::Product<SiPixelClustersCUDA>>()),
    trackToken_(reg.consumes<PixelTrackHeterogeneous>()),
    vertexToken_(reg.consumes<ZVertexHeterogeneous>()),
    outputToken_(reg.produces<OutputStorage>()),
    sizeToken_(reg.produces<SizeStorage>()),
    suppressDigis_(false),
    suppressTracks_(true){
        //output_ = new int8_t[7200000];
    }

void CountValidatorSimple::acquire(const edm::Event& iEvent,
        const edm::EventSetup& iSetup,
        edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
    cms::cuda::ScopedContextAcquire ctx{iEvent.streamID(), std::move(waitingTaskHolder)};
    const auto& gpuDigis = ctx.get(iEvent, digiToken_);  
    nDigis_ = gpuDigis.nDigis();
    store_  = gpuDigis.copyAllToHostAsync(ctx.stream());
}

void CountValidatorSimple::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    //std::cout << " calling produce function " << std::endl;
    auto output_ = std::make_unique<int8_t[]>(7500000);
    auto size_   = std::make_unique<uint64_t[]>(1);
    unsigned int pCount = 0;
    {
        uint32_t  nErrors_;
        uint32_t  nHits_;

        auto const& hits = iEvent.get(hitsToken_);
        nHits_ = hits.get()[0];
        if(nHits_ > 35000) std::cout << "----> Too many Hits #Hits  " << nHits_ << " Max! 35000 " << std::endl;
        if(nHits_ > 35000) nHits_ = 35000;
        std::memcpy(output_.get()+pCount,&nHits_,sizeof(uint32_t)); pCount += 4;
        static const unsigned maxNumModules = 2000;
        std::memcpy(output_.get() + pCount,hits.get()+1,    (maxNumModules+1)*sizeof(uint32_t)); pCount += 4*(maxNumModules+1);
        std::memcpy(output_.get() + pCount,hits.get()+maxNumModules+2, (4*nHits_)*sizeof(float)); pCount+=4*(4*nHits_);

        //std::cout << "nhits " << nHits_ << std::endl;

        //auto const& pdigis = iEvent.get(digiToken_);
        //cms::cuda::ScopedContextProduce ctx{pdigis};
        //auto const& digis = ctx.get(iEvent, digiToken_);
        //nDigis_    = digis.nDigis();
        if(nDigis_ > 150000) std::cout << "----> Too many Digis #Digis  " << nDigis_ << " Max! " << nDigis_ << std::endl;
        if(nDigis_ > 150000) nDigis_ = 150000;
        //pdigi_     = digis.pdigiToHostAsync(ctx.stream());
        //rawIdArr_  = digis.rawIdArrToHostAsync(ctx.stream());
        //adc_       = digis.adcToHostAsync(ctx.stream());
        //clus_      = digis.clusToHostAsync(ctx.stream());
        if(!suppressDigis_) {
            //std::cout << "co digis " << std::endl;
            auto tmp_view = SiPixelDigisCUDASOAView(store_, nDigis_, SiPixelDigisCUDASOAView::StorageLocationHost::kMAX);
            std::memcpy(output_.get() + pCount,&nDigis_,sizeof(uint32_t)); pCount += 4;
            std::memcpy(output_.get() + pCount,tmp_view.pdigi()   ,nDigis_*sizeof(uint32_t)); pCount+=4*nDigis_;
            std::memcpy(output_.get() + pCount,tmp_view.rawIdArr(),nDigis_*sizeof(uint32_t)); pCount+=4*nDigis_;
            std::memcpy(output_.get() + pCount,tmp_view.adc()     ,nDigis_*sizeof(uint16_t)); pCount+=2*nDigis_;
            std::memcpy(output_.get() + pCount,tmp_view.clus()    ,nDigis_*sizeof(int32_t));  pCount+=4*nDigis_;

            //std::cout << "cp digis fine " << std::endl;
        } else { 
            uint32_t pOldDigi = 0;
            uint32_t pOldRawId = 0;
            uint16_t pADC      = 0;
            int32_t  pOldClus = 0;

            //uint32_t pdigi   [100000];
            //uint32_t rawIdArr[100000];
            //uint16_t adc     [100000];
            //int32_t  clus    [100000];

            //uint32_t pdigiArrs[150000];
            //uint32_t rawIdArrs[150000];
            //uint16_t adcArrs  [150000];
            //int32_t  clusArrs [150000];

            uint32_t* pdigi = new uint32_t[100000];
            uint32_t* rawIdArr = new uint32_t[100000];
            uint16_t* adc = new uint16_t[100000];
            int32_t* clus = new int32_t[100000];

            uint32_t* pdigiArrs = new uint32_t[150000];
            uint32_t* rawIdArrs = new uint32_t[150000];
            uint16_t* adcArrs   = new uint16_t[150000];
            int32_t*  clusArrs  = new int32_t [150000];

            auto tmp_view = SiPixelDigisCUDASOAView(store_, nDigis_, SiPixelDigisCUDASOAView::StorageLocationHost::kMAX);
            std::memcpy(pdigiArrs,tmp_view.pdigi()    ,nDigis_*sizeof(uint32_t)); 
            std::memcpy(rawIdArrs,tmp_view.rawIdArr(),nDigis_*sizeof(uint32_t)); 
            std::memcpy(adcArrs,  tmp_view.adc()     ,nDigis_*sizeof(uint16_t)); 
            std::memcpy(clusArrs, tmp_view.clus()    ,nDigis_*sizeof(int32_t)); 

            uint32_t pDigiCount=0;
            for(uint32_t i0 = 0; i0 < nDigis_+1; i0++) { 
                if(pOldRawId != rawIdArrs[i0] || pOldClus != clusArrs[i0] ) {
                    if(i0 > 0) { 
                        pdigi[pDigiCount]    = pOldDigi; 
                        rawIdArr[pDigiCount] = pOldRawId;
                        adc [pDigiCount] = pADC;
                        clus[pDigiCount] = pOldClus;
                        pDigiCount++;
                    }
                    if (i0 < nDigis_) { 
                        pOldDigi  = pdigiArrs[i0];
                        pOldRawId = rawIdArrs[i0];
                        pADC      = adcArrs[i0];
                        pOldClus  = clusArrs[i0];
                    }
                } else { 
                    if(adcArrs[i0] + pADC < 65536) pADC += tmp_view.adc()[i0];
                }
            }
            nDigis_ = pDigiCount;
            std::memcpy(output_.get() + pCount,&nDigis_,sizeof(uint32_t)); pCount += 4;
            std::memcpy(output_.get() + pCount,pdigi   ,nDigis_*sizeof(uint32_t)); pCount+=4*nDigis_;
            std::memcpy(output_.get() + pCount,rawIdArr,nDigis_*sizeof(uint32_t)); pCount+=4*nDigis_;
            std::memcpy(output_.get() + pCount,adc     ,nDigis_*sizeof(uint16_t)); pCount+=2*nDigis_;
            std::memcpy(output_.get() + pCount,clus    ,nDigis_*sizeof(int32_t));  pCount+=4*nDigis_;

            delete [] pdigi;
            delete [] rawIdArr;
            delete [] adc;
            delete [] clus;

            delete [] pdigiArrs;
            delete [] rawIdArrs;
            delete [] adcArrs;
            delete [] clusArrs;
        }
        //std::cout << "nDigis_ " << nDigis_ << std::endl;
        auto const& pdigiErrors = iEvent.get(digiErrorToken_);
        //std::cout << "failed to get pdigiErros " << std::endl;
        nErrors_ = pdigiErrors.size();
        //if(nErrors_ > 0) std::cout << " ---> Found Errors " << nErrors_ << std::endl;
        //std::cout << "cp errors " << std::endl;
        std::memcpy(output_.get()+pCount,&nErrors_,sizeof(uint32_t)); pCount += 4;
        // cpp compiler would 'pad' the struct class
        std::memcpy(output_.get()+pCount,pdigiErrors.errorVector().data(),12*nErrors_);     pCount += 12*nErrors_;
        //if (nErrors_ > 0)
        //    std::cout << "SiPixelErrorCompact Struct size " << sizeof(pdigiErrors.errorVector()[0]) << std::endl;
        // check digiErrors
        //for (int ierr = 0; ierr < (int)nErrors_; ierr++) {
        //    SiPixelErrorCompact err = pdigiErrors.errorVector().data()[ierr];
        //    std::cout << "err rawId " << err.rawId << " word " << err.word << " errorType " << int(err.errorType) << " fedId " << int(err.fedId) << std::endl;
        //}
        //std::cout << "nErrors " << nErrors_ << std::endl;
        //std::cout << "cp erros fine " << std::endl;
    }
    //std::cout << "fine before suppressTracks_" << std::endl;
    if( suppressTracks_ ) {
        auto const& tracks = iEvent.get(trackToken_);
        auto const& vertices = iEvent.get(vertexToken_);
        uint32_t nTracks = tracks->stride();//25000;

        unsigned int pSize = 5000;
        uint32_t pInt = 0;
        float *chi2 = new float[pSize];
        int8_t *nLayers = new int8_t[pSize];
        uint8_t *quality = new uint8_t[pSize];
        float *eta = new float[pSize];
        float* pt = new float[pSize];
        float* stateAtBS = new float[pSize*5];
        float* stateAtBSCov = new float[pSize*15];
        uint32_t *hitsRaw = new uint32_t[pSize*5];
        int32_t* hitsOff = new int32_t[pSize+1];
        uint32_t *detsRaw = new uint32_t[pSize*5];
        int32_t  *detsOff = new int32_t[pSize+1];
        int16_t *idv = new int16_t[pSize];
        //float   chi2   [pSize];
        //int8_t  nLayers[pSize];
        //uint8_t quality[pSize];
        //float   eta    [pSize];
        //float   pt     [pSize];
        //float stateAtBS[pSize*5];
        //float stateAtBSCov[pSize*15];
        //uint32_t *hitsRaw = new uint32_t[pSize*5];
        //int32_t  hitsOff[pSize+1];
        //uint32_t *detsRaw = new uint32_t[pSize*5];
        //int32_t  detsOff[pSize+1];
        //uint32_t pInt = 0;
        //int16_t idv[pSize];
        uint32_t pHitCheck = 0;
        uint32_t pDetCheck = 0; 
        for(unsigned  i0 = 0; i0 < nTracks; i0++) {
            auto nHits = tracks->nHits(i0);
            if (nHits == 0)
                break;
            if(tracks->quality(i0) > pixelTrack::Quality::dup && tracks->nHits(i0) > 0 && pInt < pSize-1) {
                chi2[pInt]    = tracks->chi2(i0);
                nLayers[pInt] = tracks->nLayers(i0);
                quality[pInt] = uint8_t(tracks->quality(i0));
                eta[pInt]     = tracks->eta(i0);
                pt[pInt]      = tracks->pt(i0);
                for(int i1 = 0; i1 < 5; i1++) { 
                    stateAtBS[i1*pSize + pInt] = tracks->stateAtBS.state(i0)(i1);
                }
                for(int i1 = 0; i1 < 15; i1++) { 
                    stateAtBSCov[i1*pSize + pInt] = tracks->stateAtBS.covariance(i0)(i1);
                }
                uint32_t pHit = tracks->hitIndices.size(i0);
                uint32_t pDet = tracks->detIndices.size(i0);
                std::memcpy(hitsRaw + pHitCheck,tracks->hitIndices.begin() + tracks->hitIndices.off.data()[i0],sizeof(uint32_t)*pHit);
                std::memcpy(detsRaw + pDetCheck,tracks->detIndices.begin() + tracks->detIndices.off.data()[i0],sizeof(uint32_t)*pDet);      
                hitsOff[pInt] = pHitCheck;
                detsOff[pInt] = pDetCheck;
                pHitCheck    += pHit;
                pDetCheck    += pDet;
                idv[pInt]     = vertices->idv[i0];
                pInt++;
            }
        }
        hitsOff[pInt]  = pHitCheck;
        detsOff[pInt]  = pDetCheck;
        nTracks = pInt;
        //std::cout << "nTracks " << nTracks << std::endl;
        //if (nTracks > 0 ){
        //    std::cout << "track 0 pt " << pt[0] << " eta " << eta[0] << " chi2 " << chi2[0] << " nLayers " << int(nLayers[0]) << " quality " << int(quality[0]) << std::endl;
        //}
        std::memcpy(output_.get() + pCount,&nTracks,sizeof(uint32_t)); pCount += 4;
        std::memcpy(output_.get() + pCount,chi2,        nTracks*sizeof(float));             pCount+=4*nTracks;
        std::memcpy(output_.get() + pCount,nLayers,     nTracks*sizeof(int8_t));            pCount+=nTracks;
        std::memcpy(output_.get() + pCount,quality,     nTracks*sizeof(uint8_t));           pCount+=nTracks;
        std::memcpy(output_.get() + pCount,eta,         nTracks*sizeof(float));             pCount+=4*nTracks;
        std::memcpy(output_.get() + pCount,pt,          nTracks*sizeof(float));             pCount+=4*nTracks;
        for(unsigned i1 = 0; i1 < 5; i1++) { 
            std::memcpy(output_.get() + pCount,stateAtBS+i1*pSize,   nTracks*sizeof(float));  pCount+=(4*nTracks);
        }
        for(unsigned i1 = 0; i1 < 15; i1++) { 
            std::memcpy(output_.get() + pCount,stateAtBSCov+i1*pSize,nTracks*sizeof(float));  pCount+=(4*nTracks);
        }
        std::memcpy(output_.get() + pCount,hitsRaw,    5*nTracks*sizeof(uint32_t));         pCount+=4*nTracks*5;
        std::memcpy(output_.get() + pCount,hitsOff,    (nTracks+1)*sizeof(int32_t));        pCount+=4*(nTracks+1);
        std::memcpy(output_.get() + pCount,detsRaw,    5*nTracks*sizeof(uint32_t));         pCount+=4*nTracks*5;
        std::memcpy(output_.get() + pCount,detsOff,    (nTracks+1)*sizeof(int32_t));        pCount+=4*(nTracks+1);

        delete [] chi2;
        delete [] nLayers;
        delete [] quality;
        delete [] eta;
        delete [] pt;
        delete [] stateAtBS;
        delete [] stateAtBSCov;
        delete [] hitsRaw;
        delete [] hitsOff;
        delete [] detsRaw;
        delete [] detsOff;
        delete [] idv;

        uint32_t nVtx = vertices->nvFinal;
        //std::cout << "nVtx " << nVtx << std::endl;
        std::memcpy(output_.get() + pCount,&(vertices->nvFinal),sizeof(uint32_t)); pCount += 4;
        std::memcpy(output_.get() + pCount,idv              ,nTracks*sizeof(int16_t));   pCount+=2*nTracks;
        std::memcpy(output_.get() + pCount,vertices->zv     ,nVtx*sizeof(float));        pCount+=4*nVtx;
        std::memcpy(output_.get() + pCount,vertices->wv     ,nVtx*sizeof(float));        pCount+=4*nVtx;
        std::memcpy(output_.get() + pCount,vertices->chi2   ,nVtx*sizeof(float));        pCount+=4*nVtx;
        std::memcpy(output_.get() + pCount,vertices->ptv2   ,nVtx*sizeof(float));        pCount+=4*nVtx;
        std::memcpy(output_.get() + pCount,vertices->ndof   ,nVtx*sizeof(int32_t));      pCount+=4*nVtx;
        std::memcpy(output_.get() + pCount,vertices->sortInd,nVtx*sizeof(uint16_t));     pCount+=2*nVtx;
    } else { 

        auto const& tracks = iEvent.get(trackToken_);
        uint32_t nTracks = tracks->stride();
        std::memcpy(output_.get() + pCount,&nTracks,sizeof(uint32_t)); pCount += 4;
        std::memcpy(output_.get() + pCount,tracks->chi2.data()     ,nTracks*sizeof(float));                  pCount+=4*nTracks;
        std::memcpy(output_.get() + pCount,tracks->qualityData()   ,nTracks*sizeof(uint8_t));                pCount+=nTracks;
        std::memcpy(output_.get() + pCount,tracks->eta.data()      ,nTracks*sizeof(float));                  pCount+=4*nTracks;
        std::memcpy(output_.get() + pCount,tracks->pt.data()       ,nTracks*sizeof(float));                  pCount+=4*nTracks;
        std::memcpy(output_.get() + pCount,tracks->stateAtBS.state(0).data(),     nTracks*sizeof(float)*5);  pCount+=(4*nTracks*5);
        std::memcpy(output_.get() + pCount,tracks->stateAtBS.covariance(0).data(),nTracks*sizeof(float)*15); pCount+=(4*nTracks*15);
        std::memcpy(output_.get() + pCount,tracks->hitIndices.begin(),5*nTracks*sizeof(uint32_t));     pCount+=4*nTracks*5;
        std::memcpy(output_.get() + pCount,tracks->hitIndices.off.data(),    (nTracks+1)*sizeof(int32_t));        pCount+=4*(nTracks+1);
        std::memcpy(output_.get() + pCount,tracks->detIndices.begin(),5*nTracks*sizeof(uint32_t));     pCount+=4*nTracks*5;
        std::memcpy(output_.get() + pCount,tracks->detIndices.off.data(),    (nTracks+1)*sizeof(int32_t));        pCount+=4*(nTracks+1);

        auto const& vertices = iEvent.get(vertexToken_);
        static constexpr uint32_t MAXVTX = 1024;
        //std::memcpy(output_.get() + pCount,&(vertices->nvFinal),sizeof(uint32_t)); pCount += 4;
        //std::memcpy(output_.get() + pCount,vertices->idv    ,nTracks*sizeof(int16_t));     pCount+=2*nTracks;
        //std::memcpy(output_.get() + pCount,vertices->zv     ,MAXVTX*sizeof(float));        pCount+=4*MAXVTX;
        //std::memcpy(output_.get() + pCount,vertices->wv     ,MAXVTX*sizeof(float));        pCount+=4*MAXVTX;
        //std::memcpy(output_.get() + pCount,vertices->chi2   ,MAXVTX*sizeof(float));        pCount+=4*MAXVTX;
        //std::memcpy(output_.get() + pCount,vertices->ptv2   ,MAXVTX*sizeof(float));        pCount+=4*MAXVTX;
        //std::memcpy(output_.get() + pCount,vertices->ndof   ,MAXVTX*sizeof(int32_t));      pCount+=4*MAXVTX;
        //std::memcpy(output_.get() + pCount,vertices->sortInd,MAXVTX*sizeof(uint16_t));     pCount+=2*MAXVTX;
        uint32_t nVtx = vertices->nvFinal;
        //std::cout << "nVtx " << nVtx << std::endl;
        std::memcpy(output_.get() + pCount,&(vertices->nvFinal),sizeof(uint32_t)); pCount += 4;
        std::memcpy(output_.get() + pCount,vertices->idv    ,nTracks*sizeof(int16_t));   pCount+=2*nTracks;
        std::memcpy(output_.get() + pCount,vertices->zv     ,nVtx*sizeof(float));        pCount+=4*nVtx;
        std::memcpy(output_.get() + pCount,vertices->wv     ,nVtx*sizeof(float));        pCount+=4*nVtx;
        std::memcpy(output_.get() + pCount,vertices->chi2   ,nVtx*sizeof(float));        pCount+=4*nVtx;
        std::memcpy(output_.get() + pCount,vertices->ptv2   ,nVtx*sizeof(float));        pCount+=4*nVtx;
        std::memcpy(output_.get() + pCount,vertices->ndof   ,nVtx*sizeof(int32_t));      pCount+=4*nVtx;
        std::memcpy(output_.get() + pCount,vertices->sortInd,nVtx*sizeof(uint16_t));     pCount+=2*nVtx;
    }
    //std::cout << " ---> CVS size " << pCount << std::endl;
    //size_ = pCount;
    iEvent.emplace(outputToken_, std::move(output_));
    size_.get()[0] = pCount;
    iEvent.emplace(sizeToken_, std::move(size_));
    store_.reset();
}
/*
   int8_t* CountValidatorSimple::getOutput() {
   return output_;
   }
   uint64_t CountValidatorSimple::getSize() {
   return size_;
   }
   */
void CountValidatorSimple::endJob() {
}

DEFINE_FWK_MODULE(CountValidatorSimple);

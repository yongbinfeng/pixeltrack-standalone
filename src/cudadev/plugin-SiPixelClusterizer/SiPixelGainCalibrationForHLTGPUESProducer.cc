#include "CondFormats/SiPixelGainCalibrationForHLTGPU.h"
#include "CondFormats/SiPixelGainForHLTonGPU.h"
#include "Framework/ESProducer.h"
#include "Framework/EventSetup.h"
#include "Framework/ESPluginFactory.h"

#include <fstream>
#include <iostream>
#include <memory>

class SiPixelGainCalibrationForHLTGPUESProducer : public edm::ESProducer {
public:
  explicit SiPixelGainCalibrationForHLTGPUESProducer(std::filesystem::path const& datadir) : data_(datadir) {}
  void produce(edm::EventSetup& eventSetup);

private:
  std::filesystem::path data_;
};

void SiPixelGainCalibrationForHLTGPUESProducer::produce(edm::EventSetup& eventSetup) {
  std::cout << "reading gain.bin" << std::endl;
  std::ifstream in(data_ / "gain.bin", std::ios::binary);
  in.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
  SiPixelGainForHLTonGPU gain;
  in.read(reinterpret_cast<char*>(&gain), sizeof(SiPixelGainForHLTonGPU));
  std::cout << "reading SiPixelGainForHLTonGPU class fine; size " << sizeof(SiPixelGainForHLTonGPU) << std::endl;
  unsigned int nbytes;
  in.read(reinterpret_cast<char*>(&nbytes), sizeof(unsigned int));
  std::cout << "reading nbytes fine; nbytes " << nbytes << std::endl;
  std::vector<char> gainData(nbytes);
  in.read(gainData.data(), nbytes);
  std::cout << "reading gainData fine " << std::endl;
  std::cout << " -gain- " << nbytes << " -- " <<  gain.pedPrecision_ << " -- " << int(gainData.data()[0]) << std::endl;
  eventSetup.put(std::make_unique<SiPixelGainCalibrationForHLTGPU>(gain, std::move(gainData)));
  std::cout << "done reading gain.bin" << std::endl;
}

DEFINE_FWK_EVENTSETUP_MODULE(SiPixelGainCalibrationForHLTGPUESProducer);

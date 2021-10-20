#ifndef DataFormats_SiPixelDigi_interface_SiPixelErrorsSoA_h
#define DataFormats_SiPixelDigi_interface_SiPixelErrorsSoA_h

#include "DataFormats/PixelErrors.h"

#include <cstdint>
#include <vector>

class SiPixelErrorsSoA {
public:
  SiPixelErrorsSoA() = default;
  explicit SiPixelErrorsSoA(size_t nErrors, const PixelErrorCompact *error)//, const PixelFormatterErrors *err)
    : error_(error, error + nErrors) {}
	//, formatterErrors_(err) {}
  ~SiPixelErrorsSoA() = default;

  auto size() const { return error_.size(); }

  //const PixelFormatterErrors *formatterErrors() const { return formatterErrors_; }

  const PixelErrorCompact &error(size_t i) const { return error_[i]; }

  const std::vector<PixelErrorCompact> &errorVector() const { return error_; }

private:
  std::vector<PixelErrorCompact> error_;
  //const PixelFormatterErrors *formatterErrors_ = nullptr;
};

#endif

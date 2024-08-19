

#pragma once

#include "decoders/RawDecoder.h" 
#include "io/Buffer.h"           
#include "tiff/TiffIFD.h"        
#include "tiff/TiffTag.h"        
#include <memory>                
#include <string>                
#include <utility>               

namespace rawspeed {

class Buffer;
class CameraMetaData;

class AbstractTiffDecoder : public RawDecoder {
protected:
TiffRootIFDOwner mRootIFD;

public:
AbstractTiffDecoder(TiffRootIFDOwner&& root, Buffer file)
: RawDecoder(file), mRootIFD(std::move(root)) {}

TiffIFD* getRootIFD() final { return mRootIFD.get(); }

inline bool checkCameraSupported(const CameraMetaData* meta, const TiffID& id,
const std::string& mode) {
return RawDecoder::checkCameraSupported(meta, id.make, id.model, mode);
}

using RawDecoder::setMetaData;

inline void setMetaData(const CameraMetaData* meta, const TiffID& id,
const std::string& mode, int iso_speed) {
setMetaData(meta, id.make, id.model, mode, iso_speed);
}

inline void setMetaData(const CameraMetaData* meta, const std::string& mode,
int iso_speed) {
setMetaData(meta, mRootIFD->getID(), mode, iso_speed);
}

inline void checkSupportInternal(const CameraMetaData* meta) override {
checkCameraSupported(meta, mRootIFD->getID(), "");
}

[[nodiscard]] const TiffIFD*
getIFDWithLargestImage(TiffTag filter = TiffTag::IMAGEWIDTH) const;
};

} 

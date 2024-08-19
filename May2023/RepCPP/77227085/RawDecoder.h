

#pragma once

#include "common/Common.h"   
#include "common/RawImage.h" 
#include "io/Buffer.h"       
#include "metadata/Camera.h" 
#include <cstdint>           
#include <string>            

namespace rawspeed {

class CameraMetaData;
class TiffIFD;

class RawDecoder {
public:



explicit RawDecoder(Buffer file);
virtual ~RawDecoder() = default;





void checkSupport(const CameraMetaData* meta);




RawImage decodeRaw();








void decodeMetaData(const CameraMetaData* meta);



virtual TiffIFD* getRootIFD() { return nullptr; }



RawImage mRaw;






bool failOnUnknown;



bool interpolateBadPixels;



bool applyStage1DngOpcodes;


bool applyCrop;





bool uncorrectedRawValues;


bool fujiRotate;

struct {

bool quadrantMultipliers = true;

explicit operator bool() const { return quadrantMultipliers ; }
} iiq;

bool noSamples = false;

protected:




virtual RawImage decodeRawInternal() = 0;
virtual void decodeMetaDataInternal(const CameraMetaData* meta) = 0;
virtual void checkSupportInternal(const CameraMetaData* meta) = 0;


static void askForSamples(const CameraMetaData* meta, const std::string& make,
const std::string& model, const std::string& mode);




bool checkCameraSupported(const CameraMetaData* meta, const std::string& make,
const std::string& model, const std::string& mode);



virtual void setMetaData(const CameraMetaData* meta, const std::string& make,
const std::string& model, const std::string& mode,
int iso_speed = 0);



void decodeUncompressed(const TiffIFD* rawIFD, BitOrder order) const;


Buffer mFile;






[[nodiscard]] virtual int getDecoderVersion() const = 0;


Hints hints;

struct RawSlice;
};

struct RawDecoder::RawSlice {
uint32_t h = 0;
uint32_t offset = 0;
uint32_t count = 0;
};

} 

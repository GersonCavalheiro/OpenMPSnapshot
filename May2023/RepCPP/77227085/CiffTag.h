

#pragma once

#include <initializer_list> 

namespace rawspeed {

enum class CiffTag {
NULL_TAG = 0x0000,
MAKEMODEL = 0x080a,
SHOTINFO = 0x102a,
WHITEBALANCE = 0x10a9,
SENSORINFO = 0x1031,
IMAGEINFO = 0x1810,
DECODERTABLE = 0x1835,
RAWDATA = 0x2005,
SUBIFD = 0x300a,
EXIF = 0x300b,
};

static constexpr std::initializer_list<CiffTag> CiffTagsWeCareAbout = {
CiffTag::DECODERTABLE,        CiffTag::MAKEMODEL, CiffTag::RAWDATA,
CiffTag::SENSORINFO,          CiffTag::SHOTINFO,  CiffTag::WHITEBALANCE,
static_cast<CiffTag>(0x0032), 
static_cast<CiffTag>(0x102c), 
};

} 

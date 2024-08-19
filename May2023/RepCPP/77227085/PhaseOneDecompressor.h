

#pragma once

#include "common/RawImage.h"                    
#include "decompressors/AbstractDecompressor.h" 
#include "io/ByteStream.h"                      
#include <utility>                              
#include <vector>                               

namespace rawspeed {

struct PhaseOneStrip {
int n;
ByteStream bs;

PhaseOneStrip() = default;
PhaseOneStrip(int block, ByteStream bs_) : n(block), bs(bs_) {}
};

class PhaseOneDecompressor final : public AbstractDecompressor {
RawImage mRaw;

std::vector<PhaseOneStrip> strips;

void decompressStrip(const PhaseOneStrip& strip) const;

void decompressThread() const noexcept;

void prepareStrips();

public:
PhaseOneDecompressor(const RawImage& img,
std::vector<PhaseOneStrip>&& strips_);

void decompress() const;
};

} 

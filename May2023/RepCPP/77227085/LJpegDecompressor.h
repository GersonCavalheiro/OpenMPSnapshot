

#pragma once

#include "adt/Point.h"               
#include "codes/PrefixCodeDecoder.h" 
#include "common/RawImage.h"         
#include "io/ByteStream.h"           
#include <array>                     
#include <cstdint>                   
#include <functional>                
#include <stddef.h>                  
#include <tuple>                     
#include <vector>                    

namespace rawspeed {


class LJpegDecompressor final {
public:
struct Frame {
const int cps;
const iPoint2D dim;
};
struct PerComponentRecipe {
const PrefixCodeDecoder<>& ht;
const uint16_t initPred;
};

private:
RawImage mRaw;
ByteStream input;

const iRectangle2D imgFrame;

const Frame frame;
const std::vector<PerComponentRecipe> rec;

int fullBlocks = 0;
int trailingPixels = 0;

template <int N_COMP, size_t... I>
[[nodiscard]] std::array<std::reference_wrapper<const PrefixCodeDecoder<>>,
N_COMP>
getPrefixCodeDecodersImpl(std::index_sequence<I...> ) const;

template <int N_COMP>
[[nodiscard]] std::array<std::reference_wrapper<const PrefixCodeDecoder<>>,
N_COMP>
getPrefixCodeDecoders() const;

template <int N_COMP>
[[nodiscard]] std::array<uint16_t, N_COMP> getInitialPreds() const;

template <int N_COMP, bool WeirdWidth = false> void decodeN();

public:
LJpegDecompressor(const RawImage& img, iRectangle2D imgFrame, Frame frame,
std::vector<PerComponentRecipe> rec, ByteStream bs);

void decode();
};

} 

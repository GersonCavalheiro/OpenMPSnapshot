

#pragma once

#include "adt/Invariant.h"                
#include "adt/Point.h"                    
#include "adt/iterator_range.h"           
#include "codes/DummyPrefixCodeDecoder.h" 
#include "codes/PrefixCodeDecoder.h"      
#include "common/RawImage.h"              
#include "common/RawspeedException.h"     
#include "decoders/RawDecoderException.h" 
#include "io/ByteStream.h"                
#include <array>                          
#include <cassert>                        
#include <cstddef>                        
#include <cstdint>                        
#include <functional>                     
#include <iterator>                       
#include <tuple>                          
#include <utility>                        
#include <vector>                         

namespace rawspeed {

class ByteStream;
class RawImage;
struct Cr2SliceWidthIterator;
struct Cr2SliceIterator;
struct Cr2OutputTileIterator;
class Cr2VerticalOutputStripIterator;

class Cr2SliceWidths {
int numSlices = 0;
int sliceWidth = 0;
int lastSliceWidth = 0;

friend class Cr2LJpegDecoder;
friend struct Cr2SliceWidthIterator;

template <typename PrefixCodeDecoder> friend class Cr2Decompressor;

public:
Cr2SliceWidths() = default;

Cr2SliceWidths(uint16_t numSlices_, uint16_t sliceWidth_,
uint16_t lastSliceWidth_)
: numSlices(numSlices_), sliceWidth(sliceWidth_),
lastSliceWidth(lastSliceWidth_) {
if (numSlices < 1)
ThrowRDE("Bad slice count: %u", numSlices);
}

[[nodiscard]] bool empty() const {
return 0 == numSlices && 0 == sliceWidth && 0 == lastSliceWidth;
}

[[nodiscard]] int widthOfSlice(int sliceId) const {
invariant(sliceId >= 0 && sliceId < numSlices);
if ((sliceId + 1) == numSlices)
return lastSliceWidth;
return sliceWidth;
}

[[nodiscard]] Cr2SliceWidthIterator begin() const;
[[nodiscard]] Cr2SliceWidthIterator end() const;
};

struct Cr2SliceWidthIterator final {
const Cr2SliceWidths& slicing;

int sliceId;

using iterator_category = std::input_iterator_tag;
using difference_type = std::ptrdiff_t;
using value_type = int;
using pointer = const value_type*;   
using reference = const value_type&; 

Cr2SliceWidthIterator(const Cr2SliceWidths& slicing_, int sliceId_)
: slicing(slicing_), sliceId(sliceId_) {
invariant(sliceId >= 0 && sliceId <= slicing.numSlices &&
"Iterator overflow");
}

value_type operator*() const {
invariant(sliceId >= 0 && sliceId < slicing.numSlices &&
"Iterator overflow");
return slicing.widthOfSlice(sliceId);
}
Cr2SliceWidthIterator& operator++() {
++sliceId;
return *this;
}
friend bool operator==(const Cr2SliceWidthIterator& a,
const Cr2SliceWidthIterator& b) {
invariant(&a.slicing == &b.slicing && "Comparing unrelated iterators.");
return a.sliceId == b.sliceId;
}
friend bool operator!=(const Cr2SliceWidthIterator& a,
const Cr2SliceWidthIterator& b) {
return !(a == b);
}
};

inline Cr2SliceWidthIterator Cr2SliceWidths::begin() const {
return {*this, 0};
}
inline Cr2SliceWidthIterator Cr2SliceWidths::end() const {
return {*this, numSlices};
}

template <typename PrefixCodeDecoder> class Cr2Decompressor final {
public:
struct PerComponentRecipe {
const PrefixCodeDecoder& ht;
const uint16_t initPred;
};

private:
const RawImage mRaw;
const std::tuple<int , int , int > format;
iPoint2D dim;
iPoint2D frame;
Cr2SliceWidths slicing;

const std::vector<PerComponentRecipe> rec;

const ByteStream input;

template <int N_COMP, size_t... I>
[[nodiscard]] std::array<std::reference_wrapper<const PrefixCodeDecoder>,
N_COMP>
getPrefixCodeDecodersImpl(std::index_sequence<I...> ) const;

template <int N_COMP>
[[nodiscard]] std::array<std::reference_wrapper<const PrefixCodeDecoder>,
N_COMP>
getPrefixCodeDecoders() const;

template <int N_COMP>
[[nodiscard]] std::array<uint16_t, N_COMP> getInitialPreds() const;

template <int N_COMP, int X_S_F, int Y_S_F> void decompressN_X_Y();

[[nodiscard]] iterator_range<Cr2SliceIterator> getSlices();
[[nodiscard]] iterator_range<Cr2OutputTileIterator> getAllOutputTiles();
[[nodiscard]] iterator_range<Cr2OutputTileIterator> getOutputTiles();
[[nodiscard]] iterator_range<Cr2VerticalOutputStripIterator>
getVerticalOutputStrips();

public:
Cr2Decompressor(
const RawImage& mRaw,
std::tuple<int , int , int > format,
iPoint2D frame, Cr2SliceWidths slicing,
std::vector<PerComponentRecipe> rec, ByteStream input);

void decompress();
};

extern template class Cr2Decompressor<PrefixCodeDecoder<>>;

} 

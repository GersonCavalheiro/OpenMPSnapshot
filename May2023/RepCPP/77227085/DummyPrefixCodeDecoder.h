

#pragma once

#include "codes/AbstractPrefixCode.h" 
#include "codes/HuffmanCode.h"        
#include "codes/PrefixCode.h"         
#include "io/BitStream.h"             
#include <cassert>                    
#include <cstdint>                    
#include <tuple>                      
#include <utility>                    
#include <vector>                     

namespace rawspeed {
class Buffer;

template <typename CodeTag = BaselineCodeTag>
class DummyPrefixCodeDecoder final {
public:
using Tag = CodeTag;
using Traits = CodeTraits<CodeTag>;

explicit DummyPrefixCodeDecoder(HuffmanCode<CodeTag> code) {}
explicit DummyPrefixCodeDecoder(PrefixCode<CodeTag> code) {}

private:
bool fullDecode = true;
bool fixDNGBug16 = false;

public:
void setup(bool fullDecode_, bool fixDNGBug16_) {
fullDecode = fullDecode_;
fixDNGBug16 = fixDNGBug16_;
}

[[nodiscard]] bool isFullDecode() const { return fullDecode; }

template <typename BIT_STREAM>
inline typename Traits::CodeValueTy decodeCodeValue(BIT_STREAM& bs) const {
static_assert(
BitStreamTraits<typename BIT_STREAM::tag>::canUseWithPrefixCodeDecoder,
"This BitStream specialization is not marked as usable here");
invariant(!fullDecode);
return decode<BIT_STREAM, false>(bs);
}

template <typename BIT_STREAM>
inline int decodeDifference(BIT_STREAM& bs) const {
static_assert(
BitStreamTraits<typename BIT_STREAM::tag>::canUseWithPrefixCodeDecoder,
"This BitStream specialization is not marked as usable here");
invariant(fullDecode);
return decode<BIT_STREAM, true>(bs);
}

template <typename BIT_STREAM, bool FULL_DECODE>
inline int decode(BIT_STREAM& bs) const {
static_assert(
BitStreamTraits<typename BIT_STREAM::tag>::canUseWithPrefixCodeDecoder,
"This BitStream specialization is not marked as usable here");
invariant(FULL_DECODE == fullDecode);

(void)bs;

return 0; 
}
};

} 

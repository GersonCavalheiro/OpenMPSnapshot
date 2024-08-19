

#ifndef IMPL
#error IMPL must be defined to one of rawspeeds huffman table implementations
#endif

#include "codes/BinaryPrefixTree.h"         
#include "codes/PrefixCodeDecoder.h"        
#include "codes/PrefixCodeDecoder/Common.h" 
#include "codes/PrefixCodeLUTDecoder.h"     
#include "codes/PrefixCodeLookupDecoder.h"  
#include "codes/PrefixCodeTreeDecoder.h"    
#include "codes/PrefixCodeVectorDecoder.h"  
#include "common/RawspeedException.h"       
#include "io/BitPumpJPEG.h"                 
#include "io/BitPumpMSB.h"                  
#include "io/BitPumpMSB32.h"                
#include "io/Buffer.h"                      
#include "io/ByteStream.h"                  
#include "io/Endianness.h"                  
#include <algorithm>                        
#include <cassert>                          
#include <cstdint>                          
#include <cstdio>                           
#include <initializer_list>                 
#include <vector>                           

namespace rawspeed {
struct BaselineCodeTag;
struct VC5CodeTag;
} 

template <typename Pump, bool IsFullDecode, typename HT>
static void workloop(rawspeed::ByteStream bs, const HT& ht) {
Pump bits(bs);
while (true)
ht.template decode<Pump, IsFullDecode>(bits);
}

template <typename Pump, typename HT>
static void checkPump(rawspeed::ByteStream bs, const HT& ht) {
if (ht.isFullDecode())
workloop<Pump, true>(bs, ht);
else
workloop<Pump, false>(bs, ht);
}

template <typename CodeTag> static void checkFlavour(rawspeed::ByteStream bs) {
#ifndef BACKIMPL
const auto ht = createPrefixCodeDecoder<rawspeed::IMPL<CodeTag>>(bs);
#else
const auto ht = createPrefixCodeDecoder<
rawspeed::IMPL<CodeTag, rawspeed::BACKIMPL<CodeTag>>>(bs);
#endif

switch (bs.getByte()) {
case 0:
checkPump<rawspeed::BitPumpMSB>(bs, ht);
break;
case 1:
checkPump<rawspeed::BitPumpMSB32>(bs, ht);
break;
case 2:
checkPump<rawspeed::BitPumpJPEG>(bs, ht);
break;
default:
ThrowRSE("Unknown bit pump");
}
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* Data, size_t Size);

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* Data, size_t Size) {
assert(Data);

try {
const rawspeed::Buffer b(Data, Size);
const rawspeed::DataBuffer db(b, rawspeed::Endianness::little);
rawspeed::ByteStream bs(db);

switch (bs.getByte()) {
case 0:
checkFlavour<rawspeed::BaselineCodeTag>(bs);
break;
case 1:
checkFlavour<rawspeed::VC5CodeTag>(bs);
break;
default:
ThrowRSE("Unknown flavor");
}
} catch (const rawspeed::RawspeedException&) {
return 0;
}

__builtin_unreachable();
}

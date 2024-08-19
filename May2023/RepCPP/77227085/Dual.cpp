

#ifndef IMPL0
#error IMPL0 must be defined to one of rawspeeds huffman table implementations
#endif
#ifndef IMPL1
#error IMPL1 must be defined to one of rawspeeds huffman table implementations
#endif

#include "codes/BinaryPrefixTree.h"         
#include "codes/PrefixCodeDecoder.h"        
#include "codes/PrefixCodeDecoder/Common.h" 
#include "codes/PrefixCodeLUTDecoder.h"     
#include "codes/PrefixCodeLookupDecoder.h"  
#include "codes/PrefixCodeTreeDecoder.h"    
#include "codes/PrefixCodeVectorDecoder.h"  
#include "io/BitPumpJPEG.h"                 
#include "io/BitPumpMSB.h"                  
#include "io/BitPumpMSB32.h"                
#include "io/Buffer.h"                      
#include "io/ByteStream.h"                  
#include "io/Endianness.h"                  
#include "io/IOException.h"                 
#include <algorithm>                        
#include <cassert>                          
#include <cstdint>                          
#include <cstdio>                           
#include <initializer_list>                 
#include <optional>                         
#include <vector>                           

namespace rawspeed {
struct BaselineCodeTag;
struct VC5CodeTag;
} 

template <typename Pump, bool IsFullDecode, typename HT0, typename HT1>
static void workloop(rawspeed::ByteStream bs0, rawspeed::ByteStream bs1,
const HT0& ht0, const HT1& ht1) {
Pump bits0(bs0);
Pump bits1(bs1);

while (true) {
int decoded0;
int decoded1;

bool failure0 = false;
bool failure1 = false;

try {
decoded1 = ht1.template decode<decltype(bits1), IsFullDecode>(bits1);
} catch (const rawspeed::IOException&) {
throw;
} catch (const rawspeed::RawspeedException&) {
failure1 = true;
}

try {
decoded0 = ht0.template decode<decltype(bits0), IsFullDecode>(bits0);
} catch (const rawspeed::IOException&) {
throw;
} catch (const rawspeed::RawspeedException&) {
failure0 = true;
}

assert(failure0 == failure1);

if (failure0 || failure1)
ThrowRSE("Failure detected");

(void)decoded0;
(void)decoded1;

assert(decoded0 == decoded1);
}
}

template <typename Pump, typename HT0, typename HT1>
static void checkPump(rawspeed::ByteStream bs0, rawspeed::ByteStream bs1,
const HT0& ht0, const HT1& ht1) {
assert(bs0.getPosition() == bs1.getPosition());
assert(ht0.isFullDecode() == ht1.isFullDecode());
if (ht0.isFullDecode())
workloop<Pump, true>(bs0, bs1, ht0, ht1);
else
workloop<Pump, false>(bs0, bs1, ht0, ht1);
}

template <typename CodeTag> static void checkFlavour(rawspeed::ByteStream bs) {
rawspeed::ByteStream bs0 = bs;
rawspeed::ByteStream bs1 = bs;

#ifndef BACKIMPL0
const auto ht0 = createPrefixCodeDecoder<rawspeed::IMPL0<CodeTag>>(bs0);
#else
const auto ht0 = createPrefixCodeDecoder<
rawspeed::IMPL0<CodeTag, rawspeed::BACKIMPL0<CodeTag>>>(bs0);
#endif

#ifndef BACKIMPL1
const auto ht1 = createPrefixCodeDecoder<rawspeed::IMPL1<CodeTag>>(bs1);
#else
const auto ht1 = createPrefixCodeDecoder<
rawspeed::IMPL1<CodeTag, rawspeed::BACKIMPL1<CodeTag>>>(bs1);
#endif

assert(bs0.getPosition() == bs1.getPosition());

bs1.skipBytes(1);
switch (bs0.getByte()) {
case 0:
checkPump<rawspeed::BitPumpMSB>(bs0, bs1, ht0, ht1);
break;
case 1:
checkPump<rawspeed::BitPumpMSB32>(bs0, bs1, ht0, ht1);
break;
case 2:
checkPump<rawspeed::BitPumpJPEG>(bs0, bs1, ht0, ht1);
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

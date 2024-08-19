

#include "decompressors/Cr2Decompressor.h"     
#include "codes/PrefixCodeDecoder.h"           
#include "decompressors/Cr2DecompressorImpl.h" 

namespace rawspeed {

template class Cr2Decompressor<PrefixCodeDecoder<>>;

} 

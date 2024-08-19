

#pragma once


#include "codes/PrefixCodeLUTDecoder.h" 


namespace rawspeed {

template <typename CodeTag = BaselineCodeTag>
using PrefixCodeDecoder =
PrefixCodeLUTDecoder<CodeTag, PrefixCodeLookupDecoder<CodeTag>>;




} 

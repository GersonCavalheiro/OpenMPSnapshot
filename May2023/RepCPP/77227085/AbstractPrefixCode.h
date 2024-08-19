

#pragma once

#include "adt/BitIterator.h"              
#include "adt/iterator_range.h"           
#include "common/Common.h"                
#include "decoders/RawDecoderException.h" 
#include <cstdint>                        
#include <type_traits>                    

namespace rawspeed {

template <typename CodeTag> struct CodeTraits final {



};

struct BaselineCodeTag;

template <> struct CodeTraits<BaselineCodeTag> final {
using CodeTy = uint16_t;
static constexpr int MaxCodeLenghtBits = 16;
static constexpr int MaxNumCodeValues = 162;

using CodeValueTy = uint8_t;
static constexpr int MaxCodeValueLenghtBits = 8;
static constexpr CodeValueTy MaxCodeValue = 255;

static constexpr int MaxDiffLengthBits = 5;
static constexpr CodeValueTy MaxDiffLength = 16;

static constexpr bool SupportsFullDecode = true;
};

struct VC5CodeTag;

template <> struct CodeTraits<VC5CodeTag> final {
using CodeTy = uint32_t;
static constexpr int MaxCodeLenghtBits = 26;
static constexpr int MaxNumCodeValues = 264;

using CodeValueTy = uint32_t;
static constexpr int MaxCodeValueLenghtBits = 19;
static constexpr CodeValueTy MaxCodeValue = 524287;

static constexpr int MaxDiffLengthBits = -1;     
static constexpr CodeValueTy MaxDiffLength = -1; 

static constexpr bool SupportsFullDecode = false;
};

template <typename CodeTag> struct CodeTraitsValidator final {
using Traits = CodeTraits<CodeTag>;

static_assert(std::is_integral<typename Traits::CodeTy>::value);
static_assert(std::is_unsigned<typename Traits::CodeTy>::value);
static_assert(std::is_same<typename Traits::CodeTy, uint16_t>::value ||
std::is_same<typename Traits::CodeTy, uint32_t>::value);

static_assert(Traits::MaxCodeLenghtBits > 0 &&
Traits::MaxCodeLenghtBits <=
bitwidth<typename Traits::CodeTy>());
static_assert(Traits::MaxCodeLenghtBits == 16 ||
Traits::MaxCodeLenghtBits == 26);

static_assert(Traits::MaxNumCodeValues > 0 &&
Traits::MaxNumCodeValues <=
((1ULL << Traits::MaxCodeLenghtBits) - 1ULL));
static_assert(Traits::MaxNumCodeValues == 162 ||
Traits::MaxNumCodeValues == 264);

static_assert(std::is_integral<typename Traits::CodeValueTy>::value);
static_assert(std::is_unsigned<typename Traits::CodeValueTy>::value);
static_assert(std::is_same<typename Traits::CodeValueTy, uint8_t>::value ||
std::is_same<typename Traits::CodeValueTy, uint32_t>::value);

static_assert(Traits::MaxCodeValueLenghtBits > 0 &&
Traits::MaxCodeValueLenghtBits <=
bitwidth<typename Traits::CodeValueTy>());
static_assert(Traits::MaxCodeValueLenghtBits == 8 ||
Traits::MaxCodeValueLenghtBits == 19);

static_assert(Traits::MaxCodeValue > 0 &&
Traits::MaxCodeValue <=
((1ULL << Traits::MaxCodeValueLenghtBits) - 1ULL));
static_assert(Traits::MaxCodeValue == 255 || Traits::MaxCodeValue == 524287);

static_assert(
std::is_same<decltype(Traits::SupportsFullDecode), const bool>::value);

static_assert(!Traits::SupportsFullDecode ||
(Traits::MaxDiffLengthBits > 0 &&
Traits::MaxDiffLengthBits <=
bitwidth<typename Traits::CodeValueTy>()));
static_assert(!Traits::SupportsFullDecode ||
(Traits::MaxDiffLengthBits == 5));

static_assert(!Traits::SupportsFullDecode ||
(Traits::MaxDiffLength > 0 &&
Traits::MaxDiffLength <=
((1ULL << Traits::MaxDiffLengthBits) - 1ULL)));
static_assert(!Traits::SupportsFullDecode || (Traits::MaxDiffLength == 16));

static constexpr bool validate() { return true; }
};

template <typename CodeTag> class AbstractPrefixCode {
public:
using Traits = CodeTraits<CodeTag>;
using CodeValueTy = typename Traits::CodeValueTy;
static_assert(CodeTraitsValidator<CodeTag>::validate());

struct CodeSymbol final {
typename Traits::CodeTy code; 
uint8_t code_len;             

CodeSymbol() = default;

CodeSymbol(typename Traits::CodeTy code_, uint8_t code_len_)
: code(code_), code_len(code_len_) {
assert(code_len > 0);
assert(code_len <= Traits::MaxCodeLenghtBits);
assert(code <= ((1U << code_len) - 1U));
}

[[nodiscard]] iterator_range<BitMSBIterator<typename Traits::CodeTy>>
getBitsMSB() const {
return {{code, code_len - 1}, {code, -1}};
}

static bool HaveCommonPrefix(const CodeSymbol& symbol,
const CodeSymbol& partial) {
assert(partial.code_len <= symbol.code_len);

const auto s0 = extractHighBits(symbol.code, partial.code_len,
symbol.code_len);
const auto s1 = partial.code;

return s0 == s1;
}

inline bool RAWSPEED_READONLY operator==(const CodeSymbol& other) const {
return code == other.code && code_len == other.code_len;
}
};

AbstractPrefixCode() = default;

explicit AbstractPrefixCode(std::vector<CodeValueTy> codeValues_)
: codeValues(std::move(codeValues_)) {
if (codeValues.empty())
ThrowRDE("Empty code alphabet?");
assert(
all_of(codeValues.begin(), codeValues.end(),
[](const CodeValueTy& v) { return v <= Traits::MaxCodeValue; }));
}

std::vector<CodeValueTy> codeValues;
};

} 

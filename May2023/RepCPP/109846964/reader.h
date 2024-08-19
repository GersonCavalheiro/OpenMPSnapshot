
#ifndef RAPIDJSON_READER_H_
#define RAPIDJSON_READER_H_



#include "rapidjson.h"
#include "encodings.h"
#include "internal/meta.h"
#include "internal/pow10.h"
#include "internal/stack.h"

#if defined(RAPIDJSON_SIMD) && defined(_MSC_VER)
#include <intrin.h>
#pragma intrinsic(_BitScanForward)
#endif
#ifdef RAPIDJSON_SSE42
#include <nmmintrin.h>
#elif defined(RAPIDJSON_SSE2)
#include <emmintrin.h>
#endif

#ifdef _MSC_VER
RAPIDJSON_DIAG_PUSH
RAPIDJSON_DIAG_OFF(4127)  
RAPIDJSON_DIAG_OFF(4702)  
#endif

#define RAPIDJSON_NOTHING 
#ifndef RAPIDJSON_PARSE_ERROR_EARLY_RETURN
#define RAPIDJSON_PARSE_ERROR_EARLY_RETURN(value) \
RAPIDJSON_MULTILINEMACRO_BEGIN \
if (HasParseError()) { return value; } \
RAPIDJSON_MULTILINEMACRO_END
#endif
#define RAPIDJSON_PARSE_ERROR_EARLY_RETURN_VOID \
RAPIDJSON_PARSE_ERROR_EARLY_RETURN(RAPIDJSON_NOTHING)


#ifndef RAPIDJSON_PARSE_ERROR_NORETURN
#define RAPIDJSON_PARSE_ERROR_NORETURN(parseErrorCode, offset) \
RAPIDJSON_MULTILINEMACRO_BEGIN \
RAPIDJSON_ASSERT(!HasParseError());  \
SetParseError(parseErrorCode, offset); \
RAPIDJSON_MULTILINEMACRO_END
#endif


#ifndef RAPIDJSON_PARSE_ERROR
#define RAPIDJSON_PARSE_ERROR(parseErrorCode, offset) \
RAPIDJSON_MULTILINEMACRO_BEGIN \
RAPIDJSON_PARSE_ERROR_NORETURN(parseErrorCode, offset); \
RAPIDJSON_PARSE_ERROR_EARLY_RETURN_VOID; \
RAPIDJSON_MULTILINEMACRO_END
#endif

#include "error/error.h" 

RAPIDJSON_NAMESPACE_BEGIN



enum ParseFlag {
kParseDefaultFlags = 0,         
kParseInsituFlag = 1,           
kParseValidateEncodingFlag = 2, 
kParseIterativeFlag = 4,        
kParseStopWhenDoneFlag = 8      
};





template<typename Encoding = UTF8<>, typename Derived = void>
struct BaseReaderHandler {
typedef typename Encoding::Ch Ch;

typedef typename internal::SelectIf<internal::IsSame<Derived, void>, BaseReaderHandler, Derived>::Type Override;

bool Default() { return true; }
bool Null() { return static_cast<Override&>(*this).Default(); }
bool Bool(bool) { return static_cast<Override&>(*this).Default(); }
bool Int(int) { return static_cast<Override&>(*this).Default(); }
bool Uint(unsigned) { return static_cast<Override&>(*this).Default(); }
bool Int64(int64_t) { return static_cast<Override&>(*this).Default(); }
bool Uint64(uint64_t) { return static_cast<Override&>(*this).Default(); }
bool Double(double) { return static_cast<Override&>(*this).Default(); }
bool String(const Ch*, SizeType, bool) { return static_cast<Override&>(*this).Default(); }
bool StartObject() { return static_cast<Override&>(*this).Default(); }
bool Key(const Ch* str, SizeType len, bool copy) { return static_cast<Override&>(*this).String(str, len, copy); }
bool EndObject(SizeType) { return static_cast<Override&>(*this).Default(); }
bool StartArray() { return static_cast<Override&>(*this).Default(); }
bool EndArray(SizeType) { return static_cast<Override&>(*this).Default(); }
};


namespace internal {

template<typename Stream, int = StreamTraits<Stream>::copyOptimization>
class StreamLocalCopy;

template<typename Stream>
class StreamLocalCopy<Stream, 1> {
public:
StreamLocalCopy(Stream& original) : s(original), original_(original) {}
~StreamLocalCopy() { original_ = s; }

Stream s;

private:
StreamLocalCopy& operator=(const StreamLocalCopy&) ;

Stream& original_;
};

template<typename Stream>
class StreamLocalCopy<Stream, 0> {
public:
StreamLocalCopy(Stream& original) : s(original) {}

Stream& s;

private:
StreamLocalCopy& operator=(const StreamLocalCopy&) ;
};

} 



template<typename InputStream>
void SkipWhitespace(InputStream& is) {
internal::StreamLocalCopy<InputStream> copy(is);
InputStream& s(copy.s);

while (s.Peek() == ' ' || s.Peek() == '\n' || s.Peek() == '\r' || s.Peek() == '\t')
s.Take();
}

#ifdef RAPIDJSON_SSE42
inline const char *SkipWhitespace_SIMD(const char* p) {
if (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')
++p;
else
return p;

const char* nextAligned = reinterpret_cast<const char*>((reinterpret_cast<size_t>(p) + 15) & ~15);
while (p != nextAligned)
if (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')
++p;
else
return p;

static const char whitespace[16] = " \n\r\t";
const __m128i w = _mm_loadu_si128((const __m128i *)&whitespace[0]);

for (;; p += 16) {
const __m128i s = _mm_load_si128((const __m128i *)p);
const unsigned r = _mm_cvtsi128_si32(_mm_cmpistrm(w, s, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK | _SIDD_NEGATIVE_POLARITY));
if (r != 0) {   
#ifdef _MSC_VER         
unsigned long offset;
_BitScanForward(&offset, r);
return p + offset;
#else
return p + __builtin_ffs(r) - 1;
#endif
}
}
}

#elif defined(RAPIDJSON_SSE2)

inline const char *SkipWhitespace_SIMD(const char* p) {
if (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')
++p;
else
return p;

const char* nextAligned = reinterpret_cast<const char*>((reinterpret_cast<size_t>(p) + 15) & ~15);
while (p != nextAligned)
if (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')
++p;
else
return p;

static const char whitespaces[4][17] = {
"                ",
"\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
"\r\r\r\r\r\r\r\r\r\r\r\r\r\r\r\r",
"\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t"};

const __m128i w0 = _mm_loadu_si128((const __m128i *)&whitespaces[0][0]);
const __m128i w1 = _mm_loadu_si128((const __m128i *)&whitespaces[1][0]);
const __m128i w2 = _mm_loadu_si128((const __m128i *)&whitespaces[2][0]);
const __m128i w3 = _mm_loadu_si128((const __m128i *)&whitespaces[3][0]);

for (;; p += 16) {
const __m128i s = _mm_load_si128((const __m128i *)p);
__m128i x = _mm_cmpeq_epi8(s, w0);
x = _mm_or_si128(x, _mm_cmpeq_epi8(s, w1));
x = _mm_or_si128(x, _mm_cmpeq_epi8(s, w2));
x = _mm_or_si128(x, _mm_cmpeq_epi8(s, w3));
unsigned short r = (unsigned short)~_mm_movemask_epi8(x);
if (r != 0) {   
#ifdef _MSC_VER         
unsigned long offset;
_BitScanForward(&offset, r);
return p + offset;
#else
return p + __builtin_ffs(r) - 1;
#endif
}
}
}

#endif 

#ifdef RAPIDJSON_SIMD
template<> inline void SkipWhitespace(InsituStringStream& is) { 
is.src_ = const_cast<char*>(SkipWhitespace_SIMD(is.src_));
}

template<> inline void SkipWhitespace(StringStream& is) {
is.src_ = SkipWhitespace_SIMD(is.src_);
}
#endif 



template <typename SourceEncoding, typename TargetEncoding, typename StackAllocator = CrtAllocator>
class GenericReader {
public:
typedef typename SourceEncoding::Ch Ch; 


GenericReader(StackAllocator* stackAllocator = 0, size_t stackCapacity = kDefaultStackCapacity) : stack_(stackAllocator, stackCapacity), parseResult_() {}


template <unsigned parseFlags, typename InputStream, typename Handler>
ParseResult Parse(InputStream& is, Handler& handler) {
if (parseFlags & kParseIterativeFlag)
return IterativeParse<parseFlags>(is, handler);

parseResult_.Clear();

ClearStackOnExit scope(*this);

SkipWhitespace(is);

if (is.Peek() == '\0') {
RAPIDJSON_PARSE_ERROR_NORETURN(kParseErrorDocumentEmpty, is.Tell());
RAPIDJSON_PARSE_ERROR_EARLY_RETURN(parseResult_);
}
else {
ParseValue<parseFlags>(is, handler);
RAPIDJSON_PARSE_ERROR_EARLY_RETURN(parseResult_);

if (!(parseFlags & kParseStopWhenDoneFlag)) {
SkipWhitespace(is);

if (is.Peek() != '\0') {
RAPIDJSON_PARSE_ERROR_NORETURN(kParseErrorDocumentRootNotSingular, is.Tell());
RAPIDJSON_PARSE_ERROR_EARLY_RETURN(parseResult_);
}
}
}

return parseResult_;
}


template <typename InputStream, typename Handler>
ParseResult Parse(InputStream& is, Handler& handler) {
return Parse<kParseDefaultFlags>(is, handler);
}

bool HasParseError() const { return parseResult_.IsError(); }

ParseErrorCode GetParseErrorCode() const { return parseResult_.Code(); }

size_t GetErrorOffset() const { return parseResult_.Offset(); }

protected:
void SetParseError(ParseErrorCode code, size_t offset) { parseResult_.Set(code, offset); }

private:
GenericReader(const GenericReader&);
GenericReader& operator=(const GenericReader&);

void ClearStack() { stack_.Clear(); }

struct ClearStackOnExit {
explicit ClearStackOnExit(GenericReader& r) : r_(r) {}
~ClearStackOnExit() { r_.ClearStack(); }
private:
GenericReader& r_;
ClearStackOnExit(const ClearStackOnExit&);
ClearStackOnExit& operator=(const ClearStackOnExit&);
};

template<unsigned parseFlags, typename InputStream, typename Handler>
void ParseObject(InputStream& is, Handler& handler) {
RAPIDJSON_ASSERT(is.Peek() == '{');
is.Take();  

if (!handler.StartObject())
RAPIDJSON_PARSE_ERROR(kParseErrorTermination, is.Tell());

SkipWhitespace(is);

if (is.Peek() == '}') {
is.Take();
if (!handler.EndObject(0))  
RAPIDJSON_PARSE_ERROR(kParseErrorTermination, is.Tell());
return;
}

for (SizeType memberCount = 0;;) {
if (is.Peek() != '"')
RAPIDJSON_PARSE_ERROR(kParseErrorObjectMissName, is.Tell());

ParseString<parseFlags>(is, handler, true);
RAPIDJSON_PARSE_ERROR_EARLY_RETURN_VOID;

SkipWhitespace(is);

if (is.Take() != ':')
RAPIDJSON_PARSE_ERROR(kParseErrorObjectMissColon, is.Tell());

SkipWhitespace(is);

ParseValue<parseFlags>(is, handler);
RAPIDJSON_PARSE_ERROR_EARLY_RETURN_VOID;

SkipWhitespace(is);

++memberCount;

switch (is.Take()) {
case ',': SkipWhitespace(is); break;
case '}': 
if (!handler.EndObject(memberCount))
RAPIDJSON_PARSE_ERROR(kParseErrorTermination, is.Tell());
else
return;
default:  RAPIDJSON_PARSE_ERROR(kParseErrorObjectMissCommaOrCurlyBracket, is.Tell());
}
}
}

template<unsigned parseFlags, typename InputStream, typename Handler>
void ParseArray(InputStream& is, Handler& handler) {
RAPIDJSON_ASSERT(is.Peek() == '[');
is.Take();  

if (!handler.StartArray())
RAPIDJSON_PARSE_ERROR(kParseErrorTermination, is.Tell());

SkipWhitespace(is);

if (is.Peek() == ']') {
is.Take();
if (!handler.EndArray(0)) 
RAPIDJSON_PARSE_ERROR(kParseErrorTermination, is.Tell());
return;
}

for (SizeType elementCount = 0;;) {
ParseValue<parseFlags>(is, handler);
RAPIDJSON_PARSE_ERROR_EARLY_RETURN_VOID;

++elementCount;
SkipWhitespace(is);

switch (is.Take()) {
case ',': SkipWhitespace(is); break;
case ']': 
if (!handler.EndArray(elementCount))
RAPIDJSON_PARSE_ERROR(kParseErrorTermination, is.Tell());
else
return;
default:  RAPIDJSON_PARSE_ERROR(kParseErrorArrayMissCommaOrSquareBracket, is.Tell());
}
}
}

template<unsigned parseFlags, typename InputStream, typename Handler>
void ParseNull(InputStream& is, Handler& handler) {
RAPIDJSON_ASSERT(is.Peek() == 'n');
is.Take();

if (is.Take() == 'u' && is.Take() == 'l' && is.Take() == 'l') {
if (!handler.Null())
RAPIDJSON_PARSE_ERROR(kParseErrorTermination, is.Tell());
}
else
RAPIDJSON_PARSE_ERROR(kParseErrorValueInvalid, is.Tell() - 1);
}

template<unsigned parseFlags, typename InputStream, typename Handler>
void ParseTrue(InputStream& is, Handler& handler) {
RAPIDJSON_ASSERT(is.Peek() == 't');
is.Take();

if (is.Take() == 'r' && is.Take() == 'u' && is.Take() == 'e') {
if (!handler.Bool(true))
RAPIDJSON_PARSE_ERROR(kParseErrorTermination, is.Tell());
}
else
RAPIDJSON_PARSE_ERROR(kParseErrorValueInvalid, is.Tell() - 1);
}

template<unsigned parseFlags, typename InputStream, typename Handler>
void ParseFalse(InputStream& is, Handler& handler) {
RAPIDJSON_ASSERT(is.Peek() == 'f');
is.Take();

if (is.Take() == 'a' && is.Take() == 'l' && is.Take() == 's' && is.Take() == 'e') {
if (!handler.Bool(false))
RAPIDJSON_PARSE_ERROR(kParseErrorTermination, is.Tell());
}
else
RAPIDJSON_PARSE_ERROR(kParseErrorValueInvalid, is.Tell() - 1);
}

template<typename InputStream>
unsigned ParseHex4(InputStream& is) {
unsigned codepoint = 0;
for (int i = 0; i < 4; i++) {
Ch c = is.Take();
codepoint <<= 4;
codepoint += static_cast<unsigned>(c);
if (c >= '0' && c <= '9')
codepoint -= '0';
else if (c >= 'A' && c <= 'F')
codepoint -= 'A' - 10;
else if (c >= 'a' && c <= 'f')
codepoint -= 'a' - 10;
else {
RAPIDJSON_PARSE_ERROR_NORETURN(kParseErrorStringUnicodeEscapeInvalidHex, is.Tell() - 1);
RAPIDJSON_PARSE_ERROR_EARLY_RETURN(0);
}
}
return codepoint;
}

class StackStream {
public:
typedef typename TargetEncoding::Ch Ch;

StackStream(internal::Stack<StackAllocator>& stack) : stack_(stack), length_(0) {}
RAPIDJSON_FORCEINLINE void Put(Ch c) {
*stack_.template Push<Ch>() = c;
++length_;
}
internal::Stack<StackAllocator>& stack_;
SizeType length_;

private:
StackStream(const StackStream&);
StackStream& operator=(const StackStream&);
};

template<unsigned parseFlags, typename InputStream, typename Handler>
void ParseString(InputStream& is, Handler& handler, bool isKey = false) {
internal::StreamLocalCopy<InputStream> copy(is);
InputStream& s(copy.s);

bool success = false;
if (parseFlags & kParseInsituFlag) {
typename InputStream::Ch *head = s.PutBegin();
ParseStringToStream<parseFlags, SourceEncoding, SourceEncoding>(s, s);
RAPIDJSON_PARSE_ERROR_EARLY_RETURN_VOID;
size_t length = s.PutEnd(head) - 1;
RAPIDJSON_ASSERT(length <= 0xFFFFFFFF);
const typename TargetEncoding::Ch* const str = (typename TargetEncoding::Ch*)head;
success = (isKey ? handler.Key(str, SizeType(length), false) : handler.String(str, SizeType(length), false));
}
else {
StackStream stackStream(stack_);
ParseStringToStream<parseFlags, SourceEncoding, TargetEncoding>(s, stackStream);
RAPIDJSON_PARSE_ERROR_EARLY_RETURN_VOID;
const typename TargetEncoding::Ch* const str = stack_.template Pop<typename TargetEncoding::Ch>(stackStream.length_);
success = (isKey ? handler.Key(str, stackStream.length_ - 1, true) : handler.String(str, stackStream.length_ - 1, true));
}
if (!success)
RAPIDJSON_PARSE_ERROR(kParseErrorTermination, s.Tell());
}

template<unsigned parseFlags, typename SEncoding, typename TEncoding, typename InputStream, typename OutputStream>
RAPIDJSON_FORCEINLINE void ParseStringToStream(InputStream& is, OutputStream& os) {
#define Z16 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
static const char escape[256] = {
Z16, Z16, 0, 0,'\"', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,'/', 
Z16, Z16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,'\\', 0, 0, 0, 
0, 0,'\b', 0, 0, 0,'\f', 0, 0, 0, 0, 0, 0, 0,'\n', 0, 
0, 0,'\r', 0,'\t', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
Z16, Z16, Z16, Z16, Z16, Z16, Z16, Z16
};
#undef Z16

RAPIDJSON_ASSERT(is.Peek() == '\"');
is.Take();  

for (;;) {
Ch c = is.Peek();
if (c == '\\') {    
is.Take();
Ch e = is.Take();
if ((sizeof(Ch) == 1 || unsigned(e) < 256) && escape[(unsigned char)e]) {
os.Put(escape[(unsigned char)e]);
}
else if (e == 'u') {    
unsigned codepoint = ParseHex4(is);
if (codepoint >= 0xD800 && codepoint <= 0xDBFF) {
if (is.Take() != '\\' || is.Take() != 'u')
RAPIDJSON_PARSE_ERROR(kParseErrorStringUnicodeSurrogateInvalid, is.Tell() - 2);
unsigned codepoint2 = ParseHex4(is);
if (codepoint2 < 0xDC00 || codepoint2 > 0xDFFF)
RAPIDJSON_PARSE_ERROR(kParseErrorStringUnicodeSurrogateInvalid, is.Tell() - 2);
codepoint = (((codepoint - 0xD800) << 10) | (codepoint2 - 0xDC00)) + 0x10000;
}
TEncoding::Encode(os, codepoint);
}
else
RAPIDJSON_PARSE_ERROR(kParseErrorStringEscapeInvalid, is.Tell() - 1);
}
else if (c == '"') {    
is.Take();
os.Put('\0');   
return;
}
else if (c == '\0')
RAPIDJSON_PARSE_ERROR(kParseErrorStringMissQuotationMark, is.Tell() - 1);
else if ((unsigned)c < 0x20) 
RAPIDJSON_PARSE_ERROR(kParseErrorStringEscapeInvalid, is.Tell() - 1);
else {
if (parseFlags & kParseValidateEncodingFlag ? 
!Transcoder<SEncoding, TEncoding>::Validate(is, os) : 
!Transcoder<SEncoding, TEncoding>::Transcode(is, os))
RAPIDJSON_PARSE_ERROR(kParseErrorStringInvalidEncoding, is.Tell());
}
}
}

inline double StrtodFastPath(double significand, int exp) {
if (exp < -308)
return 0.0;
else if (exp >= 0)
return significand * internal::Pow10(exp);
else
return significand / internal::Pow10(-exp);
}

template<unsigned parseFlags, typename InputStream, typename Handler>
void ParseNumber(InputStream& is, Handler& handler) {
internal::StreamLocalCopy<InputStream> copy(is);
InputStream& s(copy.s);

bool minus = false;
if (s.Peek() == '-') {
minus = true;
s.Take();
}

unsigned i = 0;
uint64_t i64 = 0;
bool use64bit = false;
if (s.Peek() == '0') {
i = 0;
s.Take();
}
else if (s.Peek() >= '1' && s.Peek() <= '9') {
i = static_cast<unsigned>(s.Take() - '0');

if (minus)
while (s.Peek() >= '0' && s.Peek() <= '9') {
if (i >= 214748364) { 
if (i != 214748364 || s.Peek() > '8') {
i64 = i;
use64bit = true;
break;
}
}
i = i * 10 + static_cast<unsigned>(s.Take() - '0');
}
else
while (s.Peek() >= '0' && s.Peek() <= '9') {
if (i >= 429496729) { 
if (i != 429496729 || s.Peek() > '5') {
i64 = i;
use64bit = true;
break;
}
}
i = i * 10 + static_cast<unsigned>(s.Take() - '0');
}
}
else
RAPIDJSON_PARSE_ERROR(kParseErrorValueInvalid, s.Tell());

double d = 0.0;
bool useDouble = false;
if (use64bit) {
if (minus) 
while (s.Peek() >= '0' && s.Peek() <= '9') {                    
if (i64 >= RAPIDJSON_UINT64_C2(0x0CCCCCCC, 0xCCCCCCCC)) 
if (i64 != RAPIDJSON_UINT64_C2(0x0CCCCCCC, 0xCCCCCCCC) || s.Peek() > '8') {
d = (double)i64;
useDouble = true;
break;
}
i64 = i64 * 10 + static_cast<unsigned>(s.Take() - '0');
}
else
while (s.Peek() >= '0' && s.Peek() <= '9') {                    
if (i64 >= RAPIDJSON_UINT64_C2(0x19999999, 0x99999999)) 
if (i64 != RAPIDJSON_UINT64_C2(0x19999999, 0x99999999) || s.Peek() > '5') {
d = (double)i64;
useDouble = true;
break;
}
i64 = i64 * 10 + static_cast<unsigned>(s.Take() - '0');
}
}

if (useDouble) {
while (s.Peek() >= '0' && s.Peek() <= '9') {
if (d >= 1.7976931348623157e307) 
RAPIDJSON_PARSE_ERROR(kParseErrorNumberTooBig, s.Tell());
d = d * 10 + (s.Take() - '0');
}
}

int expFrac = 0;
if (s.Peek() == '.') {
s.Take();

#if RAPIDJSON_64BIT
if (!useDouble) {
if (!use64bit)
i64 = i;

while (s.Peek() >= '0' && s.Peek() <= '9') {
if (i64 >= RAPIDJSON_UINT64_C2(0x19999999, 0x99999999))
break;
else {
i64 = i64 * 10 + static_cast<unsigned>(s.Take() - '0');
--expFrac;
}
}

d = (double)i64;
}
#else
if (!useDouble)
d = use64bit ? (double)i64 : (double)i;
#endif
useDouble = true;

while (s.Peek() >= '0' && s.Peek() <= '9') {
d = d * 10 + (s.Take() - '0');
--expFrac;
}

if (expFrac == 0)
RAPIDJSON_PARSE_ERROR(kParseErrorNumberMissFraction, s.Tell());
}

int exp = 0;
if (s.Peek() == 'e' || s.Peek() == 'E') {
if (!useDouble) {
d = use64bit ? (double)i64 : (double)i;
useDouble = true;
}
s.Take();

bool expMinus = false;
if (s.Peek() == '+')
s.Take();
else if (s.Peek() == '-') {
s.Take();
expMinus = true;
}

if (s.Peek() >= '0' && s.Peek() <= '9') {
exp = s.Take() - '0';
while (s.Peek() >= '0' && s.Peek() <= '9') {
exp = exp * 10 + (s.Take() - '0');
if (exp > 308 && !expMinus) 
RAPIDJSON_PARSE_ERROR(kParseErrorNumberTooBig, s.Tell());
}
}
else
RAPIDJSON_PARSE_ERROR(kParseErrorNumberMissExponent, s.Tell());

if (expMinus)
exp = -exp;
}

bool cont = true;
if (useDouble) {
int expSum = exp + expFrac;
if (expSum < -308) {
d = StrtodFastPath(d, exp);
d = StrtodFastPath(d, expFrac);
}
else
d = StrtodFastPath(d, expSum);

cont = handler.Double(minus ? -d : d);
}
else {
if (use64bit) {
if (minus)
cont = handler.Int64(-(int64_t)i64);
else
cont = handler.Uint64(i64);
}
else {
if (minus)
cont = handler.Int(-(int)i);
else
cont = handler.Uint(i);
}
}
if (!cont)
RAPIDJSON_PARSE_ERROR(kParseErrorTermination, s.Tell());
}

template<unsigned parseFlags, typename InputStream, typename Handler>
void ParseValue(InputStream& is, Handler& handler) {
switch (is.Peek()) {
case 'n': ParseNull  <parseFlags>(is, handler); break;
case 't': ParseTrue  <parseFlags>(is, handler); break;
case 'f': ParseFalse <parseFlags>(is, handler); break;
case '"': ParseString<parseFlags>(is, handler); break;
case '{': ParseObject<parseFlags>(is, handler); break;
case '[': ParseArray <parseFlags>(is, handler); break;
default : ParseNumber<parseFlags>(is, handler);
}
}


enum IterativeParsingState {
IterativeParsingStartState = 0,
IterativeParsingFinishState,
IterativeParsingErrorState,

IterativeParsingObjectInitialState,
IterativeParsingMemberKeyState,
IterativeParsingKeyValueDelimiterState,
IterativeParsingMemberValueState,
IterativeParsingMemberDelimiterState,
IterativeParsingObjectFinishState,

IterativeParsingArrayInitialState,
IterativeParsingElementState,
IterativeParsingElementDelimiterState,
IterativeParsingArrayFinishState,

IterativeParsingValueState,

cIterativeParsingStateCount
};

enum Token {
LeftBracketToken = 0,
RightBracketToken,

LeftCurlyBracketToken,
RightCurlyBracketToken,

CommaToken,
ColonToken,

StringToken,
FalseToken,
TrueToken,
NullToken,
NumberToken,

kTokenCount
};

RAPIDJSON_FORCEINLINE Token Tokenize(Ch c) {

#define N NumberToken
#define N16 N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N
static const unsigned char tokenMap[256] = {
N16, 
N16, 
N, N, StringToken, N, N, N, N, N, N, N, N, N, CommaToken, N, N, N, 
N, N, N, N, N, N, N, N, N, N, ColonToken, N, N, N, N, N, 
N16, 
N, N, N, N, N, N, N, N, N, N, N, LeftBracketToken, N, RightBracketToken, N, N, 
N, N, N, N, N, N, FalseToken, N, N, N, N, N, N, N, NullToken, N, 
N, N, N, N, TrueToken, N, N, N, N, N, N, LeftCurlyBracketToken, N, RightCurlyBracketToken, N, N, 
N16, N16, N16, N16, N16, N16, N16, N16 
};
#undef N
#undef N16

if (sizeof(Ch) == 1 || unsigned(c) < 256)
return (Token)tokenMap[(unsigned char)c];
else
return NumberToken;
}

RAPIDJSON_FORCEINLINE IterativeParsingState Predict(IterativeParsingState state, Token token) {
static const char G[cIterativeParsingStateCount][kTokenCount] = {
{
IterativeParsingArrayInitialState,  
IterativeParsingErrorState,         
IterativeParsingObjectInitialState, 
IterativeParsingErrorState,         
IterativeParsingErrorState,         
IterativeParsingErrorState,         
IterativeParsingValueState,         
IterativeParsingValueState,         
IterativeParsingValueState,         
IterativeParsingValueState,         
IterativeParsingValueState          
},
{
IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState,
IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState,
IterativeParsingErrorState
},
{
IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState,
IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState,
IterativeParsingErrorState
},
{
IterativeParsingErrorState,         
IterativeParsingErrorState,         
IterativeParsingErrorState,         
IterativeParsingObjectFinishState,  
IterativeParsingErrorState,         
IterativeParsingErrorState,         
IterativeParsingMemberKeyState,     
IterativeParsingErrorState,         
IterativeParsingErrorState,         
IterativeParsingErrorState,         
IterativeParsingErrorState          
},
{
IterativeParsingErrorState,             
IterativeParsingErrorState,             
IterativeParsingErrorState,             
IterativeParsingErrorState,             
IterativeParsingErrorState,             
IterativeParsingKeyValueDelimiterState, 
IterativeParsingErrorState,             
IterativeParsingErrorState,             
IterativeParsingErrorState,             
IterativeParsingErrorState,             
IterativeParsingErrorState              
},
{
IterativeParsingArrayInitialState,      
IterativeParsingErrorState,             
IterativeParsingObjectInitialState,     
IterativeParsingErrorState,             
IterativeParsingErrorState,             
IterativeParsingErrorState,             
IterativeParsingMemberValueState,       
IterativeParsingMemberValueState,       
IterativeParsingMemberValueState,       
IterativeParsingMemberValueState,       
IterativeParsingMemberValueState        
},
{
IterativeParsingErrorState,             
IterativeParsingErrorState,             
IterativeParsingErrorState,             
IterativeParsingObjectFinishState,      
IterativeParsingMemberDelimiterState,   
IterativeParsingErrorState,             
IterativeParsingErrorState,             
IterativeParsingErrorState,             
IterativeParsingErrorState,             
IterativeParsingErrorState,             
IterativeParsingErrorState              
},
{
IterativeParsingErrorState,         
IterativeParsingErrorState,         
IterativeParsingErrorState,         
IterativeParsingErrorState,         
IterativeParsingErrorState,         
IterativeParsingErrorState,         
IterativeParsingMemberKeyState,     
IterativeParsingErrorState,         
IterativeParsingErrorState,         
IterativeParsingErrorState,         
IterativeParsingErrorState          
},
{
IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState,
IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState,
IterativeParsingErrorState
},
{
IterativeParsingArrayInitialState,      
IterativeParsingArrayFinishState,       
IterativeParsingObjectInitialState,     
IterativeParsingErrorState,             
IterativeParsingErrorState,             
IterativeParsingErrorState,             
IterativeParsingElementState,           
IterativeParsingElementState,           
IterativeParsingElementState,           
IterativeParsingElementState,           
IterativeParsingElementState            
},
{
IterativeParsingErrorState,             
IterativeParsingArrayFinishState,       
IterativeParsingErrorState,             
IterativeParsingErrorState,             
IterativeParsingElementDelimiterState,  
IterativeParsingErrorState,             
IterativeParsingErrorState,             
IterativeParsingErrorState,             
IterativeParsingErrorState,             
IterativeParsingErrorState,             
IterativeParsingErrorState              
},
{
IterativeParsingArrayInitialState,      
IterativeParsingErrorState,             
IterativeParsingObjectInitialState,     
IterativeParsingErrorState,             
IterativeParsingErrorState,             
IterativeParsingErrorState,             
IterativeParsingElementState,           
IterativeParsingElementState,           
IterativeParsingElementState,           
IterativeParsingElementState,           
IterativeParsingElementState            
},
{
IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState,
IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState,
IterativeParsingErrorState
},
{
IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState,
IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState, IterativeParsingErrorState,
IterativeParsingErrorState
}
}; 

return (IterativeParsingState)G[state][token];
}

template <unsigned parseFlags, typename InputStream, typename Handler>
RAPIDJSON_FORCEINLINE IterativeParsingState Transit(IterativeParsingState src, Token token, IterativeParsingState dst, InputStream& is, Handler& handler) {
switch (dst) {
case IterativeParsingStartState:
RAPIDJSON_ASSERT(false);
return IterativeParsingErrorState;

case IterativeParsingFinishState:
return dst;

case IterativeParsingErrorState:
return dst;

case IterativeParsingObjectInitialState:
case IterativeParsingArrayInitialState:
{
IterativeParsingState n = src;
if (src == IterativeParsingArrayInitialState || src == IterativeParsingElementDelimiterState)
n = IterativeParsingElementState;
else if (src == IterativeParsingKeyValueDelimiterState)
n = IterativeParsingMemberValueState;
*stack_.template Push<SizeType>(1) = n;
*stack_.template Push<SizeType>(1) = 0;
bool hr = (dst == IterativeParsingObjectInitialState) ? handler.StartObject() : handler.StartArray();
if (!hr) {
RAPIDJSON_PARSE_ERROR_NORETURN(kParseErrorTermination, is.Tell());
return IterativeParsingErrorState;
}
else {
is.Take();
return dst;
}
}

case IterativeParsingMemberKeyState:
ParseString<parseFlags>(is, handler, true);
if (HasParseError())
return IterativeParsingErrorState;
else
return dst;

case IterativeParsingKeyValueDelimiterState:
if (token == ColonToken) {
is.Take();
return dst;
}
else
return IterativeParsingErrorState;

case IterativeParsingMemberValueState:
ParseValue<parseFlags>(is, handler);
if (HasParseError()) {
return IterativeParsingErrorState;
}
return dst;

case IterativeParsingElementState:
ParseValue<parseFlags>(is, handler);
if (HasParseError()) {
return IterativeParsingErrorState;
}
return dst;

case IterativeParsingMemberDelimiterState:
case IterativeParsingElementDelimiterState:
is.Take();
*stack_.template Top<SizeType>() = *stack_.template Top<SizeType>() + 1;
return dst;

case IterativeParsingObjectFinishState:
{
SizeType c = *stack_.template Pop<SizeType>(1);
if (src == IterativeParsingMemberValueState)
++c;
IterativeParsingState n = static_cast<IterativeParsingState>(*stack_.template Pop<SizeType>(1));
if (n == IterativeParsingStartState)
n = IterativeParsingFinishState;
bool hr = handler.EndObject(c);
if (!hr) {
RAPIDJSON_PARSE_ERROR_NORETURN(kParseErrorTermination, is.Tell());
return IterativeParsingErrorState;
}
else {
is.Take();
return n;
}
}

case IterativeParsingArrayFinishState:
{
SizeType c = *stack_.template Pop<SizeType>(1);
if (src == IterativeParsingElementState)
++c;
IterativeParsingState n = static_cast<IterativeParsingState>(*stack_.template Pop<SizeType>(1));
if (n == IterativeParsingStartState)
n = IterativeParsingFinishState;
bool hr = handler.EndArray(c);
if (!hr) {
RAPIDJSON_PARSE_ERROR_NORETURN(kParseErrorTermination, is.Tell());
return IterativeParsingErrorState;
}
else {
is.Take();
return n;
}
}

case IterativeParsingValueState:
ParseValue<parseFlags>(is, handler);
if (HasParseError()) {
return IterativeParsingErrorState;
}
return IterativeParsingFinishState;

default:
RAPIDJSON_ASSERT(false);
return IterativeParsingErrorState;
}
}

template <typename InputStream>
void HandleError(IterativeParsingState src, InputStream& is) {
if (HasParseError()) {
return;
}

switch (src) {
case IterativeParsingStartState:            RAPIDJSON_PARSE_ERROR(kParseErrorDocumentEmpty, is.Tell());
case IterativeParsingFinishState:           RAPIDJSON_PARSE_ERROR(kParseErrorDocumentRootNotSingular, is.Tell());
case IterativeParsingObjectInitialState:
case IterativeParsingMemberDelimiterState:  RAPIDJSON_PARSE_ERROR(kParseErrorObjectMissName, is.Tell());
case IterativeParsingMemberKeyState:        RAPIDJSON_PARSE_ERROR(kParseErrorObjectMissColon, is.Tell());
case IterativeParsingMemberValueState:      RAPIDJSON_PARSE_ERROR(kParseErrorObjectMissCommaOrCurlyBracket, is.Tell());
case IterativeParsingElementState:          RAPIDJSON_PARSE_ERROR(kParseErrorArrayMissCommaOrSquareBracket, is.Tell());
default:                                    RAPIDJSON_PARSE_ERROR(kParseErrorUnspecificSyntaxError, is.Tell());
}       
}

template <unsigned parseFlags, typename InputStream, typename Handler>
ParseResult IterativeParse(InputStream& is, Handler& handler) {
parseResult_.Clear();
ClearStackOnExit scope(*this);
IterativeParsingState state = IterativeParsingStartState;

SkipWhitespace(is);
while (is.Peek() != '\0') {
Token t = Tokenize(is.Peek());
IterativeParsingState n = Predict(state, t);
IterativeParsingState d = Transit<parseFlags>(state, t, n, is, handler);

if (d == IterativeParsingErrorState) {
HandleError(state, is);
break;
}

state = d;

if ((parseFlags & kParseStopWhenDoneFlag) && state == IterativeParsingFinishState)
break;

SkipWhitespace(is);
}

if (state != IterativeParsingFinishState)
HandleError(state, is);

return parseResult_;
}

static const size_t kDefaultStackCapacity = 256;    
internal::Stack<StackAllocator> stack_;  
ParseResult parseResult_;
}; 

typedef GenericReader<UTF8<>, UTF8<> > Reader;

RAPIDJSON_NAMESPACE_END

#ifdef _MSC_VER
RAPIDJSON_DIAG_POP
#endif

#endif 

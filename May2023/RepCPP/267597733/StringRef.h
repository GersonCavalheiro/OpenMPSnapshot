
#ifndef LLVM_ADT_STRINGREF_H
#define LLVM_ADT_STRINGREF_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Compiler.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

namespace llvm {

class APInt;
class hash_code;
template <typename T> class SmallVectorImpl;
class StringRef;

bool getAsUnsignedInteger(StringRef Str, unsigned Radix,
unsigned long long &Result);

bool getAsSignedInteger(StringRef Str, unsigned Radix, long long &Result);

bool consumeUnsignedInteger(StringRef &Str, unsigned Radix,
unsigned long long &Result);
bool consumeSignedInteger(StringRef &Str, unsigned Radix, long long &Result);

class StringRef {
public:
static const size_t npos = ~size_t(0);

using iterator = const char *;
using const_iterator = const char *;
using size_type = size_t;

private:
const char *Data = nullptr;

size_t Length = 0;

LLVM_ATTRIBUTE_ALWAYS_INLINE
static int compareMemory(const char *Lhs, const char *Rhs, size_t Length) {
if (Length == 0) { return 0; }
return ::memcmp(Lhs,Rhs,Length);
}

public:

StringRef() = default;

StringRef(std::nullptr_t) = delete;

LLVM_ATTRIBUTE_ALWAYS_INLINE
StringRef(const char *Str)
: Data(Str), Length(Str ? ::strlen(Str) : 0) {}

LLVM_ATTRIBUTE_ALWAYS_INLINE
constexpr StringRef(const char *data, size_t length)
: Data(data), Length(length) {}

LLVM_ATTRIBUTE_ALWAYS_INLINE
StringRef(const std::string &Str)
: Data(Str.data()), Length(Str.length()) {}

static StringRef withNullAsEmpty(const char *data) {
return StringRef(data ? data : "");
}


iterator begin() const { return Data; }

iterator end() const { return Data + Length; }

const unsigned char *bytes_begin() const {
return reinterpret_cast<const unsigned char *>(begin());
}
const unsigned char *bytes_end() const {
return reinterpret_cast<const unsigned char *>(end());
}
iterator_range<const unsigned char *> bytes() const {
return make_range(bytes_begin(), bytes_end());
}


LLVM_NODISCARD
LLVM_ATTRIBUTE_ALWAYS_INLINE
const char *data() const { return Data; }

LLVM_NODISCARD
LLVM_ATTRIBUTE_ALWAYS_INLINE
bool empty() const { return Length == 0; }

LLVM_NODISCARD
LLVM_ATTRIBUTE_ALWAYS_INLINE
size_t size() const { return Length; }

LLVM_NODISCARD
char front() const {
assert(!empty());
return Data[0];
}

LLVM_NODISCARD
char back() const {
assert(!empty());
return Data[Length-1];
}

template <typename Allocator>
LLVM_NODISCARD StringRef copy(Allocator &A) const {
if (empty())
return StringRef();
char *S = A.template Allocate<char>(Length);
std::copy(begin(), end(), S);
return StringRef(S, Length);
}

LLVM_NODISCARD
LLVM_ATTRIBUTE_ALWAYS_INLINE
bool equals(StringRef RHS) const {
return (Length == RHS.Length &&
compareMemory(Data, RHS.Data, RHS.Length) == 0);
}

LLVM_NODISCARD
bool equals_lower(StringRef RHS) const {
return Length == RHS.Length && compare_lower(RHS) == 0;
}

LLVM_NODISCARD
LLVM_ATTRIBUTE_ALWAYS_INLINE
int compare(StringRef RHS) const {
if (int Res = compareMemory(Data, RHS.Data, std::min(Length, RHS.Length)))
return Res < 0 ? -1 : 1;

if (Length == RHS.Length)
return 0;
return Length < RHS.Length ? -1 : 1;
}

LLVM_NODISCARD
int compare_lower(StringRef RHS) const;

LLVM_NODISCARD
int compare_numeric(StringRef RHS) const;

LLVM_NODISCARD
unsigned edit_distance(StringRef Other, bool AllowReplacements = true,
unsigned MaxEditDistance = 0) const;

LLVM_NODISCARD
std::string str() const {
if (!Data) return std::string();
return std::string(Data, Length);
}


LLVM_NODISCARD
char operator[](size_t Index) const {
assert(Index < Length && "Invalid index!");
return Data[Index];
}

template <typename T>
typename std::enable_if<std::is_same<T, std::string>::value,
StringRef>::type &
operator=(T &&Str) = delete;


operator std::string() const {
return str();
}


LLVM_NODISCARD
LLVM_ATTRIBUTE_ALWAYS_INLINE
bool startswith(StringRef Prefix) const {
return Length >= Prefix.Length &&
compareMemory(Data, Prefix.Data, Prefix.Length) == 0;
}

LLVM_NODISCARD
bool startswith_lower(StringRef Prefix) const;

LLVM_NODISCARD
LLVM_ATTRIBUTE_ALWAYS_INLINE
bool endswith(StringRef Suffix) const {
return Length >= Suffix.Length &&
compareMemory(end() - Suffix.Length, Suffix.Data, Suffix.Length) == 0;
}

LLVM_NODISCARD
bool endswith_lower(StringRef Suffix) const;


LLVM_NODISCARD
LLVM_ATTRIBUTE_ALWAYS_INLINE
size_t find(char C, size_t From = 0) const {
size_t FindBegin = std::min(From, Length);
if (FindBegin < Length) { 
if (const void *P = ::memchr(Data + FindBegin, C, Length - FindBegin))
return static_cast<const char *>(P) - Data;
}
return npos;
}

LLVM_NODISCARD
size_t find_lower(char C, size_t From = 0) const;

LLVM_NODISCARD
LLVM_ATTRIBUTE_ALWAYS_INLINE
size_t find_if(function_ref<bool(char)> F, size_t From = 0) const {
StringRef S = drop_front(From);
while (!S.empty()) {
if (F(S.front()))
return size() - S.size();
S = S.drop_front();
}
return npos;
}

LLVM_NODISCARD
LLVM_ATTRIBUTE_ALWAYS_INLINE
size_t find_if_not(function_ref<bool(char)> F, size_t From = 0) const {
return find_if([F](char c) { return !F(c); }, From);
}

LLVM_NODISCARD
size_t find(StringRef Str, size_t From = 0) const;

LLVM_NODISCARD
size_t find_lower(StringRef Str, size_t From = 0) const;

LLVM_NODISCARD
size_t rfind(char C, size_t From = npos) const {
From = std::min(From, Length);
size_t i = From;
while (i != 0) {
--i;
if (Data[i] == C)
return i;
}
return npos;
}

LLVM_NODISCARD
size_t rfind_lower(char C, size_t From = npos) const;

LLVM_NODISCARD
size_t rfind(StringRef Str) const;

LLVM_NODISCARD
size_t rfind_lower(StringRef Str) const;

LLVM_NODISCARD
size_t find_first_of(char C, size_t From = 0) const {
return find(C, From);
}

LLVM_NODISCARD
size_t find_first_of(StringRef Chars, size_t From = 0) const;

LLVM_NODISCARD
size_t find_first_not_of(char C, size_t From = 0) const;

LLVM_NODISCARD
size_t find_first_not_of(StringRef Chars, size_t From = 0) const;

LLVM_NODISCARD
size_t find_last_of(char C, size_t From = npos) const {
return rfind(C, From);
}

LLVM_NODISCARD
size_t find_last_of(StringRef Chars, size_t From = npos) const;

LLVM_NODISCARD
size_t find_last_not_of(char C, size_t From = npos) const;

LLVM_NODISCARD
size_t find_last_not_of(StringRef Chars, size_t From = npos) const;

LLVM_NODISCARD
LLVM_ATTRIBUTE_ALWAYS_INLINE
bool contains(StringRef Other) const { return find(Other) != npos; }

LLVM_NODISCARD
LLVM_ATTRIBUTE_ALWAYS_INLINE
bool contains(char C) const { return find_first_of(C) != npos; }

LLVM_NODISCARD
LLVM_ATTRIBUTE_ALWAYS_INLINE
bool contains_lower(StringRef Other) const {
return find_lower(Other) != npos;
}

LLVM_NODISCARD
LLVM_ATTRIBUTE_ALWAYS_INLINE
bool contains_lower(char C) const { return find_lower(C) != npos; }


LLVM_NODISCARD
size_t count(char C) const {
size_t Count = 0;
for (size_t i = 0, e = Length; i != e; ++i)
if (Data[i] == C)
++Count;
return Count;
}

size_t count(StringRef Str) const;

template <typename T>
typename std::enable_if<std::numeric_limits<T>::is_signed, bool>::type
getAsInteger(unsigned Radix, T &Result) const {
long long LLVal;
if (getAsSignedInteger(*this, Radix, LLVal) ||
static_cast<T>(LLVal) != LLVal)
return true;
Result = LLVal;
return false;
}

template <typename T>
typename std::enable_if<!std::numeric_limits<T>::is_signed, bool>::type
getAsInteger(unsigned Radix, T &Result) const {
unsigned long long ULLVal;
if (getAsUnsignedInteger(*this, Radix, ULLVal) ||
static_cast<unsigned long long>(static_cast<T>(ULLVal)) != ULLVal)
return true;
Result = ULLVal;
return false;
}

template <typename T>
typename std::enable_if<std::numeric_limits<T>::is_signed, bool>::type
consumeInteger(unsigned Radix, T &Result) {
long long LLVal;
if (consumeSignedInteger(*this, Radix, LLVal) ||
static_cast<long long>(static_cast<T>(LLVal)) != LLVal)
return true;
Result = LLVal;
return false;
}

template <typename T>
typename std::enable_if<!std::numeric_limits<T>::is_signed, bool>::type
consumeInteger(unsigned Radix, T &Result) {
unsigned long long ULLVal;
if (consumeUnsignedInteger(*this, Radix, ULLVal) ||
static_cast<unsigned long long>(static_cast<T>(ULLVal)) != ULLVal)
return true;
Result = ULLVal;
return false;
}

bool getAsInteger(unsigned Radix, APInt &Result) const;

bool getAsDouble(double &Result, bool AllowInexact = true) const;


LLVM_NODISCARD
std::string lower() const;

LLVM_NODISCARD
std::string upper() const;


LLVM_NODISCARD
LLVM_ATTRIBUTE_ALWAYS_INLINE
StringRef substr(size_t Start, size_t N = npos) const {
Start = std::min(Start, Length);
return StringRef(Data + Start, std::min(N, Length - Start));
}

LLVM_NODISCARD
LLVM_ATTRIBUTE_ALWAYS_INLINE
StringRef take_front(size_t N = 1) const {
if (N >= size())
return *this;
return drop_back(size() - N);
}

LLVM_NODISCARD
LLVM_ATTRIBUTE_ALWAYS_INLINE
StringRef take_back(size_t N = 1) const {
if (N >= size())
return *this;
return drop_front(size() - N);
}

LLVM_NODISCARD
LLVM_ATTRIBUTE_ALWAYS_INLINE
StringRef take_while(function_ref<bool(char)> F) const {
return substr(0, find_if_not(F));
}

LLVM_NODISCARD
LLVM_ATTRIBUTE_ALWAYS_INLINE
StringRef take_until(function_ref<bool(char)> F) const {
return substr(0, find_if(F));
}

LLVM_NODISCARD
LLVM_ATTRIBUTE_ALWAYS_INLINE
StringRef drop_front(size_t N = 1) const {
assert(size() >= N && "Dropping more elements than exist");
return substr(N);
}

LLVM_NODISCARD
LLVM_ATTRIBUTE_ALWAYS_INLINE
StringRef drop_back(size_t N = 1) const {
assert(size() >= N && "Dropping more elements than exist");
return substr(0, size()-N);
}

LLVM_NODISCARD
LLVM_ATTRIBUTE_ALWAYS_INLINE
StringRef drop_while(function_ref<bool(char)> F) const {
return substr(find_if_not(F));
}

LLVM_NODISCARD
LLVM_ATTRIBUTE_ALWAYS_INLINE
StringRef drop_until(function_ref<bool(char)> F) const {
return substr(find_if(F));
}

LLVM_ATTRIBUTE_ALWAYS_INLINE
bool consume_front(StringRef Prefix) {
if (!startswith(Prefix))
return false;

*this = drop_front(Prefix.size());
return true;
}

LLVM_ATTRIBUTE_ALWAYS_INLINE
bool consume_back(StringRef Suffix) {
if (!endswith(Suffix))
return false;

*this = drop_back(Suffix.size());
return true;
}

LLVM_NODISCARD
LLVM_ATTRIBUTE_ALWAYS_INLINE
StringRef slice(size_t Start, size_t End) const {
Start = std::min(Start, Length);
End = std::min(std::max(Start, End), Length);
return StringRef(Data + Start, End - Start);
}

LLVM_NODISCARD
std::pair<StringRef, StringRef> split(char Separator) const {
return split(StringRef(&Separator, 1));
}

LLVM_NODISCARD
std::pair<StringRef, StringRef> split(StringRef Separator) const {
size_t Idx = find(Separator);
if (Idx == npos)
return std::make_pair(*this, StringRef());
return std::make_pair(slice(0, Idx), slice(Idx + Separator.size(), npos));
}

LLVM_NODISCARD
std::pair<StringRef, StringRef> rsplit(StringRef Separator) const {
size_t Idx = rfind(Separator);
if (Idx == npos)
return std::make_pair(*this, StringRef());
return std::make_pair(slice(0, Idx), slice(Idx + Separator.size(), npos));
}

void split(SmallVectorImpl<StringRef> &A,
StringRef Separator, int MaxSplit = -1,
bool KeepEmpty = true) const;

void split(SmallVectorImpl<StringRef> &A, char Separator, int MaxSplit = -1,
bool KeepEmpty = true) const;

LLVM_NODISCARD
std::pair<StringRef, StringRef> rsplit(char Separator) const {
return rsplit(StringRef(&Separator, 1));
}

LLVM_NODISCARD
StringRef ltrim(char Char) const {
return drop_front(std::min(Length, find_first_not_of(Char)));
}

LLVM_NODISCARD
StringRef ltrim(StringRef Chars = " \t\n\v\f\r") const {
return drop_front(std::min(Length, find_first_not_of(Chars)));
}

LLVM_NODISCARD
StringRef rtrim(char Char) const {
return drop_back(Length - std::min(Length, find_last_not_of(Char) + 1));
}

LLVM_NODISCARD
StringRef rtrim(StringRef Chars = " \t\n\v\f\r") const {
return drop_back(Length - std::min(Length, find_last_not_of(Chars) + 1));
}

LLVM_NODISCARD
StringRef trim(char Char) const {
return ltrim(Char).rtrim(Char);
}

LLVM_NODISCARD
StringRef trim(StringRef Chars = " \t\n\v\f\r") const {
return ltrim(Chars).rtrim(Chars);
}

};

class StringLiteral : public StringRef {
private:
constexpr StringLiteral(const char *Str, size_t N) : StringRef(Str, N) {
}

public:
template <size_t N>
constexpr StringLiteral(const char (&Str)[N])
#if defined(__clang__) && __has_attribute(enable_if)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgcc-compat"
__attribute((enable_if(__builtin_strlen(Str) == N - 1,
"invalid string literal")))
#pragma clang diagnostic pop
#endif
: StringRef(Str, N - 1) {
}

template <size_t N>
static constexpr StringLiteral withInnerNUL(const char (&Str)[N]) {
return StringLiteral(Str, N - 1);
}
};


LLVM_ATTRIBUTE_ALWAYS_INLINE
inline bool operator==(StringRef LHS, StringRef RHS) {
return LHS.equals(RHS);
}

LLVM_ATTRIBUTE_ALWAYS_INLINE
inline bool operator!=(StringRef LHS, StringRef RHS) { return !(LHS == RHS); }

inline bool operator<(StringRef LHS, StringRef RHS) {
return LHS.compare(RHS) == -1;
}

inline bool operator<=(StringRef LHS, StringRef RHS) {
return LHS.compare(RHS) != 1;
}

inline bool operator>(StringRef LHS, StringRef RHS) {
return LHS.compare(RHS) == 1;
}

inline bool operator>=(StringRef LHS, StringRef RHS) {
return LHS.compare(RHS) != -1;
}

inline std::string &operator+=(std::string &buffer, StringRef string) {
return buffer.append(string.data(), string.size());
}


LLVM_NODISCARD
hash_code hash_value(StringRef S);

template <typename T> struct isPodLike;
template <> struct isPodLike<StringRef> { static const bool value = true; };

} 

#endif 


#ifndef RE2_RE2_H_
#define RE2_RE2_H_


#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <map>
#include <mutex>
#include <string>

#include "re2/stringpiece.h"

namespace re2 {
class Prog;
class Regexp;
}  

namespace re2 {

using std::string;

class RE2 {
public:
class Arg;
class Options;

class Set;

enum ErrorCode {
NoError = 0,

ErrorInternal,

ErrorBadEscape,          
ErrorBadCharClass,       
ErrorBadCharRange,       
ErrorMissingBracket,     
ErrorMissingParen,       
ErrorTrailingBackslash,  
ErrorRepeatArgument,     
ErrorRepeatSize,         
ErrorRepeatOp,           
ErrorBadPerlOp,          
ErrorBadUTF8,            
ErrorBadNamedCapture,    
ErrorPatternTooLarge     
};

enum CannedOptions {
DefaultOptions = 0,
Latin1, 
POSIX, 
Quiet 
};

#ifndef SWIG
RE2(const char* pattern);
RE2(const string& pattern);
#endif
RE2(const StringPiece& pattern);
RE2(const StringPiece& pattern, const Options& options);
~RE2();

bool ok() const { return error_code() == NoError; }

const string& pattern() const { return pattern_; }

const string& error() const { return *error_; }

ErrorCode error_code() const { return error_code_; }

const string& error_arg() const { return error_arg_; }

int ProgramSize() const;

int ProgramFanout(std::map<int, int>* histogram) const;

re2::Regexp* Regexp() const { return entire_regexp_; }



static bool FullMatchN(const StringPiece& text, const RE2& re,
const Arg* const args[], int argc);
static bool PartialMatchN(const StringPiece& text, const RE2& re,
const Arg* const args[], int argc);
static bool ConsumeN(StringPiece* input, const RE2& re,
const Arg* const args[], int argc);
static bool FindAndConsumeN(StringPiece* input, const RE2& re,
const Arg* const args[], int argc);

#ifndef SWIG
private:
template <typename F, typename SP>
static inline bool Apply(F f, SP sp, const RE2& re) {
return f(sp, re, NULL, 0);
}

template <typename F, typename SP, typename... A>
static inline bool Apply(F f, SP sp, const RE2& re, const A&... a) {
const Arg* const args[] = {&a...};
const int argc = sizeof...(a);
return f(sp, re, args, argc);
}

public:



template <typename... A>
static bool FullMatch(const StringPiece& text, const RE2& re, A&&... a) {
return Apply(FullMatchN, text, re, Arg(std::forward<A>(a))...);
}

template <typename... A>
static bool PartialMatch(const StringPiece& text, const RE2& re, A&&... a) {
return Apply(PartialMatchN, text, re, Arg(std::forward<A>(a))...);
}

template <typename... A>
static bool Consume(StringPiece* input, const RE2& re, A&&... a) {
return Apply(ConsumeN, input, re, Arg(std::forward<A>(a))...);
}

template <typename... A>
static bool FindAndConsume(StringPiece* input, const RE2& re, A&&... a) {
return Apply(FindAndConsumeN, input, re, Arg(std::forward<A>(a))...);
}
#endif

static bool Replace(string* str,
const RE2& re,
const StringPiece& rewrite);

static int GlobalReplace(string* str,
const RE2& re,
const StringPiece& rewrite);

static bool Extract(const StringPiece& text,
const RE2& re,
const StringPiece& rewrite,
string* out);

static string QuoteMeta(const StringPiece& unquoted);

bool PossibleMatchRange(string* min, string* max, int maxlen) const;


enum Anchor {
UNANCHORED,         
ANCHOR_START,       
ANCHOR_BOTH         
};

int NumberOfCapturingGroups() const;

const std::map<string, int>& NamedCapturingGroups() const;

const std::map<int, string>& CapturingGroupNames() const;

bool Match(const StringPiece& text,
size_t startpos,
size_t endpos,
Anchor re_anchor,
StringPiece* submatch,
int nsubmatch) const;

bool CheckRewriteString(const StringPiece& rewrite, string* error) const;

static int MaxSubmatch(const StringPiece& rewrite);

bool Rewrite(string* out,
const StringPiece& rewrite,
const StringPiece* vec,
int veclen) const;

class Options {
public:

static const int kDefaultMaxMem = 8<<20;

enum Encoding {
EncodingUTF8 = 1,
EncodingLatin1
};

Options() :
encoding_(EncodingUTF8),
posix_syntax_(false),
longest_match_(false),
log_errors_(true),
max_mem_(kDefaultMaxMem),
literal_(false),
never_nl_(false),
dot_nl_(false),
never_capture_(false),
case_sensitive_(true),
perl_classes_(false),
word_boundary_(false),
one_line_(false) {
}

Options(CannedOptions);

Encoding encoding() const { return encoding_; }
void set_encoding(Encoding encoding) { encoding_ = encoding; }

bool utf8() const { return encoding_ == EncodingUTF8; }
void set_utf8(bool b) {
if (b) {
encoding_ = EncodingUTF8;
} else {
encoding_ = EncodingLatin1;
}
}

bool posix_syntax() const { return posix_syntax_; }
void set_posix_syntax(bool b) { posix_syntax_ = b; }

bool longest_match() const { return longest_match_; }
void set_longest_match(bool b) { longest_match_ = b; }

bool log_errors() const { return log_errors_; }
void set_log_errors(bool b) { log_errors_ = b; }

int64_t max_mem() const { return max_mem_; }
void set_max_mem(int64_t m) { max_mem_ = m; }

bool literal() const { return literal_; }
void set_literal(bool b) { literal_ = b; }

bool never_nl() const { return never_nl_; }
void set_never_nl(bool b) { never_nl_ = b; }

bool dot_nl() const { return dot_nl_; }
void set_dot_nl(bool b) { dot_nl_ = b; }

bool never_capture() const { return never_capture_; }
void set_never_capture(bool b) { never_capture_ = b; }

bool case_sensitive() const { return case_sensitive_; }
void set_case_sensitive(bool b) { case_sensitive_ = b; }

bool perl_classes() const { return perl_classes_; }
void set_perl_classes(bool b) { perl_classes_ = b; }

bool word_boundary() const { return word_boundary_; }
void set_word_boundary(bool b) { word_boundary_ = b; }

bool one_line() const { return one_line_; }
void set_one_line(bool b) { one_line_ = b; }

void Copy(const Options& src) {
*this = src;
}

int ParseFlags() const;

private:
Encoding encoding_;
bool posix_syntax_;
bool longest_match_;
bool log_errors_;
int64_t max_mem_;
bool literal_;
bool never_nl_;
bool dot_nl_;
bool never_capture_;
bool case_sensitive_;
bool perl_classes_;
bool word_boundary_;
bool one_line_;
};

const Options& options() const { return options_; };

static inline Arg CRadix(short* x);
static inline Arg CRadix(unsigned short* x);
static inline Arg CRadix(int* x);
static inline Arg CRadix(unsigned int* x);
static inline Arg CRadix(long* x);
static inline Arg CRadix(unsigned long* x);
static inline Arg CRadix(long long* x);
static inline Arg CRadix(unsigned long long* x);

static inline Arg Hex(short* x);
static inline Arg Hex(unsigned short* x);
static inline Arg Hex(int* x);
static inline Arg Hex(unsigned int* x);
static inline Arg Hex(long* x);
static inline Arg Hex(unsigned long* x);
static inline Arg Hex(long long* x);
static inline Arg Hex(unsigned long long* x);

static inline Arg Octal(short* x);
static inline Arg Octal(unsigned short* x);
static inline Arg Octal(int* x);
static inline Arg Octal(unsigned int* x);
static inline Arg Octal(long* x);
static inline Arg Octal(unsigned long* x);
static inline Arg Octal(long long* x);
static inline Arg Octal(unsigned long long* x);

private:
void Init(const StringPiece& pattern, const Options& options);

bool DoMatch(const StringPiece& text,
Anchor re_anchor,
size_t* consumed,
const Arg* const args[],
int n) const;

re2::Prog* ReverseProg() const;

string        pattern_;          
Options       options_;          
string        prefix_;           
bool          prefix_foldcase_;  
re2::Regexp*  entire_regexp_;    
re2::Regexp*  suffix_regexp_;    
re2::Prog*    prog_;             
bool          is_one_pass_;      

mutable re2::Prog*     rprog_;         
mutable const string*  error_;         
mutable ErrorCode      error_code_;    
mutable string         error_arg_;     
mutable int            num_captures_;  

mutable const std::map<string, int>* named_groups_;

mutable const std::map<int, string>* group_names_;

mutable std::once_flag rprog_once_;
mutable std::once_flag num_captures_once_;
mutable std::once_flag named_groups_once_;
mutable std::once_flag group_names_once_;

RE2(const RE2&) = delete;
RE2& operator=(const RE2&) = delete;
};




template <class T>
class _RE2_MatchObject {
public:
static inline bool Parse(const char* str, size_t n, void* dest) {
if (dest == NULL) return true;
T* object = reinterpret_cast<T*>(dest);
return object->ParseFrom(str, n);
}
};

class RE2::Arg {
public:
Arg();

Arg(void*);
Arg(std::nullptr_t);

typedef bool (*Parser)(const char* str, size_t n, void* dest);

#define MAKE_PARSER(type, name)            \
Arg(type* p) : arg_(p), parser_(name) {} \
Arg(type* p, Parser parser) : arg_(p), parser_(parser) {}

MAKE_PARSER(char,               parse_char);
MAKE_PARSER(signed char,        parse_schar);
MAKE_PARSER(unsigned char,      parse_uchar);
MAKE_PARSER(float,              parse_float);
MAKE_PARSER(double,             parse_double);
MAKE_PARSER(string,             parse_string);
MAKE_PARSER(StringPiece,        parse_stringpiece);

MAKE_PARSER(short,              parse_short);
MAKE_PARSER(unsigned short,     parse_ushort);
MAKE_PARSER(int,                parse_int);
MAKE_PARSER(unsigned int,       parse_uint);
MAKE_PARSER(long,               parse_long);
MAKE_PARSER(unsigned long,      parse_ulong);
MAKE_PARSER(long long,          parse_longlong);
MAKE_PARSER(unsigned long long, parse_ulonglong);

#undef MAKE_PARSER

template <class T> Arg(T* p)
: arg_(p), parser_(_RE2_MatchObject<T>::Parse) { }
template <class T> Arg(T* p, Parser parser)
: arg_(p), parser_(parser) { }

bool Parse(const char* str, size_t n) const;

private:
void*         arg_;
Parser        parser_;

static bool parse_null          (const char* str, size_t n, void* dest);
static bool parse_char          (const char* str, size_t n, void* dest);
static bool parse_schar         (const char* str, size_t n, void* dest);
static bool parse_uchar         (const char* str, size_t n, void* dest);
static bool parse_float         (const char* str, size_t n, void* dest);
static bool parse_double        (const char* str, size_t n, void* dest);
static bool parse_string        (const char* str, size_t n, void* dest);
static bool parse_stringpiece   (const char* str, size_t n, void* dest);

#define DECLARE_INTEGER_PARSER(name)                                       \
private:                                                                  \
static bool parse_##name(const char* str, size_t n, void* dest);         \
static bool parse_##name##_radix(const char* str, size_t n, void* dest,  \
int radix);                             \
\
public:                                                                   \
static bool parse_##name##_hex(const char* str, size_t n, void* dest);   \
static bool parse_##name##_octal(const char* str, size_t n, void* dest); \
static bool parse_##name##_cradix(const char* str, size_t n, void* dest)

DECLARE_INTEGER_PARSER(short);
DECLARE_INTEGER_PARSER(ushort);
DECLARE_INTEGER_PARSER(int);
DECLARE_INTEGER_PARSER(uint);
DECLARE_INTEGER_PARSER(long);
DECLARE_INTEGER_PARSER(ulong);
DECLARE_INTEGER_PARSER(longlong);
DECLARE_INTEGER_PARSER(ulonglong);

#undef DECLARE_INTEGER_PARSER

};

inline RE2::Arg::Arg() : arg_(NULL), parser_(parse_null) { }
inline RE2::Arg::Arg(void* p) : arg_(p), parser_(parse_null) { }
inline RE2::Arg::Arg(std::nullptr_t p) : arg_(p), parser_(parse_null) { }

inline bool RE2::Arg::Parse(const char* str, size_t n) const {
return (*parser_)(str, n, arg_);
}

#define MAKE_INTEGER_PARSER(type, name)                    \
inline RE2::Arg RE2::Hex(type* ptr) {                    \
return RE2::Arg(ptr, RE2::Arg::parse_##name##_hex);    \
}                                                        \
inline RE2::Arg RE2::Octal(type* ptr) {                  \
return RE2::Arg(ptr, RE2::Arg::parse_##name##_octal);  \
}                                                        \
inline RE2::Arg RE2::CRadix(type* ptr) {                 \
return RE2::Arg(ptr, RE2::Arg::parse_##name##_cradix); \
}

MAKE_INTEGER_PARSER(short,              short)
MAKE_INTEGER_PARSER(unsigned short,     ushort)
MAKE_INTEGER_PARSER(int,                int)
MAKE_INTEGER_PARSER(unsigned int,       uint)
MAKE_INTEGER_PARSER(long,               long)
MAKE_INTEGER_PARSER(unsigned long,      ulong)
MAKE_INTEGER_PARSER(long long,          longlong)
MAKE_INTEGER_PARSER(unsigned long long, ulonglong)

#undef MAKE_INTEGER_PARSER

#ifndef SWIG

#if defined(__clang__)
#elif defined(__GNUC__) && __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

class LazyRE2 {
private:
struct NoArg {};

public:
typedef RE2 element_type;  


RE2& operator*() const { return *get(); }
RE2* operator->() const { return get(); }

RE2* get() const {
std::call_once(once_, &LazyRE2::Init, this);
return ptr_;
}

const char* pattern_;
RE2::CannedOptions options_;
NoArg barrier_against_excess_initializers_;

mutable RE2* ptr_;
mutable std::once_flag once_;

private:
static void Init(const LazyRE2* lazy_re2) {
lazy_re2->ptr_ = new RE2(lazy_re2->pattern_, lazy_re2->options_);
}

void operator=(const LazyRE2&);  
};
#endif  

}  

using re2::RE2;
using re2::LazyRE2;

#endif  

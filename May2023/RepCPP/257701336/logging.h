

#ifndef TENSORFLOW_CORE_PLATFORM_DEFAULT_LOGGING_H_
#define TENSORFLOW_CORE_PLATFORM_DEFAULT_LOGGING_H_


#include <limits>
#include <sstream>
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

#undef ERROR

namespace tensorflow {
const int INFO = 0;            
const int WARNING = 1;         
const int ERROR = 2;           
const int FATAL = 3;           
const int NUM_SEVERITIES = 4;  

namespace internal {

class LogMessage : public std::basic_ostringstream<char> {
public:
LogMessage(const char* fname, int line, int severity);
~LogMessage();

static int64 MinVLogLevel();

static bool VmoduleActivated(const char* fname, int level);

protected:
void GenerateLogMessage();

private:
const char* fname_;
int line_;
int severity_;
};

struct Voidifier {
template <typename T>
void operator&(const T&)const {}
};

class LogMessageFatal : public LogMessage {
public:
LogMessageFatal(const char* file, int line) TF_ATTRIBUTE_COLD;
TF_ATTRIBUTE_NORETURN ~LogMessageFatal();
};

#define _TF_LOG_INFO \
::tensorflow::internal::LogMessage(__FILE__, __LINE__, ::tensorflow::INFO)
#define _TF_LOG_WARNING \
::tensorflow::internal::LogMessage(__FILE__, __LINE__, ::tensorflow::WARNING)
#define _TF_LOG_ERROR \
::tensorflow::internal::LogMessage(__FILE__, __LINE__, ::tensorflow::ERROR)
#define _TF_LOG_FATAL \
::tensorflow::internal::LogMessageFatal(__FILE__, __LINE__)

#define _TF_LOG_QFATAL _TF_LOG_FATAL

#define LOG(severity) _TF_LOG_##severity

#ifdef IS_MOBILE_PLATFORM

#define VLOG_IS_ON(lvl) ((lvl) <= 0)

#else

#define VLOG_IS_ON(lvl)                                                     \
(([](int level, const char* fname) {                                      \
static const bool vmodule_activated =                                   \
::tensorflow::internal::LogMessage::VmoduleActivated(fname, level); \
return vmodule_activated;                                               \
})(lvl, __FILE__))

#endif

#define VLOG(level)                                              \
TF_PREDICT_TRUE(!VLOG_IS_ON(level))                            \
? (void)0                                                      \
: ::tensorflow::internal::Voidifier() &                        \
::tensorflow::internal::LogMessage(__FILE__, __LINE__, \
tensorflow::INFO)

#define CHECK(condition)              \
if (TF_PREDICT_FALSE(!(condition))) \
LOG(FATAL) << "Check failed: " #condition " "

template <typename T>
inline const T& GetReferenceableValue(const T& t) {
return t;
}
inline char GetReferenceableValue(char t) { return t; }
inline unsigned char GetReferenceableValue(unsigned char t) { return t; }
inline signed char GetReferenceableValue(signed char t) { return t; }
inline short GetReferenceableValue(short t) { return t; }
inline unsigned short GetReferenceableValue(unsigned short t) { return t; }
inline int GetReferenceableValue(int t) { return t; }
inline unsigned int GetReferenceableValue(unsigned int t) { return t; }
inline long GetReferenceableValue(long t) { return t; }
inline unsigned long GetReferenceableValue(unsigned long t) { return t; }
inline long long GetReferenceableValue(long long t) { return t; }
inline unsigned long long GetReferenceableValue(unsigned long long t) {
return t;
}

template <typename T>
inline void MakeCheckOpValueString(std::ostream* os, const T& v) {
(*os) << v;
}

template <>
void MakeCheckOpValueString(std::ostream* os, const char& v);
template <>
void MakeCheckOpValueString(std::ostream* os, const signed char& v);
template <>
void MakeCheckOpValueString(std::ostream* os, const unsigned char& v);

#if LANG_CXX11
template <>
void MakeCheckOpValueString(std::ostream* os, const std::nullptr_t& p);
#endif

struct CheckOpString {
CheckOpString(string* str) : str_(str) {}
operator bool() const { return TF_PREDICT_FALSE(str_ != NULL); }
string* str_;
};

template <typename T1, typename T2>
string* MakeCheckOpString(const T1& v1, const T2& v2,
const char* exprtext) TF_ATTRIBUTE_NOINLINE;

class CheckOpMessageBuilder {
public:
explicit CheckOpMessageBuilder(const char* exprtext);
~CheckOpMessageBuilder();
std::ostream* ForVar1() { return stream_; }
std::ostream* ForVar2();
string* NewString();

private:
std::ostringstream* stream_;
};

template <typename T1, typename T2>
string* MakeCheckOpString(const T1& v1, const T2& v2, const char* exprtext) {
CheckOpMessageBuilder comb(exprtext);
MakeCheckOpValueString(comb.ForVar1(), v1);
MakeCheckOpValueString(comb.ForVar2(), v2);
return comb.NewString();
}

#define TF_DEFINE_CHECK_OP_IMPL(name, op)                                 \
template <typename T1, typename T2>                                     \
inline string* name##Impl(const T1& v1, const T2& v2,                   \
const char* exprtext) {                       \
if (TF_PREDICT_TRUE(v1 op v2))                                        \
return NULL;                                                        \
else                                                                  \
return ::tensorflow::internal::MakeCheckOpString(v1, v2, exprtext); \
}                                                                       \
inline string* name##Impl(int v1, int v2, const char* exprtext) {       \
return name##Impl<int, int>(v1, v2, exprtext);                        \
}                                                                       \
inline string* name##Impl(const size_t v1, const int v2,                \
const char* exprtext) {                       \
if (TF_PREDICT_FALSE(v2 < 0)) {                                       \
return ::tensorflow::internal::MakeCheckOpString(v1, v2, exprtext); \
}                                                                     \
const size_t uval = (size_t)((unsigned)v1);                           \
return name##Impl<size_t, size_t>(uval, v2, exprtext);                \
}                                                                       \
inline string* name##Impl(const int v1, const size_t v2,                \
const char* exprtext) {                       \
if (TF_PREDICT_FALSE(v2 >= std::numeric_limits<int>::max())) {        \
return ::tensorflow::internal::MakeCheckOpString(v1, v2, exprtext); \
}                                                                     \
const size_t uval = (size_t)((unsigned)v2);                           \
return name##Impl<size_t, size_t>(v1, uval, exprtext);                \
}

TF_DEFINE_CHECK_OP_IMPL(Check_EQ,
==)  
TF_DEFINE_CHECK_OP_IMPL(Check_NE, !=)  
TF_DEFINE_CHECK_OP_IMPL(Check_LE, <=)
TF_DEFINE_CHECK_OP_IMPL(Check_LT, <)
TF_DEFINE_CHECK_OP_IMPL(Check_GE, >=)
TF_DEFINE_CHECK_OP_IMPL(Check_GT, >)
#undef TF_DEFINE_CHECK_OP_IMPL

#define CHECK_OP_LOG(name, op, val1, val2)                            \
while (::tensorflow::internal::CheckOpString _result =              \
::tensorflow::internal::name##Impl(                      \
::tensorflow::internal::GetReferenceableValue(val1), \
::tensorflow::internal::GetReferenceableValue(val2), \
#val1 " " #op " " #val2))                            \
::tensorflow::internal::LogMessageFatal(__FILE__, __LINE__) << *(_result.str_)

#define CHECK_OP(name, op, val1, val2) CHECK_OP_LOG(name, op, val1, val2)

#define CHECK_EQ(val1, val2) CHECK_OP(Check_EQ, ==, val1, val2)
#define CHECK_NE(val1, val2) CHECK_OP(Check_NE, !=, val1, val2)
#define CHECK_LE(val1, val2) CHECK_OP(Check_LE, <=, val1, val2)
#define CHECK_LT(val1, val2) CHECK_OP(Check_LT, <, val1, val2)
#define CHECK_GE(val1, val2) CHECK_OP(Check_GE, >=, val1, val2)
#define CHECK_GT(val1, val2) CHECK_OP(Check_GT, >, val1, val2)
#define CHECK_NOTNULL(val)                                 \
::tensorflow::internal::CheckNotNull(__FILE__, __LINE__, \
"'" #val "' Must be non NULL", (val))

#ifndef NDEBUG
#define DCHECK(condition) CHECK(condition)
#define DCHECK_EQ(val1, val2) CHECK_EQ(val1, val2)
#define DCHECK_NE(val1, val2) CHECK_NE(val1, val2)
#define DCHECK_LE(val1, val2) CHECK_LE(val1, val2)
#define DCHECK_LT(val1, val2) CHECK_LT(val1, val2)
#define DCHECK_GE(val1, val2) CHECK_GE(val1, val2)
#define DCHECK_GT(val1, val2) CHECK_GT(val1, val2)

#else

#define DCHECK(condition) \
while (false && (condition)) LOG(FATAL)

#define _TF_DCHECK_NOP(x, y) \
while (false && ((void)(x), (void)(y), 0)) LOG(FATAL)

#define DCHECK_EQ(x, y) _TF_DCHECK_NOP(x, y)
#define DCHECK_NE(x, y) _TF_DCHECK_NOP(x, y)
#define DCHECK_LE(x, y) _TF_DCHECK_NOP(x, y)
#define DCHECK_LT(x, y) _TF_DCHECK_NOP(x, y)
#define DCHECK_GE(x, y) _TF_DCHECK_NOP(x, y)
#define DCHECK_GT(x, y) _TF_DCHECK_NOP(x, y)

#endif

#define QCHECK(condition) CHECK(condition)
#define QCHECK_EQ(x, y) CHECK_EQ(x, y)
#define QCHECK_NE(x, y) CHECK_NE(x, y)
#define QCHECK_LE(x, y) CHECK_LE(x, y)
#define QCHECK_LT(x, y) CHECK_LT(x, y)
#define QCHECK_GE(x, y) CHECK_GE(x, y)
#define QCHECK_GT(x, y) CHECK_GT(x, y)

template <typename T>
T&& CheckNotNull(const char* file, int line, const char* exprtext, T&& t) {
if (t == nullptr) {
LogMessageFatal(file, line) << string(exprtext);
}
return std::forward<T>(t);
}

int64 MinLogLevelFromEnv();

int64 MinVLogLevelFromEnv();

}  
}  

#endif  

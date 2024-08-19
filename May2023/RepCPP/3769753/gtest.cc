
#include "gtest/gtest.h"
#include "gtest/gtest-spi.h"

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <wchar.h>
#include <wctype.h>

#include <algorithm>
#include <iomanip>
#include <limits>
#include <ostream>  
#include <sstream>
#include <vector>

#if GTEST_OS_LINUX

# define GTEST_HAS_GETTIMEOFDAY_ 1

# include <fcntl.h>  
# include <limits.h>  
# include <sched.h>  
# include <strings.h>  
# include <sys/mman.h>  
# include <sys/time.h>  
# include <unistd.h>  
# include <string>

#elif GTEST_OS_SYMBIAN
# define GTEST_HAS_GETTIMEOFDAY_ 1
# include <sys/time.h>  

#elif GTEST_OS_ZOS
# define GTEST_HAS_GETTIMEOFDAY_ 1
# include <sys/time.h>  

# include <strings.h>  

#elif GTEST_OS_WINDOWS_MOBILE  

# include <windows.h>  

#elif GTEST_OS_WINDOWS  

# include <io.h>  
# include <sys/timeb.h>  
# include <sys/types.h>  
# include <sys/stat.h>  

# if GTEST_OS_WINDOWS_MINGW
#  define GTEST_HAS_GETTIMEOFDAY_ 1
#  include <sys/time.h>  
# endif  

# include <windows.h>  

#else

# define GTEST_HAS_GETTIMEOFDAY_ 1

# include <sys/time.h>  
# include <unistd.h>  

#endif  

#if GTEST_HAS_EXCEPTIONS
# include <stdexcept>
#endif

#if GTEST_CAN_STREAM_RESULTS_
# include <arpa/inet.h>  
# include <netdb.h>  
#endif

#define GTEST_IMPLEMENTATION_ 1
#include "src/gtest-internal-inl.h"
#undef GTEST_IMPLEMENTATION_

#if GTEST_OS_WINDOWS
# define vsnprintf _vsnprintf
#endif  

namespace testing {

using internal::CountIf;
using internal::ForEach;
using internal::GetElementOr;
using internal::Shuffle;


static const char kDisableTestFilter[] = "DISABLED_*:*/DISABLED_*";

static const char kDeathTestCaseFilter[] = "*DeathTest:*DeathTest,
const char* ,
const char* ,
const TestPartResultArray& results,
TestPartResult::Type type,
const string& substr) {
const std::string expected(type == TestPartResult::kFatalFailure ?
"1 fatal failure" :
"1 non-fatal failure");
Message msg;
if (results.size() != 1) {
msg << "Expected: " << expected << "\n"
<< "  Actual: " << results.size() << " failures";
for (int i = 0; i < results.size(); i++) {
msg << "\n" << results.GetTestPartResult(i);
}
return AssertionFailure() << msg;
}

const TestPartResult& r = results.GetTestPartResult(0);
if (r.type() != type) {
return AssertionFailure() << "Expected: " << expected << "\n"
<< "  Actual:\n"
<< r;
}

if (strstr(r.message(), substr.c_str()) == NULL) {
return AssertionFailure() << "Expected: " << expected << " containing \""
<< substr << "\"\n"
<< "  Actual:\n"
<< r;
}

return AssertionSuccess();
}

SingleFailureChecker:: SingleFailureChecker(
const TestPartResultArray* results,
TestPartResult::Type type,
const string& substr)
: results_(results),
type_(type),
substr_(substr) {}

SingleFailureChecker::~SingleFailureChecker() {
EXPECT_PRED_FORMAT3(HasOneFailure, *results_, type_, substr_);
}

DefaultGlobalTestPartResultReporter::DefaultGlobalTestPartResultReporter(
UnitTestImpl* unit_test) : unit_test_(unit_test) {}

void DefaultGlobalTestPartResultReporter::ReportTestPartResult(
const TestPartResult& result) {
unit_test_->current_test_result()->AddTestPartResult(result);
unit_test_->listeners()->repeater()->OnTestPartResult(result);
}

DefaultPerThreadTestPartResultReporter::DefaultPerThreadTestPartResultReporter(
UnitTestImpl* unit_test) : unit_test_(unit_test) {}

void DefaultPerThreadTestPartResultReporter::ReportTestPartResult(
const TestPartResult& result) {
unit_test_->GetGlobalTestPartResultReporter()->ReportTestPartResult(result);
}

TestPartResultReporterInterface*
UnitTestImpl::GetGlobalTestPartResultReporter() {
internal::MutexLock lock(&global_test_part_result_reporter_mutex_);
return global_test_part_result_repoter_;
}

void UnitTestImpl::SetGlobalTestPartResultReporter(
TestPartResultReporterInterface* reporter) {
internal::MutexLock lock(&global_test_part_result_reporter_mutex_);
global_test_part_result_repoter_ = reporter;
}

TestPartResultReporterInterface*
UnitTestImpl::GetTestPartResultReporterForCurrentThread() {
return per_thread_test_part_result_reporter_.get();
}

void UnitTestImpl::SetTestPartResultReporterForCurrentThread(
TestPartResultReporterInterface* reporter) {
per_thread_test_part_result_reporter_.set(reporter);
}

int UnitTestImpl::successful_test_case_count() const {
return CountIf(test_cases_, TestCasePassed);
}

int UnitTestImpl::failed_test_case_count() const {
return CountIf(test_cases_, TestCaseFailed);
}

int UnitTestImpl::total_test_case_count() const {
return static_cast<int>(test_cases_.size());
}

int UnitTestImpl::test_case_to_run_count() const {
return CountIf(test_cases_, ShouldRunTestCase);
}

int UnitTestImpl::successful_test_count() const {
return SumOverTestCaseList(test_cases_, &TestCase::successful_test_count);
}

int UnitTestImpl::failed_test_count() const {
return SumOverTestCaseList(test_cases_, &TestCase::failed_test_count);
}

int UnitTestImpl::reportable_disabled_test_count() const {
return SumOverTestCaseList(test_cases_,
&TestCase::reportable_disabled_test_count);
}

int UnitTestImpl::disabled_test_count() const {
return SumOverTestCaseList(test_cases_, &TestCase::disabled_test_count);
}

int UnitTestImpl::reportable_test_count() const {
return SumOverTestCaseList(test_cases_, &TestCase::reportable_test_count);
}

int UnitTestImpl::total_test_count() const {
return SumOverTestCaseList(test_cases_, &TestCase::total_test_count);
}

int UnitTestImpl::test_to_run_count() const {
return SumOverTestCaseList(test_cases_, &TestCase::test_to_run_count);
}

std::string UnitTestImpl::CurrentOsStackTraceExceptTop(int skip_count) {
(void)skip_count;
return "";
}

TimeInMillis GetTimeInMillis() {
#if GTEST_OS_WINDOWS_MOBILE || defined(__BORLANDC__)
const TimeInMillis kJavaEpochToWinFileTimeDelta =
static_cast<TimeInMillis>(116444736UL) * 100000UL;
const DWORD kTenthMicrosInMilliSecond = 10000;

SYSTEMTIME now_systime;
FILETIME now_filetime;
ULARGE_INTEGER now_int64;
GetSystemTime(&now_systime);
if (SystemTimeToFileTime(&now_systime, &now_filetime)) {
now_int64.LowPart = now_filetime.dwLowDateTime;
now_int64.HighPart = now_filetime.dwHighDateTime;
now_int64.QuadPart = (now_int64.QuadPart / kTenthMicrosInMilliSecond) -
kJavaEpochToWinFileTimeDelta;
return now_int64.QuadPart;
}
return 0;
#elif GTEST_OS_WINDOWS && !GTEST_HAS_GETTIMEOFDAY_
__timeb64 now;

# ifdef _MSC_VER

#  pragma warning(push)          
#  pragma warning(disable:4996)  
_ftime64(&now);
#  pragma warning(pop)           
# else

_ftime64(&now);

# endif  

return static_cast<TimeInMillis>(now.time) * 1000 + now.millitm;
#elif GTEST_HAS_GETTIMEOFDAY_
struct timeval now;
gettimeofday(&now, NULL);
return static_cast<TimeInMillis>(now.tv_sec) * 1000 + now.tv_usec / 1000;
#else
# error "Don't know how to get the current time on your system."
#endif
}



#if GTEST_OS_WINDOWS_MOBILE
LPCWSTR String::AnsiToUtf16(const char* ansi) {
if (!ansi) return NULL;
const int length = strlen(ansi);
const int unicode_length =
MultiByteToWideChar(CP_ACP, 0, ansi, length,
NULL, 0);
WCHAR* unicode = new WCHAR[unicode_length + 1];
MultiByteToWideChar(CP_ACP, 0, ansi, length,
unicode, unicode_length);
unicode[unicode_length] = 0;
return unicode;
}

const char* String::Utf16ToAnsi(LPCWSTR utf16_str)  {
if (!utf16_str) return NULL;
const int ansi_length =
WideCharToMultiByte(CP_ACP, 0, utf16_str, -1,
NULL, 0, NULL, NULL);
char* ansi = new char[ansi_length + 1];
WideCharToMultiByte(CP_ACP, 0, utf16_str, -1,
ansi, ansi_length, NULL, NULL);
ansi[ansi_length] = 0;
return ansi;
}

#endif  

bool String::CStringEquals(const char * lhs, const char * rhs) {
if ( lhs == NULL ) return rhs == NULL;

if ( rhs == NULL ) return false;

return strcmp(lhs, rhs) == 0;
}

#if GTEST_HAS_STD_WSTRING || GTEST_HAS_GLOBAL_WSTRING

static void StreamWideCharsToMessage(const wchar_t* wstr, size_t length,
Message* msg) {
for (size_t i = 0; i != length; ) {  
if (wstr[i] != L'\0') {
*msg << WideStringToUtf8(wstr + i, static_cast<int>(length - i));
while (i != length && wstr[i] != L'\0')
i++;
} else {
*msg << '\0';
i++;
}
}
}

#endif  

}  

Message::Message() : ss_(new ::std::stringstream) {
*ss_ << std::setprecision(std::numeric_limits<double>::digits10 + 2);
}

Message& Message::operator <<(const wchar_t* wide_c_str) {
return *this << internal::String::ShowWideCString(wide_c_str);
}
Message& Message::operator <<(wchar_t* wide_c_str) {
return *this << internal::String::ShowWideCString(wide_c_str);
}

#if GTEST_HAS_STD_WSTRING
Message& Message::operator <<(const ::std::wstring& wstr) {
internal::StreamWideCharsToMessage(wstr.c_str(), wstr.length(), this);
return *this;
}
#endif  

#if GTEST_HAS_GLOBAL_WSTRING
Message& Message::operator <<(const ::wstring& wstr) {
internal::StreamWideCharsToMessage(wstr.c_str(), wstr.length(), this);
return *this;
}
#endif  

std::string Message::GetString() const {
return internal::StringStreamToString(ss_.get());
}

AssertionResult::AssertionResult(const AssertionResult& other)
: success_(other.success_),
message_(other.message_.get() != NULL ?
new ::std::string(*other.message_) :
static_cast< ::std::string*>(NULL)) {
}

AssertionResult AssertionResult::operator!() const {
AssertionResult negation(!success_);
if (message_.get() != NULL)
negation << *message_;
return negation;
}

AssertionResult AssertionSuccess() {
return AssertionResult(true);
}

AssertionResult AssertionFailure() {
return AssertionResult(false);
}

AssertionResult AssertionFailure(const Message& message) {
return AssertionFailure() << message;
}

namespace internal {

AssertionResult EqFailure(const char* expected_expression,
const char* actual_expression,
const std::string& expected_value,
const std::string& actual_value,
bool ignoring_case) {
Message msg;
msg << "Value of: " << actual_expression;
if (actual_value != actual_expression) {
msg << "\n  Actual: " << actual_value;
}

msg << "\nExpected: " << expected_expression;
if (ignoring_case) {
msg << " (ignoring case)";
}
if (expected_value != expected_expression) {
msg << "\nWhich is: " << expected_value;
}

return AssertionFailure() << msg;
}

std::string GetBoolAssertionFailureMessage(
const AssertionResult& assertion_result,
const char* expression_text,
const char* actual_predicate_value,
const char* expected_predicate_value) {
const char* actual_message = assertion_result.message();
Message msg;
msg << "Value of: " << expression_text
<< "\n  Actual: " << actual_predicate_value;
if (actual_message[0] != '\0')
msg << " (" << actual_message << ")";
msg << "\nExpected: " << expected_predicate_value;
return msg.GetString();
}

AssertionResult DoubleNearPredFormat(const char* expr1,
const char* expr2,
const char* abs_error_expr,
double val1,
double val2,
double abs_error) {
const double diff = fabs(val1 - val2);
if (diff <= abs_error) return AssertionSuccess();

return AssertionFailure()
<< "The difference between " << expr1 << " and " << expr2
<< " is " << diff << ", which exceeds " << abs_error_expr << ", where\n"
<< expr1 << " evaluates to " << val1 << ",\n"
<< expr2 << " evaluates to " << val2 << ", and\n"
<< abs_error_expr << " evaluates to " << abs_error << ".";
}


template <typename RawType>
AssertionResult FloatingPointLE(const char* expr1,
const char* expr2,
RawType val1,
RawType val2) {
if (val1 < val2) {
return AssertionSuccess();
}

const FloatingPoint<RawType> lhs(val1), rhs(val2);
if (lhs.AlmostEquals(rhs)) {
return AssertionSuccess();
}


::std::stringstream val1_ss;
val1_ss << std::setprecision(std::numeric_limits<RawType>::digits10 + 2)
<< val1;

::std::stringstream val2_ss;
val2_ss << std::setprecision(std::numeric_limits<RawType>::digits10 + 2)
<< val2;

return AssertionFailure()
<< "Expected: (" << expr1 << ") <= (" << expr2 << ")\n"
<< "  Actual: " << StringStreamToString(&val1_ss) << " vs "
<< StringStreamToString(&val2_ss);
}

}  

AssertionResult FloatLE(const char* expr1, const char* expr2,
float val1, float val2) {
return internal::FloatingPointLE<float>(expr1, expr2, val1, val2);
}

AssertionResult DoubleLE(const char* expr1, const char* expr2,
double val1, double val2) {
return internal::FloatingPointLE<double>(expr1, expr2, val1, val2);
}

namespace internal {

AssertionResult CmpHelperEQ(const char* expected_expression,
const char* actual_expression,
BiggestInt expected,
BiggestInt actual) {
if (expected == actual) {
return AssertionSuccess();
}

return EqFailure(expected_expression,
actual_expression,
FormatForComparisonFailureMessage(expected, actual),
FormatForComparisonFailureMessage(actual, expected),
false);
}

#define GTEST_IMPL_CMP_HELPER_(op_name, op)\
AssertionResult CmpHelper##op_name(const char* expr1, const char* expr2, \
BiggestInt val1, BiggestInt val2) {\
if (val1 op val2) {\
return AssertionSuccess();\
} else {\
return AssertionFailure() \
<< "Expected: (" << expr1 << ") " #op " (" << expr2\
<< "), actual: " << FormatForComparisonFailureMessage(val1, val2)\
<< " vs " << FormatForComparisonFailureMessage(val2, val1);\
}\
}

GTEST_IMPL_CMP_HELPER_(NE, !=)
GTEST_IMPL_CMP_HELPER_(LE, <=)
GTEST_IMPL_CMP_HELPER_(LT, < )
GTEST_IMPL_CMP_HELPER_(GE, >=)
GTEST_IMPL_CMP_HELPER_(GT, > )

#undef GTEST_IMPL_CMP_HELPER_

AssertionResult CmpHelperSTREQ(const char* expected_expression,
const char* actual_expression,
const char* expected,
const char* actual) {
if (String::CStringEquals(expected, actual)) {
return AssertionSuccess();
}

return EqFailure(expected_expression,
actual_expression,
PrintToString(expected),
PrintToString(actual),
false);
}

AssertionResult CmpHelperSTRCASEEQ(const char* expected_expression,
const char* actual_expression,
const char* expected,
const char* actual) {
if (String::CaseInsensitiveCStringEquals(expected, actual)) {
return AssertionSuccess();
}

return EqFailure(expected_expression,
actual_expression,
PrintToString(expected),
PrintToString(actual),
true);
}

AssertionResult CmpHelperSTRNE(const char* s1_expression,
const char* s2_expression,
const char* s1,
const char* s2) {
if (!String::CStringEquals(s1, s2)) {
return AssertionSuccess();
} else {
return AssertionFailure() << "Expected: (" << s1_expression << ") != ("
<< s2_expression << "), actual: \""
<< s1 << "\" vs \"" << s2 << "\"";
}
}

AssertionResult CmpHelperSTRCASENE(const char* s1_expression,
const char* s2_expression,
const char* s1,
const char* s2) {
if (!String::CaseInsensitiveCStringEquals(s1, s2)) {
return AssertionSuccess();
} else {
return AssertionFailure()
<< "Expected: (" << s1_expression << ") != ("
<< s2_expression << ") (ignoring case), actual: \""
<< s1 << "\" vs \"" << s2 << "\"";
}
}

}  

namespace {



bool IsSubstringPred(const char* needle, const char* haystack) {
if (needle == NULL || haystack == NULL)
return needle == haystack;

return strstr(haystack, needle) != NULL;
}

bool IsSubstringPred(const wchar_t* needle, const wchar_t* haystack) {
if (needle == NULL || haystack == NULL)
return needle == haystack;

return wcsstr(haystack, needle) != NULL;
}

template <typename StringType>
bool IsSubstringPred(const StringType& needle,
const StringType& haystack) {
return haystack.find(needle) != StringType::npos;
}

template <typename StringType>
AssertionResult IsSubstringImpl(
bool expected_to_be_substring,
const char* needle_expr, const char* haystack_expr,
const StringType& needle, const StringType& haystack) {
if (IsSubstringPred(needle, haystack) == expected_to_be_substring)
return AssertionSuccess();

const bool is_wide_string = sizeof(needle[0]) > 1;
const char* const begin_string_quote = is_wide_string ? "L\"" : "\"";
return AssertionFailure()
<< "Value of: " << needle_expr << "\n"
<< "  Actual: " << begin_string_quote << needle << "\"\n"
<< "Expected: " << (expected_to_be_substring ? "" : "not ")
<< "a substring of " << haystack_expr << "\n"
<< "Which is: " << begin_string_quote << haystack << "\"";
}

}  


AssertionResult IsSubstring(
const char* needle_expr, const char* haystack_expr,
const char* needle, const char* haystack) {
return IsSubstringImpl(true, needle_expr, haystack_expr, needle, haystack);
}

AssertionResult IsSubstring(
const char* needle_expr, const char* haystack_expr,
const wchar_t* needle, const wchar_t* haystack) {
return IsSubstringImpl(true, needle_expr, haystack_expr, needle, haystack);
}

AssertionResult IsNotSubstring(
const char* needle_expr, const char* haystack_expr,
const char* needle, const char* haystack) {
return IsSubstringImpl(false, needle_expr, haystack_expr, needle, haystack);
}

AssertionResult IsNotSubstring(
const char* needle_expr, const char* haystack_expr,
const wchar_t* needle, const wchar_t* haystack) {
return IsSubstringImpl(false, needle_expr, haystack_expr, needle, haystack);
}

AssertionResult IsSubstring(
const char* needle_expr, const char* haystack_expr,
const ::std::string& needle, const ::std::string& haystack) {
return IsSubstringImpl(true, needle_expr, haystack_expr, needle, haystack);
}

AssertionResult IsNotSubstring(
const char* needle_expr, const char* haystack_expr,
const ::std::string& needle, const ::std::string& haystack) {
return IsSubstringImpl(false, needle_expr, haystack_expr, needle, haystack);
}

#if GTEST_HAS_STD_WSTRING
AssertionResult IsSubstring(
const char* needle_expr, const char* haystack_expr,
const ::std::wstring& needle, const ::std::wstring& haystack) {
return IsSubstringImpl(true, needle_expr, haystack_expr, needle, haystack);
}

AssertionResult IsNotSubstring(
const char* needle_expr, const char* haystack_expr,
const ::std::wstring& needle, const ::std::wstring& haystack) {
return IsSubstringImpl(false, needle_expr, haystack_expr, needle, haystack);
}
#endif  

namespace internal {

#if GTEST_OS_WINDOWS

namespace {

AssertionResult HRESULTFailureHelper(const char* expr,
const char* expected,
long hr) {  
# if GTEST_OS_WINDOWS_MOBILE

const char error_text[] = "";

# else

const DWORD kFlags = FORMAT_MESSAGE_FROM_SYSTEM |
FORMAT_MESSAGE_IGNORE_INSERTS;
const DWORD kBufSize = 4096;
char error_text[kBufSize] = { '\0' };
DWORD message_length = ::FormatMessageA(kFlags,
0,  
hr,  
0,  
error_text,  
kBufSize,  
NULL);  
for (; message_length && IsSpace(error_text[message_length - 1]);
--message_length) {
error_text[message_length - 1] = '\0';
}

# endif  

const std::string error_hex("0x" + String::FormatHexInt(hr));
return ::testing::AssertionFailure()
<< "Expected: " << expr << " " << expected << ".\n"
<< "  Actual: " << error_hex << " " << error_text << "\n";
}

}  

AssertionResult IsHRESULTSuccess(const char* expr, long hr) {  
if (SUCCEEDED(hr)) {
return AssertionSuccess();
}
return HRESULTFailureHelper(expr, "succeeds", hr);
}

AssertionResult IsHRESULTFailure(const char* expr, long hr) {  
if (FAILED(hr)) {
return AssertionSuccess();
}
return HRESULTFailureHelper(expr, "fails", hr);
}

#endif  



const UInt32 kMaxCodePoint1 = (static_cast<UInt32>(1) <<  7) - 1;

const UInt32 kMaxCodePoint2 = (static_cast<UInt32>(1) << (5 + 6)) - 1;

const UInt32 kMaxCodePoint3 = (static_cast<UInt32>(1) << (4 + 2*6)) - 1;

const UInt32 kMaxCodePoint4 = (static_cast<UInt32>(1) << (3 + 3*6)) - 1;

inline UInt32 ChopLowBits(UInt32* bits, int n) {
const UInt32 low_bits = *bits & ((static_cast<UInt32>(1) << n) - 1);
*bits >>= n;
return low_bits;
}

std::string CodePointToUtf8(UInt32 code_point) {
if (code_point > kMaxCodePoint4) {
return "(Invalid Unicode 0x" + String::FormatHexInt(code_point) + ")";
}

char str[5];  
if (code_point <= kMaxCodePoint1) {
str[1] = '\0';
str[0] = static_cast<char>(code_point);                          
} else if (code_point <= kMaxCodePoint2) {
str[2] = '\0';
str[1] = static_cast<char>(0x80 | ChopLowBits(&code_point, 6));  
str[0] = static_cast<char>(0xC0 | code_point);                   
} else if (code_point <= kMaxCodePoint3) {
str[3] = '\0';
str[2] = static_cast<char>(0x80 | ChopLowBits(&code_point, 6));  
str[1] = static_cast<char>(0x80 | ChopLowBits(&code_point, 6));  
str[0] = static_cast<char>(0xE0 | code_point);                   
} else {  
str[4] = '\0';
str[3] = static_cast<char>(0x80 | ChopLowBits(&code_point, 6));  
str[2] = static_cast<char>(0x80 | ChopLowBits(&code_point, 6));  
str[1] = static_cast<char>(0x80 | ChopLowBits(&code_point, 6));  
str[0] = static_cast<char>(0xF0 | code_point);                   
}
return str;
}


inline bool IsUtf16SurrogatePair(wchar_t first, wchar_t second) {
return sizeof(wchar_t) == 2 &&
(first & 0xFC00) == 0xD800 && (second & 0xFC00) == 0xDC00;
}

inline UInt32 CreateCodePointFromUtf16SurrogatePair(wchar_t first,
wchar_t second) {
const UInt32 mask = (1 << 10) - 1;
return (sizeof(wchar_t) == 2) ?
(((first & mask) << 10) | (second & mask)) + 0x10000 :
static_cast<UInt32>(first);
}

std::string WideStringToUtf8(const wchar_t* str, int num_chars) {
if (num_chars == -1)
num_chars = static_cast<int>(wcslen(str));

::std::stringstream stream;
for (int i = 0; i < num_chars; ++i) {
UInt32 unicode_code_point;

if (str[i] == L'\0') {
break;
} else if (i + 1 < num_chars && IsUtf16SurrogatePair(str[i], str[i + 1])) {
unicode_code_point = CreateCodePointFromUtf16SurrogatePair(str[i],
str[i + 1]);
i++;
} else {
unicode_code_point = static_cast<UInt32>(str[i]);
}

stream << CodePointToUtf8(unicode_code_point);
}
return StringStreamToString(&stream);
}

std::string String::ShowWideCString(const wchar_t * wide_c_str) {
if (wide_c_str == NULL)  return "(null)";

return internal::WideStringToUtf8(wide_c_str, -1);
}

bool String::WideCStringEquals(const wchar_t * lhs, const wchar_t * rhs) {
if (lhs == NULL) return rhs == NULL;

if (rhs == NULL) return false;

return wcscmp(lhs, rhs) == 0;
}

AssertionResult CmpHelperSTREQ(const char* expected_expression,
const char* actual_expression,
const wchar_t* expected,
const wchar_t* actual) {
if (String::WideCStringEquals(expected, actual)) {
return AssertionSuccess();
}

return EqFailure(expected_expression,
actual_expression,
PrintToString(expected),
PrintToString(actual),
false);
}

AssertionResult CmpHelperSTRNE(const char* s1_expression,
const char* s2_expression,
const wchar_t* s1,
const wchar_t* s2) {
if (!String::WideCStringEquals(s1, s2)) {
return AssertionSuccess();
}

return AssertionFailure() << "Expected: (" << s1_expression << ") != ("
<< s2_expression << "), actual: "
<< PrintToString(s1)
<< " vs " << PrintToString(s2);
}

bool String::CaseInsensitiveCStringEquals(const char * lhs, const char * rhs) {
if (lhs == NULL)
return rhs == NULL;
if (rhs == NULL)
return false;
return posix::StrCaseCmp(lhs, rhs) == 0;
}

bool String::CaseInsensitiveWideCStringEquals(const wchar_t* lhs,
const wchar_t* rhs) {
if (lhs == NULL) return rhs == NULL;

if (rhs == NULL) return false;

#if GTEST_OS_WINDOWS
return _wcsicmp(lhs, rhs) == 0;
#elif GTEST_OS_LINUX && !GTEST_OS_LINUX_ANDROID
return wcscasecmp(lhs, rhs) == 0;
#else
wint_t left, right;
do {
left = towlower(*lhs++);
right = towlower(*rhs++);
} while (left && left == right);
return left == right;
#endif  
}

bool String::EndsWithCaseInsensitive(
const std::string& str, const std::string& suffix) {
const size_t str_len = str.length();
const size_t suffix_len = suffix.length();
return (str_len >= suffix_len) &&
CaseInsensitiveCStringEquals(str.c_str() + str_len - suffix_len,
suffix.c_str());
}

std::string String::FormatIntWidth2(int value) {
std::stringstream ss;
ss << std::setfill('0') << std::setw(2) << value;
return ss.str();
}

std::string String::FormatHexInt(int value) {
std::stringstream ss;
ss << std::hex << std::uppercase << value;
return ss.str();
}

std::string String::FormatByte(unsigned char value) {
std::stringstream ss;
ss << std::setfill('0') << std::setw(2) << std::hex << std::uppercase
<< static_cast<unsigned int>(value);
return ss.str();
}

std::string StringStreamToString(::std::stringstream* ss) {
const ::std::string& str = ss->str();
const char* const start = str.c_str();
const char* const end = start + str.length();

std::string result;
result.reserve(2 * (end - start));
for (const char* ch = start; ch != end; ++ch) {
if (*ch == '\0') {
result += "\\0";  
} else {
result += *ch;
}
}

return result;
}

std::string AppendUserMessage(const std::string& gtest_msg,
const Message& user_msg) {
const std::string user_msg_string = user_msg.GetString();
if (user_msg_string.empty()) {
return gtest_msg;
}

return gtest_msg + "\n" + user_msg_string;
}

}  


TestResult::TestResult()
: death_test_count_(0),
elapsed_time_(0) {
}

TestResult::~TestResult() {
}

const TestPartResult& TestResult::GetTestPartResult(int i) const {
if (i < 0 || i >= total_part_count())
internal::posix::Abort();
return test_part_results_.at(i);
}

const TestProperty& TestResult::GetTestProperty(int i) const {
if (i < 0 || i >= test_property_count())
internal::posix::Abort();
return test_properties_.at(i);
}

void TestResult::ClearTestPartResults() {
test_part_results_.clear();
}

void TestResult::AddTestPartResult(const TestPartResult& test_part_result) {
test_part_results_.push_back(test_part_result);
}

void TestResult::RecordProperty(const std::string& xml_element,
const TestProperty& test_property) {
if (!ValidateTestProperty(xml_element, test_property)) {
return;
}
internal::MutexLock lock(&test_properites_mutex_);
const std::vector<TestProperty>::iterator property_with_matching_key =
std::find_if(test_properties_.begin(), test_properties_.end(),
internal::TestPropertyKeyIs(test_property.key()));
if (property_with_matching_key == test_properties_.end()) {
test_properties_.push_back(test_property);
return;
}
property_with_matching_key->SetValue(test_property.value());
}

static const char* const kReservedTestSuitesAttributes[] = {
"disabled",
"errors",
"failures",
"name",
"random_seed",
"tests",
"time",
"timestamp"
};

static const char* const kReservedTestSuiteAttributes[] = {
"disabled",
"errors",
"failures",
"name",
"tests",
"time"
};

static const char* const kReservedTestCaseAttributes[] = {
"classname",
"name",
"status",
"time",
"type_param",
"value_param"
};

template <int kSize>
std::vector<std::string> ArrayAsVector(const char* const (&array)[kSize]) {
return std::vector<std::string>(array, array + kSize);
}

static std::vector<std::string> GetReservedAttributesForElement(
const std::string& xml_element) {
if (xml_element == "testsuites") {
return ArrayAsVector(kReservedTestSuitesAttributes);
} else if (xml_element == "testsuite") {
return ArrayAsVector(kReservedTestSuiteAttributes);
} else if (xml_element == "testcase") {
return ArrayAsVector(kReservedTestCaseAttributes);
} else {
GTEST_CHECK_(false) << "Unrecognized xml_element provided: " << xml_element;
}
return std::vector<std::string>();
}

static std::string FormatWordList(const std::vector<std::string>& words) {
Message word_list;
for (size_t i = 0; i < words.size(); ++i) {
if (i > 0 && words.size() > 2) {
word_list << ", ";
}
if (i == words.size() - 1) {
word_list << "and ";
}
word_list << "'" << words[i] << "'";
}
return word_list.GetString();
}

bool ValidateTestPropertyName(const std::string& property_name,
const std::vector<std::string>& reserved_names) {
if (std::find(reserved_names.begin(), reserved_names.end(), property_name) !=
reserved_names.end()) {
ADD_FAILURE() << "Reserved key used in RecordProperty(): " << property_name
<< " (" << FormatWordList(reserved_names)
<< " are reserved by " << GTEST_NAME_ << ")";
return false;
}
return true;
}

bool TestResult::ValidateTestProperty(const std::string& xml_element,
const TestProperty& test_property) {
return ValidateTestPropertyName(test_property.key(),
GetReservedAttributesForElement(xml_element));
}

void TestResult::Clear() {
test_part_results_.clear();
test_properties_.clear();
death_test_count_ = 0;
elapsed_time_ = 0;
}

bool TestResult::Failed() const {
for (int i = 0; i < total_part_count(); ++i) {
if (GetTestPartResult(i).failed())
return true;
}
return false;
}

static bool TestPartFatallyFailed(const TestPartResult& result) {
return result.fatally_failed();
}

bool TestResult::HasFatalFailure() const {
return CountIf(test_part_results_, TestPartFatallyFailed) > 0;
}

static bool TestPartNonfatallyFailed(const TestPartResult& result) {
return result.nonfatally_failed();
}

bool TestResult::HasNonfatalFailure() const {
return CountIf(test_part_results_, TestPartNonfatallyFailed) > 0;
}

int TestResult::total_part_count() const {
return static_cast<int>(test_part_results_.size());
}

int TestResult::test_property_count() const {
return static_cast<int>(test_properties_.size());
}



Test::Test()
: gtest_flag_saver_(new internal::GTestFlagSaver) {
}

Test::~Test() {
delete gtest_flag_saver_;
}

void Test::SetUp() {
}

void Test::TearDown() {
}

void Test::RecordProperty(const std::string& key, const std::string& value) {
UnitTest::GetInstance()->RecordProperty(key, value);
}

void Test::RecordProperty(const std::string& key, int value) {
Message value_message;
value_message << value;
RecordProperty(key, value_message.GetString().c_str());
}

namespace internal {

void ReportFailureInUnknownLocation(TestPartResult::Type result_type,
const std::string& message) {
UnitTest::GetInstance()->AddTestPartResult(
result_type,
NULL,  
-1,    
message,
"");   
}

}  

bool Test::HasSameFixtureClass() {
internal::UnitTestImpl* const impl = internal::GetUnitTestImpl();
const TestCase* const test_case = impl->current_test_case();

const TestInfo* const first_test_info = test_case->test_info_list()[0];
const internal::TypeId first_fixture_id = first_test_info->fixture_class_id_;
const char* const first_test_name = first_test_info->name();

const TestInfo* const this_test_info = impl->current_test_info();
const internal::TypeId this_fixture_id = this_test_info->fixture_class_id_;
const char* const this_test_name = this_test_info->name();

if (this_fixture_id != first_fixture_id) {
const bool first_is_TEST = first_fixture_id == internal::GetTestTypeId();
const bool this_is_TEST = this_fixture_id == internal::GetTestTypeId();

if (first_is_TEST || this_is_TEST) {

const char* const TEST_name =
first_is_TEST ? first_test_name : this_test_name;
const char* const TEST_F_name =
first_is_TEST ? this_test_name : first_test_name;

ADD_FAILURE()
<< "All tests in the same test case must use the same test fixture\n"
<< "class, so mixing TEST_F and TEST in the same test case is\n"
<< "illegal.  In test case " << this_test_info->test_case_name()
<< ",\n"
<< "test " << TEST_F_name << " is defined using TEST_F but\n"
<< "test " << TEST_name << " is defined using TEST.  You probably\n"
<< "want to change the TEST to TEST_F or move it to another test\n"
<< "case.";
} else {
ADD_FAILURE()
<< "All tests in the same test case must use the same test fixture\n"
<< "class.  However, in test case "
<< this_test_info->test_case_name() << ",\n"
<< "you defined test " << first_test_name
<< " and test " << this_test_name << "\n"
<< "using two different test fixture classes.  This can happen if\n"
<< "the two classes are from different namespaces or translation\n"
<< "units and have the same name.  You should probably rename one\n"
<< "of the classes to put the tests into different test cases.";
}
return false;
}

return true;
}

#if GTEST_HAS_SEH

static std::string* FormatSehExceptionMessage(DWORD exception_code,
const char* location) {
Message message;
message << "SEH exception with code 0x" << std::setbase(16) <<
exception_code << std::setbase(10) << " thrown in " << location << ".";

return new std::string(message.GetString());
}

#endif  

namespace internal {

#if GTEST_HAS_EXCEPTIONS

static std::string FormatCxxExceptionMessage(const char* description,
const char* location) {
Message message;
if (description != NULL) {
message << "C++ exception with description \"" << description << "\"";
} else {
message << "Unknown C++ exception";
}
message << " thrown in " << location << ".";

return message.GetString();
}

static std::string PrintTestPartResultToString(
const TestPartResult& test_part_result);

GoogleTestFailureException::GoogleTestFailureException(
const TestPartResult& failure)
: ::std::runtime_error(PrintTestPartResultToString(failure).c_str()) {}

#endif  


template <class T, typename Result>
Result HandleSehExceptionsInMethodIfSupported(
T* object, Result (T::*method)(), const char* location) {
#if GTEST_HAS_SEH
__try {
return (object->*method)();
} __except (internal::UnitTestOptions::GTestShouldProcessSEH(  
GetExceptionCode())) {
std::string* exception_message = FormatSehExceptionMessage(
GetExceptionCode(), location);
internal::ReportFailureInUnknownLocation(TestPartResult::kFatalFailure,
*exception_message);
delete exception_message;
return static_cast<Result>(0);
}
#else
(void)location;
return (object->*method)();
#endif  
}

template <class T, typename Result>
Result HandleExceptionsInMethodIfSupported(
T* object, Result (T::*method)(), const char* location) {
if (internal::GetUnitTestImpl()->catch_exceptions()) {
#if GTEST_HAS_EXCEPTIONS
try {
return HandleSehExceptionsInMethodIfSupported(object, method, location);
} catch (const internal::GoogleTestFailureException&) {  
throw;
} catch (const std::exception& e) {  
internal::ReportFailureInUnknownLocation(
TestPartResult::kFatalFailure,
FormatCxxExceptionMessage(e.what(), location));
} catch (...) {  
internal::ReportFailureInUnknownLocation(
TestPartResult::kFatalFailure,
FormatCxxExceptionMessage(NULL, location));
}
return static_cast<Result>(0);
#else
return HandleSehExceptionsInMethodIfSupported(object, method, location);
#endif  
} else {
return (object->*method)();
}
}

}  

void Test::Run() {
if (!HasSameFixtureClass()) return;

internal::UnitTestImpl* const impl = internal::GetUnitTestImpl();
impl->os_stack_trace_getter()->UponLeavingGTest();
internal::HandleExceptionsInMethodIfSupported(this, &Test::SetUp, "SetUp()");
if (!HasFatalFailure()) {
impl->os_stack_trace_getter()->UponLeavingGTest();
internal::HandleExceptionsInMethodIfSupported(
this, &Test::TestBody, "the test body");
}

impl->os_stack_trace_getter()->UponLeavingGTest();
internal::HandleExceptionsInMethodIfSupported(
this, &Test::TearDown, "TearDown()");
}

bool Test::HasFatalFailure() {
return internal::GetUnitTestImpl()->current_test_result()->HasFatalFailure();
}

bool Test::HasNonfatalFailure() {
return internal::GetUnitTestImpl()->current_test_result()->
HasNonfatalFailure();
}


TestInfo::TestInfo(const std::string& a_test_case_name,
const std::string& a_name,
const char* a_type_param,
const char* a_value_param,
internal::TypeId fixture_class_id,
internal::TestFactoryBase* factory)
: test_case_name_(a_test_case_name),
name_(a_name),
type_param_(a_type_param ? new std::string(a_type_param) : NULL),
value_param_(a_value_param ? new std::string(a_value_param) : NULL),
fixture_class_id_(fixture_class_id),
should_run_(false),
is_disabled_(false),
matches_filter_(false),
factory_(factory),
result_() {}

TestInfo::~TestInfo() { delete factory_; }

namespace internal {

TestInfo* MakeAndRegisterTestInfo(
const char* test_case_name,
const char* name,
const char* type_param,
const char* value_param,
TypeId fixture_class_id,
SetUpTestCaseFunc set_up_tc,
TearDownTestCaseFunc tear_down_tc,
TestFactoryBase* factory) {
TestInfo* const test_info =
new TestInfo(test_case_name, name, type_param, value_param,
fixture_class_id, factory);
GetUnitTestImpl()->AddTestInfo(set_up_tc, tear_down_tc, test_info);
return test_info;
}

#if GTEST_HAS_PARAM_TEST
void ReportInvalidTestCaseType(const char* test_case_name,
const char* file, int line) {
Message errors;
errors
<< "Attempted redefinition of test case " << test_case_name << ".\n"
<< "All tests in the same test case must use the same test fixture\n"
<< "class.  However, in test case " << test_case_name << ", you tried\n"
<< "to define a test using a fixture class different from the one\n"
<< "used earlier. This can happen if the two fixture classes are\n"
<< "from different namespaces and have the same name. You should\n"
<< "probably rename one of the classes to put the tests into different\n"
<< "test cases.";

fprintf(stderr, "%s %s", FormatFileLocation(file, line).c_str(),
errors.GetString().c_str());
}
#endif  

}  

namespace {

class TestNameIs {
public:
explicit TestNameIs(const char* name)
: name_(name) {}

bool operator()(const TestInfo * test_info) const {
return test_info && test_info->name() == name_;
}

private:
std::string name_;
};

}  

namespace internal {

void UnitTestImpl::RegisterParameterizedTests() {
#if GTEST_HAS_PARAM_TEST
if (!parameterized_tests_registered_) {
parameterized_test_registry_.RegisterTests();
parameterized_tests_registered_ = true;
}
#endif
}

}  

void TestInfo::Run() {
if (!should_run_) return;

internal::UnitTestImpl* const impl = internal::GetUnitTestImpl();
impl->set_current_test_info(this);

TestEventListener* repeater = UnitTest::GetInstance()->listeners().repeater();

repeater->OnTestStart(*this);

const TimeInMillis start = internal::GetTimeInMillis();

impl->os_stack_trace_getter()->UponLeavingGTest();

Test* const test = internal::HandleExceptionsInMethodIfSupported(
factory_, &internal::TestFactoryBase::CreateTest,
"the test fixture's constructor");

if ((test != NULL) && !Test::HasFatalFailure()) {
test->Run();
}

impl->os_stack_trace_getter()->UponLeavingGTest();
internal::HandleExceptionsInMethodIfSupported(
test, &Test::DeleteSelf_, "the test fixture's destructor");

result_.set_elapsed_time(internal::GetTimeInMillis() - start);

repeater->OnTestEnd(*this);

impl->set_current_test_info(NULL);
}


int TestCase::successful_test_count() const {
return CountIf(test_info_list_, TestPassed);
}

int TestCase::failed_test_count() const {
return CountIf(test_info_list_, TestFailed);
}

int TestCase::reportable_disabled_test_count() const {
return CountIf(test_info_list_, TestReportableDisabled);
}

int TestCase::disabled_test_count() const {
return CountIf(test_info_list_, TestDisabled);
}

int TestCase::reportable_test_count() const {
return CountIf(test_info_list_, TestReportable);
}

int TestCase::test_to_run_count() const {
return CountIf(test_info_list_, ShouldRunTest);
}

int TestCase::total_test_count() const {
return static_cast<int>(test_info_list_.size());
}

TestCase::TestCase(const char* a_name, const char* a_type_param,
Test::SetUpTestCaseFunc set_up_tc,
Test::TearDownTestCaseFunc tear_down_tc)
: name_(a_name),
type_param_(a_type_param ? new std::string(a_type_param) : NULL),
set_up_tc_(set_up_tc),
tear_down_tc_(tear_down_tc),
should_run_(false),
elapsed_time_(0) {
}

TestCase::~TestCase() {
ForEach(test_info_list_, internal::Delete<TestInfo>);
}

const TestInfo* TestCase::GetTestInfo(int i) const {
const int index = GetElementOr(test_indices_, i, -1);
return index < 0 ? NULL : test_info_list_[index];
}

TestInfo* TestCase::GetMutableTestInfo(int i) {
const int index = GetElementOr(test_indices_, i, -1);
return index < 0 ? NULL : test_info_list_[index];
}

void TestCase::AddTestInfo(TestInfo * test_info) {
test_info_list_.push_back(test_info);
test_indices_.push_back(static_cast<int>(test_indices_.size()));
}

void TestCase::Run() {
if (!should_run_) return;

internal::UnitTestImpl* const impl = internal::GetUnitTestImpl();
impl->set_current_test_case(this);

TestEventListener* repeater = UnitTest::GetInstance()->listeners().repeater();

repeater->OnTestCaseStart(*this);
impl->os_stack_trace_getter()->UponLeavingGTest();
internal::HandleExceptionsInMethodIfSupported(
this, &TestCase::RunSetUpTestCase, "SetUpTestCase()");

const internal::TimeInMillis start = internal::GetTimeInMillis();
for (int i = 0; i < total_test_count(); i++) {
GetMutableTestInfo(i)->Run();
}
elapsed_time_ = internal::GetTimeInMillis() - start;

impl->os_stack_trace_getter()->UponLeavingGTest();
internal::HandleExceptionsInMethodIfSupported(
this, &TestCase::RunTearDownTestCase, "TearDownTestCase()");

repeater->OnTestCaseEnd(*this);
impl->set_current_test_case(NULL);
}

void TestCase::ClearResult() {
ad_hoc_test_result_.Clear();
ForEach(test_info_list_, TestInfo::ClearTestResult);
}

void TestCase::ShuffleTests(internal::Random* random) {
Shuffle(random, &test_indices_);
}

void TestCase::UnshuffleTests() {
for (size_t i = 0; i < test_indices_.size(); i++) {
test_indices_[i] = static_cast<int>(i);
}
}

static std::string FormatCountableNoun(int count,
const char * singular_form,
const char * plural_form) {
return internal::StreamableToString(count) + " " +
(count == 1 ? singular_form : plural_form);
}

static std::string FormatTestCount(int test_count) {
return FormatCountableNoun(test_count, "test", "tests");
}

static std::string FormatTestCaseCount(int test_case_count) {
return FormatCountableNoun(test_case_count, "test case", "test cases");
}

static const char * TestPartResultTypeToString(TestPartResult::Type type) {
switch (type) {
case TestPartResult::kSuccess:
return "Success";

case TestPartResult::kNonFatalFailure:
case TestPartResult::kFatalFailure:
#ifdef _MSC_VER
return "error: ";
#else
return "Failure\n";
#endif
default:
return "Unknown result type";
}
}

namespace internal {

static std::string PrintTestPartResultToString(
const TestPartResult& test_part_result) {
return (Message()
<< internal::FormatFileLocation(test_part_result.file_name(),
test_part_result.line_number())
<< " " << TestPartResultTypeToString(test_part_result.type())
<< test_part_result.message()).GetString();
}

static void PrintTestPartResult(const TestPartResult& test_part_result) {
const std::string& result =
PrintTestPartResultToString(test_part_result);
printf("%s\n", result.c_str());
fflush(stdout);
#if GTEST_OS_WINDOWS && !GTEST_OS_WINDOWS_MOBILE
::OutputDebugStringA(result.c_str());
::OutputDebugStringA("\n");
#endif
}


enum GTestColor {
COLOR_DEFAULT,
COLOR_RED,
COLOR_GREEN,
COLOR_YELLOW
};

#if GTEST_OS_WINDOWS && !GTEST_OS_WINDOWS_MOBILE

WORD GetColorAttribute(GTestColor color) {
switch (color) {
case COLOR_RED:    return FOREGROUND_RED;
case COLOR_GREEN:  return FOREGROUND_GREEN;
case COLOR_YELLOW: return FOREGROUND_RED | FOREGROUND_GREEN;
default:           return 0;
}
}

#else

const char* GetAnsiColorCode(GTestColor color) {
switch (color) {
case COLOR_RED:     return "1";
case COLOR_GREEN:   return "2";
case COLOR_YELLOW:  return "3";
default:            return NULL;
};
}

#endif  

bool ShouldUseColor(bool stdout_is_tty) {
const char* const gtest_color = GTEST_FLAG(color).c_str();

if (String::CaseInsensitiveCStringEquals(gtest_color, "auto")) {
#if GTEST_OS_WINDOWS
return stdout_is_tty;
#else
const char* const term = posix::GetEnv("TERM");
const bool term_supports_color =
String::CStringEquals(term, "xterm") ||
String::CStringEquals(term, "xterm-color") ||
String::CStringEquals(term, "xterm-256color") ||
String::CStringEquals(term, "screen") ||
String::CStringEquals(term, "screen-256color") ||
String::CStringEquals(term, "linux") ||
String::CStringEquals(term, "cygwin");
return stdout_is_tty && term_supports_color;
#endif  
}

return String::CaseInsensitiveCStringEquals(gtest_color, "yes") ||
String::CaseInsensitiveCStringEquals(gtest_color, "true") ||
String::CaseInsensitiveCStringEquals(gtest_color, "t") ||
String::CStringEquals(gtest_color, "1");
}

void ColoredPrintf(GTestColor color, const char* fmt, ...) {
va_list args;
va_start(args, fmt);

#if GTEST_OS_WINDOWS_MOBILE || GTEST_OS_SYMBIAN || GTEST_OS_ZOS || GTEST_OS_IOS
const bool use_color = false;
#else
static const bool in_color_mode =
ShouldUseColor(posix::IsATTY(posix::FileNo(stdout)) != 0);
const bool use_color = in_color_mode && (color != COLOR_DEFAULT);
#endif  

if (!use_color) {
vprintf(fmt, args);
va_end(args);
return;
}

#if GTEST_OS_WINDOWS && !GTEST_OS_WINDOWS_MOBILE
const HANDLE stdout_handle = GetStdHandle(STD_OUTPUT_HANDLE);

CONSOLE_SCREEN_BUFFER_INFO buffer_info;
GetConsoleScreenBufferInfo(stdout_handle, &buffer_info);
const WORD old_color_attrs = buffer_info.wAttributes;

fflush(stdout);
SetConsoleTextAttribute(stdout_handle,
GetColorAttribute(color) | FOREGROUND_INTENSITY);
vprintf(fmt, args);

fflush(stdout);
SetConsoleTextAttribute(stdout_handle, old_color_attrs);
#else
printf("\033[0;3%sm", GetAnsiColorCode(color));
vprintf(fmt, args);
printf("\033[m");  
#endif  
va_end(args);
}

static const char kTypeParamLabel[] = "TypeParam";
static const char kValueParamLabel[] = "GetParam()";

void PrintFullTestCommentIfPresent(const TestInfo& test_info) {
const char* const type_param = test_info.type_param();
const char* const value_param = test_info.value_param();

if (type_param != NULL || value_param != NULL) {
printf(", where ");
if (type_param != NULL) {
printf("%s = %s", kTypeParamLabel, type_param);
if (value_param != NULL)
printf(" and ");
}
if (value_param != NULL) {
printf("%s = %s", kValueParamLabel, value_param);
}
}
}

class PrettyUnitTestResultPrinter : public TestEventListener {
public:
PrettyUnitTestResultPrinter() {}
static void PrintTestName(const char * test_case, const char * test) {
printf("%s.%s", test_case, test);
}

virtual void OnTestProgramStart(const UnitTest& ) {}
virtual void OnTestIterationStart(const UnitTest& unit_test, int iteration);
virtual void OnEnvironmentsSetUpStart(const UnitTest& unit_test);
virtual void OnEnvironmentsSetUpEnd(const UnitTest& ) {}
virtual void OnTestCaseStart(const TestCase& test_case);
virtual void OnTestStart(const TestInfo& test_info);
virtual void OnTestPartResult(const TestPartResult& result);
virtual void OnTestEnd(const TestInfo& test_info);
virtual void OnTestCaseEnd(const TestCase& test_case);
virtual void OnEnvironmentsTearDownStart(const UnitTest& unit_test);
virtual void OnEnvironmentsTearDownEnd(const UnitTest& ) {}
virtual void OnTestIterationEnd(const UnitTest& unit_test, int iteration);
virtual void OnTestProgramEnd(const UnitTest& ) {}

private:
static void PrintFailedTests(const UnitTest& unit_test);
};

void PrettyUnitTestResultPrinter::OnTestIterationStart(
const UnitTest& unit_test, int iteration) {
if (GTEST_FLAG(repeat) != 1)
printf("\nRepeating all tests (iteration %d) . . .\n\n", iteration + 1);

const char* const filter = GTEST_FLAG(filter).c_str();

if (!String::CStringEquals(filter, kUniversalFilter)) {
ColoredPrintf(COLOR_YELLOW,
"Note: %s filter = %s\n", GTEST_NAME_, filter);
}

if (internal::ShouldShard(kTestTotalShards, kTestShardIndex, false)) {
const Int32 shard_index = Int32FromEnvOrDie(kTestShardIndex, -1);
ColoredPrintf(COLOR_YELLOW,
"Note: This is test shard %d of %s.\n",
static_cast<int>(shard_index) + 1,
internal::posix::GetEnv(kTestTotalShards));
}

if (GTEST_FLAG(shuffle)) {
ColoredPrintf(COLOR_YELLOW,
"Note: Randomizing tests' orders with a seed of %d .\n",
unit_test.random_seed());
}

ColoredPrintf(COLOR_GREEN,  "[==========] ");
printf("Running %s from %s.\n",
FormatTestCount(unit_test.test_to_run_count()).c_str(),
FormatTestCaseCount(unit_test.test_case_to_run_count()).c_str());
fflush(stdout);
}

void PrettyUnitTestResultPrinter::OnEnvironmentsSetUpStart(
const UnitTest& ) {
ColoredPrintf(COLOR_GREEN,  "[----------] ");
printf("Global test environment set-up.\n");
fflush(stdout);
}

void PrettyUnitTestResultPrinter::OnTestCaseStart(const TestCase& test_case) {
const std::string counts =
FormatCountableNoun(test_case.test_to_run_count(), "test", "tests");
ColoredPrintf(COLOR_GREEN, "[----------] ");
printf("%s from %s", counts.c_str(), test_case.name());
if (test_case.type_param() == NULL) {
printf("\n");
} else {
printf(", where %s = %s\n", kTypeParamLabel, test_case.type_param());
}
fflush(stdout);
}

void PrettyUnitTestResultPrinter::OnTestStart(const TestInfo& test_info) {
ColoredPrintf(COLOR_GREEN,  "[ RUN      ] ");
PrintTestName(test_info.test_case_name(), test_info.name());
printf("\n");
fflush(stdout);
}

void PrettyUnitTestResultPrinter::OnTestPartResult(
const TestPartResult& result) {
if (result.type() == TestPartResult::kSuccess)
return;

PrintTestPartResult(result);
fflush(stdout);
}

void PrettyUnitTestResultPrinter::OnTestEnd(const TestInfo& test_info) {
if (test_info.result()->Passed()) {
ColoredPrintf(COLOR_GREEN, "[       OK ] ");
} else {
ColoredPrintf(COLOR_RED, "[  FAILED  ] ");
}
PrintTestName(test_info.test_case_name(), test_info.name());
if (test_info.result()->Failed())
PrintFullTestCommentIfPresent(test_info);

if (GTEST_FLAG(print_time)) {
printf(" (%s ms)\n", internal::StreamableToString(
test_info.result()->elapsed_time()).c_str());
} else {
printf("\n");
}
fflush(stdout);
}

void PrettyUnitTestResultPrinter::OnTestCaseEnd(const TestCase& test_case) {
if (!GTEST_FLAG(print_time)) return;

const std::string counts =
FormatCountableNoun(test_case.test_to_run_count(), "test", "tests");
ColoredPrintf(COLOR_GREEN, "[----------] ");
printf("%s from %s (%s ms total)\n\n",
counts.c_str(), test_case.name(),
internal::StreamableToString(test_case.elapsed_time()).c_str());
fflush(stdout);
}

void PrettyUnitTestResultPrinter::OnEnvironmentsTearDownStart(
const UnitTest& ) {
ColoredPrintf(COLOR_GREEN,  "[----------] ");
printf("Global test environment tear-down\n");
fflush(stdout);
}

void PrettyUnitTestResultPrinter::PrintFailedTests(const UnitTest& unit_test) {
const int failed_test_count = unit_test.failed_test_count();
if (failed_test_count == 0) {
return;
}

for (int i = 0; i < unit_test.total_test_case_count(); ++i) {
const TestCase& test_case = *unit_test.GetTestCase(i);
if (!test_case.should_run() || (test_case.failed_test_count() == 0)) {
continue;
}
for (int j = 0; j < test_case.total_test_count(); ++j) {
const TestInfo& test_info = *test_case.GetTestInfo(j);
if (!test_info.should_run() || test_info.result()->Passed()) {
continue;
}
ColoredPrintf(COLOR_RED, "[  FAILED  ] ");
printf("%s.%s", test_case.name(), test_info.name());
PrintFullTestCommentIfPresent(test_info);
printf("\n");
}
}
}

void PrettyUnitTestResultPrinter::OnTestIterationEnd(const UnitTest& unit_test,
int ) {
ColoredPrintf(COLOR_GREEN,  "[==========] ");
printf("%s from %s ran.",
FormatTestCount(unit_test.test_to_run_count()).c_str(),
FormatTestCaseCount(unit_test.test_case_to_run_count()).c_str());
if (GTEST_FLAG(print_time)) {
printf(" (%s ms total)",
internal::StreamableToString(unit_test.elapsed_time()).c_str());
}
printf("\n");
ColoredPrintf(COLOR_GREEN,  "[  PASSED  ] ");
printf("%s.\n", FormatTestCount(unit_test.successful_test_count()).c_str());

int num_failures = unit_test.failed_test_count();
if (!unit_test.Passed()) {
const int failed_test_count = unit_test.failed_test_count();
ColoredPrintf(COLOR_RED,  "[  FAILED  ] ");
printf("%s, listed below:\n", FormatTestCount(failed_test_count).c_str());
PrintFailedTests(unit_test);
printf("\n%2d FAILED %s\n", num_failures,
num_failures == 1 ? "TEST" : "TESTS");
}

int num_disabled = unit_test.reportable_disabled_test_count();
if (num_disabled && !GTEST_FLAG(also_run_disabled_tests)) {
if (!num_failures) {
printf("\n");  
}
ColoredPrintf(COLOR_YELLOW,
"  YOU HAVE %d DISABLED %s\n\n",
num_disabled,
num_disabled == 1 ? "TEST" : "TESTS");
}
fflush(stdout);
}


class TestEventRepeater : public TestEventListener {
public:
TestEventRepeater() : forwarding_enabled_(true) {}
virtual ~TestEventRepeater();
void Append(TestEventListener *listener);
TestEventListener* Release(TestEventListener* listener);

bool forwarding_enabled() const { return forwarding_enabled_; }
void set_forwarding_enabled(bool enable) { forwarding_enabled_ = enable; }

virtual void OnTestProgramStart(const UnitTest& unit_test);
virtual void OnTestIterationStart(const UnitTest& unit_test, int iteration);
virtual void OnEnvironmentsSetUpStart(const UnitTest& unit_test);
virtual void OnEnvironmentsSetUpEnd(const UnitTest& unit_test);
virtual void OnTestCaseStart(const TestCase& test_case);
virtual void OnTestStart(const TestInfo& test_info);
virtual void OnTestPartResult(const TestPartResult& result);
virtual void OnTestEnd(const TestInfo& test_info);
virtual void OnTestCaseEnd(const TestCase& test_case);
virtual void OnEnvironmentsTearDownStart(const UnitTest& unit_test);
virtual void OnEnvironmentsTearDownEnd(const UnitTest& unit_test);
virtual void OnTestIterationEnd(const UnitTest& unit_test, int iteration);
virtual void OnTestProgramEnd(const UnitTest& unit_test);

private:
bool forwarding_enabled_;
std::vector<TestEventListener*> listeners_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(TestEventRepeater);
};

TestEventRepeater::~TestEventRepeater() {
ForEach(listeners_, Delete<TestEventListener>);
}

void TestEventRepeater::Append(TestEventListener *listener) {
listeners_.push_back(listener);
}

TestEventListener* TestEventRepeater::Release(TestEventListener *listener) {
for (size_t i = 0; i < listeners_.size(); ++i) {
if (listeners_[i] == listener) {
listeners_.erase(listeners_.begin() + i);
return listener;
}
}

return NULL;
}

#define GTEST_REPEATER_METHOD_(Name, Type) \
void TestEventRepeater::Name(const Type& parameter) { \
if (forwarding_enabled_) { \
for (size_t i = 0; i < listeners_.size(); i++) { \
listeners_[i]->Name(parameter); \
} \
} \
}
#define GTEST_REVERSE_REPEATER_METHOD_(Name, Type) \
void TestEventRepeater::Name(const Type& parameter) { \
if (forwarding_enabled_) { \
for (int i = static_cast<int>(listeners_.size()) - 1; i >= 0; i--) { \
listeners_[i]->Name(parameter); \
} \
} \
}

GTEST_REPEATER_METHOD_(OnTestProgramStart, UnitTest)
GTEST_REPEATER_METHOD_(OnEnvironmentsSetUpStart, UnitTest)
GTEST_REPEATER_METHOD_(OnTestCaseStart, TestCase)
GTEST_REPEATER_METHOD_(OnTestStart, TestInfo)
GTEST_REPEATER_METHOD_(OnTestPartResult, TestPartResult)
GTEST_REPEATER_METHOD_(OnEnvironmentsTearDownStart, UnitTest)
GTEST_REVERSE_REPEATER_METHOD_(OnEnvironmentsSetUpEnd, UnitTest)
GTEST_REVERSE_REPEATER_METHOD_(OnEnvironmentsTearDownEnd, UnitTest)
GTEST_REVERSE_REPEATER_METHOD_(OnTestEnd, TestInfo)
GTEST_REVERSE_REPEATER_METHOD_(OnTestCaseEnd, TestCase)
GTEST_REVERSE_REPEATER_METHOD_(OnTestProgramEnd, UnitTest)

#undef GTEST_REPEATER_METHOD_
#undef GTEST_REVERSE_REPEATER_METHOD_

void TestEventRepeater::OnTestIterationStart(const UnitTest& unit_test,
int iteration) {
if (forwarding_enabled_) {
for (size_t i = 0; i < listeners_.size(); i++) {
listeners_[i]->OnTestIterationStart(unit_test, iteration);
}
}
}

void TestEventRepeater::OnTestIterationEnd(const UnitTest& unit_test,
int iteration) {
if (forwarding_enabled_) {
for (int i = static_cast<int>(listeners_.size()) - 1; i >= 0; i--) {
listeners_[i]->OnTestIterationEnd(unit_test, iteration);
}
}
}


class XmlUnitTestResultPrinter : public EmptyTestEventListener {
public:
explicit XmlUnitTestResultPrinter(const char* output_file);

virtual void OnTestIterationEnd(const UnitTest& unit_test, int iteration);

private:
static bool IsNormalizableWhitespace(char c) {
return c == 0x9 || c == 0xA || c == 0xD;
}

static bool IsValidXmlCharacter(char c) {
return IsNormalizableWhitespace(c) || c >= 0x20;
}

static std::string EscapeXml(const std::string& str, bool is_attribute);

static std::string RemoveInvalidXmlCharacters(const std::string& str);

static std::string EscapeXmlAttribute(const std::string& str) {
return EscapeXml(str, true);
}

static std::string EscapeXmlText(const char* str) {
return EscapeXml(str, false);
}

static void OutputXmlAttribute(std::ostream* stream,
const std::string& element_name,
const std::string& name,
const std::string& value);

static void OutputXmlCDataSection(::std::ostream* stream, const char* data);

static void OutputXmlTestInfo(::std::ostream* stream,
const char* test_case_name,
const TestInfo& test_info);

static void PrintXmlTestCase(::std::ostream* stream,
const TestCase& test_case);

static void PrintXmlUnitTest(::std::ostream* stream,
const UnitTest& unit_test);

static std::string TestPropertiesAsXmlAttributes(const TestResult& result);

const std::string output_file_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(XmlUnitTestResultPrinter);
};

XmlUnitTestResultPrinter::XmlUnitTestResultPrinter(const char* output_file)
: output_file_(output_file) {
if (output_file_.c_str() == NULL || output_file_.empty()) {
fprintf(stderr, "XML output file may not be null\n");
fflush(stderr);
exit(EXIT_FAILURE);
}
}

void XmlUnitTestResultPrinter::OnTestIterationEnd(const UnitTest& unit_test,
int ) {
FILE* xmlout = NULL;
FilePath output_file(output_file_);
FilePath output_dir(output_file.RemoveFileName());

if (output_dir.CreateDirectoriesRecursively()) {
xmlout = posix::FOpen(output_file_.c_str(), "w");
}
if (xmlout == NULL) {
fprintf(stderr,
"Unable to open file \"%s\"\n",
output_file_.c_str());
fflush(stderr);
exit(EXIT_FAILURE);
}
std::stringstream stream;
PrintXmlUnitTest(&stream, unit_test);
fprintf(xmlout, "%s", StringStreamToString(&stream).c_str());
fclose(xmlout);
}

std::string XmlUnitTestResultPrinter::EscapeXml(
const std::string& str, bool is_attribute) {
Message m;

for (size_t i = 0; i < str.size(); ++i) {
const char ch = str[i];
switch (ch) {
case '<':
m << "&lt;";
break;
case '>':
m << "&gt;";
break;
case '&':
m << "&amp;";
break;
case '\'':
if (is_attribute)
m << "&apos;";
else
m << '\'';
break;
case '"':
if (is_attribute)
m << "&quot;";
else
m << '"';
break;
default:
if (IsValidXmlCharacter(ch)) {
if (is_attribute && IsNormalizableWhitespace(ch))
m << "&#x" << String::FormatByte(static_cast<unsigned char>(ch))
<< ";";
else
m << ch;
}
break;
}
}

return m.GetString();
}

std::string XmlUnitTestResultPrinter::RemoveInvalidXmlCharacters(
const std::string& str) {
std::string output;
output.reserve(str.size());
for (std::string::const_iterator it = str.begin(); it != str.end(); ++it)
if (IsValidXmlCharacter(*it))
output.push_back(*it);

return output;
}


std::string FormatTimeInMillisAsSeconds(TimeInMillis ms) {
::std::stringstream ss;
ss << ms/1000.0;
return ss.str();
}

std::string FormatEpochTimeInMillisAsIso8601(TimeInMillis ms) {
time_t seconds = static_cast<time_t>(ms / 1000);
#ifdef _MSC_VER
# pragma warning(push)          
# pragma warning(disable:4996)  
const struct tm* const time_struct = localtime(&seconds);  
# pragma warning(pop)           
#else
const struct tm* const time_struct = localtime(&seconds);  
#endif
if (time_struct == NULL)
return "";  

return StreamableToString(time_struct->tm_year + 1900) + "-" +
String::FormatIntWidth2(time_struct->tm_mon + 1) + "-" +
String::FormatIntWidth2(time_struct->tm_mday) + "T" +
String::FormatIntWidth2(time_struct->tm_hour) + ":" +
String::FormatIntWidth2(time_struct->tm_min) + ":" +
String::FormatIntWidth2(time_struct->tm_sec);
}

void XmlUnitTestResultPrinter::OutputXmlCDataSection(::std::ostream* stream,
const char* data) {
const char* segment = data;
*stream << "<![CDATA[";
for (;;) {
const char* const next_segment = strstr(segment, "]]>");
if (next_segment != NULL) {
stream->write(
segment, static_cast<std::streamsize>(next_segment - segment));
*stream << "]]>]]&gt;<![CDATA[";
segment = next_segment + strlen("]]>");
} else {
*stream << segment;
break;
}
}
*stream << "]]>";
}

void XmlUnitTestResultPrinter::OutputXmlAttribute(
std::ostream* stream,
const std::string& element_name,
const std::string& name,
const std::string& value) {
const std::vector<std::string>& allowed_names =
GetReservedAttributesForElement(element_name);

GTEST_CHECK_(std::find(allowed_names.begin(), allowed_names.end(), name) !=
allowed_names.end())
<< "Attribute " << name << " is not allowed for element <" << element_name
<< ">.";

*stream << " " << name << "=\"" << EscapeXmlAttribute(value) << "\"";
}

void XmlUnitTestResultPrinter::OutputXmlTestInfo(::std::ostream* stream,
const char* test_case_name,
const TestInfo& test_info) {
const TestResult& result = *test_info.result();
const std::string kTestcase = "testcase";

*stream << "    <testcase";
OutputXmlAttribute(stream, kTestcase, "name", test_info.name());

if (test_info.value_param() != NULL) {
OutputXmlAttribute(stream, kTestcase, "value_param",
test_info.value_param());
}
if (test_info.type_param() != NULL) {
OutputXmlAttribute(stream, kTestcase, "type_param", test_info.type_param());
}

OutputXmlAttribute(stream, kTestcase, "status",
test_info.should_run() ? "run" : "notrun");
OutputXmlAttribute(stream, kTestcase, "time",
FormatTimeInMillisAsSeconds(result.elapsed_time()));
OutputXmlAttribute(stream, kTestcase, "classname", test_case_name);
*stream << TestPropertiesAsXmlAttributes(result);

int failures = 0;
for (int i = 0; i < result.total_part_count(); ++i) {
const TestPartResult& part = result.GetTestPartResult(i);
if (part.failed()) {
if (++failures == 1) {
*stream << ">\n";
}
const string location = internal::FormatCompilerIndependentFileLocation(
part.file_name(), part.line_number());
const string summary = location + "\n" + part.summary();
*stream << "      <failure message=\""
<< EscapeXmlAttribute(summary.c_str())
<< "\" type=\"\">";
const string detail = location + "\n" + part.message();
OutputXmlCDataSection(stream, RemoveInvalidXmlCharacters(detail).c_str());
*stream << "</failure>\n";
}
}

if (failures == 0)
*stream << " />\n";
else
*stream << "    </testcase>\n";
}

void XmlUnitTestResultPrinter::PrintXmlTestCase(std::ostream* stream,
const TestCase& test_case) {
const std::string kTestsuite = "testsuite";
*stream << "  <" << kTestsuite;
OutputXmlAttribute(stream, kTestsuite, "name", test_case.name());
OutputXmlAttribute(stream, kTestsuite, "tests",
StreamableToString(test_case.reportable_test_count()));
OutputXmlAttribute(stream, kTestsuite, "failures",
StreamableToString(test_case.failed_test_count()));
OutputXmlAttribute(
stream, kTestsuite, "disabled",
StreamableToString(test_case.reportable_disabled_test_count()));
OutputXmlAttribute(stream, kTestsuite, "errors", "0");
OutputXmlAttribute(stream, kTestsuite, "time",
FormatTimeInMillisAsSeconds(test_case.elapsed_time()));
*stream << TestPropertiesAsXmlAttributes(test_case.ad_hoc_test_result())
<< ">\n";

for (int i = 0; i < test_case.total_test_count(); ++i) {
if (test_case.GetTestInfo(i)->is_reportable())
OutputXmlTestInfo(stream, test_case.name(), *test_case.GetTestInfo(i));
}
*stream << "  </" << kTestsuite << ">\n";
}

void XmlUnitTestResultPrinter::PrintXmlUnitTest(std::ostream* stream,
const UnitTest& unit_test) {
const std::string kTestsuites = "testsuites";

*stream << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
*stream << "<" << kTestsuites;

OutputXmlAttribute(stream, kTestsuites, "tests",
StreamableToString(unit_test.reportable_test_count()));
OutputXmlAttribute(stream, kTestsuites, "failures",
StreamableToString(unit_test.failed_test_count()));
OutputXmlAttribute(
stream, kTestsuites, "disabled",
StreamableToString(unit_test.reportable_disabled_test_count()));
OutputXmlAttribute(stream, kTestsuites, "errors", "0");
OutputXmlAttribute(
stream, kTestsuites, "timestamp",
FormatEpochTimeInMillisAsIso8601(unit_test.start_timestamp()));
OutputXmlAttribute(stream, kTestsuites, "time",
FormatTimeInMillisAsSeconds(unit_test.elapsed_time()));

if (GTEST_FLAG(shuffle)) {
OutputXmlAttribute(stream, kTestsuites, "random_seed",
StreamableToString(unit_test.random_seed()));
}

*stream << TestPropertiesAsXmlAttributes(unit_test.ad_hoc_test_result());

OutputXmlAttribute(stream, kTestsuites, "name", "AllTests");
*stream << ">\n";

for (int i = 0; i < unit_test.total_test_case_count(); ++i) {
if (unit_test.GetTestCase(i)->reportable_test_count() > 0)
PrintXmlTestCase(stream, *unit_test.GetTestCase(i));
}
*stream << "</" << kTestsuites << ">\n";
}

std::string XmlUnitTestResultPrinter::TestPropertiesAsXmlAttributes(
const TestResult& result) {
Message attributes;
for (int i = 0; i < result.test_property_count(); ++i) {
const TestProperty& property = result.GetTestProperty(i);
attributes << " " << property.key() << "="
<< "\"" << EscapeXmlAttribute(property.value()) << "\"";
}
return attributes.GetString();
}


#if GTEST_CAN_STREAM_RESULTS_

string StreamingListener::UrlEncode(const char* str) {
string result;
result.reserve(strlen(str) + 1);
for (char ch = *str; ch != '\0'; ch = *++str) {
switch (ch) {
case '%':
case '=':
case '&':
case '\n':
result.append("%" + String::FormatByte(static_cast<unsigned char>(ch)));
break;
default:
result.push_back(ch);
break;
}
}
return result;
}

void StreamingListener::SocketWriter::MakeConnection() {
GTEST_CHECK_(sockfd_ == -1)
<< "MakeConnection() can't be called when there is already a connection.";

addrinfo hints;
memset(&hints, 0, sizeof(hints));
hints.ai_family = AF_UNSPEC;    
hints.ai_socktype = SOCK_STREAM;
addrinfo* servinfo = NULL;

const int error_num = getaddrinfo(
host_name_.c_str(), port_num_.c_str(), &hints, &servinfo);
if (error_num != 0) {
GTEST_LOG_(WARNING) << "stream_result_to: getaddrinfo() failed: "
<< gai_strerror(error_num);
}

for (addrinfo* cur_addr = servinfo; sockfd_ == -1 && cur_addr != NULL;
cur_addr = cur_addr->ai_next) {
sockfd_ = socket(
cur_addr->ai_family, cur_addr->ai_socktype, cur_addr->ai_protocol);
if (sockfd_ != -1) {
if (connect(sockfd_, cur_addr->ai_addr, cur_addr->ai_addrlen) == -1) {
close(sockfd_);
sockfd_ = -1;
}
}
}

freeaddrinfo(servinfo);  

if (sockfd_ == -1) {
GTEST_LOG_(WARNING) << "stream_result_to: failed to connect to "
<< host_name_ << ":" << port_num_;
}
}

#endif  


ScopedTrace::ScopedTrace(const char* file, int line, const Message& message)
GTEST_LOCK_EXCLUDED_(&UnitTest::mutex_) {
TraceInfo trace;
trace.file = file;
trace.line = line;
trace.message = message.GetString();

UnitTest::GetInstance()->PushGTestTrace(trace);
}

ScopedTrace::~ScopedTrace()
GTEST_LOCK_EXCLUDED_(&UnitTest::mutex_) {
UnitTest::GetInstance()->PopGTestTrace();
}



string OsStackTraceGetter::CurrentStackTrace(int ,
int )
GTEST_LOCK_EXCLUDED_(mutex_) {
return "";
}

void OsStackTraceGetter::UponLeavingGTest()
GTEST_LOCK_EXCLUDED_(mutex_) {
}

const char* const
OsStackTraceGetter::kElidedFramesMarker =
"... " GTEST_NAME_ " internal frames ...";

class ScopedPrematureExitFile {
public:
explicit ScopedPrematureExitFile(const char* premature_exit_filepath)
: premature_exit_filepath_(premature_exit_filepath) {
if (premature_exit_filepath != NULL && *premature_exit_filepath != '\0') {
FILE* pfile = posix::FOpen(premature_exit_filepath, "w");
fwrite("0", 1, 1, pfile);
fclose(pfile);
}
}

~ScopedPrematureExitFile() {
if (premature_exit_filepath_ != NULL && *premature_exit_filepath_ != '\0') {
remove(premature_exit_filepath_);
}
}

private:
const char* const premature_exit_filepath_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(ScopedPrematureExitFile);
};

}  


TestEventListeners::TestEventListeners()
: repeater_(new internal::TestEventRepeater()),
default_result_printer_(NULL),
default_xml_generator_(NULL) {
}

TestEventListeners::~TestEventListeners() { delete repeater_; }

void TestEventListeners::Append(TestEventListener* listener) {
repeater_->Append(listener);
}

TestEventListener* TestEventListeners::Release(TestEventListener* listener) {
if (listener == default_result_printer_)
default_result_printer_ = NULL;
else if (listener == default_xml_generator_)
default_xml_generator_ = NULL;
return repeater_->Release(listener);
}

TestEventListener* TestEventListeners::repeater() { return repeater_; }

void TestEventListeners::SetDefaultResultPrinter(TestEventListener* listener) {
if (default_result_printer_ != listener) {
delete Release(default_result_printer_);
default_result_printer_ = listener;
if (listener != NULL)
Append(listener);
}
}

void TestEventListeners::SetDefaultXmlGenerator(TestEventListener* listener) {
if (default_xml_generator_ != listener) {
delete Release(default_xml_generator_);
default_xml_generator_ = listener;
if (listener != NULL)
Append(listener);
}
}

bool TestEventListeners::EventForwardingEnabled() const {
return repeater_->forwarding_enabled();
}

void TestEventListeners::SuppressEventForwarding() {
repeater_->set_forwarding_enabled(false);
}


UnitTest* UnitTest::GetInstance() {


#if (_MSC_VER == 1310 && !defined(_DEBUG)) || defined(__BORLANDC__)
static UnitTest* const instance = new UnitTest;
return instance;
#else
static UnitTest instance;
return &instance;
#endif  
}

int UnitTest::successful_test_case_count() const {
return impl()->successful_test_case_count();
}

int UnitTest::failed_test_case_count() const {
return impl()->failed_test_case_count();
}

int UnitTest::total_test_case_count() const {
return impl()->total_test_case_count();
}

int UnitTest::test_case_to_run_count() const {
return impl()->test_case_to_run_count();
}

int UnitTest::successful_test_count() const {
return impl()->successful_test_count();
}

int UnitTest::failed_test_count() const { return impl()->failed_test_count(); }

int UnitTest::reportable_disabled_test_count() const {
return impl()->reportable_disabled_test_count();
}

int UnitTest::disabled_test_count() const {
return impl()->disabled_test_count();
}

int UnitTest::reportable_test_count() const {
return impl()->reportable_test_count();
}

int UnitTest::total_test_count() const { return impl()->total_test_count(); }

int UnitTest::test_to_run_count() const { return impl()->test_to_run_count(); }

internal::TimeInMillis UnitTest::start_timestamp() const {
return impl()->start_timestamp();
}

internal::TimeInMillis UnitTest::elapsed_time() const {
return impl()->elapsed_time();
}

bool UnitTest::Passed() const { return impl()->Passed(); }

bool UnitTest::Failed() const { return impl()->Failed(); }

const TestCase* UnitTest::GetTestCase(int i) const {
return impl()->GetTestCase(i);
}

const TestResult& UnitTest::ad_hoc_test_result() const {
return *impl()->ad_hoc_test_result();
}

TestCase* UnitTest::GetMutableTestCase(int i) {
return impl()->GetMutableTestCase(i);
}

TestEventListeners& UnitTest::listeners() {
return *impl()->listeners();
}

Environment* UnitTest::AddEnvironment(Environment* env) {
if (env == NULL) {
return NULL;
}

impl_->environments().push_back(env);
return env;
}

void UnitTest::AddTestPartResult(
TestPartResult::Type result_type,
const char* file_name,
int line_number,
const std::string& message,
const std::string& os_stack_trace) GTEST_LOCK_EXCLUDED_(mutex_) {
Message msg;
msg << message;

internal::MutexLock lock(&mutex_);
if (impl_->gtest_trace_stack().size() > 0) {
msg << "\n" << GTEST_NAME_ << " trace:";

for (int i = static_cast<int>(impl_->gtest_trace_stack().size());
i > 0; --i) {
const internal::TraceInfo& trace = impl_->gtest_trace_stack()[i - 1];
msg << "\n" << internal::FormatFileLocation(trace.file, trace.line)
<< " " << trace.message;
}
}

if (os_stack_trace.c_str() != NULL && !os_stack_trace.empty()) {
msg << internal::kStackTraceMarker << os_stack_trace;
}

const TestPartResult result =
TestPartResult(result_type, file_name, line_number,
msg.GetString().c_str());
impl_->GetTestPartResultReporterForCurrentThread()->
ReportTestPartResult(result);

if (result_type != TestPartResult::kSuccess) {
if (GTEST_FLAG(break_on_failure)) {
#if GTEST_OS_WINDOWS
DebugBreak();
#else
*static_cast<volatile int*>(NULL) = 1;
#endif  
} else if (GTEST_FLAG(throw_on_failure)) {
#if GTEST_HAS_EXCEPTIONS
throw internal::GoogleTestFailureException(result);
#else
exit(1);
#endif
}
}
}

void UnitTest::RecordProperty(const std::string& key,
const std::string& value) {
impl_->RecordProperty(TestProperty(key, value));
}

int UnitTest::Run() {
const bool in_death_test_child_process =
internal::GTEST_FLAG(internal_run_death_test).length() > 0;


const internal::ScopedPrematureExitFile premature_exit_file(
in_death_test_child_process ?
NULL : internal::posix::GetEnv("TEST_PREMATURE_EXIT_FILE"));

impl()->set_catch_exceptions(GTEST_FLAG(catch_exceptions));

#if GTEST_HAS_SEH
if (impl()->catch_exceptions() || in_death_test_child_process) {
# if !GTEST_OS_WINDOWS_MOBILE
SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOALIGNMENTFAULTEXCEPT |
SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
# endif  

# if (defined(_MSC_VER) || GTEST_OS_WINDOWS_MINGW) && !GTEST_OS_WINDOWS_MOBILE
_set_error_mode(_OUT_TO_STDERR);
# endif

# if _MSC_VER >= 1400 && !GTEST_OS_WINDOWS_MOBILE
if (!GTEST_FLAG(break_on_failure))
_set_abort_behavior(
0x0,                                    
_WRITE_ABORT_MSG | _CALL_REPORTFAULT);  
# endif
}
#endif  

return internal::HandleExceptionsInMethodIfSupported(
impl(),
&internal::UnitTestImpl::RunAllTests,
"auxiliary test code (environments or event listeners)") ? 0 : 1;
}

const char* UnitTest::original_working_dir() const {
return impl_->original_working_dir_.c_str();
}

const TestCase* UnitTest::current_test_case() const
GTEST_LOCK_EXCLUDED_(mutex_) {
internal::MutexLock lock(&mutex_);
return impl_->current_test_case();
}

const TestInfo* UnitTest::current_test_info() const
GTEST_LOCK_EXCLUDED_(mutex_) {
internal::MutexLock lock(&mutex_);
return impl_->current_test_info();
}

int UnitTest::random_seed() const { return impl_->random_seed(); }

#if GTEST_HAS_PARAM_TEST
internal::ParameterizedTestCaseRegistry&
UnitTest::parameterized_test_registry()
GTEST_LOCK_EXCLUDED_(mutex_) {
return impl_->parameterized_test_registry();
}
#endif  

UnitTest::UnitTest() {
impl_ = new internal::UnitTestImpl(this);
}

UnitTest::~UnitTest() {
delete impl_;
}

void UnitTest::PushGTestTrace(const internal::TraceInfo& trace)
GTEST_LOCK_EXCLUDED_(mutex_) {
internal::MutexLock lock(&mutex_);
impl_->gtest_trace_stack().push_back(trace);
}

void UnitTest::PopGTestTrace()
GTEST_LOCK_EXCLUDED_(mutex_) {
internal::MutexLock lock(&mutex_);
impl_->gtest_trace_stack().pop_back();
}

namespace internal {

UnitTestImpl::UnitTestImpl(UnitTest* parent)
: parent_(parent),
#ifdef _MSC_VER
# pragma warning(push)                    
# pragma warning(disable:4355)            
default_global_test_part_result_reporter_(this),
default_per_thread_test_part_result_reporter_(this),
# pragma warning(pop)                     
#else
default_global_test_part_result_reporter_(this),
default_per_thread_test_part_result_reporter_(this),
#endif  
global_test_part_result_repoter_(
&default_global_test_part_result_reporter_),
per_thread_test_part_result_reporter_(
&default_per_thread_test_part_result_reporter_),
#if GTEST_HAS_PARAM_TEST
parameterized_test_registry_(),
parameterized_tests_registered_(false),
#endif  
last_death_test_case_(-1),
current_test_case_(NULL),
current_test_info_(NULL),
ad_hoc_test_result_(),
os_stack_trace_getter_(NULL),
post_flag_parse_init_performed_(false),
random_seed_(0),  
random_(0),  
start_timestamp_(0),
elapsed_time_(0),
#if GTEST_HAS_DEATH_TEST
death_test_factory_(new DefaultDeathTestFactory),
#endif
catch_exceptions_(false) {
listeners()->SetDefaultResultPrinter(new PrettyUnitTestResultPrinter);
}

UnitTestImpl::~UnitTestImpl() {
ForEach(test_cases_, internal::Delete<TestCase>);

ForEach(environments_, internal::Delete<Environment>);

delete os_stack_trace_getter_;
}

void UnitTestImpl::RecordProperty(const TestProperty& test_property) {
std::string xml_element;
TestResult* test_result;  

if (current_test_info_ != NULL) {
xml_element = "testcase";
test_result = &(current_test_info_->result_);
} else if (current_test_case_ != NULL) {
xml_element = "testsuite";
test_result = &(current_test_case_->ad_hoc_test_result_);
} else {
xml_element = "testsuites";
test_result = &ad_hoc_test_result_;
}
test_result->RecordProperty(xml_element, test_property);
}

#if GTEST_HAS_DEATH_TEST
void UnitTestImpl::SuppressTestEventsIfInSubprocess() {
if (internal_run_death_test_flag_.get() != NULL)
listeners()->SuppressEventForwarding();
}
#endif  

void UnitTestImpl::ConfigureXmlOutput() {
const std::string& output_format = UnitTestOptions::GetOutputFormat();
if (output_format == "xml") {
listeners()->SetDefaultXmlGenerator(new XmlUnitTestResultPrinter(
UnitTestOptions::GetAbsolutePathToOutputFile().c_str()));
} else if (output_format != "") {
printf("WARNING: unrecognized output format \"%s\" ignored.\n",
output_format.c_str());
fflush(stdout);
}
}

#if GTEST_CAN_STREAM_RESULTS_
void UnitTestImpl::ConfigureStreamingOutput() {
const std::string& target = GTEST_FLAG(stream_result_to);
if (!target.empty()) {
const size_t pos = target.find(':');
if (pos != std::string::npos) {
listeners()->Append(new StreamingListener(target.substr(0, pos),
target.substr(pos+1)));
} else {
printf("WARNING: unrecognized streaming target \"%s\" ignored.\n",
target.c_str());
fflush(stdout);
}
}
}
#endif  

void UnitTestImpl::PostFlagParsingInit() {
if (!post_flag_parse_init_performed_) {
post_flag_parse_init_performed_ = true;

#if GTEST_HAS_DEATH_TEST
InitDeathTestSubprocessControlInfo();
SuppressTestEventsIfInSubprocess();
#endif  

RegisterParameterizedTests();

ConfigureXmlOutput();

#if GTEST_CAN_STREAM_RESULTS_
ConfigureStreamingOutput();
#endif  
}
}

class TestCaseNameIs {
public:
explicit TestCaseNameIs(const std::string& name)
: name_(name) {}

bool operator()(const TestCase* test_case) const {
return test_case != NULL && strcmp(test_case->name(), name_.c_str()) == 0;
}

private:
std::string name_;
};

TestCase* UnitTestImpl::GetTestCase(const char* test_case_name,
const char* type_param,
Test::SetUpTestCaseFunc set_up_tc,
Test::TearDownTestCaseFunc tear_down_tc) {
const std::vector<TestCase*>::const_iterator test_case =
std::find_if(test_cases_.begin(), test_cases_.end(),
TestCaseNameIs(test_case_name));

if (test_case != test_cases_.end())
return *test_case;

TestCase* const new_test_case =
new TestCase(test_case_name, type_param, set_up_tc, tear_down_tc);

if (internal::UnitTestOptions::MatchesFilter(test_case_name,
kDeathTestCaseFilter)) {
++last_death_test_case_;
test_cases_.insert(test_cases_.begin() + last_death_test_case_,
new_test_case);
} else {
test_cases_.push_back(new_test_case);
}

test_case_indices_.push_back(static_cast<int>(test_case_indices_.size()));
return new_test_case;
}

static void SetUpEnvironment(Environment* env) { env->SetUp(); }
static void TearDownEnvironment(Environment* env) { env->TearDown(); }

bool UnitTestImpl::RunAllTests() {
if (!GTestIsInitialized()) {
printf("%s",
"\nThis test program did NOT call ::testing::InitGoogleTest "
"before calling RUN_ALL_TESTS().  Please fix it.\n");
return false;
}

if (g_help_flag)
return true;

PostFlagParsingInit();

internal::WriteToShardStatusFileIfNeeded();

bool in_subprocess_for_death_test = false;

#if GTEST_HAS_DEATH_TEST
in_subprocess_for_death_test = (internal_run_death_test_flag_.get() != NULL);
#endif  

const bool should_shard = ShouldShard(kTestTotalShards, kTestShardIndex,
in_subprocess_for_death_test);

const bool has_tests_to_run = FilterTests(should_shard
? HONOR_SHARDING_PROTOCOL
: IGNORE_SHARDING_PROTOCOL) > 0;

if (GTEST_FLAG(list_tests)) {
ListTestsMatchingFilter();
return true;
}

random_seed_ = GTEST_FLAG(shuffle) ?
GetRandomSeedFromFlag(GTEST_FLAG(random_seed)) : 0;

bool failed = false;

TestEventListener* repeater = listeners()->repeater();

start_timestamp_ = GetTimeInMillis();
repeater->OnTestProgramStart(*parent_);

const int repeat = in_subprocess_for_death_test ? 1 : GTEST_FLAG(repeat);
const bool forever = repeat < 0;
for (int i = 0; forever || i != repeat; i++) {
ClearNonAdHocTestResult();

const TimeInMillis start = GetTimeInMillis();

if (has_tests_to_run && GTEST_FLAG(shuffle)) {
random()->Reseed(random_seed_);
ShuffleTests();
}

repeater->OnTestIterationStart(*parent_, i);

if (has_tests_to_run) {
repeater->OnEnvironmentsSetUpStart(*parent_);
ForEach(environments_, SetUpEnvironment);
repeater->OnEnvironmentsSetUpEnd(*parent_);

if (!Test::HasFatalFailure()) {
for (int test_index = 0; test_index < total_test_case_count();
test_index++) {
GetMutableTestCase(test_index)->Run();
}
}

repeater->OnEnvironmentsTearDownStart(*parent_);
std::for_each(environments_.rbegin(), environments_.rend(),
TearDownEnvironment);
repeater->OnEnvironmentsTearDownEnd(*parent_);
}

elapsed_time_ = GetTimeInMillis() - start;

repeater->OnTestIterationEnd(*parent_, i);

if (!Passed()) {
failed = true;
}

UnshuffleTests();

if (GTEST_FLAG(shuffle)) {
random_seed_ = GetNextRandomSeed(random_seed_);
}
}

repeater->OnTestProgramEnd(*parent_);

return !failed;
}

void WriteToShardStatusFileIfNeeded() {
const char* const test_shard_file = posix::GetEnv(kTestShardStatusFile);
if (test_shard_file != NULL) {
FILE* const file = posix::FOpen(test_shard_file, "w");
if (file == NULL) {
ColoredPrintf(COLOR_RED,
"Could not write to the test shard status file \"%s\" "
"specified by the %s environment variable.\n",
test_shard_file, kTestShardStatusFile);
fflush(stdout);
exit(EXIT_FAILURE);
}
fclose(file);
}
}

bool ShouldShard(const char* total_shards_env,
const char* shard_index_env,
bool in_subprocess_for_death_test) {
if (in_subprocess_for_death_test) {
return false;
}

const Int32 total_shards = Int32FromEnvOrDie(total_shards_env, -1);
const Int32 shard_index = Int32FromEnvOrDie(shard_index_env, -1);

if (total_shards == -1 && shard_index == -1) {
return false;
} else if (total_shards == -1 && shard_index != -1) {
const Message msg = Message()
<< "Invalid environment variables: you have "
<< kTestShardIndex << " = " << shard_index
<< ", but have left " << kTestTotalShards << " unset.\n";
ColoredPrintf(COLOR_RED, msg.GetString().c_str());
fflush(stdout);
exit(EXIT_FAILURE);
} else if (total_shards != -1 && shard_index == -1) {
const Message msg = Message()
<< "Invalid environment variables: you have "
<< kTestTotalShards << " = " << total_shards
<< ", but have left " << kTestShardIndex << " unset.\n";
ColoredPrintf(COLOR_RED, msg.GetString().c_str());
fflush(stdout);
exit(EXIT_FAILURE);
} else if (shard_index < 0 || shard_index >= total_shards) {
const Message msg = Message()
<< "Invalid environment variables: we require 0 <= "
<< kTestShardIndex << " < " << kTestTotalShards
<< ", but you have " << kTestShardIndex << "=" << shard_index
<< ", " << kTestTotalShards << "=" << total_shards << ".\n";
ColoredPrintf(COLOR_RED, msg.GetString().c_str());
fflush(stdout);
exit(EXIT_FAILURE);
}

return total_shards > 1;
}

Int32 Int32FromEnvOrDie(const char* var, Int32 default_val) {
const char* str_val = posix::GetEnv(var);
if (str_val == NULL) {
return default_val;
}

Int32 result;
if (!ParseInt32(Message() << "The value of environment variable " << var,
str_val, &result)) {
exit(EXIT_FAILURE);
}
return result;
}

bool ShouldRunTestOnShard(int total_shards, int shard_index, int test_id) {
return (test_id % total_shards) == shard_index;
}

int UnitTestImpl::FilterTests(ReactionToSharding shard_tests) {
const Int32 total_shards = shard_tests == HONOR_SHARDING_PROTOCOL ?
Int32FromEnvOrDie(kTestTotalShards, -1) : -1;
const Int32 shard_index = shard_tests == HONOR_SHARDING_PROTOCOL ?
Int32FromEnvOrDie(kTestShardIndex, -1) : -1;

int num_runnable_tests = 0;
int num_selected_tests = 0;
for (size_t i = 0; i < test_cases_.size(); i++) {
TestCase* const test_case = test_cases_[i];
const std::string &test_case_name = test_case->name();
test_case->set_should_run(false);

for (size_t j = 0; j < test_case->test_info_list().size(); j++) {
TestInfo* const test_info = test_case->test_info_list()[j];
const std::string test_name(test_info->name());
const bool is_disabled =
internal::UnitTestOptions::MatchesFilter(test_case_name,
kDisableTestFilter) ||
internal::UnitTestOptions::MatchesFilter(test_name,
kDisableTestFilter);
test_info->is_disabled_ = is_disabled;

const bool matches_filter =
internal::UnitTestOptions::FilterMatchesTest(test_case_name,
test_name);
test_info->matches_filter_ = matches_filter;

const bool is_runnable =
(GTEST_FLAG(also_run_disabled_tests) || !is_disabled) &&
matches_filter;

const bool is_selected = is_runnable &&
(shard_tests == IGNORE_SHARDING_PROTOCOL ||
ShouldRunTestOnShard(total_shards, shard_index,
num_runnable_tests));

num_runnable_tests += is_runnable;
num_selected_tests += is_selected;

test_info->should_run_ = is_selected;
test_case->set_should_run(test_case->should_run() || is_selected);
}
}
return num_selected_tests;
}

static void PrintOnOneLine(const char* str, int max_length) {
if (str != NULL) {
for (int i = 0; *str != '\0'; ++str) {
if (i >= max_length) {
printf("...");
break;
}
if (*str == '\n') {
printf("\\n");
i += 2;
} else {
printf("%c", *str);
++i;
}
}
}
}

void UnitTestImpl::ListTestsMatchingFilter() {
const int kMaxParamLength = 250;

for (size_t i = 0; i < test_cases_.size(); i++) {
const TestCase* const test_case = test_cases_[i];
bool printed_test_case_name = false;

for (size_t j = 0; j < test_case->test_info_list().size(); j++) {
const TestInfo* const test_info =
test_case->test_info_list()[j];
if (test_info->matches_filter_) {
if (!printed_test_case_name) {
printed_test_case_name = true;
printf("%s.", test_case->name());
if (test_case->type_param() != NULL) {
printf("  # %s = ", kTypeParamLabel);
PrintOnOneLine(test_case->type_param(), kMaxParamLength);
}
printf("\n");
}
printf("  %s", test_info->name());
if (test_info->value_param() != NULL) {
printf("  # %s = ", kValueParamLabel);
PrintOnOneLine(test_info->value_param(), kMaxParamLength);
}
printf("\n");
}
}
}
fflush(stdout);
}

void UnitTestImpl::set_os_stack_trace_getter(
OsStackTraceGetterInterface* getter) {
if (os_stack_trace_getter_ != getter) {
delete os_stack_trace_getter_;
os_stack_trace_getter_ = getter;
}
}

OsStackTraceGetterInterface* UnitTestImpl::os_stack_trace_getter() {
if (os_stack_trace_getter_ == NULL) {
os_stack_trace_getter_ = new OsStackTraceGetter;
}

return os_stack_trace_getter_;
}

TestResult* UnitTestImpl::current_test_result() {
return current_test_info_ ?
&(current_test_info_->result_) : &ad_hoc_test_result_;
}

void UnitTestImpl::ShuffleTests() {
ShuffleRange(random(), 0, last_death_test_case_ + 1, &test_case_indices_);

ShuffleRange(random(), last_death_test_case_ + 1,
static_cast<int>(test_cases_.size()), &test_case_indices_);

for (size_t i = 0; i < test_cases_.size(); i++) {
test_cases_[i]->ShuffleTests(random());
}
}

void UnitTestImpl::UnshuffleTests() {
for (size_t i = 0; i < test_cases_.size(); i++) {
test_cases_[i]->UnshuffleTests();
test_case_indices_[i] = static_cast<int>(i);
}
}

std::string GetCurrentOsStackTraceExceptTop(UnitTest* ,
int skip_count) {
return GetUnitTestImpl()->CurrentOsStackTraceExceptTop(skip_count + 1);
}

namespace {
class ClassUniqueToAlwaysTrue {};
}

bool IsTrue(bool condition) { return condition; }

bool AlwaysTrue() {
#if GTEST_HAS_EXCEPTIONS
if (IsTrue(false))
throw ClassUniqueToAlwaysTrue();
#endif  
return true;
}

bool SkipPrefix(const char* prefix, const char** pstr) {
const size_t prefix_len = strlen(prefix);
if (strncmp(*pstr, prefix, prefix_len) == 0) {
*pstr += prefix_len;
return true;
}
return false;
}

const char* ParseFlagValue(const char* str,
const char* flag,
bool def_optional) {
if (str == NULL || flag == NULL) return NULL;

const std::string flag_str = std::string("--") + GTEST_FLAG_PREFIX_ + flag;
const size_t flag_len = flag_str.length();
if (strncmp(str, flag_str.c_str(), flag_len) != 0) return NULL;

const char* flag_end = str + flag_len;

if (def_optional && (flag_end[0] == '\0')) {
return flag_end;
}

if (flag_end[0] != '=') return NULL;

return flag_end + 1;
}

bool ParseBoolFlag(const char* str, const char* flag, bool* value) {
const char* const value_str = ParseFlagValue(str, flag, true);

if (value_str == NULL) return false;

*value = !(*value_str == '0' || *value_str == 'f' || *value_str == 'F');
return true;
}

bool ParseInt32Flag(const char* str, const char* flag, Int32* value) {
const char* const value_str = ParseFlagValue(str, flag, false);

if (value_str == NULL) return false;

return ParseInt32(Message() << "The value of flag --" << flag,
value_str, value);
}

bool ParseStringFlag(const char* str, const char* flag, std::string* value) {
const char* const value_str = ParseFlagValue(str, flag, false);

if (value_str == NULL) return false;

*value = value_str;
return true;
}

static bool HasGoogleTestFlagPrefix(const char* str) {
return (SkipPrefix("--", &str) ||
SkipPrefix("-", &str) ||
SkipPrefix("/", &str)) &&
!SkipPrefix(GTEST_FLAG_PREFIX_ "internal_", &str) &&
(SkipPrefix(GTEST_FLAG_PREFIX_, &str) ||
SkipPrefix(GTEST_FLAG_PREFIX_DASH_, &str));
}

static void PrintColorEncoded(const char* str) {
GTestColor color = COLOR_DEFAULT;  

for (;;) {
const char* p = strchr(str, '@');
if (p == NULL) {
ColoredPrintf(color, "%s", str);
return;
}

ColoredPrintf(color, "%s", std::string(str, p).c_str());

const char ch = p[1];
str = p + 2;
if (ch == '@') {
ColoredPrintf(color, "@");
} else if (ch == 'D') {
color = COLOR_DEFAULT;
} else if (ch == 'R') {
color = COLOR_RED;
} else if (ch == 'G') {
color = COLOR_GREEN;
} else if (ch == 'Y') {
color = COLOR_YELLOW;
} else {
--str;
}
}
}

static const char kColorEncodedHelpMessage[] =
"This program contains tests written using " GTEST_NAME_ ". You can use the\n"
"following command line flags to control its behavior:\n"
"\n"
"Test Selection:\n"
"  @G--" GTEST_FLAG_PREFIX_ "list_tests@D\n"
"      List the names of all tests instead of running them. The name of\n"
"      TEST(Foo, Bar) is \"Foo.Bar\".\n"
"  @G--" GTEST_FLAG_PREFIX_ "filter=@YPOSTIVE_PATTERNS"
"[@G-@YNEGATIVE_PATTERNS]@D\n"
"      Run only the tests whose name matches one of the positive patterns but\n"
"      none of the negative patterns. '?' matches any single character; '*'\n"
"      matches any substring; ':' separates two patterns.\n"
"  @G--" GTEST_FLAG_PREFIX_ "also_run_disabled_tests@D\n"
"      Run all disabled tests too.\n"
"\n"
"Test Execution:\n"
"  @G--" GTEST_FLAG_PREFIX_ "repeat=@Y[COUNT]@D\n"
"      Run the tests repeatedly; use a negative count to repeat forever.\n"
"  @G--" GTEST_FLAG_PREFIX_ "shuffle@D\n"
"      Randomize tests' orders on every iteration.\n"
"  @G--" GTEST_FLAG_PREFIX_ "random_seed=@Y[NUMBER]@D\n"
"      Random number seed to use for shuffling test orders (between 1 and\n"
"      99999, or 0 to use a seed based on the current time).\n"
"\n"
"Test Output:\n"
"  @G--" GTEST_FLAG_PREFIX_ "color=@Y(@Gyes@Y|@Gno@Y|@Gauto@Y)@D\n"
"      Enable/disable colored output. The default is @Gauto@D.\n"
"  -@G-" GTEST_FLAG_PREFIX_ "print_time=0@D\n"
"      Don't print the elapsed time of each test.\n"
"  @G--" GTEST_FLAG_PREFIX_ "output=xml@Y[@G:@YDIRECTORY_PATH@G"
GTEST_PATH_SEP_ "@Y|@G:@YFILE_PATH]@D\n"
"      Generate an XML report in the given directory or with the given file\n"
"      name. @YFILE_PATH@D defaults to @Gtest_details.xml@D.\n"
#if GTEST_CAN_STREAM_RESULTS_
"  @G--" GTEST_FLAG_PREFIX_ "stream_result_to=@YHOST@G:@YPORT@D\n"
"      Stream test results to the given server.\n"
#endif  
"\n"
"Assertion Behavior:\n"
#if GTEST_HAS_DEATH_TEST && !GTEST_OS_WINDOWS
"  @G--" GTEST_FLAG_PREFIX_ "death_test_style=@Y(@Gfast@Y|@Gthreadsafe@Y)@D\n"
"      Set the default death test style.\n"
#endif  
"  @G--" GTEST_FLAG_PREFIX_ "break_on_failure@D\n"
"      Turn assertion failures into debugger break-points.\n"
"  @G--" GTEST_FLAG_PREFIX_ "throw_on_failure@D\n"
"      Turn assertion failures into C++ exceptions.\n"
"  @G--" GTEST_FLAG_PREFIX_ "catch_exceptions=0@D\n"
"      Do not report exceptions as test failures. Instead, allow them\n"
"      to crash the program or throw a pop-up (on Windows).\n"
"\n"
"Except for @G--" GTEST_FLAG_PREFIX_ "list_tests@D, you can alternatively set "
"the corresponding\n"
"environment variable of a flag (all letters in upper-case). For example, to\n"
"disable colored text output, you can either specify @G--" GTEST_FLAG_PREFIX_
"color=no@D or set\n"
"the @G" GTEST_FLAG_PREFIX_UPPER_ "COLOR@D environment variable to @Gno@D.\n"
"\n"
"For more information, please read the " GTEST_NAME_ " documentation at\n"
"@G" GTEST_PROJECT_URL_ "@D. If you find a bug in " GTEST_NAME_ "\n"
"(not one in your own code or tests), please report it to\n"
"@G<" GTEST_DEV_EMAIL_ ">@D.\n";

template <typename CharType>
void ParseGoogleTestFlagsOnlyImpl(int* argc, CharType** argv) {
for (int i = 1; i < *argc; i++) {
const std::string arg_string = StreamableToString(argv[i]);
const char* const arg = arg_string.c_str();

using internal::ParseBoolFlag;
using internal::ParseInt32Flag;
using internal::ParseStringFlag;

if (ParseBoolFlag(arg, kAlsoRunDisabledTestsFlag,
&GTEST_FLAG(also_run_disabled_tests)) ||
ParseBoolFlag(arg, kBreakOnFailureFlag,
&GTEST_FLAG(break_on_failure)) ||
ParseBoolFlag(arg, kCatchExceptionsFlag,
&GTEST_FLAG(catch_exceptions)) ||
ParseStringFlag(arg, kColorFlag, &GTEST_FLAG(color)) ||
ParseStringFlag(arg, kDeathTestStyleFlag,
&GTEST_FLAG(death_test_style)) ||
ParseBoolFlag(arg, kDeathTestUseFork,
&GTEST_FLAG(death_test_use_fork)) ||
ParseStringFlag(arg, kFilterFlag, &GTEST_FLAG(filter)) ||
ParseStringFlag(arg, kInternalRunDeathTestFlag,
&GTEST_FLAG(internal_run_death_test)) ||
ParseBoolFlag(arg, kListTestsFlag, &GTEST_FLAG(list_tests)) ||
ParseStringFlag(arg, kOutputFlag, &GTEST_FLAG(output)) ||
ParseBoolFlag(arg, kPrintTimeFlag, &GTEST_FLAG(print_time)) ||
ParseInt32Flag(arg, kRandomSeedFlag, &GTEST_FLAG(random_seed)) ||
ParseInt32Flag(arg, kRepeatFlag, &GTEST_FLAG(repeat)) ||
ParseBoolFlag(arg, kShuffleFlag, &GTEST_FLAG(shuffle)) ||
ParseInt32Flag(arg, kStackTraceDepthFlag,
&GTEST_FLAG(stack_trace_depth)) ||
ParseStringFlag(arg, kStreamResultToFlag,
&GTEST_FLAG(stream_result_to)) ||
ParseBoolFlag(arg, kThrowOnFailureFlag,
&GTEST_FLAG(throw_on_failure))
) {
for (int j = i; j != *argc; j++) {
argv[j] = argv[j + 1];
}

(*argc)--;

i--;
} else if (arg_string == "--help" || arg_string == "-h" ||
arg_string == "-?" || arg_string == "/?" ||
HasGoogleTestFlagPrefix(arg)) {
g_help_flag = true;
}
}

if (g_help_flag) {
PrintColorEncoded(kColorEncodedHelpMessage);
}
}

void ParseGoogleTestFlagsOnly(int* argc, char** argv) {
ParseGoogleTestFlagsOnlyImpl(argc, argv);
}
void ParseGoogleTestFlagsOnly(int* argc, wchar_t** argv) {
ParseGoogleTestFlagsOnlyImpl(argc, argv);
}

template <typename CharType>
void InitGoogleTestImpl(int* argc, CharType** argv) {
g_init_gtest_count++;

if (g_init_gtest_count != 1) return;

if (*argc <= 0) return;

internal::g_executable_path = internal::StreamableToString(argv[0]);

#if GTEST_HAS_DEATH_TEST

g_argvs.clear();
for (int i = 0; i != *argc; i++) {
g_argvs.push_back(StreamableToString(argv[i]));
}

#endif  

ParseGoogleTestFlagsOnly(argc, argv);
GetUnitTestImpl()->PostFlagParsingInit();
}

}  

void InitGoogleTest(int* argc, char** argv) {
internal::InitGoogleTestImpl(argc, argv);
}

void InitGoogleTest(int* argc, wchar_t** argv) {
internal::InitGoogleTestImpl(argc, argv);
}

}  

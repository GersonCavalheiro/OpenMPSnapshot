
#include "gtest/internal/gtest-port.h"

#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#if GTEST_OS_WINDOWS_MOBILE
# include <windows.h>  
#elif GTEST_OS_WINDOWS
# include <io.h>
# include <sys/stat.h>
#else
# include <unistd.h>
#endif  

#if GTEST_OS_MAC
# include <mach/mach_init.h>
# include <mach/task.h>
# include <mach/vm_map.h>
#endif  

#if GTEST_OS_QNX
# include <devctl.h>
# include <sys/procfs.h>
#endif  

#include "gtest/gtest-spi.h"
#include "gtest/gtest-message.h"
#include "gtest/internal/gtest-internal.h"
#include "gtest/internal/gtest-string.h"

#define GTEST_IMPLEMENTATION_ 1
#include "src/gtest-internal-inl.h"
#undef GTEST_IMPLEMENTATION_

namespace testing {
namespace internal {

#if defined(_MSC_VER) || defined(__BORLANDC__)
const int kStdOutFileno = 1;
const int kStdErrFileno = 2;
#else
const int kStdOutFileno = STDOUT_FILENO;
const int kStdErrFileno = STDERR_FILENO;
#endif  

#if GTEST_OS_MAC

size_t GetThreadCount() {
const task_t task = mach_task_self();
mach_msg_type_number_t thread_count;
thread_act_array_t thread_list;
const kern_return_t status = task_threads(task, &thread_list, &thread_count);
if (status == KERN_SUCCESS) {
vm_deallocate(task,
reinterpret_cast<vm_address_t>(thread_list),
sizeof(thread_t) * thread_count);
return static_cast<size_t>(thread_count);
} else {
return 0;
}
}

#elif GTEST_OS_QNX

size_t GetThreadCount() {
const int fd = open("/proc/self/as", O_RDONLY);
if (fd < 0) {
return 0;
}
procfs_info process_info;
const int status =
devctl(fd, DCMD_PROC_INFO, &process_info, sizeof(process_info), NULL);
close(fd);
if (status == EOK) {
return static_cast<size_t>(process_info.num_threads);
} else {
return 0;
}
}

#else

size_t GetThreadCount() {
return 0;
}

#endif  

#if GTEST_USES_POSIX_RE


RE::~RE() {
if (is_valid_) {
regfree(&partial_regex_);
regfree(&full_regex_);
}
free(const_cast<char*>(pattern_));
}

bool RE::FullMatch(const char* str, const RE& re) {
if (!re.is_valid_) return false;

regmatch_t match;
return regexec(&re.full_regex_, str, 1, &match, 0) == 0;
}

bool RE::PartialMatch(const char* str, const RE& re) {
if (!re.is_valid_) return false;

regmatch_t match;
return regexec(&re.partial_regex_, str, 1, &match, 0) == 0;
}

void RE::Init(const char* regex) {
pattern_ = posix::StrDup(regex);

const size_t full_regex_len = strlen(regex) + 10;
char* const full_pattern = new char[full_regex_len];

snprintf(full_pattern, full_regex_len, "^(%s)$", regex);
is_valid_ = regcomp(&full_regex_, full_pattern, REG_EXTENDED) == 0;
if (is_valid_) {
const char* const partial_regex = (*regex == '\0') ? "()" : regex;
is_valid_ = regcomp(&partial_regex_, partial_regex, REG_EXTENDED) == 0;
}
EXPECT_TRUE(is_valid_)
<< "Regular expression \"" << regex
<< "\" is not a valid POSIX Extended regular expression.";

delete[] full_pattern;
}

#elif GTEST_USES_SIMPLE_RE

bool IsInSet(char ch, const char* str) {
return ch != '\0' && strchr(str, ch) != NULL;
}

bool IsAsciiDigit(char ch) { return '0' <= ch && ch <= '9'; }
bool IsAsciiPunct(char ch) {
return IsInSet(ch, "^-!\"#$%&'()*+,./:;<=>?@[\\]_`{|}~");
}
bool IsRepeat(char ch) { return IsInSet(ch, "?*+"); }
bool IsAsciiWhiteSpace(char ch) { return IsInSet(ch, " \f\n\r\t\v"); }
bool IsAsciiWordChar(char ch) {
return ('a' <= ch && ch <= 'z') || ('A' <= ch && ch <= 'Z') ||
('0' <= ch && ch <= '9') || ch == '_';
}

bool IsValidEscape(char c) {
return (IsAsciiPunct(c) || IsInSet(c, "dDfnrsStvwW"));
}

bool AtomMatchesChar(bool escaped, char pattern_char, char ch) {
if (escaped) {  
switch (pattern_char) {
case 'd': return IsAsciiDigit(ch);
case 'D': return !IsAsciiDigit(ch);
case 'f': return ch == '\f';
case 'n': return ch == '\n';
case 'r': return ch == '\r';
case 's': return IsAsciiWhiteSpace(ch);
case 'S': return !IsAsciiWhiteSpace(ch);
case 't': return ch == '\t';
case 'v': return ch == '\v';
case 'w': return IsAsciiWordChar(ch);
case 'W': return !IsAsciiWordChar(ch);
}
return IsAsciiPunct(pattern_char) && pattern_char == ch;
}

return (pattern_char == '.' && ch != '\n') || pattern_char == ch;
}

std::string FormatRegexSyntaxError(const char* regex, int index) {
return (Message() << "Syntax error at index " << index
<< " in simple regular expression \"" << regex << "\": ").GetString();
}

bool ValidateRegex(const char* regex) {
if (regex == NULL) {
ADD_FAILURE() << "NULL is not a valid simple regular expression.";
return false;
}

bool is_valid = true;

bool prev_repeatable = false;
for (int i = 0; regex[i]; i++) {
if (regex[i] == '\\') {  
i++;
if (regex[i] == '\0') {
ADD_FAILURE() << FormatRegexSyntaxError(regex, i - 1)
<< "'\\' cannot appear at the end.";
return false;
}

if (!IsValidEscape(regex[i])) {
ADD_FAILURE() << FormatRegexSyntaxError(regex, i - 1)
<< "invalid escape sequence \"\\" << regex[i] << "\".";
is_valid = false;
}
prev_repeatable = true;
} else {  
const char ch = regex[i];

if (ch == '^' && i > 0) {
ADD_FAILURE() << FormatRegexSyntaxError(regex, i)
<< "'^' can only appear at the beginning.";
is_valid = false;
} else if (ch == '$' && regex[i + 1] != '\0') {
ADD_FAILURE() << FormatRegexSyntaxError(regex, i)
<< "'$' can only appear at the end.";
is_valid = false;
} else if (IsInSet(ch, "()[]{}|")) {
ADD_FAILURE() << FormatRegexSyntaxError(regex, i)
<< "'" << ch << "' is unsupported.";
is_valid = false;
} else if (IsRepeat(ch) && !prev_repeatable) {
ADD_FAILURE() << FormatRegexSyntaxError(regex, i)
<< "'" << ch << "' can only follow a repeatable token.";
is_valid = false;
}

prev_repeatable = !IsInSet(ch, "^$?*+");
}
}

return is_valid;
}

bool MatchRepetitionAndRegexAtHead(
bool escaped, char c, char repeat, const char* regex,
const char* str) {
const size_t min_count = (repeat == '+') ? 1 : 0;
const size_t max_count = (repeat == '?') ? 1 :
static_cast<size_t>(-1) - 1;

for (size_t i = 0; i <= max_count; ++i) {
if (i >= min_count && MatchRegexAtHead(regex, str + i)) {
return true;
}
if (str[i] == '\0' || !AtomMatchesChar(escaped, c, str[i]))
return false;
}
return false;
}

bool MatchRegexAtHead(const char* regex, const char* str) {
if (*regex == '\0')  
return true;

if (*regex == '$')
return *str == '\0';

const bool escaped = *regex == '\\';
if (escaped)
++regex;
if (IsRepeat(regex[1])) {
return MatchRepetitionAndRegexAtHead(
escaped, regex[0], regex[1], regex + 2, str);
} else {
return (*str != '\0') && AtomMatchesChar(escaped, *regex, *str) &&
MatchRegexAtHead(regex + 1, str + 1);
}
}

bool MatchRegexAnywhere(const char* regex, const char* str) {
if (regex == NULL || str == NULL)
return false;

if (*regex == '^')
return MatchRegexAtHead(regex + 1, str);

do {
if (MatchRegexAtHead(regex, str))
return true;
} while (*str++ != '\0');
return false;
}


RE::~RE() {
free(const_cast<char*>(pattern_));
free(const_cast<char*>(full_pattern_));
}

bool RE::FullMatch(const char* str, const RE& re) {
return re.is_valid_ && MatchRegexAnywhere(re.full_pattern_, str);
}

bool RE::PartialMatch(const char* str, const RE& re) {
return re.is_valid_ && MatchRegexAnywhere(re.pattern_, str);
}

void RE::Init(const char* regex) {
pattern_ = full_pattern_ = NULL;
if (regex != NULL) {
pattern_ = posix::StrDup(regex);
}

is_valid_ = ValidateRegex(regex);
if (!is_valid_) {
return;
}

const size_t len = strlen(regex);
char* buffer = static_cast<char*>(malloc(len + 3));
full_pattern_ = buffer;

if (*regex != '^')
*buffer++ = '^';  

memcpy(buffer, regex, len);
buffer += len;

if (len == 0 || regex[len - 1] != '$')
*buffer++ = '$';  

*buffer = '\0';
}

#endif  

const char kUnknownFile[] = "unknown file";

GTEST_API_ ::std::string FormatFileLocation(const char* file, int line) {
const std::string file_name(file == NULL ? kUnknownFile : file);

if (line < 0) {
return file_name + ":";
}
#ifdef _MSC_VER
return file_name + "(" + StreamableToString(line) + "):";
#else
return file_name + ":" + StreamableToString(line) + ":";
#endif  
}

GTEST_API_ ::std::string FormatCompilerIndependentFileLocation(
const char* file, int line) {
const std::string file_name(file == NULL ? kUnknownFile : file);

if (line < 0)
return file_name;
else
return file_name + ":" + StreamableToString(line);
}


GTestLog::GTestLog(GTestLogSeverity severity, const char* file, int line)
: severity_(severity) {
const char* const marker =
severity == GTEST_INFO ?    "[  INFO ]" :
severity == GTEST_WARNING ? "[WARNING]" :
severity == GTEST_ERROR ?   "[ ERROR ]" : "[ FATAL ]";
GetStream() << ::std::endl << marker << " "
<< FormatFileLocation(file, line).c_str() << ": ";
}

GTestLog::~GTestLog() {
GetStream() << ::std::endl;
if (severity_ == GTEST_FATAL) {
fflush(stderr);
posix::Abort();
}
}
#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 4996)
#endif  

#if GTEST_HAS_STREAM_REDIRECTION

class CapturedStream {
public:
explicit CapturedStream(int fd) : fd_(fd), uncaptured_fd_(dup(fd)) {
# if GTEST_OS_WINDOWS
char temp_dir_path[MAX_PATH + 1] = { '\0' };  
char temp_file_path[MAX_PATH + 1] = { '\0' };  

::GetTempPathA(sizeof(temp_dir_path), temp_dir_path);
const UINT success = ::GetTempFileNameA(temp_dir_path,
"gtest_redir",
0,  
temp_file_path);
GTEST_CHECK_(success != 0)
<< "Unable to create a temporary file in " << temp_dir_path;
const int captured_fd = creat(temp_file_path, _S_IREAD | _S_IWRITE);
GTEST_CHECK_(captured_fd != -1) << "Unable to open temporary file "
<< temp_file_path;
filename_ = temp_file_path;
# else
#  if GTEST_OS_LINUX_ANDROID
char name_template[] = "/sdcard/gtest_captured_stream.XXXXXX";
#  else
char name_template[] = "/tmp/captured_stream.XXXXXX";
#  endif  
const int captured_fd = mkstemp(name_template);
filename_ = name_template;
# endif  
fflush(NULL);
dup2(captured_fd, fd_);
close(captured_fd);
}

~CapturedStream() {
remove(filename_.c_str());
}

std::string GetCapturedString() {
if (uncaptured_fd_ != -1) {
fflush(NULL);
dup2(uncaptured_fd_, fd_);
close(uncaptured_fd_);
uncaptured_fd_ = -1;
}

FILE* const file = posix::FOpen(filename_.c_str(), "r");
const std::string content = ReadEntireFile(file);
posix::FClose(file);
return content;
}

private:
static std::string ReadEntireFile(FILE* file);

static size_t GetFileSize(FILE* file);

const int fd_;  
int uncaptured_fd_;
::std::string filename_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(CapturedStream);
};

size_t CapturedStream::GetFileSize(FILE* file) {
fseek(file, 0, SEEK_END);
return static_cast<size_t>(ftell(file));
}

std::string CapturedStream::ReadEntireFile(FILE* file) {
const size_t file_size = GetFileSize(file);
char* const buffer = new char[file_size];

size_t bytes_last_read = 0;  
size_t bytes_read = 0;       

fseek(file, 0, SEEK_SET);

do {
bytes_last_read = fread(buffer+bytes_read, 1, file_size-bytes_read, file);
bytes_read += bytes_last_read;
} while (bytes_last_read > 0 && bytes_read < file_size);

const std::string content(buffer, bytes_read);
delete[] buffer;

return content;
}

# ifdef _MSC_VER
#  pragma warning(pop)
# endif  

static CapturedStream* g_captured_stderr = NULL;
static CapturedStream* g_captured_stdout = NULL;

void CaptureStream(int fd, const char* stream_name, CapturedStream** stream) {
if (*stream != NULL) {
GTEST_LOG_(FATAL) << "Only one " << stream_name
<< " capturer can exist at a time.";
}
*stream = new CapturedStream(fd);
}

std::string GetCapturedStream(CapturedStream** captured_stream) {
const std::string content = (*captured_stream)->GetCapturedString();

delete *captured_stream;
*captured_stream = NULL;

return content;
}

void CaptureStdout() {
CaptureStream(kStdOutFileno, "stdout", &g_captured_stdout);
}

void CaptureStderr() {
CaptureStream(kStdErrFileno, "stderr", &g_captured_stderr);
}

std::string GetCapturedStdout() {
return GetCapturedStream(&g_captured_stdout);
}

std::string GetCapturedStderr() {
return GetCapturedStream(&g_captured_stderr);
}

#endif  

#if GTEST_HAS_DEATH_TEST

::std::vector<testing::internal::string> g_argvs;

static const ::std::vector<testing::internal::string>* g_injected_test_argvs =
NULL;  

void SetInjectableArgvs(const ::std::vector<testing::internal::string>* argvs) {
if (g_injected_test_argvs != argvs)
delete g_injected_test_argvs;
g_injected_test_argvs = argvs;
}

const ::std::vector<testing::internal::string>& GetInjectableArgvs() {
if (g_injected_test_argvs != NULL) {
return *g_injected_test_argvs;
}
return g_argvs;
}
#endif  

#if GTEST_OS_WINDOWS_MOBILE
namespace posix {
void Abort() {
DebugBreak();
TerminateProcess(GetCurrentProcess(), 1);
}
}  
#endif  

static std::string FlagToEnvVar(const char* flag) {
const std::string full_flag =
(Message() << GTEST_FLAG_PREFIX_ << flag).GetString();

Message env_var;
for (size_t i = 0; i != full_flag.length(); i++) {
env_var << ToUpper(full_flag.c_str()[i]);
}

return env_var.GetString();
}

bool ParseInt32(const Message& src_text, const char* str, Int32* value) {
char* end = NULL;
const long long_value = strtol(str, &end, 10);  

if (*end != '\0') {
Message msg;
msg << "WARNING: " << src_text
<< " is expected to be a 32-bit integer, but actually"
<< " has value \"" << str << "\".\n";
printf("%s", msg.GetString().c_str());
fflush(stdout);
return false;
}

const Int32 result = static_cast<Int32>(long_value);
if (long_value == LONG_MAX || long_value == LONG_MIN ||
result != long_value
) {
Message msg;
msg << "WARNING: " << src_text
<< " is expected to be a 32-bit integer, but actually"
<< " has value " << str << ", which overflows.\n";
printf("%s", msg.GetString().c_str());
fflush(stdout);
return false;
}

*value = result;
return true;
}

bool BoolFromGTestEnv(const char* flag, bool default_value) {
const std::string env_var = FlagToEnvVar(flag);
const char* const string_value = posix::GetEnv(env_var.c_str());
return string_value == NULL ?
default_value : strcmp(string_value, "0") != 0;
}

Int32 Int32FromGTestEnv(const char* flag, Int32 default_value) {
const std::string env_var = FlagToEnvVar(flag);
const char* const string_value = posix::GetEnv(env_var.c_str());
if (string_value == NULL) {
return default_value;
}

Int32 result = default_value;
if (!ParseInt32(Message() << "Environment variable " << env_var,
string_value, &result)) {
printf("The default value %s is used.\n",
(Message() << default_value).GetString().c_str());
fflush(stdout);
return default_value;
}

return result;
}

const char* StringFromGTestEnv(const char* flag, const char* default_value) {
const std::string env_var = FlagToEnvVar(flag);
const char* const value = posix::GetEnv(env_var.c_str());
return value == NULL ? default_value : value;
}

}  
}  

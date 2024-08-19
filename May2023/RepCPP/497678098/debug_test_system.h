


#ifndef SEQAN_INCLUDE_SEQAN_BASIC_DEBUG_TEST_SYSTEM_H_
#define SEQAN_INCLUDE_SEQAN_BASIC_DEBUG_TEST_SYSTEM_H_





#ifndef SEQAN_ENABLE_TESTING
#define SEQAN_ENABLE_TESTING 0
#endif  

#if SEQAN_ENABLE_TESTING
#undef SEQAN_ENABLE_DEBUG
#define SEQAN_ENABLE_DEBUG 1
#endif  



#ifndef SEQAN_ENABLE_DEBUG
#ifdef NDEBUG
#define SEQAN_ENABLE_DEBUG 0
#else
#define SEQAN_ENABLE_DEBUG 1
#endif
#endif  

#if !SEQAN_ENABLE_DEBUG
#define NDEBUG 1
#else
#undef NDEBUG
#endif 



#if !SEQAN_ENABLE_DEBUG
#define SEQAN_TYPEDEF_FOR_DEBUG SEQAN_UNUSED
#else
#define SEQAN_TYPEDEF_FOR_DEBUG
#endif

#include <iostream>  
#include <iomanip>
#include <cstring>   
#include <cstdlib>   
#include <cstdio>
#include <cstdarg>   
#include <algorithm> 
#include <set>
#include <vector>
#include <string>
#include <typeinfo>

#ifdef STDLIB_VS
#include <Windows.h>    
#else  
#include <unistd.h>     
#include <sys/stat.h>   
#include <dirent.h>     
#if SEQAN_HAS_EXECINFO
#include <execinfo.h>   
#endif  
#include <cxxabi.h>     
#include <signal.h>
#endif  



namespace seqan {

template <typename T>
struct Demangler
{
#if !defined(STDLIB_VS)
char *data_begin;
#else
const char *data_begin;
#endif

Demangler()
{
T t;
_demangle(*this, t);
}

Demangler(T const & t)
{
_demangle(*this, t);
}

~Demangler()
{
#if !defined(STDLIB_VS)
free(data_begin);
#endif
}
};



template <typename T>
inline void _demangle(Demangler<T> & me, T const & t)
{
#if !defined(STDLIB_VS)
int status;
me.data_begin = abi::__cxa_demangle(typeid(t).name(), NULL, NULL, &status);
#else
me.data_begin = typeid(t).name();
#endif
}


template <typename T>
inline const char * toCString(Demangler<T> const & me)
{

return me.data_begin;
}

}





#define SEQAN_FAIL(...)                                                 \
do {                                                                \
::seqan::ClassTest::forceFail(__FILE__, __LINE__,               \
__VA_ARGS__);                     \
::seqan::ClassTest::fail();                                     \
} while (false)



#define SEQAN_CHECK(_arg1, ...)                                         \
do {                                                                \
if (!::seqan::ClassTest::testTrue(__FILE__, __LINE__,           \
(_arg1), # _arg1,              \
__VA_ARGS__)) {               \
::seqan::ClassTest::fail();                                 \
}                                                               \
} while (false)

namespace seqan {

#if !defined(SEQAN_CXX_FLAGS_)
#define SEQAN_CXX_FLAGS_ SEQAN_CXX_FLAGS_NOT_SET
#endif 
#define SEQAN_MKSTRING_(str) # str
#define SEQAN_MKSTRING(str) SEQAN_MKSTRING_(str)
#define SEQAN_CXX_FLAGS SEQAN_MKSTRING(SEQAN_CXX_FLAGS_)



template <typename TStream>
void printDebugLevel(TStream & stream)
{
stream << "SEQAN_ENABLE_DEBUG == " << SEQAN_ENABLE_DEBUG << std::endl;
stream << "SEQAN_ENABLE_TESTING == " << SEQAN_ENABLE_TESTING << std::endl;
stream << "SEQAN_CXX_FLAGS == \"" << SEQAN_CXX_FLAGS << "\"" << std::endl;
stream << "SEQAN_ASYNC_IO == " << SEQAN_ASYNC_IO << std::endl;
}

#if !SEQAN_HAS_EXECINFO

template <typename TSize>
void printStackTrace(TSize )
{}

#else

template <typename TSize>
void printStackTrace(TSize maxFrames)
{
void * addrlist[256];
char temp[4096];
char addr[20];
char offset[20];

size_t size;
int status;
char * symname;
char * demangled;

std::cerr << std::endl << "stack trace:" << std::endl;

int addrlist_len = backtrace(addrlist, maxFrames);
char ** symbollist = backtrace_symbols(addrlist, addrlist_len);
for (int i = 1; i < addrlist_len; ++i)
{
offset[0] = 0;
addr[0] = 0;
demangled = NULL;


if (3 == sscanf(symbollist[i], "%*[^(](%4095[^+]+%[^)]) %s", temp, offset, addr))
{
symname = temp;
if (NULL != (demangled = abi::__cxa_demangle(temp, NULL, &size, &status)))
{
symname = demangled;
}
}
else if (3 == sscanf(symbollist[i], "%*d %*s %s %s %*s %s", addr, temp, offset))
{
symname = temp;
if (NULL != (demangled = abi::__cxa_demangle(temp, NULL, &size, &status)))
{
symname = demangled;
}
}
else if (2 == sscanf(symbollist[i], "%s %s", temp, addr))
{
symname = temp;
}
else
{
symname = symbollist[i];
}

std::cerr << std::setw(3) << i - 1;
std::cerr << std::setw(20) << addr;
std::cerr << "  " << symname;
if (offset[0] != 0)
std::cerr << " + " << offset;
std::cerr << std::endl;

free(demangled);
}
std::cerr << std::endl;
free(symbollist);
}

static void signalHandlerPrintStackTrace(int signum)
{
std::cerr << std::endl;
printStackTrace(20);
signal(signum, SIG_DFL);
kill(getpid(), signum);
}

inline int _deploySignalHandlers()
{
signal(SIGSEGV, signalHandlerPrintStackTrace);      
signal(SIGFPE, signalHandlerPrintStackTrace);       
return 0;
}

#if SEQAN_ENABLE_DEBUG


template <typename T>
struct SignalHandlersDummy_
{
static const int i;
};

template <typename T>
const int SignalHandlersDummy_<T>::i = _deploySignalHandlers();

namespace {
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#endif  
volatile int signalHandlersDummy_ = SignalHandlersDummy_<void>::i;
#ifdef __clang__
#pragma clang diagnostic pop
#endif  
}

#endif  
#endif  


namespace ClassTest {
struct AssertionFailedException {};

struct StaticData
{
static int & testCount()
{
static int result = 0;
return result;
}

static int & errorCount()
{
static int result = 0;
return result;
}

static int & skippedCount()
{
static int result = 0;
return result;
}

static bool & thisTestOk()
{
static bool result = 0;
return result;
}

static bool & thisTestSkipped()
{
static bool result = 0;
return result;
}

static const char * & currentTestName()
{
const char * defaultValue = "";
static const char * result = const_cast<char *>(defaultValue);
return result;
}

static char * & basePath()
{
const char * defaultValue = ".";
static char * result = const_cast<char *>(defaultValue);
return result;
}

static char const * _computePathToRoot()
{
const char * file = __FILE__;
int pos = -1;
for (size_t i = 0; i < strlen(file) - strlen("include"); ++i)
{
if (strncmp(file + i, "include", strlen("include")) == 0)
{
pos = i;
}
}
for (; pos > 0 && *(file + pos - 1) != '/' &&  *(file + pos - 1) != '\\'; --pos)
continue;
if (pos == -1)
{
std::cerr << "Could not extrapolate path to repository from __FILE__ == \""
<< __FILE__ << "\"" << std::endl;
exit(1);
}

static char buffer[1024];
strncpy(&buffer[0], file, pos);
buffer[pos - 1] = '\0';
return &buffer[0];
}

static char const * pathToRoot()
{
const char * result = 0;
if (!result)
result = _computePathToRoot();
return result;
}

static::std::vector<std::string> & tempFileNames()
{
static::std::vector<std::string> filenames;
return filenames;
}
};



inline
const char * tempFileName()
{
static char fileNameBuffer[1000];
#ifdef STDLIB_VS
static char filePathBuffer[1000];
DWORD dwRetVal = 0;
dwRetVal = GetTempPath(1000,            
filePathBuffer); 
if (dwRetVal > 1000 || (dwRetVal == 0))
{
std::cerr << "GetTempPath failed" << std::endl;
exit(1);
}

UINT uRetVal   = 0;
uRetVal = GetTempFileName(filePathBuffer,   
TEXT("SEQAN."),   
0,                
fileNameBuffer);  

if (uRetVal == 0)
{
std::cerr << "GetTempFileName failed" << std::endl;
exit(1);
}

DeleteFile(fileNameBuffer);
CreateDirectoryA(fileNameBuffer, NULL);
StaticData::tempFileNames().push_back(fileNameBuffer);
strcat(fileNameBuffer, "\\test_file");
return fileNameBuffer;

#else  
strcpy(fileNameBuffer, "/tmp/SEQAN.XXXXXXXXXXXXXXXXXXXX");
mode_t cur_umask = umask(S_IRWXO | S_IRWXG);  
int _tmp = mkstemp(fileNameBuffer);
(void) _tmp;
umask(cur_umask);
unlink(fileNameBuffer);
mkdir(fileNameBuffer, 0777);

StaticData::tempFileNames().push_back(fileNameBuffer);

strcat(fileNameBuffer, "/test_file");
return fileNameBuffer;

#endif  
}

inline
void beginTestSuite(const char * testSuiteName, const char * argv0)
{
std::cout << "TEST SUITE " << testSuiteName << std::endl;
printDebugLevel(std::cout);
(void)testSuiteName;
StaticData::testCount() = 0;
StaticData::skippedCount() = 0;
StaticData::errorCount() = 0;
const char * end = argv0;
const char * ptr = std::min(strchr(argv0, '\\'), strchr(argv0, '/'));     
for (; ptr != 0; ptr = std::min(strchr(ptr + 1, '\\'), strchr(ptr + 1, '/')))
end = ptr;
int rpos = end - argv0;
if (rpos <= 0)
{
StaticData::basePath() = new char[2];
strcpy(StaticData::basePath(), ".");
}
else
{
int len = rpos;
StaticData::basePath() = new char[len];
strncpy(StaticData::basePath(), argv0, len);
}

#ifdef STDLIB_VS
_set_error_mode(_OUT_TO_STDERR);
_CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
_CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
_CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
_CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
_CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif  
}

inline
std::string _stripFileName(const char * tempFilename)
{
std::string s(tempFilename);
return s.substr(0, s.find_last_of("\\/"));
}

inline
int _deleteTempFile(std::string tempFilename)
{
#ifdef STDLIB_VS
HANDLE hFind;
WIN32_FIND_DATA data;

std::string temp = tempFilename.c_str() + std::string("\\*");
hFind = FindFirstFile(temp.c_str(), &data);
if (hFind != INVALID_HANDLE_VALUE)
{
do
{
std::string tempp = tempFilename.c_str() + std::string("\\") + data.cFileName;
if (strcmp(data.cFileName, ".") == 0 || strcmp(data.cFileName, "..") == 0)
continue;  
DeleteFile(tempp.c_str());
}
while (FindNextFile(hFind, &data));
FindClose(hFind);
}

if (!RemoveDirectory(tempFilename.c_str()))
{
std::cerr << "ERROR: Could not delete directory " << tempFilename << "\n";
return 0;
}
#else  
DIR * dpdf;
struct dirent * epdf;

dpdf = opendir(tempFilename.c_str());
if (dpdf != NULL)
{
while ((epdf = readdir(dpdf)) != NULL)
{
std::string temp = tempFilename.c_str() + std::string("/") + std::string(epdf->d_name);
unlink(temp.c_str());
}
}

rmdir(tempFilename.c_str());
if (closedir(dpdf) != 0)
{
std::cerr << "ERROR: Could not delete directory " << tempFilename << "\n";
return 0;
}
#endif  

return 1;
}

inline
int endTestSuite()
{
delete[] StaticData::basePath();

for (unsigned i = 0; i < StaticData::tempFileNames().size(); ++i)
if (!_deleteTempFile(StaticData::tempFileNames()[i]))
++StaticData::errorCount();

std::cout << "**************************************" << std::endl;
std::cout << " Total Tests: " << StaticData::testCount() << std::endl;
std::cout << " Skipped:     " << StaticData::skippedCount() << std::endl;
std::cout << " Errors:      " << StaticData::errorCount() << std::endl;
std::cout << "**************************************" << std::endl;

if (StaticData::errorCount() != 0)
return 1;

return 0;
}

inline
void beginTest(const char * testName)
{
StaticData::currentTestName() = testName;
StaticData::thisTestOk() = true;
StaticData::thisTestSkipped() = false;
StaticData::testCount() += 1;
}

inline
void endTest()
{
if (StaticData::thisTestSkipped())
{
std::cout << StaticData::currentTestName() << " SKIPPED" << std::endl;
}
else if (StaticData::thisTestOk())
{
std::cout << StaticData::currentTestName() << " OK" << std::endl;
}
else
{
std::cerr << StaticData::currentTestName() << " FAILED" << std::endl;
}
}

inline
void skipCurrentTest()
{
StaticData::thisTestSkipped() = true;
StaticData::skippedCount() += 1;
}

inline void forceFail(const char * file, int line,
const char * comment, ...)
{
StaticData::errorCount() += 1;
std::cerr << file << ":" << line << " FAILED! ";
if (comment)
{
std::cerr << " (";
va_list args;
va_start(args, comment);
vfprintf(stderr, comment, args);
va_end(args);
std::cerr << ")";
}
std::cerr << std::endl;
}

inline void vforceFail(const char * file, int line,
const char * comment, va_list argp)
{
StaticData::errorCount() += 1;
std::cerr << file << ":" << line << " FAILED! ";
if (comment)
{
std::cerr << " (";
vfprintf(stderr, comment, argp);
std::cerr << ")";
}
std::cerr << std::endl;
}

inline void forceFail(const char * file, int line)
{
forceFail(file, line, 0);
}

template <typename T1, typename T2>
bool testEqual(char const * file, int line,
T1 const & value1, char const * expression1,
T2 const & value2, char const * expression2,
char const * comment, ...)
{
if (!(value1 == value2))
{
StaticData::thisTestOk() = false;
StaticData::errorCount() += 1;
std::cerr << file << ":" << line << " Assertion failed : "
<< expression1 << " == " << expression2 << " was: " << value1
<< " != " << value2;
if (comment)
{
std::cerr << " (";
va_list args;
va_start(args, comment);
vfprintf(stderr, comment, args);
va_end(args);
std::cerr << ")";
}
std::cerr << std::endl;
return false;
}
return true;
}

template <typename T1, typename T2>
bool vtestEqual(const char * file, int line,
const T1 & value1, const char * expression1,
const T2 & value2, const char * expression2,
const char * comment, va_list argp)
{
if (!(value1 == value2))
{
StaticData::thisTestOk() = false;
StaticData::errorCount() += 1;
std::cerr << file << ":" << line << " Assertion failed : "
<< expression1 << " == " << expression2 << " was: " << value1
<< " != " << value2;
if (comment)
{
std::cerr << " (";
vfprintf(stderr, comment, argp);
std::cerr << ")";
}
std::cerr << std::endl;
return false;
}
return true;
}

template <typename T1, typename T2>
bool testEqual(const char * file, int line,
const T1 & value1, const char * expression1,
const T2 & value2, const char * expression2)
{
return testEqual(file, line, value1, expression1, value2, expression2, 0);
}

template <typename T1, typename T2, typename T3>
bool testInDelta(const char * file, int line,
const T1 & value1, const char * expression1,
const T2 & value2, const char * expression2,
const T3 & value3, const char * expression3,
const char * comment, ...)
{
if (!(value1 >= value2 - value3 && value1 <= value2 + value3))
{
StaticData::thisTestOk() = false;
StaticData::errorCount() += 1;
std::cerr << file << ":" << line << " Assertion failed : "
<< expression1 << " in [" << expression2 << " - " << expression3
<< ", " << expression2 << " + " << expression3 << "] was: " << value1
<< " not in [" << value2 - value3 << ", " << value2 + value3 << "]";
if (comment)
{
std::cerr << " (";
va_list args;
va_start(args, comment);
vfprintf(stderr, comment, args);
va_end(args);
std::cerr << ")";
}
std::cerr << std::endl;
return false;
}
return true;
}

template <typename T1, typename T2, typename T3>
bool vtestInDelta(const char * file, int line,
const T1 & value1, const char * expression1,
const T2 & value2, const char * expression2,
const T3 & value3, const char * expression3,
const char * comment, va_list argp)
{
if (!(value1 >= value2 - value3 && value1 <= value2 + value3))
{
StaticData::thisTestOk() = false;
StaticData::errorCount() += 1;
std::cerr << file << ":" << line << " Assertion failed : "
<< expression1 << " in [" << expression2 << " - " << expression3
<< ", " << expression2 << " + " << expression3 << "] was: " << value1
<< " not in [" << value2 - value3 << ", " << value2 + value3 << "]";
if (comment)
{
std::cerr << " (";
vfprintf(stderr, comment, argp);
std::cerr << ")";
}
std::cerr << std::endl;
return false;
}
return true;
}

template <typename T1, typename T2, typename T3>
bool testInDelta(const char * file, int line,
const T1 & value1, const char * expression1,
const T2 & value2, const char * expression2,
const T3 & value3, const char * expression3)
{
return testInDelta(file, line, value1, expression1, value2, expression2, value3, expression3, 0);
}

template <typename T1, typename T2>
bool testNotEqual(const char * file, int line,
const T1 & value1, const char * expression1,
const T2 & value2, const char * expression2,
const char * comment, ...)
{
if (!(value1 != value2))
{
StaticData::thisTestOk() = false;
StaticData::errorCount() += 1;
std::cerr << file << ":" << line << " Assertion failed : "
<< expression1 << " != " << expression2 << " was: " << value1
<< " == " << value2;
if (comment)
{
std::cerr << " (";
va_list args;
va_start(args, comment);
vfprintf(stderr, comment, args);
va_end(args);
std::cerr << ")";
}
std::cerr << std::endl;
return false;
}
return true;
}

template <typename T1, typename T2>
bool vtestNotEqual(const char * file, int line,
const T1 & value1, const char * expression1,
const T2 & value2, const char * expression2,
const char * comment, va_list argp)
{
if (!(value1 != value2))
{
StaticData::thisTestOk() = false;
StaticData::errorCount() += 1;
std::cerr << file << ":" << line << " Assertion failed : "
<< expression1 << " != " << expression2 << " was: " << value1
<< " == " << value2;
if (comment)
{
std::cerr << " (";
vfprintf(stderr, comment, argp);
std::cerr << ")";
}
std::cerr << std::endl;
return false;
}
return true;
}

template <typename T1, typename T2>
bool testNotEqual(const char * file, int line,
const T1 & value1, const char * expression1,
const T2 & value2, const char * expression2)
{
return testNotEqual(file, line, value1, expression1, value2, expression2, 0);
}

template <typename T1, typename T2>
bool testGeq(const char * file, int line,
const T1 & value1, const char * expression1,
const T2 & value2, const char * expression2,
const char * comment, ...)
{
if (!(value1 >= value2))
{
StaticData::thisTestOk() = false;
StaticData::errorCount() += 1;
std::cerr << file << ":" << line << " Assertion failed : "
<< expression1 << " >= " << expression2 << " was: " << value1
<< " < " << value2;
if (comment)
{
std::cerr << " (";
va_list args;
va_start(args, comment);
vfprintf(stderr, comment, args);
va_end(args);
std::cerr << ")";
}
std::cerr << std::endl;
return false;
}
return true;
}

template <typename T1, typename T2>
bool vtestGeq(const char * file, int line,
const T1 & value1, const char * expression1,
const T2 & value2, const char * expression2,
const char * comment, va_list argp)
{
if (!(value1 >= value2))
{
StaticData::thisTestOk() = false;
StaticData::errorCount() += 1;
std::cerr << file << ":" << line << " Assertion failed : "
<< expression1 << " >= " << expression2 << " was: " << value1
<< " < " << value2;
if (comment)
{
std::cerr << " (";
vfprintf(stderr, comment, argp);
std::cerr << ")";
}
std::cerr << std::endl;
return false;
}
return true;
}

template <typename T1, typename T2>
bool testGeq(const char * file, int line,
const T1 & value1, const char * expression1,
const T2 & value2, const char * expression2)
{
return testGeq(file, line, value1, expression1, value2, expression2, 0);
}

template <typename T1, typename T2>
bool testGt(const char * file, int line,
const T1 & value1, const char * expression1,
const T2 & value2, const char * expression2,
const char * comment, ...)
{
if (!(value1 > value2))
{
StaticData::thisTestOk() = false;
StaticData::errorCount() += 1;
std::cerr << file << ":" << line << " Assertion failed : "
<< expression1 << " > " << expression2 << " was: " << value1
<< " <= " << value2;
if (comment)
{
std::cerr << " (";
va_list args;
va_start(args, comment);
vfprintf(stderr, comment, args);
va_end(args);
std::cerr << ")";
}
std::cerr << std::endl;
return false;
}
return true;
}

template <typename T1, typename T2>
bool vtestGt(const char * file, int line,
const T1 & value1, const char * expression1,
const T2 & value2, const char * expression2,
const char * comment, va_list argp)
{
if (!(value1 > value2))
{
StaticData::thisTestOk() = false;
StaticData::errorCount() += 1;
std::cerr << file << ":" << line << " Assertion failed : "
<< expression1 << " > " << expression2 << " was: " << value1
<< " <= " << value2;
if (comment)
{
std::cerr << " (";
vfprintf(stderr, comment, argp);
std::cerr << ")";
}
std::cerr << std::endl;
return false;
}
return true;
}

template <typename T1, typename T2>
bool testGt(const char * file, int line,
const T1 & value1, const char * expression1,
const T2 & value2, const char * expression2)
{
return testGt(file, line, value1, expression1, value2, expression2, 0);
}

template <typename T1, typename T2>
bool testLeq(const char * file, int line,
const T1 & value1, const char * expression1,
const T2 & value2, const char * expression2,
const char * comment, ...)
{
if (!(value1 <= value2))
{
StaticData::thisTestOk() = false;
StaticData::errorCount() += 1;
std::cerr << file << ":" << line << " Assertion failed : "
<< expression1 << " <= " << expression2 << " was: " << value1
<< " > " << value2;
if (comment)
{
std::cerr << " (";
va_list args;
va_start(args, comment);
vfprintf(stderr, comment, args);
va_end(args);
std::cerr << ")";
}
std::cerr << std::endl;
return false;
}
return true;
}

template <typename T1, typename T2>
bool vtestLeq(const char * file, int line,
const T1 & value1, const char * expression1,
const T2 & value2, const char * expression2,
const char * comment, va_list argp)
{
if (!(value1 <= value2))
{
StaticData::thisTestOk() = false;
StaticData::errorCount() += 1;
std::cerr << file << ":" << line << " Assertion failed : "
<< expression1 << " <= " << expression2 << " was: " << value1
<< " > " << value2;
if (comment)
{
std::cerr << " (";
vfprintf(stderr, comment, argp);
std::cerr << ")";
}
std::cerr << std::endl;
return false;
}
return true;
}

template <typename T1, typename T2>
bool testLeq(const char * file, int line,
const T1 & value1, const char * expression1,
const T2 & value2, const char * expression2)
{
return testLeq(file, line, value1, expression1, value2, expression2, 0);
}

template <typename T1, typename T2>
bool testLt(const char * file, int line,
const T1 & value1, const char * expression1,
const T2 & value2, const char * expression2,
const char * comment, ...)
{
if (!(value1 < value2))
{
StaticData::thisTestOk() = false;
StaticData::errorCount() += 1;
std::cerr << file << ":" << line << " Assertion failed : "
<< expression1 << " < " << expression2 << " was: " << value1
<< " >= " << value2;
if (comment)
{
std::cerr << " (";
va_list args;
va_start(args, comment);
vfprintf(stderr, comment, args);
va_end(args);
std::cerr << ")";
}
std::cerr << std::endl;
return false;
}
return true;
}

template <typename T1, typename T2>
bool vtestLt(const char * file, int line,
const T1 & value1, const char * expression1,
const T2 & value2, const char * expression2,
const char * comment, va_list argp)
{
if (!(value1 < value2))
{
StaticData::thisTestOk() = false;
StaticData::errorCount() += 1;
std::cerr << file << ":" << line << " Assertion failed : "
<< expression1 << " < " << expression2 << " was: " << value1
<< " >= " << value2;
if (comment)
{
std::cerr << " (";
vfprintf(stderr, comment, argp);
std::cerr << ")";
}
std::cerr << std::endl;
return false;
}
return true;
}

template <typename T1, typename T2>
bool testLt(const char * file, int line,
const T1 & value1, const char * expression1,
const T2 & value2, const char * expression2)
{
return testLt(file, line, value1, expression1, value2, expression2, 0);
}

template <typename T>
bool testTrue(const char * file, int line,
const T & value_, const char * expression_,
const char * comment, ...)
{
if (!(value_))
{
StaticData::thisTestOk() = false;
StaticData::errorCount() += 1;
std::cerr << file << ":" << line << " Assertion failed : "
<< expression_ << " should be true but was " << (value_);
if (comment)
{
std::cerr << " (";
va_list args;
va_start(args, comment);
vfprintf(stderr, comment, args);
va_end(args);
std::cerr << ")";
}
std::cerr << std::endl;
return false;
}
return true;
}

template <typename T>
bool vtestTrue(const char * file, int line,
const T & value_, const char * expression_,
const char * comment, va_list argp)
{
if (!(value_))
{
StaticData::thisTestOk() = false;
StaticData::errorCount() += 1;
std::cerr << file << ":" << line << " Assertion failed : "
<< expression_ << " should be true but was " << (value_);
if (comment)
{
std::cerr << " (";
vfprintf(stderr, comment, argp);
std::cerr << ")";
}
std::cerr << std::endl;
return false;
}
return true;
}

template <typename T>
bool testTrue(const char * file, int line,
const T & value_, const char * expression_)
{
return testTrue(file, line, value_, expression_, 0);
}

template <typename T>
bool testFalse(const char * file, int line,
const T & value_, const char * expression_,
const char * comment, ...)
{
if (value_)
{
StaticData::thisTestOk() = false;
StaticData::errorCount() += 1;
std::cerr << file << ":" << line << " Assertion failed : "
<< expression_ << " should be false but was " << (value_);
if (comment)
{
std::cerr << " (";
va_list args;
va_start(args, comment);
vfprintf(stderr, comment, args);
va_end(args);
std::cerr << ")";
}
std::cerr << std::endl;
return false;
}
return true;
}

template <typename T>
bool vtestFalse(const char * file, int line,
const T & value_, const char * expression_,
const char * comment, va_list argp)
{
if (value_)
{
StaticData::thisTestOk() = false;
StaticData::errorCount() += 1;
std::cerr << file << ":" << line << " Assertion failed : "
<< expression_ << " should be false but was " << (value_);
if (comment)
{
std::cerr << " (";
vfprintf(stderr, comment, argp);
std::cerr << ")";
}
std::cerr << std::endl;
return false;
}
return true;
}

template <typename T>
bool testFalse(const char * file, int line,
const T & value_, const char * expression_)
{
return testFalse(file, line, value_, expression_, 0);
}

#if SEQAN_ENABLE_TESTING
inline void fail()
{
StaticData::thisTestOk() = false;
printStackTrace(20);
throw AssertionFailedException();
}

#else
inline void fail()
{
printStackTrace(20);
abort();
}

#endif  

}  



#define SEQAN_DEFINE_TEST(test_name)                    \
template <bool speed_up_dummy_to_prevent_compilation_of_unused_tests_> \
void SEQAN_TEST_ ## test_name()





#if SEQAN_ENABLE_TESTING
#define SEQAN_BEGIN_TESTSUITE(suite_name)                       \
int main(int argc, char ** argv) {                           \
(void) argc;                                                \
::seqan::ClassTest::beginTestSuite(# suite_name, argv[0]);



#define SEQAN_END_TESTSUITE                     \
return ::seqan::ClassTest::endTestSuite();  \
}



#define SEQAN_CALL_TEST(test_name)                                      \
do {                                                                \
seqan::ClassTest::beginTest(# test_name);                       \
try {                                                           \
SEQAN_TEST_ ## test_name<true>();                           \
} catch (seqan::ClassTest::AssertionFailedException e) {        \
\
(void) e;          \
} catch (std::exception const & e) {                            \
std::cerr << "Unexpected exception of type "                \
<< toCString(seqan::Demangler<std::exception>(e)) \
<< "; message: " << e.what() << "\n";             \
seqan::ClassTest::StaticData::thisTestOk() = false;         \
seqan::ClassTest::StaticData::errorCount() += 1;            \
} catch (...) {                                                 \
std::cerr << "Unexpected exception of unknown type\n";      \
seqan::ClassTest::StaticData::thisTestOk() = false;         \
seqan::ClassTest::StaticData::errorCount() += 1;            \
}                                                               \
seqan::ClassTest::endTest();                                    \
} while (false)



#define SEQAN_SKIP_TEST                                       \
do {                                                      \
::seqan::ClassTest::skipCurrentTest();                \
throw ::seqan::ClassTest::AssertionFailedException(); \
} while (false)
#endif  


#if SEQAN_ENABLE_DEBUG



















#define SEQAN_ASSERT_FAIL(...)                                          \
do {                                                                \
::seqan::ClassTest::forceFail(__FILE__, __LINE__,               \
__VA_ARGS__);                     \
::seqan::ClassTest::fail();                                     \
} while (false)


#define SEQAN_ASSERT_EQ(_arg1, _arg2)                                   \
do {                                                                \
if (!::seqan::ClassTest::testEqual(__FILE__, __LINE__,          \
(_arg1), # _arg1,             \
(_arg2), # _arg2)) {          \
::seqan::ClassTest::fail();                                 \
}                                                               \
} while (false)


#define SEQAN_ASSERT_EQ_MSG(_arg1, _arg2, ...)                          \
do {                                                                \
if (!::seqan::ClassTest::testEqual(__FILE__, __LINE__,          \
(_arg1), # _arg1,             \
(_arg2), # _arg2,             \
__VA_ARGS__)) {              \
::seqan::ClassTest::fail();                                 \
}                                                               \
} while (false)


#define SEQAN_ASSERT_IN_DELTA(_arg1, _arg2, _arg3)                      \
do {                                                                \
if (!::seqan::ClassTest::testInDelta(__FILE__, __LINE__,        \
(_arg1), # _arg1,           \
(_arg2), # _arg2,           \
(_arg3), # _arg3)) {        \
::seqan::ClassTest::fail();                                 \
}                                                               \
} while (false)


#define SEQAN_ASSERT_IN_DELTA_MSG(_arg1, _arg2, _arg3, ...)             \
do {                                                                \
if (!::seqan::ClassTest::testInDelta(__FILE__, __LINE__,        \
(_arg1), # _arg1,           \
(_arg2), # _arg2,           \
(_arg3), # _arg3,           \
__VA_ARGS__)) {            \
::seqan::ClassTest::fail();                                 \
}                                                               \
} while (false)


#define SEQAN_ASSERT_NEQ(_arg1, _arg2)                                  \
do {                                                                \
if (!::seqan::ClassTest::testNotEqual(__FILE__, __LINE__,       \
(_arg1), # _arg1,          \
(_arg2), # _arg2)) {       \
::seqan::ClassTest::fail();                                 \
}                                                               \
} while (false)


#define SEQAN_ASSERT_NEQ_MSG(_arg1, _arg2, ...)                         \
do {                                                                \
if (!::seqan::ClassTest::testNotEqual(__FILE__, __LINE__,       \
(_arg1), # _arg1,          \
(_arg2), # _arg2,          \
__VA_ARGS__)) {           \
::seqan::ClassTest::fail();                                 \
}                                                               \
} while (false)


#define SEQAN_ASSERT_LEQ(_arg1, _arg2)                                  \
do {                                                                \
if (!::seqan::ClassTest::testLeq(__FILE__, __LINE__,            \
(_arg1), # _arg1,               \
(_arg2), # _arg2)) {            \
::seqan::ClassTest::fail();                                 \
}                                                               \
} while (false)


#define SEQAN_ASSERT_LEQ_MSG(_arg1, _arg2, ...)                         \
do {                                                                \
if (!::seqan::ClassTest::testLeq(__FILE__, __LINE__,            \
(_arg1), # _arg1,               \
(_arg2), # _arg2,               \
__VA_ARGS__)) {                \
::seqan::ClassTest::fail();                                 \
}                                                               \
} while (false)


#define SEQAN_ASSERT_LT(_arg1, _arg2)                                   \
do {                                                                \
if (!::seqan::ClassTest::testLt(__FILE__, __LINE__,             \
(_arg1), # _arg1,                \
(_arg2), # _arg2)) {             \
::seqan::ClassTest::fail();                                 \
}                                                               \
} while (false)


#define SEQAN_ASSERT_LT_MSG(_arg1, _arg2, ...)                          \
do {                                                                \
if (!::seqan::ClassTest::testLt(__FILE__, __LINE__,             \
(_arg1), # _arg1,                \
(_arg2), # _arg2,                \
__VA_ARGS__)) {                 \
::seqan::ClassTest::fail();                                 \
}                                                               \
} while (false)


#define SEQAN_ASSERT_GEQ(_arg1, _arg2)                                  \
do {                                                                \
if (!::seqan::ClassTest::testGeq(__FILE__, __LINE__,            \
(_arg1), # _arg1,               \
(_arg2), # _arg2)) {            \
::seqan::ClassTest::fail();                                 \
}                                                               \
} while (false)


#define SEQAN_ASSERT_GEQ_MSG(_arg1, _arg2, ...)                         \
do {                                                                \
if (!::seqan::ClassTest::testGeq(__FILE__, __LINE__,            \
(_arg1), # _arg1,               \
(_arg2), # _arg2,               \
__VA_ARGS__)) {                \
::seqan::ClassTest::fail();                                 \
}                                                               \
} while (false)


#define SEQAN_ASSERT_GT(_arg1, _arg2)                                   \
do {                                                                \
if (!::seqan::ClassTest::testGt(__FILE__, __LINE__,             \
(_arg1), # _arg1,                \
(_arg2), # _arg2)) {             \
::seqan::ClassTest::fail();                                 \
}                                                               \
} while (false)


#define SEQAN_ASSERT_GT_MSG(_arg1, _arg2, ...)                          \
do {                                                                \
if (!::seqan::ClassTest::testGt(__FILE__, __LINE__,             \
(_arg1), # _arg1,                \
(_arg2), # _arg2,                \
__VA_ARGS__)) {                 \
::seqan::ClassTest::fail();                                 \
}                                                               \
} while (false)


#define SEQAN_ASSERT(_arg1)                                        \
do {                                                                \
if (!::seqan::ClassTest::testTrue(__FILE__, __LINE__,           \
(_arg1), # _arg1)) {           \
::seqan::ClassTest::fail();                                 \
}                                                               \
} while (false)


#define SEQAN_ASSERT_MSG(_arg1, ...)                               \
do {                                                                \
if (!::seqan::ClassTest::testTrue(__FILE__, __LINE__,           \
(_arg1), # _arg1,              \
__VA_ARGS__)) {             \
::seqan::ClassTest::fail();                                 \
}                                                               \
} while (false)


#define SEQAN_ASSERT_NOT(_arg1)                                       \
do {                                                              \
if (!::seqan::ClassTest::testFalse(__FILE__, __LINE__,        \
(_arg1), # _arg1)) {        \
::seqan::ClassTest::fail();                               \
}                                                             \
} while (false)


#define SEQAN_ASSERT_NOT_MSG(_arg1, ...)                              \
do {                                                              \
if (!::seqan::ClassTest::testFalse(__FILE__, __LINE__,        \
(_arg1), # _arg1,           \
__VA_ARGS__)) {          \
::seqan::ClassTest::fail();                               \
}                                                             \
} while (false)

#else

#define SEQAN_ASSERT_EQ(_arg1, _arg2) do {} while (false)
#define SEQAN_ASSERT_EQ_MSG(_arg1, _arg2, ...) do {} while (false)
#define SEQAN_ASSERT_NEQ(_arg1, _arg2) do {} while (false)
#define SEQAN_ASSERT_NEQ_MSG(_arg1, _arg2, ...) do {} while (false)
#define SEQAN_ASSERT_LEQ(_arg1, _arg2) do {} while (false)
#define SEQAN_ASSERT_LEQ_MSG(_arg1, _arg2, ...) do {} while (false)
#define SEQAN_ASSERT_LT(_arg1, _arg2) do {} while (false)
#define SEQAN_ASSERT_LT_MSG(_arg1, _arg2, ...) do {} while (false)
#define SEQAN_ASSERT_GEQ(_arg1, _arg2) do {} while (false)
#define SEQAN_ASSERT_GEQ_MSG(_arg1, _arg2, ...) do {} while (false)
#define SEQAN_ASSERT_GT(_arg1, _arg2) do {} while (false)
#define SEQAN_ASSERT_GT_MSG(_arg1, _arg2, ...) do {} while (false)
#define SEQAN_ASSERT(_arg1) do {} while (false)
#define SEQAN_ASSERT_MSG(_arg1, ...) do {} while (false)
#define SEQAN_ASSERT_NOT(_arg1) do {} while (false)
#define SEQAN_ASSERT_NOT_MSG(_arg1, ...) do {} while (false)
#define SEQAN_ASSERT_FAIL(...) do {} while (false)

#endif  

#define SEQAN_PROGRAM_PATH                      \
::seqan::ClassTest::StaticData::basePath()



#define SEQAN_PATH_TO_ROOT()                      \
::seqan::ClassTest::StaticData::pathToRoot()




#define SEQAN_TEMP_FILENAME() (::seqan::ClassTest::tempFileName())


#if !SEQAN_ENABLE_TESTING

#define SEQAN_BEGIN_TESTSUITE(suite_name)                               \
int main(int argc, char ** argv) {                                   \
(void) argc;                                                        \
(void) argv;                                                        \
fprintf(stderr, "Warning: SEQAN_ENABLE_TESTING is wrong and you used the macro SEQAN_BEGIN_TESTSUITE!\n");
#define SEQAN_END_TESTSUITE \
return 0;                                   \
}
#define SEQAN_CALL_TEST(test_name) do { SEQAN_TEST_ ## test_name(); } while (false)
#define SEQAN_SKIP_TEST do {} while (false)

#endif  





inline std::string getAbsolutePath(const char * path)
{
return std::string(::seqan::ClassTest::StaticData::pathToRoot()) + "/" + path;
}

}  

#endif  

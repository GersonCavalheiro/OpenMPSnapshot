

#pragma once

#include <ghc/filesystem.hpp>
#include <types.hpp>
#include <core.hpp>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
struct IUnknown;
#include <Dbghelp.h>
#include <Windows.h>
#include <shellapi.h>
#include <timeapi.h>
#define SET_TICKER_RESOLUTION(ms) timeBeginPeriod(ms)
#define LIBRARY_OPEN(path) LoadLibrary(path)
#define LIBRARY_GET_ADDR GetProcAddress
#define LIBRARY_FREE FreeLibrary
static LARGE_INTEGER yo;
#define BUILD_WINDOWS
#else
#include <dlfcn.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#define SET_TICKER_RESOLUTION(ms)
#define LIBRARY_OPEN(path) dlopen(path, RTLD_LAZY | RTLD_LOCAL)
#define LIBRARY_GET_ADDR dlsym
#define LIBRARY_FREE dlclose
static timeval initialTime;
#endif

#ifdef BUILD_WINDOWS
#define LIBRARY_EXT ".dll"
#define EXECUTABLE_EXT ".exe"
#elif __APPLE__
#define LIBRARY_EXT ".dylib"
#define EXECUTABLE_EXT
#else
#define LIBRARY_EXT ".so"
#define EXECUTABLE_EXT
#endif

namespace utils
{
static bool initialized = false;

#ifdef BUILD_WINDOWS
std::string GetLastErrorAsString()
{
DWORD errorMessageID = ::GetLastError();
if (errorMessageID == 0)
{
return std::string(); 
}

LPSTR messageBuffer = nullptr;

size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
NULL, errorMessageID, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);

std::string message(messageBuffer, size);

LocalFree(messageBuffer);

return message;
}
#else
std::string GetLastErrorAsString()
{
const char* lastError = dlerror();

if (lastError == nullptr)
{
return std::string();
}

return std::string(lastError);
}
#endif

void RunProcess(ILogger& logger, StringView exe, StringView args, bool runInBackground = false)
{
auto exePath = ghc::filesystem::path(exe.data());
if (!exePath.has_extension())
{
exePath.replace_extension(EXECUTABLE_EXT);
}

if (!ghc::filesystem::exists(exePath))
{
logger.logLn(LogLevel::Warning, "Bot executable not found: %.*s", PRINT_VIEW(exePath.string()));
}

#ifdef BUILD_WINDOWS
ShellExecute(nullptr, "open", exePath.string().c_str(), args.data(), nullptr, FALSE);
#else
system(("./" + exePath.string() + " " + args.data() + (runInBackground ? " &" : "")).c_str());
#endif
}

unsigned GetTickCount()
{
if (initialized == false)
{
#ifdef BUILD_WINDOWS
QueryPerformanceFrequency(&yo);
#else
gettimeofday(&initialTime, 0);
#endif
initialized = true;
}

#ifdef BUILD_WINDOWS
LARGE_INTEGER PerfVal;

QueryPerformanceCounter(&PerfVal);

return (unsigned)(PerfVal.QuadPart * 1000 / yo.QuadPart);
#else
struct timeval tp;
gettimeofday(&tp, 0);
return (tp.tv_sec - initialTime.tv_sec) * 1000 + (tp.tv_usec - initialTime.tv_usec) / 1000;
#endif
}

ghc::filesystem::path GetExecutablePath()
{
#ifdef BUILD_WINDOWS
char path[4096] = { 0 };
if (GetModuleFileNameA(nullptr, path, sizeof(path)))
{
return ghc::filesystem::canonical(path);
}
else
{
return ghc::filesystem::path();
}
#else
return ghc::filesystem::canonical("/proc/self/exe");
#endif
}

}

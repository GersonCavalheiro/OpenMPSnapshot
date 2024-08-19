




#include <catch2/internal/catch_fatal_condition_handler.hpp>

#include <catch2/internal/catch_context.hpp>
#include <catch2/internal/catch_enforce.hpp>
#include <catch2/interfaces/catch_interfaces_capture.hpp>
#include <catch2/internal/catch_windows_h_proxy.hpp>
#include <catch2/internal/catch_stdstreams.hpp>

#include <algorithm>

#if !defined( CATCH_CONFIG_WINDOWS_SEH ) && !defined( CATCH_CONFIG_POSIX_SIGNALS )

namespace Catch {

void FatalConditionHandler::engage_platform() {}
void FatalConditionHandler::disengage_platform() noexcept {}
FatalConditionHandler::FatalConditionHandler() = default;
FatalConditionHandler::~FatalConditionHandler() = default;

} 

#endif 

#if defined( CATCH_CONFIG_WINDOWS_SEH ) && defined( CATCH_CONFIG_POSIX_SIGNALS )
#error "Inconsistent configuration: Windows' SEH handling and POSIX signals cannot be enabled at the same time"
#endif 

#if defined( CATCH_CONFIG_WINDOWS_SEH ) || defined( CATCH_CONFIG_POSIX_SIGNALS )

namespace {
void reportFatal( char const * const message ) {
Catch::getCurrentContext().getResultCapture()->handleFatalErrorCondition( message );
}

constexpr std::size_t minStackSizeForErrors = 32 * 1024;
} 

#endif 

#if defined( CATCH_CONFIG_WINDOWS_SEH )

namespace Catch {

struct SignalDefs { DWORD id; const char* name; };

static SignalDefs signalDefs[] = {
{ EXCEPTION_ILLEGAL_INSTRUCTION,  "SIGILL - Illegal instruction signal" },
{ EXCEPTION_STACK_OVERFLOW, "SIGSEGV - Stack overflow" },
{ EXCEPTION_ACCESS_VIOLATION, "SIGSEGV - Segmentation violation signal" },
{ EXCEPTION_INT_DIVIDE_BY_ZERO, "Divide by zero error" },
};

static LONG CALLBACK topLevelExceptionFilter(PEXCEPTION_POINTERS ExceptionInfo) {
for (auto const& def : signalDefs) {
if (ExceptionInfo->ExceptionRecord->ExceptionCode == def.id) {
reportFatal(def.name);
}
}
return EXCEPTION_CONTINUE_SEARCH;
}

static LPTOP_LEVEL_EXCEPTION_FILTER previousTopLevelExceptionFilter = nullptr;


FatalConditionHandler::FatalConditionHandler() {
ULONG guaranteeSize = static_cast<ULONG>(minStackSizeForErrors);
if (!SetThreadStackGuarantee(&guaranteeSize)) {
Catch::cerr()
<< "Failed to reserve piece of stack."
<< " Stack overflows will not be reported successfully.";
}
}

FatalConditionHandler::~FatalConditionHandler() = default;


void FatalConditionHandler::engage_platform() {
previousTopLevelExceptionFilter = SetUnhandledExceptionFilter(topLevelExceptionFilter);
}

void FatalConditionHandler::disengage_platform() noexcept {
if (SetUnhandledExceptionFilter(previousTopLevelExceptionFilter) != topLevelExceptionFilter) {
Catch::cerr()
<< "Unexpected SEH unhandled exception filter on disengage."
<< " The filter was restored, but might be rolled back unexpectedly.";
}
previousTopLevelExceptionFilter = nullptr;
}

} 

#endif 

#if defined( CATCH_CONFIG_POSIX_SIGNALS )

#include <signal.h>

namespace Catch {

struct SignalDefs {
int id;
const char* name;
};

static SignalDefs signalDefs[] = {
{ SIGINT,  "SIGINT - Terminal interrupt signal" },
{ SIGILL,  "SIGILL - Illegal instruction signal" },
{ SIGFPE,  "SIGFPE - Floating point error signal" },
{ SIGSEGV, "SIGSEGV - Segmentation violation signal" },
{ SIGTERM, "SIGTERM - Termination request signal" },
{ SIGABRT, "SIGABRT - Abort (abnormal termination) signal" }
};

#if defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

static char* altStackMem = nullptr;
static std::size_t altStackSize = 0;
static stack_t oldSigStack{};
static struct sigaction oldSigActions[sizeof(signalDefs) / sizeof(SignalDefs)]{};

static void restorePreviousSignalHandlers() noexcept {
for (std::size_t i = 0; i < sizeof(signalDefs) / sizeof(SignalDefs); ++i) {
sigaction(signalDefs[i].id, &oldSigActions[i], nullptr);
}
sigaltstack(&oldSigStack, nullptr);
}

static void handleSignal( int sig ) {
char const * name = "<unknown signal>";
for (auto const& def : signalDefs) {
if (sig == def.id) {
name = def.name;
break;
}
}
restorePreviousSignalHandlers();
reportFatal( name );
raise( sig );
}

FatalConditionHandler::FatalConditionHandler() {
assert(!altStackMem && "Cannot initialize POSIX signal handler when one already exists");
if (altStackSize == 0) {
altStackSize = std::max(static_cast<size_t>(SIGSTKSZ), minStackSizeForErrors);
}
altStackMem = new char[altStackSize]();
}

FatalConditionHandler::~FatalConditionHandler() {
delete[] altStackMem;
altStackMem = nullptr;
}

void FatalConditionHandler::engage_platform() {
stack_t sigStack;
sigStack.ss_sp = altStackMem;
sigStack.ss_size = altStackSize;
sigStack.ss_flags = 0;
sigaltstack(&sigStack, &oldSigStack);
struct sigaction sa = { };

sa.sa_handler = handleSignal;
sa.sa_flags = SA_ONSTACK;
for (std::size_t i = 0; i < sizeof(signalDefs)/sizeof(SignalDefs); ++i) {
sigaction(signalDefs[i].id, &sa, &oldSigActions[i]);
}
}

#if defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif


void FatalConditionHandler::disengage_platform() noexcept {
restorePreviousSignalHandlers();
}

} 

#endif 

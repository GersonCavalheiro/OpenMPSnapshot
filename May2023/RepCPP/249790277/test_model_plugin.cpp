

#define HARNESS_DEFAULT_MIN_THREADS 4
#define HARNESS_DEFAULT_MAX_THREADS 4

#include "tbb/tbb_config.h"

#if !__TBB_TODO || __TBB_WIN8UI_SUPPORT
#define HARNESS_NO_PARSE_COMMAND_LINE 1
#include "harness.h"
int TestMain() {
return Harness::Skipped;
}
#else 

#if __TBB_DEFINE_MIC

#ifndef _USRDLL
#define HARNESS_NO_PARSE_COMMAND_LINE 1
#include "harness.h"
int TestMain() {
return Harness::Skipped;
}
#endif

#else 

#if _WIN32 || _WIN64
#include "tbb/machine/windows_api.h"
#else
#include <dlfcn.h>
#endif

#include <stdlib.h>
#include <stdio.h>

#if !TBB_USE_EXCEPTIONS && _MSC_VER
#pragma warning (push)
#pragma warning (disable: 4530)
#endif

#include <stdexcept>

#if !TBB_USE_EXCEPTIONS && _MSC_VER
#pragma warning (pop)
#endif

#if TBB_USE_EXCEPTIONS
#include "harness_report.h"
#endif

#ifdef _USRDLL
#include "tbb/task_scheduler_init.h"

class CModel {
public:
CModel(void) {};
static tbb::task_scheduler_init tbb_init;

void init_and_terminate( int );
};

tbb::task_scheduler_init CModel::tbb_init(1);


void CModel::init_and_terminate( int maxthread ) {
for( int i=0; i<200; ++i ) {
switch( i&3 ) {
default: {
tbb::task_scheduler_init init( rand() % maxthread + 1 );
break;
}
case 0: {
tbb::task_scheduler_init init;
break;
}
case 1: {
tbb::task_scheduler_init init( tbb::task_scheduler_init::automatic );
break;
}
case 2: {
tbb::task_scheduler_init init( tbb::task_scheduler_init::deferred );
init.initialize( rand() % maxthread + 1 );
init.terminate();
break;
}
}
}
}

extern "C"
#if _WIN32 || _WIN64
__declspec(dllexport)
#endif
void plugin_call(int maxthread)
{
srand(2);
__TBB_TRY {
CModel model;
model.init_and_terminate(maxthread);
} __TBB_CATCH( std::runtime_error& error ) {
#if TBB_USE_EXCEPTIONS
REPORT("ERROR: %s\n", error.what());
#endif 
}
}

#else 

#include "harness.h"
#include "harness_dynamic_libs.h"
#include "harness_tls.h"

extern "C" void plugin_call(int);

void report_error_in(const char* function_name)
{
#if _WIN32 || _WIN64
char* message;
int code = GetLastError();

FormatMessage(
FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
NULL, code,MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
(char*)&message, 0, NULL );
#else
char* message = (char*)dlerror();
int code = 0;
#endif
REPORT( "%s failed with error %d: %s\n", function_name, code, message);

#if _WIN32 || _WIN64
LocalFree(message);
#endif
}

typedef void (*PLUGIN_CALL)(int);

#if __linux__
#define RML_LIBRARY_NAME(base) TEST_LIBRARY_NAME(base) ".1"
#else
#define RML_LIBRARY_NAME(base) TEST_LIBRARY_NAME(base)
#endif

int TestMain () {
#if !RML_USE_WCRM
PLUGIN_CALL my_plugin_call;

LimitTLSKeysTo limitTLS(10);

Harness::LIBRARY_HANDLE hLib;
#if _WIN32 || _WIN64
hLib = LoadLibrary("irml.dll");
if ( !hLib )
hLib = LoadLibrary("irml_debug.dll");
if ( !hLib )
return Harness::Skipped; 
FreeLibrary(hLib);
#else 
#if __TBB_ARENA_PER_MASTER
hLib = dlopen(RML_LIBRARY_NAME("libirml"), RTLD_LAZY);
if ( !hLib )
hLib = dlopen(RML_LIBRARY_NAME("libirml_debug"), RTLD_LAZY);
if ( !hLib )
return Harness::Skipped;
dlclose(hLib);
#endif 
#endif 
for( int i=1; i<100; ++i ) {  
REMARK("Iteration %d, loading plugin library...\n", i);
hLib = Harness::OpenLibrary(TEST_LIBRARY_NAME("test_model_plugin_dll"));
if ( !hLib ) {
#if !__TBB_NO_IMPLICIT_LINKAGE
#if _WIN32 || _WIN64
report_error_in("LoadLibrary");
#else
report_error_in("dlopen");
#endif
return -1;
#else
return Harness::Skipped;
#endif
}
my_plugin_call = (PLUGIN_CALL)Harness::GetAddress(hLib, "plugin_call");
if (my_plugin_call==NULL) {
#if _WIN32 || _WIN64
report_error_in("GetProcAddress");
#else
report_error_in("dlsym");
#endif
return -1;
}
REMARK("Calling plugin method...\n");
my_plugin_call(MaxThread);

REMARK("Unloading plugin library...\n");
Harness::CloseLibrary(hLib);
} 

return Harness::Done;
#else
return Harness::Skipped;
#endif 
}

#endif
#endif

#endif 



#include "tbb/tbb_stddef.h"

#if __TBB_WIN8UI_SUPPORT
#define HARNESS_NO_PARSE_COMMAND_LINE 1
#include "harness.h"
int TestMain() {
return Harness::Skipped;
}
#else

#include <stdio.h>
#include <stdlib.h>

#if !TBB_USE_EXCEPTIONS && _MSC_VER
#pragma warning (push)
#pragma warning (disable: 4530)
#endif

#include <vector>
#include <string>
#include <utility>

#if !TBB_USE_EXCEPTIONS && _MSC_VER
#pragma warning (pop)
#endif

#include "tbb/task_scheduler_init.h"

#define HARNESS_CUSTOM_MAIN 1
#define HARNESS_NO_PARSE_COMMAND_LINE 1
#define HARNESS_NO_MAIN_ARGS 0
#include "harness.h"

#if defined (_WIN32) || defined (_WIN64)
#define TEST_SYSTEM_COMMAND "test_tbb_version.exe @"
#elif __APPLE__
#define TEST_SYSTEM_COMMAND "DYLD_LIBRARY_PATH=. ./test_tbb_version.exe @"
#else
#define TEST_SYSTEM_COMMAND "./test_tbb_version.exe @"
#endif

enum string_required {
required,
optional,
optional_multiple
};

typedef std::pair <std::string, string_required> string_pair;

void initialize_strings_vector(std::vector <string_pair>* vector);

const char stderr_stream[] = "version_test.err";
const char stdout_stream[] = "version_test.out";

HARNESS_EXPORT
int main(int argc, char *argv[] ) {
const size_t psBuffer_len = 2048;
char psBuffer[psBuffer_len];

#if TBB_INTERFACE_VERSION>=3014
if ( tbb::TBB_runtime_interface_version()!=TBB_INTERFACE_VERSION ){
snprintf( psBuffer, psBuffer_len,
"%s %s %d %s %d.",
"Running with the library of different version than the test was compiled against.",
"Expected",
TBB_INTERFACE_VERSION,
"- got",
tbb::TBB_runtime_interface_version()
);
ASSERT( tbb::TBB_runtime_interface_version()==TBB_INTERFACE_VERSION, psBuffer );
}
#endif
#if __TBB_MIC_OFFLOAD
(argc, argv);
REPORT("skip\n");
#elif __TBB_MPI_INTEROP || __bg__
(void) argc; 
(void) argv; 
REPORT("skip\n");
#else
__TBB_TRY {
FILE *stream_out;
FILE *stream_err;

if(argc>1 && argv[1][0] == '@' ) {
stream_err = freopen( stderr_stream, "w", stderr );
if( stream_err == NULL ){
REPORT( "Internal test error (freopen)\n" );
exit( 1 );
}
stream_out = freopen( stdout_stream, "w", stdout );
if( stream_out == NULL ){
REPORT( "Internal test error (freopen)\n" );
exit( 1 );
}
{
tbb::task_scheduler_init init(1);
}
fclose( stream_out );
fclose( stream_err );
exit(0);
}
if ( getenv("TBB_VERSION") ){
REPORT( "TBB_VERSION defined, skipping step 1 (empty output check)\n" );
}else{
if( ( system(TEST_SYSTEM_COMMAND) ) != 0 ){
REPORT( "Error (step 1): Internal test error\n" );
exit( 1 );
}
stream_err = fopen( stderr_stream, "r" );
if( stream_err == NULL ){
REPORT( "Error (step 1):Internal test error (stderr open)\n" );
exit( 1 );
}
while( !feof( stream_err ) ) {
if( fgets( psBuffer, psBuffer_len, stream_err ) != NULL ){
REPORT( "Error (step 1): stderr should be empty\n" );
exit( 1 );
}
}
fclose( stream_err );
stream_out = fopen( stdout_stream, "r" );
if( stream_out == NULL ){
REPORT( "Error (step 1):Internal test error (stdout open)\n" );
exit( 1 );
}
while( !feof( stream_out ) ) {
if( fgets( psBuffer, psBuffer_len, stream_out ) != NULL ){
REPORT( "Error (step 1): stdout should be empty\n" );
exit( 1 );
}
}
fclose( stream_out );
}

if ( !getenv("TBB_VERSION") ){
Harness::SetEnv("TBB_VERSION","1");
}

if( ( system(TEST_SYSTEM_COMMAND) ) != 0 ){
REPORT( "Error (step 2):Internal test error\n" );
exit( 1 );
}
std::vector <string_pair> strings_vector;
std::vector <string_pair>::iterator strings_iterator;

initialize_strings_vector( &strings_vector );
strings_iterator = strings_vector.begin();

stream_out = fopen( stdout_stream, "r" );
if( stream_out == NULL ){
REPORT( "Error (step 2):Internal test error (stdout open)\n" );
exit( 1 );
}
while( !feof( stream_out ) ) {
if( fgets( psBuffer, psBuffer_len, stream_out ) != NULL ){
REPORT( "Error (step 2): stdout should be empty\n" );
exit( 1 );
}
}
fclose( stream_out );

stream_err = fopen( stderr_stream, "r" );
if( stream_err == NULL ){
REPORT( "Error (step 1):Internal test error (stderr open)\n" );
exit( 1 );
}

while( !feof( stream_err ) ) {
if( fgets( psBuffer, psBuffer_len, stream_err ) != NULL ){
if (strstr( psBuffer, "TBBmalloc: " )) {
continue;
}
bool match_found = false;
do{
if ( strings_iterator == strings_vector.end() ){
REPORT( "Error: version string dictionary ended prematurely.\n" );
REPORT( "No match for: \t%s", psBuffer );
exit( 1 );
}
if ( strstr( psBuffer, strings_iterator->first.c_str() ) == NULL ){ 
if( strings_iterator->second == required ){
REPORT( "Error: version strings do not match.\n" );
REPORT( "Expected \"%s\" not found in:\n\t%s", strings_iterator->first.c_str(), psBuffer );
exit( 1 );
}
++strings_iterator;
}else{
match_found = true;
if( strings_iterator->second != optional_multiple )
++strings_iterator;
}
}while( !match_found );
}
}
fclose( stream_err );
} __TBB_CATCH(...) {
ASSERT( 0,"unexpected exception" );
}
REPORT("done\n");
#endif 
return 0;
}


void initialize_strings_vector(std::vector <string_pair>* vector)
{
vector->push_back(string_pair("TBB: VERSION\t\t2017.0", required));       
vector->push_back(string_pair("TBB: INTERFACE VERSION\t9107", required)); 
vector->push_back(string_pair("TBB: BUILD_DATE", required));
vector->push_back(string_pair("TBB: BUILD_HOST", required));
vector->push_back(string_pair("TBB: BUILD_OS", required));
#if _WIN32||_WIN64
#if !__MINGW32__
vector->push_back(string_pair("TBB: BUILD_CL", required));
vector->push_back(string_pair("TBB: BUILD_COMPILER", required));
#else
vector->push_back(string_pair("TBB: BUILD_GCC", required));
#endif
#elif __APPLE__
vector->push_back(string_pair("TBB: BUILD_KERNEL", required));
vector->push_back(string_pair("TBB: BUILD_CLANG", required));
vector->push_back(string_pair("TBB: BUILD_XCODE", optional));
vector->push_back(string_pair("TBB: BUILD_COMPILER", optional)); 
#elif __sun
vector->push_back(string_pair("TBB: BUILD_KERNEL", required));
vector->push_back(string_pair("TBB: BUILD_SUNCC", required));
vector->push_back(string_pair("TBB: BUILD_COMPILER", optional)); 
#else 
#if !__ANDROID__
vector->push_back(string_pair("TBB: BUILD_KERNEL", required));
#endif
vector->push_back(string_pair("TBB: BUILD_GCC", optional));
vector->push_back(string_pair("TBB: BUILD_CLANG", optional));
vector->push_back(string_pair("TBB: BUILD_TARGET_CXX", optional));
vector->push_back(string_pair("TBB: BUILD_COMPILER", optional)); 
#if __ANDROID__
vector->push_back(string_pair("TBB: BUILD_NDK", optional));
vector->push_back(string_pair("TBB: BUILD_LD", optional));
#else
vector->push_back(string_pair("TBB: BUILD_LIBC", required));
vector->push_back(string_pair("TBB: BUILD_LD", required));
#endif 
#endif 
vector->push_back(string_pair("TBB: BUILD_TARGET", required));
vector->push_back(string_pair("TBB: BUILD_COMMAND", required));
vector->push_back(string_pair("TBB: TBB_USE_DEBUG", required));
vector->push_back(string_pair("TBB: TBB_USE_ASSERT", required));
#if __TBB_CPF_BUILD
vector->push_back(string_pair("TBB: TBB_PREVIEW_BINARY", required));
#endif
vector->push_back(string_pair("TBB: DO_ITT_NOTIFY", required));
vector->push_back(string_pair("TBB: ITT", optional)); 
vector->push_back(string_pair("TBB: ALLOCATOR", required));
#if _WIN32||_WIN64
vector->push_back(string_pair("TBB: Processor groups", required));
vector->push_back(string_pair("TBB: ----- Group", optional_multiple));
#endif
vector->push_back(string_pair("TBB: RML", optional));
vector->push_back(string_pair("TBB: Intel(R) RML library built:", optional));
vector->push_back(string_pair("TBB: Intel(R) RML library version:", optional));
vector->push_back(string_pair("TBB: Tools support", required));
return;
}
#endif 

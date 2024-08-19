


#include <stdlib.h>
#include "../../../include/tbb/tbb_stddef.h" 
#include "harness_defs.h"
#define RML_PURE_VIRTUAL_HANDLER abort

#if _MSC_VER==1500 && !defined(__INTEL_COMPILER)
#pragma warning( push )
#pragma warning( disable: 4100 ) 
#elif __TBB_MSVC_UNREACHABLE_CODE_IGNORED
#pragma warning( push )
#pragma warning( disable: 4702 )
#endif
#include "rml_omp.h"
#if ( _MSC_VER==1500 && !defined(__INTEL_COMPILER)) || __TBB_MSVC_UNREACHABLE_CODE_IGNORED
#pragma warning( pop )
#endif

rml::versioned_object::version_type Version;

class MyClient: public __kmp::rml::omp_client {
public:
rml::versioned_object::version_type version() const __TBB_override {return 0;}
size_type max_job_count() const __TBB_override {return 1024;}
size_t min_stack_size() const __TBB_override {return 1<<20;}
rml::job* create_one_job() __TBB_override {return NULL;}
void acknowledge_close_connection() __TBB_override {}
void cleanup(job&) __TBB_override {}
policy_type policy() const __TBB_override {return throughput;}
void process( job&, void*, __kmp::rml::omp_client::size_type ) __TBB_override {}

};

__kmp::rml::omp_server* MyServerPtr;

#define HARNESS_NO_PARSE_COMMAND_LINE 1
#define HARNESS_CUSTOM_MAIN 1
#include "harness.h"

extern "C" void Cplusplus() {
MyClient client;
Version = client.version();
REPORT("done\n");
}

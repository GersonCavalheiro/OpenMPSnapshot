

#include "internal/_deprecated_header_message_guard.h"

#if !defined(__TBB_show_deprecation_message_runtime_loader_H) && defined(__TBB_show_deprecated_header_message)
#define  __TBB_show_deprecation_message_runtime_loader_H
#pragma message("TBB Warning: tbb/runtime_loader.h is deprecated. For details, please see Deprecated Features appendix in the TBB reference manual.")
#endif

#if defined(__TBB_show_deprecated_header_message)
#undef __TBB_show_deprecated_header_message
#endif

#ifndef __TBB_runtime_loader_H
#define __TBB_runtime_loader_H

#define __TBB_runtime_loader_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#if ! TBB_PREVIEW_RUNTIME_LOADER
#error Set TBB_PREVIEW_RUNTIME_LOADER to include runtime_loader.h
#endif

#include "tbb_stddef.h"
#include <climits>

#if _MSC_VER
#if ! __TBB_NO_IMPLICIT_LINKAGE
#ifdef _DEBUG
#pragma comment( linker, "/nodefaultlib:tbb_debug.lib" )
#pragma comment( linker, "/defaultlib:tbbproxy_debug.lib" )
#else
#pragma comment( linker, "/nodefaultlib:tbb.lib" )
#pragma comment( linker, "/defaultlib:tbbproxy.lib" )
#endif
#endif
#endif

namespace tbb {

namespace interface6 {



class __TBB_DEPRECATED_VERBOSE runtime_loader : tbb::internal::no_copy {

public:

enum error_mode {
em_status,     
em_throw,      
em_abort       
}; 

enum error_code {
ec_ok,         
ec_bad_call,   
ec_bad_arg,    
ec_bad_lib,    
ec_bad_ver,    
ec_no_lib      
}; 

runtime_loader( error_mode mode = em_abort );


runtime_loader(
char const * path[],                           
int          min_ver = TBB_INTERFACE_VERSION,  
int          max_ver = INT_MAX,                
error_mode   mode    = em_abort                
);

~runtime_loader();


error_code
load(
char const * path[],                           
int          min_ver = TBB_INTERFACE_VERSION,  
int          max_ver = INT_MAX                 

);



error_code status();

private:

error_mode const my_mode;
error_code       my_status;
bool             my_loaded;

}; 

} 

using interface6::runtime_loader;

} 

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_runtime_loader_H_include_area

#endif 


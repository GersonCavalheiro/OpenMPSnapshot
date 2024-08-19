

#include "rml_omp.h"
#include "omp_dynamic_link.h"
#include <assert.h>

namespace __kmp {
namespace rml {

#define MAKE_SERVER(x) DLD(__KMP_make_rml_server,x)
#define GET_INFO(x) DLD(__KMP_call_with_my_server_info,x)
#define SERVER omp_server 
#define CLIENT omp_client
#define FACTORY omp_factory

#if __TBB_WEAK_SYMBOLS_PRESENT
#pragma weak __KMP_make_rml_server
#pragma weak __KMP_call_with_my_server_info
extern "C" {
omp_factory::status_type __KMP_make_rml_server( omp_factory& f, omp_server*& server, omp_client& client );
void __KMP_call_with_my_server_info( ::rml::server_info_callback_t cb, void* arg );
}
#endif 

#include "rml_factory.h"

} 
} 

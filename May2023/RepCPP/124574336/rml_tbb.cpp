

#include "../include/rml_tbb.h"
#include "tbb/dynamic_link.h"
#include <assert.h>

namespace tbb {
namespace internal {
namespace rml {

#define MAKE_SERVER(x) DLD(__TBB_make_rml_server,x)
#define GET_INFO(x) DLD(__TBB_call_with_my_server_info,x)
#define SERVER tbb_server 
#define CLIENT tbb_client
#define FACTORY tbb_factory

#if __TBB_WEAK_SYMBOLS_PRESENT
#pragma weak __TBB_make_rml_server
#pragma weak __TBB_call_with_my_server_info
extern "C" {
::rml::factory::status_type __TBB_make_rml_server( tbb::internal::rml::tbb_factory& f, tbb::internal::rml::tbb_server*& server, tbb::internal::rml::tbb_client& client );
void __TBB_call_with_my_server_info( ::rml::server_info_callback_t cb, void* arg );
}
#endif 

#include "rml_factory.h"

} 
} 
} 

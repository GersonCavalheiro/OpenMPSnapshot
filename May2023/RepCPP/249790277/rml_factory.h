


#if TBB_USE_DEBUG
#define DEBUG_SUFFIX "_debug"
#else
#define DEBUG_SUFFIX
#endif 

#if _WIN32||_WIN64
#define RML_SERVER_NAME "irml" DEBUG_SUFFIX ".dll"
#elif __APPLE__
#define RML_SERVER_NAME "libirml" DEBUG_SUFFIX ".dylib"
#elif __linux__
#define RML_SERVER_NAME "libirml" DEBUG_SUFFIX ".so.1"
#elif __FreeBSD__ || __NetBSD__ || __sun || _AIX
#define RML_SERVER_NAME "libirml" DEBUG_SUFFIX ".so"
#else
#error Unknown OS
#endif

const ::rml::versioned_object::version_type CLIENT_VERSION = 2;

#if __TBB_WEAK_SYMBOLS_PRESENT
#pragma weak __RML_open_factory
#pragma weak __RML_close_factory
extern "C" {
::rml::factory::status_type __RML_open_factory ( ::rml::factory&, ::rml::versioned_object::version_type&, ::rml::versioned_object::version_type );
void __RML_close_factory( ::rml::factory& f );
}
#endif 

::rml::factory::status_type FACTORY::open() {
LIBRARY_ASSERT( !library_handle, NULL );
status_type (*open_factory_routine)( factory&, version_type&, version_type );
dynamic_link_descriptor server_link_table[4] = {
DLD(__RML_open_factory,open_factory_routine),
MAKE_SERVER(my_make_server_routine),
DLD(__RML_close_factory,my_wait_to_close_routine),
GET_INFO(my_call_with_server_info_routine),
};
status_type result;
if( dynamic_link( RML_SERVER_NAME, server_link_table, 4, &library_handle ) ) {
version_type server_version;
result = (*open_factory_routine)( *this, server_version, CLIENT_VERSION );
} else {
library_handle = NULL;
result = st_not_found;
}
return result;
}

void FACTORY::close() {
if( library_handle )
(*my_wait_to_close_routine)(*this);
if( (size_t)library_handle>FACTORY::c_dont_unload ) {
dynamic_unlink(library_handle);
library_handle = NULL;
}
}

::rml::factory::status_type FACTORY::make_server( SERVER*& s, CLIENT& c) {
LIBRARY_ASSERT( my_make_server_routine, NULL );
return (*my_make_server_routine)(*this,s,c);
}

void FACTORY::call_with_server_info( ::rml::server_info_callback_t cb, void* arg ) const {
LIBRARY_ASSERT( my_call_with_server_info_routine, NULL );
(*my_call_with_server_info_routine)( cb, arg );
}

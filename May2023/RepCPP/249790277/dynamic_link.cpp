

#include "dynamic_link.h"
#include "tbb/tbb_config.h"



#include <cstdarg>          
#if _WIN32
#include <malloc.h>

#define dlopen( name, flags )   LoadLibrary( name )
#define dlsym( handle, name )   GetProcAddress( handle, name )
#define dlclose( handle )       ( ! FreeLibrary( handle ) )
#define dlerror()               GetLastError()
#ifndef PATH_MAX
#define PATH_MAX                MAX_PATH
#endif
#else 
#include <dlfcn.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <stdlib.h>
#endif 

#if __TBB_WEAK_SYMBOLS_PRESENT && !__TBB_DYNAMIC_LOAD_ENABLED
#pragma weak dlopen
#pragma weak dlsym
#pragma weak dlclose
#endif 

#include "tbb/tbb_misc.h"

#define __USE_TBB_ATOMICS       ( !(__linux__&&__ia64__) || __TBB_BUILD )
#define __USE_STATIC_DL_INIT    ( !__ANDROID__ )

#if !__USE_TBB_ATOMICS
#include <pthread.h>
#endif



OPEN_INTERNAL_NAMESPACE

#if __TBB_WEAK_SYMBOLS_PRESENT || __TBB_DYNAMIC_LOAD_ENABLED

#if !defined(DYNAMIC_LINK_WARNING) && !__TBB_WIN8UI_SUPPORT && __TBB_DYNAMIC_LOAD_ENABLED
#define DYNAMIC_LINK_WARNING dynamic_link_warning
static void dynamic_link_warning( dynamic_link_error_t code, ... ) {
(void) code;
} 
#endif 

static bool resolve_symbols( dynamic_link_handle module, const dynamic_link_descriptor descriptors[], size_t required )
{
if ( !module )
return false;

#if !__TBB_DYNAMIC_LOAD_ENABLED 
if ( !dlsym ) return false;
#endif 

const size_t n_desc=20; 
LIBRARY_ASSERT( required <= n_desc, "Too many descriptors is required" );
if ( required > n_desc ) return false;
pointer_to_handler h[n_desc];

for ( size_t k = 0; k < required; ++k ) {
dynamic_link_descriptor const & desc = descriptors[k];
pointer_to_handler addr = (pointer_to_handler)dlsym( module, desc.name );
if ( !addr ) {
return false;
}
h[k] = addr;
}

for( size_t k = 0; k < required; ++k )
*descriptors[k].handler = h[k];
return true;
}

#if __TBB_WIN8UI_SUPPORT
bool dynamic_link( const char*  library, const dynamic_link_descriptor descriptors[], size_t required, dynamic_link_handle*, int flags ) {
dynamic_link_handle tmp_handle = NULL;
TCHAR wlibrary[256];
if ( MultiByteToWideChar(CP_UTF8, 0, library, -1, wlibrary, 255) == 0 ) return false;
if ( flags & DYNAMIC_LINK_LOAD )
tmp_handle = LoadPackagedLibrary( wlibrary, 0 );
if (tmp_handle != NULL){
return resolve_symbols(tmp_handle, descriptors, required);
}else{
return false;
}
}
void dynamic_unlink( dynamic_link_handle ) {}
void dynamic_unlink_all() {}
#else
#if __TBB_DYNAMIC_LOAD_ENABLED


#define MAX_LOADED_MODULES 8 

#if __USE_TBB_ATOMICS
typedef ::tbb::atomic<size_t> atomic_incrementer;
void init_atomic_incrementer( atomic_incrementer & ) {}

static void atomic_once( void( *func ) (void), tbb::atomic< tbb::internal::do_once_state > &once_state ) {
tbb::internal::atomic_do_once( func, once_state );
}
#define ATOMIC_ONCE_DECL( var ) tbb::atomic< tbb::internal::do_once_state > var
#else
static void pthread_assert( int error_code, const char* msg ) {
LIBRARY_ASSERT( error_code == 0, msg );
}

class atomic_incrementer {
size_t my_val;
pthread_spinlock_t my_lock;
public:
void init() {
my_val = 0;
pthread_assert( pthread_spin_init( &my_lock, PTHREAD_PROCESS_PRIVATE ), "pthread_spin_init failed" );
}
size_t operator++(int) {
pthread_assert( pthread_spin_lock( &my_lock ), "pthread_spin_lock failed" );
size_t prev_val = my_val++;
pthread_assert( pthread_spin_unlock( &my_lock ), "pthread_spin_unlock failed" );
return prev_val;
}
operator size_t() {
pthread_assert( pthread_spin_lock( &my_lock ), "pthread_spin_lock failed" );
size_t val = my_val;
pthread_assert( pthread_spin_unlock( &my_lock ), "pthread_spin_unlock failed" );
return val;
}
~atomic_incrementer() {
pthread_assert( pthread_spin_destroy( &my_lock ), "pthread_spin_destroy failed" );
}
};

void init_atomic_incrementer( atomic_incrementer &r ) {
r.init();
}

static void atomic_once( void( *func ) (), pthread_once_t &once_state ) {
pthread_assert( pthread_once( &once_state, func ), "pthread_once failed" );
}
#define ATOMIC_ONCE_DECL( var ) pthread_once_t var = PTHREAD_ONCE_INIT
#endif 

struct handles_t {
atomic_incrementer my_size;
dynamic_link_handle my_handles[MAX_LOADED_MODULES];

void init() {
init_atomic_incrementer( my_size );
}

void add(const dynamic_link_handle &handle) {
const size_t ind = my_size++;
LIBRARY_ASSERT( ind < MAX_LOADED_MODULES, "Too many modules are loaded" );
my_handles[ind] = handle;
}

void free() {
const size_t size = my_size;
for (size_t i=0; i<size; ++i)
dynamic_unlink( my_handles[i] );
}
} handles;

ATOMIC_ONCE_DECL( init_dl_data_state );

static struct ap_data_t {
char _path[PATH_MAX+1];
size_t _len;
} ap_data;

static void init_ap_data() {
#if _WIN32
HMODULE handle;
BOOL brc = GetModuleHandleEx(
GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
(LPCSTR)( & dynamic_link ), 
& handle
);
if ( !brc ) { 
int err = GetLastError();
DYNAMIC_LINK_WARNING( dl_sys_fail, "GetModuleHandleEx", err );
return;
}
DWORD drc = GetModuleFileName( handle, ap_data._path, static_cast< DWORD >( PATH_MAX ) );
if ( drc == 0 ) { 
int err = GetLastError();
DYNAMIC_LINK_WARNING( dl_sys_fail, "GetModuleFileName", err );
return;
}
if ( drc >= PATH_MAX ) { 
DYNAMIC_LINK_WARNING( dl_buff_too_small );
return;
}
char *backslash = strrchr( ap_data._path, '\\' );

if ( !backslash ) {    
LIBRARY_ASSERT( backslash!=NULL, "Unbelievable.");
return;
}
LIBRARY_ASSERT( backslash >= ap_data._path, "Unbelievable.");
ap_data._len = (size_t)(backslash - ap_data._path) + 1;
*(backslash+1) = 0;
#else
Dl_info dlinfo;
int res = dladdr( (void*)&dynamic_link, &dlinfo ); 
if ( !res ) {
char const * err = dlerror();
DYNAMIC_LINK_WARNING( dl_sys_fail, "dladdr", err );
return;
} else {
LIBRARY_ASSERT( dlinfo.dli_fname!=NULL, "Unbelievable." );
}

char const *slash = strrchr( dlinfo.dli_fname, '/' );
size_t fname_len=0;
if ( slash ) {
LIBRARY_ASSERT( slash >= dlinfo.dli_fname, "Unbelievable.");
fname_len = (size_t)(slash - dlinfo.dli_fname) + 1;
}

size_t rc;
if ( dlinfo.dli_fname[0]=='/' ) {
rc = 0;
ap_data._len = 0;
} else {
if ( !getcwd( ap_data._path, sizeof(ap_data._path)/sizeof(ap_data._path[0]) ) ) {
DYNAMIC_LINK_WARNING( dl_buff_too_small );
return;
}
ap_data._len = strlen( ap_data._path );
ap_data._path[ap_data._len++]='/';
rc = ap_data._len;
}

if ( fname_len>0 ) {
if ( ap_data._len>PATH_MAX ) {
DYNAMIC_LINK_WARNING( dl_buff_too_small );
ap_data._len=0;
return;
}
strncpy( ap_data._path+rc, dlinfo.dli_fname, fname_len );
ap_data._len += fname_len;
ap_data._path[ap_data._len]=0;
}
#endif 
}

static void init_dl_data() {
handles.init();
init_ap_data();
}


static size_t abs_path( char const * name, char * path, size_t len ) {
if ( !ap_data._len )
return 0;

size_t name_len = strlen( name );
size_t full_len = name_len+ap_data._len;
if ( full_len < len ) {
strncpy( path, ap_data._path, ap_data._len );
strncpy( path+ap_data._len, name, name_len );
path[full_len] = 0;
}
return full_len;
}
#endif  

void init_dynamic_link_data() {
#if __TBB_DYNAMIC_LOAD_ENABLED
atomic_once( &init_dl_data, init_dl_data_state );
#endif
}

#if __USE_STATIC_DL_INIT
static struct static_init_dl_data_t {
static_init_dl_data_t() {
init_dynamic_link_data();
}
} static_init_dl_data;
#endif

#if __TBB_WEAK_SYMBOLS_PRESENT
static bool weak_symbol_link( const dynamic_link_descriptor descriptors[], size_t required )
{
for ( size_t k = 0; k < required; ++k )
if ( !descriptors[k].ptr )
return false;
for ( size_t k = 0; k < required; ++k )
*descriptors[k].handler = (pointer_to_handler) descriptors[k].ptr;
return true;
}
#else
static bool weak_symbol_link( const dynamic_link_descriptor[], size_t ) {
return false;
}
#endif 

void dynamic_unlink( dynamic_link_handle handle ) {
#if !__TBB_DYNAMIC_LOAD_ENABLED 
if ( !dlclose ) return;
#endif
if ( handle ) {
dlclose( handle );
}
}

void dynamic_unlink_all() {
#if __TBB_DYNAMIC_LOAD_ENABLED
handles.free();
#endif
}

#if !_WIN32
#if __TBB_DYNAMIC_LOAD_ENABLED
static dynamic_link_handle pin_symbols( dynamic_link_descriptor desc, const dynamic_link_descriptor* descriptors, size_t required ) {
dynamic_link_handle library_handle = 0;
Dl_info info;
if ( dladdr( (void*)*desc.handler, &info ) ) {
library_handle = dlopen( info.dli_fname, RTLD_LAZY );
if ( library_handle ) {
if ( !resolve_symbols( library_handle, descriptors, required ) ) {
dynamic_unlink(library_handle);
library_handle = 0;
}
} else {
char const * err = dlerror();
DYNAMIC_LINK_WARNING( dl_lib_not_found, info.dli_fname, err );
}
}
return library_handle;
}
#endif 
#endif 

static dynamic_link_handle global_symbols_link( const char* library, const dynamic_link_descriptor descriptors[], size_t required ) {
::tbb::internal::suppress_unused_warning( library );
dynamic_link_handle library_handle;
#if _WIN32
if ( GetModuleHandleEx( 0, library, &library_handle ) ) {
if ( resolve_symbols( library_handle, descriptors, required ) )
return library_handle;
else
FreeLibrary( library_handle );
}
#else 
#if !__TBB_DYNAMIC_LOAD_ENABLED 
if ( !dlopen ) return 0;
#endif 
library_handle = dlopen( NULL, RTLD_LAZY );
#if !__ANDROID__
LIBRARY_ASSERT( library_handle, "The handle for the main program is NULL" );
#endif
#if __TBB_DYNAMIC_LOAD_ENABLED
pointer_to_handler handler;
dynamic_link_descriptor desc;
desc.name = descriptors[0].name;
desc.handler = &handler;
if ( resolve_symbols( library_handle, &desc, 1 ) ) {
dynamic_unlink( library_handle );
return pin_symbols( desc, descriptors, required );
}
#else  
if ( resolve_symbols( library_handle, descriptors, required ) )
return library_handle;
#endif
dynamic_unlink( library_handle );
#endif 
return 0;
}

static void save_library_handle( dynamic_link_handle src, dynamic_link_handle *dst ) {
LIBRARY_ASSERT( src, "The library handle to store must be non-zero" );
if ( dst )
*dst = src;
#if __TBB_DYNAMIC_LOAD_ENABLED
else
handles.add( src );
#endif 
}

dynamic_link_handle dynamic_load( const char* library, const dynamic_link_descriptor descriptors[], size_t required ) {
::tbb::internal::suppress_unused_warning( library, descriptors, required );
#if __TBB_DYNAMIC_LOAD_ENABLED

size_t const len = PATH_MAX + 1;
char path[ len ];
size_t rc = abs_path( library, path, len );
if ( 0 < rc && rc < len ) {
#if _WIN32
UINT prev_mode = SetErrorMode (SEM_FAILCRITICALERRORS);
#endif 
dynamic_link_handle library_handle = dlopen( path, RTLD_LAZY );
#if _WIN32
SetErrorMode (prev_mode);
#endif 
if( library_handle ) {
if( !resolve_symbols( library_handle, descriptors, required ) ) {
dynamic_unlink( library_handle );
library_handle = NULL;
}
} else
DYNAMIC_LINK_WARNING( dl_lib_not_found, path, dlerror() );
return library_handle;
} else if ( rc>=len )
DYNAMIC_LINK_WARNING( dl_buff_too_small );

#endif 
return 0;
}

bool dynamic_link( const char* library, const dynamic_link_descriptor descriptors[], size_t required, dynamic_link_handle *handle, int flags ) {
init_dynamic_link_data();

dynamic_link_handle library_handle = ( flags & DYNAMIC_LINK_GLOBAL ) ? global_symbols_link( library, descriptors, required ) : 0;

if ( !library_handle && ( flags & DYNAMIC_LINK_LOAD ) )
library_handle = dynamic_load( library, descriptors, required );

if ( !library_handle && ( flags & DYNAMIC_LINK_WEAK ) )
return weak_symbol_link( descriptors, required );

if ( library_handle ) {
save_library_handle( library_handle, handle );
return true;
}
return false;
}

#endif 
#else 
bool dynamic_link( const char*, const dynamic_link_descriptor*, size_t, dynamic_link_handle *handle, int ) {
if ( handle )
*handle=0;
return false;
}
void dynamic_unlink( dynamic_link_handle ) {}
void dynamic_unlink_all() {}
#endif 

CLOSE_INTERNAL_NAMESPACE

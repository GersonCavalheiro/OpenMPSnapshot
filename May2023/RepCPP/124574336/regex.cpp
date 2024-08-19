




#define BOOST_REGEX_SOURCE

#include <boost/config.hpp>
#include <new>
#include <boost/regex.hpp>
#include <boost/throw_exception.hpp>

#if defined(BOOST_REGEX_HAS_MS_STACK_GUARD) && defined(_MSC_VER) && (_MSC_VER >= 1300)
#  include <malloc.h>
#endif
#ifdef BOOST_REGEX_HAS_MS_STACK_GUARD
#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#  define NOMINMAX
#endif
#define NOGDI
#define NOUSER
#include <windows.h>
#endif

#if defined(BOOST_REGEX_NON_RECURSIVE) && !defined(BOOST_REGEX_V3)
#if BOOST_REGEX_MAX_CACHE_BLOCKS == 0
#include <new>
#else
#include <boost/regex/v4/mem_block_cache.hpp>
#endif
#endif

#ifdef BOOST_INTEL
#pragma warning(disable:383)
#endif

namespace boost{

regex_error::regex_error(const std::string& s, regex_constants::error_type err, std::ptrdiff_t pos) 
: std::runtime_error(s)
, m_error_code(err)
, m_position(pos) 
{
}

regex_error::regex_error(regex_constants::error_type err) 
: std::runtime_error(::boost::BOOST_REGEX_DETAIL_NS::get_default_error_string(err))
, m_error_code(err)
, m_position(0) 
{
}

regex_error::~regex_error() BOOST_NOEXCEPT_OR_NOTHROW
{
}

void regex_error::raise()const
{
#ifndef BOOST_NO_EXCEPTIONS
::boost::throw_exception(*this);
#endif
}



namespace BOOST_REGEX_DETAIL_NS{

BOOST_REGEX_DECL void BOOST_REGEX_CALL raise_runtime_error(const std::runtime_error& ex)
{
::boost::throw_exception(ex);
}
BOOST_REGEX_DECL void BOOST_REGEX_CALL verify_options(boost::regex::flag_type , match_flag_type mf)
{
#ifndef BOOST_REGEX_V3
if((mf & match_extra) && (mf & match_posix))
{
std::logic_error msg("Usage Error: Can't mix regular expression captures with POSIX matching rules");
throw_exception(msg);
}
#endif
}

#ifdef BOOST_REGEX_HAS_MS_STACK_GUARD

static void execute_eror()
{
reset_stack_guard_page();
std::runtime_error err("Out of stack space, while attempting to match a regular expression.");
raise_runtime_error(err);
}

bool BOOST_REGEX_CALL abstract_protected_call::execute()const
{
__try{
return this->call();
}__except(EXCEPTION_STACK_OVERFLOW == GetExceptionCode())
{
execute_eror();
}
return false;
}

BOOST_REGEX_DECL void BOOST_REGEX_CALL reset_stack_guard_page()
{
#if defined(BOOST_REGEX_HAS_MS_STACK_GUARD) && defined(_MSC_VER) && (_MSC_VER >= 1300)
_resetstkoflw();
#else
SYSTEM_INFO si;
GetSystemInfo(&si);
MEMORY_BASIC_INFORMATION mi;
DWORD previous_protection_status;
LPBYTE page = (LPBYTE)&page;
VirtualQuery(page, &mi, sizeof(mi));
page = (LPBYTE)(mi.BaseAddress)-si.dwPageSize;
if (!VirtualFree(mi.AllocationBase, (LPBYTE)page - (LPBYTE)mi.AllocationBase, MEM_DECOMMIT)
|| !VirtualProtect(page, si.dwPageSize, PAGE_GUARD | PAGE_READWRITE, &previous_protection_status))
{
throw std::bad_exception();
}
#endif
}
#endif

#if defined(BOOST_REGEX_NON_RECURSIVE) && !defined(BOOST_REGEX_V3)

#if BOOST_REGEX_MAX_CACHE_BLOCKS == 0

BOOST_REGEX_DECL void* BOOST_REGEX_CALL get_mem_block()
{
return ::operator new(BOOST_REGEX_BLOCKSIZE);
}

BOOST_REGEX_DECL void BOOST_REGEX_CALL put_mem_block(void* p)
{
::operator delete(p);
}

#else

#if defined(BOOST_REGEX_MEM_BLOCK_CACHE_LOCK_FREE)
mem_block_cache block_cache = { { {nullptr} } } ;
#elif defined(BOOST_HAS_THREADS)
mem_block_cache block_cache = { 0, 0, BOOST_STATIC_MUTEX_INIT, };
#else
mem_block_cache block_cache = { 0, 0, };
#endif

BOOST_REGEX_DECL void* BOOST_REGEX_CALL get_mem_block()
{
return block_cache.get();
}

BOOST_REGEX_DECL void BOOST_REGEX_CALL put_mem_block(void* p)
{
block_cache.put(p);
}

#endif

#endif

} 



} 

#if defined(BOOST_RE_USE_VCL) && defined(BOOST_REGEX_DYN_LINK)

int WINAPI DllEntryPoint(HINSTANCE , unsigned long , void*)
{
return 1;
}
#endif


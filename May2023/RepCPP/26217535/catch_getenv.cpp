


#include <catch2/internal/catch_getenv.hpp>

#include <catch2/internal/catch_platform.hpp>
#include <catch2/internal/catch_compiler_capabilities.hpp>

#include <cstdlib>

namespace Catch {
namespace Detail {

#if !defined (CATCH_CONFIG_GETENV)
char const* getEnv( char const* ) { return nullptr; }
#else

char const* getEnv( char const* varName ) {
#    if defined( _MSC_VER )
#        pragma warning( push )
#        pragma warning( disable : 4996 ) 
#    endif

return std::getenv( varName );

#    if defined( _MSC_VER )
#        pragma warning( pop )
#    endif
}
#endif
} 
} 

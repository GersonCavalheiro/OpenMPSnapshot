

#include "boost/fiber/numa/pin_thread.hpp"

#include <system_error>

#include "boost/fiber/exceptions.hpp"

#ifdef BOOST_HAS_ABI_HEADERS
# include BOOST_ABI_PREFIX
#endif

namespace boost {
namespace fibers {
namespace numa {

#if BOOST_COMP_CLANG || \
BOOST_COMP_GNUC || \
BOOST_COMP_INTEL ||  \
BOOST_COMP_MSVC 
# pragma message "pin_thread() not supported"
#endif

BOOST_FIBERS_DECL
void pin_thread( std::uint32_t) {
throw fiber_error{
std::make_error_code( std::errc::function_not_supported),
"boost fiber: pin_thread() not supported" };
}

BOOST_FIBERS_DECL
void pin_thread( std::uint32_t cpuid, std::thread::native_handle_type h) {
throw fiber_error{
std::make_error_code( std::errc::function_not_supported),
"boost fiber: pin_thread() not supported" };
}

}}}

#ifdef BOOST_HAS_ABI_HEADERS
# include BOOST_ABI_SUFFIX
#endif

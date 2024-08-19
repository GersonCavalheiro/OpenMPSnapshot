


#include <ostream>

#include <boost/config.hpp>

#ifdef BOOST_NO_STD_WSTREAMBUF
#error "wide char i/o not supported on this platform"
#else

#if (defined _MSC_VER) && (_MSC_VER == 1200)
#  pragma warning (disable : 4786) 
#endif

#define BOOST_WARCHIVE_SOURCE
#include <boost/serialization/config.hpp>
#include <boost/archive/detail/auto_link_warchive.hpp>
#include <boost/archive/impl/basic_text_oprimitive.ipp>

namespace boost {
namespace archive {

template class basic_text_oprimitive<std::wostream> ;

} 
} 

#endif 

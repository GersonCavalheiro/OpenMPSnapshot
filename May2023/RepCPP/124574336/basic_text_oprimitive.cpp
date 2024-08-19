


#if (defined _MSC_VER) && (_MSC_VER == 1200)
#  pragma warning (disable : 4786) 
#endif

#include <ostream>

#define BOOST_ARCHIVE_SOURCE
#include <boost/serialization/config.hpp>
#include <boost/archive/detail/auto_link_archive.hpp>
#include <boost/archive/impl/basic_text_oprimitive.ipp>

namespace boost {
namespace archive {

template class basic_text_oprimitive<std::ostream> ;

} 
} 

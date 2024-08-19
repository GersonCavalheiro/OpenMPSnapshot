


#if (defined _MSC_VER) && (_MSC_VER == 1200)
#  pragma warning (disable : 4786) 
#endif

#define BOOST_ARCHIVE_SOURCE
#include <boost/serialization/config.hpp>
#include <boost/archive/polymorphic_xml_oarchive.hpp>

#include <boost/archive/impl/archive_serializer_map.ipp>

namespace boost {
namespace archive {
namespace detail {

template class detail::archive_serializer_map<polymorphic_xml_oarchive>;

} 
} 
} 

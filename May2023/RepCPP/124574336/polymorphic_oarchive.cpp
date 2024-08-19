


#if (defined _MSC_VER) && (_MSC_VER == 1200)
#  pragma warning (disable : 4786) 
#endif

#define BOOST_ARCHIVE_SOURCE
#include <boost/serialization/config.hpp>
#include <boost/archive/detail/archive_serializer_map.hpp>

#include <boost/archive/impl/archive_serializer_map.ipp>
#include <boost/archive/polymorphic_oarchive.hpp>

namespace boost {
namespace archive {
namespace detail {

template class archive_serializer_map<polymorphic_oarchive>;

} 
} 
} 

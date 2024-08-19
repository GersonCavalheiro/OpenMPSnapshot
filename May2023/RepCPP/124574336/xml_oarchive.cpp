


#if (defined _MSC_VER) && (_MSC_VER == 1200)
#  pragma warning (disable : 4786) 
#endif

#define BOOST_ARCHIVE_SOURCE
#include <boost/serialization/config.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/detail/archive_serializer_map.hpp>

#include <boost/archive/impl/archive_serializer_map.ipp>
#include <boost/archive/impl/basic_xml_oarchive.ipp>
#include <boost/archive/impl/xml_oarchive_impl.ipp>

namespace boost {
namespace archive {

template class detail::archive_serializer_map<xml_oarchive>;
template class basic_xml_oarchive<xml_oarchive> ;
template class xml_oarchive_impl<xml_oarchive> ;

} 
} 

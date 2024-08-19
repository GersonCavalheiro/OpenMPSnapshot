


#if (defined _MSC_VER) && (_MSC_VER == 1200)
#  pragma warning (disable : 4786) 
#endif

#define BOOST_ARCHIVE_SOURCE
#include <boost/serialization/config.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/detail/archive_serializer_map.hpp>

#include <boost/archive/impl/archive_serializer_map.ipp>
#include <boost/archive/impl/basic_xml_iarchive.ipp>
#include <boost/archive/impl/xml_iarchive_impl.ipp>

namespace boost {
namespace archive {

template class detail::archive_serializer_map<xml_iarchive>;
template class basic_xml_iarchive<xml_iarchive> ;
template class xml_iarchive_impl<xml_iarchive> ;

} 
} 




#if (defined _MSC_VER) && (_MSC_VER == 1200)
#  pragma warning (disable : 4786) 
#endif

#include <boost/config.hpp>
#ifdef BOOST_NO_STD_WSTREAMBUF
#error "wide char i/o not supported on this platform"
#else

#define BOOST_WARCHIVE_SOURCE
#include <boost/serialization/config.hpp>
#include <boost/archive/xml_woarchive.hpp>
#include <boost/archive/detail/archive_serializer_map.hpp>

#include <boost/archive/impl/archive_serializer_map.ipp>
#include <boost/archive/impl/basic_xml_oarchive.ipp>
#include <boost/archive/impl/xml_woarchive_impl.ipp>

namespace boost {
namespace archive {

template class detail::archive_serializer_map<xml_woarchive>;
template class basic_xml_oarchive<xml_woarchive> ;
template class xml_woarchive_impl<xml_woarchive> ;

} 
} 

#endif 

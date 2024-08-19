#ifndef BOOST_ARCHIVE_POLYMORPHIC_XML_WOARCHIVE_HPP
#define BOOST_ARCHIVE_POLYMORPHIC_XML_WOARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>
#ifdef BOOST_NO_STD_WSTREAMBUF
#error "wide char i/o not supported on this platform"
#else

#include <boost/archive/xml_woarchive.hpp>
#include <boost/archive/detail/polymorphic_oarchive_route.hpp>

namespace boost {
namespace archive {

class BOOST_SYMBOL_VISIBLE polymorphic_xml_woarchive :
public detail::polymorphic_oarchive_route<xml_woarchive>
{
public:
polymorphic_xml_woarchive(std::wostream & os, unsigned int flags = 0) :
detail::polymorphic_oarchive_route<xml_woarchive>(os, flags)
{}
~polymorphic_xml_woarchive() BOOST_OVERRIDE {}
};

} 
} 

BOOST_SERIALIZATION_REGISTER_ARCHIVE(
boost::archive::polymorphic_xml_woarchive
)

#endif 
#endif 

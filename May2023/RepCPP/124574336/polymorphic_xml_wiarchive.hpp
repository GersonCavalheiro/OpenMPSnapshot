#ifndef BOOST_ARCHIVE_POLYMORPHIC_XML_WIARCHIVE_HPP
#define BOOST_ARCHIVE_POLYMORPHIC_XML_WIARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>
#ifdef BOOST_NO_STD_WSTREAMBUF
#error "wide char i/o not supported on this platform"
#else

#include <boost/archive/xml_wiarchive.hpp>
#include <boost/archive/detail/polymorphic_iarchive_route.hpp>

namespace boost {
namespace archive {

class BOOST_SYMBOL_VISIBLE polymorphic_xml_wiarchive :
public detail::polymorphic_iarchive_route<xml_wiarchive>
{
public:
polymorphic_xml_wiarchive(std::wistream & is, unsigned int flags = 0) :
detail::polymorphic_iarchive_route<xml_wiarchive>(is, flags)
{}
~polymorphic_xml_wiarchive() BOOST_OVERRIDE {}
};

} 
} 

BOOST_SERIALIZATION_REGISTER_ARCHIVE(
boost::archive::polymorphic_xml_wiarchive
)

#endif 
#endif 

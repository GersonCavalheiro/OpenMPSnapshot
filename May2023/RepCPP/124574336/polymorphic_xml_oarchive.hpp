#ifndef BOOST_ARCHIVE_POLYMORPHIC_XML_OARCHIVE_HPP
#define BOOST_ARCHIVE_POLYMORPHIC_XML_OARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/detail/polymorphic_oarchive_route.hpp>

namespace boost {
namespace archive {

class BOOST_SYMBOL_VISIBLE polymorphic_xml_oarchive :
public detail::polymorphic_oarchive_route<xml_oarchive>
{
public:
polymorphic_xml_oarchive(std::ostream & os, unsigned int flags = 0) :
detail::polymorphic_oarchive_route<xml_oarchive>(os, flags)
{}
~polymorphic_xml_oarchive() BOOST_OVERRIDE {}
};
} 
} 

BOOST_SERIALIZATION_REGISTER_ARCHIVE(
boost::archive::polymorphic_xml_oarchive
)

#endif 

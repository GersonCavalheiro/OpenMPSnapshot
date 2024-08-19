#ifndef BOOST_ARCHIVE_POLYMORPHIC_XML_IARCHIVE_HPP
#define BOOST_ARCHIVE_POLYMORPHIC_XML_IARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/detail/polymorphic_iarchive_route.hpp>

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif

namespace boost {
namespace archive {

class BOOST_SYMBOL_VISIBLE polymorphic_xml_iarchive :
public detail::polymorphic_iarchive_route<xml_iarchive>
{
public:
polymorphic_xml_iarchive(std::istream & is, unsigned int flags = 0) :
detail::polymorphic_iarchive_route<xml_iarchive>(is, flags)
{}
~polymorphic_xml_iarchive() BOOST_OVERRIDE {}
};

} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

BOOST_SERIALIZATION_REGISTER_ARCHIVE(
boost::archive::polymorphic_xml_iarchive
)

#endif 

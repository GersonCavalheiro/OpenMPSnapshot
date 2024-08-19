#ifndef BOOST_ARCHIVE_POLYMORPHIC_TEXT_WIARCHIVE_HPP
#define BOOST_ARCHIVE_POLYMORPHIC_TEXT_WIARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>
#ifdef BOOST_NO_STD_WSTREAMBUF
#error "wide char i/o not supported on this platform"
#else

#include <boost/archive/text_wiarchive.hpp>
#include <boost/archive/detail/polymorphic_iarchive_route.hpp>

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif

namespace boost {
namespace archive {

class BOOST_SYMBOL_VISIBLE polymorphic_text_wiarchive :
public detail::polymorphic_iarchive_route<text_wiarchive>
{
public:
polymorphic_text_wiarchive(std::wistream & is, unsigned int flags = 0) :
detail::polymorphic_iarchive_route<text_wiarchive>(is, flags)
{}
~polymorphic_text_wiarchive() BOOST_OVERRIDE {}
};

} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

BOOST_SERIALIZATION_REGISTER_ARCHIVE(
boost::archive::polymorphic_text_wiarchive
)

#endif 
#endif 

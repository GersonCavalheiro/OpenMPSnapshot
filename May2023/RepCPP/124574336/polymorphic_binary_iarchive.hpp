#ifndef BOOST_ARCHIVE_POLYMORPHIC_BINARY_IARCHIVE_HPP
#define BOOST_ARCHIVE_POLYMORPHIC_BINARY_IARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/detail/polymorphic_iarchive_route.hpp>

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif

namespace boost {
namespace archive {

class BOOST_SYMBOL_VISIBLE polymorphic_binary_iarchive :
public detail::polymorphic_iarchive_route<binary_iarchive>
{
public:
polymorphic_binary_iarchive(std::istream & is, unsigned int flags = 0) :
detail::polymorphic_iarchive_route<binary_iarchive>(is, flags)
{}
~polymorphic_binary_iarchive() BOOST_OVERRIDE {}
};

} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

BOOST_SERIALIZATION_REGISTER_ARCHIVE(
boost::archive::polymorphic_binary_iarchive
)

#endif 

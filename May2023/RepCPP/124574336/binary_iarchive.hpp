#ifndef BOOST_ARCHIVE_BINARY_IARCHIVE_HPP
#define BOOST_ARCHIVE_BINARY_IARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <istream>
#include <boost/archive/binary_iarchive_impl.hpp>
#include <boost/archive/detail/register_archive.hpp>

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif

namespace boost {
namespace archive {

class BOOST_SYMBOL_VISIBLE binary_iarchive :
public binary_iarchive_impl<
boost::archive::binary_iarchive,
std::istream::char_type,
std::istream::traits_type
>{
public:
binary_iarchive(std::istream & is, unsigned int flags = 0) :
binary_iarchive_impl<
binary_iarchive, std::istream::char_type, std::istream::traits_type
>(is, flags)
{
init(flags);
}
binary_iarchive(std::streambuf & bsb, unsigned int flags = 0) :
binary_iarchive_impl<
binary_iarchive, std::istream::char_type, std::istream::traits_type
>(bsb, flags)
{
init(flags);
}
};

} 
} 

BOOST_SERIALIZATION_REGISTER_ARCHIVE(boost::archive::binary_iarchive)
BOOST_SERIALIZATION_USE_ARRAY_OPTIMIZATION(boost::archive::binary_iarchive)

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif 

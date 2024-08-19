#ifndef BOOST_ARCHIVE_BINARY_OARCHIVE_HPP
#define BOOST_ARCHIVE_BINARY_OARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <ostream>
#include <boost/config.hpp>
#include <boost/archive/binary_oarchive_impl.hpp>
#include <boost/archive/detail/register_archive.hpp>

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif

namespace boost {
namespace archive {

class BOOST_SYMBOL_VISIBLE binary_oarchive :
public binary_oarchive_impl<
binary_oarchive, std::ostream::char_type, std::ostream::traits_type
>
{
public:
binary_oarchive(std::ostream & os, unsigned int flags = 0) :
binary_oarchive_impl<
binary_oarchive, std::ostream::char_type, std::ostream::traits_type
>(os, flags)
{
init(flags);
}
binary_oarchive(std::streambuf & bsb, unsigned int flags = 0) :
binary_oarchive_impl<
binary_oarchive, std::ostream::char_type, std::ostream::traits_type
>(bsb, flags)
{
init(flags);
}
};

} 
} 

BOOST_SERIALIZATION_REGISTER_ARCHIVE(boost::archive::binary_oarchive)
BOOST_SERIALIZATION_USE_ARRAY_OPTIMIZATION(boost::archive::binary_oarchive)

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif 

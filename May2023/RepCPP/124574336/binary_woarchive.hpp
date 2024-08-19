#ifndef BOOST_ARCHIVE_BINARY_WOARCHIVE_HPP
#define BOOST_ARCHIVE_BINARY_WOARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>
#ifdef BOOST_NO_STD_WSTREAMBUF
#error "wide char i/o not supported on this platform"
#else

#include <ostream>
#include <boost/archive/binary_oarchive_impl.hpp>
#include <boost/archive/detail/register_archive.hpp>

namespace boost {
namespace archive {

class binary_woarchive :
public binary_oarchive_impl<
binary_woarchive, std::wostream::char_type, std::wostream::traits_type
>
{
public:
binary_woarchive(std::wostream & os, unsigned int flags = 0) :
binary_oarchive_impl<
binary_woarchive, std::wostream::char_type, std::wostream::traits_type
>(os, flags)
{}
binary_woarchive(std::wstreambuf & bsb, unsigned int flags = 0) :
binary_oarchive_impl<
binary_woarchive, std::wostream::char_type, std::wostream::traits_type
>(bsb, flags)
{}
};

} 
} 

BOOST_SERIALIZATION_REGISTER_ARCHIVE(boost::archive::binary_woarchive)

#endif 
#endif 

#ifndef BOOST_ARCHIVE_BINARY_WIARCHIVE_HPP
#define BOOST_ARCHIVE_BINARY_WIARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>
#ifdef BOOST_NO_STD_WSTREAMBUF
#error "wide char i/o not supported on this platform"
#else

#include <istream> 
#include <boost/archive/binary_iarchive_impl.hpp>
#include <boost/archive/detail/register_archive.hpp>

namespace boost {
namespace archive {

class binary_wiarchive :
public binary_iarchive_impl<
binary_wiarchive, std::wistream::char_type, std::wistream::traits_type
>
{
public:
binary_wiarchive(std::wistream & is, unsigned int flags = 0) :
binary_iarchive_impl<
binary_wiarchive, std::wistream::char_type, std::wistream::traits_type
>(is, flags)
{}
binary_wiarchive(std::wstreambuf & bsb, unsigned int flags = 0) :
binary_iarchive_impl<
binary_wiarchive, std::wistream::char_type, std::wistream::traits_type
>(bsb, flags)
{}
};

} 
} 

BOOST_SERIALIZATION_REGISTER_ARCHIVE(boost::archive::binary_wiarchive)

#endif 
#endif 

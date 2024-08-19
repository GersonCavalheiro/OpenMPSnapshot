#ifndef  BOOST_ARCHIVE_BASIC_SERIALIZER_HPP
#define BOOST_ARCHIVE_BASIC_SERIALIZER_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/assert.hpp>
#include <cstddef> 

#include <boost/noncopyable.hpp>
#include <boost/config.hpp>
#include <boost/serialization/extended_type_info.hpp>

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif

namespace boost {
namespace archive {
namespace detail {

class basic_serializer :
private boost::noncopyable
{
const boost::serialization::extended_type_info * m_eti;
protected:
explicit basic_serializer(
const boost::serialization::extended_type_info & eti
) :
m_eti(& eti)
{}
public:
inline bool
operator<(const basic_serializer & rhs) const {
return get_eti() < rhs.get_eti();
}
const char * get_debug_info() const {
return m_eti->get_debug_info();
}
const boost::serialization::extended_type_info & get_eti() const {
return * m_eti;
}
};

class basic_serializer_arg : public basic_serializer {
public:
basic_serializer_arg(const serialization::extended_type_info & eti) :
basic_serializer(eti)
{}
};

} 
} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif 

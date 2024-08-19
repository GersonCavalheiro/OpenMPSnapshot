#ifndef  BOOST_SERIALIZER_MAP_HPP
#define BOOST_SERIALIZER_MAP_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <set>

#include <boost/config.hpp>
#include <boost/noncopyable.hpp>
#include <boost/archive/detail/auto_link_archive.hpp>

#include <boost/archive/detail/abi_prefix.hpp> 

namespace boost {
namespace serialization {
class extended_type_info;
}

namespace archive {
namespace detail {

class basic_serializer;

class BOOST_SYMBOL_VISIBLE
basic_serializer_map : public
boost::noncopyable
{
struct type_info_pointer_compare
{
bool operator()(
const basic_serializer * lhs, const basic_serializer * rhs
) const ;
};
typedef std::set<
const basic_serializer *,
type_info_pointer_compare
> map_type;
map_type m_map;
public:
BOOST_ARCHIVE_DECL bool insert(const basic_serializer * bs);
BOOST_ARCHIVE_DECL void erase(const basic_serializer * bs);
BOOST_ARCHIVE_DECL const basic_serializer * find(
const boost::serialization::extended_type_info & type_
) const;
private:
basic_serializer_map& operator=(basic_serializer_map const&);
};

} 
} 
} 

#include <boost/archive/detail/abi_suffix.hpp> 

#endif 

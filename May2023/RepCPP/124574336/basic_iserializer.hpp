#ifndef BOOST_ARCHIVE_DETAIL_BASIC_ISERIALIZER_HPP
#define BOOST_ARCHIVE_DETAIL_BASIC_ISERIALIZER_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <cstdlib> 
#include <boost/config.hpp>

#include <boost/archive/basic_archive.hpp>
#include <boost/archive/detail/decl.hpp>
#include <boost/archive/detail/basic_serializer.hpp>
#include <boost/archive/detail/auto_link_archive.hpp>
#include <boost/archive/detail/abi_prefix.hpp> 

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif

namespace boost {
namespace serialization {
class extended_type_info;
} 

namespace archive {
namespace detail {

class basic_iarchive;
class basic_pointer_iserializer;

class BOOST_SYMBOL_VISIBLE basic_iserializer :
public basic_serializer
{
private:
basic_pointer_iserializer *m_bpis;
protected:
explicit BOOST_ARCHIVE_DECL basic_iserializer(
const boost::serialization::extended_type_info & type
);
virtual BOOST_ARCHIVE_DECL ~basic_iserializer();
public:
bool serialized_as_pointer() const {
return m_bpis != NULL;
}
void set_bpis(basic_pointer_iserializer *bpis){
m_bpis = bpis;
}
const basic_pointer_iserializer * get_bpis_ptr() const {
return m_bpis;
}
virtual void load_object_data(
basic_iarchive & ar,
void *x,
const unsigned int file_version
) const = 0;
virtual bool class_info() const = 0 ;
virtual bool tracking(const unsigned int) const = 0 ;
virtual version_type version() const = 0 ;
virtual bool is_polymorphic() const = 0;
virtual void destroy( void *address) const = 0 ;
};

} 
} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#include <boost/archive/detail/abi_suffix.hpp> 

#endif 

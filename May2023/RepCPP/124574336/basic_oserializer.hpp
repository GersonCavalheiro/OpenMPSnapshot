#ifndef BOOST_SERIALIZATION_BASIC_OSERIALIZER_HPP
#define BOOST_SERIALIZATION_BASIC_OSERIALIZER_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <cstddef> 
#include <boost/config.hpp>
#include <boost/noncopyable.hpp>

#include <boost/archive/basic_archive.hpp>
#include <boost/archive/detail/auto_link_archive.hpp>
#include <boost/archive/detail/basic_serializer.hpp>

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

class basic_oarchive;
class basic_pointer_oserializer;

class BOOST_SYMBOL_VISIBLE basic_oserializer :
public basic_serializer
{
private:
basic_pointer_oserializer *m_bpos;
protected:
explicit BOOST_ARCHIVE_DECL basic_oserializer(
const boost::serialization::extended_type_info & type_
);
virtual BOOST_ARCHIVE_DECL ~basic_oserializer();
public:
bool serialized_as_pointer() const {
return m_bpos != NULL;
}
void set_bpos(basic_pointer_oserializer *bpos){
m_bpos = bpos;
}
const basic_pointer_oserializer * get_bpos() const {
return m_bpos;
}
virtual void save_object_data(
basic_oarchive & ar, const void * x
) const = 0;
virtual bool class_info() const = 0;
virtual bool tracking(const unsigned int flags) const = 0;
virtual version_type version() const = 0;
virtual bool is_polymorphic() const = 0;
};

} 
} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#include <boost/archive/detail/abi_suffix.hpp> 

#endif 

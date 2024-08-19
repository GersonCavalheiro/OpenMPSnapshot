#ifndef BOOST_SERIALIZATION_EXTENDED_TYPE_INFO_TYPEID_HPP
#define BOOST_SERIALIZATION_EXTENDED_TYPE_INFO_TYPEID_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <typeinfo>
#include <cstdarg>
#include <boost/assert.hpp>
#include <boost/config.hpp>

#include <boost/static_assert.hpp>
#include <boost/serialization/static_warning.hpp>
#include <boost/type_traits/is_polymorphic.hpp>
#include <boost/type_traits/remove_const.hpp>

#include <boost/serialization/config.hpp>
#include <boost/serialization/singleton.hpp>
#include <boost/serialization/extended_type_info.hpp>
#include <boost/serialization/factory.hpp>

#include <boost/serialization/access.hpp>

#include <boost/mpl/if.hpp>

#include <boost/config/abi_prefix.hpp> 

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4251 4231 4660 4275 4511 4512)
#endif

namespace boost {
namespace serialization {
namespace typeid_system {

class BOOST_SYMBOL_VISIBLE extended_type_info_typeid_0 :
public extended_type_info
{
const char * get_debug_info() const BOOST_OVERRIDE {
if(static_cast<const std::type_info *>(0) == m_ti)
return static_cast<const char *>(0);
return m_ti->name();
}
protected:
const std::type_info * m_ti;
BOOST_SERIALIZATION_DECL extended_type_info_typeid_0(const char * key);
BOOST_SERIALIZATION_DECL ~extended_type_info_typeid_0() BOOST_OVERRIDE;
BOOST_SERIALIZATION_DECL void type_register(const std::type_info & ti);
BOOST_SERIALIZATION_DECL void type_unregister();
BOOST_SERIALIZATION_DECL const extended_type_info *
get_extended_type_info(const std::type_info & ti) const;
public:
BOOST_SERIALIZATION_DECL bool
is_less_than(const extended_type_info &rhs) const BOOST_OVERRIDE;
BOOST_SERIALIZATION_DECL bool
is_equal(const extended_type_info &rhs) const BOOST_OVERRIDE;
const std::type_info & get_typeid() const {
return *m_ti;
}
};

} 

template<class T>
class extended_type_info_typeid :
public typeid_system::extended_type_info_typeid_0,
public singleton<extended_type_info_typeid< T > >
{
public:
extended_type_info_typeid() :
typeid_system::extended_type_info_typeid_0(
boost::serialization::guid< T >()
)
{
type_register(typeid(T));
key_register();
}
~extended_type_info_typeid() BOOST_OVERRIDE {
key_unregister();
type_unregister();
}
const extended_type_info *
get_derived_extended_type_info(const T & t) const {
BOOST_STATIC_WARNING(boost::is_polymorphic< T >::value);
return
typeid_system::extended_type_info_typeid_0::get_extended_type_info(
typeid(t)
);
}
const char * get_key() const {
return boost::serialization::guid< T >();
}
void * construct(unsigned int count, ...) const BOOST_OVERRIDE {
std::va_list ap;
va_start(ap, count);
switch(count){
case 0:
return factory<typename boost::remove_const< T >::type, 0>(ap);
case 1:
return factory<typename boost::remove_const< T >::type, 1>(ap);
case 2:
return factory<typename boost::remove_const< T >::type, 2>(ap);
case 3:
return factory<typename boost::remove_const< T >::type, 3>(ap);
case 4:
return factory<typename boost::remove_const< T >::type, 4>(ap);
default:
BOOST_ASSERT(false); 
return NULL;
}
}
void destroy(void const * const p) const BOOST_OVERRIDE {
boost::serialization::access::destroy(
static_cast<T const *>(p)
);
}
};

} 
} 

#ifndef BOOST_SERIALIZATION_DEFAULT_TYPE_INFO
#define BOOST_SERIALIZATION_DEFAULT_TYPE_INFO
namespace boost {
namespace serialization {
template<class T>
struct extended_type_info_impl {
typedef typename
boost::serialization::extended_type_info_typeid< T > type;
};
} 
} 
#endif

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif
#include <boost/config/abi_suffix.hpp> 

#endif 

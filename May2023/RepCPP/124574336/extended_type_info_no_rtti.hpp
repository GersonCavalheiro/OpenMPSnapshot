#ifndef BOOST_EXTENDED_TYPE_INFO_NO_RTTI_HPP
#define BOOST_EXTENDED_TYPE_INFO_NO_RTTI_HPP

#if defined(_MSC_VER)
# pragma once
#endif



#include <boost/assert.hpp>

#include <boost/config.hpp>
#include <boost/static_assert.hpp>

#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_polymorphic.hpp>
#include <boost/type_traits/remove_const.hpp>

#include <boost/serialization/static_warning.hpp>
#include <boost/serialization/singleton.hpp>
#include <boost/serialization/extended_type_info.hpp>
#include <boost/serialization/factory.hpp>
#include <boost/serialization/throw_exception.hpp>

#include <boost/serialization/config.hpp>
#include <boost/serialization/access.hpp>

#include <boost/config/abi_prefix.hpp> 
#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4251 4231 4660 4275 4511 4512)
#endif

namespace boost {
namespace serialization {

namespace no_rtti_system {

class BOOST_SYMBOL_VISIBLE extended_type_info_no_rtti_0 :
public extended_type_info
{
protected:
BOOST_SERIALIZATION_DECL extended_type_info_no_rtti_0(const char * key);
BOOST_SERIALIZATION_DECL ~extended_type_info_no_rtti_0() BOOST_OVERRIDE;
public:
BOOST_SERIALIZATION_DECL bool
is_less_than(const boost::serialization::extended_type_info &rhs) const BOOST_OVERRIDE;
BOOST_SERIALIZATION_DECL bool
is_equal(const boost::serialization::extended_type_info &rhs) const BOOST_OVERRIDE;
};

} 

template<class T>
class extended_type_info_no_rtti :
public no_rtti_system::extended_type_info_no_rtti_0,
public singleton<extended_type_info_no_rtti< T > >
{
template<bool tf>
struct action {
struct defined {
static const char * invoke(){
return guid< T >();
}
};
struct undefined {
BOOST_STATIC_ASSERT(0 == sizeof(T));
static const char * invoke();
};
static const char * invoke(){
typedef
typename boost::mpl::if_c<
tf,
defined,
undefined
>::type type;
return type::invoke();
}
};
public:
extended_type_info_no_rtti() :
no_rtti_system::extended_type_info_no_rtti_0(get_key())
{
key_register();
}
~extended_type_info_no_rtti() BOOST_OVERRIDE {
key_unregister();
}
const extended_type_info *
get_derived_extended_type_info(const T & t) const {
BOOST_STATIC_WARNING(boost::is_polymorphic< T >::value);
const char * derived_key = t.get_key();
BOOST_ASSERT(NULL != derived_key);
return boost::serialization::extended_type_info::find(derived_key);
}
const char * get_key() const{
return action<guid_defined< T >::value >::invoke();
}
const char * get_debug_info() const BOOST_OVERRIDE {
return action<guid_defined< T >::value >::invoke();
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
boost::serialization::extended_type_info_no_rtti< T > type;
};
} 
} 
#endif

#ifdef BOOST_MSVC
#  pragma warning(pop)
#endif
#include <boost/config/abi_suffix.hpp> 

#endif 

#ifndef BOOST_SERIALIZATION_EXTENDED_TYPE_INFO_HPP
#define BOOST_SERIALIZATION_EXTENDED_TYPE_INFO_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <cstdarg>
#include <boost/assert.hpp>
#include <cstddef> 
#include <boost/config.hpp>
#include <boost/noncopyable.hpp>
#include <boost/mpl/bool.hpp>

#include <boost/serialization/config.hpp>
#include <boost/config/abi_prefix.hpp> 
#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4251 4231 4660 4275)
#endif

#define BOOST_SERIALIZATION_MAX_KEY_SIZE 128

namespace boost {
namespace serialization {

namespace void_cast_detail{
class void_caster;
}

class BOOST_SYMBOL_VISIBLE extended_type_info :
private boost::noncopyable
{
private:
friend class boost::serialization::void_cast_detail::void_caster;

const unsigned int m_type_info_key;
virtual bool is_less_than(const extended_type_info & ) const = 0;
virtual bool is_equal(const extended_type_info & ) const = 0;
const char * m_key;

protected:
BOOST_SERIALIZATION_DECL void key_unregister() const;
BOOST_SERIALIZATION_DECL void key_register() const;
BOOST_SERIALIZATION_DECL extended_type_info(
const unsigned int type_info_key,
const char * key
);
virtual BOOST_SERIALIZATION_DECL ~extended_type_info();
public:
const char * get_key() const {
return m_key;
}
virtual const char * get_debug_info() const = 0;
BOOST_SERIALIZATION_DECL bool operator<(const extended_type_info &rhs) const;
BOOST_SERIALIZATION_DECL bool operator==(const extended_type_info &rhs) const;
bool operator!=(const extended_type_info &rhs) const {
return !(operator==(rhs));
}
static BOOST_SERIALIZATION_DECL const extended_type_info *
find(const char *key);
virtual void * construct(unsigned int  = 0, ...) const = 0;
virtual void destroy(void const * const ) const = 0;
};

template<class T>
struct guid_defined : boost::mpl::false_ {};

namespace ext {
template <typename T>
struct guid_impl
{
static inline const char * call()
{
return NULL;
}
};
}

template<class T>
inline const char * guid(){
return ext::guid_impl<T>::call();
}

} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#include <boost/config/abi_suffix.hpp> 

#endif 

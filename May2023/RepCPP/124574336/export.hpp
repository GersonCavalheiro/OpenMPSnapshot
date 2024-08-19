#ifndef BOOST_SERIALIZATION_EXPORT_HPP
#define BOOST_SERIALIZATION_EXPORT_HPP

#if defined(_MSC_VER)
# pragma once
#endif





#include <utility>
#include <cstddef> 

#include <boost/config.hpp>
#include <boost/static_assert.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/type_traits/is_polymorphic.hpp>

#include <boost/mpl/assert.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/not.hpp>
#include <boost/mpl/bool_fwd.hpp>

#include <boost/serialization/extended_type_info.hpp> 
#include <boost/serialization/static_warning.hpp>
#include <boost/serialization/assume_abstract.hpp>
#include <boost/serialization/force_include.hpp>
#include <boost/serialization/singleton.hpp>

#include <boost/archive/detail/register_archive.hpp>

namespace boost {
namespace archive {
namespace detail {

class basic_pointer_iserializer;
class basic_pointer_oserializer;

template<class Archive, class T>
class pointer_iserializer;
template<class Archive, class T>
class pointer_oserializer;

template <class Archive, class Serializable>
struct export_impl
{
static const basic_pointer_iserializer &
enable_load(mpl::true_){
return boost::serialization::singleton<
pointer_iserializer<Archive, Serializable>
>::get_const_instance();
}

static const basic_pointer_oserializer &
enable_save(mpl::true_){
return boost::serialization::singleton<
pointer_oserializer<Archive, Serializable>
>::get_const_instance();
}
inline static void enable_load(mpl::false_) {}
inline static void enable_save(mpl::false_) {}
};

template <void(*)()>
struct instantiate_function {};

template <class Archive, class Serializable>
struct ptr_serialization_support
{
# if defined(BOOST_MSVC) || defined(__SUNPRO_CC)
virtual BOOST_DLLEXPORT void instantiate() BOOST_USED;
# else
static BOOST_DLLEXPORT void instantiate() BOOST_USED;
typedef instantiate_function<
&ptr_serialization_support::instantiate
> x;
# endif
};

template <class Archive, class Serializable>
BOOST_DLLEXPORT void
ptr_serialization_support<Archive,Serializable>::instantiate()
{
export_impl<Archive,Serializable>::enable_save(
typename Archive::is_saving()
);

export_impl<Archive,Serializable>::enable_load(
typename Archive::is_loading()
);
}


namespace extra_detail {

template<class T>
struct guid_initializer
{
void export_guid(mpl::false_) const {
instantiate_ptr_serialization((T*)0, 0, adl_tag());
}
void export_guid(mpl::true_) const {
}
guid_initializer const & export_guid() const {
BOOST_STATIC_WARNING(boost::is_polymorphic< T >::value);
export_guid(boost::serialization::is_abstract< T >());
return *this;
}
};

template<typename T>
struct init_guid;

} 
} 
} 
} 

#define BOOST_CLASS_EXPORT_IMPLEMENT(T)                      \
namespace boost {                                        \
namespace archive {                                      \
namespace detail {                                       \
namespace extra_detail {                                 \
template<>                                               \
struct init_guid< T > {                                  \
static guid_initializer< T > const & g;              \
};                                                       \
guid_initializer< T > const & init_guid< T >::g =        \
::boost::serialization::singleton<                   \
guid_initializer< T >                            \
>::get_mutable_instance().export_guid();             \
}}}}                                                     \


#define BOOST_CLASS_EXPORT_KEY2(T, K)          \
namespace boost {                              \
namespace serialization {                      \
template<>                                     \
struct guid_defined< T > : boost::mpl::true_ {}; \
template<>                                     \
inline const char * guid< T >(){                 \
return K;                                  \
}                                              \
}                           \
}                                   \


#define BOOST_CLASS_EXPORT_KEY(T)                                      \
BOOST_CLASS_EXPORT_KEY2(T, BOOST_PP_STRINGIZE(T))                                                                  \


#define BOOST_CLASS_EXPORT_GUID(T, K)                                  \
BOOST_CLASS_EXPORT_KEY2(T, K)                                          \
BOOST_CLASS_EXPORT_IMPLEMENT(T)                                        \


#if BOOST_WORKAROUND(__MWERKS__, BOOST_TESTED_AT(0x3205))

# define BOOST_SERIALIZATION_MWERKS_BASE_AND_DERIVED(Base,Derived)             \
namespace {                                                                    \
static int BOOST_PP_CAT(boost_serialization_mwerks_init_, __LINE__) =        \
(::boost::archive::detail::instantiate_ptr_serialization((Derived*)0,0), 3); \
static int BOOST_PP_CAT(boost_serialization_mwerks_init2_, __LINE__) = (     \
::boost::serialization::void_cast_register((Derived*)0,(Base*)0)         \
, 3);                                                                      \
}

#else

# define BOOST_SERIALIZATION_MWERKS_BASE_AND_DERIVED(Base,Derived)

#endif

#define BOOST_CLASS_EXPORT_CHECK(T)                              \
BOOST_STATIC_WARNING(                                        \
boost::is_polymorphic<U>::value                          \
);                                                           \


#define BOOST_CLASS_EXPORT(T)                   \
BOOST_CLASS_EXPORT_GUID(                    \
T,                                      \
BOOST_PP_STRINGIZE(T)                   \
)                                           \


#endif 


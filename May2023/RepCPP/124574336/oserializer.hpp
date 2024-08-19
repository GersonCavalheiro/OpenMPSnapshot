#ifndef BOOST_ARCHIVE_OSERIALIZER_HPP
#define BOOST_ARCHIVE_OSERIALIZER_HPP

#if defined(_MSC_VER)
# pragma once
#pragma inline_depth(255)
#pragma inline_recursion(on)
#endif

#if defined(__MWERKS__)
#pragma inline_depth(255)
#endif




#include <boost/assert.hpp>
#include <cstddef> 

#include <boost/config.hpp>

#include <boost/static_assert.hpp>
#include <boost/detail/workaround.hpp>

#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/equal_to.hpp>
#include <boost/mpl/greater_equal.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/mpl/bool_fwd.hpp>

#ifndef BOOST_SERIALIZATION_DEFAULT_TYPE_INFO
#include <boost/serialization/extended_type_info_typeid.hpp>
#endif
#include <boost/serialization/throw_exception.hpp>
#include <boost/serialization/smart_cast.hpp>
#include <boost/serialization/assume_abstract.hpp>
#include <boost/serialization/static_warning.hpp>

#include <boost/type_traits/is_pointer.hpp>
#include <boost/type_traits/is_enum.hpp>
#include <boost/type_traits/is_const.hpp>
#include <boost/type_traits/is_polymorphic.hpp>
#include <boost/type_traits/remove_extent.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/level.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/serialization/type_info_implementation.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/void_cast.hpp>
#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/array_wrapper.hpp>

#include <boost/serialization/singleton.hpp>

#include <boost/archive/archive_exception.hpp>
#include <boost/archive/detail/basic_oarchive.hpp>
#include <boost/archive/detail/basic_oserializer.hpp>
#include <boost/archive/detail/basic_pointer_oserializer.hpp>
#include <boost/archive/detail/archive_serializer_map.hpp>
#include <boost/archive/detail/check.hpp>

#include <boost/core/addressof.hpp>

namespace boost {

namespace serialization {
class extended_type_info;
} 

namespace archive {

class save_access {
public:
template<class Archive>
static void end_preamble(Archive & ar){
ar.end_preamble();
}
template<class Archive, class T>
static void save_primitive(Archive & ar, const  T & t){
ar.end_preamble();
ar.save(t);
}
};

namespace detail {

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif

template<class Archive, class T>
class oserializer : public basic_oserializer
{
private:
public:
explicit BOOST_DLLEXPORT oserializer() :
basic_oserializer(
boost::serialization::singleton<
typename
boost::serialization::type_info_implementation< T >::type
>::get_const_instance()
)
{}
BOOST_DLLEXPORT void save_object_data(
basic_oarchive & ar,
const void *x
) const BOOST_OVERRIDE BOOST_USED;
bool class_info() const BOOST_OVERRIDE {
return boost::serialization::implementation_level< T >::value
>= boost::serialization::object_class_info;
}
bool tracking(const unsigned int ) const BOOST_OVERRIDE {
return boost::serialization::tracking_level< T >::value == boost::serialization::track_always
|| (boost::serialization::tracking_level< T >::value == boost::serialization::track_selectively
&& serialized_as_pointer());
}
version_type version() const BOOST_OVERRIDE {
return version_type(::boost::serialization::version< T >::value);
}
bool is_polymorphic() const BOOST_OVERRIDE {
return boost::is_polymorphic< T >::value;
}
~oserializer() BOOST_OVERRIDE {}
};

#ifdef BOOST_MSVC
#  pragma warning(pop)
#endif

template<class Archive, class T>
BOOST_DLLEXPORT void oserializer<Archive, T>::save_object_data(
basic_oarchive & ar,
const void *x
) const {
BOOST_STATIC_ASSERT(boost::is_const< T >::value == false);
boost::serialization::serialize_adl(
boost::serialization::smart_cast_reference<Archive &>(ar),
* static_cast<T *>(const_cast<void *>(x)),
version()
);
}

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif

template<class Archive, class T>
class pointer_oserializer :
public basic_pointer_oserializer
{
private:
const basic_oserializer &
get_basic_serializer() const BOOST_OVERRIDE {
return boost::serialization::singleton<
oserializer<Archive, T>
>::get_const_instance();
}
BOOST_DLLEXPORT void save_object_ptr(
basic_oarchive & ar,
const void * x
) const BOOST_OVERRIDE BOOST_USED;
public:
pointer_oserializer();
~pointer_oserializer() BOOST_OVERRIDE;
};

#ifdef BOOST_MSVC
#  pragma warning(pop)
#endif

template<class Archive, class T>
BOOST_DLLEXPORT void pointer_oserializer<Archive, T>::save_object_ptr(
basic_oarchive & ar,
const void * x
) const {
BOOST_ASSERT(NULL != x);
T * t = static_cast<T *>(const_cast<void *>(x));
const unsigned int file_version = boost::serialization::version< T >::value;
Archive & ar_impl
= boost::serialization::smart_cast_reference<Archive &>(ar);
boost::serialization::save_construct_data_adl<Archive, T>(
ar_impl,
t,
file_version
);
ar_impl << boost::serialization::make_nvp(NULL, * t);
}

template<class Archive, class T>
pointer_oserializer<Archive, T>::pointer_oserializer() :
basic_pointer_oserializer(
boost::serialization::singleton<
typename
boost::serialization::type_info_implementation< T >::type
>::get_const_instance()
)
{
boost::serialization::singleton<
oserializer<Archive, T>
>::get_mutable_instance().set_bpos(this);
archive_serializer_map<Archive>::insert(this);
}

template<class Archive, class T>
pointer_oserializer<Archive, T>::~pointer_oserializer(){
archive_serializer_map<Archive>::erase(this);
}

template<class Archive>
struct save_non_pointer_type {
struct save_primitive {
template<class T>
static void invoke(Archive & ar, const T & t){
save_access::save_primitive(ar, t);
}
};
struct save_only {
template<class T>
static void invoke(Archive & ar, const T & t){
boost::serialization::serialize_adl(
ar,
const_cast<T &>(t),
::boost::serialization::version< T >::value
);
}
};
struct save_standard {
template<class T>
static void invoke(Archive &ar, const T & t){
ar.save_object(
boost::addressof(t),
boost::serialization::singleton<
oserializer<Archive, T>
>::get_const_instance()
);
}
};



struct save_conditional {
template<class T>
static void invoke(Archive &ar, const T &t){
save_standard::invoke(ar, t);
}
};


template<class T>
static void invoke(Archive & ar, const T & t){
typedef
typename mpl::eval_if<
mpl::equal_to<
boost::serialization::implementation_level< T >,
mpl::int_<boost::serialization::primitive_type>
>,
mpl::identity<save_primitive>,
typename mpl::eval_if<
mpl::greater_equal<
boost::serialization::implementation_level< T >,
mpl::int_<boost::serialization::object_class_info>
>,
mpl::identity<save_standard>,
typename mpl::eval_if<
mpl::equal_to<
boost::serialization::tracking_level< T >,
mpl::int_<boost::serialization::track_never>
>,
mpl::identity<save_only>,
mpl::identity<save_conditional>
> > >::type typex;
check_object_versioning< T >();
typex::invoke(ar, t);
}
template<class T>
static void invoke(Archive & ar, T & t){
check_object_level< T >();
check_object_tracking< T >();
invoke(ar, const_cast<const T &>(t));
}
};

template<class Archive>
struct save_pointer_type {
struct abstract
{
template<class T>
static const basic_pointer_oserializer * register_type(Archive & ){
BOOST_STATIC_ASSERT(boost::is_polymorphic< T >::value);
return NULL;
}
};

struct non_abstract
{
template<class T>
static const basic_pointer_oserializer * register_type(Archive & ar){
return ar.register_type(static_cast<T *>(NULL));
}
};

template<class T>
static const basic_pointer_oserializer * register_type(Archive &ar, T* const ){
typedef
typename mpl::eval_if<
boost::serialization::is_abstract< T >,
mpl::identity<abstract>,
mpl::identity<non_abstract>
>::type typex;
return typex::template register_type< T >(ar);
}

struct non_polymorphic
{
template<class T>
static void save(
Archive &ar,
T & t
){
const basic_pointer_oserializer & bpos =
boost::serialization::singleton<
pointer_oserializer<Archive, T>
>::get_const_instance();
ar.save_pointer(& t, & bpos);
}
};

struct polymorphic
{
template<class T>
static void save(
Archive &ar,
T & t
){
typename
boost::serialization::type_info_implementation< T >::type const
& i = boost::serialization::singleton<
typename
boost::serialization::type_info_implementation< T >::type
>::get_const_instance();

boost::serialization::extended_type_info const * const this_type = & i;

BOOST_ASSERT(NULL != this_type);

const boost::serialization::extended_type_info * true_type =
i.get_derived_extended_type_info(t);

if(NULL == true_type){
boost::serialization::throw_exception(
archive_exception(
archive_exception::unregistered_class,
"derived class not registered or exported"
)
);
}

const void *vp = static_cast<const void *>(&t);
if(*this_type == *true_type){
const basic_pointer_oserializer * bpos = register_type(ar, &t);
ar.save_pointer(vp, bpos);
return;
}
vp = serialization::void_downcast(
*true_type,
*this_type,
static_cast<const void *>(&t)
);
if(NULL == vp){
boost::serialization::throw_exception(
archive_exception(
archive_exception::unregistered_cast,
true_type->get_debug_info(),
this_type->get_debug_info()
)
);
}

const basic_pointer_oserializer * bpos
= static_cast<const basic_pointer_oserializer *>(
boost::serialization::singleton<
archive_serializer_map<Archive>
>::get_const_instance().find(*true_type)
);
BOOST_ASSERT(NULL != bpos);
if(NULL == bpos)
boost::serialization::throw_exception(
archive_exception(
archive_exception::unregistered_class,
"derived class not registered or exported"
)
);
ar.save_pointer(vp, bpos);
}
};

template<class T>
static void save(
Archive & ar,
const T & t
){
check_pointer_level< T >();
check_pointer_tracking< T >();
typedef typename mpl::eval_if<
is_polymorphic< T >,
mpl::identity<polymorphic>,
mpl::identity<non_polymorphic>
>::type type;
type::save(ar, const_cast<T &>(t));
}

template<class TPtr>
static void invoke(Archive &ar, const TPtr t){
register_type(ar, t);
if(NULL == t){
basic_oarchive & boa
= boost::serialization::smart_cast_reference<basic_oarchive &>(ar);
boa.save_null_pointer();
save_access::end_preamble(ar);
return;
}
save(ar, * t);
}
};

template<class Archive>
struct save_enum_type
{
template<class T>
static void invoke(Archive &ar, const T &t){
const int i = static_cast<int>(t);
ar << boost::serialization::make_nvp(NULL, i);
}
};

template<class Archive>
struct save_array_type
{
template<class T>
static void invoke(Archive &ar, const T &t){
typedef typename boost::remove_extent< T >::type value_type;

save_access::end_preamble(ar);
std::size_t c = sizeof(t) / (
static_cast<const char *>(static_cast<const void *>(&t[1]))
- static_cast<const char *>(static_cast<const void *>(&t[0]))
);
boost::serialization::collection_size_type count(c);
ar << BOOST_SERIALIZATION_NVP(count);
ar << serialization::make_array<
const value_type,
boost::serialization::collection_size_type
>(
static_cast<const value_type *>(&t[0]),
count
);
}
};

} 

template<class Archive, class T>
inline void save(Archive & ar,  T &t){
typedef
typename mpl::eval_if<is_pointer< T >,
mpl::identity<detail::save_pointer_type<Archive> >,
typename mpl::eval_if<is_enum< T >,
mpl::identity<detail::save_enum_type<Archive> >,
typename mpl::eval_if<is_array< T >,
mpl::identity<detail::save_array_type<Archive> >,
mpl::identity<detail::save_non_pointer_type<Archive> >
>
>
>::type typex;
typex::invoke(ar, t);
}

} 
} 

#endif 

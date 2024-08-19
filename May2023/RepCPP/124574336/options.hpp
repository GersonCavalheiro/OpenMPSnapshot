
#ifndef BOOST_INTRUSIVE_OPTIONS_HPP
#define BOOST_INTRUSIVE_OPTIONS_HPP

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/intrusive_fwd.hpp>
#include <boost/intrusive/link_mode.hpp>
#include <boost/intrusive/pack_options.hpp>

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace intrusive {

#ifndef BOOST_INTRUSIVE_DOXYGEN_INVOKED

struct empty
{};

template<class Functor>
struct fhtraits;

template<class T, class Hook, Hook T::* P>
struct mhtraits;

struct dft_tag;
struct member_tag;

template<class SupposedValueTraits>
struct is_default_hook_tag;

#endif   

BOOST_INTRUSIVE_OPTION_CONSTANT(constant_time_size, bool, Enabled, constant_time_size)

BOOST_INTRUSIVE_OPTION_TYPE(header_holder_type, HeaderHolder, HeaderHolder, header_holder_type)

BOOST_INTRUSIVE_OPTION_TYPE(size_type, SizeType, SizeType, size_type)

BOOST_INTRUSIVE_OPTION_TYPE(compare, Compare, Compare, compare)

BOOST_INTRUSIVE_OPTION_TYPE(key_of_value, KeyOfValue, KeyOfValue, key_of_value)

BOOST_INTRUSIVE_OPTION_TYPE(priority_of_value, PrioOfValue, PrioOfValue, priority_of_value)

BOOST_INTRUSIVE_OPTION_CONSTANT(floating_point, bool, Enabled, floating_point)

BOOST_INTRUSIVE_OPTION_TYPE(equal, Equal, Equal, equal)

BOOST_INTRUSIVE_OPTION_TYPE(priority, Priority, Priority, priority)

BOOST_INTRUSIVE_OPTION_TYPE(hash, Hash, Hash, hash)

BOOST_INTRUSIVE_OPTION_TYPE(value_traits, ValueTraits, ValueTraits, proto_value_traits)


template< typename Parent
, typename MemberHook
, MemberHook Parent::* PtrToMember>
struct member_hook
{
typedef mhtraits <Parent, MemberHook, PtrToMember> member_value_traits;
template<class Base>
struct pack : Base
{
typedef member_value_traits proto_value_traits;
};
};

BOOST_INTRUSIVE_OPTION_TYPE(function_hook, Functor, fhtraits<Functor>, proto_value_traits)

BOOST_INTRUSIVE_OPTION_TYPE(base_hook, BaseHook, BaseHook, proto_value_traits)

BOOST_INTRUSIVE_OPTION_TYPE(void_pointer, VoidPointer, VoidPointer, void_pointer)

BOOST_INTRUSIVE_OPTION_TYPE(tag, Tag, Tag, tag)

BOOST_INTRUSIVE_OPTION_CONSTANT(link_mode, link_mode_type, LinkType, link_mode)

BOOST_INTRUSIVE_OPTION_CONSTANT(optimize_size, bool, Enabled, optimize_size)

BOOST_INTRUSIVE_OPTION_CONSTANT(linear, bool, Enabled, linear)

BOOST_INTRUSIVE_OPTION_CONSTANT(cache_last, bool, Enabled, cache_last)

BOOST_INTRUSIVE_OPTION_TYPE(bucket_traits, BucketTraits, BucketTraits, bucket_traits)

BOOST_INTRUSIVE_OPTION_CONSTANT(store_hash, bool, Enabled, store_hash)

BOOST_INTRUSIVE_OPTION_CONSTANT(optimize_multikey, bool, Enabled, optimize_multikey)

BOOST_INTRUSIVE_OPTION_CONSTANT(power_2_buckets, bool, Enabled, power_2_buckets)

BOOST_INTRUSIVE_OPTION_CONSTANT(cache_begin, bool, Enabled, cache_begin)

BOOST_INTRUSIVE_OPTION_CONSTANT(compare_hash, bool, Enabled, compare_hash)

BOOST_INTRUSIVE_OPTION_CONSTANT(incremental, bool, Enabled, incremental)


struct hook_defaults
{
typedef void* void_pointer;
static const link_mode_type link_mode = safe_link;
typedef dft_tag tag;
static const bool optimize_size = false;
static const bool store_hash = false;
static const bool linear = false;
static const bool optimize_multikey = false;
};


}  
}  

#include <boost/intrusive/detail/config_end.hpp>

#endif   

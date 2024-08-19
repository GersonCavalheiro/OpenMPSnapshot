
#ifndef BOOST_INTRUSIVE_FWD_HPP
#define BOOST_INTRUSIVE_FWD_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#ifndef BOOST_CSTDINT_HPP
#  include <boost/cstdint.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif


#if !defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)

#include <cstddef>
#include <boost/intrusive/link_mode.hpp>
#include <boost/intrusive/detail/workaround.hpp>

namespace boost {
namespace intrusive {

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
#  ifdef BOOST_HAS_INTPTR_T
using ::boost::uintptr_t;
#  else
typedef std::size_t uintptr_t;
#  endif
#endif


template<class NodeTraits>
class circular_list_algorithms;

template<class NodeTraits>
class circular_slist_algorithms;

template<class NodeTraits>
class linear_slist_algorithms;

template<class NodeTraits>
class bstree_algorithms;

template<class NodeTraits>
class rbtree_algorithms;

template<class NodeTraits>
class avltree_algorithms;

template<class NodeTraits>
class sgtree_algorithms;

template<class NodeTraits>
class splaytree_algorithms;

template<class NodeTraits>
class treap_algorithms;


#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class T
, class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
, class O5  = void
, class O6  = void
>
#else
template<class T, class ...Options>
#endif
class slist;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class O1  = void
, class O2  = void
, class O3  = void
>
#else
template<class ...Options>
#endif
class slist_base_hook;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class O1  = void
, class O2  = void
, class O3  = void
>
#else
template<class ...Options>
#endif
class slist_member_hook;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class T
, class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
>
#else
template<class T, class ...Options>
#endif
class list;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class O1  = void
, class O2  = void
, class O3  = void
>
#else
template<class ...Options>
#endif
class list_base_hook;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class O1  = void
, class O2  = void
, class O3  = void
>
#else
template<class ...Options>
#endif
class list_member_hook;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class T
, class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
, class O5  = void
, class O6  = void
>
#else
template<class T, class ...Options>
#endif
class rbtree;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class T
, class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
, class O5  = void
, class O6  = void
>
#else
template<class T, class ...Options>
#endif
class set;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class T
, class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
, class O5  = void
, class O6  = void
>
#else
template<class T, class ...Options>
#endif
class multiset;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
>
#else
template<class ...Options>
#endif
class set_base_hook;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
>
#else
template<class ...Options>
#endif
class set_member_hook;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class T
, class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
, class O5  = void
, class O6  = void
>
#else
template<class T, class ...Options>
#endif
class splaytree;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class T
, class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
, class O5  = void
, class O6  = void
>
#else
template<class T, class ...Options>
#endif
class splay_set;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class T
, class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
, class O5  = void
, class O6  = void
>
#else
template<class T, class ...Options>
#endif
class splay_multiset;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class T
, class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
, class O5  = void
, class O6  = void
>
#else
template<class T, class ...Options>
#endif
class avltree;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class T
, class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
, class O5  = void
, class O6  = void
>
#else
template<class T, class ...Options>
#endif
class avl_set;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class T
, class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
, class O5  = void
, class O6  = void
>
#else
template<class T, class ...Options>
#endif
class avl_multiset;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
>
#else
template<class ...Options>
#endif
class avl_set_base_hook;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
>
#else
template<class ...Options>
#endif
class avl_set_member_hook;


#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class T
, class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
, class O5  = void
, class O6  = void
, class O7  = void
>
#else
template<class T, class ...Options>
#endif
class treap;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class T
, class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
, class O5  = void
, class O6  = void
, class O7  = void
>
#else
template<class T, class ...Options>
#endif
class treap_set;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class T
, class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
, class O5  = void
, class O6  = void
, class O7  = void
>
#else
template<class T, class ...Options>
#endif
class treap_multiset;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class T
, class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
, class O5  = void
, class O6  = void
>
#else
template<class T, class ...Options>
#endif
class sgtree;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class T
, class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
, class O5  = void
, class O6  = void
>
#else
template<class T, class ...Options>
#endif
class sg_set;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class T
, class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
, class O5  = void
, class O6  = void
>
#else
template<class T, class ...Options>
#endif
class sg_multiset;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class T
, class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
, class O5  = void
, class O6  = void
>
#else
template<class T, class ...Options>
#endif
class bstree;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class T
, class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
, class O5  = void
, class O6  = void
>
#else
template<class T, class ...Options>
#endif
class bs_set;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class T
, class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
, class O5  = void
, class O6  = void
>
#else
template<class T, class ...Options>
#endif
class bs_multiset;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class O1  = void
, class O2  = void
, class O3  = void
>
#else
template<class ...Options>
#endif
class bs_set_base_hook;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class O1  = void
, class O2  = void
, class O3  = void
>
#else
template<class ...Options>
#endif
class bs_set_member_hook;


#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class T
, class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
, class O5  = void
, class O6  = void
, class O7  = void
, class O8  = void
, class O9  = void
, class O10 = void
>
#else
template<class T, class ...Options>
#endif
class hashtable;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class T
, class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
, class O5  = void
, class O6  = void
, class O7  = void
, class O8  = void
, class O9  = void
, class O10 = void
>
#else
template<class T, class ...Options>
#endif
class unordered_set;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class T
, class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
, class O5  = void
, class O6  = void
, class O7  = void
, class O8  = void
, class O9  = void
, class O10 = void
>
#else
template<class T, class ...Options>
#endif
class unordered_multiset;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
>
#else
template<class ...Options>
#endif
class unordered_set_base_hook;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class O1  = void
, class O2  = void
, class O3  = void
, class O4  = void
>
#else
template<class ...Options>
#endif
class unordered_set_member_hook;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class O1  = void
, class O2  = void
, class O3  = void
>
#else
template<class ...Options>
#endif
class any_base_hook;

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template
< class O1  = void
, class O2  = void
, class O3  = void
>
#else
template<class ...Options>
#endif
class any_member_hook;


template<bool Enabled>
struct constant_time_size;

template<typename SizeType>
struct size_type;

template<typename Compare>
struct compare;

template<bool Enabled>
struct floating_point;

template<typename Equal>
struct equal;

template<typename Priority>
struct priority;

template<typename Hash>
struct hash;

template<typename ValueTraits> struct value_traits;

template< typename Parent
, typename MemberHook
, MemberHook Parent::* PtrToMember>
struct member_hook;

template<typename Functor>
struct function_hook;

template<typename BaseHook>
struct base_hook;

template<typename VoidPointer>
struct void_pointer;

template<typename Tag>
struct tag;

template<link_mode_type LinkType>
struct link_mode;

template<bool Enabled> struct
optimize_size;

template<bool Enabled>
struct linear;

template<bool Enabled>
struct cache_last;

template<typename BucketTraits>
struct bucket_traits;

template<bool Enabled>
struct store_hash;

template<bool Enabled>
struct optimize_multikey;

template<bool Enabled>
struct power_2_buckets;

template<bool Enabled>
struct cache_begin;

template<bool Enabled>
struct compare_hash;

template<bool Enabled>
struct incremental;


template<typename ValueTraits>
struct value_traits;

template< typename Parent
, typename MemberHook
, MemberHook Parent::* PtrToMember>
struct member_hook;

template< typename Functor>
struct function_hook;

template<typename BaseHook>
struct base_hook;

template<class T, class NodeTraits, link_mode_type LinkMode = safe_link>
struct derivation_value_traits;

template<class NodeTraits, link_mode_type LinkMode = normal_link>
struct trivial_value_traits;


template<typename VoidPointer, std::size_t Alignment>
struct max_pointer_plus_bits;

template<std::size_t Alignment>
struct max_pointer_plus_bits<void *, Alignment>;

template<typename Pointer, std::size_t NumBits>
struct pointer_plus_bits;

template<typename T, std::size_t NumBits>
struct pointer_plus_bits<T *, NumBits>;

template<typename Ptr>
struct pointer_traits;

template<typename T>
struct pointer_traits<T *>;

}  
}  

#endif   

#endif   

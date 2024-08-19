
#ifndef BOOST_INTRUSIVE_UNORDERED_SET_HOOK_HPP
#define BOOST_INTRUSIVE_UNORDERED_SET_HOOK_HPP

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/intrusive_fwd.hpp>

#include <boost/intrusive/pointer_traits.hpp>
#include <boost/intrusive/slist_hook.hpp>
#include <boost/intrusive/options.hpp>
#include <boost/intrusive/detail/generic_hook.hpp>

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace intrusive {


template<class VoidPointer, bool StoreHash, bool OptimizeMultiKey>
struct unordered_node
:  public slist_node<VoidPointer>
{
typedef typename pointer_traits
<VoidPointer>::template rebind_pointer
< unordered_node<VoidPointer, StoreHash, OptimizeMultiKey> >::type
node_ptr;
node_ptr    prev_in_group_;
std::size_t hash_;
};

template<class VoidPointer>
struct unordered_node<VoidPointer, false, true>
:  public slist_node<VoidPointer>
{
typedef typename pointer_traits
<VoidPointer>::template rebind_pointer
< unordered_node<VoidPointer, false, true> >::type
node_ptr;
node_ptr    prev_in_group_;
};

template<class VoidPointer>
struct unordered_node<VoidPointer, true, false>
:  public slist_node<VoidPointer>
{
typedef typename pointer_traits
<VoidPointer>::template rebind_pointer
< unordered_node<VoidPointer, true, false> >::type
node_ptr;
std::size_t hash_;
};

template<class VoidPointer, bool StoreHash, bool OptimizeMultiKey>
struct unordered_node_traits
:  public slist_node_traits<VoidPointer>
{
typedef slist_node_traits<VoidPointer> reduced_slist_node_traits;
typedef unordered_node<VoidPointer, StoreHash, OptimizeMultiKey> node;

typedef typename pointer_traits
<VoidPointer>::template rebind_pointer
< node >::type node_ptr;
typedef typename pointer_traits
<VoidPointer>::template rebind_pointer
< const node >::type const_node_ptr;

static const bool store_hash        = StoreHash;
static const bool optimize_multikey = OptimizeMultiKey;

BOOST_INTRUSIVE_FORCEINLINE static node_ptr get_next(const const_node_ptr & n)
{  return pointer_traits<node_ptr>::static_cast_from(n->next_);  }

BOOST_INTRUSIVE_FORCEINLINE static void set_next(node_ptr n, node_ptr next)
{  n->next_ = next;  }

BOOST_INTRUSIVE_FORCEINLINE static node_ptr get_prev_in_group(const const_node_ptr & n)
{  return n->prev_in_group_;  }

BOOST_INTRUSIVE_FORCEINLINE static void set_prev_in_group(node_ptr n, node_ptr prev)
{  n->prev_in_group_ = prev;  }

BOOST_INTRUSIVE_FORCEINLINE static std::size_t get_hash(const const_node_ptr & n)
{  return n->hash_;  }

BOOST_INTRUSIVE_FORCEINLINE static void set_hash(const node_ptr & n, std::size_t h)
{  n->hash_ = h;  }
};

template<class NodeTraits>
struct unordered_group_adapter
{
typedef typename NodeTraits::node            node;
typedef typename NodeTraits::node_ptr        node_ptr;
typedef typename NodeTraits::const_node_ptr  const_node_ptr;

static node_ptr get_next(const const_node_ptr & n)
{  return NodeTraits::get_prev_in_group(n);  }

static void set_next(node_ptr n, node_ptr next)
{  NodeTraits::set_prev_in_group(n, next);   }
};

template<class NodeTraits>
struct unordered_algorithms
: public circular_slist_algorithms<NodeTraits>
{
typedef circular_slist_algorithms<NodeTraits>   base_type;
typedef unordered_group_adapter<NodeTraits>     group_traits;
typedef circular_slist_algorithms<group_traits> group_algorithms;
typedef NodeTraits                              node_traits;
typedef typename NodeTraits::node               node;
typedef typename NodeTraits::node_ptr           node_ptr;
typedef typename NodeTraits::const_node_ptr     const_node_ptr;

BOOST_INTRUSIVE_FORCEINLINE static void init(typename base_type::node_ptr n)
{
base_type::init(n);
group_algorithms::init(n);
}

BOOST_INTRUSIVE_FORCEINLINE static void init_header(typename base_type::node_ptr n)
{
base_type::init_header(n);
group_algorithms::init_header(n);
}

BOOST_INTRUSIVE_FORCEINLINE static void unlink(typename base_type::node_ptr n)
{
base_type::unlink(n);
group_algorithms::unlink(n);
}
};

template<class Algo>
struct uset_algo_wrapper : public Algo
{};

template<class VoidPointer, bool StoreHash, bool OptimizeMultiKey>
struct get_uset_node_traits
{
typedef typename detail::if_c
< (StoreHash || OptimizeMultiKey)
, unordered_node_traits<VoidPointer, StoreHash, OptimizeMultiKey>
, slist_node_traits<VoidPointer>
>::type type;
};

template<bool OptimizeMultiKey>
struct get_uset_algo_type
{
static const algo_types value = OptimizeMultiKey ? UnorderedAlgorithms : UnorderedCircularSlistAlgorithms;
};

template<class NodeTraits>
struct get_algo<UnorderedAlgorithms, NodeTraits>
{
typedef unordered_algorithms<NodeTraits> type;
};

template<class NodeTraits>
struct get_algo<UnorderedCircularSlistAlgorithms, NodeTraits>
{
typedef uset_algo_wrapper< circular_slist_algorithms<NodeTraits> > type;
};


#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED) || defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template<class ...Options>
#else
template<class O1 = void, class O2 = void, class O3 = void, class O4 = void>
#endif
struct make_unordered_set_base_hook
{
typedef typename pack_options
< hook_defaults,
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3, O4
#else
Options...
#endif
>::type packed_options;

typedef generic_hook
< get_uset_algo_type <packed_options::optimize_multikey>::value
, typename get_uset_node_traits < typename packed_options::void_pointer
, packed_options::store_hash
, packed_options::optimize_multikey
>::type
, typename packed_options::tag
, packed_options::link_mode
, HashBaseHookId
> implementation_defined;
typedef implementation_defined type;
};

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED) || defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template<class ...Options>
#else
template<class O1, class O2, class O3, class O4>
#endif
class unordered_set_base_hook
:  public make_unordered_set_base_hook<
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3, O4
#else
Options...
#endif
>::type
{
#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
public:
unordered_set_base_hook();

unordered_set_base_hook(const unordered_set_base_hook& );

unordered_set_base_hook& operator=(const unordered_set_base_hook& );

~unordered_set_base_hook();

void swap_nodes(unordered_set_base_hook &other);

bool is_linked() const;

void unlink();
#endif
};


#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED) || defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template<class ...Options>
#else
template<class O1 = void, class O2 = void, class O3 = void, class O4 = void>
#endif
struct make_unordered_set_member_hook
{
typedef typename pack_options
< hook_defaults,
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3, O4
#else
Options...
#endif
>::type packed_options;

typedef generic_hook
< get_uset_algo_type <packed_options::optimize_multikey>::value
, typename get_uset_node_traits < typename packed_options::void_pointer
, packed_options::store_hash
, packed_options::optimize_multikey
>::type
, member_tag
, packed_options::link_mode
, NoBaseHookId
> implementation_defined;
typedef implementation_defined type;
};

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED) || defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template<class ...Options>
#else
template<class O1, class O2, class O3, class O4>
#endif
class unordered_set_member_hook
:  public make_unordered_set_member_hook<
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3, O4
#else
Options...
#endif
>::type
{
#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
public:
unordered_set_member_hook();

unordered_set_member_hook(const unordered_set_member_hook& );

unordered_set_member_hook& operator=(const unordered_set_member_hook& );

~unordered_set_member_hook();

void swap_nodes(unordered_set_member_hook &other);

bool is_linked() const;

void unlink();
#endif
};

} 
} 

#include <boost/intrusive/detail/config_end.hpp>

#endif 

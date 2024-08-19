
#ifndef BOOST_INTRUSIVE_AVL_SET_HOOK_HPP
#define BOOST_INTRUSIVE_AVL_SET_HOOK_HPP

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/intrusive_fwd.hpp>

#include <boost/intrusive/detail/avltree_node.hpp>
#include <boost/intrusive/avltree_algorithms.hpp>
#include <boost/intrusive/options.hpp>
#include <boost/intrusive/detail/generic_hook.hpp>

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace intrusive {

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED) || defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template<class ...Options>
#else
template<class O1 = void, class O2 = void, class O3 = void, class O4 = void>
#endif
struct make_avl_set_base_hook
{
typedef typename pack_options
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
<hook_defaults, O1, O2, O3, O4>
#else
<hook_defaults, Options...>
#endif
::type packed_options;

typedef generic_hook
< AvlTreeAlgorithms
, avltree_node_traits<typename packed_options::void_pointer, packed_options::optimize_size>
, typename packed_options::tag
, packed_options::link_mode
, AvlTreeBaseHookId
> implementation_defined;
typedef implementation_defined type;
};

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED) || defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template<class ...Options>
#else
template<class O1, class O2, class O3, class O4>
#endif
class avl_set_base_hook
:  public make_avl_set_base_hook
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
<O1, O2, O3, O4>
#else
<Options...>
#endif
::type
{
#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
public:
avl_set_base_hook();

avl_set_base_hook(const avl_set_base_hook& );

avl_set_base_hook& operator=(const avl_set_base_hook& );

~avl_set_base_hook();

void swap_nodes(avl_set_base_hook &other);

bool is_linked() const;

void unlink();
#endif
};

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED) || defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template<class ...Options>
#else
template<class O1 = void, class O2 = void, class O3 = void, class O4 = void>
#endif
struct make_avl_set_member_hook
{
typedef typename pack_options
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
<hook_defaults, O1, O2, O3, O4>
#else
<hook_defaults, Options...>
#endif
::type packed_options;

typedef generic_hook
< AvlTreeAlgorithms
, avltree_node_traits<typename packed_options::void_pointer, packed_options::optimize_size>
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
class avl_set_member_hook
:  public make_avl_set_member_hook
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
<O1, O2, O3, O4>
#else
<Options...>
#endif
::type
{
#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
public:
avl_set_member_hook();

avl_set_member_hook(const avl_set_member_hook& );

avl_set_member_hook& operator=(const avl_set_member_hook& );

~avl_set_member_hook();

void swap_nodes(avl_set_member_hook &other);

bool is_linked() const;

void unlink();
#endif
};

} 
} 

#include <boost/intrusive/detail/config_end.hpp>

#endif 


#ifndef BOOST_INTRUSIVE_LIST_HOOK_HPP
#define BOOST_INTRUSIVE_LIST_HOOK_HPP

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/intrusive_fwd.hpp>

#include <boost/intrusive/detail/list_node.hpp>
#include <boost/intrusive/circular_list_algorithms.hpp>
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
template<class O1 = void, class O2 = void, class O3 = void>
#endif
struct make_list_base_hook
{
typedef typename pack_options
< hook_defaults,
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3
#else
Options...
#endif
>::type packed_options;

typedef generic_hook
< CircularListAlgorithms
, list_node_traits<typename packed_options::void_pointer>
, typename packed_options::tag
, packed_options::link_mode
, ListBaseHookId
> implementation_defined;
typedef implementation_defined type;
};

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED) || defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template<class ...Options>
#else
template<class O1, class O2, class O3>
#endif
class list_base_hook
:  public make_list_base_hook
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
<O1, O2, O3>
#else
<Options...>
#endif
::type
{
#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
public:
list_base_hook();

list_base_hook(const list_base_hook& );

list_base_hook& operator=(const list_base_hook& );

~list_base_hook();

void swap_nodes(list_base_hook &other);

bool is_linked() const;

void unlink();
#endif
};

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED) || defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template<class ...Options>
#else
template<class O1 = void, class O2 = void, class O3 = void>
#endif
struct make_list_member_hook
{
typedef typename pack_options
< hook_defaults,
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3
#else
Options...
#endif
>::type packed_options;

typedef generic_hook
< CircularListAlgorithms
, list_node_traits<typename packed_options::void_pointer>
, member_tag
, packed_options::link_mode
, NoBaseHookId
> implementation_defined;
typedef implementation_defined type;
};

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED) || defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template<class ...Options>
#else
template<class O1, class O2, class O3>
#endif
class list_member_hook
:  public make_list_member_hook
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
<O1, O2, O3>
#else
<Options...>
#endif
::type
{
#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
public:
list_member_hook();

list_member_hook(const list_member_hook& );

list_member_hook& operator=(const list_member_hook& );

~list_member_hook();

void swap_nodes(list_member_hook &other);

bool is_linked() const;

void unlink();
#endif
};

} 
} 

#include <boost/intrusive/detail/config_end.hpp>

#endif 

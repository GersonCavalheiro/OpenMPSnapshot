
#ifndef BOOST_INTRUSIVE_SLIST_HOOK_HPP
#define BOOST_INTRUSIVE_SLIST_HOOK_HPP

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/intrusive_fwd.hpp>

#include <boost/intrusive/detail/slist_node.hpp>
#include <boost/intrusive/circular_slist_algorithms.hpp>
#include <boost/intrusive/link_mode.hpp>
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
struct make_slist_base_hook
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
< CircularSListAlgorithms
, slist_node_traits<typename packed_options::void_pointer>
, typename packed_options::tag
, packed_options::link_mode
, SlistBaseHookId
> implementation_defined;
typedef implementation_defined type;
};

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED) || defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template<class ...Options>
#else
template<class O1, class O2, class O3>
#endif
class slist_base_hook
:  public make_slist_base_hook<
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3
#else
Options...
#endif
>::type
{
#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
public:
slist_base_hook();

slist_base_hook(const slist_base_hook& );

slist_base_hook& operator=(const slist_base_hook& );

~slist_base_hook();

void swap_nodes(slist_base_hook &other);

bool is_linked() const;

void unlink();
#endif
};

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED) || defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template<class ...Options>
#else
template<class O1 = void, class O2 = void, class O3 = void>
#endif
struct make_slist_member_hook
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
< CircularSListAlgorithms
, slist_node_traits<typename packed_options::void_pointer>
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
class slist_member_hook
:  public make_slist_member_hook<
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3
#else
Options...
#endif
>::type
{
#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
public:
slist_member_hook();

slist_member_hook(const slist_member_hook& );

slist_member_hook& operator=(const slist_member_hook& );

~slist_member_hook();

void swap_nodes(slist_member_hook &other);

bool is_linked() const;

void unlink();
#endif
};

} 
} 

#include <boost/intrusive/detail/config_end.hpp>

#endif 


#ifndef BOOST_INTRUSIVE_ANY_HOOK_HPP
#define BOOST_INTRUSIVE_ANY_HOOK_HPP

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/intrusive_fwd.hpp>
#include <boost/intrusive/detail/any_node_and_algorithms.hpp>
#include <boost/intrusive/options.hpp>
#include <boost/intrusive/detail/generic_hook.hpp>
#include <boost/intrusive/detail/mpl.hpp>
#include <boost/intrusive/pointer_rebind.hpp>

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
struct make_any_base_hook
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
< AnyAlgorithm
, any_node_traits<typename packed_options::void_pointer>
, typename packed_options::tag
, packed_options::link_mode
, AnyBaseHookId
> implementation_defined;
typedef implementation_defined type;
};

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED) || defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template<class ...Options>
#else
template<class O1, class O2, class O3>
#endif
class any_base_hook
:  public make_any_base_hook
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
<O1, O2, O3>
#else
<Options...>
#endif
::type
{
#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
public:
any_base_hook();

any_base_hook(const any_base_hook& );

any_base_hook& operator=(const any_base_hook& );

~any_base_hook();

bool is_linked() const;
#endif
};

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED) || defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template<class ...Options>
#else
template<class O1 = void, class O2 = void, class O3 = void>
#endif
struct make_any_member_hook
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
< AnyAlgorithm
, any_node_traits<typename packed_options::void_pointer>
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
class any_member_hook
:  public make_any_member_hook
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
<O1, O2, O3>
#else
<Options...>
#endif
::type
{
#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
public:
any_member_hook();

any_member_hook(const any_member_hook& );

any_member_hook& operator=(const any_member_hook& );

~any_member_hook();

bool is_linked() const;
#endif
};


namespace detail{

BOOST_INTRUSIVE_INTERNAL_STATIC_BOOL_IS_TRUE(old_proto_value_traits_base_hook, hooktags::is_base_hook)

template<class BasicHook, template <class> class NodeTraits>
struct any_to_some_hook
{
typedef typename BasicHook::template pack<empty>::proto_value_traits old_proto_value_traits;

template<class Base>
struct pack : public Base
{
struct proto_value_traits
{
struct hooktags
{
static const bool is_base_hook = old_proto_value_traits_base_hook_bool_is_true
<old_proto_value_traits>::value;
};

typedef old_proto_value_traits basic_hook_t;
static const bool is_any_hook = true;

template<class VoidPtr>
struct node_traits_from_voidptr
{  typedef NodeTraits<VoidPtr> type;  };
};
};
};

}  


template<class BasicHook>
struct any_to_slist_hook
:  public detail::any_to_some_hook<BasicHook, any_slist_node_traits>
{};

template<class BasicHook>
struct any_to_list_hook
:  public detail::any_to_some_hook<BasicHook, any_list_node_traits>
{};

template<class BasicHook>
struct any_to_set_hook
:  public detail::any_to_some_hook<BasicHook, any_rbtree_node_traits>
{};

template<class BasicHook>
struct any_to_avl_set_hook
:  public detail::any_to_some_hook<BasicHook, any_avltree_node_traits>
{};

template<class BasicHook>
struct any_to_bs_set_hook
:  public detail::any_to_some_hook<BasicHook, any_tree_node_traits>
{};

template<class BasicHook>
struct any_to_unordered_set_hook
:  public detail::any_to_some_hook<BasicHook, any_unordered_node_traits>
{};


} 
} 

#include <boost/intrusive/detail/config_end.hpp>

#endif 

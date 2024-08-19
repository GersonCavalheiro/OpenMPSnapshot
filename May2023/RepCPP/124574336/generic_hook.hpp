
#ifndef BOOST_INTRUSIVE_GENERIC_HOOK_HPP
#define BOOST_INTRUSIVE_GENERIC_HOOK_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/intrusive/pointer_traits.hpp>
#include <boost/intrusive/link_mode.hpp>
#include <boost/intrusive/detail/mpl.hpp>
#include <boost/intrusive/detail/assert.hpp>
#include <boost/intrusive/detail/node_holder.hpp>
#include <boost/intrusive/detail/algo_type.hpp>
#include <boost/static_assert.hpp>

namespace boost {
namespace intrusive {


namespace detail {

template <link_mode_type LinkMode>
struct link_dispatch
{};

template<class Hook>
BOOST_INTRUSIVE_FORCEINLINE void destructor_impl(Hook &hook, detail::link_dispatch<safe_link>)
{  
(void)hook; BOOST_INTRUSIVE_SAFE_HOOK_DESTRUCTOR_ASSERT(!hook.is_linked());
}

template<class Hook>
BOOST_INTRUSIVE_FORCEINLINE void destructor_impl(Hook &hook, detail::link_dispatch<auto_unlink>)
{  hook.unlink();  }

template<class Hook>
BOOST_INTRUSIVE_FORCEINLINE void destructor_impl(Hook &, detail::link_dispatch<normal_link>)
{}

}  

enum base_hook_type
{  NoBaseHookId
,  ListBaseHookId
,  SlistBaseHookId
,  RbTreeBaseHookId
,  HashBaseHookId
,  AvlTreeBaseHookId
,  BsTreeBaseHookId
,  TreapTreeBaseHookId
,  AnyBaseHookId
};


template <class HookTags, unsigned int>
struct hook_tags_definer{};

template <class HookTags>
struct hook_tags_definer<HookTags, ListBaseHookId>
{  typedef HookTags default_list_hook;  };

template <class HookTags>
struct hook_tags_definer<HookTags, SlistBaseHookId>
{  typedef HookTags default_slist_hook;  };

template <class HookTags>
struct hook_tags_definer<HookTags, RbTreeBaseHookId>
{  typedef HookTags default_rbtree_hook;  };

template <class HookTags>
struct hook_tags_definer<HookTags, HashBaseHookId>
{  typedef HookTags default_hashtable_hook;  };

template <class HookTags>
struct hook_tags_definer<HookTags, AvlTreeBaseHookId>
{  typedef HookTags default_avltree_hook;  };

template <class HookTags>
struct hook_tags_definer<HookTags, BsTreeBaseHookId>
{  typedef HookTags default_bstree_hook;  };

template <class HookTags>
struct hook_tags_definer<HookTags, AnyBaseHookId>
{  typedef HookTags default_any_hook;  };

template
< class NodeTraits
, class Tag
, link_mode_type LinkMode
, base_hook_type BaseHookType
>
struct hooktags_impl
{
static const link_mode_type link_mode = LinkMode;
typedef Tag tag;
typedef NodeTraits node_traits;
static const bool is_base_hook = !detail::is_same<Tag, member_tag>::value;
static const bool safemode_or_autounlink = is_safe_autounlink<link_mode>::value;
static const unsigned int type = BaseHookType;
};


template
< boost::intrusive::algo_types Algo
, class NodeTraits
, class Tag
, link_mode_type LinkMode
, base_hook_type BaseHookType
>
class generic_hook
: public detail::if_c
< detail::is_same<Tag, member_tag>::value
, typename NodeTraits::node
, node_holder<typename NodeTraits::node, Tag, BaseHookType>
>::type
, public hook_tags_definer
< generic_hook<Algo, NodeTraits, Tag, LinkMode, BaseHookType>
, detail::is_same<Tag, dft_tag>::value ? BaseHookType : NoBaseHookId>
{
typedef typename get_algo<Algo, NodeTraits>::type  node_algorithms;
typedef typename node_algorithms::node             node;
typedef typename node_algorithms::node_ptr         node_ptr;
typedef typename node_algorithms::const_node_ptr   const_node_ptr;

public:

typedef hooktags_impl
< NodeTraits
, Tag, LinkMode, BaseHookType>                  hooktags;

BOOST_INTRUSIVE_FORCEINLINE node_ptr this_ptr()
{  return pointer_traits<node_ptr>::pointer_to(static_cast<node&>(*this)); }

BOOST_INTRUSIVE_FORCEINLINE const_node_ptr this_ptr() const
{  return pointer_traits<const_node_ptr>::pointer_to(static_cast<const node&>(*this)); }

public:

BOOST_INTRUSIVE_FORCEINLINE generic_hook()
{
if(hooktags::safemode_or_autounlink){
node_algorithms::init(this->this_ptr());
}
}

BOOST_INTRUSIVE_FORCEINLINE generic_hook(const generic_hook& )
{
if(hooktags::safemode_or_autounlink){
node_algorithms::init(this->this_ptr());
}
}

BOOST_INTRUSIVE_FORCEINLINE generic_hook& operator=(const generic_hook& )
{  return *this;  }

BOOST_INTRUSIVE_FORCEINLINE ~generic_hook()
{
destructor_impl
(*this, detail::link_dispatch<hooktags::link_mode>());
}

BOOST_INTRUSIVE_FORCEINLINE void swap_nodes(generic_hook &other)
{
node_algorithms::swap_nodes
(this->this_ptr(), other.this_ptr());
}

BOOST_INTRUSIVE_FORCEINLINE bool is_linked() const
{
BOOST_STATIC_ASSERT(( hooktags::safemode_or_autounlink ));
return !node_algorithms::unique(this->this_ptr());
}

BOOST_INTRUSIVE_FORCEINLINE void unlink()
{
BOOST_STATIC_ASSERT(( (int)hooktags::link_mode == (int)auto_unlink ));
node_ptr n(this->this_ptr());
if(!node_algorithms::inited(n)){
node_algorithms::unlink(n);
node_algorithms::init(n);
}
}
};

} 
} 

#endif 

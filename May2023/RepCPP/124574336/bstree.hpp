#ifndef BOOST_INTRUSIVE_BSTREE_HPP
#define BOOST_INTRUSIVE_BSTREE_HPP

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/intrusive_fwd.hpp>

#include <boost/intrusive/detail/assert.hpp>
#include <boost/static_assert.hpp>
#include <boost/intrusive/intrusive_fwd.hpp>
#include <boost/intrusive/bs_set_hook.hpp>
#include <boost/intrusive/detail/tree_node.hpp>
#include <boost/intrusive/detail/tree_iterator.hpp>
#include <boost/intrusive/detail/ebo_functor_holder.hpp>
#include <boost/intrusive/detail/mpl.hpp>
#include <boost/intrusive/pointer_traits.hpp>
#include <boost/intrusive/detail/is_stateful_value_traits.hpp>
#include <boost/intrusive/detail/empty_node_checker.hpp>
#include <boost/intrusive/detail/default_header_holder.hpp>
#include <boost/intrusive/detail/reverse_iterator.hpp>
#include <boost/intrusive/detail/exception_disposer.hpp>
#include <boost/intrusive/detail/node_cloner_disposer.hpp>
#include <boost/intrusive/detail/key_nodeptr_comp.hpp>
#include <boost/intrusive/detail/simple_disposers.hpp>
#include <boost/intrusive/detail/size_holder.hpp>
#include <boost/intrusive/detail/algo_type.hpp>
#include <boost/intrusive/detail/algorithm.hpp>
#include <boost/intrusive/detail/tree_value_compare.hpp>

#include <boost/intrusive/detail/get_value_traits.hpp>
#include <boost/intrusive/bstree_algorithms.hpp>
#include <boost/intrusive/link_mode.hpp>
#include <boost/intrusive/parent_from_member.hpp>
#include <boost/move/utility_core.hpp>
#include <boost/move/adl_move_swap.hpp>

#include <boost/intrusive/detail/minimal_pair_header.hpp>
#include <cstddef>   
#include <boost/intrusive/detail/minimal_less_equal_header.hpp>

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace intrusive {


struct default_bstree_hook_applier
{  template <class T> struct apply{ typedef typename T::default_bstree_hook type;  };  };

template<>
struct is_default_hook_tag<default_bstree_hook_applier>
{  static const bool value = true;  };

struct bstree_defaults
{
typedef default_bstree_hook_applier proto_value_traits;
static const bool constant_time_size = true;
typedef std::size_t size_type;
typedef void compare;
typedef void key_of_value;
static const bool floating_point = true;  
typedef void priority;  
typedef void header_holder_type;
};

template<class ValueTraits, algo_types AlgoType, typename HeaderHolder>
struct bstbase3
{
typedef ValueTraits                                               value_traits;
typedef typename value_traits::node_traits                        node_traits;
typedef typename node_traits::node                                node_type;
typedef typename get_algo<AlgoType, node_traits>::type            node_algorithms;
typedef typename node_traits::node_ptr                            node_ptr;
typedef typename node_traits::const_node_ptr                      const_node_ptr;
typedef tree_iterator<value_traits, false>                                                   iterator;
typedef tree_iterator<value_traits, true>                                                    const_iterator;
typedef boost::intrusive::reverse_iterator<iterator>                                         reverse_iterator;
typedef boost::intrusive::reverse_iterator<const_iterator>                                   const_reverse_iterator;
typedef BOOST_INTRUSIVE_IMPDEF(typename value_traits::pointer)                               pointer;
typedef BOOST_INTRUSIVE_IMPDEF(typename value_traits::const_pointer)                         const_pointer;
typedef BOOST_INTRUSIVE_IMPDEF(typename pointer_traits<pointer>::element_type)               value_type;
typedef BOOST_INTRUSIVE_IMPDEF(typename pointer_traits<pointer>::reference)                  reference;
typedef BOOST_INTRUSIVE_IMPDEF(typename pointer_traits<const_pointer>::reference)            const_reference;
typedef BOOST_INTRUSIVE_IMPDEF(typename pointer_traits<const_pointer>::difference_type)      difference_type;
typedef typename detail::get_header_holder_type
< value_traits,HeaderHolder >::type                                                       header_holder_type;

static const bool safemode_or_autounlink = is_safe_autounlink<value_traits::link_mode>::value;
static const bool stateful_value_traits = detail::is_stateful_value_traits<value_traits>::value;
static const bool has_container_from_iterator =
detail::is_same< header_holder_type, detail::default_header_holder< node_traits > >::value;

struct holder_t : public ValueTraits
{
BOOST_INTRUSIVE_FORCEINLINE explicit holder_t(const ValueTraits &vtraits)
: ValueTraits(vtraits)
{}
header_holder_type root;
} holder;

static bstbase3 &get_tree_base_from_end_iterator(const const_iterator &end_iterator)
{
BOOST_STATIC_ASSERT(has_container_from_iterator);
node_ptr p = end_iterator.pointed_node();
header_holder_type* h = header_holder_type::get_holder(p);
holder_t *holder = get_parent_from_member<holder_t, header_holder_type>(h, &holder_t::root);
bstbase3 *base   = get_parent_from_member<bstbase3, holder_t> (holder, &bstbase3::holder);
return *base;
}

BOOST_INTRUSIVE_FORCEINLINE bstbase3(const ValueTraits &vtraits)
: holder(vtraits)
{
node_algorithms::init_header(this->header_ptr());
}

BOOST_INTRUSIVE_FORCEINLINE node_ptr header_ptr()
{ return holder.root.get_node(); }

BOOST_INTRUSIVE_FORCEINLINE const_node_ptr header_ptr() const
{ return holder.root.get_node(); }

BOOST_INTRUSIVE_FORCEINLINE const value_traits &get_value_traits() const
{  return this->holder;  }

BOOST_INTRUSIVE_FORCEINLINE value_traits &get_value_traits()
{  return this->holder;  }

typedef typename boost::intrusive::value_traits_pointers
<ValueTraits>::const_value_traits_ptr const_value_traits_ptr;

BOOST_INTRUSIVE_FORCEINLINE const_value_traits_ptr priv_value_traits_ptr() const
{  return pointer_traits<const_value_traits_ptr>::pointer_to(this->get_value_traits());  }

iterator begin()
{  return iterator(node_algorithms::begin_node(this->header_ptr()), this->priv_value_traits_ptr());   }

BOOST_INTRUSIVE_FORCEINLINE const_iterator begin() const
{  return cbegin();   }

const_iterator cbegin() const
{  return const_iterator(node_algorithms::begin_node(this->header_ptr()), this->priv_value_traits_ptr());   }

iterator end()
{  return iterator(node_algorithms::end_node(this->header_ptr()), this->priv_value_traits_ptr());   }

BOOST_INTRUSIVE_FORCEINLINE const_iterator end() const
{  return cend();  }

BOOST_INTRUSIVE_FORCEINLINE const_iterator cend() const
{  return const_iterator(node_algorithms::end_node(this->header_ptr()), this->priv_value_traits_ptr());   }

BOOST_INTRUSIVE_FORCEINLINE iterator root()
{  return iterator(node_algorithms::root_node(this->header_ptr()), this->priv_value_traits_ptr());   }

BOOST_INTRUSIVE_FORCEINLINE const_iterator root() const
{  return croot();   }

BOOST_INTRUSIVE_FORCEINLINE const_iterator croot() const
{  return const_iterator(node_algorithms::root_node(this->header_ptr()), this->priv_value_traits_ptr());   }

BOOST_INTRUSIVE_FORCEINLINE reverse_iterator rbegin()
{  return reverse_iterator(end());  }

BOOST_INTRUSIVE_FORCEINLINE const_reverse_iterator rbegin() const
{  return const_reverse_iterator(end());  }

BOOST_INTRUSIVE_FORCEINLINE const_reverse_iterator crbegin() const
{  return const_reverse_iterator(end());  }

BOOST_INTRUSIVE_FORCEINLINE reverse_iterator rend()
{  return reverse_iterator(begin());   }

BOOST_INTRUSIVE_FORCEINLINE const_reverse_iterator rend() const
{  return const_reverse_iterator(begin());   }

BOOST_INTRUSIVE_FORCEINLINE const_reverse_iterator crend() const
{  return const_reverse_iterator(begin());   }

void replace_node(iterator replace_this, reference with_this)
{
node_algorithms::replace_node( get_value_traits().to_node_ptr(*replace_this)
, this->header_ptr()
, get_value_traits().to_node_ptr(with_this));
if(safemode_or_autounlink)
node_algorithms::init(replace_this.pointed_node());
}

BOOST_INTRUSIVE_FORCEINLINE void rebalance()
{  node_algorithms::rebalance(this->header_ptr()); }

iterator rebalance_subtree(iterator root)
{  return iterator(node_algorithms::rebalance_subtree(root.pointed_node()), this->priv_value_traits_ptr()); }

static iterator s_iterator_to(reference value)
{
BOOST_STATIC_ASSERT((!stateful_value_traits));
return iterator (value_traits::to_node_ptr(value), const_value_traits_ptr());
}

static const_iterator s_iterator_to(const_reference value)
{
BOOST_STATIC_ASSERT((!stateful_value_traits));
return const_iterator (value_traits::to_node_ptr(*pointer_traits<pointer>::const_cast_from(pointer_traits<const_pointer>::pointer_to(value))), const_value_traits_ptr());
}

iterator iterator_to(reference value)
{  return iterator (this->get_value_traits().to_node_ptr(value), this->priv_value_traits_ptr()); }

const_iterator iterator_to(const_reference value) const
{  return const_iterator (this->get_value_traits().to_node_ptr(*pointer_traits<pointer>::const_cast_from(pointer_traits<const_pointer>::pointer_to(value))), this->priv_value_traits_ptr()); }

BOOST_INTRUSIVE_FORCEINLINE static void init_node(reference value)
{ node_algorithms::init(value_traits::to_node_ptr(value)); }

};

template<class Less, class T>
struct get_compare
{
typedef Less type;
};

template<class T>
struct get_compare<void, T>
{
typedef ::std::less<T> type;
};

template<class KeyOfValue, class T>
struct get_key_of_value
{
typedef KeyOfValue type;
};

template<class T>
struct get_key_of_value<void, T>
{
typedef ::boost::intrusive::detail::identity<T> type;
};

template<class ValuePtr, class VoidOrKeyOfValue, class VoidOrKeyComp>
struct bst_key_types
{
typedef typename
boost::movelib::pointer_element<ValuePtr>::type value_type;
typedef typename get_key_of_value
< VoidOrKeyOfValue, value_type>::type           key_of_value;
typedef typename key_of_value::type                key_type;
typedef typename get_compare< VoidOrKeyComp
, key_type
>::type                         key_compare;
typedef tree_value_compare
<ValuePtr, key_compare, key_of_value>           value_compare;
};

template<class ValueTraits, class VoidOrKeyOfValue, class VoidOrKeyComp, algo_types AlgoType, typename HeaderHolder>
struct bstbase2
: public detail::ebo_functor_holder
< typename bst_key_types
< typename ValueTraits::pointer
, VoidOrKeyOfValue
, VoidOrKeyComp

>::value_compare
>
, public bstbase3<ValueTraits, AlgoType, HeaderHolder>
{
typedef bstbase3<ValueTraits, AlgoType, HeaderHolder>             treeheader_t;
typedef bst_key_types< typename ValueTraits::pointer
, VoidOrKeyOfValue
, VoidOrKeyComp>                             key_types;
typedef typename treeheader_t::value_traits                       value_traits;
typedef typename treeheader_t::node_algorithms                    node_algorithms;
typedef typename ValueTraits::value_type                          value_type;
typedef typename key_types::key_type                              key_type;
typedef typename key_types::key_of_value                          key_of_value;
typedef typename key_types::key_compare                           key_compare;
typedef typename key_types::value_compare                         value_compare;
typedef typename treeheader_t::iterator                           iterator;
typedef typename treeheader_t::const_iterator                     const_iterator;
typedef typename treeheader_t::node_ptr                           node_ptr;
typedef typename treeheader_t::const_node_ptr                     const_node_ptr;

bstbase2(const key_compare &comp, const ValueTraits &vtraits)
: detail::ebo_functor_holder<value_compare>(value_compare(comp)), treeheader_t(vtraits)
{}

const value_compare &comp() const
{  return this->get();  }

value_compare &comp()
{  return this->get();  }

typedef BOOST_INTRUSIVE_IMPDEF(typename value_traits::pointer)                               pointer;
typedef BOOST_INTRUSIVE_IMPDEF(typename value_traits::const_pointer)                         const_pointer;
typedef BOOST_INTRUSIVE_IMPDEF(typename pointer_traits<pointer>::reference)                  reference;
typedef BOOST_INTRUSIVE_IMPDEF(typename pointer_traits<const_pointer>::reference)            const_reference;
typedef BOOST_INTRUSIVE_IMPDEF(typename pointer_traits<const_pointer>::difference_type)      difference_type;
typedef typename node_algorithms::insert_commit_data insert_commit_data;

BOOST_INTRUSIVE_FORCEINLINE value_compare value_comp() const
{  return this->comp();   }

BOOST_INTRUSIVE_FORCEINLINE key_compare key_comp() const
{  return this->comp().key_comp();   }

BOOST_INTRUSIVE_FORCEINLINE iterator lower_bound(const key_type &key)
{  return this->lower_bound(key, this->key_comp());   }

BOOST_INTRUSIVE_FORCEINLINE const_iterator lower_bound(const key_type &key) const
{  return this->lower_bound(key, this->key_comp());   }

template<class KeyType, class KeyTypeKeyCompare>
iterator lower_bound(const KeyType &key, KeyTypeKeyCompare comp)
{
return iterator(node_algorithms::lower_bound
(this->header_ptr(), key, this->key_node_comp(comp)), this->priv_value_traits_ptr());
}

template<class KeyType, class KeyTypeKeyCompare>
const_iterator lower_bound(const KeyType &key, KeyTypeKeyCompare comp) const
{
return const_iterator(node_algorithms::lower_bound
(this->header_ptr(), key, this->key_node_comp(comp)), this->priv_value_traits_ptr());
}

BOOST_INTRUSIVE_FORCEINLINE iterator upper_bound(const key_type &key)
{  return this->upper_bound(key, this->key_comp());   }

template<class KeyType, class KeyTypeKeyCompare>
iterator upper_bound(const KeyType &key, KeyTypeKeyCompare comp)
{
return iterator(node_algorithms::upper_bound
(this->header_ptr(), key, this->key_node_comp(comp)), this->priv_value_traits_ptr());
}

BOOST_INTRUSIVE_FORCEINLINE const_iterator upper_bound(const key_type &key) const
{  return this->upper_bound(key, this->key_comp());   }

template<class KeyType, class KeyTypeKeyCompare>
const_iterator upper_bound(const KeyType &key, KeyTypeKeyCompare comp) const
{
return const_iterator(node_algorithms::upper_bound
(this->header_ptr(), key, this->key_node_comp(comp)), this->priv_value_traits_ptr());
}

template<class KeyTypeKeyCompare>
struct key_node_comp_ret
{  typedef detail::key_nodeptr_comp<KeyTypeKeyCompare, value_traits, key_of_value> type;  };

template<class KeyTypeKeyCompare>
BOOST_INTRUSIVE_FORCEINLINE typename key_node_comp_ret<KeyTypeKeyCompare>::type key_node_comp(KeyTypeKeyCompare comp) const
{
return detail::key_nodeptr_comp<KeyTypeKeyCompare, value_traits, key_of_value>(comp, &this->get_value_traits());
}

BOOST_INTRUSIVE_FORCEINLINE iterator find(const key_type &key)
{  return this->find(key, this->key_comp()); }

template<class KeyType, class KeyTypeKeyCompare>
iterator find(const KeyType &key, KeyTypeKeyCompare comp)
{
return iterator
(node_algorithms::find(this->header_ptr(), key, this->key_node_comp(comp)), this->priv_value_traits_ptr());
}

BOOST_INTRUSIVE_FORCEINLINE const_iterator find(const key_type &key) const
{  return this->find(key, this->key_comp()); }

template<class KeyType, class KeyTypeKeyCompare>
const_iterator find(const KeyType &key, KeyTypeKeyCompare comp) const
{
return const_iterator
(node_algorithms::find(this->header_ptr(), key, this->key_node_comp(comp)), this->priv_value_traits_ptr());
}

BOOST_INTRUSIVE_FORCEINLINE std::pair<iterator,iterator> equal_range(const key_type &key)
{  return this->equal_range(key, this->key_comp());   }

template<class KeyType, class KeyTypeKeyCompare>
std::pair<iterator,iterator> equal_range(const KeyType &key, KeyTypeKeyCompare comp)
{
std::pair<node_ptr, node_ptr> ret
(node_algorithms::equal_range(this->header_ptr(), key, this->key_node_comp(comp)));
return std::pair<iterator, iterator>( iterator(ret.first, this->priv_value_traits_ptr())
, iterator(ret.second, this->priv_value_traits_ptr()));
}

BOOST_INTRUSIVE_FORCEINLINE std::pair<const_iterator, const_iterator>
equal_range(const key_type &key) const
{  return this->equal_range(key, this->key_comp());   }

template<class KeyType, class KeyTypeKeyCompare>
std::pair<const_iterator, const_iterator>
equal_range(const KeyType &key, KeyTypeKeyCompare comp) const
{
std::pair<node_ptr, node_ptr> ret
(node_algorithms::equal_range(this->header_ptr(), key, this->key_node_comp(comp)));
return std::pair<const_iterator, const_iterator>( const_iterator(ret.first, this->priv_value_traits_ptr())
, const_iterator(ret.second, this->priv_value_traits_ptr()));
}

BOOST_INTRUSIVE_FORCEINLINE std::pair<iterator,iterator> lower_bound_range(const key_type &key)
{  return this->lower_bound_range(key, this->key_comp());   }

template<class KeyType, class KeyTypeKeyCompare>
std::pair<iterator,iterator> lower_bound_range(const KeyType &key, KeyTypeKeyCompare comp)
{
std::pair<node_ptr, node_ptr> ret
(node_algorithms::lower_bound_range(this->header_ptr(), key, this->key_node_comp(comp)));
return std::pair<iterator, iterator>( iterator(ret.first, this->priv_value_traits_ptr())
, iterator(ret.second, this->priv_value_traits_ptr()));
}

BOOST_INTRUSIVE_FORCEINLINE std::pair<const_iterator, const_iterator>
lower_bound_range(const key_type &key) const
{  return this->lower_bound_range(key, this->key_comp());   }

template<class KeyType, class KeyTypeKeyCompare>
std::pair<const_iterator, const_iterator>
lower_bound_range(const KeyType &key, KeyTypeKeyCompare comp) const
{
std::pair<node_ptr, node_ptr> ret
(node_algorithms::lower_bound_range(this->header_ptr(), key, this->key_node_comp(comp)));
return std::pair<const_iterator, const_iterator>( const_iterator(ret.first, this->priv_value_traits_ptr())
, const_iterator(ret.second, this->priv_value_traits_ptr()));
}

BOOST_INTRUSIVE_FORCEINLINE std::pair<iterator,iterator> bounded_range
(const key_type &lower_key, const key_type &upper_key, bool left_closed, bool right_closed)
{  return this->bounded_range(lower_key, upper_key, this->key_comp(), left_closed, right_closed);   }

template<class KeyType, class KeyTypeKeyCompare>
std::pair<iterator,iterator> bounded_range
(const KeyType &lower_key, const KeyType &upper_key, KeyTypeKeyCompare comp, bool left_closed, bool right_closed)
{
std::pair<node_ptr, node_ptr> ret
(node_algorithms::bounded_range
(this->header_ptr(), lower_key, upper_key, this->key_node_comp(comp), left_closed, right_closed));
return std::pair<iterator, iterator>( iterator(ret.first, this->priv_value_traits_ptr())
, iterator(ret.second, this->priv_value_traits_ptr()));
}

BOOST_INTRUSIVE_FORCEINLINE std::pair<const_iterator,const_iterator> bounded_range
(const key_type &lower_key, const key_type &upper_key, bool left_closed, bool right_closed) const
{  return this->bounded_range(lower_key, upper_key, this->key_comp(), left_closed, right_closed);   }

template<class KeyType, class KeyTypeKeyCompare>
std::pair<const_iterator,const_iterator> bounded_range
(const KeyType &lower_key, const KeyType &upper_key, KeyTypeKeyCompare comp, bool left_closed, bool right_closed) const
{
std::pair<node_ptr, node_ptr> ret
(node_algorithms::bounded_range
(this->header_ptr(), lower_key, upper_key, this->key_node_comp(comp), left_closed, right_closed));
return std::pair<const_iterator, const_iterator>( const_iterator(ret.first, this->priv_value_traits_ptr())
, const_iterator(ret.second, this->priv_value_traits_ptr()));
}

BOOST_INTRUSIVE_FORCEINLINE std::pair<iterator, bool> insert_unique_check
(const key_type &key, insert_commit_data &commit_data)
{  return this->insert_unique_check(key, this->key_comp(), commit_data);   }

BOOST_INTRUSIVE_FORCEINLINE std::pair<iterator, bool> insert_unique_check
(const_iterator hint, const key_type &key, insert_commit_data &commit_data)
{  return this->insert_unique_check(hint, key, this->key_comp(), commit_data);   }

template<class KeyType, class KeyTypeKeyCompare>
BOOST_INTRUSIVE_DOC1ST(std::pair<iterator BOOST_INTRUSIVE_I bool>
, typename detail::disable_if_convertible
<KeyType BOOST_INTRUSIVE_I const_iterator BOOST_INTRUSIVE_I 
std::pair<iterator BOOST_INTRUSIVE_I bool> >::type)
insert_unique_check
(const KeyType &key, KeyTypeKeyCompare comp, insert_commit_data &commit_data)
{
std::pair<node_ptr, bool> ret =
(node_algorithms::insert_unique_check
(this->header_ptr(), key, this->key_node_comp(comp), commit_data));
return std::pair<iterator, bool>(iterator(ret.first, this->priv_value_traits_ptr()), ret.second);
}

template<class KeyType, class KeyTypeKeyCompare>
std::pair<iterator, bool> insert_unique_check
(const_iterator hint, const KeyType &key, KeyTypeKeyCompare comp, insert_commit_data &commit_data)
{
std::pair<node_ptr, bool> ret =
(node_algorithms::insert_unique_check
(this->header_ptr(), hint.pointed_node(), key, this->key_node_comp(comp), commit_data));
return std::pair<iterator, bool>(iterator(ret.first, this->priv_value_traits_ptr()), ret.second);
}
};

template<class ValueTraits, class VoidOrKeyOfValue, class VoidOrKeyComp, bool ConstantTimeSize, class SizeType, algo_types AlgoType, typename HeaderHolder>
struct bstbase_hack
: public detail::size_holder<ConstantTimeSize, SizeType>
, public bstbase2 < ValueTraits, VoidOrKeyOfValue, VoidOrKeyComp, AlgoType, HeaderHolder>
{
typedef bstbase2< ValueTraits, VoidOrKeyOfValue, VoidOrKeyComp, AlgoType, HeaderHolder> base_type;
typedef typename base_type::key_compare         key_compare;
typedef typename base_type::value_compare       value_compare;
typedef SizeType                                size_type;
typedef typename base_type::node_traits         node_traits;
typedef typename get_algo
<AlgoType, node_traits>::type                algo_type;

BOOST_INTRUSIVE_FORCEINLINE bstbase_hack(const key_compare & comp, const ValueTraits &vtraits)
: base_type(comp, vtraits)
{
this->sz_traits().set_size(size_type(0));
}

typedef detail::size_holder<ConstantTimeSize, SizeType>     size_traits;

BOOST_INTRUSIVE_FORCEINLINE size_traits &sz_traits()
{  return static_cast<size_traits &>(*this);  }

BOOST_INTRUSIVE_FORCEINLINE const size_traits &sz_traits() const
{  return static_cast<const size_traits &>(*this);  }
};

template<class ValueTraits, class VoidOrKeyOfValue, class VoidOrKeyComp, class SizeType, algo_types AlgoType, typename HeaderHolder>
struct bstbase_hack<ValueTraits, VoidOrKeyOfValue, VoidOrKeyComp, false, SizeType, AlgoType, HeaderHolder>
: public bstbase2 < ValueTraits, VoidOrKeyOfValue, VoidOrKeyComp, AlgoType, HeaderHolder>
{
typedef bstbase2< ValueTraits, VoidOrKeyOfValue, VoidOrKeyComp, AlgoType, HeaderHolder> base_type;
typedef typename base_type::value_compare       value_compare;
typedef typename base_type::key_compare         key_compare;
BOOST_INTRUSIVE_FORCEINLINE bstbase_hack(const key_compare & comp, const ValueTraits &vtraits)
: base_type(comp, vtraits)
{}

typedef detail::size_holder<false, SizeType>     size_traits;

BOOST_INTRUSIVE_FORCEINLINE size_traits sz_traits() const
{  return size_traits();  }
};

template<class ValueTraits, class VoidOrKeyOfValue, class VoidOrKeyComp, bool ConstantTimeSize, class SizeType, algo_types AlgoType, typename HeaderHolder>
struct bstbase
: public bstbase_hack< ValueTraits, VoidOrKeyOfValue, VoidOrKeyComp, ConstantTimeSize, SizeType, AlgoType, HeaderHolder>
{
typedef bstbase_hack< ValueTraits, VoidOrKeyOfValue, VoidOrKeyComp, ConstantTimeSize, SizeType, AlgoType, HeaderHolder> base_type;
typedef ValueTraits                             value_traits;
typedef typename base_type::value_compare       value_compare;
typedef typename base_type::key_compare         key_compare;
typedef typename base_type::const_reference     const_reference;
typedef typename base_type::reference           reference;
typedef typename base_type::iterator            iterator;
typedef typename base_type::const_iterator      const_iterator;
typedef typename base_type::node_traits         node_traits;
typedef typename get_algo
<AlgoType, node_traits>::type                node_algorithms;
typedef SizeType                                size_type;

BOOST_INTRUSIVE_FORCEINLINE bstbase(const key_compare & comp, const ValueTraits &vtraits)
: base_type(comp, vtraits)
{}

~bstbase()
{
if(is_safe_autounlink<value_traits::link_mode>::value){
node_algorithms::clear_and_dispose
( this->header_ptr()
, detail::node_disposer<detail::null_disposer, value_traits, AlgoType>
(detail::null_disposer(), &this->get_value_traits()));
node_algorithms::init(this->header_ptr());
}
}
};



#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
template<class T, class ...Options>
#else
template<class ValueTraits, class VoidOrKeyOfValue, class VoidOrKeyComp, class SizeType, bool ConstantTimeSize, algo_types AlgoType, typename HeaderHolder>
#endif
class bstree_impl
:  public bstbase<ValueTraits, VoidOrKeyOfValue, VoidOrKeyComp, ConstantTimeSize, SizeType, AlgoType, HeaderHolder>
{
public:
typedef bstbase<ValueTraits, VoidOrKeyOfValue, VoidOrKeyComp, ConstantTimeSize, SizeType, AlgoType, HeaderHolder> data_type;
typedef tree_iterator<ValueTraits, false> iterator_type;
typedef tree_iterator<ValueTraits, true>  const_iterator_type;

typedef BOOST_INTRUSIVE_IMPDEF(ValueTraits)                                                  value_traits;
typedef BOOST_INTRUSIVE_IMPDEF(typename value_traits::pointer)                               pointer;
typedef BOOST_INTRUSIVE_IMPDEF(typename value_traits::const_pointer)                         const_pointer;
typedef BOOST_INTRUSIVE_IMPDEF(typename pointer_traits<pointer>::element_type)               value_type;
typedef BOOST_INTRUSIVE_IMPDEF(typename data_type::key_type)                                 key_type;
typedef BOOST_INTRUSIVE_IMPDEF(typename data_type::key_of_value)                             key_of_value;
typedef BOOST_INTRUSIVE_IMPDEF(typename pointer_traits<pointer>::reference)                  reference;
typedef BOOST_INTRUSIVE_IMPDEF(typename pointer_traits<const_pointer>::reference)            const_reference;
typedef BOOST_INTRUSIVE_IMPDEF(typename pointer_traits<const_pointer>::difference_type)      difference_type;
typedef BOOST_INTRUSIVE_IMPDEF(SizeType)                                                     size_type;
typedef BOOST_INTRUSIVE_IMPDEF(typename data_type::value_compare)                            value_compare;
typedef BOOST_INTRUSIVE_IMPDEF(typename data_type::key_compare)                              key_compare;
typedef BOOST_INTRUSIVE_IMPDEF(iterator_type)                                                iterator;
typedef BOOST_INTRUSIVE_IMPDEF(const_iterator_type)                                          const_iterator;
typedef BOOST_INTRUSIVE_IMPDEF(boost::intrusive::reverse_iterator<iterator>)                 reverse_iterator;
typedef BOOST_INTRUSIVE_IMPDEF(boost::intrusive::reverse_iterator<const_iterator>)           const_reverse_iterator;
typedef BOOST_INTRUSIVE_IMPDEF(typename value_traits::node_traits)                           node_traits;
typedef BOOST_INTRUSIVE_IMPDEF(typename node_traits::node)                                   node;
typedef BOOST_INTRUSIVE_IMPDEF(typename node_traits::node_ptr)                               node_ptr;
typedef BOOST_INTRUSIVE_IMPDEF(typename node_traits::const_node_ptr)                         const_node_ptr;
typedef typename get_algo<AlgoType, node_traits>::type                                       algo_type;
typedef BOOST_INTRUSIVE_IMPDEF(algo_type)                                                    node_algorithms;

static const bool constant_time_size = ConstantTimeSize;
static const bool stateful_value_traits = detail::is_stateful_value_traits<value_traits>::value;
private:

BOOST_MOVABLE_BUT_NOT_COPYABLE(bstree_impl)

static const bool safemode_or_autounlink = is_safe_autounlink<value_traits::link_mode>::value;

BOOST_STATIC_ASSERT(!(constant_time_size && ((int)value_traits::link_mode == (int)auto_unlink)));


protected:



public:

typedef typename node_algorithms::insert_commit_data insert_commit_data;

bstree_impl()
:  data_type(key_compare(), value_traits())
{}

explicit bstree_impl( const key_compare &cmp, const value_traits &v_traits = value_traits())
:  data_type(cmp, v_traits)
{}

template<class Iterator>
bstree_impl( bool unique, Iterator b, Iterator e
, const key_compare &cmp     = key_compare()
, const value_traits &v_traits = value_traits())
: data_type(cmp, v_traits)
{
if(unique)
this->insert_unique(b, e);
else
this->insert_equal(b, e);
}

bstree_impl(BOOST_RV_REF(bstree_impl) x)
: data_type(::boost::move(x.comp()), ::boost::move(x.get_value_traits()))
{
this->swap(x);
}

BOOST_INTRUSIVE_FORCEINLINE bstree_impl& operator=(BOOST_RV_REF(bstree_impl) x)
{  this->swap(x); return *this;  }

#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED
~bstree_impl()
{}

iterator begin();

const_iterator begin() const;

const_iterator cbegin() const;

iterator end();

const_iterator end() const;

const_iterator cend() const;

reverse_iterator rbegin();

const_reverse_iterator rbegin() const;

const_reverse_iterator crbegin() const;

reverse_iterator rend();

const_reverse_iterator rend() const;

const_reverse_iterator crend() const;

iterator root();

const_iterator root() const;

const_iterator croot() const;

#endif   

static bstree_impl &container_from_end_iterator(iterator end_iterator)
{
return static_cast<bstree_impl&>
(data_type::get_tree_base_from_end_iterator(end_iterator));
}

static const bstree_impl &container_from_end_iterator(const_iterator end_iterator)
{
return static_cast<bstree_impl&>
(data_type::get_tree_base_from_end_iterator(end_iterator));
}

static bstree_impl &container_from_iterator(iterator it)
{  return container_from_end_iterator(it.end_iterator_from_it());   }

static const bstree_impl &container_from_iterator(const_iterator it)
{  return container_from_end_iterator(it.end_iterator_from_it());   }

#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED

key_compare key_comp() const;

value_compare value_comp() const;

#endif   

bool empty() const
{
if(ConstantTimeSize){
return !this->data_type::sz_traits().get_size();
}
else{
return algo_type::unique(this->header_ptr());
}
}

size_type size() const
{
if(constant_time_size)
return this->sz_traits().get_size();
else{
return (size_type)node_algorithms::size(this->header_ptr());
}
}

void swap(bstree_impl& other)
{
::boost::adl_move_swap(this->comp(), other.comp());
node_algorithms::swap_tree(this->header_ptr(), node_ptr(other.header_ptr()));
this->sz_traits().swap(other.sz_traits());
}

template <class Cloner, class Disposer>
void clone_from(const bstree_impl &src, Cloner cloner, Disposer disposer)
{
this->clear_and_dispose(disposer);
if(!src.empty()){
detail::exception_disposer<bstree_impl, Disposer>
rollback(*this, disposer);
node_algorithms::clone
(src.header_ptr()
,this->header_ptr()
,detail::node_cloner <Cloner,    value_traits, AlgoType>(cloner,   &this->get_value_traits())
,detail::node_disposer<Disposer, value_traits, AlgoType>(disposer, &this->get_value_traits()));
this->sz_traits().set_size(src.sz_traits().get_size());
this->comp() = src.comp();
rollback.release();
}
}

template <class Cloner, class Disposer>
void clone_from(BOOST_RV_REF(bstree_impl) src, Cloner cloner, Disposer disposer)
{
this->clear_and_dispose(disposer);
if(!src.empty()){
detail::exception_disposer<bstree_impl, Disposer>
rollback(*this, disposer);
node_algorithms::clone
(src.header_ptr()
,this->header_ptr()
,detail::node_cloner <Cloner,    value_traits, AlgoType, false>(cloner,   &this->get_value_traits())
,detail::node_disposer<Disposer, value_traits, AlgoType>(disposer, &this->get_value_traits()));
this->sz_traits().set_size(src.sz_traits().get_size());
this->comp() = src.comp();
rollback.release();
}
}

iterator insert_equal(reference value)
{
node_ptr to_insert(this->get_value_traits().to_node_ptr(value));
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || node_algorithms::unique(to_insert));
iterator ret(node_algorithms::insert_equal_upper_bound
(this->header_ptr(), to_insert, this->key_node_comp(this->key_comp())), this->priv_value_traits_ptr());
this->sz_traits().increment();
return ret;
}

iterator insert_equal(const_iterator hint, reference value)
{
node_ptr to_insert(this->get_value_traits().to_node_ptr(value));
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || node_algorithms::unique(to_insert));
iterator ret(node_algorithms::insert_equal
(this->header_ptr(), hint.pointed_node(), to_insert, this->key_node_comp(this->key_comp())), this->priv_value_traits_ptr());
this->sz_traits().increment();
return ret;
}

template<class Iterator>
void insert_equal(Iterator b, Iterator e)
{
iterator iend(this->end());
for (; b != e; ++b)
this->insert_equal(iend, *b);
}

std::pair<iterator, bool> insert_unique(reference value)
{
insert_commit_data commit_data;
std::pair<node_ptr, bool> ret =
(node_algorithms::insert_unique_check
(this->header_ptr(), key_of_value()(value), this->key_node_comp(this->key_comp()), commit_data));
return std::pair<iterator, bool>
( ret.second ? this->insert_unique_commit(value, commit_data)
: iterator(ret.first, this->priv_value_traits_ptr())
, ret.second);
}

iterator insert_unique(const_iterator hint, reference value)
{
insert_commit_data commit_data;
std::pair<node_ptr, bool> ret =
(node_algorithms::insert_unique_check
(this->header_ptr(), hint.pointed_node(), key_of_value()(value), this->key_node_comp(this->key_comp()), commit_data));
return ret.second ? this->insert_unique_commit(value, commit_data)
: iterator(ret.first, this->priv_value_traits_ptr());
}

template<class Iterator>
void insert_unique(Iterator b, Iterator e)
{
if(this->empty()){
iterator iend(this->end());
for (; b != e; ++b)
this->insert_unique(iend, *b);
}
else{
for (; b != e; ++b)
this->insert_unique(*b);
}
}

#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED

std::pair<iterator, bool> insert_unique_check(const key_type &key, insert_commit_data &commit_data);

std::pair<iterator, bool> insert_unique_check(const_iterator hint, const key_type &key, insert_commit_data &commit_data);

template<class KeyType, class KeyTypeKeyCompare>
std::pair<iterator, bool> insert_unique_check
(const KeyType &key, KeyTypeKeyCompare comp, insert_commit_data &commit_data);

template<class KeyType, class KeyTypeKeyCompare>
std::pair<iterator, bool> insert_unique_check
(const_iterator hint, const KeyType &key
,KeyTypeKeyCompare comp, insert_commit_data &commit_data);

#endif   

iterator insert_unique_commit(reference value, const insert_commit_data &commit_data)
{
node_ptr to_insert(this->get_value_traits().to_node_ptr(value));
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || node_algorithms::unique(to_insert));

#if !(defined(BOOST_DISABLE_ASSERTS) || ( defined(BOOST_ENABLE_ASSERT_DEBUG_HANDLER) && defined(NDEBUG) ))
iterator p(commit_data.node, this->priv_value_traits_ptr());
if(!commit_data.link_left){
++p;
}
BOOST_ASSERT(( p == this->end()   || !this->comp()(*p, value)   ));
BOOST_ASSERT(( p == this->begin() || !this->comp()(value, *--p) ));
#endif

node_algorithms::insert_unique_commit
(this->header_ptr(), to_insert, commit_data);
this->sz_traits().increment();
return iterator(to_insert, this->priv_value_traits_ptr());
}

iterator insert_before(const_iterator pos, reference value)
{
node_ptr to_insert(this->get_value_traits().to_node_ptr(value));
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || node_algorithms::unique(to_insert));
this->sz_traits().increment();
return iterator(node_algorithms::insert_before
(this->header_ptr(), pos.pointed_node(), to_insert), this->priv_value_traits_ptr());
}

void push_back(reference value)
{
node_ptr to_insert(this->get_value_traits().to_node_ptr(value));
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || node_algorithms::unique(to_insert));
this->sz_traits().increment();
node_algorithms::push_back(this->header_ptr(), to_insert);
}

void push_front(reference value)
{
node_ptr to_insert(this->get_value_traits().to_node_ptr(value));
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || node_algorithms::unique(to_insert));
this->sz_traits().increment();
node_algorithms::push_front(this->header_ptr(), to_insert);
}

iterator erase(const_iterator i)
{
const_iterator ret(i);
++ret;
node_ptr to_erase(i.pointed_node());
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || !node_algorithms::unique(to_erase));
node_algorithms::erase(this->header_ptr(), to_erase);
this->sz_traits().decrement();
if(safemode_or_autounlink)
node_algorithms::init(to_erase);
return ret.unconst();
}

iterator erase(const_iterator b, const_iterator e)
{  size_type n;   return this->private_erase(b, e, n);   }

size_type erase(const key_type &key)
{  return this->erase(key, this->key_comp());   }

template<class KeyType, class KeyTypeKeyCompare>
BOOST_INTRUSIVE_DOC1ST(size_type
, typename detail::disable_if_convertible<KeyTypeKeyCompare BOOST_INTRUSIVE_I const_iterator BOOST_INTRUSIVE_I size_type>::type)
erase(const KeyType& key, KeyTypeKeyCompare comp)
{
std::pair<iterator,iterator> p = this->equal_range(key, comp);
size_type n;
this->private_erase(p.first, p.second, n);
return n;
}

template<class Disposer>
iterator erase_and_dispose(const_iterator i, Disposer disposer)
{
node_ptr to_erase(i.pointed_node());
iterator ret(this->erase(i));
disposer(this->get_value_traits().to_value_ptr(to_erase));
return ret;
}

template<class Disposer>
size_type erase_and_dispose(const key_type &key, Disposer disposer)
{
std::pair<iterator,iterator> p = this->equal_range(key);
size_type n;
this->private_erase(p.first, p.second, n, disposer);
return n;
}

template<class Disposer>
iterator erase_and_dispose(const_iterator b, const_iterator e, Disposer disposer)
{  size_type n;   return this->private_erase(b, e, n, disposer);   }

template<class KeyType, class KeyTypeKeyCompare, class Disposer>
BOOST_INTRUSIVE_DOC1ST(size_type
, typename detail::disable_if_convertible<KeyTypeKeyCompare BOOST_INTRUSIVE_I const_iterator BOOST_INTRUSIVE_I size_type>::type)
erase_and_dispose(const KeyType& key, KeyTypeKeyCompare comp, Disposer disposer)
{
std::pair<iterator,iterator> p = this->equal_range(key, comp);
size_type n;
this->private_erase(p.first, p.second, n, disposer);
return n;
}

void clear()
{
if(safemode_or_autounlink){
this->clear_and_dispose(detail::null_disposer());
}
else{
node_algorithms::init_header(this->header_ptr());
this->sz_traits().set_size(0);
}
}

template<class Disposer>
void clear_and_dispose(Disposer disposer)
{
node_algorithms::clear_and_dispose(this->header_ptr()
, detail::node_disposer<Disposer, value_traits, AlgoType>(disposer, &this->get_value_traits()));
node_algorithms::init_header(this->header_ptr());
this->sz_traits().set_size(0);
}

size_type count(const key_type &key) const
{  return size_type(this->count(key, this->key_comp()));   }

template<class KeyType, class KeyTypeKeyCompare>
size_type count(const KeyType &key, KeyTypeKeyCompare comp) const
{
std::pair<const_iterator, const_iterator> ret = this->equal_range(key, comp);
size_type n = 0;
for(; ret.first != ret.second; ++ret.first){ ++n; }
return n;
}

#if !defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)

size_type count(const key_type &key)
{  return size_type(this->count(key, this->key_comp()));   }

template<class KeyType, class KeyTypeKeyCompare>
size_type count(const KeyType &key, KeyTypeKeyCompare comp)
{
std::pair<const_iterator, const_iterator> ret = this->equal_range(key, comp);
size_type n = 0;
for(; ret.first != ret.second; ++ret.first){ ++n; }
return n;
}

#else 

iterator lower_bound(const key_type &key);

const_iterator lower_bound(const key_type &key) const;

template<class KeyType, class KeyTypeKeyCompare>
iterator lower_bound(const KeyType &key, KeyTypeKeyCompare comp);

template<class KeyType, class KeyTypeKeyCompare>
const_iterator lower_bound(const KeyType &key, KeyTypeKeyCompare comp) const;

iterator upper_bound(const key_type &key);

template<class KeyType, class KeyTypeKeyCompare>
iterator upper_bound(const KeyType &key, KeyTypeKeyCompare comp);

const_iterator upper_bound(const key_type &key) const;

template<class KeyType, class KeyTypeKeyCompare>
const_iterator upper_bound(const KeyType &key, KeyTypeKeyCompare comp) const;

iterator find(const key_type &key);

template<class KeyType, class KeyTypeKeyCompare>
iterator find(const KeyType &key, KeyTypeKeyCompare comp);

const_iterator find(const key_type &key) const;

template<class KeyType, class KeyTypeKeyCompare>
const_iterator find(const KeyType &key, KeyTypeKeyCompare comp) const;

std::pair<iterator,iterator> equal_range(const key_type &key);

template<class KeyType, class KeyTypeKeyCompare>
std::pair<iterator,iterator> equal_range(const KeyType &key, KeyTypeKeyCompare comp);

std::pair<const_iterator, const_iterator> equal_range(const key_type &key) const;

template<class KeyType, class KeyTypeKeyCompare>
std::pair<const_iterator, const_iterator>
equal_range(const KeyType &key, KeyTypeKeyCompare comp) const;

std::pair<iterator,iterator> bounded_range
(const key_type &lower_key, const key_type &upper_value, bool left_closed, bool right_closed);

template<class KeyType, class KeyTypeKeyCompare>
std::pair<iterator,iterator> bounded_range
(const KeyType &lower_key, const KeyType &upper_key, KeyTypeKeyCompare comp, bool left_closed, bool right_closed);

std::pair<const_iterator,const_iterator> bounded_range
(const key_type &lower_key, const key_type &upper_key, bool left_closed, bool right_closed) const;

template<class KeyType, class KeyTypeKeyCompare>
std::pair<const_iterator,const_iterator> bounded_range
(const KeyType &lower_key, const KeyType &upper_key, KeyTypeKeyCompare comp, bool left_closed, bool right_closed) const;

static iterator s_iterator_to(reference value);

static const_iterator s_iterator_to(const_reference value);

iterator iterator_to(reference value);

const_iterator iterator_to(const_reference value) const;

static void init_node(reference value);

#endif   

pointer unlink_leftmost_without_rebalance()
{
node_ptr to_be_disposed(node_algorithms::unlink_leftmost_without_rebalance
(this->header_ptr()));
if(!to_be_disposed)
return 0;
this->sz_traits().decrement();
if(safemode_or_autounlink)
node_algorithms::init(to_be_disposed);
return this->get_value_traits().to_value_ptr(to_be_disposed);
}

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)

void replace_node(iterator replace_this, reference with_this);

void rebalance();

iterator rebalance_subtree(iterator root);

#endif   

static void remove_node(reference value)
{
BOOST_STATIC_ASSERT((!constant_time_size));
node_ptr to_remove(value_traits::to_node_ptr(value));
node_algorithms::unlink(to_remove);
if(safemode_or_autounlink)
node_algorithms::init(to_remove);
}

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
template<class T, class ...Options2> void merge_unique(bstree<T, Options2...> &);
#else
template<class Compare2>
void merge_unique(bstree_impl
<ValueTraits, VoidOrKeyOfValue, Compare2, SizeType, ConstantTimeSize, AlgoType, HeaderHolder> &source)
#endif
{
node_ptr it   (node_algorithms::begin_node(source.header_ptr()))
, itend(node_algorithms::end_node  (source.header_ptr()));

while(it != itend){
node_ptr const p(it);
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || !node_algorithms::unique(p));
it = node_algorithms::next_node(it);
if( node_algorithms::transfer_unique(this->header_ptr(), this->key_node_comp(this->key_comp()), source.header_ptr(), p) ){
source.sz_traits().decrement();
this->sz_traits().increment();
}
}
}

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
template<class T, class ...Options2> void merge_equal(bstree<T, Options2...> &);
#else
template<class Compare2>
void merge_equal(bstree_impl
<ValueTraits, VoidOrKeyOfValue, Compare2, SizeType, ConstantTimeSize, AlgoType, HeaderHolder> &source)
#endif
{
node_ptr it   (node_algorithms::begin_node(source.header_ptr()))
, itend(node_algorithms::end_node  (source.header_ptr()));

while(it != itend){
node_ptr const p(it);
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || !node_algorithms::unique(p));
it = node_algorithms::next_node(it);
node_algorithms::transfer_equal(this->header_ptr(), this->key_node_comp(this->key_comp()), source.header_ptr(), p);
source.sz_traits().decrement();
this->sz_traits().increment();
}
}

template <class ExtraChecker>
void check(ExtraChecker extra_checker) const
{
typedef detail::key_nodeptr_comp<key_compare, value_traits, key_of_value> nodeptr_comp_t;
nodeptr_comp_t nodeptr_comp(this->key_comp(), &this->get_value_traits());
typedef typename get_node_checker<AlgoType, ValueTraits, nodeptr_comp_t, ExtraChecker>::type node_checker_t;
typename node_checker_t::return_type checker_return;
node_algorithms::check(this->header_ptr(), node_checker_t(nodeptr_comp, extra_checker), checker_return);
BOOST_INTRUSIVE_INVARIANT_ASSERT(!constant_time_size || this->sz_traits().get_size() == checker_return.node_count);
}

void check() const
{
check(detail::empty_node_checker<ValueTraits>());
}

friend bool operator==(const bstree_impl &x, const bstree_impl &y)
{
if(constant_time_size && x.size() != y.size()){
return false;
}
return boost::intrusive::algo_equal(x.cbegin(), x.cend(), y.cbegin(), y.cend());
}

friend bool operator!=(const bstree_impl &x, const bstree_impl &y)
{  return !(x == y); }

friend bool operator<(const bstree_impl &x, const bstree_impl &y)
{  return ::boost::intrusive::algo_lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());  }

friend bool operator>(const bstree_impl &x, const bstree_impl &y)
{  return y < x;  }

friend bool operator<=(const bstree_impl &x, const bstree_impl &y)
{  return !(x > y);  }

friend bool operator>=(const bstree_impl &x, const bstree_impl &y)
{  return !(x < y);  }

friend void swap(bstree_impl &x, bstree_impl &y)
{  x.swap(y);  }

private:
template<class Disposer>
iterator private_erase(const_iterator b, const_iterator e, size_type &n, Disposer disposer)
{
for(n = 0; b != e; ++n)
this->erase_and_dispose(b++, disposer);
return b.unconst();
}

iterator private_erase(const_iterator b, const_iterator e, size_type &n)
{
for(n = 0; b != e; ++n)
this->erase(b++);
return b.unconst();
}
};

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED) || defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template<class T, class ...Options>
#else
template<class T, class O1 = void, class O2 = void
, class O3 = void, class O4 = void
, class O5 = void, class O6 = void>
#endif
struct make_bstree
{
typedef typename pack_options
< bstree_defaults,
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3, O4, O5, O6
#else
Options...
#endif
>::type packed_options;

typedef typename detail::get_value_traits
<T, typename packed_options::proto_value_traits>::type value_traits;

typedef bstree_impl
< value_traits
, typename packed_options::key_of_value
, typename packed_options::compare
, typename packed_options::size_type
, packed_options::constant_time_size
, BsTreeAlgorithms
, typename packed_options::header_holder_type
> implementation_defined;
typedef implementation_defined type;
};


#ifndef BOOST_INTRUSIVE_DOXYGEN_INVOKED

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template<class T, class O1, class O2, class O3, class O4, class O5, class O6>
#else
template<class T, class ...Options>
#endif
class bstree
:  public make_bstree<T,
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3, O4, O5, O6
#else
Options...
#endif
>::type
{
typedef typename make_bstree
<T,
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3, O4, O5, O6
#else
Options...
#endif
>::type   Base;
BOOST_MOVABLE_BUT_NOT_COPYABLE(bstree)

public:
typedef typename Base::key_compare        key_compare;
typedef typename Base::value_traits       value_traits;
typedef typename Base::iterator           iterator;
typedef typename Base::const_iterator     const_iterator;

BOOST_STATIC_ASSERT((detail::is_same<typename value_traits::value_type, T>::value));

BOOST_INTRUSIVE_FORCEINLINE bstree()
:  Base()
{}

BOOST_INTRUSIVE_FORCEINLINE explicit bstree( const key_compare &cmp, const value_traits &v_traits = value_traits())
:  Base(cmp, v_traits)
{}

template<class Iterator>
BOOST_INTRUSIVE_FORCEINLINE bstree( bool unique, Iterator b, Iterator e
, const key_compare &cmp = key_compare()
, const value_traits &v_traits = value_traits())
:  Base(unique, b, e, cmp, v_traits)
{}

BOOST_INTRUSIVE_FORCEINLINE bstree(BOOST_RV_REF(bstree) x)
:  Base(BOOST_MOVE_BASE(Base, x))
{}

BOOST_INTRUSIVE_FORCEINLINE bstree& operator=(BOOST_RV_REF(bstree) x)
{  return static_cast<bstree &>(this->Base::operator=(BOOST_MOVE_BASE(Base, x)));  }

template <class Cloner, class Disposer>
BOOST_INTRUSIVE_FORCEINLINE void clone_from(const bstree &src, Cloner cloner, Disposer disposer)
{  Base::clone_from(src, cloner, disposer);  }

template <class Cloner, class Disposer>
BOOST_INTRUSIVE_FORCEINLINE void clone_from(BOOST_RV_REF(bstree) src, Cloner cloner, Disposer disposer)
{  Base::clone_from(BOOST_MOVE_BASE(Base, src), cloner, disposer);  }

BOOST_INTRUSIVE_FORCEINLINE static bstree &container_from_end_iterator(iterator end_iterator)
{  return static_cast<bstree &>(Base::container_from_end_iterator(end_iterator));   }

BOOST_INTRUSIVE_FORCEINLINE static const bstree &container_from_end_iterator(const_iterator end_iterator)
{  return static_cast<const bstree &>(Base::container_from_end_iterator(end_iterator));   }

BOOST_INTRUSIVE_FORCEINLINE static bstree &container_from_iterator(iterator it)
{  return static_cast<bstree &>(Base::container_from_iterator(it));   }

BOOST_INTRUSIVE_FORCEINLINE static const bstree &container_from_iterator(const_iterator it)
{  return static_cast<const bstree &>(Base::container_from_iterator(it));   }
};

#endif
} 
} 

#include <boost/intrusive/detail/config_end.hpp>

#endif 

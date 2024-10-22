#ifndef BOOST_INTRUSIVE_AVLTREE_HPP
#define BOOST_INTRUSIVE_AVLTREE_HPP

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/intrusive_fwd.hpp>
#include <cstddef>
#include <boost/intrusive/detail/minimal_less_equal_header.hpp>
#include <boost/intrusive/detail/minimal_pair_header.hpp>

#include <boost/static_assert.hpp>
#include <boost/intrusive/avl_set_hook.hpp>
#include <boost/intrusive/detail/avltree_node.hpp>
#include <boost/intrusive/bstree.hpp>
#include <boost/intrusive/detail/tree_node.hpp>
#include <boost/intrusive/detail/ebo_functor_holder.hpp>
#include <boost/intrusive/detail/mpl.hpp>
#include <boost/intrusive/pointer_traits.hpp>
#include <boost/intrusive/detail/get_value_traits.hpp>
#include <boost/intrusive/avltree_algorithms.hpp>
#include <boost/intrusive/link_mode.hpp>
#include <boost/move/utility_core.hpp>

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace intrusive {


struct default_avltree_hook_applier
{  template <class T> struct apply{ typedef typename T::default_avltree_hook type;  };  };

template<>
struct is_default_hook_tag<default_avltree_hook_applier>
{  static const bool value = true;  };

struct avltree_defaults
: bstree_defaults
{
typedef default_avltree_hook_applier proto_value_traits;
};


#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
template<class T, class ...Options>
#else
template<class ValueTraits, class VoidOrKeyOfValue, class VoidOrKeyComp, class SizeType, bool ConstantTimeSize, typename HeaderHolder>
#endif
class avltree_impl
:  public bstree_impl<ValueTraits, VoidOrKeyOfValue, VoidOrKeyComp, SizeType, ConstantTimeSize, AvlTreeAlgorithms, HeaderHolder>
{
public:
typedef ValueTraits value_traits;
typedef bstree_impl< ValueTraits, VoidOrKeyOfValue, VoidOrKeyComp, SizeType
, ConstantTimeSize, AvlTreeAlgorithms
, HeaderHolder>                                tree_type;
typedef tree_type                                                 implementation_defined;

typedef typename implementation_defined::pointer                  pointer;
typedef typename implementation_defined::const_pointer            const_pointer;
typedef typename implementation_defined::value_type               value_type;
typedef typename implementation_defined::key_type                 key_type;
typedef typename implementation_defined::key_of_value             key_of_value;
typedef typename implementation_defined::reference                reference;
typedef typename implementation_defined::const_reference          const_reference;
typedef typename implementation_defined::difference_type          difference_type;
typedef typename implementation_defined::size_type                size_type;
typedef typename implementation_defined::value_compare            value_compare;
typedef typename implementation_defined::key_compare              key_compare;
typedef typename implementation_defined::iterator                 iterator;
typedef typename implementation_defined::const_iterator           const_iterator;
typedef typename implementation_defined::reverse_iterator         reverse_iterator;
typedef typename implementation_defined::const_reverse_iterator   const_reverse_iterator;
typedef typename implementation_defined::node_traits              node_traits;
typedef typename implementation_defined::node                     node;
typedef typename implementation_defined::node_ptr                 node_ptr;
typedef typename implementation_defined::const_node_ptr           const_node_ptr;
typedef typename implementation_defined::node_algorithms          node_algorithms;

static const bool constant_time_size = implementation_defined::constant_time_size;
private:

BOOST_MOVABLE_BUT_NOT_COPYABLE(avltree_impl)


public:

typedef typename implementation_defined::insert_commit_data insert_commit_data;

avltree_impl()
:  tree_type()
{}

explicit avltree_impl( const key_compare &cmp, const value_traits &v_traits = value_traits())
:  tree_type(cmp, v_traits)
{}

template<class Iterator>
avltree_impl( bool unique, Iterator b, Iterator e
, const key_compare &cmp     = key_compare()
, const value_traits &v_traits = value_traits())
: tree_type(unique, b, e, cmp, v_traits)
{}

avltree_impl(BOOST_RV_REF(avltree_impl) x)
:  tree_type(BOOST_MOVE_BASE(tree_type, x))
{}

avltree_impl& operator=(BOOST_RV_REF(avltree_impl) x)
{  return static_cast<avltree_impl&>(tree_type::operator=(BOOST_MOVE_BASE(tree_type, x))); }

#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED

~avltree_impl();

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

static avltree_impl &container_from_end_iterator(iterator end_iterator);

static const avltree_impl &container_from_end_iterator(const_iterator end_iterator);

static avltree_impl &container_from_iterator(iterator it);

static const avltree_impl &container_from_iterator(const_iterator it);

key_compare key_comp() const;

value_compare value_comp() const;

bool empty() const;

size_type size() const;

void swap(avltree_impl& other);

template <class Cloner, class Disposer>
void clone_from(const avltree_impl &src, Cloner cloner, Disposer disposer);

#else 

using tree_type::clone_from;

#endif   

template <class Cloner, class Disposer>
void clone_from(BOOST_RV_REF(avltree_impl) src, Cloner cloner, Disposer disposer)
{  tree_type::clone_from(BOOST_MOVE_BASE(tree_type, src), cloner, disposer);  }

#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED

iterator insert_equal(reference value);

iterator insert_equal(const_iterator hint, reference value);

template<class Iterator>
void insert_equal(Iterator b, Iterator e);

std::pair<iterator, bool> insert_unique(reference value);

iterator insert_unique(const_iterator hint, reference value);

template<class KeyType, class KeyTypeKeyCompare>
std::pair<iterator, bool> insert_unique_check
(const KeyType &key, KeyTypeKeyCompare comp, insert_commit_data &commit_data);

template<class KeyType, class KeyTypeKeyCompare>
std::pair<iterator, bool> insert_unique_check
(const_iterator hint, const KeyType &key
,KeyTypeKeyCompare comp, insert_commit_data &commit_data);

std::pair<iterator, bool> insert_unique_check
(const key_type &key, insert_commit_data &commit_data);

std::pair<iterator, bool> insert_unique_check
(const_iterator hint, const key_type &key, insert_commit_data &commit_data);

iterator insert_unique_commit(reference value, const insert_commit_data &commit_data);

template<class Iterator>
void insert_unique(Iterator b, Iterator e);

iterator insert_before(const_iterator pos, reference value);

void push_back(reference value);

void push_front(reference value);

iterator erase(const_iterator i);

iterator erase(const_iterator b, const_iterator e);

size_type erase(const key_type &key);

template<class KeyType, class KeyTypeKeyCompare>
size_type erase(const KeyType& key, KeyTypeKeyCompare comp);

template<class Disposer>
iterator erase_and_dispose(const_iterator i, Disposer disposer);

template<class Disposer>
iterator erase_and_dispose(const_iterator b, const_iterator e, Disposer disposer);

template<class Disposer>
size_type erase_and_dispose(const key_type &key, Disposer disposer);

template<class KeyType, class KeyTypeKeyCompare, class Disposer>
size_type erase_and_dispose(const KeyType& key, KeyTypeKeyCompare comp, Disposer disposer);

void clear();

template<class Disposer>
void clear_and_dispose(Disposer disposer);

size_type count(const key_type &key) const;

template<class KeyType, class KeyTypeKeyCompare>
size_type count(const KeyType& key, KeyTypeKeyCompare comp) const;

iterator lower_bound(const key_type &key);

template<class KeyType, class KeyTypeKeyCompare>
iterator lower_bound(const KeyType& key, KeyTypeKeyCompare comp);

const_iterator lower_bound(const key_type &key) const;

template<class KeyType, class KeyTypeKeyCompare>
const_iterator lower_bound(const KeyType& key, KeyTypeKeyCompare comp) const;

iterator upper_bound(const key_type &key);

template<class KeyType, class KeyTypeKeyCompare>
iterator upper_bound(const KeyType& key, KeyTypeKeyCompare comp);

const_iterator upper_bound(const key_type &key) const;

template<class KeyType, class KeyTypeKeyCompare>
const_iterator upper_bound(const KeyType& key, KeyTypeKeyCompare comp) const;

iterator find(const key_type &key);

template<class KeyType, class KeyTypeKeyCompare>
iterator find(const KeyType& key, KeyTypeKeyCompare comp);

const_iterator find(const key_type &key) const;

template<class KeyType, class KeyTypeKeyCompare>
const_iterator find(const KeyType& key, KeyTypeKeyCompare comp) const;

std::pair<iterator,iterator> equal_range(const key_type &key);

template<class KeyType, class KeyTypeKeyCompare>
std::pair<iterator,iterator> equal_range(const KeyType& key, KeyTypeKeyCompare comp);

std::pair<const_iterator, const_iterator>
equal_range(const key_type &key) const;

template<class KeyType, class KeyTypeKeyCompare>
std::pair<const_iterator, const_iterator>
equal_range(const KeyType& key, KeyTypeKeyCompare comp) const;

std::pair<iterator,iterator> bounded_range
(const key_type &lower, const key_type &upper_key, bool left_closed, bool right_closed);

template<class KeyType, class KeyTypeKeyCompare>
std::pair<iterator,iterator> bounded_range
(const KeyType& lower_key, const KeyType& upper_key, KeyTypeKeyCompare comp, bool left_closed, bool right_closed);

std::pair<const_iterator, const_iterator>
bounded_range(const key_type &lower_key, const key_type &upper_key, bool left_closed, bool right_closed) const;

template<class KeyType, class KeyTypeKeyCompare>
std::pair<const_iterator, const_iterator> bounded_range
(const KeyType& lower_key, const KeyType& upper_key, KeyTypeKeyCompare comp, bool left_closed, bool right_closed) const;

static iterator s_iterator_to(reference value);

static const_iterator s_iterator_to(const_reference value);

iterator iterator_to(reference value);

const_iterator iterator_to(const_reference value) const;

static void init_node(reference value);

pointer unlink_leftmost_without_rebalance();

void replace_node(iterator replace_this, reference with_this);

void remove_node(reference value);

template<class T, class ...Options2>
void merge_unique(avltree<T, Options2...> &);

template<class T, class ...Options2>
void merge_equal(avltree<T, Options2...> &);

friend bool operator< (const avltree_impl &x, const avltree_impl &y);

friend bool operator==(const avltree_impl &x, const avltree_impl &y);

friend bool operator!= (const avltree_impl &x, const avltree_impl &y);

friend bool operator>(const avltree_impl &x, const avltree_impl &y);

friend bool operator<=(const avltree_impl &x, const avltree_impl &y);

friend bool operator>=(const avltree_impl &x, const avltree_impl &y);

friend void swap(avltree_impl &x, avltree_impl &y);
#endif   
};


#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED) || defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template<class T, class ...Options>
#else
template<class T, class O1 = void, class O2 = void
, class O3 = void, class O4 = void
, class O5 = void, class O6 = void>
#endif
struct make_avltree
{
typedef typename pack_options
< avltree_defaults,
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3, O4, O5, O6
#else
Options...
#endif
>::type packed_options;

typedef typename detail::get_value_traits
<T, typename packed_options::proto_value_traits>::type value_traits;

typedef avltree_impl
< value_traits
, typename packed_options::key_of_value
, typename packed_options::compare
, typename packed_options::size_type
, packed_options::constant_time_size
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
class avltree
:  public make_avltree<T,
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3, O4, O5, O6
#else
Options...
#endif
>::type
{
typedef typename make_avltree
<T,
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3, O4, O5, O6
#else
Options...
#endif
>::type   Base;
BOOST_MOVABLE_BUT_NOT_COPYABLE(avltree)

public:
typedef typename Base::key_compare        key_compare;
typedef typename Base::value_traits       value_traits;
typedef typename Base::iterator           iterator;
typedef typename Base::const_iterator     const_iterator;
typedef typename Base::reverse_iterator           reverse_iterator;
typedef typename Base::const_reverse_iterator     const_reverse_iterator;

BOOST_STATIC_ASSERT((detail::is_same<typename value_traits::value_type, T>::value));

BOOST_INTRUSIVE_FORCEINLINE avltree()
:  Base()
{}

BOOST_INTRUSIVE_FORCEINLINE explicit avltree( const key_compare &cmp, const value_traits &v_traits = value_traits())
:  Base(cmp, v_traits)
{}

template<class Iterator>
BOOST_INTRUSIVE_FORCEINLINE avltree( bool unique, Iterator b, Iterator e
, const key_compare &cmp = key_compare()
, const value_traits &v_traits = value_traits())
:  Base(unique, b, e, cmp, v_traits)
{}

BOOST_INTRUSIVE_FORCEINLINE avltree(BOOST_RV_REF(avltree) x)
:  Base(BOOST_MOVE_BASE(Base, x))
{}

BOOST_INTRUSIVE_FORCEINLINE avltree& operator=(BOOST_RV_REF(avltree) x)
{  return static_cast<avltree &>(this->Base::operator=(BOOST_MOVE_BASE(Base, x)));  }

template <class Cloner, class Disposer>
BOOST_INTRUSIVE_FORCEINLINE void clone_from(const avltree &src, Cloner cloner, Disposer disposer)
{  Base::clone_from(src, cloner, disposer);  }

template <class Cloner, class Disposer>
BOOST_INTRUSIVE_FORCEINLINE void clone_from(BOOST_RV_REF(avltree) src, Cloner cloner, Disposer disposer)
{  Base::clone_from(BOOST_MOVE_BASE(Base, src), cloner, disposer);  }

BOOST_INTRUSIVE_FORCEINLINE static avltree &container_from_end_iterator(iterator end_iterator)
{  return static_cast<avltree &>(Base::container_from_end_iterator(end_iterator));   }

BOOST_INTRUSIVE_FORCEINLINE static const avltree &container_from_end_iterator(const_iterator end_iterator)
{  return static_cast<const avltree &>(Base::container_from_end_iterator(end_iterator));   }

BOOST_INTRUSIVE_FORCEINLINE static avltree &container_from_iterator(iterator it)
{  return static_cast<avltree &>(Base::container_from_iterator(it));   }

BOOST_INTRUSIVE_FORCEINLINE static const avltree &container_from_iterator(const_iterator it)
{  return static_cast<const avltree &>(Base::container_from_iterator(it));   }
};

#endif

} 
} 

#include <boost/intrusive/detail/config_end.hpp>

#endif 

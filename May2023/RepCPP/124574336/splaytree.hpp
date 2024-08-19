#ifndef BOOST_INTRUSIVE_SPLAYTREE_HPP
#define BOOST_INTRUSIVE_SPLAYTREE_HPP

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/intrusive_fwd.hpp>
#include <cstddef>
#include <boost/intrusive/detail/minimal_less_equal_header.hpp>
#include <boost/intrusive/detail/minimal_pair_header.hpp>   

#include <boost/static_assert.hpp>
#include <boost/intrusive/bstree.hpp>
#include <boost/intrusive/detail/tree_node.hpp>
#include <boost/intrusive/detail/mpl.hpp>
#include <boost/intrusive/pointer_traits.hpp>
#include <boost/intrusive/detail/function_detector.hpp>
#include <boost/intrusive/detail/get_value_traits.hpp>
#include <boost/intrusive/splaytree_algorithms.hpp>
#include <boost/intrusive/link_mode.hpp>
#include <boost/intrusive/detail/key_nodeptr_comp.hpp>
#include <boost/move/utility_core.hpp>

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace intrusive {


struct splaytree_defaults
: bstree_defaults
{};


#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
template<class T, class ...Options>
#else
template<class ValueTraits, class VoidOrKeyOfValue, class VoidOrKeyComp, class SizeType, bool ConstantTimeSize, typename HeaderHolder>
#endif
class splaytree_impl
:  public bstree_impl<ValueTraits, VoidOrKeyOfValue, VoidOrKeyComp, SizeType, ConstantTimeSize, SplayTreeAlgorithms, HeaderHolder>
{
public:
typedef ValueTraits                                               value_traits;
typedef bstree_impl< ValueTraits, VoidOrKeyOfValue, VoidOrKeyComp, SizeType
, ConstantTimeSize, SplayTreeAlgorithms
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

BOOST_MOVABLE_BUT_NOT_COPYABLE(splaytree_impl)


public:

typedef typename implementation_defined::insert_commit_data insert_commit_data;

splaytree_impl()
:  tree_type()
{}

explicit splaytree_impl( const key_compare &cmp, const value_traits &v_traits = value_traits())
:  tree_type(cmp, v_traits)
{}

template<class Iterator>
splaytree_impl( bool unique, Iterator b, Iterator e
, const key_compare &cmp     = key_compare()
, const value_traits &v_traits = value_traits())
: tree_type(cmp, v_traits)
{
if(unique)
this->insert_unique(b, e);
else
this->insert_equal(b, e);
}

splaytree_impl(BOOST_RV_REF(splaytree_impl) x)
:  tree_type(BOOST_MOVE_BASE(tree_type, x))
{}

splaytree_impl& operator=(BOOST_RV_REF(splaytree_impl) x)
{  return static_cast<splaytree_impl&>(tree_type::operator=(BOOST_MOVE_BASE(tree_type, x))); }

#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED
~splaytree_impl();

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

static splaytree_impl &container_from_end_iterator(iterator end_iterator);

static const splaytree_impl &container_from_end_iterator(const_iterator end_iterator);

static splaytree_impl &container_from_iterator(iterator it);

static const splaytree_impl &container_from_iterator(const_iterator it);

key_compare key_comp() const;

value_compare value_comp() const;

bool empty() const;

size_type size() const;

void swap(splaytree_impl& other);

template <class Cloner, class Disposer>
void clone_from(const splaytree_impl &src, Cloner cloner, Disposer disposer);

#else 

using tree_type::clone_from;

#endif   

template <class Cloner, class Disposer>
void clone_from(BOOST_RV_REF(splaytree_impl) src, Cloner cloner, Disposer disposer)
{  tree_type::clone_from(BOOST_MOVE_BASE(tree_type, src), cloner, disposer);  }

#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED

iterator insert_equal(reference value);

iterator insert_equal(const_iterator hint, reference value);

template<class Iterator>
void insert_equal(Iterator b, Iterator e);

std::pair<iterator, bool> insert_unique(reference value);

iterator insert_unique(const_iterator hint, reference value);

std::pair<iterator, bool> insert_unique_check
(const key_type &key, insert_commit_data &commit_data);

std::pair<iterator, bool> insert_unique_check
(const_iterator hint, const key_type &key, insert_commit_data &commit_data);

template<class KeyType, class KeyTypeKeyCompare>
std::pair<iterator, bool> insert_unique_check
(const KeyType &key, KeyTypeKeyCompare comp, insert_commit_data &commit_data);

template<class KeyType, class KeyTypeKeyCompare>
std::pair<iterator, bool> insert_unique_check
(const_iterator hint, const KeyType &key
,KeyTypeKeyCompare comp, insert_commit_data &commit_data);

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

size_type count(const key_type &key);

template<class KeyType, class KeyTypeKeyCompare>
size_type count(const KeyType &key, KeyTypeKeyCompare comp);

size_type count(const key_type &key) const;

template<class KeyType, class KeyTypeKeyCompare>
size_type count(const KeyType &key, KeyTypeKeyCompare comp) const;

iterator lower_bound(const key_type &key);

const_iterator lower_bound(const key_type &key) const;

template<class KeyType, class KeyTypeKeyCompare>
iterator lower_bound(const KeyType &key, KeyTypeKeyCompare comp);

template<class KeyType, class KeyTypeKeyCompare>
const_iterator lower_bound(const KeyType &key, KeyTypeKeyCompare comp) const;

iterator upper_bound(const key_type &key);

const_iterator upper_bound(const key_type &key) const;

template<class KeyType, class KeyTypeKeyCompare>
iterator upper_bound(const KeyType &key, KeyTypeKeyCompare comp);

template<class KeyType, class KeyTypeKeyCompare>
const_iterator upper_bound(const KeyType &key, KeyTypeKeyCompare comp) const;

iterator find(const key_type &key);

const_iterator find(const key_type &key) const;

template<class KeyType, class KeyTypeKeyCompare>
iterator find(const KeyType &key, KeyTypeKeyCompare comp);

template<class KeyType, class KeyTypeKeyCompare>
const_iterator find(const KeyType &key, KeyTypeKeyCompare comp) const;

std::pair<iterator, iterator> equal_range(const key_type &key);

std::pair<const_iterator, const_iterator> equal_range(const key_type &key) const;

template<class KeyType, class KeyTypeKeyCompare>
std::pair<iterator, iterator> equal_range(const KeyType &key, KeyTypeKeyCompare comp);

template<class KeyType, class KeyTypeKeyCompare>
std::pair<const_iterator, const_iterator> equal_range(const KeyType &key, KeyTypeKeyCompare comp) const;

std::pair<iterator,iterator> bounded_range
(const key_type &lower_key, const key_type &upper_key, bool left_closed, bool right_closed);

template<class KeyType, class KeyTypeKeyCompare>
std::pair<iterator,iterator> bounded_range
(const KeyType& lower_key, const KeyType& upper_key, KeyTypeKeyCompare comp, bool left_closed, bool right_closed);

std::pair<const_iterator, const_iterator> bounded_range
(const key_type &lower_key, const key_type &upper_key, bool left_closed, bool right_closed) const;

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
void merge_unique(splaytree<T, Options2...> &);

template<class T, class ...Options2>
void merge_equal(splaytree<T, Options2...> &);

#endif   

void splay_up(iterator i)
{  return node_algorithms::splay_up(i.pointed_node(), tree_type::header_ptr());   }

template<class KeyType, class KeyTypeKeyCompare>
iterator splay_down(const KeyType &key, KeyTypeKeyCompare comp)
{
detail::key_nodeptr_comp<value_compare, value_traits>
key_node_comp(comp, &this->get_value_traits());
node_ptr r = node_algorithms::splay_down(tree_type::header_ptr(), key, key_node_comp);
return iterator(r, this->priv_value_traits_ptr());
}

iterator splay_down(const key_type &key)
{  return this->splay_down(key, this->key_comp());   }

#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED
void rebalance();

iterator rebalance_subtree(iterator root);

friend bool operator< (const splaytree_impl &x, const splaytree_impl &y);

friend bool operator==(const splaytree_impl &x, const splaytree_impl &y);

friend bool operator!= (const splaytree_impl &x, const splaytree_impl &y);

friend bool operator>(const splaytree_impl &x, const splaytree_impl &y);

friend bool operator<=(const splaytree_impl &x, const splaytree_impl &y);

friend bool operator>=(const splaytree_impl &x, const splaytree_impl &y);

friend void swap(splaytree_impl &x, splaytree_impl &y);

#endif   
};

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED) || defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template<class T, class ...Options>
#else
template<class T, class O1 = void, class O2 = void
, class O3 = void, class O4 = void
, class O5 = void, class O6 = void>
#endif
struct make_splaytree
{
typedef typename pack_options
< splaytree_defaults,
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3, O4, O5, O6
#else
Options...
#endif
>::type packed_options;

typedef typename detail::get_value_traits
<T, typename packed_options::proto_value_traits>::type value_traits;

typedef splaytree_impl
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
class splaytree
:  public make_splaytree<T,
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3, O4, O5, O6
#else
Options...
#endif
>::type
{
typedef typename make_splaytree
<T,
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3, O4, O5, O6
#else
Options...
#endif
>::type   Base;
BOOST_MOVABLE_BUT_NOT_COPYABLE(splaytree)

public:
typedef typename Base::key_compare        key_compare;
typedef typename Base::value_traits       value_traits;
typedef typename Base::iterator           iterator;
typedef typename Base::const_iterator     const_iterator;
typedef typename Base::reverse_iterator           reverse_iterator;
typedef typename Base::const_reverse_iterator     const_reverse_iterator;

BOOST_STATIC_ASSERT((detail::is_same<typename value_traits::value_type, T>::value));

BOOST_INTRUSIVE_FORCEINLINE splaytree()
:  Base()
{}

BOOST_INTRUSIVE_FORCEINLINE explicit splaytree( const key_compare &cmp, const value_traits &v_traits = value_traits())
:  Base(cmp, v_traits)
{}

template<class Iterator>
BOOST_INTRUSIVE_FORCEINLINE splaytree( bool unique, Iterator b, Iterator e
, const key_compare &cmp = key_compare()
, const value_traits &v_traits = value_traits())
:  Base(unique, b, e, cmp, v_traits)
{}

BOOST_INTRUSIVE_FORCEINLINE splaytree(BOOST_RV_REF(splaytree) x)
:  Base(BOOST_MOVE_BASE(Base, x))
{}

BOOST_INTRUSIVE_FORCEINLINE splaytree& operator=(BOOST_RV_REF(splaytree) x)
{  return static_cast<splaytree &>(this->Base::operator=(BOOST_MOVE_BASE(Base, x)));  }

template <class Cloner, class Disposer>
BOOST_INTRUSIVE_FORCEINLINE void clone_from(const splaytree &src, Cloner cloner, Disposer disposer)
{  Base::clone_from(src, cloner, disposer);  }

template <class Cloner, class Disposer>
BOOST_INTRUSIVE_FORCEINLINE void clone_from(BOOST_RV_REF(splaytree) src, Cloner cloner, Disposer disposer)
{  Base::clone_from(BOOST_MOVE_BASE(Base, src), cloner, disposer);  }

BOOST_INTRUSIVE_FORCEINLINE static splaytree &container_from_end_iterator(iterator end_iterator)
{  return static_cast<splaytree &>(Base::container_from_end_iterator(end_iterator));   }

BOOST_INTRUSIVE_FORCEINLINE static const splaytree &container_from_end_iterator(const_iterator end_iterator)
{  return static_cast<const splaytree &>(Base::container_from_end_iterator(end_iterator));   }

BOOST_INTRUSIVE_FORCEINLINE static splaytree &container_from_iterator(iterator it)
{  return static_cast<splaytree &>(Base::container_from_iterator(it));   }

BOOST_INTRUSIVE_FORCEINLINE static const splaytree &container_from_iterator(const_iterator it)
{  return static_cast<const splaytree &>(Base::container_from_iterator(it));   }
};

#endif

} 
} 

#include <boost/intrusive/detail/config_end.hpp>

#endif 

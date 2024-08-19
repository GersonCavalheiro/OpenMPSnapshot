#ifndef BOOST_INTRUSIVE_SPLAY_SET_HPP
#define BOOST_INTRUSIVE_SPLAY_SET_HPP

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/intrusive_fwd.hpp>
#include <boost/intrusive/splaytree.hpp>
#include <boost/intrusive/detail/mpl.hpp>
#include <boost/move/utility_core.hpp>
#include <boost/static_assert.hpp>

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#if !defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
template<class ValueTraits, class VoidOrKeyOfValue, class Compare, class SizeType, bool ConstantTimeSize, typename HeaderHolder>
class splay_multiset_impl;
#endif

namespace boost {
namespace intrusive {

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
template<class T, class ...Options>
#else
template<class ValueTraits, class VoidOrKeyOfValue, class Compare, class SizeType, bool ConstantTimeSize, typename HeaderHolder>
#endif
class splay_set_impl
#ifndef BOOST_INTRUSIVE_DOXYGEN_INVOKED
: public splaytree_impl<ValueTraits, VoidOrKeyOfValue, Compare, SizeType, ConstantTimeSize, HeaderHolder>
#endif
{
typedef splaytree_impl<ValueTraits, VoidOrKeyOfValue, Compare, SizeType, ConstantTimeSize, HeaderHolder> tree_type;
BOOST_MOVABLE_BUT_NOT_COPYABLE(splay_set_impl)

typedef tree_type implementation_defined;

public:
typedef typename implementation_defined::value_type               value_type;
typedef typename implementation_defined::key_type                 key_type;
typedef typename implementation_defined::key_of_value             key_of_value;
typedef typename implementation_defined::value_traits             value_traits;
typedef typename implementation_defined::pointer                  pointer;
typedef typename implementation_defined::const_pointer            const_pointer;
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
typedef typename implementation_defined::insert_commit_data       insert_commit_data;
typedef typename implementation_defined::node_traits              node_traits;
typedef typename implementation_defined::node                     node;
typedef typename implementation_defined::node_ptr                 node_ptr;
typedef typename implementation_defined::const_node_ptr           const_node_ptr;
typedef typename implementation_defined::node_algorithms          node_algorithms;

static const bool constant_time_size = tree_type::constant_time_size;

public:
splay_set_impl()
:  tree_type()
{}

explicit splay_set_impl( const key_compare &cmp, const value_traits &v_traits = value_traits())
:  tree_type(cmp, v_traits)
{}

template<class Iterator>
splay_set_impl( Iterator b, Iterator e
, const key_compare &cmp = key_compare()
, const value_traits &v_traits = value_traits())
: tree_type(true, b, e, cmp, v_traits)
{}

splay_set_impl(BOOST_RV_REF(splay_set_impl) x)
:  tree_type(BOOST_MOVE_BASE(tree_type, x))
{}

splay_set_impl& operator=(BOOST_RV_REF(splay_set_impl) x)
{  return static_cast<splay_set_impl&>(tree_type::operator=(BOOST_MOVE_BASE(tree_type, x))); }

#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED
~splay_set_impl();

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

static splay_set_impl &container_from_end_iterator(iterator end_iterator);

static const splay_set_impl &container_from_end_iterator(const_iterator end_iterator);

static splay_set_impl &container_from_iterator(iterator it);

static const splay_set_impl &container_from_iterator(const_iterator it);

key_compare key_comp() const;

value_compare value_comp() const;

bool empty() const;

size_type size() const;

void swap(splay_set_impl& other);

template <class Cloner, class Disposer>
void clone_from(const splay_set_impl &src, Cloner cloner, Disposer disposer);

#else

using tree_type::clone_from;

#endif   

template <class Cloner, class Disposer>
void clone_from(BOOST_RV_REF(splay_set_impl) src, Cloner cloner, Disposer disposer)
{  tree_type::clone_from(BOOST_MOVE_BASE(tree_type, src), cloner, disposer);  }

std::pair<iterator, bool> insert(reference value)
{  return tree_type::insert_unique(value);  }

iterator insert(const_iterator hint, reference value)
{  return tree_type::insert_unique(hint, value);  }

std::pair<iterator, bool> insert_check
(const key_type &key, insert_commit_data &commit_data)
{  return tree_type::insert_unique_check(key, commit_data); }

std::pair<iterator, bool> insert_check
(const_iterator hint, const key_type &key
,insert_commit_data &commit_data)
{  return tree_type::insert_unique_check(hint, key, commit_data); }

template<class KeyType, class KeyTypeKeyCompare>
std::pair<iterator, bool> insert_check
(const KeyType &key, KeyTypeKeyCompare comp, insert_commit_data &commit_data)
{  return tree_type::insert_unique_check(key, comp, commit_data); }

template<class KeyType, class KeyTypeKeyCompare>
std::pair<iterator, bool> insert_check
(const_iterator hint, const KeyType &key
,KeyTypeKeyCompare comp, insert_commit_data &commit_data)
{  return tree_type::insert_unique_check(hint, key, comp, commit_data); }

template<class Iterator>
void insert(Iterator b, Iterator e)
{  tree_type::insert_unique(b, e);  }

iterator insert_commit(reference value, const insert_commit_data &commit_data)
{  return tree_type::insert_unique_commit(value, commit_data);  }

#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED
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

#endif   

size_type count(const key_type &key) const
{  return static_cast<size_type>(this->tree_type::find(key) != this->tree_type::cend()); }

template<class KeyType, class KeyTypeKeyCompare>
size_type count(const KeyType& key, KeyTypeKeyCompare comp) const
{  return static_cast<size_type>(this->tree_type::find(key, comp) != this->tree_type::cend()); }

#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED

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

#endif   

std::pair<iterator,iterator> equal_range(const key_type &key)
{  return this->tree_type::lower_bound_range(key); }

template<class KeyType, class KeyTypeKeyCompare>
std::pair<iterator,iterator> equal_range(const KeyType& key, KeyTypeKeyCompare comp)
{  return this->tree_type::equal_range(key, comp); }

std::pair<const_iterator, const_iterator>
equal_range(const key_type &key) const
{  return this->tree_type::lower_bound_range(key); }

template<class KeyType, class KeyTypeKeyCompare>
std::pair<const_iterator, const_iterator>
equal_range(const KeyType& key, KeyTypeKeyCompare comp) const
{  return this->tree_type::equal_range(key, comp); }

#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED

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

void splay_up(iterator i);

template<class KeyType, class KeyTypeKeyCompare>
iterator splay_down(const KeyType &key, KeyTypeKeyCompare comp);

iterator splay_down(const key_type &key);

void rebalance();

iterator rebalance_subtree(iterator root);

template<class ...Options2>
void merge(splay_set<T, Options2...> &source);

template<class ...Options2>
void merge(splay_multiset<T, Options2...> &source);

#else

template<class Compare2>
void merge(splay_set_impl<ValueTraits, VoidOrKeyOfValue, Compare2, SizeType, ConstantTimeSize, HeaderHolder> &source)
{  return tree_type::merge_unique(source);  }


template<class Compare2>
void merge(splay_multiset_impl<ValueTraits, VoidOrKeyOfValue, Compare2, SizeType, ConstantTimeSize, HeaderHolder> &source)
{  return tree_type::merge_unique(source);  }

#endif   
};

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)

template<class T, class ...Options>
bool operator!= (const splay_set_impl<T, Options...> &x, const splay_set_impl<T, Options...> &y);

template<class T, class ...Options>
bool operator>(const splay_set_impl<T, Options...> &x, const splay_set_impl<T, Options...> &y);

template<class T, class ...Options>
bool operator<=(const splay_set_impl<T, Options...> &x, const splay_set_impl<T, Options...> &y);

template<class T, class ...Options>
bool operator>=(const splay_set_impl<T, Options...> &x, const splay_set_impl<T, Options...> &y);

template<class T, class ...Options>
void swap(splay_set_impl<T, Options...> &x, splay_set_impl<T, Options...> &y);

#endif   

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED) || defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template<class T, class ...Options>
#else
template<class T, class O1 = void, class O2 = void
, class O3 = void, class O4 = void
, class O5 = void, class O6 = void>
#endif
struct make_splay_set
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

typedef splay_set_impl
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
class splay_set
:  public make_splay_set<T,
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3, O4, O5, O6
#else
Options...
#endif
>::type
{
typedef typename make_splay_set
<T,
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3, O4, O5, O6
#else
Options...
#endif
>::type   Base;

BOOST_MOVABLE_BUT_NOT_COPYABLE(splay_set)
public:
typedef typename Base::key_compare        key_compare;
typedef typename Base::value_traits       value_traits;
typedef typename Base::iterator           iterator;
typedef typename Base::const_iterator     const_iterator;

BOOST_STATIC_ASSERT((detail::is_same<typename value_traits::value_type, T>::value));

BOOST_INTRUSIVE_FORCEINLINE splay_set()
:  Base()
{}

BOOST_INTRUSIVE_FORCEINLINE explicit splay_set( const key_compare &cmp, const value_traits &v_traits = value_traits())
:  Base(cmp, v_traits)
{}

template<class Iterator>
BOOST_INTRUSIVE_FORCEINLINE splay_set( Iterator b, Iterator e
, const key_compare &cmp = key_compare()
, const value_traits &v_traits = value_traits())
:  Base(b, e, cmp, v_traits)
{}

BOOST_INTRUSIVE_FORCEINLINE splay_set(BOOST_RV_REF(splay_set) x)
:  Base(::boost::move(static_cast<Base&>(x)))
{}

BOOST_INTRUSIVE_FORCEINLINE splay_set& operator=(BOOST_RV_REF(splay_set) x)
{  return static_cast<splay_set &>(this->Base::operator=(::boost::move(static_cast<Base&>(x))));  }

template <class Cloner, class Disposer>
BOOST_INTRUSIVE_FORCEINLINE void clone_from(const splay_set &src, Cloner cloner, Disposer disposer)
{  Base::clone_from(src, cloner, disposer);  }

template <class Cloner, class Disposer>
BOOST_INTRUSIVE_FORCEINLINE void clone_from(BOOST_RV_REF(splay_set) src, Cloner cloner, Disposer disposer)
{  Base::clone_from(BOOST_MOVE_BASE(Base, src), cloner, disposer);  }

BOOST_INTRUSIVE_FORCEINLINE static splay_set &container_from_end_iterator(iterator end_iterator)
{  return static_cast<splay_set &>(Base::container_from_end_iterator(end_iterator));   }

BOOST_INTRUSIVE_FORCEINLINE static const splay_set &container_from_end_iterator(const_iterator end_iterator)
{  return static_cast<const splay_set &>(Base::container_from_end_iterator(end_iterator));   }

BOOST_INTRUSIVE_FORCEINLINE static splay_set &container_from_iterator(iterator it)
{  return static_cast<splay_set &>(Base::container_from_iterator(it));   }

BOOST_INTRUSIVE_FORCEINLINE static const splay_set &container_from_iterator(const_iterator it)
{  return static_cast<const splay_set &>(Base::container_from_iterator(it));   }
};

#endif

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
template<class T, class ...Options>
#else
template<class ValueTraits, class VoidOrKeyOfValue, class Compare, class SizeType, bool ConstantTimeSize, typename HeaderHolder>
#endif
class splay_multiset_impl
#ifndef BOOST_INTRUSIVE_DOXYGEN_INVOKED
: public splaytree_impl<ValueTraits, VoidOrKeyOfValue, Compare, SizeType, ConstantTimeSize, HeaderHolder>
#endif
{
typedef splaytree_impl<ValueTraits, VoidOrKeyOfValue, Compare, SizeType, ConstantTimeSize, HeaderHolder> tree_type;

BOOST_MOVABLE_BUT_NOT_COPYABLE(splay_multiset_impl)
typedef tree_type implementation_defined;

public:
typedef typename implementation_defined::value_type               value_type;
typedef typename implementation_defined::key_type                 key_type;
typedef typename implementation_defined::key_of_value             key_of_value;
typedef typename implementation_defined::value_traits             value_traits;
typedef typename implementation_defined::pointer                  pointer;
typedef typename implementation_defined::const_pointer            const_pointer;
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
typedef typename implementation_defined::insert_commit_data       insert_commit_data;
typedef typename implementation_defined::node_traits              node_traits;
typedef typename implementation_defined::node                     node;
typedef typename implementation_defined::node_ptr                 node_ptr;
typedef typename implementation_defined::const_node_ptr           const_node_ptr;
typedef typename implementation_defined::node_algorithms          node_algorithms;

static const bool constant_time_size = tree_type::constant_time_size;

public:
splay_multiset_impl()
:  tree_type()
{}

explicit splay_multiset_impl(const key_compare &cmp, const value_traits &v_traits = value_traits())
:  tree_type(cmp, v_traits)
{}

template<class Iterator>
splay_multiset_impl( Iterator b, Iterator e
, const key_compare &cmp = key_compare()
, const value_traits &v_traits = value_traits())
: tree_type(false, b, e, cmp, v_traits)
{}

splay_multiset_impl(BOOST_RV_REF(splay_multiset_impl) x)
:  tree_type(::boost::move(static_cast<tree_type&>(x)))
{}

splay_multiset_impl& operator=(BOOST_RV_REF(splay_multiset_impl) x)
{  return static_cast<splay_multiset_impl&>(tree_type::operator=(::boost::move(static_cast<tree_type&>(x)))); }

#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED
~splay_multiset_impl();

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

static splay_multiset_impl &container_from_end_iterator(iterator end_iterator);

static const splay_multiset_impl &container_from_end_iterator(const_iterator end_iterator);

static splay_multiset_impl &container_from_iterator(iterator it);

static const splay_multiset_impl &container_from_iterator(const_iterator it);

key_compare key_comp() const;

value_compare value_comp() const;

bool empty() const;

size_type size() const;

void swap(splay_multiset_impl& other);

template <class Cloner, class Disposer>
void clone_from(const splay_multiset_impl &src, Cloner cloner, Disposer disposer);

#else

using tree_type::clone_from;

#endif   

template <class Cloner, class Disposer>
void clone_from(BOOST_RV_REF(splay_multiset_impl) src, Cloner cloner, Disposer disposer)
{  tree_type::clone_from(BOOST_MOVE_BASE(tree_type, src), cloner, disposer);  }

iterator insert(reference value)
{  return tree_type::insert_equal(value);  }

iterator insert(const_iterator hint, reference value)
{  return tree_type::insert_equal(hint, value);  }

template<class Iterator>
void insert(Iterator b, Iterator e)
{  tree_type::insert_equal(b, e);  }

#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED
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

size_type count(const key_type&);

template<class KeyType, class KeyTypeKeyCompare>
size_type count(const KeyType& key, KeyTypeKeyCompare comp);

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
(const_reference lower_value, const_reference upper_value, bool left_closed, bool right_closed);

template<class KeyType, class KeyTypeKeyCompare>
std::pair<iterator,iterator> bounded_range
(const KeyType& lower_key, const KeyType& upper_key, KeyTypeKeyCompare comp, bool left_closed, bool right_closed);

std::pair<const_iterator, const_iterator> bounded_range
(const_reference lower_value, const_reference upper_value, bool left_closed, bool right_closed) const;

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

void splay_up(iterator i);

template<class KeyType, class KeyTypeKeyCompare>
iterator splay_down(const KeyType &key, KeyTypeKeyCompare comp);

iterator splay_down(const key_type &key);

void rebalance();

iterator rebalance_subtree(iterator root);

template<class ...Options2>
void merge(splay_multiset<T, Options2...> &source);

template<class ...Options2>
void merge(splay_set<T, Options2...> &source);

#else

template<class Compare2>
void merge(splay_multiset_impl<ValueTraits, VoidOrKeyOfValue, Compare2, SizeType, ConstantTimeSize, HeaderHolder> &source)
{  return tree_type::merge_equal(source);  }

template<class Compare2>
void merge(splay_set_impl<ValueTraits, VoidOrKeyOfValue, Compare2, SizeType, ConstantTimeSize, HeaderHolder> &source)
{  return tree_type::merge_equal(source);  }

#endif   
};

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)

template<class T, class ...Options>
bool operator!= (const splay_multiset_impl<T, Options...> &x, const splay_multiset_impl<T, Options...> &y);

template<class T, class ...Options>
bool operator>(const splay_multiset_impl<T, Options...> &x, const splay_multiset_impl<T, Options...> &y);

template<class T, class ...Options>
bool operator<=(const splay_multiset_impl<T, Options...> &x, const splay_multiset_impl<T, Options...> &y);

template<class T, class ...Options>
bool operator>=(const splay_multiset_impl<T, Options...> &x, const splay_multiset_impl<T, Options...> &y);

template<class T, class ...Options>
void swap(splay_multiset_impl<T, Options...> &x, splay_multiset_impl<T, Options...> &y);

#endif   

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED) || defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template<class T, class ...Options>
#else
template<class T, class O1 = void, class O2 = void
, class O3 = void, class O4 = void
, class O5 = void, class O6 = void>
#endif
struct make_splay_multiset
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

typedef splay_multiset_impl
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
class splay_multiset
:  public make_splay_multiset<T,
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3, O4, O5, O6
#else
Options...
#endif
>::type
{
typedef typename make_splay_multiset<T,
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3, O4, O5, O6
#else
Options...
#endif
>::type   Base;

BOOST_MOVABLE_BUT_NOT_COPYABLE(splay_multiset)

public:
typedef typename Base::key_compare        key_compare;
typedef typename Base::value_traits       value_traits;
typedef typename Base::iterator           iterator;
typedef typename Base::const_iterator     const_iterator;

BOOST_STATIC_ASSERT((detail::is_same<typename value_traits::value_type, T>::value));

BOOST_INTRUSIVE_FORCEINLINE splay_multiset()
:  Base()
{}

BOOST_INTRUSIVE_FORCEINLINE explicit splay_multiset( const key_compare &cmp, const value_traits &v_traits = value_traits())
:  Base(cmp, v_traits)
{}

template<class Iterator>
BOOST_INTRUSIVE_FORCEINLINE splay_multiset( Iterator b, Iterator e
, const key_compare &cmp = key_compare()
, const value_traits &v_traits = value_traits())
:  Base(b, e, cmp, v_traits)
{}

BOOST_INTRUSIVE_FORCEINLINE splay_multiset(BOOST_RV_REF(splay_multiset) x)
:  Base(::boost::move(static_cast<Base&>(x)))
{}

BOOST_INTRUSIVE_FORCEINLINE splay_multiset& operator=(BOOST_RV_REF(splay_multiset) x)
{  return static_cast<splay_multiset &>(this->Base::operator=(::boost::move(static_cast<Base&>(x))));  }

template <class Cloner, class Disposer>
BOOST_INTRUSIVE_FORCEINLINE void clone_from(const splay_multiset &src, Cloner cloner, Disposer disposer)
{  Base::clone_from(src, cloner, disposer);  }

template <class Cloner, class Disposer>
BOOST_INTRUSIVE_FORCEINLINE void clone_from(BOOST_RV_REF(splay_multiset) src, Cloner cloner, Disposer disposer)
{  Base::clone_from(BOOST_MOVE_BASE(Base, src), cloner, disposer);  }

BOOST_INTRUSIVE_FORCEINLINE static splay_multiset &container_from_end_iterator(iterator end_iterator)
{  return static_cast<splay_multiset &>(Base::container_from_end_iterator(end_iterator));   }

BOOST_INTRUSIVE_FORCEINLINE static const splay_multiset &container_from_end_iterator(const_iterator end_iterator)
{  return static_cast<const splay_multiset &>(Base::container_from_end_iterator(end_iterator));   }

BOOST_INTRUSIVE_FORCEINLINE static splay_multiset &container_from_iterator(iterator it)
{  return static_cast<splay_multiset &>(Base::container_from_iterator(it));   }

BOOST_INTRUSIVE_FORCEINLINE static const splay_multiset &container_from_iterator(const_iterator it)
{  return static_cast<const splay_multiset &>(Base::container_from_iterator(it));   }
};

#endif

} 
} 

#include <boost/intrusive/detail/config_end.hpp>

#endif 

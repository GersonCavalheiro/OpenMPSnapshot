
#ifndef BOOST_INTRUSIVE_SGTREE_HPP
#define BOOST_INTRUSIVE_SGTREE_HPP

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/intrusive_fwd.hpp>
#include <boost/intrusive/detail/assert.hpp>
#include <boost/static_assert.hpp>
#include <boost/intrusive/bs_set_hook.hpp>
#include <boost/intrusive/bstree.hpp>
#include <boost/intrusive/detail/tree_node.hpp>
#include <boost/intrusive/pointer_traits.hpp>
#include <boost/intrusive/detail/mpl.hpp>
#include <boost/intrusive/detail/math.hpp>
#include <boost/intrusive/detail/get_value_traits.hpp>
#include <boost/intrusive/sgtree_algorithms.hpp>
#include <boost/intrusive/detail/key_nodeptr_comp.hpp>
#include <boost/intrusive/link_mode.hpp>

#include <boost/move/utility_core.hpp>
#include <boost/move/adl_move_swap.hpp>

#include <cstddef>
#include <boost/intrusive/detail/minimal_less_equal_header.hpp>
#include <boost/intrusive/detail/minimal_pair_header.hpp>   
#include <cmath>
#include <cstddef>

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace intrusive {


namespace detail{


inline std::size_t calculate_h_sqrt2 (std::size_t n)
{
std::size_t f_log2 = detail::floor_log2(n);
return (2*f_log2) + static_cast<std::size_t>(n >= detail::sqrt2_pow_2xplus1(f_log2));
}

struct h_alpha_sqrt2_t
{
h_alpha_sqrt2_t(void){}
std::size_t operator()(std::size_t n) const
{  return calculate_h_sqrt2(n);  }
};

struct alpha_0_75_by_max_size_t
{
alpha_0_75_by_max_size_t(void){}

std::size_t operator()(std::size_t max_tree_size) const
{
const std::size_t max_tree_size_limit = ((~std::size_t(0))/std::size_t(3));
return max_tree_size > max_tree_size_limit ? max_tree_size/4*3 : max_tree_size*3/4;
}
};


struct h_alpha_t
{
explicit h_alpha_t(float inv_minus_logalpha)
:  inv_minus_logalpha_(inv_minus_logalpha)
{}

std::size_t operator()(std::size_t n) const
{
return static_cast<std::size_t>(detail::fast_log2(float(n))*inv_minus_logalpha_);
}

private:
float inv_minus_logalpha_;
};

struct alpha_by_max_size_t
{
explicit alpha_by_max_size_t(float alpha)
:  alpha_(alpha)
{}

float operator()(std::size_t max_tree_size) const
{  return float(max_tree_size)*alpha_;   }

private:
float alpha_;
};

template<bool Activate, class SizeType>
struct alpha_holder
{
typedef boost::intrusive::detail::h_alpha_t           h_alpha_t;
typedef boost::intrusive::detail::alpha_by_max_size_t multiply_by_alpha_t;

alpha_holder()
: max_tree_size_()
{  set_alpha(0.70711f);   } 

float get_alpha() const
{  return alpha_;  }

void set_alpha(float alpha)
{
alpha_ = alpha;
inv_minus_logalpha_ = 1/(-detail::fast_log2(alpha));
}

h_alpha_t get_h_alpha_t() const
{  return h_alpha_t(inv_minus_logalpha_);  }

multiply_by_alpha_t get_multiply_by_alpha_t() const
{  return multiply_by_alpha_t(alpha_);  }

SizeType &get_max_tree_size()
{  return max_tree_size_;  }

protected:
float alpha_;
float inv_minus_logalpha_;
SizeType max_tree_size_;
};

template<class SizeType>
struct alpha_holder<false, SizeType>
{
typedef boost::intrusive::detail::h_alpha_sqrt2_t           h_alpha_t;
typedef boost::intrusive::detail::alpha_0_75_by_max_size_t  multiply_by_alpha_t;

alpha_holder()
: max_tree_size_()
{}

float get_alpha() const
{  return 0.70710677f;  }

void set_alpha(float)
{  
BOOST_INTRUSIVE_INVARIANT_ASSERT(0);
}

h_alpha_t get_h_alpha_t() const
{  return h_alpha_t();  }

multiply_by_alpha_t get_multiply_by_alpha_t() const
{  return multiply_by_alpha_t();  }

SizeType &get_max_tree_size()
{  return max_tree_size_;  }

protected:
SizeType max_tree_size_;
};

}  

struct sgtree_defaults
: bstree_defaults
{
static const bool floating_point = true;
};


#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
template<class T, class ...Options>
#else
template<class ValueTraits, class VoidOrKeyOfValue, class VoidOrKeyComp, class SizeType, bool FloatingPoint, typename HeaderHolder>
#endif
class sgtree_impl
:  public bstree_impl<ValueTraits, VoidOrKeyOfValue, VoidOrKeyComp, SizeType, true, SgTreeAlgorithms, HeaderHolder>
,  public detail::alpha_holder<FloatingPoint, SizeType>
{
public:
typedef ValueTraits                                               value_traits;
typedef bstree_impl< ValueTraits, VoidOrKeyOfValue, VoidOrKeyComp, SizeType
, true, SgTreeAlgorithms, HeaderHolder>        tree_type;
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
typedef BOOST_INTRUSIVE_IMPDEF(sgtree_algorithms<node_traits>)    node_algorithms;

static const bool constant_time_size      = implementation_defined::constant_time_size;
static const bool floating_point          = FloatingPoint;
static const bool stateful_value_traits   = implementation_defined::stateful_value_traits;

private:

typedef detail::alpha_holder<FloatingPoint, SizeType>    alpha_traits;
typedef typename alpha_traits::h_alpha_t                 h_alpha_t;
typedef typename alpha_traits::multiply_by_alpha_t       multiply_by_alpha_t;

BOOST_MOVABLE_BUT_NOT_COPYABLE(sgtree_impl)
BOOST_STATIC_ASSERT(((int)value_traits::link_mode != (int)auto_unlink));

enum { safemode_or_autounlink  =
(int)value_traits::link_mode == (int)auto_unlink   ||
(int)value_traits::link_mode == (int)safe_link     };


public:

typedef BOOST_INTRUSIVE_IMPDEF(typename node_algorithms::insert_commit_data) insert_commit_data;

sgtree_impl()
:  tree_type()
{}

explicit sgtree_impl( const key_compare &cmp, const value_traits &v_traits = value_traits())
:  tree_type(cmp, v_traits)
{}

template<class Iterator>
sgtree_impl( bool unique, Iterator b, Iterator e
, const key_compare &cmp     = key_compare()
, const value_traits &v_traits = value_traits())
: tree_type(cmp, v_traits)
{
if(unique)
this->insert_unique(b, e);
else
this->insert_equal(b, e);
}

sgtree_impl(BOOST_RV_REF(sgtree_impl) x)
:  tree_type(BOOST_MOVE_BASE(tree_type, x)), alpha_traits(x.get_alpha_traits())
{  ::boost::adl_move_swap(this->get_alpha_traits(), x.get_alpha_traits());   }

sgtree_impl& operator=(BOOST_RV_REF(sgtree_impl) x)
{
this->get_alpha_traits() = x.get_alpha_traits();
return static_cast<sgtree_impl&>(tree_type::operator=(BOOST_MOVE_BASE(tree_type, x)));
}

private:

const alpha_traits &get_alpha_traits() const
{  return *this;  }

alpha_traits &get_alpha_traits()
{  return *this;  }

h_alpha_t get_h_alpha_func() const
{  return this->get_alpha_traits().get_h_alpha_t();  }

multiply_by_alpha_t get_alpha_by_max_size_func() const
{  return this->get_alpha_traits().get_multiply_by_alpha_t(); }


public:

#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED
~sgtree_impl();

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

static sgtree_impl &container_from_end_iterator(iterator end_iterator);

static const sgtree_impl &container_from_end_iterator(const_iterator end_iterator);

static sgtree_impl &container_from_iterator(iterator it);

static const sgtree_impl &container_from_iterator(const_iterator it);

key_compare key_comp() const;

value_compare value_comp() const;

bool empty() const;

size_type size() const;

#endif   

void swap(sgtree_impl& other)
{
this->tree_type::swap(static_cast<tree_type&>(other));
::boost::adl_move_swap(this->get_alpha_traits(), other.get_alpha_traits());
}

template <class Cloner, class Disposer>
void clone_from(const sgtree_impl &src, Cloner cloner, Disposer disposer)
{
tree_type::clone_from(src, cloner, disposer);
this->get_alpha_traits() = src.get_alpha_traits();
}

template <class Cloner, class Disposer>
void clone_from(BOOST_RV_REF(sgtree_impl) src, Cloner cloner, Disposer disposer)
{
tree_type::clone_from(BOOST_MOVE_BASE(tree_type, src), cloner, disposer);
this->get_alpha_traits() = ::boost::move(src.get_alpha_traits());
}

iterator insert_equal(reference value)
{
node_ptr to_insert(this->get_value_traits().to_node_ptr(value));
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || node_algorithms::unique(to_insert));
std::size_t max_tree_size = (std::size_t)this->max_tree_size_;
node_ptr p = node_algorithms::insert_equal_upper_bound
(this->tree_type::header_ptr(), to_insert, this->key_node_comp(this->key_comp())
, (size_type)this->size(), this->get_h_alpha_func(), max_tree_size);
this->tree_type::sz_traits().increment();
this->max_tree_size_ = (size_type)max_tree_size;
return iterator(p, this->priv_value_traits_ptr());
}

iterator insert_equal(const_iterator hint, reference value)
{
node_ptr to_insert(this->get_value_traits().to_node_ptr(value));
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || node_algorithms::unique(to_insert));
std::size_t max_tree_size = (std::size_t)this->max_tree_size_;
node_ptr p = node_algorithms::insert_equal
( this->tree_type::header_ptr(), hint.pointed_node(), to_insert, this->key_node_comp(this->key_comp())
, (std::size_t)this->size(), this->get_h_alpha_func(), max_tree_size);
this->tree_type::sz_traits().increment();
this->max_tree_size_ = (size_type)max_tree_size;
return iterator(p, this->priv_value_traits_ptr());
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
std::pair<iterator, bool> ret = this->insert_unique_check
(key_of_value()(value), this->key_comp(), commit_data);
if(!ret.second)
return ret;
return std::pair<iterator, bool> (this->insert_unique_commit(value, commit_data), true);
}

iterator insert_unique(const_iterator hint, reference value)
{
insert_commit_data commit_data;
std::pair<iterator, bool> ret = this->insert_unique_check
(hint, key_of_value()(value), this->key_comp(), commit_data);
if(!ret.second)
return ret.first;
return this->insert_unique_commit(value, commit_data);
}

template<class KeyType, class KeyTypeKeyCompare>
BOOST_INTRUSIVE_DOC1ST(std::pair<iterator BOOST_INTRUSIVE_I bool>
, typename detail::disable_if_convertible
<KeyType BOOST_INTRUSIVE_I const_iterator BOOST_INTRUSIVE_I 
std::pair<iterator BOOST_INTRUSIVE_I bool> >::type)
insert_unique_check
(const KeyType &key, KeyTypeKeyCompare comp, insert_commit_data &commit_data)
{
std::pair<node_ptr, bool> ret =
node_algorithms::insert_unique_check
(this->tree_type::header_ptr(), key, this->key_node_comp(comp), commit_data);
return std::pair<iterator, bool>(iterator(ret.first, this->priv_value_traits_ptr()), ret.second);
}

template<class KeyType, class KeyTypeKeyCompare>
std::pair<iterator, bool> insert_unique_check
(const_iterator hint, const KeyType &key
,KeyTypeKeyCompare comp, insert_commit_data &commit_data)
{
std::pair<node_ptr, bool> ret =
node_algorithms::insert_unique_check
(this->tree_type::header_ptr(), hint.pointed_node(), key, this->key_node_comp(comp), commit_data);
return std::pair<iterator, bool>(iterator(ret.first, this->priv_value_traits_ptr()), ret.second);
}

std::pair<iterator, bool> insert_unique_check
(const key_type &key, insert_commit_data &commit_data)
{  return this->insert_unique_check(key, this->key_comp(), commit_data);   }

std::pair<iterator, bool> insert_unique_check
(const_iterator hint, const key_type &key, insert_commit_data &commit_data)
{  return this->insert_unique_check(hint, key, this->key_comp(), commit_data);   }

iterator insert_unique_commit(reference value, const insert_commit_data &commit_data)
{
node_ptr to_insert(this->get_value_traits().to_node_ptr(value));
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || node_algorithms::unique(to_insert));
std::size_t max_tree_size = (std::size_t)this->max_tree_size_;
node_algorithms::insert_unique_commit
( this->tree_type::header_ptr(), to_insert, commit_data
, (std::size_t)this->size(), this->get_h_alpha_func(), max_tree_size);
this->tree_type::sz_traits().increment();
this->max_tree_size_ = (size_type)max_tree_size;
return iterator(to_insert, this->priv_value_traits_ptr());
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

iterator insert_before(const_iterator pos, reference value)
{
node_ptr to_insert(this->get_value_traits().to_node_ptr(value));
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || node_algorithms::unique(to_insert));
std::size_t max_tree_size = (std::size_t)this->max_tree_size_;
node_ptr p = node_algorithms::insert_before
( this->tree_type::header_ptr(), pos.pointed_node(), to_insert
, (size_type)this->size(), this->get_h_alpha_func(), max_tree_size);
this->tree_type::sz_traits().increment();
this->max_tree_size_ = (size_type)max_tree_size;
return iterator(p, this->priv_value_traits_ptr());
}

void push_back(reference value)
{
node_ptr to_insert(this->get_value_traits().to_node_ptr(value));
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || node_algorithms::unique(to_insert));
std::size_t max_tree_size = (std::size_t)this->max_tree_size_;
node_algorithms::push_back
( this->tree_type::header_ptr(), to_insert
, (size_type)this->size(), this->get_h_alpha_func(), max_tree_size);
this->tree_type::sz_traits().increment();
this->max_tree_size_ = (size_type)max_tree_size;
}

void push_front(reference value)
{
node_ptr to_insert(this->get_value_traits().to_node_ptr(value));
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || node_algorithms::unique(to_insert));
std::size_t max_tree_size = (std::size_t)this->max_tree_size_;
node_algorithms::push_front
( this->tree_type::header_ptr(), to_insert
, (size_type)this->size(), this->get_h_alpha_func(), max_tree_size);
this->tree_type::sz_traits().increment();
this->max_tree_size_ = (size_type)max_tree_size;
}


iterator erase(const_iterator i)
{
const_iterator ret(i);
++ret;
node_ptr to_erase(i.pointed_node());
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || !node_algorithms::unique(to_erase));
std::size_t max_tree_size = this->max_tree_size_;
node_algorithms::erase
( this->tree_type::header_ptr(), to_erase, (std::size_t)this->size()
, max_tree_size, this->get_alpha_by_max_size_func());
this->max_tree_size_ = (size_type)max_tree_size;
this->tree_type::sz_traits().decrement();
if(safemode_or_autounlink)
node_algorithms::init(to_erase);
return ret.unconst();
}

iterator erase(const_iterator b, const_iterator e)
{  size_type n;   return private_erase(b, e, n);   }

size_type erase(const key_type &key)
{  return this->erase(key, this->key_comp());   }

template<class KeyType, class KeyTypeKeyCompare>
BOOST_INTRUSIVE_DOC1ST(size_type
, typename detail::disable_if_convertible<KeyTypeKeyCompare BOOST_INTRUSIVE_I const_iterator BOOST_INTRUSIVE_I size_type>::type)
erase(const KeyType& key, KeyTypeKeyCompare comp)
{
std::pair<iterator,iterator> p = this->equal_range(key, comp);
size_type n;
private_erase(p.first, p.second, n);
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

#if !defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
template<class Disposer>
iterator erase_and_dispose(iterator i, Disposer disposer)
{  return this->erase_and_dispose(const_iterator(i), disposer);   }
#endif

template<class Disposer>
iterator erase_and_dispose(const_iterator b, const_iterator e, Disposer disposer)
{  size_type n;   return private_erase(b, e, n, disposer);   }

template<class Disposer>
size_type erase_and_dispose(const key_type &key, Disposer disposer)
{
std::pair<iterator,iterator> p = this->equal_range(key);
size_type n;
private_erase(p.first, p.second, n, disposer);
return n;
}

template<class KeyType, class KeyTypeKeyCompare, class Disposer>
BOOST_INTRUSIVE_DOC1ST(size_type
, typename detail::disable_if_convertible<KeyTypeKeyCompare BOOST_INTRUSIVE_I const_iterator BOOST_INTRUSIVE_I size_type>::type)
erase_and_dispose(const KeyType& key, KeyTypeKeyCompare comp, Disposer disposer)
{
std::pair<iterator,iterator> p = this->equal_range(key, comp);
size_type n;
private_erase(p.first, p.second, n, disposer);
return n;
}

void clear()
{
tree_type::clear();
this->max_tree_size_ = 0;
}

template<class Disposer>
void clear_and_dispose(Disposer disposer)
{
tree_type::clear_and_dispose(disposer);
this->max_tree_size_ = 0;
}

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
template<class T, class ...Options2> void merge_unique(sgtree<T, Options2...> &);
#else
template<class Compare2>
void merge_unique(sgtree_impl
<ValueTraits, VoidOrKeyOfValue, Compare2, SizeType, FloatingPoint, HeaderHolder> &source)
#endif
{
node_ptr it   (node_algorithms::begin_node(source.header_ptr()))
, itend(node_algorithms::end_node  (source.header_ptr()));

while(it != itend){
node_ptr const p(it);
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || !node_algorithms::unique(p));
it = node_algorithms::next_node(it);

std::size_t max_tree1_size = this->max_tree_size_;
std::size_t max_tree2_size = source.get_max_tree_size();
if( node_algorithms::transfer_unique
( this->header_ptr(), this->key_node_comp(this->key_comp()), this->size(), max_tree1_size
, source.header_ptr(), p, source.size(), max_tree2_size
, this->get_h_alpha_func(), this->get_alpha_by_max_size_func()) ){
this->max_tree_size_  = (size_type)max_tree1_size;
this->sz_traits().increment();
source.get_max_tree_size() = (size_type)max_tree2_size;
source.sz_traits().decrement();
}
}
}

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
template<class T, class ...Options2> void merge_equal(sgtree<T, Options2...> &);
#else
template<class Compare2>
void merge_equal(sgtree_impl
<ValueTraits, VoidOrKeyOfValue, Compare2, SizeType, FloatingPoint, HeaderHolder> &source)
#endif
{
node_ptr it   (node_algorithms::begin_node(source.header_ptr()))
, itend(node_algorithms::end_node  (source.header_ptr()));

while(it != itend){
node_ptr const p(it);
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || !node_algorithms::unique(p));
it = node_algorithms::next_node(it);
std::size_t max_tree1_size = this->max_tree_size_;
std::size_t max_tree2_size = source.get_max_tree_size();
node_algorithms::transfer_equal
( this->header_ptr(), this->key_node_comp(this->key_comp()), this->size(), max_tree1_size
, source.header_ptr(), p, source.size(), max_tree2_size
, this->get_h_alpha_func(), this->get_alpha_by_max_size_func());
this->max_tree_size_  = (size_type)max_tree1_size;
this->sz_traits().increment();
source.get_max_tree_size() = (size_type)max_tree2_size;
source.sz_traits().decrement();
}
}

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

std::pair<iterator,iterator> equal_range(const key_type &key);

template<class KeyType, class KeyTypeKeyCompare>
std::pair<iterator,iterator> equal_range(const KeyType& key, KeyTypeKeyCompare comp);

std::pair<const_iterator, const_iterator>
equal_range(const key_type &key) const;

template<class KeyType, class KeyTypeKeyCompare>
std::pair<const_iterator, const_iterator>
equal_range(const KeyType& key, KeyTypeKeyCompare comp) const;

std::pair<iterator,iterator> bounded_range
(const key_type &lower_key, const key_type &upper_key, bool left_closed, bool right_closed);

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

void rebalance();

iterator rebalance_subtree(iterator root);

friend bool operator< (const sgtree_impl &x, const sgtree_impl &y);

friend bool operator==(const sgtree_impl &x, const sgtree_impl &y);

friend bool operator!= (const sgtree_impl &x, const sgtree_impl &y);

friend bool operator>(const sgtree_impl &x, const sgtree_impl &y);

friend bool operator<=(const sgtree_impl &x, const sgtree_impl &y);

friend bool operator>=(const sgtree_impl &x, const sgtree_impl &y);

friend void swap(sgtree_impl &x, sgtree_impl &y);

#endif   

float balance_factor() const
{  return this->get_alpha_traits().get_alpha(); }

void balance_factor(float new_alpha)
{
BOOST_STATIC_ASSERT((floating_point));
BOOST_INTRUSIVE_INVARIANT_ASSERT((new_alpha > 0.5f && new_alpha < 1.0f));
if(new_alpha >= 0.5f && new_alpha < 1.0f){
float old_alpha = this->get_alpha_traits().get_alpha();
this->get_alpha_traits().set_alpha(new_alpha);
if(new_alpha < old_alpha){
this->max_tree_size_ = this->size();
this->rebalance();
}
}
}

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
struct make_sgtree
{
typedef typename pack_options
< sgtree_defaults,
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3, O4, O5, O6
#else
Options...
#endif
>::type packed_options;

typedef typename detail::get_value_traits
<T, typename packed_options::proto_value_traits>::type value_traits;

typedef sgtree_impl
< value_traits
, typename packed_options::key_of_value
, typename packed_options::compare
, typename packed_options::size_type
, packed_options::floating_point
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
class sgtree
:  public make_sgtree<T,
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3, O4, O5, O6
#else
Options...
#endif
>::type
{
typedef typename make_sgtree
<T,
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3, O4, O5, O6
#else
Options...
#endif
>::type   Base;
BOOST_MOVABLE_BUT_NOT_COPYABLE(sgtree)

public:
typedef typename Base::key_compare        key_compare;
typedef typename Base::value_traits       value_traits;
typedef typename Base::iterator           iterator;
typedef typename Base::const_iterator     const_iterator;
typedef typename Base::reverse_iterator           reverse_iterator;
typedef typename Base::const_reverse_iterator     const_reverse_iterator;

BOOST_STATIC_ASSERT((detail::is_same<typename value_traits::value_type, T>::value));

BOOST_INTRUSIVE_FORCEINLINE sgtree()
:  Base()
{}

BOOST_INTRUSIVE_FORCEINLINE explicit sgtree(const key_compare &cmp, const value_traits &v_traits = value_traits())
:  Base(cmp, v_traits)
{}

template<class Iterator>
BOOST_INTRUSIVE_FORCEINLINE sgtree( bool unique, Iterator b, Iterator e
, const key_compare &cmp = key_compare()
, const value_traits &v_traits = value_traits())
:  Base(unique, b, e, cmp, v_traits)
{}

BOOST_INTRUSIVE_FORCEINLINE sgtree(BOOST_RV_REF(sgtree) x)
:  Base(BOOST_MOVE_BASE(Base, x))
{}

BOOST_INTRUSIVE_FORCEINLINE sgtree& operator=(BOOST_RV_REF(sgtree) x)
{  return static_cast<sgtree &>(this->Base::operator=(BOOST_MOVE_BASE(Base, x)));  }

template <class Cloner, class Disposer>
BOOST_INTRUSIVE_FORCEINLINE void clone_from(const sgtree &src, Cloner cloner, Disposer disposer)
{  Base::clone_from(src, cloner, disposer);  }

template <class Cloner, class Disposer>
BOOST_INTRUSIVE_FORCEINLINE void clone_from(BOOST_RV_REF(sgtree) src, Cloner cloner, Disposer disposer)
{  Base::clone_from(BOOST_MOVE_BASE(Base, src), cloner, disposer);  }

BOOST_INTRUSIVE_FORCEINLINE static sgtree &container_from_end_iterator(iterator end_iterator)
{  return static_cast<sgtree &>(Base::container_from_end_iterator(end_iterator));   }

BOOST_INTRUSIVE_FORCEINLINE static const sgtree &container_from_end_iterator(const_iterator end_iterator)
{  return static_cast<const sgtree &>(Base::container_from_end_iterator(end_iterator));   }

BOOST_INTRUSIVE_FORCEINLINE static sgtree &container_from_iterator(iterator it)
{  return static_cast<sgtree &>(Base::container_from_iterator(it));   }

BOOST_INTRUSIVE_FORCEINLINE static const sgtree &container_from_iterator(const_iterator it)
{  return static_cast<const sgtree &>(Base::container_from_iterator(it));   }
};

#endif

} 
} 

#include <boost/intrusive/detail/config_end.hpp>

#endif 

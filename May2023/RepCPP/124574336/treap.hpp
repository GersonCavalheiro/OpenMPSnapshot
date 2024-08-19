#ifndef BOOST_INTRUSIVE_TREAP_HPP
#define BOOST_INTRUSIVE_TREAP_HPP

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/intrusive_fwd.hpp>

#include <boost/intrusive/detail/assert.hpp>
#include <boost/intrusive/bs_set_hook.hpp>
#include <boost/intrusive/bstree.hpp>
#include <boost/intrusive/detail/tree_node.hpp>
#include <boost/intrusive/detail/ebo_functor_holder.hpp>
#include <boost/intrusive/pointer_traits.hpp>
#include <boost/intrusive/detail/get_value_traits.hpp>
#include <boost/intrusive/detail/mpl.hpp>
#include <boost/intrusive/treap_algorithms.hpp>
#include <boost/intrusive/link_mode.hpp>
#include <boost/intrusive/priority_compare.hpp>
#include <boost/intrusive/detail/node_cloner_disposer.hpp>
#include <boost/intrusive/detail/key_nodeptr_comp.hpp>

#include <boost/static_assert.hpp>
#include <boost/move/utility_core.hpp>
#include <boost/move/adl_move_swap.hpp>

#include <cstddef>
#include <boost/intrusive/detail/minimal_less_equal_header.hpp>
#include <boost/intrusive/detail/minimal_pair_header.hpp>   

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace intrusive {


struct treap_defaults
: bstree_defaults
{
typedef void priority;
typedef void priority_of_value;
};

template<class ValuePtr, class VoidOrPrioOfValue, class VoidOrPrioComp>
struct treap_prio_types
{
typedef typename
boost::movelib::pointer_element<ValuePtr>::type value_type;
typedef typename get_key_of_value
< VoidOrPrioOfValue, value_type>::type          priority_of_value;
typedef typename priority_of_value::type           priority_type;
typedef typename get_prio_comp< VoidOrPrioComp
, priority_type
>::type                         priority_compare;
};

struct treap_tag;


#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
template<class T, class ...Options>
#else
template<class ValueTraits, class VoidOrKeyOfValue, class VoidOrKeyComp, class VoidOrPrioOfValue, class VoidOrPrioComp, class SizeType, bool ConstantTimeSize, typename HeaderHolder>
#endif
class treap_impl
: public bstree_impl<ValueTraits, VoidOrKeyOfValue, VoidOrKeyComp, SizeType, ConstantTimeSize, BsTreeAlgorithms, HeaderHolder>
, public detail::ebo_functor_holder
< typename treap_prio_types<typename ValueTraits::pointer, VoidOrPrioOfValue, VoidOrPrioComp>::priority_compare
, treap_tag>
{
public:
typedef ValueTraits                                               value_traits;
typedef bstree_impl< ValueTraits, VoidOrKeyOfValue, VoidOrKeyComp, SizeType
, ConstantTimeSize, BsTreeAlgorithms
, HeaderHolder>                                tree_type;
typedef tree_type                                                 implementation_defined;
typedef treap_prio_types
< typename ValueTraits::pointer
, VoidOrPrioOfValue, VoidOrPrioComp>                           treap_prio_types_t;

typedef detail::ebo_functor_holder
<typename treap_prio_types_t::priority_compare, treap_tag>     prio_base;


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
typedef BOOST_INTRUSIVE_IMPDEF(treap_algorithms<node_traits>)     node_algorithms;
typedef BOOST_INTRUSIVE_IMPDEF
(typename treap_prio_types_t::priority_type)                   priority_type;
typedef BOOST_INTRUSIVE_IMPDEF
(typename treap_prio_types_t::priority_of_value)               priority_of_value;
typedef BOOST_INTRUSIVE_IMPDEF
(typename treap_prio_types_t::priority_compare)                priority_compare;

static const bool constant_time_size      = implementation_defined::constant_time_size;
static const bool stateful_value_traits   = implementation_defined::stateful_value_traits;
static const bool safemode_or_autounlink = is_safe_autounlink<value_traits::link_mode>::value;

typedef detail::key_nodeptr_comp<priority_compare, value_traits, priority_of_value> prio_node_prio_comp_t;

template<class PrioPrioComp>
detail::key_nodeptr_comp<PrioPrioComp, value_traits, priority_of_value> prio_node_prio_comp(PrioPrioComp priopriocomp) const
{  return detail::key_nodeptr_comp<PrioPrioComp, value_traits, priority_of_value>(priopriocomp, &this->get_value_traits());  }

private:

BOOST_MOVABLE_BUT_NOT_COPYABLE(treap_impl)

const priority_compare &priv_pcomp() const
{  return static_cast<const prio_base&>(*this).get();  }

priority_compare &priv_pcomp()
{  return static_cast<prio_base&>(*this).get();  }


public:
typedef typename node_algorithms::insert_commit_data insert_commit_data;

treap_impl()
: tree_type(), prio_base()
{}

explicit treap_impl( const key_compare &cmp
, const priority_compare &pcmp = priority_compare()
, const value_traits &v_traits = value_traits())
: tree_type(cmp, v_traits), prio_base(pcmp)
{}

template<class Iterator>
treap_impl( bool unique, Iterator b, Iterator e
, const key_compare &cmp     = key_compare()
, const priority_compare &pcmp = priority_compare()
, const value_traits &v_traits = value_traits())
: tree_type(cmp, v_traits), prio_base(pcmp)
{
if(unique)
this->insert_unique(b, e);
else
this->insert_equal(b, e);
}

treap_impl(BOOST_RV_REF(treap_impl) x)
: tree_type(BOOST_MOVE_BASE(tree_type, x))
, prio_base(::boost::move(x.priv_pcomp()))
{}

treap_impl& operator=(BOOST_RV_REF(treap_impl) x)
{  this->swap(x); return *this;  }

#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED
~treap_impl();

iterator begin();

const_iterator begin() const;

const_iterator cbegin() const;

iterator end();

const_iterator end() const;

const_iterator cend() const;
#endif

iterator top()
{  return this->tree_type::root();   }

const_iterator top() const
{  return this->ctop();   }

const_iterator ctop() const
{  return this->tree_type::root();   }

#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED
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

reverse_iterator rtop()
{  return reverse_iterator(this->top());  }

const_reverse_iterator rtop() const
{  return const_reverse_iterator(this->top());  }

const_reverse_iterator crtop() const
{  return const_reverse_iterator(this->top());  }

#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED
static treap_impl &container_from_end_iterator(iterator end_iterator);

static const treap_impl &container_from_end_iterator(const_iterator end_iterator);

static treap_impl &container_from_iterator(iterator it);

static const treap_impl &container_from_iterator(const_iterator it);

key_compare key_comp() const;

value_compare value_comp() const;

bool empty() const;

size_type size() const;
#endif   

priority_compare priority_comp() const
{  return this->priv_pcomp();   }

void swap(treap_impl& other)
{
::boost::adl_move_swap(this->priv_pcomp(), other.priv_pcomp());
tree_type::swap(other);
}

template <class Cloner, class Disposer>
void clone_from(const treap_impl &src, Cloner cloner, Disposer disposer)
{
tree_type::clone_from(src, cloner, disposer);
this->priv_pcomp() = src.priv_pcomp();
}

template <class Cloner, class Disposer>
void clone_from(BOOST_RV_REF(treap_impl) src, Cloner cloner, Disposer disposer)
{
tree_type::clone_from(BOOST_MOVE_BASE(tree_type, src), cloner, disposer);
this->priv_pcomp() = ::boost::move(src.priv_pcomp());
}

iterator insert_equal(reference value)
{
node_ptr to_insert(this->get_value_traits().to_node_ptr(value));
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || node_algorithms::unique(to_insert));
iterator ret
( node_algorithms::insert_equal_upper_bound
( this->tree_type::header_ptr()
, to_insert
, this->key_node_comp(this->key_comp())
, this->prio_node_prio_comp(this->priv_pcomp()))
, this->priv_value_traits_ptr());
this->tree_type::sz_traits().increment();
return ret;
}

iterator insert_equal(const_iterator hint, reference value)
{
node_ptr to_insert(this->get_value_traits().to_node_ptr(value));
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || node_algorithms::unique(to_insert));
iterator ret
(node_algorithms::insert_equal
( this->tree_type::header_ptr()
, hint.pointed_node()
, to_insert
, this->key_node_comp(this->key_comp())
, this->prio_node_prio_comp(this->priv_pcomp()))
, this->priv_value_traits_ptr());
this->tree_type::sz_traits().increment();
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
std::pair<iterator, bool> ret = this->insert_unique_check(key_of_value()(value), priority_of_value()(value), commit_data);
if(!ret.second)
return ret;
return std::pair<iterator, bool> (this->insert_unique_commit(value, commit_data), true);
}

iterator insert_unique(const_iterator hint, reference value)
{
insert_commit_data commit_data;
std::pair<iterator, bool> ret = this->insert_unique_check(hint, key_of_value()(value), priority_of_value()(value), commit_data);
if(!ret.second)
return ret.first;
return this->insert_unique_commit(value, commit_data);
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

std::pair<iterator, bool> insert_unique_check
( const key_type &key, const priority_type &prio, insert_commit_data &commit_data)
{  return this->insert_unique_check(key, this->key_comp(), prio, this->priv_pcomp(), commit_data); }

std::pair<iterator, bool> insert_unique_check
( const_iterator hint, const key_type &key, const priority_type &prio, insert_commit_data &commit_data)
{  return this->insert_unique_check(hint, key, this->key_comp(), prio, this->priv_pcomp(), commit_data); }

template<class KeyType, class KeyTypeKeyCompare, class PrioType, class PrioValuePrioCompare>
BOOST_INTRUSIVE_DOC1ST(std::pair<iterator BOOST_INTRUSIVE_I bool>
, typename detail::disable_if_convertible
<KeyType BOOST_INTRUSIVE_I const_iterator BOOST_INTRUSIVE_I 
std::pair<iterator BOOST_INTRUSIVE_I bool> >::type)
insert_unique_check
( const KeyType &key, KeyTypeKeyCompare comp
, const PrioType &prio, PrioValuePrioCompare prio_value_pcomp, insert_commit_data &commit_data)
{
std::pair<node_ptr, bool> const ret =
(node_algorithms::insert_unique_check
( this->tree_type::header_ptr()
, key, this->key_node_comp(comp)
, prio, this->prio_node_prio_comp(prio_value_pcomp)
, commit_data));
return std::pair<iterator, bool>(iterator(ret.first, this->priv_value_traits_ptr()), ret.second);
}

template<class KeyType, class KeyTypeKeyCompare, class PrioType, class PrioValuePrioCompare>
std::pair<iterator, bool> insert_unique_check
( const_iterator hint
, const KeyType &key
, KeyTypeKeyCompare comp
, const PrioType &prio
, PrioValuePrioCompare prio_value_pcomp
, insert_commit_data &commit_data)
{
std::pair<node_ptr, bool> const ret =
(node_algorithms::insert_unique_check
( this->tree_type::header_ptr(), hint.pointed_node()
, key, this->key_node_comp(comp)
, prio, this->prio_node_prio_comp(prio_value_pcomp)
, commit_data));
return std::pair<iterator, bool>(iterator(ret.first, this->priv_value_traits_ptr()), ret.second);
}

iterator insert_unique_commit(reference value, const insert_commit_data &commit_data)
{
node_ptr to_insert(this->get_value_traits().to_node_ptr(value));
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || node_algorithms::unique(to_insert));
node_algorithms::insert_unique_commit(this->tree_type::header_ptr(), to_insert, commit_data);
this->tree_type::sz_traits().increment();
return iterator(to_insert, this->priv_value_traits_ptr());
}

iterator insert_before(const_iterator pos, reference value)
{
node_ptr to_insert(this->get_value_traits().to_node_ptr(value));
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || node_algorithms::unique(to_insert));
iterator ret
( node_algorithms::insert_before
( this->tree_type::header_ptr()
, pos.pointed_node()
, to_insert
, this->prio_node_prio_comp(this->priv_pcomp())
)
, this->priv_value_traits_ptr());
this->tree_type::sz_traits().increment();
return ret;
}

void push_back(reference value)
{
node_ptr to_insert(this->get_value_traits().to_node_ptr(value));
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || node_algorithms::unique(to_insert));
node_algorithms::push_back
(this->tree_type::header_ptr(), to_insert, this->prio_node_prio_comp(this->priv_pcomp()));
this->tree_type::sz_traits().increment();
}

void push_front(reference value)
{
node_ptr to_insert(this->get_value_traits().to_node_ptr(value));
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || node_algorithms::unique(to_insert));
node_algorithms::push_front
(this->tree_type::header_ptr(), to_insert, this->prio_node_prio_comp(this->priv_pcomp()));
this->tree_type::sz_traits().increment();
}

iterator erase(const_iterator i)
{
const_iterator ret(i);
++ret;
node_ptr to_erase(i.pointed_node());
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || !node_algorithms::unique(to_erase));
node_algorithms::erase
(this->tree_type::header_ptr(), to_erase, this->prio_node_prio_comp(this->priv_pcomp()));
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
{  tree_type::clear(); }

template<class Disposer>
void clear_and_dispose(Disposer disposer)
{
node_algorithms::clear_and_dispose(this->tree_type::header_ptr()
, detail::node_disposer<Disposer, value_traits, TreapAlgorithms>(disposer, &this->get_value_traits()));
node_algorithms::init_header(this->tree_type::header_ptr());
this->tree_type::sz_traits().set_size(0);
}

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
template<class T, class ...Options2> void merge_unique(sgtree<T, Options2...> &);
#else
template<class Compare2>
void merge_unique(treap_impl
<ValueTraits, VoidOrKeyOfValue, Compare2, VoidOrPrioOfValue, VoidOrPrioComp, SizeType, ConstantTimeSize, HeaderHolder> &source)
#endif
{
node_ptr it   (node_algorithms::begin_node(source.header_ptr()))
, itend(node_algorithms::end_node  (source.header_ptr()));

while(it != itend){
node_ptr const p(it);
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || !node_algorithms::unique(p));
it = node_algorithms::next_node(it);

if( node_algorithms::transfer_unique
( this->header_ptr(), this->key_node_comp(this->key_comp())
, this->prio_node_prio_comp(this->priv_pcomp()), source.header_ptr(), p) ){
this->sz_traits().increment();
source.sz_traits().decrement();
}
}
}

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
template<class T, class ...Options2> void merge_equal(sgtree<T, Options2...> &);
#else
template<class Compare2>
void merge_equal(treap_impl
<ValueTraits, VoidOrKeyOfValue, Compare2, VoidOrPrioOfValue, VoidOrPrioComp, SizeType, ConstantTimeSize, HeaderHolder> &source)
#endif
{
node_ptr it   (node_algorithms::begin_node(source.header_ptr()))
, itend(node_algorithms::end_node  (source.header_ptr()));

while(it != itend){
node_ptr const p(it);
BOOST_INTRUSIVE_SAFE_HOOK_DEFAULT_ASSERT(!safemode_or_autounlink || !node_algorithms::unique(p));
it = node_algorithms::next_node(it);
node_algorithms::transfer_equal
( this->header_ptr(), this->key_node_comp(this->key_comp())
, this->prio_node_prio_comp(this->priv_pcomp()), source.header_ptr(), p);
this->sz_traits().increment();
source.sz_traits().decrement();
}
}

template <class ExtraChecker>
void check(ExtraChecker extra_checker) const
{
typedef detail::key_nodeptr_comp<priority_compare, value_traits, priority_of_value> nodeptr_prio_comp_t;
tree_type::check(detail::treap_node_extra_checker
<ValueTraits, nodeptr_prio_comp_t, ExtraChecker>
(this->prio_node_prio_comp(this->priv_pcomp()), extra_checker));
}

void check() const
{  check(detail::empty_node_checker<ValueTraits>());  }

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

friend bool operator< (const treap_impl &x, const treap_impl &y);

friend bool operator==(const treap_impl &x, const treap_impl &y);

friend bool operator!= (const treap_impl &x, const treap_impl &y);

friend bool operator>(const treap_impl &x, const treap_impl &y);

friend bool operator<=(const treap_impl &x, const treap_impl &y);

friend bool operator>=(const treap_impl &x, const treap_impl &y);

friend void swap(treap_impl &x, treap_impl &y);

#endif   

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
, class O5 = void, class O6 = void
, class O7 = void>
#endif
struct make_treap
{
typedef typename pack_options
< treap_defaults,
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3, O4, O5, O6, O7
#else
Options...
#endif
>::type packed_options;

typedef typename detail::get_value_traits
<T, typename packed_options::proto_value_traits>::type value_traits;

typedef treap_impl
< value_traits
, typename packed_options::key_of_value
, typename packed_options::compare
, typename packed_options::priority_of_value
, typename packed_options::priority
, typename packed_options::size_type
, packed_options::constant_time_size
, typename packed_options::header_holder_type
> implementation_defined;
typedef implementation_defined type;
};

#ifndef BOOST_INTRUSIVE_DOXYGEN_INVOKED

#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
template<class T, class O1, class O2, class O3, class O4, class O5, class O6, class O7>
#else
template<class T, class ...Options>
#endif
class treap
:  public make_treap<T,
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3, O4, O5, O6, O7
#else
Options...
#endif
>::type
{
typedef typename make_treap
<T,
#if !defined(BOOST_INTRUSIVE_VARIADIC_TEMPLATES)
O1, O2, O3, O4, O5, O6, O7
#else
Options...
#endif
>::type   Base;
BOOST_MOVABLE_BUT_NOT_COPYABLE(treap)

public:
typedef typename Base::key_compare        key_compare;
typedef typename Base::priority_compare   priority_compare;
typedef typename Base::value_traits       value_traits;
typedef typename Base::iterator           iterator;
typedef typename Base::const_iterator     const_iterator;
typedef typename Base::reverse_iterator           reverse_iterator;
typedef typename Base::const_reverse_iterator     const_reverse_iterator;

BOOST_STATIC_ASSERT((detail::is_same<typename value_traits::value_type, T>::value));

BOOST_INTRUSIVE_FORCEINLINE treap()
:  Base()
{}

BOOST_INTRUSIVE_FORCEINLINE explicit treap( const key_compare &cmp
, const priority_compare &pcmp = priority_compare()
, const value_traits &v_traits = value_traits())
:  Base(cmp, pcmp, v_traits)
{}

template<class Iterator>
BOOST_INTRUSIVE_FORCEINLINE treap( bool unique, Iterator b, Iterator e
, const key_compare &cmp = key_compare()
, const priority_compare &pcmp = priority_compare()
, const value_traits &v_traits = value_traits())
:  Base(unique, b, e, cmp, pcmp, v_traits)
{}

BOOST_INTRUSIVE_FORCEINLINE treap(BOOST_RV_REF(treap) x)
:  Base(BOOST_MOVE_BASE(Base, x))
{}

BOOST_INTRUSIVE_FORCEINLINE treap& operator=(BOOST_RV_REF(treap) x)
{  return static_cast<treap&>(this->Base::operator=(BOOST_MOVE_BASE(Base, x)));  }

template <class Cloner, class Disposer>
BOOST_INTRUSIVE_FORCEINLINE void clone_from(const treap &src, Cloner cloner, Disposer disposer)
{  Base::clone_from(src, cloner, disposer);  }

template <class Cloner, class Disposer>
BOOST_INTRUSIVE_FORCEINLINE void clone_from(BOOST_RV_REF(treap) src, Cloner cloner, Disposer disposer)
{  Base::clone_from(BOOST_MOVE_BASE(Base, src), cloner, disposer);  }

BOOST_INTRUSIVE_FORCEINLINE static treap &container_from_end_iterator(iterator end_iterator)
{  return static_cast<treap &>(Base::container_from_end_iterator(end_iterator));   }

BOOST_INTRUSIVE_FORCEINLINE static const treap &container_from_end_iterator(const_iterator end_iterator)
{  return static_cast<const treap &>(Base::container_from_end_iterator(end_iterator));   }

BOOST_INTRUSIVE_FORCEINLINE static treap &container_from_iterator(iterator it)
{  return static_cast<treap &>(Base::container_from_iterator(it));   }

BOOST_INTRUSIVE_FORCEINLINE static const treap &container_from_iterator(const_iterator it)
{  return static_cast<const treap &>(Base::container_from_iterator(it));   }
};

#endif

} 
} 

#include <boost/intrusive/detail/config_end.hpp>

#endif 

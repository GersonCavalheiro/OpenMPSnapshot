
#ifndef BOOST_HEAP_FIBONACCI_HEAP_HPP
#define BOOST_HEAP_FIBONACCI_HEAP_HPP

#include <algorithm>
#include <utility>
#include <vector>

#include <boost/array.hpp>
#include <boost/assert.hpp>

#include <boost/heap/detail/heap_comparison.hpp>
#include <boost/heap/detail/heap_node.hpp>
#include <boost/heap/detail/stable_heap.hpp>
#include <boost/heap/detail/tree_iterator.hpp>
#include <boost/type_traits/integral_constant.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif


#ifndef BOOST_DOXYGEN_INVOKED
#ifdef BOOST_HEAP_SANITYCHECKS
#define BOOST_HEAP_ASSERT BOOST_ASSERT
#else
#define BOOST_HEAP_ASSERT(expression)
#endif
#endif

namespace boost  {
namespace heap   {
namespace detail {

typedef parameter::parameters<boost::parameter::optional<tag::allocator>,
boost::parameter::optional<tag::compare>,
boost::parameter::optional<tag::stable>,
boost::parameter::optional<tag::constant_time_size>,
boost::parameter::optional<tag::stability_counter_type>
> fibonacci_heap_signature;

template <typename T, typename Parspec>
struct make_fibonacci_heap_base
{
static const bool constant_time_size = parameter::binding<Parspec,
tag::constant_time_size,
boost::true_type
>::type::value;

typedef typename detail::make_heap_base<T, Parspec, constant_time_size>::type base_type;
typedef typename detail::make_heap_base<T, Parspec, constant_time_size>::allocator_argument allocator_argument;
typedef typename detail::make_heap_base<T, Parspec, constant_time_size>::compare_argument compare_argument;
typedef marked_heap_node<typename base_type::internal_type> node_type;

typedef typename boost::allocator_rebind<allocator_argument, node_type>::type allocator_type;

struct type:
base_type,
allocator_type
{
type(compare_argument const & arg):
base_type(arg)
{}

type(type const & rhs):
base_type(static_cast<base_type const &>(rhs)),
allocator_type(static_cast<allocator_type const &>(rhs))
{}

type & operator=(type const & rhs)
{
base_type::operator=(static_cast<base_type const &>(rhs));
allocator_type::operator=(static_cast<allocator_type const &>(rhs));
return *this;
}

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
type(type && rhs):
base_type(std::move(static_cast<base_type&>(rhs))),
allocator_type(std::move(static_cast<allocator_type&>(rhs)))
{}

type & operator=(type && rhs)
{
base_type::operator=(std::move(static_cast<base_type&>(rhs)));
allocator_type::operator=(std::move(static_cast<allocator_type&>(rhs)));
return *this;
}
#endif
};
};

}




#ifdef BOOST_DOXYGEN_INVOKED
template<class T, class ...Options>
#else
template <typename T,
class A0 = boost::parameter::void_,
class A1 = boost::parameter::void_,
class A2 = boost::parameter::void_,
class A3 = boost::parameter::void_,
class A4 = boost::parameter::void_
>
#endif
class fibonacci_heap:
private detail::make_fibonacci_heap_base<T,
typename detail::fibonacci_heap_signature::bind<A0, A1, A2, A3, A4>::type
>::type
{
typedef typename detail::fibonacci_heap_signature::bind<A0, A1, A2, A3, A4>::type bound_args;
typedef detail::make_fibonacci_heap_base<T, bound_args> base_maker;
typedef typename base_maker::type super_t;

typedef typename super_t::size_holder_type size_holder;
typedef typename super_t::internal_type internal_type;
typedef typename base_maker::allocator_argument allocator_argument;

template <typename Heap1, typename Heap2>
friend struct heap_merge_emulate;

private:
#ifndef BOOST_DOXYGEN_INVOKED
struct implementation_defined:
detail::extract_allocator_types<typename base_maker::allocator_argument>
{
typedef T value_type;
typedef typename detail::extract_allocator_types<typename base_maker::allocator_argument>::size_type size_type;
typedef typename detail::extract_allocator_types<typename base_maker::allocator_argument>::reference reference;

typedef typename base_maker::compare_argument value_compare;
typedef typename base_maker::allocator_type allocator_type;

typedef typename boost::allocator_pointer<allocator_type>::type node_pointer;
typedef typename boost::allocator_const_pointer<allocator_type>::type const_node_pointer;

typedef detail::heap_node_list node_list_type;
typedef typename node_list_type::iterator node_list_iterator;
typedef typename node_list_type::const_iterator node_list_const_iterator;

typedef typename base_maker::node_type node;

typedef detail::value_extractor<value_type, internal_type, super_t> value_extractor;
typedef typename super_t::internal_compare internal_compare;
typedef detail::node_handle<node_pointer, super_t, reference> handle_type;

typedef detail::recursive_tree_iterator<node,
node_list_const_iterator,
const value_type,
value_extractor,
detail::list_iterator_converter<node, node_list_type>
> iterator;
typedef iterator const_iterator;

typedef detail::tree_iterator<node,
const value_type,
allocator_type,
value_extractor,
detail::list_iterator_converter<node, node_list_type>,
true,
true,
value_compare
> ordered_iterator;
};

typedef typename implementation_defined::node node;
typedef typename implementation_defined::node_pointer node_pointer;
typedef typename implementation_defined::node_list_type node_list_type;
typedef typename implementation_defined::node_list_iterator node_list_iterator;
typedef typename implementation_defined::node_list_const_iterator node_list_const_iterator;
typedef typename implementation_defined::internal_compare internal_compare;
#endif

public:
typedef T value_type;

typedef typename implementation_defined::size_type size_type;
typedef typename implementation_defined::difference_type difference_type;
typedef typename implementation_defined::value_compare value_compare;
typedef typename implementation_defined::allocator_type allocator_type;
typedef typename implementation_defined::reference reference;
typedef typename implementation_defined::const_reference const_reference;
typedef typename implementation_defined::pointer pointer;
typedef typename implementation_defined::const_pointer const_pointer;
typedef typename implementation_defined::iterator iterator;
typedef typename implementation_defined::const_iterator const_iterator;
typedef typename implementation_defined::ordered_iterator ordered_iterator;

typedef typename implementation_defined::handle_type handle_type;

static const bool constant_time_size = base_maker::constant_time_size;
static const bool has_ordered_iterators = true;
static const bool is_mergable = true;
static const bool is_stable = detail::extract_stable<bound_args>::value;
static const bool has_reserve = false;

explicit fibonacci_heap(value_compare const & cmp = value_compare()):
super_t(cmp), top_element(0)
{}

fibonacci_heap(fibonacci_heap const & rhs):
super_t(rhs), top_element(0)
{
if (rhs.empty())
return;

clone_forest(rhs);
size_holder::set_size(rhs.size());
}

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
fibonacci_heap(fibonacci_heap && rhs):
super_t(std::move(rhs)), top_element(rhs.top_element)
{
roots.splice(roots.begin(), rhs.roots);
rhs.top_element = NULL;
}

fibonacci_heap & operator=(fibonacci_heap && rhs)
{
clear();

super_t::operator=(std::move(rhs));
roots.splice(roots.begin(), rhs.roots);
top_element = rhs.top_element;
rhs.top_element = NULL;
return *this;
}
#endif

fibonacci_heap & operator=(fibonacci_heap const & rhs)
{
clear();
size_holder::set_size(rhs.size());
static_cast<super_t&>(*this) = rhs;

if (rhs.empty())
top_element = NULL;
else
clone_forest(rhs);
return *this;
}

~fibonacci_heap(void)
{
clear();
}

bool empty(void) const
{
if (constant_time_size)
return size() == 0;
else
return roots.empty();
}

size_type size(void) const
{
if (constant_time_size)
return size_holder::get_size();

if (empty())
return 0;
else
return detail::count_list_nodes<node, node_list_type>(roots);
}

size_type max_size(void) const
{
const allocator_type& alloc = *this;
return boost::allocator_max_size(alloc);
}

void clear(void)
{
typedef detail::node_disposer<node, typename node_list_type::value_type, allocator_type> disposer;
roots.clear_and_dispose(disposer(*this));

size_holder::set_size(0);
top_element = NULL;
}

allocator_type get_allocator(void) const
{
return *this;
}

void swap(fibonacci_heap & rhs)
{
super_t::swap(rhs);
std::swap(top_element, rhs.top_element);
roots.swap(rhs.roots);
}


value_type const & top(void) const
{
BOOST_ASSERT(!empty());

return super_t::get_value(top_element->value);
}


handle_type push(value_type const & v)
{
size_holder::increment();

allocator_type& alloc = *this;
node_pointer n = alloc.allocate(1);
new(n) node(super_t::make_node(v));
roots.push_front(*n);

if (!top_element || super_t::operator()(top_element->value, n->value))
top_element = n;
return handle_type(n);
}

#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES) && !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES)

template <class... Args>
handle_type emplace(Args&&... args)
{
size_holder::increment();

allocator_type& alloc = *this;
node_pointer n = alloc.allocate(1);
new(n) node(super_t::make_node(std::forward<Args>(args)...));
roots.push_front(*n);

if (!top_element || super_t::operator()(top_element->value, n->value))
top_element = n;
return handle_type(n);
}
#endif


void pop(void)
{
BOOST_ASSERT(!empty());

node_pointer element = top_element;
roots.erase(node_list_type::s_iterator_to(*element));

finish_erase_or_pop(element);
}


void update (handle_type handle, const_reference v)
{
if (super_t::operator()(super_t::get_value(handle.node_->value), v))
increase(handle, v);
else
decrease(handle, v);
}


void update_lazy(handle_type handle, const_reference v)
{
handle.node_->value = super_t::make_node(v);
update_lazy(handle);
}


void update (handle_type handle)
{
update_lazy(handle);
consolidate();
}


void update_lazy (handle_type handle)
{
node_pointer n = handle.node_;
node_pointer parent = n->get_parent();

if (parent) {
n->parent = NULL;
roots.splice(roots.begin(), parent->children, node_list_type::s_iterator_to(*n));
}
add_children_to_root(n);

if (super_t::operator()(top_element->value, n->value))
top_element = n;
}



void increase (handle_type handle, const_reference v)
{
handle.node_->value = super_t::make_node(v);
increase(handle);
}


void increase (handle_type handle)
{
node_pointer n = handle.node_;

if (n->parent) {
if (super_t::operator()(n->get_parent()->value, n->value)) {
node_pointer parent = n->get_parent();
cut(n);
cascading_cut(parent);
}
}

if (super_t::operator()(top_element->value, n->value)) {
top_element = n;
return;
}
}


void decrease (handle_type handle, const_reference v)
{
handle.node_->value = super_t::make_node(v);
decrease(handle);
}


void decrease (handle_type handle)
{
update(handle);
}


void erase(handle_type const & handle)
{
node_pointer element = handle.node_;
node_pointer parent = element->get_parent();

if (parent)
parent->children.erase(node_list_type::s_iterator_to(*element));
else
roots.erase(node_list_type::s_iterator_to(*element));

finish_erase_or_pop(element);
}

iterator begin(void) const
{
return iterator(roots.begin());
}

iterator end(void) const
{
return iterator(roots.end());
}



ordered_iterator ordered_begin(void) const
{
return ordered_iterator(roots.begin(), roots.end(), top_element, super_t::value_comp());
}


ordered_iterator ordered_end(void) const
{
return ordered_iterator(NULL, super_t::value_comp());
}


void merge(fibonacci_heap & rhs)
{
size_holder::add(rhs.get_size());

if (!top_element ||
(rhs.top_element && super_t::operator()(top_element->value, rhs.top_element->value)))
top_element = rhs.top_element;

roots.splice(roots.end(), rhs.roots);

rhs.top_element = NULL;
rhs.set_size(0);

super_t::set_stability_count((std::max)(super_t::get_stability_count(),
rhs.get_stability_count()));
rhs.set_stability_count(0);
}

static handle_type s_handle_from_iterator(iterator const & it)
{
node * ptr = const_cast<node *>(it.get_node());
return handle_type(ptr);
}

value_compare const & value_comp(void) const
{
return super_t::value_comp();
}

template <typename HeapType>
bool operator<(HeapType const & rhs) const
{
return detail::heap_compare(*this, rhs);
}

template <typename HeapType>
bool operator>(HeapType const & rhs) const
{
return detail::heap_compare(rhs, *this);
}

template <typename HeapType>
bool operator>=(HeapType const & rhs) const
{
return !operator<(rhs);
}

template <typename HeapType>
bool operator<=(HeapType const & rhs) const
{
return !operator>(rhs);
}

template <typename HeapType>
bool operator==(HeapType const & rhs) const
{
return detail::heap_equality(*this, rhs);
}

template <typename HeapType>
bool operator!=(HeapType const & rhs) const
{
return !(*this == rhs);
}

private:
#if !defined(BOOST_DOXYGEN_INVOKED)
void clone_forest(fibonacci_heap const & rhs)
{
BOOST_HEAP_ASSERT(roots.empty());
typedef typename node::template node_cloner<allocator_type> node_cloner;
roots.clone_from(rhs.roots, node_cloner(*this, NULL), detail::nop_disposer());

top_element = detail::find_max_child<node_list_type, node, internal_compare>(roots, super_t::get_internal_cmp());
}

void cut(node_pointer n)
{
node_pointer parent = n->get_parent();
roots.splice(roots.begin(), parent->children, node_list_type::s_iterator_to(*n));
n->parent = 0;
n->mark = false;
}

void cascading_cut(node_pointer n)
{
node_pointer parent = n->get_parent();

if (parent) {
if (!parent->mark)
parent->mark = true;
else {
cut(n);
cascading_cut(parent);
}
}
}

void add_children_to_root(node_pointer n)
{
for (node_list_iterator it = n->children.begin(); it != n->children.end(); ++it) {
node_pointer child = static_cast<node_pointer>(&*it);
child->parent = 0;
}

roots.splice(roots.end(), n->children);
}

void consolidate(void)
{
if (roots.empty())
return;

static const size_type max_log2 = sizeof(size_type) * 8;
boost::array<node_pointer, max_log2> aux;
aux.assign(NULL);

node_list_iterator it = roots.begin();
top_element = static_cast<node_pointer>(&*it);

do {
node_pointer n = static_cast<node_pointer>(&*it);
++it;
size_type node_rank = n->child_count();

if (aux[node_rank] == NULL)
aux[node_rank] = n;
else {
do {
node_pointer other = aux[node_rank];
if (super_t::operator()(n->value, other->value))
std::swap(n, other);

if (other->parent)
n->children.splice(n->children.end(), other->parent->children, node_list_type::s_iterator_to(*other));
else
n->children.splice(n->children.end(), roots, node_list_type::s_iterator_to(*other));

other->parent = n;

aux[node_rank] = NULL;
node_rank = n->child_count();
} while (aux[node_rank] != NULL);
aux[node_rank] = n;
}

if (!super_t::operator()(n->value, top_element->value))
top_element = n;
}
while (it != roots.end());
}

void finish_erase_or_pop(node_pointer erased_node)
{
add_children_to_root(erased_node);

erased_node->~node();
allocator_type& alloc = *this;
alloc.deallocate(erased_node, 1);

size_holder::decrement();
if (!empty())
consolidate();
else
top_element = NULL;
}

mutable node_pointer top_element;
node_list_type roots;
#endif
};

} 
} 

#undef BOOST_HEAP_ASSERT

#endif 

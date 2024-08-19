
#ifndef BOOST_HEAP_MERGE_HPP
#define BOOST_HEAP_MERGE_HPP

#include <algorithm>

#include <boost/concept/assert.hpp>
#include <boost/heap/heap_concepts.hpp>
#include <boost/type_traits/conditional.hpp>
#include <boost/type_traits/is_same.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif


namespace boost  {
namespace heap   {
namespace detail {

template <typename Heap1, typename Heap2>
struct heap_merge_emulate
{
struct dummy_reserver
{
static void reserve (Heap1 & lhs, std::size_t required_size)
{}
};

struct reserver
{
static void reserve (Heap1 & lhs, std::size_t required_size)
{
lhs.reserve(required_size);
}
};

typedef typename boost::conditional<Heap1::has_reserve,
reserver,
dummy_reserver>::type space_reserver;

static void merge(Heap1 & lhs, Heap2 & rhs)
{
if (Heap1::constant_time_size && Heap2::constant_time_size) {
if (Heap1::has_reserve) {
std::size_t required_size = lhs.size() + rhs.size();
space_reserver::reserve(lhs, required_size);
}
}


while (!rhs.empty()) {
lhs.push(rhs.top());
rhs.pop();
}

lhs.set_stability_count((std::max)(lhs.get_stability_count(),
rhs.get_stability_count()));
rhs.set_stability_count(0);
}

};


template <typename Heap>
struct heap_merge_same_mergable
{
static void merge(Heap & lhs, Heap & rhs)
{
lhs.merge(rhs);
}
};


template <typename Heap>
struct heap_merge_same
{
static const bool is_mergable = Heap::is_mergable;
typedef typename boost::conditional<is_mergable,
heap_merge_same_mergable<Heap>,
heap_merge_emulate<Heap, Heap>
>::type heap_merger;

static void merge(Heap & lhs, Heap & rhs)
{
heap_merger::merge(lhs, rhs);
}
};

} 



template <typename Heap1,
typename Heap2
>
void heap_merge(Heap1 & lhs, Heap2 & rhs)
{
BOOST_CONCEPT_ASSERT((boost::heap::PriorityQueue<Heap1>));
BOOST_CONCEPT_ASSERT((boost::heap::PriorityQueue<Heap2>));

BOOST_STATIC_ASSERT((boost::is_same<typename Heap1::value_compare, typename Heap2::value_compare>::value));

const bool same_heaps = boost::is_same<Heap1, Heap2>::value;

typedef typename boost::conditional<same_heaps,
detail::heap_merge_same<Heap1>,
detail::heap_merge_emulate<Heap1, Heap2>
>::type heap_merger;

heap_merger::merge(lhs, rhs);
}


} 
} 

#endif 

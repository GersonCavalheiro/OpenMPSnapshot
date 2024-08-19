#pragma once

#include <numeric>
#include <map>
#include <cereal/types/map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/base_class.hpp>

#include "sample/Sample.hpp"
#include "pipeline/Token.hpp"

class WorkIterator;


class Work : public Token
{
public:
using key_type              = size_t;
using value_type            = size_t;
using container_value_type  = std::vector<value_type>;
using container_type        = std::map<key_type, container_value_type>;
using const_iterator        = WorkIterator;

struct Work_Pair
{
key_type    branch_id;
value_type  sequence_id;
};


template<class T>
Work(Sample<T>& sample)
{
for (auto& pq : sample)
{
const auto seq_id = pq.sequence_id();
for (auto& placement : pq)
{
const auto branch_id = placement.branch_id();
work_set_[branch_id].push_back(seq_id);
}
}
}


Work(std::pair<key_type, key_type>&& branch_range, std::pair<value_type, value_type>&& seq_range)
{
for (key_type branch_id = branch_range.first; branch_id < branch_range.second; ++branch_id) {
for (value_type seq_id = seq_range.first; seq_id < seq_range.second; ++seq_id) {
work_set_[branch_id].push_back(seq_id);
}
}
}

Work(Work const& other) = default;
Work(Work && other) = default;

Work& operator=(Work const&) = default;
Work& operator=(Work &&) = default;

Work() = default;
~Work() = default;

void clear() {work_set_.clear();}

size_t size() const
{
return std::accumulate(work_set_.begin(), work_set_.end(), 0,
[](size_t a, std::pair<key_type, container_value_type> b){return a + b.second.size();}
);
}

bool empty() const { return work_set_.empty();}

inline void add(key_type branch_id, value_type seq_id)
{
work_set_[branch_id].emplace_back(seq_id);
}

inline void add(Work_Pair& it);


auto bin_begin() { return work_set_.begin(); }
auto bin_end() { return work_set_.end(); }
auto bin_cbegin() const { return work_set_.begin(); }
auto bin_cend() const { return work_set_.end(); }
const_iterator begin() const;
const_iterator end() const;

const container_value_type& at (const key_type index) const { return work_set_.at(index); }
container_value_type& operator[] (const key_type index) { return work_set_[index]; }

template <class Archive>
void serialize(Archive & ar)
{ ar( *static_cast<Token*>( this ), work_set_ ); }


private:
container_type work_set_;
};

class WorkIterator
{
public:
using WorkType      = Work::container_type;
using MapItType     = typename WorkType::const_iterator;
using VecItType     = typename WorkType::mapped_type::const_iterator;
using self_type     = WorkIterator;
using element_type  = Work::Work_Pair;
using iterator_tag  = std::forward_iterator_tag;


WorkIterator() = delete;

WorkIterator( WorkType const& target, bool is_end )
: map_it( std::begin( target ) )
, end_it( std::end( target ) )
, vec_it()
{
if( is_end ) {
map_it = end_it;
}
if( map_it != end_it && ! target.empty() ) {
while( map_it != end_it && map_it->second.empty() ) {
++map_it;
}
if( map_it != end_it && ! map_it->second.empty() ) {
vec_it = std::begin( map_it->second );
}
}
}

~WorkIterator() = default;

WorkIterator( WorkIterator const& ) = default;
WorkIterator( WorkIterator&& ) = default;
WorkIterator& operator= ( WorkIterator const& ) = default;
WorkIterator& operator= ( WorkIterator&& ) = default;


element_type operator * ()
{
return { map_it->first, *vec_it };
}


size_t current_branch_id()
{

return map_it->first;
}

size_t current_sequence_id()
{
return *vec_it;
}

self_type operator ++ ()
{
++vec_it;
if( vec_it == map_it->second.end() ) {
do {
++map_it;
if( map_it == end_it ) {
break;
} else {
vec_it = std::begin( map_it->second );
}
} while( vec_it == std::end( map_it->second ) );
}
return *this;
}

self_type operator ++ (int)
{
self_type tmp = *this;
++(*this);
return tmp;
}

bool operator == (const self_type &other) const
{
return ( other.map_it == map_it && other.end_it == end_it ) &&
( map_it == end_it || other.vec_it == vec_it );
}

bool operator != (const self_type &other) const
{
return !(other == *this);
}


private:

MapItType map_it;
MapItType end_it;

VecItType vec_it;
};

inline void Work::add(Work_Pair& it)
{
add(it.branch_id, it.sequence_id);
}

inline Work::const_iterator Work::begin() const
{
return WorkIterator( work_set_, false );
}

inline Work::const_iterator Work::end() const
{
return WorkIterator( work_set_, true );
}
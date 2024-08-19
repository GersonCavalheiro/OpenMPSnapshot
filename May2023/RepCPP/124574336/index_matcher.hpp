

#ifndef BOOST_MULTI_INDEX_DETAIL_INDEX_MATCHER_HPP
#define BOOST_MULTI_INDEX_DETAIL_INDEX_MATCHER_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <algorithm>
#include <boost/noncopyable.hpp>
#include <boost/multi_index/detail/auto_space.hpp>
#include <boost/multi_index/detail/raw_ptr.hpp>
#include <cstddef>
#include <functional>

namespace boost{

namespace multi_index{

namespace detail{



namespace index_matcher{



struct entry
{
entry(void* node_,std::size_t pos_=0):node(node_),pos(pos_){}



void*       node;
std::size_t pos;
entry*      previous;
bool        ordered;

struct less_by_node
{
bool operator()(
const entry& x,const entry& y)const
{
return std::less<void*>()(x.node,y.node);
}
};



std::size_t pile_top;
entry*      pile_top_entry;

struct less_by_pile_top
{
bool operator()(
const entry& x,const entry& y)const
{
return x.pile_top<y.pile_top;
}
};
};



template<typename Allocator>
class algorithm_base:private noncopyable
{
protected:
algorithm_base(const Allocator& al,std::size_t size):
spc(al,size),size_(size),n_(0),sorted(false)
{
}

void add(void* node)
{
entries()[n_]=entry(node,n_);
++n_;
}

void begin_algorithm()const
{
if(!sorted){
std::sort(entries(),entries()+size_,entry::less_by_node());
sorted=true;
}
num_piles=0;
}

void add_node_to_algorithm(void* node)const
{
entry* ent=
std::lower_bound(
entries(),entries()+size_,
entry(node),entry::less_by_node()); 
ent->ordered=false;
std::size_t n=ent->pos;                 

entry dummy(0);
dummy.pile_top=n;

entry* pile_ent=                        
std::lower_bound(                     
entries(),entries()+num_piles,
dummy,entry::less_by_pile_top());

pile_ent->pile_top=n;                   
pile_ent->pile_top_entry=ent;        


if(pile_ent>&entries()[0]){ 
ent->previous=(pile_ent-1)->pile_top_entry;
}

if(pile_ent==&entries()[num_piles]){    
++num_piles;
}
}

void finish_algorithm()const
{
if(num_piles>0){


entry* ent=entries()[num_piles-1].pile_top_entry;
for(std::size_t n=num_piles;n--;){
ent->ordered=true;
ent=ent->previous;
}
}
}

bool is_ordered(void * node)const
{
return std::lower_bound(
entries(),entries()+size_,
entry(node),entry::less_by_node())->ordered;
}

private:
entry* entries()const{return raw_ptr<entry*>(spc.data());}

auto_space<entry,Allocator> spc;
std::size_t                 size_;
std::size_t                 n_;
mutable bool                sorted;
mutable std::size_t         num_piles;
};



template<typename Node,typename Allocator>
class algorithm:private algorithm_base<Allocator>
{
typedef algorithm_base<Allocator> super;

public:
algorithm(const Allocator& al,std::size_t size):super(al,size){}

void add(Node* node)
{
super::add(node);
}

template<typename IndexIterator>
void execute(IndexIterator first,IndexIterator last)const
{
super::begin_algorithm();

for(IndexIterator it=first;it!=last;++it){
add_node_to_algorithm(get_node(it));
}

super::finish_algorithm();
}

bool is_ordered(Node* node)const
{
return super::is_ordered(node);
}

private:
void add_node_to_algorithm(Node* node)const
{
super::add_node_to_algorithm(node);
}

template<typename IndexIterator>
static Node* get_node(IndexIterator it)
{
return static_cast<Node*>(it.get_node());
}
};

} 

} 

} 

} 

#endif



#ifndef BOOST_MULTI_INDEX_DETAIL_COPY_MAP_HPP
#define BOOST_MULTI_INDEX_DETAIL_COPY_MAP_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <algorithm>
#include <boost/core/addressof.hpp>
#include <boost/core/no_exceptions_support.hpp>
#include <boost/move/core.hpp>
#include <boost/move/utility_core.hpp>
#include <boost/multi_index/detail/allocator_traits.hpp>
#include <boost/multi_index/detail/auto_space.hpp>
#include <boost/multi_index/detail/raw_ptr.hpp>
#include <boost/noncopyable.hpp>
#include <functional>

namespace boost{

namespace multi_index{

namespace detail{



template <typename Node>
struct copy_map_entry
{
copy_map_entry(Node* f,Node* s):first(f),second(s){}

Node* first;
Node* second;

bool operator<(const copy_map_entry<Node>& x)const
{
return std::less<Node*>()(first,x.first);
}
};

struct copy_map_value_copier
{
template<typename Value>
const Value& operator()(Value& x)const{return x;}
};

struct copy_map_value_mover
{
template<typename Value>
BOOST_RV_REF(Value) operator()(Value& x)const{return boost::move(x);}
};

template <typename Node,typename Allocator>
class copy_map:private noncopyable
{
typedef typename rebind_alloc_for<
Allocator,Node
>::type                                  allocator_type;
typedef allocator_traits<allocator_type> alloc_traits;
typedef typename alloc_traits::pointer   pointer;

public:
typedef const copy_map_entry<Node>*      const_iterator;
typedef typename alloc_traits::size_type size_type;

copy_map(
const Allocator& al,size_type size,Node* header_org,Node* header_cpy):
al_(al),size_(size),spc(al_,size_),n(0),
header_org_(header_org),header_cpy_(header_cpy),released(false)
{}

~copy_map()
{
if(!released){
for(size_type i=0;i<n;++i){
alloc_traits::destroy(
al_,boost::addressof((spc.data()+i)->second->value()));
deallocate((spc.data()+i)->second);
}
}
}

const_iterator begin()const{return raw_ptr<const_iterator>(spc.data());}
const_iterator end()const{return raw_ptr<const_iterator>(spc.data()+n);}

void copy_clone(Node* node){clone(node,copy_map_value_copier());}
void move_clone(Node* node){clone(node,copy_map_value_mover());}

Node* find(Node* node)const
{
if(node==header_org_)return header_cpy_;
return std::lower_bound(
begin(),end(),copy_map_entry<Node>(node,0))->second;
}

void release()
{
released=true;
}

private:
allocator_type                             al_;
size_type                                  size_;
auto_space<copy_map_entry<Node>,Allocator> spc;
size_type                                  n;
Node*                                      header_org_;
Node*                                      header_cpy_;
bool                                       released;

pointer allocate()
{
return alloc_traits::allocate(al_,1);
}

void deallocate(Node* node)
{
alloc_traits::deallocate(al_,static_cast<pointer>(node),1);
}

template<typename ValueAccess>
void clone(Node* node,ValueAccess access)
{
(spc.data()+n)->first=node;
(spc.data()+n)->second=raw_ptr<Node*>(allocate());
BOOST_TRY{
alloc_traits::construct(
al_,boost::addressof((spc.data()+n)->second->value()),
access(node->value()));
}
BOOST_CATCH(...){
deallocate((spc.data()+n)->second);
BOOST_RETHROW;
}
BOOST_CATCH_END
++n;

if(n==size_){
std::sort(
raw_ptr<copy_map_entry<Node>*>(spc.data()),
raw_ptr<copy_map_entry<Node>*>(spc.data())+size_);
}
}
};

} 

} 

} 

#endif

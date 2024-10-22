

#ifndef BOOST_MULTI_INDEX_DETAIL_HASH_INDEX_NODE_HPP
#define BOOST_MULTI_INDEX_DETAIL_HASH_INDEX_NODE_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/multi_index/detail/allocator_traits.hpp>
#include <boost/multi_index/detail/raw_ptr.hpp>
#include <utility>

namespace boost{

namespace multi_index{

namespace detail{



template<typename Allocator>
struct hashed_index_node_impl;



template<typename Allocator>
struct hashed_index_base_node_impl
{
typedef typename rebind_alloc_for<
Allocator,hashed_index_base_node_impl
>::type                                             base_allocator;
typedef typename rebind_alloc_for<
Allocator,hashed_index_node_impl<Allocator>
>::type                                             node_allocator;
typedef allocator_traits<base_allocator>            base_alloc_traits;
typedef allocator_traits<node_allocator>            node_alloc_traits;
typedef typename base_alloc_traits::pointer         base_pointer;
typedef typename base_alloc_traits::const_pointer   const_base_pointer;
typedef typename node_alloc_traits::pointer         pointer;
typedef typename node_alloc_traits::const_pointer   const_pointer;
typedef typename node_alloc_traits::difference_type difference_type;

pointer& prior(){return prior_;}
pointer  prior()const{return prior_;}

private:
pointer prior_;
};



template<typename Allocator>
struct hashed_index_node_impl:hashed_index_base_node_impl<Allocator>
{
private:
typedef hashed_index_base_node_impl<Allocator> super;

public:  
typedef typename super::base_pointer           base_pointer;
typedef typename super::const_base_pointer     const_base_pointer;
typedef typename super::pointer                pointer;
typedef typename super::const_pointer          const_pointer;

base_pointer& next(){return next_;}
base_pointer  next()const{return next_;}

static pointer pointer_from(base_pointer x)
{
return static_cast<pointer>(
static_cast<hashed_index_node_impl*>(
raw_ptr<super*>(x)));
}

static base_pointer base_pointer_from(pointer x)
{
return static_cast<base_pointer>(
raw_ptr<hashed_index_node_impl*>(x));
}

private:
base_pointer next_;
};



struct default_assigner
{
template<typename T> void operator()(T& x,const T& val){x=val;}
};

template<typename Node>
struct unlink_undo_assigner
{
typedef typename Node::base_pointer base_pointer;
typedef typename Node::pointer      pointer;

unlink_undo_assigner():pointer_track_count(0),base_pointer_track_count(0){}

void operator()(pointer& x,pointer val)
{
pointer_tracks[pointer_track_count].x=&x;
pointer_tracks[pointer_track_count++].val=x;
x=val;
}

void operator()(base_pointer& x,base_pointer val)
{
base_pointer_tracks[base_pointer_track_count].x=&x;
base_pointer_tracks[base_pointer_track_count++].val=x;
x=val;
}

void operator()() 
{


while(pointer_track_count--){
*(pointer_tracks[pointer_track_count].x)=
pointer_tracks[pointer_track_count].val;
}
while(base_pointer_track_count--){
*(base_pointer_tracks[base_pointer_track_count].x)=
base_pointer_tracks[base_pointer_track_count].val;
}
}

struct pointer_track     {pointer*      x; pointer      val;};
struct base_pointer_track{base_pointer* x; base_pointer val;};



pointer_track      pointer_tracks[3];
int                pointer_track_count;
base_pointer_track base_pointer_tracks[2];
int                base_pointer_track_count;
};



struct hashed_unique_tag{};
struct hashed_non_unique_tag{};

template<typename Node,typename Category>
struct hashed_index_node_alg;

template<typename Node>
struct hashed_index_node_alg<Node,hashed_unique_tag>
{
typedef typename Node::base_pointer       base_pointer;
typedef typename Node::const_base_pointer const_base_pointer;
typedef typename Node::pointer            pointer;
typedef typename Node::const_pointer      const_pointer;

static bool is_first_of_bucket(pointer x)
{
return x->prior()->next()!=base_pointer_from(x);
}

static pointer after(pointer x)
{
return is_last_of_bucket(x)?x->next()->prior():pointer_from(x->next());
}

static pointer after_local(pointer x)
{
return is_last_of_bucket(x)?pointer(0):pointer_from(x->next());
}

static pointer next_to_inspect(pointer x)
{
return is_last_of_bucket(x)?pointer(0):pointer_from(x->next());
}

static void link(pointer x,base_pointer buc,pointer end)
{
if(buc->prior()==pointer(0)){ 
x->prior()=end->prior();
x->next()=end->prior()->next();
x->prior()->next()=buc;
buc->prior()=x;
end->prior()=x;
}
else{
x->prior()=buc->prior()->prior();
x->next()=base_pointer_from(buc->prior());
buc->prior()=x;
x->next()->prior()=x;
}
}

static void unlink(pointer x)
{
default_assigner assign;
unlink(x,assign);
}

typedef unlink_undo_assigner<Node> unlink_undo;

template<typename Assigner>
static void unlink(pointer x,Assigner& assign)
{
if(is_first_of_bucket(x)){
if(is_last_of_bucket(x)){
assign(x->prior()->next()->prior(),pointer(0));
assign(x->prior()->next(),x->next());
assign(x->next()->prior()->prior(),x->prior());
}
else{
assign(x->prior()->next()->prior(),pointer_from(x->next()));
assign(x->next()->prior(),x->prior());
}
}
else if(is_last_of_bucket(x)){
assign(x->prior()->next(),x->next());
assign(x->next()->prior()->prior(),x->prior());
}
else{
assign(x->prior()->next(),x->next());
assign(x->next()->prior(),x->prior());
}
}



static void append(pointer x,pointer end)
{
x->prior()=end->prior();
x->next()=end->prior()->next();
x->prior()->next()=base_pointer_from(x);
end->prior()=x;
}

static bool unlink_last(pointer end)
{


pointer x=end->prior();
if(x->prior()->next()==base_pointer_from(x)){
x->prior()->next()=x->next();
end->prior()=x->prior();
return false;
}
else{
x->prior()->next()->prior()=pointer(0);
x->prior()->next()=x->next();
end->prior()=x->prior();
return true;
}
}

private:
static pointer pointer_from(base_pointer x)
{
return Node::pointer_from(x);
}

static base_pointer base_pointer_from(pointer x)
{
return Node::base_pointer_from(x);
}

static bool is_last_of_bucket(pointer x)
{
return x->next()->prior()!=x;
}
};

template<typename Node>
struct hashed_index_node_alg<Node,hashed_non_unique_tag>
{
typedef typename Node::base_pointer       base_pointer;
typedef typename Node::const_base_pointer const_base_pointer;
typedef typename Node::pointer            pointer;
typedef typename Node::const_pointer      const_pointer;

static bool is_first_of_bucket(pointer x)
{
return x->prior()->next()->prior()==x;
}

static bool is_first_of_group(pointer x)
{
return
x->next()->prior()!=x&&
x->next()->prior()->prior()->next()==base_pointer_from(x);
}

static pointer after(pointer x)
{
if(x->next()->prior()==x)return pointer_from(x->next());
if(x->next()->prior()->prior()==x)return x->next()->prior();
if(x->next()->prior()->prior()->next()==base_pointer_from(x))
return pointer_from(x->next());
return pointer_from(x->next())->next()->prior();
}

static pointer after_local(pointer x)
{
if(x->next()->prior()==x)return pointer_from(x->next());
if(x->next()->prior()->prior()==x)return pointer(0);
if(x->next()->prior()->prior()->next()==base_pointer_from(x))
return pointer_from(x->next());
return pointer_from(x->next())->next()->prior();
}

static pointer next_to_inspect(pointer x)
{
if(x->next()->prior()==x)return pointer_from(x->next());
if(x->next()->prior()->prior()==x)return pointer(0);
if(x->next()->prior()->next()->prior()!=x->next()->prior())
return pointer(0);
return pointer_from(x->next()->prior()->next());
}

static void link(pointer x,base_pointer buc,pointer end)
{
if(buc->prior()==pointer(0)){ 
x->prior()=end->prior();
x->next()=end->prior()->next();
x->prior()->next()=buc;
buc->prior()=x;
end->prior()=x;
}
else{
x->prior()=buc->prior()->prior();
x->next()=base_pointer_from(buc->prior());
buc->prior()=x;
x->next()->prior()=x;
}
}

static void link(pointer x,pointer first,pointer last)
{
x->prior()=first->prior();
x->next()=base_pointer_from(first);
if(is_first_of_bucket(first)){
x->prior()->next()->prior()=x;
}
else{
x->prior()->next()=base_pointer_from(x);
}

if(first==last){
last->prior()=x;
}
else if(first->next()==base_pointer_from(last)){
first->prior()=last;
first->next()=base_pointer_from(x);
}
else{
pointer second=pointer_from(first->next()),
lastbutone=last->prior();
second->prior()=first;
first->prior()=last;
lastbutone->next()=base_pointer_from(x);
}
}

static void unlink(pointer x)
{
default_assigner assign;
unlink(x,assign);
}

typedef unlink_undo_assigner<Node> unlink_undo;

template<typename Assigner>
static void unlink(pointer x,Assigner& assign)
{
if(x->prior()->next()==base_pointer_from(x)){
if(x->next()->prior()==x){
left_unlink(x,assign);
right_unlink(x,assign);
}
else if(x->next()->prior()->prior()==x){           
left_unlink(x,assign);
right_unlink_last_of_bucket(x,assign);
}
else if(x->next()->prior()->prior()->next()==
base_pointer_from(x)){                
left_unlink(x,assign);
right_unlink_first_of_group(x,assign);
}
else{                                                
unlink_last_but_one_of_group(x,assign);
}
}
else if(x->prior()->next()->prior()==x){            
if(x->next()->prior()==x){
left_unlink_first_of_bucket(x,assign);
right_unlink(x,assign);
}
else if(x->next()->prior()->prior()==x){           
assign(x->prior()->next()->prior(),pointer(0));
assign(x->prior()->next(),x->next());
assign(x->next()->prior()->prior(),x->prior());
}
else{                                              
left_unlink_first_of_bucket(x,assign);
right_unlink_first_of_group(x,assign);
}
}
else if(x->next()->prior()->prior()==x){   
left_unlink_last_of_group(x,assign);
right_unlink_last_of_bucket(x,assign);
}
else if(pointer_from(x->prior()->prior()->next())
->next()==base_pointer_from(x)){            
unlink_second_of_group(x,assign);
}
else{                              
left_unlink_last_of_group(x,assign);
right_unlink(x,assign);
}
}



static void link_range(
pointer first,pointer last,base_pointer buc,pointer cend)
{
if(buc->prior()==pointer(0)){ 
first->prior()=cend->prior();
last->next()=cend->prior()->next();
first->prior()->next()=buc;
buc->prior()=first;
cend->prior()=last;
}
else{
first->prior()=buc->prior()->prior();
last->next()=base_pointer_from(buc->prior());
buc->prior()=first;
last->next()->prior()=last;
}
}

static void append_range(pointer first,pointer last,pointer cend)
{
first->prior()=cend->prior();
last->next()=cend->prior()->next();
first->prior()->next()=base_pointer_from(first);
cend->prior()=last;
}

static std::pair<pointer,bool> unlink_last_group(pointer end)
{


pointer x=end->prior();
if(x->prior()->next()==base_pointer_from(x)){
x->prior()->next()=x->next();
end->prior()=x->prior();
return std::make_pair(x,false);
}
else if(x->prior()->next()->prior()==x){
x->prior()->next()->prior()=pointer(0);
x->prior()->next()=x->next();
end->prior()=x->prior();
return std::make_pair(x,true);
}
else{
pointer y=pointer_from(x->prior()->next());

if(y->prior()->next()==base_pointer_from(y)){
y->prior()->next()=x->next();
end->prior()=y->prior();
return std::make_pair(y,false);
}
else{
y->prior()->next()->prior()=pointer(0);
y->prior()->next()=x->next();
end->prior()=y->prior();
return std::make_pair(y,true);
}
}
}

static void unlink_range(pointer first,pointer last)
{
if(is_first_of_bucket(first)){
if(is_last_of_bucket(last)){
first->prior()->next()->prior()=pointer(0);
first->prior()->next()=last->next();
last->next()->prior()->prior()=first->prior();
}
else{
first->prior()->next()->prior()=pointer_from(last->next());
last->next()->prior()=first->prior();
}
}
else if(is_last_of_bucket(last)){
first->prior()->next()=last->next();
last->next()->prior()->prior()=first->prior();
}
else{
first->prior()->next()=last->next();
last->next()->prior()=first->prior();
}
}

private:
static pointer pointer_from(base_pointer x)
{
return Node::pointer_from(x);
}

static base_pointer base_pointer_from(pointer x)
{
return Node::base_pointer_from(x);
}

static bool is_last_of_bucket(pointer x)
{
return x->next()->prior()->prior()==x;
}

template<typename Assigner>
static void left_unlink(pointer x,Assigner& assign)
{
assign(x->prior()->next(),x->next());
}

template<typename Assigner>
static void right_unlink(pointer x,Assigner& assign)
{
assign(x->next()->prior(),x->prior());
}

template<typename Assigner>
static void left_unlink_first_of_bucket(pointer x,Assigner& assign)
{
assign(x->prior()->next()->prior(),pointer_from(x->next()));
}

template<typename Assigner>
static void right_unlink_last_of_bucket(pointer x,Assigner& assign)
{
assign(x->next()->prior()->prior(),x->prior());
}

template<typename Assigner>
static void right_unlink_first_of_group(pointer x,Assigner& assign)
{
pointer second=pointer_from(x->next()),
last=second->prior(),
lastbutone=last->prior();
if(second==lastbutone){
assign(second->next(),base_pointer_from(last));
assign(second->prior(),x->prior());
}
else{
assign(lastbutone->next(),base_pointer_from(second));
assign(second->next()->prior(),last);
assign(second->prior(),x->prior());
}
}

template<typename Assigner>
static void left_unlink_last_of_group(pointer x,Assigner& assign)
{
pointer lastbutone=x->prior(),
first=pointer_from(lastbutone->next()),
second=pointer_from(first->next());
if(lastbutone==second){
assign(lastbutone->prior(),first);
assign(lastbutone->next(),x->next());
}
else{
assign(second->prior(),lastbutone);
assign(lastbutone->prior()->next(),base_pointer_from(first));
assign(lastbutone->next(),x->next());
}
}

template<typename Assigner>
static void unlink_last_but_one_of_group(pointer x,Assigner& assign)
{
pointer first=pointer_from(x->next()),
second=pointer_from(first->next()),
last=second->prior();
if(second==x){
assign(last->prior(),first);
assign(first->next(),base_pointer_from(last));
}
else{
assign(last->prior(),x->prior());
assign(x->prior()->next(),base_pointer_from(first));
}
}

template<typename Assigner>
static void unlink_second_of_group(pointer x,Assigner& assign)
{
pointer last=x->prior(),
lastbutone=last->prior(),
first=pointer_from(lastbutone->next());
if(lastbutone==x){
assign(first->next(),base_pointer_from(last));
assign(last->prior(),first);
}
else{
assign(first->next(),x->next());
assign(x->next()->prior(),last);
}
}
};

template<typename Super>
struct hashed_index_node_trampoline:
hashed_index_node_impl<
typename rebind_alloc_for<
typename Super::allocator_type,char
>::type
>
{
typedef typename rebind_alloc_for<
typename Super::allocator_type,char
>::type                                             impl_allocator_type;
typedef hashed_index_node_impl<impl_allocator_type> impl_type;
};

template<typename Super>
struct hashed_index_node:
Super,hashed_index_node_trampoline<Super>
{
private:
typedef hashed_index_node_trampoline<Super> trampoline;

public:
typedef typename trampoline::impl_type          impl_type;
typedef typename trampoline::base_pointer       impl_base_pointer;
typedef typename trampoline::const_base_pointer const_impl_base_pointer;
typedef typename trampoline::pointer            impl_pointer;
typedef typename trampoline::const_pointer      const_impl_pointer;
typedef typename trampoline::difference_type    difference_type;

template<typename Category>
struct node_alg{
typedef hashed_index_node_alg<impl_type,Category> type;
};

impl_pointer&      prior(){return trampoline::prior();}
impl_pointer       prior()const{return trampoline::prior();}
impl_base_pointer& next(){return trampoline::next();}
impl_base_pointer  next()const{return trampoline::next();}

impl_pointer impl()
{
return static_cast<impl_pointer>(
static_cast<impl_type*>(static_cast<trampoline*>(this)));
}

const_impl_pointer impl()const
{
return static_cast<const_impl_pointer>(
static_cast<const impl_type*>(static_cast<const trampoline*>(this)));
}

static hashed_index_node* from_impl(impl_pointer x)
{
return
static_cast<hashed_index_node*>(
static_cast<trampoline*>(
raw_ptr<impl_type*>(x)));
}

static const hashed_index_node* from_impl(const_impl_pointer x)
{
return 
static_cast<const hashed_index_node*>(
static_cast<const trampoline*>(
raw_ptr<const impl_type*>(x)));
}



template<typename Category>
static void increment(hashed_index_node*& x)
{
x=from_impl(node_alg<Category>::type::after(x->impl()));
}

template<typename Category>
static void increment_local(hashed_index_node*& x)
{
x=from_impl(node_alg<Category>::type::after_local(x->impl()));
}
};

} 

} 

} 

#endif

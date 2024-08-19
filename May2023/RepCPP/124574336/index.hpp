#ifndef BOOST_LOCALE_BOUNDARY_INDEX_HPP_INCLUDED
#define BOOST_LOCALE_BOUNDARY_INDEX_HPP_INCLUDED

#include <boost/locale/config.hpp>
#include <boost/locale/boundary/types.hpp>
#include <boost/locale/boundary/facets.hpp>
#include <boost/locale/boundary/segment.hpp>
#include <boost/locale/boundary/boundary_point.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/cstdint.hpp>
#include <boost/assert.hpp>
#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4275 4251 4231 4660)
#endif
#include <string>
#include <locale>
#include <vector>
#include <iterator>
#include <algorithm>
#include <stdexcept>

#include <iostream>

namespace boost {

namespace locale {

namespace boundary {


namespace details {

template<typename IteratorType,typename CategoryType = typename std::iterator_traits<IteratorType>::iterator_category>
struct mapping_traits {
typedef typename std::iterator_traits<IteratorType>::value_type char_type;
static index_type map(boundary_type t,IteratorType b,IteratorType e,std::locale const &l)
{
std::basic_string<char_type> str(b,e);
return std::use_facet<boundary_indexing<char_type> >(l).map(t,str.c_str(),str.c_str()+str.size());
}
};

template<typename CharType,typename SomeIteratorType>
struct linear_iterator_traits {
static const bool is_linear =
is_same<SomeIteratorType,CharType*>::value
|| is_same<SomeIteratorType,CharType const*>::value
|| is_same<SomeIteratorType,typename std::basic_string<CharType>::iterator>::value
|| is_same<SomeIteratorType,typename std::basic_string<CharType>::const_iterator>::value
|| is_same<SomeIteratorType,typename std::vector<CharType>::iterator>::value
|| is_same<SomeIteratorType,typename std::vector<CharType>::const_iterator>::value
;
};



template<typename IteratorType>
struct mapping_traits<IteratorType,std::random_access_iterator_tag> {

typedef typename std::iterator_traits<IteratorType>::value_type char_type;



static index_type map(boundary_type t,IteratorType b,IteratorType e,std::locale const &l)
{
index_type result;


if(linear_iterator_traits<char_type,IteratorType>::is_linear && b!=e)
{
char_type const *begin = &*b;
char_type const *end = begin + (e-b);
index_type tmp=std::use_facet<boundary_indexing<char_type> >(l).map(t,begin,end);
result.swap(tmp);
}
else {
std::basic_string<char_type> str(b,e);
index_type tmp = std::use_facet<boundary_indexing<char_type> >(l).map(t,str.c_str(),str.c_str()+str.size());
result.swap(tmp);
}
return result;
}
};

template<typename BaseIterator>
class mapping {
public:
typedef BaseIterator base_iterator;
typedef typename std::iterator_traits<base_iterator>::value_type char_type;


mapping(boundary_type type,
base_iterator begin,
base_iterator end,
std::locale const &loc) 
:   
index_(new index_type()),
begin_(begin),
end_(end)
{
index_type idx=details::mapping_traits<base_iterator>::map(type,begin,end,loc);
index_->swap(idx);
}

mapping()
{
}

index_type const &index() const
{
return *index_;
}

base_iterator begin() const
{
return begin_;
}

base_iterator end() const
{
return end_;
}

private:
boost::shared_ptr<index_type> index_;
base_iterator begin_,end_;
};

template<typename BaseIterator>
class segment_index_iterator : 
public boost::iterator_facade<
segment_index_iterator<BaseIterator>,
segment<BaseIterator>,
boost::bidirectional_traversal_tag,
segment<BaseIterator> const &
>
{
public:
typedef BaseIterator base_iterator;
typedef mapping<base_iterator> mapping_type;
typedef segment<base_iterator> segment_type;

segment_index_iterator() : current_(0,0),map_(0)
{
}

segment_index_iterator(base_iterator p,mapping_type const *map,rule_type mask,bool full_select) :
map_(map),
mask_(mask),
full_select_(full_select)
{
set(p);
}
segment_index_iterator(bool is_begin,mapping_type const *map,rule_type mask,bool full_select) :
map_(map),
mask_(mask),
full_select_(full_select)
{
if(is_begin)
set_begin();
else
set_end();
}

segment_type const &dereference() const
{
return value_;
}

bool equal(segment_index_iterator const &other) const
{
return map_ == other.map_ && current_.second == other.current_.second;
}

void increment()
{
std::pair<size_t,size_t> next = current_;
if(full_select_) {
next.first = next.second;
while(next.second < size()) {
next.second++;
if(valid_offset(next.second))
break;
}
if(next.second == size())
next.first = next.second - 1;
}
else {
while(next.second < size()) {
next.first = next.second;
next.second++;
if(valid_offset(next.second))
break;
}
}
update_current(next);
}

void decrement()
{
std::pair<size_t,size_t> next = current_;
if(full_select_) {
while(next.second >1) {
next.second--;
if(valid_offset(next.second))
break;
}
next.first = next.second;
while(next.first >0) {
next.first--;
if(valid_offset(next.first))
break;
}
}
else {
while(next.second >1) {
next.second--;
if(valid_offset(next.second))
break;
}
next.first = next.second - 1;
}
update_current(next);
}

private:

void set_end()
{
current_.first  = size() - 1;
current_.second = size();
value_ = segment_type(map_->end(),map_->end(),0);
}
void set_begin()
{
current_.first = current_.second = 0;
value_ = segment_type(map_->begin(),map_->begin(),0);
increment();
}

void set(base_iterator p)
{
size_t dist=std::distance(map_->begin(),p);
index_type::const_iterator b=map_->index().begin(),e=map_->index().end();
index_type::const_iterator 
boundary_point=std::upper_bound(b,e,break_info(dist));
while(boundary_point != e && (boundary_point->rule & mask_)==0)
boundary_point++;

current_.first = current_.second = boundary_point - b;

if(full_select_) {
while(current_.first > 0) {
current_.first --;
if(valid_offset(current_.first))
break;
}
}
else {
if(current_.first > 0)
current_.first --;
}
value_.first = map_->begin();
std::advance(value_.first,get_offset(current_.first));
value_.second = value_.first;
std::advance(value_.second,get_offset(current_.second) - get_offset(current_.first));

update_rule();
}

void update_current(std::pair<size_t,size_t> pos)
{
std::ptrdiff_t first_diff = get_offset(pos.first) - get_offset(current_.first);
std::ptrdiff_t second_diff = get_offset(pos.second) - get_offset(current_.second);
std::advance(value_.first,first_diff);
std::advance(value_.second,second_diff);
current_ = pos;
update_rule();
}

void update_rule()
{
if(current_.second != size()) {
value_.rule(index()[current_.second].rule);
}
}
size_t get_offset(size_t ind) const
{
if(ind == size())
return index().back().offset;
return index()[ind].offset;
}

bool valid_offset(size_t offset) const
{
return  offset == 0 
|| offset == size() 
|| (index()[offset].rule & mask_)!=0;
}

size_t size() const
{
return index().size();
}

index_type const &index() const
{
return map_->index();
}


segment_type value_;
std::pair<size_t,size_t> current_;
mapping_type const *map_;
rule_type mask_;
bool full_select_;
};

template<typename BaseIterator>
class boundary_point_index_iterator : 
public boost::iterator_facade<
boundary_point_index_iterator<BaseIterator>,
boundary_point<BaseIterator>,
boost::bidirectional_traversal_tag,
boundary_point<BaseIterator> const &
>
{
public:
typedef BaseIterator base_iterator;
typedef mapping<base_iterator> mapping_type;
typedef boundary_point<base_iterator> boundary_point_type;

boundary_point_index_iterator() : current_(0),map_(0)
{
}

boundary_point_index_iterator(bool is_begin,mapping_type const *map,rule_type mask) :
map_(map),
mask_(mask)
{
if(is_begin)
set_begin();
else
set_end();
}
boundary_point_index_iterator(base_iterator p,mapping_type const *map,rule_type mask) :
map_(map),
mask_(mask)
{
set(p);
}

boundary_point_type const &dereference() const
{
return value_;
}

bool equal(boundary_point_index_iterator const &other) const
{
return map_ == other.map_ && current_ == other.current_;
}

void increment()
{
size_t next = current_;
while(next < size()) {
next++;
if(valid_offset(next))
break;
}
update_current(next);
}

void decrement()
{
size_t next = current_;
while(next>0) {
next--;
if(valid_offset(next))
break;
}
update_current(next);
}

private:
void set_end()
{
current_ = size();
value_ = boundary_point_type(map_->end(),0);
}
void set_begin()
{
current_ = 0;
value_ = boundary_point_type(map_->begin(),0);
}

void set(base_iterator p)
{
size_t dist =  std::distance(map_->begin(),p);

index_type::const_iterator b=index().begin();
index_type::const_iterator e=index().end();
index_type::const_iterator ptr = std::lower_bound(b,e,break_info(dist));

if(ptr==index().end())
current_=size()-1;
else
current_=ptr - index().begin();

while(!valid_offset(current_))
current_ ++;

std::ptrdiff_t diff = get_offset(current_) - dist;
std::advance(p,diff);
value_.iterator(p);
update_rule();
}

void update_current(size_t pos)
{
std::ptrdiff_t diff = get_offset(pos) - get_offset(current_);
base_iterator i=value_.iterator();
std::advance(i,diff);
current_ = pos;
value_.iterator(i);
update_rule();
}

void update_rule()
{
if(current_ != size()) {
value_.rule(index()[current_].rule);
}
}
size_t get_offset(size_t ind) const
{
if(ind == size())
return index().back().offset;
return index()[ind].offset;
}

bool valid_offset(size_t offset) const
{
return  offset == 0 
|| offset + 1 >= size() 
|| (index()[offset].rule & mask_)!=0;
}

size_t size() const
{
return index().size();
}

index_type const &index() const
{
return map_->index();
}


boundary_point_type value_;
size_t current_;
mapping_type const *map_;
rule_type mask_;
};


} 


template<typename BaseIterator>
class segment_index;

template<typename BaseIterator>
class boundary_point_index;



template<typename BaseIterator>
class segment_index {
public:

typedef BaseIterator base_iterator;
#ifdef BOOST_LOCALE_DOXYGEN
typedef unspecified_iterator_type iterator;
typedef unspecified_iterator_type const_iterator;
#else
typedef details::segment_index_iterator<base_iterator> iterator;
typedef details::segment_index_iterator<base_iterator> const_iterator;
#endif
typedef segment<base_iterator> value_type;

segment_index() : mask_(0xFFFFFFFFu),full_select_(false)
{
}
segment_index(boundary_type type,
base_iterator begin,
base_iterator end,
rule_type mask,
std::locale const &loc=std::locale()) 
:
map_(type,begin,end,loc),
mask_(mask),
full_select_(false)
{
}
segment_index(boundary_type type,
base_iterator begin,
base_iterator end,
std::locale const &loc=std::locale()) 
:
map_(type,begin,end,loc),
mask_(0xFFFFFFFFu),
full_select_(false)
{
}

segment_index(boundary_point_index<base_iterator> const &);
segment_index const &operator = (boundary_point_index<base_iterator> const &);


void map(boundary_type type,base_iterator begin,base_iterator end,std::locale const &loc=std::locale())
{
map_ = mapping_type(type,begin,end,loc);
}

iterator begin() const
{
return iterator(true,&map_,mask_,full_select_);
}

iterator end() const
{
return iterator(false,&map_,mask_,full_select_);
}

iterator find(base_iterator p) const
{
return iterator(p,&map_,mask_,full_select_);
}

rule_type rule() const
{
return mask_;
}
void rule(rule_type v)
{
mask_ = v;
}


bool full_select()  const 
{
return full_select_;
}


void full_select(bool v) 
{
full_select_ = v;
}

private:
friend class boundary_point_index<base_iterator>;
typedef details::mapping<base_iterator> mapping_type;
mapping_type  map_;
rule_type mask_;
bool full_select_;
};



template<typename BaseIterator>
class boundary_point_index {
public:
typedef BaseIterator base_iterator;
#ifdef BOOST_LOCALE_DOXYGEN
typedef unspecified_iterator_type iterator;
typedef unspecified_iterator_type const_iterator;
#else
typedef details::boundary_point_index_iterator<base_iterator> iterator;
typedef details::boundary_point_index_iterator<base_iterator> const_iterator;
#endif
typedef boundary_point<base_iterator> value_type;

boundary_point_index() : mask_(0xFFFFFFFFu)
{
}

boundary_point_index(boundary_type type,
base_iterator begin,
base_iterator end,
rule_type mask,
std::locale const &loc=std::locale()) 
:
map_(type,begin,end,loc),
mask_(mask)
{
}
boundary_point_index(boundary_type type,
base_iterator begin,
base_iterator end,
std::locale const &loc=std::locale()) 
:
map_(type,begin,end,loc),
mask_(0xFFFFFFFFu)
{
}

boundary_point_index(segment_index<base_iterator> const &other);
boundary_point_index const &operator=(segment_index<base_iterator> const &other);

void map(boundary_type type,base_iterator begin,base_iterator end,std::locale const &loc=std::locale())
{
map_ = mapping_type(type,begin,end,loc);
}

iterator begin() const
{
return iterator(true,&map_,mask_);
}

iterator end() const
{
return iterator(false,&map_,mask_);
}

iterator find(base_iterator p) const
{
return iterator(p,&map_,mask_);
}

rule_type rule() const
{
return mask_;
}
void rule(rule_type v)
{
mask_ = v;
}

private:

friend class segment_index<base_iterator>;
typedef details::mapping<base_iterator> mapping_type;
mapping_type  map_;
rule_type mask_;
};

template<typename BaseIterator>
segment_index<BaseIterator>::segment_index(boundary_point_index<BaseIterator> const &other) :
map_(other.map_),
mask_(0xFFFFFFFFu),
full_select_(false)
{
}

template<typename BaseIterator>
boundary_point_index<BaseIterator>::boundary_point_index(segment_index<BaseIterator> const &other) :
map_(other.map_),
mask_(0xFFFFFFFFu)
{
}

template<typename BaseIterator>
segment_index<BaseIterator> const &segment_index<BaseIterator>::operator=(boundary_point_index<BaseIterator> const &other)
{
map_ = other.map_;
return *this;
}

template<typename BaseIterator>
boundary_point_index<BaseIterator> const &boundary_point_index<BaseIterator>::operator=(segment_index<BaseIterator> const &other)
{
map_ = other.map_;
return *this;
}

typedef segment_index<std::string::const_iterator> ssegment_index;      
typedef segment_index<std::wstring::const_iterator> wssegment_index;    
#ifdef BOOST_LOCALE_ENABLE_CHAR16_T
typedef segment_index<std::u16string::const_iterator> u16ssegment_index;
#endif
#ifdef BOOST_LOCALE_ENABLE_CHAR32_T
typedef segment_index<std::u32string::const_iterator> u32ssegment_index;
#endif

typedef segment_index<char const *> csegment_index;                     
typedef segment_index<wchar_t const *> wcsegment_index;                 
#ifdef BOOST_LOCALE_ENABLE_CHAR16_T
typedef segment_index<char16_t const *> u16csegment_index;              
#endif
#ifdef BOOST_LOCALE_ENABLE_CHAR32_T
typedef segment_index<char32_t const *> u32csegment_index;              
#endif

typedef boundary_point_index<std::string::const_iterator> sboundary_point_index;
typedef boundary_point_index<std::wstring::const_iterator> wsboundary_point_index;
#ifdef BOOST_LOCALE_ENABLE_CHAR16_T
typedef boundary_point_index<std::u16string::const_iterator> u16sboundary_point_index;
#endif
#ifdef BOOST_LOCALE_ENABLE_CHAR32_T
typedef boundary_point_index<std::u32string::const_iterator> u32sboundary_point_index;
#endif

typedef boundary_point_index<char const *> cboundary_point_index;       
typedef boundary_point_index<wchar_t const *> wcboundary_point_index;   
#ifdef BOOST_LOCALE_ENABLE_CHAR16_T
typedef boundary_point_index<char16_t const *> u16cboundary_point_index;
#endif
#ifdef BOOST_LOCALE_ENABLE_CHAR32_T
typedef boundary_point_index<char32_t const *> u32cboundary_point_index;
#endif



} 

} 
} 


#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif

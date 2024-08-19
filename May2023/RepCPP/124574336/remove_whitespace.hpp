#ifndef BOOST_ARCHIVE_ITERATORS_REMOVE_WHITESPACE_HPP
#define BOOST_ARCHIVE_ITERATORS_REMOVE_WHITESPACE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/assert.hpp>

#include <boost/iterator/iterator_adaptor.hpp>
#include <boost/iterator/filter_iterator.hpp>
#include <boost/iterator/iterator_traits.hpp>


#ifndef BOOST_NO_CWCTYPE
#include <cwctype> 
#if defined(BOOST_NO_STDC_NAMESPACE)
namespace std{ using ::iswspace; }
#endif
#endif

#include <cctype> 
#if defined(BOOST_NO_STDC_NAMESPACE)
namespace std{ using ::isspace; }
#endif

#if defined(__STD_RWCOMPILER_H__) || defined(_RWSTD_VER)
#undef isspace
#undef iswspace
#endif

namespace { 

template<class CharType>
struct remove_whitespace_predicate;

template<>
struct remove_whitespace_predicate<char>
{
bool operator()(unsigned char t){
return ! std::isspace(t);
}
};

#ifndef BOOST_NO_CWCHAR
template<>
struct remove_whitespace_predicate<wchar_t>
{
bool operator()(wchar_t t){
return ! std::iswspace(t);
}
};
#endif

} 


namespace boost {
namespace archive {
namespace iterators {


template<class Predicate, class Base>
class filter_iterator
: public boost::iterator_adaptor<
filter_iterator<Predicate, Base>,
Base,
use_default,
single_pass_traversal_tag
>
{
friend class boost::iterator_core_access;
typedef typename boost::iterator_adaptor<
filter_iterator<Predicate, Base>,
Base,
use_default,
single_pass_traversal_tag
> super_t;
typedef filter_iterator<Predicate, Base> this_t;
typedef typename super_t::reference reference_type;

reference_type dereference_impl(){
if(! m_full){
while(! m_predicate(* this->base_reference()))
++(this->base_reference());
m_full = true;
}
return * this->base_reference();
}

reference_type dereference() const {
return const_cast<this_t *>(this)->dereference_impl();
}

Predicate m_predicate;
bool m_full;
public:
void increment(){
m_full = false;
++(this->base_reference());
}
filter_iterator(Base start) :
super_t(start),
m_full(false)
{}
filter_iterator(){}
};

template<class Base>
class remove_whitespace :
public filter_iterator<
remove_whitespace_predicate<
typename boost::iterator_value<Base>::type
>,
Base
>
{
friend class boost::iterator_core_access;
typedef filter_iterator<
remove_whitespace_predicate<
typename boost::iterator_value<Base>::type
>,
Base
> super_t;
public:
template<class T>
remove_whitespace(T start) :
super_t(Base(static_cast< T >(start)))
{}
remove_whitespace(const remove_whitespace & rhs) :
super_t(rhs.base_reference())
{}
};

} 
} 
} 

#endif 

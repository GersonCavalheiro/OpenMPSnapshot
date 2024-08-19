#ifndef BOOST_ARCHIVE_ITERATORS_INSERT_LINEBREAKS_HPP
#define BOOST_ARCHIVE_ITERATORS_INSERT_LINEBREAKS_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/assert.hpp>

#include <boost/config.hpp>
#if defined(BOOST_NO_STDC_NAMESPACE)
namespace std{ using ::memcpy; }
#endif

#include <boost/iterator/iterator_adaptor.hpp>
#include <boost/iterator/iterator_traits.hpp>

namespace boost {
namespace archive {
namespace iterators {

template<
class Base,
int N,
class CharType = typename boost::iterator_value<Base>::type
>
class insert_linebreaks :
public iterator_adaptor<
insert_linebreaks<Base, N, CharType>,
Base,
CharType,
single_pass_traversal_tag,
CharType
>
{
private:
friend class boost::iterator_core_access;
typedef iterator_adaptor<
insert_linebreaks<Base, N, CharType>,
Base,
CharType,
single_pass_traversal_tag,
CharType
> super_t;

bool equal(const insert_linebreaks<Base, N, CharType> & rhs) const {
return
this->base_reference() == rhs.base_reference()
;
}

void increment() {
if(m_count == N){
m_count = 0;
return;
}
++m_count;
++(this->base_reference());
}
CharType dereference() const {
if(m_count == N)
return '\n';
return * (this->base_reference());
}
unsigned int m_count;
public:
template<class T>
insert_linebreaks(T  start) :
super_t(Base(static_cast< T >(start))),
m_count(0)
{}
insert_linebreaks(const insert_linebreaks & rhs) :
super_t(rhs.base_reference()),
m_count(rhs.m_count)
{}
};

} 
} 
} 

#endif 

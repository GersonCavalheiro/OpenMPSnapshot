#ifndef BOOST_ARCHIVE_ITERATORS_MB_FROM_WCHAR_HPP
#define BOOST_ARCHIVE_ITERATORS_MB_FROM_WCHAR_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/assert.hpp>
#include <cstddef> 
#ifndef BOOST_NO_CWCHAR
#include <cwchar> 
#endif
#include <boost/config.hpp>
#if defined(BOOST_NO_STDC_NAMESPACE)
namespace std{
using ::mbstate_t;
} 
#endif

#include <boost/archive/detail/utf8_codecvt_facet.hpp>
#include <boost/iterator/iterator_adaptor.hpp>

namespace boost {
namespace archive {
namespace iterators {

template<class Base>    
class mb_from_wchar
: public boost::iterator_adaptor<
mb_from_wchar<Base>,
Base,
wchar_t,
single_pass_traversal_tag,
char
>
{
friend class boost::iterator_core_access;

typedef typename boost::iterator_adaptor<
mb_from_wchar<Base>,
Base,
wchar_t,
single_pass_traversal_tag,
char
> super_t;

typedef mb_from_wchar<Base> this_t;

char dereference_impl() {
if(! m_full){
fill();
m_full = true;
}
return m_buffer[m_bnext];
}

char dereference() const {
return (const_cast<this_t *>(this))->dereference_impl();
}
bool equal(const mb_from_wchar<Base> & rhs) const {
return
0 == m_bend
&& 0 == m_bnext
&& this->base_reference() == rhs.base_reference()
;
}

void fill(){
wchar_t value = * this->base_reference();
const wchar_t *wend;
char *bend;
BOOST_VERIFY(
m_codecvt_facet.out(
m_mbs,
& value, & value + 1, wend,
m_buffer, m_buffer + sizeof(m_buffer), bend
)
==
std::codecvt_base::ok
);
m_bnext = 0;
m_bend = bend - m_buffer;
}

void increment(){
if(++m_bnext < m_bend)
return;
m_bend =
m_bnext = 0;
++(this->base_reference());
m_full = false;
}

boost::archive::detail::utf8_codecvt_facet m_codecvt_facet;
std::mbstate_t m_mbs;
char m_buffer[9 ];
std::size_t m_bend;
std::size_t m_bnext;
bool m_full;

public:
template<class T>
mb_from_wchar(T start) :
super_t(Base(static_cast< T >(start))),
m_mbs(std::mbstate_t()),
m_bend(0),
m_bnext(0),
m_full(false)
{}
mb_from_wchar(const mb_from_wchar & rhs) :
super_t(rhs.base_reference()),
m_bend(rhs.m_bend),
m_bnext(rhs.m_bnext),
m_full(rhs.m_full)
{}
};

} 
} 
} 

#endif 

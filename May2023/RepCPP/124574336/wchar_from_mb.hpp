#ifndef BOOST_ARCHIVE_ITERATORS_WCHAR_FROM_MB_HPP
#define BOOST_ARCHIVE_ITERATORS_WCHAR_FROM_MB_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <cctype>
#include <cstddef> 
#ifndef BOOST_NO_CWCHAR
#include <cwchar>  
#endif
#include <algorithm> 

#include <boost/config.hpp>
#if defined(BOOST_NO_STDC_NAMESPACE)
namespace std{
using ::mbstate_t;
} 
#endif
#include <boost/assert.hpp>
#include <boost/core/ignore_unused.hpp>
#include <boost/array.hpp>
#include <boost/iterator/iterator_adaptor.hpp>
#include <boost/archive/detail/utf8_codecvt_facet.hpp>
#include <boost/archive/iterators/dataflow_exception.hpp>
#include <boost/serialization/throw_exception.hpp>

#include <iostream>

namespace boost {
namespace archive {
namespace iterators {

template<class Base>
class wchar_from_mb
: public boost::iterator_adaptor<
wchar_from_mb<Base>,
Base,
wchar_t,
single_pass_traversal_tag,
wchar_t
>
{
friend class boost::iterator_core_access;
typedef typename boost::iterator_adaptor<
wchar_from_mb<Base>,
Base,
wchar_t,
single_pass_traversal_tag,
wchar_t
> super_t;

typedef wchar_from_mb<Base> this_t;

void drain();

wchar_t dereference() const {
if(m_output.m_next == m_output.m_next_available)
return static_cast<wchar_t>(0);
return * m_output.m_next;
}

void increment(){
if(m_output.m_next == m_output.m_next_available)
return;
if(++m_output.m_next == m_output.m_next_available){
if(m_input.m_done)
return;
drain();
}
}

bool equal(this_t const & rhs) const {
return dereference() == rhs.dereference();
}

boost::archive::detail::utf8_codecvt_facet m_codecvt_facet;
std::mbstate_t m_mbs;

template<typename T>
struct sliding_buffer {
boost::array<T, 32> m_buffer;
typename boost::array<T, 32>::const_iterator m_next_available;
typename boost::array<T, 32>::iterator m_next;
bool m_done;
sliding_buffer() :
m_next_available(m_buffer.begin()),
m_next(m_buffer.begin()),
m_done(false)
{}
sliding_buffer(const sliding_buffer & rhs) :
m_next_available(
std::copy(
rhs.m_buffer.begin(),
rhs.m_next_available,
m_buffer.begin()
)
),
m_next(
m_buffer.begin() + (rhs.m_next - rhs.m_buffer.begin())
),
m_done(rhs.m_done)
{}
};

sliding_buffer<typename iterator_value<Base>::type> m_input;
sliding_buffer<typename iterator_value<this_t>::type> m_output;

public:
template<class T>
wchar_from_mb(T start) :
super_t(Base(static_cast< T >(start))),
m_mbs(std::mbstate_t())
{
BOOST_ASSERT(std::mbsinit(&m_mbs));
drain();
}
wchar_from_mb(){}

wchar_from_mb(const wchar_from_mb & rhs) :
super_t(rhs.base_reference()),
m_mbs(rhs.m_mbs),
m_input(rhs.m_input),
m_output(rhs.m_output)
{}
};

template<class Base>
void wchar_from_mb<Base>::drain(){
BOOST_ASSERT(! m_input.m_done);
for(;;){
typename boost::iterators::iterator_reference<Base>::type c = *(this->base_reference());
if(0 == c){
m_input.m_done = true;
break;
}
++(this->base_reference());
* const_cast<typename iterator_value<Base>::type *>(
(m_input.m_next_available++)
) = c;
if(m_input.m_buffer.end() == m_input.m_next_available)
break;
}
const typename boost::iterators::iterator_value<Base>::type * input_new_start;
typename iterator_value<this_t>::type * next_available;

BOOST_ATTRIBUTE_UNUSED 
std::codecvt_base::result r = m_codecvt_facet.in(
m_mbs,
m_input.m_buffer.begin(),
m_input.m_next_available,
input_new_start,
m_output.m_buffer.begin(),
m_output.m_buffer.end(),
next_available
);
BOOST_ASSERT(std::codecvt_base::ok == r);
m_output.m_next_available = next_available;
m_output.m_next = m_output.m_buffer.begin();

m_input.m_next_available = std::copy(
input_new_start,
m_input.m_next_available,
m_input.m_buffer.begin()
);
m_input.m_next = m_input.m_buffer.begin();
}

} 
} 
} 

#endif 

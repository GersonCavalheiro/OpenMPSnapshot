#ifndef BOOST_ARCHIVE_ITERATORS_ISTREAM_ITERATOR_HPP
#define BOOST_ARCHIVE_ITERATORS_ISTREAM_ITERATOR_HPP

#if defined(_MSC_VER)
# pragma once
#endif





#include <cstddef> 
#include <istream>
#include <boost/iterator/iterator_facade.hpp>

namespace boost {
namespace archive {
namespace iterators {

template<class Elem = char>
class istream_iterator :
public boost::iterator_facade<
istream_iterator<Elem>,
Elem,
std::input_iterator_tag,
Elem
>
{
friend class boost::iterator_core_access;
typedef istream_iterator this_t ;
typedef typename boost::iterator_facade<
istream_iterator<Elem>,
Elem,
std::input_iterator_tag,
Elem
> super_t;
typedef typename std::basic_istream<Elem> istream_type;

bool equal(const this_t & rhs) const {
return m_istream == rhs.m_istream;
}

Elem dereference() const {
return static_cast<Elem>(m_istream->peek());
}

void increment(){
if(NULL != m_istream){
m_istream->ignore(1);
}
}

istream_type *m_istream;
Elem m_current_value;
public:
istream_iterator(istream_type & is) :
m_istream(& is)
{
}

istream_iterator() :
m_istream(NULL),
m_current_value(NULL)
{}

istream_iterator(const istream_iterator<Elem> & rhs) :
m_istream(rhs.m_istream),
m_current_value(rhs.m_current_value)
{}
};

} 
} 
} 

#endif 

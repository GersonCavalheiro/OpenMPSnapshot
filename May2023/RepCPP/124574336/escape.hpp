#ifndef BOOST_ARCHIVE_ITERATORS_ESCAPE_HPP
#define BOOST_ARCHIVE_ITERATORS_ESCAPE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/assert.hpp>
#include <cstddef> 

#include <boost/iterator/iterator_adaptor.hpp>
#include <boost/iterator/iterator_traits.hpp>

namespace boost {
namespace archive {
namespace iterators {


template<class Derived, class Base>
class escape :
public boost::iterator_adaptor<
Derived,
Base,
typename boost::iterator_value<Base>::type,
single_pass_traversal_tag,
typename boost::iterator_value<Base>::type
>
{
typedef typename boost::iterator_value<Base>::type base_value_type;
typedef typename boost::iterator_reference<Base>::type reference_type;
friend class boost::iterator_core_access;

typedef typename boost::iterator_adaptor<
Derived,
Base,
base_value_type,
single_pass_traversal_tag,
base_value_type
> super_t;

typedef escape<Derived, Base> this_t;

void dereference_impl() {
m_current_value = static_cast<Derived *>(this)->fill(m_bnext, m_bend);
m_full = true;
}

reference_type dereference() const {
if(!m_full)
const_cast<this_t *>(this)->dereference_impl();
return m_current_value;
}

bool equal(const this_t & rhs) const {
if(m_full){
if(! rhs.m_full)
const_cast<this_t *>(& rhs)->dereference_impl();
}
else{
if(rhs.m_full)
const_cast<this_t *>(this)->dereference_impl();
}
if(m_bnext != rhs.m_bnext)
return false;
if(this->base_reference() != rhs.base_reference())
return false;
return true;
}

void increment(){
if(++m_bnext < m_bend){
m_current_value = *m_bnext;
return;
}
++(this->base_reference());
m_bnext = NULL;
m_bend = NULL;
m_full = false;
}

const base_value_type *m_bnext;
const base_value_type *m_bend;
bool m_full;
base_value_type m_current_value;
public:
escape(Base base) :
super_t(base),
m_bnext(NULL),
m_bend(NULL),
m_full(false),
m_current_value(0)
{
}
};

} 
} 
} 

#endif 

#ifndef BOOST_ARCHIVE_ITERATORS_UNESCAPE_HPP
#define BOOST_ARCHIVE_ITERATORS_UNESCAPE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/assert.hpp>

#include <boost/iterator/iterator_adaptor.hpp>
#include <boost/pointee.hpp>

namespace boost {
namespace archive {
namespace iterators {

template<class Derived, class Base>
class unescape
: public boost::iterator_adaptor<
unescape<Derived, Base>,
Base,
typename pointee<Base>::type,
single_pass_traversal_tag,
typename pointee<Base>::type
>
{
friend class boost::iterator_core_access;
typedef typename boost::iterator_adaptor<
unescape<Derived, Base>,
Base,
typename pointee<Base>::type,
single_pass_traversal_tag,
typename pointee<Base>::type
> super_t;

typedef unescape<Derived, Base> this_t;
public:
typedef typename this_t::value_type value_type;
typedef typename this_t::reference reference;
private:
value_type dereference_impl() {
if(! m_full){
m_current_value = static_cast<Derived *>(this)->drain();
m_full = true;
}
return m_current_value;
}

reference dereference() const {
return const_cast<this_t *>(this)->dereference_impl();
}

value_type m_current_value;
bool m_full;

void increment(){
++(this->base_reference());
dereference_impl();
m_full = false;
}

public:

unescape(Base base) :
super_t(base),
m_full(false)
{}

};

} 
} 
} 

#endif 

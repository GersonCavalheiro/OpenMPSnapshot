#ifndef BOOST_ARCHIVE_ITERATORS_DATAFLOW_HPP
#define BOOST_ARCHIVE_ITERATORS_DATAFLOW_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/assert.hpp>

#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/apply.hpp>
#include <boost/mpl/plus.hpp>
#include <boost/mpl/int.hpp>

#include <boost/type_traits/is_convertible.hpp>
#include <boost/type_traits/is_base_and_derived.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/iterator/iterator_adaptor.hpp>
#include <boost/iterator/iterator_traits.hpp>
#include <boost/static_assert.hpp>

namespace boost {
namespace archive {
namespace iterators {

struct tri_state {
enum state_enum {
is_false = false,
is_true = true,
is_indeterminant
} m_state;
operator bool (){
BOOST_ASSERT(is_indeterminant != m_state);
return is_true == m_state ? true : false;
}
tri_state & operator=(bool rhs) {
m_state = rhs ? is_true : is_false;
return *this;
}
tri_state(bool rhs) :
m_state(rhs ? is_true : is_false)
{}
tri_state(state_enum state) :
m_state(state)
{}
bool operator==(const tri_state & rhs) const {
return m_state == rhs.m_state;
}
bool operator!=(const tri_state & rhs) const {
return m_state != rhs.m_state;
}
};

template<class Derived>
class dataflow {
bool m_eoi;
protected:
tri_state equal(const Derived & rhs) const {
if(m_eoi && rhs.m_eoi)
return true;
if(m_eoi || rhs.m_eoi)
return false;
return tri_state(tri_state::is_indeterminant);
}
void eoi(bool tf){
m_eoi = tf;
}
bool eoi() const {
return m_eoi;
}
public:
dataflow(bool tf) :
m_eoi(tf)
{}
dataflow() : 
m_eoi(true)
{}
};

} 
} 
} 

#endif 

#pragma once

#include <cusp/array1d.h>
#include "tensor_traits.h"

namespace dg
{



template<class ThrustVector >
struct View
{
using iterator = std::conditional_t<std::is_const<ThrustVector>::value,
typename ThrustVector::const_iterator,
typename ThrustVector::iterator>;
using const_iterator = typename ThrustVector::const_iterator;
using pointer = std::conditional_t<std::is_const<ThrustVector>::value,
typename ThrustVector::const_pointer,
typename ThrustVector::pointer>;
using const_pointer = typename ThrustVector::const_pointer;
View( void): m_ptr(), m_size(0){}


template<class OtherView>
View( OtherView& src): m_ptr(src.data()), m_size(src.size()){}

template<class InputIterator>
View( InputIterator data, unsigned size): m_ptr(pointer(data)),m_size(size){ }


template<class InputIterator>
void construct( InputIterator data, unsigned size)
{
m_ptr = pointer(data);
m_size = size;
}


pointer data() const {
return m_ptr;
}

iterator begin() const{
return iterator(m_ptr);
}

const_iterator cbegin() const{
return const_iterator(m_ptr);
}

iterator end() const{
return iterator(m_ptr + m_size);
}

const_iterator cend() const{
return const_iterator(m_ptr + m_size);
}

unsigned size() const{
return m_size;
}


void swap( View& src){
std::swap( m_ptr, src.m_ptr);
std::swap( m_size, src.m_size);
}
private:
pointer m_ptr;
unsigned m_size;
};


template<class ThrustVector>
struct TensorTraits< View<ThrustVector>>
{
using value_type = get_value_type<ThrustVector>;
using tensor_category = ThrustVectorTag;
using execution_policy = get_execution_policy<ThrustVector>;
};

}

#ifndef BOOST_ARCHIVE_ITERATORS_TRANSFORM_WIDTH_HPP
#define BOOST_ARCHIVE_ITERATORS_TRANSFORM_WIDTH_HPP

#if defined(_MSC_VER)
# pragma once
#endif





#include <boost/iterator/iterator_adaptor.hpp>
#include <boost/iterator/iterator_traits.hpp>

#include <algorithm> 

namespace boost {
namespace archive {
namespace iterators {

template<
class Base,
int BitsOut,
int BitsIn,
class CharType = typename boost::iterator_value<Base>::type 
>
class transform_width :
public boost::iterator_adaptor<
transform_width<Base, BitsOut, BitsIn, CharType>,
Base,
CharType,
single_pass_traversal_tag,
CharType
>
{
friend class boost::iterator_core_access;
typedef typename boost::iterator_adaptor<
transform_width<Base, BitsOut, BitsIn, CharType>,
Base,
CharType,
single_pass_traversal_tag,
CharType
> super_t;

typedef transform_width<Base, BitsOut, BitsIn, CharType> this_t;
typedef typename iterator_value<Base>::type base_value_type;

void fill();

CharType dereference() const {
if(!m_buffer_out_full)
const_cast<this_t *>(this)->fill();
return m_buffer_out;
}

bool equal_impl(const this_t & rhs){
if(BitsIn < BitsOut) 
return this->base_reference() == rhs.base_reference();
else{
if(this->base_reference() == rhs.base_reference()){
m_end_of_sequence = true;
return 0 == m_remaining_bits;
}
return false;
}
}

bool equal(const this_t & rhs) const {
return const_cast<this_t *>(this)->equal_impl(rhs);
}

void increment(){
m_buffer_out_full = false;
}

bool m_buffer_out_full;
CharType m_buffer_out;

base_value_type m_buffer_in;

unsigned int m_remaining_bits;

bool m_end_of_sequence;

public:
template<class T>
transform_width(T start) :
super_t(Base(static_cast< T >(start))),
m_buffer_out_full(false),
m_buffer_out(0),
m_buffer_in(0),
m_remaining_bits(0),
m_end_of_sequence(false)
{}
transform_width(const transform_width & rhs) :
super_t(rhs.base_reference()),
m_buffer_out_full(rhs.m_buffer_out_full),
m_buffer_out(rhs.m_buffer_out),
m_buffer_in(rhs.m_buffer_in),
m_remaining_bits(rhs.m_remaining_bits),
m_end_of_sequence(false)
{}
};

template<
class Base,
int BitsOut,
int BitsIn,
class CharType
>
void transform_width<Base, BitsOut, BitsIn, CharType>::fill() {
unsigned int missing_bits = BitsOut;
m_buffer_out = 0;
do{
if(0 == m_remaining_bits){
if(m_end_of_sequence){
m_buffer_in = 0;
m_remaining_bits = missing_bits;
}
else{
m_buffer_in = * this->base_reference()++;
m_remaining_bits = BitsIn;
}
}

unsigned int i = (std::min)(missing_bits, m_remaining_bits);
base_value_type j = m_buffer_in >> (m_remaining_bits - i);
j &= (1 << i) - 1;
m_buffer_out <<= i;
m_buffer_out |= j;

missing_bits -= i;
m_remaining_bits -= i;
}while(0 < missing_bits);
m_buffer_out_full = true;
}

} 
} 
} 

#endif 

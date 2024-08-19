#ifndef  BOOST_SERIALIZATION_DETAIL_STACK_CONSTRUCTOR_HPP
#define BOOST_SERIALIZATION_DETAIL_STACK_CONSTRUCTOR_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/aligned_storage.hpp>
#include <boost/serialization/serialization.hpp>

namespace boost{
namespace serialization {
namespace detail {

template<typename T >
struct stack_allocate
{
T * address() {
return static_cast<T*>(storage_.address());
}
T & reference() {
return * address();
}
private:
typedef typename boost::aligned_storage<
sizeof(T),
boost::alignment_of<T>::value
> type;
type storage_;
};

template<class Archive, class T>
struct stack_construct : public stack_allocate<T>
{
stack_construct(Archive & ar, const unsigned int version){
boost::serialization::load_construct_data_adl(
ar,
this->address(),
version
);
}
~stack_construct(){
this->address()->~T(); 
}
};

} 
} 
} 

#endif 

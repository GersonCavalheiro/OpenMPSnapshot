

#ifndef BOOST_MOVE_HPP_INCLUDED
#define BOOST_MOVE_HPP_INCLUDED

namespace boost { namespace ptr_container_detail {

namespace move_ptrs {

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)    
#pragma warning(push)    
#pragma warning(disable:4512)  
#endif  

template<typename Ptr>
class move_source {
public:
move_source(Ptr& ptr) : ptr_(ptr) {}
Ptr& ptr() const { return ptr_; }
private:
Ptr& ptr_;
move_source(const Ptr&);
};

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)    
#pragma warning(pop)    
#endif  

} 


template<typename T>
move_ptrs::move_source<T> move(T& x) 
{ return move_ptrs::move_source<T>(x); }

} 
} 

#endif 

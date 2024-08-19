#pragma once

#include <memory>

namespace dg
{


template<class Cloneable>
struct ClonePtr
{
ClonePtr( std::nullptr_t value = nullptr):m_ptr(nullptr){}

ClonePtr( Cloneable* ptr): m_ptr(ptr){}


ClonePtr( const Cloneable& src) : m_ptr( src.clone() ) { }


ClonePtr( const ClonePtr& src) : m_ptr( src.m_ptr.get() == nullptr ? nullptr : src.m_ptr->clone() ) { }

ClonePtr& operator=( const ClonePtr& src)
{
ClonePtr tmp(src);
swap( *this, tmp );
return *this;
}

ClonePtr( ClonePtr&& src) noexcept : m_ptr( nullptr)
{
swap( *this, src); 
}

ClonePtr& operator=( ClonePtr&& src) noexcept
{
swap( *this, src );
return *this;
}

friend void swap( ClonePtr& first, ClonePtr& second)
{
std::swap(first.m_ptr,second.m_ptr);
}


void reset( Cloneable* ptr){
m_ptr.reset( ptr);
}
Cloneable* release() noexcept { m_ptr.release();}

void reset( const Cloneable& src){
ClonePtr tmp(src);
swap(*this, tmp);
}


Cloneable * get() {return m_ptr.get();}

const Cloneable* get() const {return m_ptr.get();}

Cloneable& operator*() { return *m_ptr;}
const Cloneable& operator*() const { return *m_ptr;}
Cloneable* operator->() { return m_ptr.operator->();}
const Cloneable* operator->()const { return m_ptr.operator->();}
explicit operator bool() const{ return (bool)m_ptr;}


private:
std::unique_ptr<Cloneable> m_ptr;
};


template< class T>
struct Buffer
{
Buffer(){
ptr = new T;
}
Buffer( const T& t){
ptr = new T(t);
}
Buffer( const Buffer& src){ 
ptr = new T(*src.ptr);
}
Buffer( Buffer&& t): ptr( t.ptr){ 
t.ptr = nullptr;
}
Buffer& operator=( Buffer src){ 
swap( *this, src);
return *this;
}
~Buffer(){
delete ptr; 
}
friend void swap( Buffer& first, Buffer& second) 
{
using std::swap;
swap( first.ptr, second.ptr);
}



T& data( )const { return *ptr;}

private:
T* ptr;
};

}

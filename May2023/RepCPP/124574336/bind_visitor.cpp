#include <boost/config.hpp>

#if defined(BOOST_MSVC)
#pragma warning(disable: 4786)  
#pragma warning(disable: 4710)  
#pragma warning(disable: 4711)  
#pragma warning(disable: 4514)  
#endif


#include <boost/bind/bind.hpp>
#include <boost/ref.hpp>
#include <iostream>
#include <typeinfo>

using namespace boost::placeholders;


struct visitor
{
template<class T> void operator()( boost::reference_wrapper<T> const & r ) const
{
std::cout << "Reference to " << typeid(T).name() << " @ " << &r.get() << " (with value " << r.get() << ")\n";
}

template<class T> void operator()( T const & t ) const
{
std::cout << "Value of type " << typeid(T).name() << " (with value " << t << ")\n";
}
};

int f(int & i, int & j, int)
{
++i;
--j;
return i + j;
}

int x = 2;
int y = 7;

int main()
{
using namespace boost;

visitor v;
visit_each(v, bind<int>(bind(f, ref(x), _1, 42), ref(y)), 0);
}

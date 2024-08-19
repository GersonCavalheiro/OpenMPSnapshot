#include <boost/config.hpp>

#if defined(BOOST_MSVC)
#pragma warning(disable: 4786)  
#pragma warning(disable: 4710)  
#pragma warning(disable: 4711)  
#pragma warning(disable: 4514)  
#endif


#include <boost/bind/bind.hpp>
#include <iostream>
#include <string>

using namespace boost::placeholders;

std::string f(std::string const & x)
{
return "f(" + x + ")";
}

std::string g(std::string const & x)
{
return "g(" + x + ")";
}

std::string h(std::string const & x, std::string const & y)
{
return "h(" + x + ", " + y + ")";
}

std::string k()
{
return "k()";
}

template<class F> void test(F f)
{
std::cout << f("x", "y") << '\n';
}

int main()
{
using namespace boost;


test( bind(f, bind(g, _1)) );


test( bind(f, bind(h, _1, _2)) );


test( bind(h, bind(f, _1), bind(g, _1)) );


test( bind(h, bind(f, _1), bind(g, _2)) );


test( bind(f, bind(k)) );

return 0;
}

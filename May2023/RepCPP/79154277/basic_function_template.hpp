
#pragma once

#include <iostream>
#include <string>
namespace advscicomp {



template <typename T>
T max(T a, T b)
{
if (a > b)
return a;
else
return b;
}


template <typename T1, typename T2>
void g(T1 const& a, T2 const& b)
{
if (a < 0)
std::cout << b << '\n';
else if (b/5 > 10)
std::cout << a << '\n';
else
std::cout << "twiilggut" << '\n';
}


#include <algorithm> 
std::string tolower(std::string s)
{
std::transform(s.begin(), s.end(), s.begin(), ::tolower);
return s;
}


template <typename T2>
void g(std::string const& a, T2 const& b)
{

std::string m = "meerkat";

if (tolower(a) < m)
std::cout << b << '\n';
else if (b/5 > 10)
std::cout << a << '\n';
else
std::cout << "twiilggut" << '\n';
}

}

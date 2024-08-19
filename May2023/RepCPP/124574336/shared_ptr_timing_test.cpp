#include <boost/config.hpp>

#if defined(BOOST_MSVC)
#pragma warning(disable: 4786)  
#pragma warning(disable: 4710)  
#pragma warning(disable: 4711)  
#pragma warning(disable: 4514)  
#endif


#include <boost/shared_ptr.hpp>
#include <iostream>
#include <vector>
#include <ctime>

int const n = 8 * 1024 * 1024;

int main()
{
using namespace std;

std::vector< boost::shared_ptr<int> > v;
boost::shared_ptr<int> pi(new int);

clock_t t = clock();

for(int i = 0; i < n; ++i)
{
v.push_back(pi);
}

t = clock() - t;

std::cout << static_cast<double>(t) / CLOCKS_PER_SEC << '\n';

return 0;
}

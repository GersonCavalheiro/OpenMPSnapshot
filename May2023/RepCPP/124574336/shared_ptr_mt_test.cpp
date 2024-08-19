#include <boost/config.hpp>

#if defined(BOOST_MSVC)
#pragma warning(disable: 4786)  
#pragma warning(disable: 4710)  
#pragma warning(disable: 4711)  
#pragma warning(disable: 4514)  
#endif


#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>

#include <vector>

#include <cstdio>
#include <ctime>

#include <boost/detail/lightweight_thread.hpp>


int const n = 1024 * 1024;

void test( boost::shared_ptr<int> const & pi )
{
std::vector< boost::shared_ptr<int> > v;

for( int i = 0; i < n; ++i )
{
v.push_back( pi );
}
}

int const m = 16; 

#if defined( BOOST_HAS_PTHREADS )

char const * thmodel = "POSIX";

#else

char const * thmodel = "Windows";

#endif

int main()
{
using namespace std; 

printf( "Using %s threads: %d threads, %d iterations: ", thmodel, m, n );

boost::shared_ptr<int> pi( new int(42) );

clock_t t = clock();

boost::detail::lw_thread_t a[ m ];

for( int i = 0; i < m; ++i )
{
boost::detail::lw_thread_create( a[ i ], boost::bind( test, pi ) );
}

for( int j = 0; j < m; ++j )
{
boost::detail::lw_thread_join( a[j] );
}

t = clock() - t;

printf( "\n\n%.3f seconds.\n", static_cast<double>(t) / CLOCKS_PER_SEC );

return 0;
}

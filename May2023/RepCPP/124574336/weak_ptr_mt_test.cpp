#include <boost/config.hpp>

#if defined(BOOST_MSVC)
#pragma warning(disable: 4786)  
#pragma warning(disable: 4710)  
#pragma warning(disable: 4711)  
#pragma warning(disable: 4514)  
#endif


#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/bind.hpp>

#include <vector>

#include <cstdio>
#include <ctime>
#include <cstdlib>

#include <boost/detail/lightweight_thread.hpp>


int const n = 16384;
int const k = 512; 
int const m = 16; 

void test( std::vector< boost::shared_ptr<int> > & v )
{
using namespace std; 

std::vector< boost::weak_ptr<int> > w( v.begin(), v.end() );

int s = 0, f = 0, r = 0;

for( int i = 0; i < n; ++i )
{

v[ rand() % k ].reset();
++s;

for( int j = 0; j < k; ++j )
{
if( boost::shared_ptr<int> px = w[ j ].lock() )
{
++s;

if( rand() & 4 )
{
continue;
}

++f;
}
else
{
++r;
}

w[ j ] = v[ rand() % k ];
}
}

printf( "\n%d locks, %d forced rebinds, %d normal rebinds.", s, f, r );
}

#if defined( BOOST_HAS_PTHREADS )

char const * thmodel = "POSIX";

#else

char const * thmodel = "Windows";

#endif

int main()
{
using namespace std; 

printf("Using %s threads: %d threads, %d * %d iterations: ", thmodel, m, n, k );

std::vector< boost::shared_ptr<int> > v( k );

for( int i = 0; i < k; ++i )
{
v[ i ].reset( new int( 0 ) );
}

clock_t t = clock();

boost::detail::lw_thread_t a[ m ];

for( int i = 0; i < m; ++i )
{
boost::detail::lw_thread_create( a[ i ], boost::bind( test, v ) );
}

v.resize( 0 ); 

for( int j = 0; j < m; ++j )
{
boost::detail::lw_thread_join( a[j] );
}

t = clock() - t;

printf("\n\n%.3f seconds.\n", static_cast<double>(t) / CLOCKS_PER_SEC);

return 0;
}




#define HARNESS_SKIP_TEST ( __INTEL_COMPILER < 1400  || TBB_USE_DEBUG )

#define __TBB_ASSERT_ON_VECTORIZATION_FAILURE ( !HARNESS_SKIP_TEST )
#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"

#include "harness.h"
#include "harness_assert.h"

#include <algorithm>

class Body : NoAssign {
int *out_, *in_;
public:
Body( int* out, int *in ) : out_(out), in_(in) {}
void operator() ( int i ) const {
out_[i] = in_[i] + 1;
}
};

int TestMain () {
const int N = 10000;
tbb::task_scheduler_init init(1);
int array1[N];
std::fill( array1, array1+N, 0 );
tbb::parallel_for( 0, N-1, Body(array1+1, array1) );

int array2[N];
std::fill( array2, array2+N, 0 );
Body b(array2+1, array2);
for ( int i=0; i<N-1; ++i )
b(i);

ASSERT( !std::equal( array1, array1+N, array2 ), "The loop was not vectorized." );

return  Harness::Done;
}

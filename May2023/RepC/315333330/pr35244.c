int v1;
namespace N1
{
int v2;
}
namespace N2
{
int v3;
}
using N1::v2;
using namespace N2;
struct A;
typedef int i;
#pragma omp threadprivate (i)	
#pragma omp threadprivate (A)	
#pragma omp threadprivate (v1, v2, v3)
void foo ()
{
static int v4;
{
static int v5;
#pragma omp threadprivate (v4, v5)
}
}

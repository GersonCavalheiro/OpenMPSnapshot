#define thr threadprivate
struct S
{
static int s;
};
struct T : public S
{
static int t;
#pragma omp thr (s)	
};
#pragma omp thr (T::t)	

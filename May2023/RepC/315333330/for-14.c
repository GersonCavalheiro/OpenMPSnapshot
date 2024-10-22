extern "C" void abort ();
#define M(x, y, z) O(x, y, z)
#define O(x, y, z) x ## _ ## y ## _ ## z
#pragma omp declare target
#define F for
#define G f
#define S
#define N(x) M(x, G, normal)
#include "../libgomp.c/for-2.h"
#undef S
#undef N
#undef F
#undef G
#pragma omp end declare target
#undef OMPTGT
#undef OMPFROM
#undef OMPTO
#define DO_PRAGMA(x) _Pragma (#x)
#define OMPTGT DO_PRAGMA (omp target)
#define OMPFROM(v) DO_PRAGMA (omp target update from(v))
#define OMPTO(v) DO_PRAGMA (omp target update to(v))
#define F teams distribute
#define G td
#define S
#define N(x) M(x, G, normal)
#include "../libgomp.c/for-2.h"
#undef S
#undef N
#undef F
#undef G
#define F teams distribute
#define G td_ds128
#define S dist_schedule(static, 128)
#define N(x) M(x, G, normal)
#include "../libgomp.c/for-2.h"
#undef S
#undef N
#undef F
#undef G
#define F teams distribute simd
#define G tds
#define S
#define N(x) M(x, G, normal)
#include "../libgomp.c/for-2.h"
#undef S
#undef N
#undef F
#undef G
#define F teams distribute simd
#define G tds_ds128
#define S dist_schedule(static, 128)
#define N(x) M(x, G, normal)
#include "../libgomp.c/for-2.h"
#undef S
#undef N
#undef F
#undef G
#define F teams distribute parallel for
#define G tdpf
#include "../libgomp.c/for-1.h"
#undef F
#undef G
#define F teams distribute parallel for dist_schedule(static, 128)
#define G tdpf_ds128
#include "../libgomp.c/for-1.h"
#undef F
#undef G
#define F teams distribute parallel for simd
#define G tdpfs
#include "../libgomp.c/for-1.h"
#undef F
#undef G
#define F teams distribute parallel for simd dist_schedule(static, 128)
#define G tdpfs_ds128
#include "../libgomp.c/for-1.h"
#undef F
#undef G
int
main ()
{
if (test_td_normal ()
|| test_td_ds128_normal ()
|| test_tds_normal ()
|| test_tds_ds128_normal ()
|| test_tdpf_static ()
|| test_tdpf_static32 ()
|| test_tdpf_auto ()
|| test_tdpf_guided32 ()
|| test_tdpf_runtime ()
|| test_tdpf_ds128_static ()
|| test_tdpf_ds128_static32 ()
|| test_tdpf_ds128_auto ()
|| test_tdpf_ds128_guided32 ()
|| test_tdpf_ds128_runtime ()
|| test_tdpfs_static ()
|| test_tdpfs_static32 ()
|| test_tdpfs_auto ()
|| test_tdpfs_guided32 ()
|| test_tdpfs_runtime ()
|| test_tdpfs_ds128_static ()
|| test_tdpfs_ds128_static32 ()
|| test_tdpfs_ds128_auto ()
|| test_tdpfs_ds128_guided32 ()
|| test_tdpfs_ds128_runtime ())
abort ();
}

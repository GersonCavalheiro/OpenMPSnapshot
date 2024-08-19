#ifndef GCC_REALGMP_H
#define GCC_REALGMP_H
#include <mpfr.h>
#include <mpc.h>
extern void real_from_mpfr (REAL_VALUE_TYPE *, mpfr_srcptr, tree, mp_rnd_t);
extern void real_from_mpfr (REAL_VALUE_TYPE *, mpfr_srcptr,
const real_format *, mp_rnd_t);
extern void mpfr_from_real (mpfr_ptr, const REAL_VALUE_TYPE *, mp_rnd_t);
#if (GCC_VERSION >= 3000)
#pragma GCC poison MPFR_RNDN MPFR_RNDZ MPFR_RNDU MPFR_RNDD
#endif
#endif 

#ifdef __NO_MATH_ERRNO__
#error "__NO_MATH_ERRNO__ defined"
#endif
#pragma GCC optimize "-fno-math-errno"
#ifndef __NO_MATH_ERRNO__
#error "__NO_MATH_ERRNO__ not defined"
#endif
#pragma GCC optimize "-fmath-errno"
#ifdef __NO_MATH_ERRNO__
#error "__NO_MATH_ERRNO__ defined"
#endif

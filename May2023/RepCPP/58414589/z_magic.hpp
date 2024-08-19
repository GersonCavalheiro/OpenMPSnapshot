

#ifndef COMMON_Z_MAGIC_HPP
#define COMMON_Z_MAGIC_HPP

#define for_ for

#define CHAIn2(a, b) a b
#define CHAIN2(a, b) CHAIn2(a, b)

#define CONCAt2(a, b) a##b
#define CONCAT2(a, b) CONCAt2(a, b)

#define STRINGIFy(s) #s
#define STRINGIFY(s) STRINGIFy(s)

#ifdef _MSC_VER
#define PRAGMA_MACRo(x) __pragma(x)
#define PRAGMA_MACRO(x) PRAGMA_MACRo(x)
#else
#define PRAGMA_MACRo(x) _Pragma(#x)
#define PRAGMA_MACRO(x) PRAGMA_MACRo(x)
#endif

#define UNUSED(x) ((void)x)
#define MAYBE_UNUSED(x) UNUSED(x)

#if defined(_WIN32) && !defined(__GNUC__)
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

#define Z_NOT_0 1
#define Z_NOT_1 0
#define Z_NOT(cond) CONCAT2(Z_NOT_, cond)

#define Z_DO_IF_0(...)
#define Z_DO_IF_1(...) __VA_ARGS__
#define Z_CONDITIONAL_DO(cond, ...) CONCAT2(Z_DO_IF_, cond)(__VA_ARGS__)

#endif


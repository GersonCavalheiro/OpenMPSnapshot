#ifndef IGL_DEPRECATED_H
#define IGL_DEPRECATED_H
#ifdef __GNUC__
#define IGL_DEPRECATED(func) func __attribute__ ((deprecated))
#elif defined(_MSC_VER)
#define IGL_DEPRECATED(func) __declspec(deprecated) func
#else
#pragma message("WARNING: You need to implement IGL_DEPRECATED for this compiler")
#define IGL_DEPRECATED(func) func
#endif
#endif

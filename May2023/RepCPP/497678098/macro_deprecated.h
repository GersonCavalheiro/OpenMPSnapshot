
#ifndef INCLUDE_SEQAN_BASIC_MACRO_DEPRECATED_H_
#define INCLUDE_SEQAN_BASIC_MACRO_DEPRECATED_H_

namespace seqan {



#if defined(__clang__)
#define SEQAN_DEPRECATED_PRE(msg)
#define SEQAN_DEPRECATED_POST(msg) __attribute__((deprecated(msg)))
#elif defined(__GNUC__)
#define SEQAN_DEPRECATED_PRE(msg)
#define SEQAN_DEPRECATED_POST(msg) __attribute__((__deprecated__))
#elif defined(_MSC_VER)
#define SEQAN_DEPRECATED_PRE(msg) __declspec(deprecated(msg))
#define SEQAN_DEPRECATED_POST(msg)
#else
#pragma message("WARNING: You need to implement DEPRECATED_PRE and DEPRECATED_POST for this compiler")
#define SEQAN_DEPRECATED_PRE(func)
#define SEQAN_DEPRECATED_POST(func)
#endif

}  

#endif  



#pragma once


#if defined (_MSC_VER)

#define AWS_UNREFERENCED_PARAM(x) (&reinterpret_cast<const int &>(x))

#else

#define AWS_UNREFERENCED_PARAM(x) ((void)(x))

#endif 

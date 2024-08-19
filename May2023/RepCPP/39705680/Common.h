


#ifndef DC_COMMON_H
#define DC_COMMON_H

#include <cstdio>

template<typename... Args>
void parallel_printf(const char* fmt, Args... args )
{
{
std::printf( fmt, args... ) ;
}
}


#endif 

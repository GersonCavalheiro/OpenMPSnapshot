#ifndef FOR_EACH_H
#define FOR_EACH_H

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename Iterator, typename F>
void for_each(Iterator begin, Iterator end, F f) {
#pragma omp parallel if(omp_get_level() == 0)
{
#pragma omp for
for (auto it = begin; it != end; ++it) {
f(*it);
}
}
}

#endif 

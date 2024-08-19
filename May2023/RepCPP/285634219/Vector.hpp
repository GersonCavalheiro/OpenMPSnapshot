#ifndef _Vector_hpp_
#define _Vector_hpp_


#include <vector>

namespace miniFE {


template<typename Scalar,
typename LocalOrdinal,
typename GlobalOrdinal>
struct Vector {
typedef Scalar ScalarType;
typedef LocalOrdinal LocalOrdinalType;
typedef GlobalOrdinal GlobalOrdinalType;

Vector(GlobalOrdinal startIdx, LocalOrdinal local_sz)
: startIndex(startIdx),
local_size(local_sz),
coefs(local_size)
{
#pragma omp parallel for
for(MINIFE_LOCAL_ORDINAL i=0; i < local_size; ++i) {
coefs[i] = 0;	 
}
}

~Vector()
{
}

GlobalOrdinal startIndex;
LocalOrdinal local_size;
std::vector<Scalar> coefs;
};


}

#endif


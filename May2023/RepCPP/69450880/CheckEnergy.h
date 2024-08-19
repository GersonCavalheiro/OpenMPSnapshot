



#ifndef CHECKENERGY_H_
#define CHECKENERGY_H_

namespace hydra {

namespace detail {


template <size_t N>
struct CheckEnergy
{
GReal_t fMasses[N];

CheckEnergy(const GReal_t (&masses)[N] )
{
for(size_t i=0; i<N; i++)
fMasses[i] = masses[i];
}

__hydra_host__      __hydra_device__
CheckEnergy(CheckEnergy<N> const& other)
{
for(size_t i=0; i<N; i++)
fMasses[i] = other.fMasses[i];
}

template<typename Type>
__hydra_host__ __hydra_device__
inline bool operator()(Type& particle)
{

Vector4R mother = particle;
GReal_t fTeCmTm =  mother.mass();

for (size_t n = 0; n < N; n++)
{
fTeCmTm -= fMasses[n];
}

return (bool) fTeCmTm > 0.0;
}
};

}

}

#endif 

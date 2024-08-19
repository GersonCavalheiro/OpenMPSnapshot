



#ifndef AVERAGEMOTHER_H_
#define AVERAGEMOTHER_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>

#include <hydra/Vector3R.h>
#include <hydra/Vector4R.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/functors/StatsPHSP.h>

#include <hydra/detail/external/hydra_thrust/tuple.h>
#include <hydra/detail/external/hydra_thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/hydra_thrust/random.h>

#include <type_traits>
#include <utility>


namespace hydra {

namespace detail {


template <size_t N, typename GRND, typename FUNCTOR>
struct AverageMother
{

size_t  fSeed;

GReal_t fECM;
GReal_t fMaxWeight;
GReal_t fBeta0;
GReal_t fBeta1;
GReal_t fBeta2;


GReal_t fMasses[N];
FUNCTOR fFunctor ;

AverageMother(Vector4R const& mother, const GReal_t (&masses)[N],
double maxweight, double ecm, size_t seed, FUNCTOR const& functor):
fMaxWeight(maxweight),
fECM(ecm),
fSeed(seed),
fFunctor(functor)
{

for(size_t i=0; i<N; i++)
fMasses[i]=masses[i];

GReal_t beta = mother.d3mag() / mother.get(0);

if (beta)
{
GReal_t w = beta / mother.d3mag();
fBeta0 = mother.get(0) * w;
fBeta1 = mother.get(1) * w;
fBeta2 = mother.get(2) * w;
}
else
fBeta0 = fBeta1 = fBeta2 = 0.0;

}

__hydra_host__ __hydra_device__
AverageMother( AverageMother<N, GRND, FUNCTOR> const& other ):
fFunctor(other.fFunctor),
fSeed(other.fSeed ),
fECM(other.fECM ),
fMaxWeight(other.fMaxWeight ),
fBeta0(other.fBeta0 ),
fBeta1(other.fBeta1 ),
fBeta2(other.fBeta2 )
{ for(size_t i=0; i<N; i++) fMasses[i]=other.fMasses[i]; }



__hydra_host__      __hydra_device__ inline
static GReal_t pdk(const GReal_t a, const GReal_t b,
const GReal_t c)
{
return ::sqrt( (a - b - c) * (a + b + c) * (a - b + c) * (a + b - c) ) / (2 * a);
}

__hydra_host__ __hydra_device__ inline
void bbsort( GReal_t *array, GInt_t n)
{

for (GInt_t c = 0; c < n; c++)
{
GInt_t nswap = 0;

for (GInt_t d = 0; d < n - c - 1; d++)
{
if (array[d] > array[d + 1]) 
{
GReal_t swap = array[d];
array[d] = array[d + 1];
array[d + 1] = swap;
nswap++;
}
}
if (nswap == 0)
break;
}

}


__hydra_host__   __hydra_device__ inline
constexpr static size_t hash(const size_t a, const size_t b)
{
return   (((2 * a) >=  (2 * b) ? (2 * a) * (2 * a) + (2 * a) + (2 * b) : (2 * a) + (2 * b) * (2 * b)) / 2);
}

__hydra_host__   __hydra_device__ inline
GReal_t process(size_t evt, Vector4R (&daugters)[N])
{

GRND randEng(fSeed);
randEng.discard(evt+3*N);

hydra_thrust::uniform_real_distribution<GReal_t> uniDist(0.0, 1.0);

GReal_t rno[N];
rno[0] = 0.0;
rno[N - 1] = 1.0;

if (N > 2)
{
for (size_t n = 1; n < N - 1; n++)
{
rno[n] =  uniDist(randEng) ;

}

bbsort(&rno[1], N -2);

}


GReal_t invMas[N], sum = 0.0;

for (size_t n = 0; n < N; n++)
{
sum += fMasses[n];
invMas[n] = rno[n] * fECM + sum;
}


GReal_t wt = fMaxWeight;

GReal_t pd[N];

for (size_t n = 0; n < N - 1; n++)
{
pd[n] = pdk(invMas[n + 1], invMas[n], fMasses[n + 1]);
wt *= pd[n];
}


daugters[0].set(::sqrt((GReal_t) pd[0] * pd[0] + fMasses[0] * fMasses[0]), 0.0,
pd[0], 0.0);

for (size_t i = 1; i < N; i++)
{

daugters[i].set(
::sqrt(pd[i - 1] * pd[i - 1] + fMasses[i] * fMasses[i]), 0.0,
-pd[i - 1], 0.0);

GReal_t cZ = 2 * uniDist(randEng) -1 ;
GReal_t sZ = ::sqrt(1 - cZ * cZ);
GReal_t angY = 2 * PI* uniDist(randEng);
GReal_t cY = ::cos(angY);
GReal_t sY = ::sin(angY);
for (size_t j = 0; j <= i; j++)
{

GReal_t x = daugters[j].get(1);
GReal_t y = daugters[j].get(2);
daugters[j].set(1, cZ * x - sZ * y);
daugters[j].set(2, sZ * x + cZ * y); 

x = daugters[j].get(1);
GReal_t z = daugters[j].get(3);
daugters[j].set(1, cY * x - sY * z);
daugters[j].set(3, sY * x + cY * z); 
}

if (i == (N - 1))
break;

GReal_t beta = pd[i] / ::sqrt(pd[i] * pd[i] + invMas[i] * invMas[i]);
for (size_t j = 0; j <= i; j++)
{

daugters[j].applyBoostTo(Vector3R(0, beta, 0));
}

}

for (size_t n = 0; n < N; n++)
{

daugters[n].applyBoostTo(Vector3R(fBeta0, fBeta1, fBeta2));

}


return wt;

}

template< typename I>
__hydra_host__  __hydra_device__ inline
StatsPHSP operator()(I evt)
{
typedef typename hydra::detail::tuple_type<N, Vector4R>::type Tuple_t;

Vector4R Particles[N];

GReal_t weight = process(evt, Particles);
Tuple_t particles_tuple{};
detail::assignArrayToTuple(particles_tuple, Particles);

StatsPHSP result;

result.fMean = fFunctor(particles_tuple);
result.fW    = weight;
result.fM2   = 0.0;

return result;

}

};

}

}



#endif 

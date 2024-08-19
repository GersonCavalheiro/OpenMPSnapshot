







#ifndef _DECAYMOTHERS_H_
#define _DECAYMOTHERS_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>

#include <hydra/Vector3R.h>
#include <hydra/Vector4R.h>
#include <hydra/detail/utility/Utility_Tuple.h>

#include <hydra/detail/external/hydra_thrust/tuple.h>
#include <hydra/detail/external/hydra_thrust/iterator/zip_iterator.h>


namespace hydra {

namespace detail {

template <size_t N, typename GRND>
struct DecayMothers
{
typedef typename tuple_type<N, hydra::Vector4R>::type  particles_tuple_type;

size_t fSeed;
GReal_t fECM;
GReal_t fMaxWeight;
GReal_t fMasses[N];

DecayMothers(const GReal_t (&masses)[N], double maxweight, double ecm, size_t _seed ):
fMaxWeight(maxweight),
fECM(ecm),
fSeed(_seed)
{
for(size_t i=0; i<N; i++)
fMasses[i] = masses[i];
}

__hydra_host__      __hydra_device__
DecayMothers(DecayMothers<N, GRND> const& other):
fSeed(other.fSeed ),
fECM(other.fECM ),
fMaxWeight(other.fMaxWeight )
{
for(size_t i=0; i<N; i++)
fMasses[i] = other.fMasses[i];
}



__hydra_host__      __hydra_device__ inline
static GReal_t pdk(const GReal_t a, const GReal_t b,
const GReal_t c)
{
return ::sqrt( (a - b - c) * (a + b + c) * (a - b + c) * (a + b - c) ) / (2 * a);;
}

__hydra_host__ __hydra_device__ void bbsort(GReal_t *array, GInt_t n)
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

__hydra_host__      __hydra_device__ GReal_t process(size_t evt,
Vector4R (&particles)[N+1])
{

GRND randEng( fSeed );
randEng.discard(evt+3*N);

hydra_thrust::uniform_real_distribution<GReal_t> uniDist(0.0, 1.0);

GReal_t rno[N];
rno[0] = 0.0;



if (N > 2)
{
for (size_t n = 1; n < N - 1; n++)
rno[n] = uniDist(randEng) ;
bbsort(&rno[1], N - 2);

}

rno[N - 1] = 1;
GReal_t invMas[N], sum = 0.0;

for (size_t n = 0; n < N; n++)
{
sum += fMasses[n];
invMas[n] = rno[n] * fECM + sum;
}



GReal_t wt  = fMaxWeight;

GReal_t pd[N];

for (size_t n = 0; n < N - 1; n++)
{
pd[n] = pdk(invMas[n + 1], invMas[n], fMasses[n + 1]);
wt *= pd[n];
}




particles[1].set(::sqrt(pd[0] * pd[0] + fMasses[0] * fMasses[0]), 0.0,
pd[0], 0.0);

for (size_t i = 1; i < N; i++)
{

particles[i + 1].set(
::sqrt(pd[i - 1] * pd[i - 1] + fMasses[i] * fMasses[i]), 0.0,
-pd[i - 1], 0.0);

GReal_t cZ = 2	* uniDist(randEng) -1 ;
GReal_t sZ = ::sqrt(1 - cZ * cZ);
GReal_t angY = 2.0 * PI	* uniDist(randEng);
GReal_t cY = ::cos(angY);
GReal_t sY = ::sin(angY);
for (size_t j = 0; j <= i; j++)
{

GReal_t x = particles[j + 1].get(1);
GReal_t y = particles[j + 1].get(2);
particles[j + 1].set(1, cZ * x - sZ * y);
particles[j + 1].set(2, sZ * x + cZ * y); 

x = particles[j + 1].get(1);
GReal_t z = particles[j + 1].get(3);
particles[j + 1].set(1, cY * x - sY * z);
particles[j + 1].set(3, sY * x + cY * z); 
}

if (i == (N - 1))
break;

GReal_t beta = pd[i] / ::sqrt(pd[i] * pd[i] + invMas[i] * invMas[i]);
for (size_t j = 0; j <= i; j++)
{

particles[j+1 ].applyBoostTo(0, beta, 0);
}

}

for (size_t n = 0; n < N; n++)
{

particles[n+1].applyBoostTo(particles[0]);

}


return wt;


}


__hydra_host__      __hydra_device__
inline particles_tuple_type operator()(size_t evt, Vector4R const& mother )
{

Vector4R Particles[N+1];
Particles[0] = mother;

process(evt, Particles);

particles_tuple_type particles{};

assignArrayToTuple(particles,  &Particles[1] );

return particles;

}



};

}

}

#endif 







#ifndef SOBOL_H_
#define SOBOL_H_

#include <algorithm>
#include<hydra/detail/Config.h>
#include<hydra/detail/SobolTable.h>
#include<hydra/detail/GrayCode.h>
#include<hydra/detail/utility/MSB.h>
#include <cassert>

namespace hydra {


namespace detail {



template<typename UIntType, unsigned D, unsigned W, typename SobolTables>
struct sobol_lattice
{
typedef UIntType value_type;

static_assert(D > 0u, "[hydra::sobol_lattice] Problem: D < 0. (D) - dimension has to be greater than zero.");
static_assert(D <= HYDRA_SOBOL_MAX_DIMENSION, "[hydra::sobol_lattice] Problem: D > SOBOL_MAX_DIMENSION. (D) - dimension has to be greater than zero.");
static_assert(W > 0u, "[hydra::sobol_lattice] Problem: W < 0. (W) - bit count has to be greater than zero.");

static const unsigned bit_count = W;
static const unsigned lattice_dimension= D;
static const unsigned storage_size=W*D;


public:
__hydra_host__ __hydra_device__
sobol_lattice(){

init();
}

__hydra_host__ __hydra_device__
sobol_lattice(sobol_lattice< UIntType,D,W,SobolTables> const& other ){

#pragma unroll storage_size
for(unsigned i=0;i<storage_size; ++i)
bits[i] = other.GetBits()[i];
}


sobol_lattice<UIntType,D,W,SobolTables>
__hydra_host__ __hydra_device__
operator=(sobol_lattice< UIntType,D,W,SobolTables> const& other ){
if(this == &other) return *this;

#pragma unroll storage_size
for(unsigned i=0;i<storage_size; ++i)
bits[i] = other.GetBits()[i];
return *this;

}

__hydra_host__ __hydra_device__
const  value_type* iter_at(std::size_t n) const
{
assert(!(n > storage_size-1 ));
return bits + n;
}

__hydra_host__ __hydra_device__
const value_type* GetBits() const {
return bits;
}

private:

__hydra_host__ __hydra_device__
inline void init()
{

for (unsigned k = 0; k != bit_count; ++k)
bits[lattice_dimension*k] = static_cast<value_type>(1);

for (std::size_t dim = 1; dim < lattice_dimension; ++dim)
{
const typename SobolTables::value_type poly = SobolTables::polynomial(dim-1);

const unsigned degree = msb(poly); 

for (unsigned k = 0; k != degree; ++k)
bits[lattice_dimension*k + dim] = SobolTables::minit(dim-1, k);

for (unsigned j = degree; j < bit_count; ++j)
{
typename SobolTables::value_type p_i = poly;
const std::size_t bit_offset = lattice_dimension*j + dim;

bits[bit_offset] = bits[lattice_dimension*(j-degree) + dim];
for (unsigned k = 0; k != degree; ++k, p_i >>= 1)
{
int rem = degree - k;
bits[bit_offset] ^= ((p_i & 1) * bits[lattice_dimension*(j-rem) + dim]) << rem;
}
}
}

unsigned p = 1u;
for (int j = bit_count-1-1; j >= 0; --j, ++p)
{
const std::size_t bit_offset = lattice_dimension * j;
for (std::size_t dim = 0; dim != lattice_dimension; ++dim)
bits[bit_offset + dim] <<= p;
}

}

value_type bits[storage_size];

};

} 

typedef detail::SobolTable default_sobol_table;



template<typename UIntType,  unsigned D, unsigned W, typename SobolTables = default_sobol_table>
class sobol_engine
: public detail::gray_code<detail::sobol_lattice<UIntType, D, W, SobolTables>>
{
typedef detail::sobol_lattice<UIntType, D, W, SobolTables> lattice_t;
typedef detail::gray_code<lattice_t> base_t;

public:

static const  UIntType min=0;
static const  UIntType max=base_t::max;

__hydra_host__ __hydra_device__
sobol_engine() : base_t() {}

__hydra_host__ __hydra_device__
sobol_engine( UIntType s) : base_t() {
base_t::seed(s);
}



};


template<unsigned D>
using sobol= sobol_engine<uint_least64_t, D, 64u, default_sobol_table> ;

}  

#endif 







#ifndef GRAYCODE_H_
#define GRAYCODE_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/utility/LSB.h>
#include <hydra/detail/utility/IntegerMask.h>

#include <hydra/detail/QuasiRandomBase.h>
#include <functional> 
#include <type_traits>



namespace hydra {

namespace detail {

template<typename LatticeT>
class gray_code: public quasi_random_base< gray_code<LatticeT>, LatticeT, typename LatticeT::value_type>
{

public:
typedef typename LatticeT::value_type result_type;
typedef result_type size_type;

static const result_type min=0;
static const result_type max=
low_bits_mask_t<LatticeT::bit_count>::sig_bits;

private:
typedef gray_code<LatticeT> self_t;
typedef quasi_random_base<self_t, LatticeT, size_type> base_t;

friend class quasi_random_base<self_t, LatticeT, size_type>;

struct check_nothing {
__hydra_host__ __hydra_device__
inline static void bit_pos(unsigned) {}

__hydra_host__ __hydra_device__
inline static void code_size(size_type) {}
};
struct check_bit_range {

__hydra_host__ __hydra_device__
static void raise_bit_count() {
HYDRA_EXCEPTION("GrayCode: bit_count" );
}

__hydra_host__ __hydra_device__
inline static void bit_pos(unsigned bit_pos) {
if (bit_pos >= LatticeT::bit_count)
raise_bit_count();
}

__hydra_host__ __hydra_device__
inline static void code_size(size_type code) {
if (code > (self_t::max))
raise_bit_count();
}
};

static_assert(LatticeT::bit_count <= std::numeric_limits<size_type>::digits,
"hydra::gray_code : bit_count in LatticeT' > digits");

typedef typename std::conditional<
std::integral_constant<bool,((LatticeT::bit_count) < std::numeric_limits<size_type>::digits)>::value
, check_bit_range
, check_nothing
>::type check_bit_range_t;


public:

__hydra_host__ __hydra_device__
explicit gray_code(): base_t() {}



__hydra_host__ __hydra_device__
inline  void seed()
{
set_zero_state();
update_quasi(0);
base_t::reset_seq(0);
}

__hydra_host__ __hydra_device__
inline  void seed(const size_type init)
{
if (init != this->curr_seq())
{

size_type seq_code =  init+1;
if(HYDRA_HOST_UNLIKELY(!(init < seq_code))){
HYDRA_EXCEPTION("hydra::gray_code : seed overflow. Returning without set seed")
return ;
}

seq_code ^= (seq_code >> 1);
check_bit_range_t::code_size(seq_code); 

set_zero_state();
for (unsigned r = 0; seq_code != 0; ++r, seq_code >>= 1)
{
if (seq_code & static_cast<size_type>(1))
update_quasi(r);
}
}
base_t::reset_seq(init);
}

private:

__hydra_host__ __hydra_device__
inline  void compute_seq(size_type seq)
{
unsigned r = lsb(seq ^ (self_t::max));
check_bit_range_t::bit_pos(r); 
update_quasi(r);
}

__hydra_host__ __hydra_device__
inline void update_quasi(unsigned r){


result_type* i= this->state_begin();
const  result_type* j= this->lattice.iter_at(r * this->dimension());

#pragma unroll LatticeT::lattice_dimension
for(size_t s=0;s<LatticeT::lattice_dimension; ++s)
i[s]=(i[s])^(j[s]);

}

__hydra_host__ __hydra_device__
inline void set_zero_state(){

result_type* s= this->state_begin();

#pragma unroll LatticeT::lattice_dimension
for(size_t i=0;i<LatticeT::lattice_dimension; ++i)
s[i]=result_type{};
}

};


}  

}  
#endif 

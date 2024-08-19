






#ifndef QUASIRANDOMBASE_H_
#define QUASIRANDOMBASE_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/utility/Exception.h>
#include <hydra/Tuple.h>

#include <limits>
#include <type_traits>
#include <cstdint>


namespace hydra {

namespace detail {



template<typename DerivedT, typename LatticeT, typename SizeT>
class quasi_random_base
{

public:

typedef SizeT size_type;
typedef typename LatticeT::value_type result_type;

__hydra_host__ __hydra_device__
quasi_random_base():
lattice(),
quasi_state()
{
derived().seed();
}

__hydra_host__ __hydra_device__
quasi_random_base(quasi_random_base<DerivedT,LatticeT, SizeT> const& other):
lattice(other.lattice),
curr_elem(other.curr_elem),
seq_count(other.seq_count)
{
#pragma unroll LatticeT::lattice_dimension
for(size_t i=0;i<LatticeT::lattice_dimension; ++i)
quasi_state[i] = other.quasi_state[i];
}

__hydra_host__ __hydra_device__
quasi_random_base<DerivedT,LatticeT, SizeT>
operator=(quasi_random_base<DerivedT,LatticeT, SizeT> const& other){

if(this==&other) return *this;

lattice = other.lattice;
curr_elem = other.curr_elem;
seq_count = other.seq_count;

#pragma unroll LatticeT::lattice_dimension
for(size_t i=0;i<LatticeT::lattice_dimension; ++i)
quasi_state[i] = other.quasi_state[i];

return *this;
}


__hydra_host__ __hydra_device__
constexpr unsigned dimension() const {
return LatticeT::lattice_dimension;
}


__hydra_host__ __hydra_device__
inline result_type operator()()
{
return curr_elem != dimension() ? load_cached(): next_state();
}

template<typename ...T>
__hydra_host__ __hydra_device__
inline typename std::enable_if<
sizeof...(T)==LatticeT::lattice_dimension, void>::type
generate(hydra::tuple<T...>& data){
generate_helper(data);
}

__hydra_host__ __hydra_device__
inline void discard(uintmax_t z)
{
const std::size_t dimension_value = dimension();


uintmax_t vec_n = z / dimension_value;
std::size_t carry = curr_elem + (z % dimension_value);

vec_n += carry / dimension_value;
carry  = carry % dimension_value;

const bool corr = (!carry) & static_cast<bool>(vec_n);

discard_vector(vec_n - static_cast<uintmax_t>(corr));

curr_elem = carry ^ (-static_cast<std::size_t>(corr) & dimension_value);

}

__hydra_host__ __hydra_device__
friend bool operator==(const quasi_random_base& x,
const quasi_random_base& y)
{
const std::size_t dimension_value = x.dimension();


return (dimension_value == y.dimension()) &&
!((x.seq_count < y.seq_count ?
y.seq_count - x.seq_count :
x.seq_count - y.seq_count)> static_cast<size_type>(1)) &&
(x.seq_count + (x.curr_elem / dimension_value) == y.seq_count + (y.curr_elem / dimension_value)) &&
(x.curr_elem % dimension_value == y.curr_elem % dimension_value);
}

__hydra_host__ __hydra_device__
friend bool operator!=(const quasi_random_base& lhs,
const quasi_random_base& rhs)
{  return !(lhs == rhs); }

protected:

typedef  result_type* state_iterator;


__hydra_host__ __hydra_device__
inline size_type curr_seq() const {
return seq_count;
}

__hydra_host__ __hydra_device__
inline state_iterator state_begin() {
return &(quasi_state[0]);
}

__hydra_host__ __hydra_device__
inline state_iterator state_end() {
return &(quasi_state[0]) + LatticeT::lattice_dimension;
}

__hydra_host__ __hydra_device__
inline void reset_seq(size_type seq){

seq_count = seq;
curr_elem = 0u;
}

private:

template<typename T, unsigned I>
__hydra_host__ __hydra_device__
inline typename std::enable_if< (I== LatticeT::lattice_dimension), void>::type
generate_helper(T& ){ }

template<typename T, unsigned I=0>
__hydra_host__ __hydra_device__
inline typename std::enable_if< (I< LatticeT::lattice_dimension), void>::type
generate_helper(T& data){
hydra::get<I>(data)=this->operator()();
generate_helper<T,I+1>(data);
}



__hydra_host__ __hydra_device__
inline DerivedT& derived()
{
return *const_cast< DerivedT*>( (static_cast<DerivedT* >(this)));
}

__hydra_host__ __hydra_device__
inline result_type load_cached()
{
return quasi_state[curr_elem++];
}

__hydra_host__ __hydra_device__
inline 	result_type next_state()
{
size_type new_seq = seq_count;

if (HYDRA_HOST_LIKELY(++new_seq > seq_count))
{
derived().compute_seq(new_seq);
reset_seq(new_seq);
return load_cached();
}

HYDRA_EXCEPTION("hydra::quasi_random_base: next_state overflow. Returning current state.")
return load_cached();
}

__hydra_host__ __hydra_device__
inline void discard_vector(uintmax_t z)
{
const uintmax_t max_z = std::numeric_limits<size_type>::max() - seq_count;

if (max_z < z){
HYDRA_EXCEPTION("hydra::quasi_random_base: discard_vector. Returning without doing nothing.")
return ;
}
std::size_t tmp = curr_elem;
derived().seed(static_cast<size_type>(seq_count + z));
curr_elem = tmp;
}


private:
std::size_t curr_elem;
size_type seq_count;
protected:
LatticeT lattice;
private:
result_type quasi_state[LatticeT::lattice_dimension];
};


}  

}  


#endif 

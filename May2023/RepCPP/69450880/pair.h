



#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <utility>

namespace hydra_thrust
{






template <typename T1, typename T2>
struct pair
{

typedef T1 first_type;


typedef T2 second_type;


first_type first;


second_type second;


__host__ __device__ pair(void);


inline __host__ __device__
pair(const T1 &x, const T2 &y);


template <typename U1, typename U2>
inline __host__ __device__
pair(const pair<U1,U2> &p);


template <typename U1, typename U2>
inline __host__ __device__
pair(const std::pair<U1,U2> &p);


inline __host__ __device__
void swap(pair &p);
}; 



template <typename T1, typename T2>
inline __host__ __device__
bool operator==(const pair<T1,T2> &x, const pair<T1,T2> &y);



template <typename T1, typename T2>
inline __host__ __device__
bool operator<(const pair<T1,T2> &x, const pair<T1,T2> &y);



template <typename T1, typename T2>
inline __host__ __device__
bool operator!=(const pair<T1,T2> &x, const pair<T1,T2> &y);



template <typename T1, typename T2>
inline __host__ __device__
bool operator>(const pair<T1,T2> &x, const pair<T1,T2> &y);



template <typename T1, typename T2>
inline __host__ __device__
bool operator<=(const pair<T1,T2> &x, const pair<T1,T2> &y);



template <typename T1, typename T2>
inline __host__ __device__
bool operator>=(const pair<T1,T2> &x, const pair<T1,T2> &y);



template <typename T1, typename T2>
inline __host__ __device__
void swap(pair<T1,T2> &x, pair<T1,T2> &y);



template <typename T1, typename T2>
inline __host__ __device__
pair<T1,T2> make_pair(T1 x, T2 y);



template<size_t N, typename T> struct tuple_element;



template<typename Pair> struct tuple_size;











} 

#include <hydra/detail/external/hydra_thrust/detail/pair.inl>


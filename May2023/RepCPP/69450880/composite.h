


#pragma once

#include <hydra/detail/external/hydra_thrust/detail/functional/actor.h>
#include <hydra/detail/external/hydra_thrust/tuple.h>

namespace hydra_thrust
{
namespace detail
{
namespace functional
{

template<typename Eval0,
typename Eval1  = hydra_thrust::null_type,
typename Eval2  = hydra_thrust::null_type,
typename Eval3  = hydra_thrust::null_type,
typename Eval4  = hydra_thrust::null_type,
typename Eval5  = hydra_thrust::null_type,
typename Eval6  = hydra_thrust::null_type,
typename Eval7  = hydra_thrust::null_type,
typename Eval8  = hydra_thrust::null_type,
typename Eval9  = hydra_thrust::null_type,
typename Eval10 = hydra_thrust::null_type>
class composite;

template<typename Eval0, typename Eval1>
class composite<
Eval0,
Eval1,
hydra_thrust::null_type,
hydra_thrust::null_type,
hydra_thrust::null_type,
hydra_thrust::null_type,
hydra_thrust::null_type,
hydra_thrust::null_type,
hydra_thrust::null_type,
hydra_thrust::null_type
>
{
public:
template<typename Env>
struct result
{
typedef typename Eval0::template result<
hydra_thrust::tuple<
typename Eval1::template result<Env>::type
>
>::type type;
};

__host__ __device__
composite(const Eval0 &e0, const Eval1 &e1)
: m_eval0(e0),
m_eval1(e1)
{}

template<typename Env>
__host__ __device__
typename result<Env>::type
eval(const Env &x) const
{
typename Eval1::template result<Env>::type result1 = m_eval1.eval(x);
return m_eval0.eval(hydra_thrust::tie(result1));
}

private:
Eval0 m_eval0;
Eval1 m_eval1;
}; 

template<typename Eval0, typename Eval1, typename Eval2>
class composite<
Eval0,
Eval1,
Eval2,
hydra_thrust::null_type,
hydra_thrust::null_type,
hydra_thrust::null_type,
hydra_thrust::null_type,
hydra_thrust::null_type,
hydra_thrust::null_type,
hydra_thrust::null_type
>
{
public:
template<typename Env>
struct result
{
typedef typename Eval0::template result<
hydra_thrust::tuple<
typename Eval1::template result<Env>::type,
typename Eval2::template result<Env>::type
>
>::type type;
};

__host__ __device__
composite(const Eval0 &e0, const Eval1 &e1, const Eval2 &e2)
: m_eval0(e0),
m_eval1(e1),
m_eval2(e2)
{}

template<typename Env>
__host__ __device__
typename result<Env>::type
eval(const Env &x) const
{
typename Eval1::template result<Env>::type result1 = m_eval1.eval(x);
typename Eval2::template result<Env>::type result2 = m_eval2.eval(x);
return m_eval0.eval(hydra_thrust::tie(result1,result2));
}

private:
Eval0 m_eval0;
Eval1 m_eval1;
Eval2 m_eval2;
}; 

template<typename Eval0, typename Eval1>
__host__ __device__
actor<composite<Eval0,Eval1> > compose(const Eval0 &e0, const Eval1 &e1)
{
return actor<composite<Eval0,Eval1> >(composite<Eval0,Eval1>(e0,e1));
}

template<typename Eval0, typename Eval1, typename Eval2>
__host__ __device__
actor<composite<Eval0,Eval1,Eval2> > compose(const Eval0 &e0, const Eval1 &e1, const Eval2 &e2)
{
return actor<composite<Eval0,Eval1,Eval2> >(composite<Eval0,Eval1,Eval2>(e0,e1,e2));
}

} 
} 
} 




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/function.h>
#include <hydra/detail/external/hydra_thrust/system/tbb/detail/copy_if.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/distance.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_scan.h>

namespace hydra_thrust
{
namespace system
{
namespace tbb
{
namespace detail
{
namespace copy_if_detail
{

template<typename InputIterator1,
typename InputIterator2,
typename OutputIterator,
typename Predicate,
typename Size>
struct body
{

InputIterator1 first;
InputIterator2 stencil;
OutputIterator result;
hydra_thrust::detail::wrapped_function<Predicate,bool> pred;
Size sum;

body(InputIterator1 first, InputIterator2 stencil, OutputIterator result, Predicate pred)
: first(first), stencil(stencil), result(result), pred(pred), sum(0)
{}

body(body& b, ::tbb::split)
: first(b.first), stencil(b.stencil), result(b.result), pred(b.pred), sum(0)
{}

void operator()(const ::tbb::blocked_range<Size>& r, ::tbb::pre_scan_tag)
{
InputIterator2 iter = stencil + r.begin();

for (Size i = r.begin(); i != r.end(); ++i, ++iter)
{
if (pred(*iter))
++sum;
}
}

void operator()(const ::tbb::blocked_range<Size>& r, ::tbb::final_scan_tag)
{
InputIterator1  iter1 = first   + r.begin();
InputIterator2  iter2 = stencil + r.begin();
OutputIterator  iter3 = result  + sum;

for (Size i = r.begin(); i != r.end(); ++i, ++iter1, ++iter2)
{
if (pred(*iter2))
{
*iter3 = *iter1;
++sum;
++iter3;
}
}
}

void reverse_join(body& b)
{
sum = b.sum + sum;
} 

void assign(body& b)
{
sum = b.sum;
} 
}; 

} 

template<typename InputIterator1,
typename InputIterator2,
typename OutputIterator,
typename Predicate>
OutputIterator copy_if(tag,
InputIterator1 first,
InputIterator1 last,
InputIterator2 stencil,
OutputIterator result,
Predicate pred)
{
typedef typename hydra_thrust::iterator_difference<InputIterator1>::type Size; 
typedef typename copy_if_detail::body<InputIterator1,InputIterator2,OutputIterator,Predicate,Size> Body;

Size n = hydra_thrust::distance(first, last);

if (n != 0)
{
Body body(first, stencil, result, pred);
::tbb::parallel_scan(::tbb::blocked_range<Size>(0,n), body);
hydra_thrust::advance(result, body.sum);
}

return result;
} 

} 
} 
} 
} 


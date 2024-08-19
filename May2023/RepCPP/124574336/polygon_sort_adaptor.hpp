
#ifndef BOOST_POLYGON_SORT_ADAPTOR_HPP
#define BOOST_POLYGON_SORT_ADAPTOR_HPP
#ifdef __ICC
#pragma warning(disable:2022)
#pragma warning(disable:2023)
#endif

#include <algorithm>

namespace boost {
namespace polygon {

template<typename iterator_type>
struct dummy_to_delay_instantiation{
typedef int unit_type; 
};

template<typename T>
struct polygon_sort_adaptor {
template<typename RandomAccessIterator_Type>
static void sort(RandomAccessIterator_Type _First,
RandomAccessIterator_Type _Last)
{
std::sort(_First, _Last);
}
template<typename RandomAccessIterator_Type, typename Pred_Type>
static void sort(RandomAccessIterator_Type _First,
RandomAccessIterator_Type _Last,
const Pred_Type& _Comp)
{
std::sort(_First, _Last, _Comp);
}
};

template <typename iter_type>
void polygon_sort(iter_type _b_, iter_type _e_)
{
polygon_sort_adaptor<typename dummy_to_delay_instantiation<iter_type>::unit_type>::sort(_b_, _e_);
}

template <typename iter_type, typename pred_type>
void polygon_sort(iter_type _b_, iter_type _e_, const pred_type& _pred_)
{
polygon_sort_adaptor<typename dummy_to_delay_instantiation<iter_type>::unit_type>::sort(_b_, _e_, _pred_);
}



} 
}   
#endif

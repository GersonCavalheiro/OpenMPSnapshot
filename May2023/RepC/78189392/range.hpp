#pragma once
#include "../detail/setup.hpp"
#if !GLM_HAS_RANGE_FOR
#	error "GLM_GTX_range requires C++11 suppport or 'range for'"
#endif
#include "../gtc/type_ptr.hpp"
namespace glm{
namespace detail
{
template <typename T, precision P>
detail::component_count_t number_of_elements_(tvec2<T, P> const & v){
return detail::component_count(v);
}
template <typename T, precision P>
detail::component_count_t number_of_elements_(tvec3<T, P> const & v){
return detail::component_count(v);
}
template <typename T, precision P>
detail::component_count_t number_of_elements_(tvec4<T, P> const & v){
return detail::component_count(v);
}
template <typename genType>
detail::component_count_t number_of_elements_(genType const & m){
return detail::component_count(m) * detail::component_count(m[0]);
}
}
template <typename genType>
const typename genType::value_type * begin(genType const & v){
return value_ptr(v);
}
template <typename genType>
const typename genType::value_type * end(genType const & v){
return begin(v) + detail::number_of_elements_(v);
}
template <typename genType>
typename genType::value_type * begin(genType& v){
return value_ptr(v);
}
template <typename genType>
typename genType::value_type * end(genType& v){
return begin(v) + detail::number_of_elements_(v);
}
}


#ifndef BOOST_MP_NO_ET_OPS_HPP
#define BOOST_MP_NO_ET_OPS_HPP

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable : 4714)
#endif

namespace boost {
namespace multiprecision {

template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR number<B, et_off> operator-(const number<B, et_off>& v)
{
BOOST_STATIC_ASSERT_MSG(is_signed_number<B>::value, "Negating an unsigned type results in ill-defined behavior.");
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(v);
number<B, et_off>                                                    result(v);
result.backend().negate();
return result;
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR number<B, et_off> operator~(const number<B, et_off>& v)
{
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(v);
number<B, et_off>                                                    result;
eval_complement(result.backend(), v.backend());
return result;
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR number<B, et_off> operator+(const number<B, et_off>& a, const number<B, et_off>& b)
{
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a, b);
number<B, et_off>                                                    result;
using default_ops::eval_add;
eval_add(result.backend(), a.backend(), b.backend());
return result;
}
template <class B, class V>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if<is_compatible_arithmetic_type<V, number<B, et_off> >, number<B, et_off> >::type
operator+(const number<B, et_off>& a, const V& b)
{
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a);
number<B, et_off>                                                    result;
using default_ops::eval_add;
eval_add(result.backend(), a.backend(), number<B, et_off>::canonical_value(b));
return result;
}
template <class V, class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if<is_compatible_arithmetic_type<V, number<B, et_off> >, number<B, et_off> >::type
operator+(const V& a, const number<B, et_off>& b)
{
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(b);
number<B, et_off>                                                    result;
using default_ops::eval_add;
eval_add(result.backend(), b.backend(), number<B, et_off>::canonical_value(a));
return result;
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR number<B, et_off> operator-(const number<B, et_off>& a, const number<B, et_off>& b)
{
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a, b);
number<B, et_off>                                                    result;
using default_ops::eval_subtract;
eval_subtract(result.backend(), a.backend(), b.backend());
return result;
}
template <class B, class V>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if<is_compatible_arithmetic_type<V, number<B, et_off> >, number<B, et_off> >::type
operator-(const number<B, et_off>& a, const V& b)
{
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a);
number<B, et_off>                                                    result;
using default_ops::eval_subtract;
eval_subtract(result.backend(), a.backend(), number<B, et_off>::canonical_value(b));
return result;
}
template <class V, class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if<is_compatible_arithmetic_type<V, number<B, et_off> >, number<B, et_off> >::type
operator-(const V& a, const number<B, et_off>& b)
{
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(b);
number<B, et_off>                                                    result;
using default_ops::eval_subtract;
eval_subtract(result.backend(), number<B, et_off>::canonical_value(a), b.backend());
return result;
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR number<B, et_off> operator*(const number<B, et_off>& a, const number<B, et_off>& b)
{
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a, b);
number<B, et_off>                                                    result;
using default_ops::eval_multiply;
eval_multiply(result.backend(), a.backend(), b.backend());
return result;
}
template <class B, class V>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if<is_compatible_arithmetic_type<V, number<B, et_off> >, number<B, et_off> >::type
operator*(const number<B, et_off>& a, const V& b)
{
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a);
number<B, et_off>                                                    result;
using default_ops::eval_multiply;
eval_multiply(result.backend(), a.backend(), number<B, et_off>::canonical_value(b));
return result;
}
template <class V, class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if<is_compatible_arithmetic_type<V, number<B, et_off> >, number<B, et_off> >::type
operator*(const V& a, const number<B, et_off>& b)
{
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(b);
number<B, et_off>                                                    result;
using default_ops::eval_multiply;
eval_multiply(result.backend(), b.backend(), number<B, et_off>::canonical_value(a));
return result;
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR number<B, et_off> operator/(const number<B, et_off>& a, const number<B, et_off>& b)
{
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a, b);
number<B, et_off>                                                    result;
using default_ops::eval_divide;
eval_divide(result.backend(), a.backend(), b.backend());
return result;
}
template <class B, class V>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if<is_compatible_arithmetic_type<V, number<B, et_off> >, number<B, et_off> >::type
operator/(const number<B, et_off>& a, const V& b)
{
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a);
number<B, et_off>                                                    result;
using default_ops::eval_divide;
eval_divide(result.backend(), a.backend(), number<B, et_off>::canonical_value(b));
return result;
}
template <class V, class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if<is_compatible_arithmetic_type<V, number<B, et_off> >, number<B, et_off> >::type
operator/(const V& a, const number<B, et_off>& b)
{
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(b);
number<B, et_off>                                                    result;
using default_ops::eval_divide;
eval_divide(result.backend(), number<B, et_off>::canonical_value(a), b.backend());
return result;
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<number_category<B>::value == number_kind_integer, number<B, et_off> >::type operator%(const number<B, et_off>& a, const number<B, et_off>& b)
{
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a, b);
number<B, et_off>                                                    result;
using default_ops::eval_modulus;
eval_modulus(result.backend(), a.backend(), b.backend());
return result;
}
template <class B, class V>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<is_compatible_arithmetic_type<V, number<B, et_off> >::value && (number_category<B>::value == number_kind_integer), number<B, et_off> >::type
operator%(const number<B, et_off>& a, const V& b)
{
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a);
number<B, et_off>                                                    result;
using default_ops::eval_modulus;
eval_modulus(result.backend(), a.backend(), number<B, et_off>::canonical_value(b));
return result;
}
template <class V, class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<is_compatible_arithmetic_type<V, number<B, et_off> >::value && (number_category<B>::value == number_kind_integer), number<B, et_off> >::type
operator%(const V& a, const number<B, et_off>& b)
{
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(b);
number<B, et_off>                                                    result;
using default_ops::eval_modulus;
eval_modulus(result.backend(), number<B, et_off>::canonical_value(a), b.backend());
return result;
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<number_category<B>::value == number_kind_integer, number<B, et_off> >::type operator|(const number<B, et_off>& a, const number<B, et_off>& b)
{
number<B, et_off> result;
using default_ops::eval_bitwise_or;
eval_bitwise_or(result.backend(), a.backend(), b.backend());
return result;
}
template <class B, class V>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<is_compatible_arithmetic_type<V, number<B, et_off> >::value && (number_category<B>::value == number_kind_integer), number<B, et_off> >::type
operator|(const number<B, et_off>& a, const V& b)
{
number<B, et_off> result;
using default_ops::eval_bitwise_or;
eval_bitwise_or(result.backend(), a.backend(), number<B, et_off>::canonical_value(b));
return result;
}
template <class V, class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<is_compatible_arithmetic_type<V, number<B, et_off> >::value && (number_category<B>::value == number_kind_integer), number<B, et_off> >::type
operator|(const V& a, const number<B, et_off>& b)
{
number<B, et_off> result;
using default_ops::eval_bitwise_or;
eval_bitwise_or(result.backend(), b.backend(), number<B, et_off>::canonical_value(a));
return result;
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<number_category<B>::value == number_kind_integer, number<B, et_off> >::type operator^(const number<B, et_off>& a, const number<B, et_off>& b)
{
number<B, et_off> result;
using default_ops::eval_bitwise_xor;
eval_bitwise_xor(result.backend(), a.backend(), b.backend());
return result;
}
template <class B, class V>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<is_compatible_arithmetic_type<V, number<B, et_off> >::value && (number_category<B>::value == number_kind_integer), number<B, et_off> >::type
operator^(const number<B, et_off>& a, const V& b)
{
number<B, et_off> result;
using default_ops::eval_bitwise_xor;
eval_bitwise_xor(result.backend(), a.backend(), number<B, et_off>::canonical_value(b));
return result;
}
template <class V, class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<is_compatible_arithmetic_type<V, number<B, et_off> >::value && (number_category<B>::value == number_kind_integer), number<B, et_off> >::type
operator^(const V& a, const number<B, et_off>& b)
{
number<B, et_off> result;
using default_ops::eval_bitwise_xor;
eval_bitwise_xor(result.backend(), b.backend(), number<B, et_off>::canonical_value(a));
return result;
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<number_category<B>::value == number_kind_integer, number<B, et_off> >::type operator&(const number<B, et_off>& a, const number<B, et_off>& b)
{
number<B, et_off> result;
using default_ops::eval_bitwise_and;
eval_bitwise_and(result.backend(), a.backend(), b.backend());
return result;
}
template <class B, class V>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<is_compatible_arithmetic_type<V, number<B, et_off> >::value && (number_category<B>::value == number_kind_integer), number<B, et_off> >::type
operator&(const number<B, et_off>& a, const V& b)
{
number<B, et_off> result;
using default_ops::eval_bitwise_and;
eval_bitwise_and(result.backend(), a.backend(), number<B, et_off>::canonical_value(b));
return result;
}
template <class V, class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<is_compatible_arithmetic_type<V, number<B, et_off> >::value && (number_category<B>::value == number_kind_integer), number<B, et_off> >::type
operator&(const V& a, const number<B, et_off>& b)
{
number<B, et_off> result;
using default_ops::eval_bitwise_and;
eval_bitwise_and(result.backend(), b.backend(), number<B, et_off>::canonical_value(a));
return result;
}
template <class B, class I>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<is_integral<I>::value && (number_category<B>::value == number_kind_integer), number<B, et_off> >::type
operator<<(const number<B, et_off>& a, const I& b)
{
number<B, et_off> result(a);
using default_ops::eval_left_shift;
detail::check_shift_range(b, mpl::bool_<(sizeof(I) > sizeof(std::size_t))>(), mpl::bool_<is_signed<I>::value>());
eval_left_shift(result.backend(), b);
return result;
}
template <class B, class I>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<is_integral<I>::value && (number_category<B>::value == number_kind_integer), number<B, et_off> >::type
operator>>(const number<B, et_off>& a, const I& b)
{
number<B, et_off> result(a);
using default_ops::eval_right_shift;
detail::check_shift_range(b, mpl::bool_<(sizeof(I) > sizeof(std::size_t))>(), mpl::bool_<is_signed<I>::value>());
eval_right_shift(result.backend(), b);
return result;
}

#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES) && !(defined(__GNUC__) && ((__GNUC__ == 4) && (__GNUC_MINOR__ < 5)))
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR number<B, et_off> operator-(number<B, et_off>&& v)
{
BOOST_STATIC_ASSERT_MSG(is_signed_number<B>::value, "Negating an unsigned type results in ill-defined behavior.");
v.backend().negate();
return static_cast<number<B, et_off>&&>(v);
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<number_category<B>::value == number_kind_integer, number<B, et_off> >::type operator~(number<B, et_off>&& v)
{
eval_complement(v.backend(), v.backend());
return static_cast<number<B, et_off>&&>(v);
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR number<B, et_off> operator+(number<B, et_off>&& a, const number<B, et_off>& b)
{
using default_ops::eval_add;
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a, b);
eval_add(a.backend(), b.backend());
return static_cast<number<B, et_off>&&>(a);
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR number<B, et_off> operator+(const number<B, et_off>& a, number<B, et_off>&& b)
{
using default_ops::eval_add;
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a, b);
eval_add(b.backend(), a.backend());
return static_cast<number<B, et_off>&&>(b);
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR number<B, et_off> operator+(number<B, et_off>&& a, number<B, et_off>&& b)
{
using default_ops::eval_add;
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a, b);
eval_add(a.backend(), b.backend());
return static_cast<number<B, et_off>&&>(a);
}
template <class B, class V>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if<is_compatible_arithmetic_type<V, number<B, et_off> >, number<B, et_off> >::type
operator+(number<B, et_off>&& a, const V& b)
{
using default_ops::eval_add;
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a, b);
eval_add(a.backend(), number<B, et_off>::canonical_value(b));
return static_cast<number<B, et_off>&&>(a);
}
template <class V, class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if<is_compatible_arithmetic_type<V, number<B, et_off> >, number<B, et_off> >::type
operator+(const V& a, number<B, et_off>&& b)
{
using default_ops::eval_add;
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a, b);
eval_add(b.backend(), number<B, et_off>::canonical_value(a));
return static_cast<number<B, et_off>&&>(b);
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR number<B, et_off> operator-(number<B, et_off>&& a, const number<B, et_off>& b)
{
using default_ops::eval_subtract;
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a, b);
eval_subtract(a.backend(), b.backend());
return static_cast<number<B, et_off>&&>(a);
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if<is_signed_number<B>, number<B, et_off> >::type operator-(const number<B, et_off>& a, number<B, et_off>&& b)
{
using default_ops::eval_subtract;
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a, b);
eval_subtract(b.backend(), a.backend());
b.backend().negate();
return static_cast<number<B, et_off>&&>(b);
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR number<B, et_off> operator-(number<B, et_off>&& a, number<B, et_off>&& b)
{
using default_ops::eval_subtract;
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a, b);
eval_subtract(a.backend(), b.backend());
return static_cast<number<B, et_off>&&>(a);
}
template <class B, class V>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if<is_compatible_arithmetic_type<V, number<B, et_off> >, number<B, et_off> >::type
operator-(number<B, et_off>&& a, const V& b)
{
using default_ops::eval_subtract;
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a, b);
eval_subtract(a.backend(), number<B, et_off>::canonical_value(b));
return static_cast<number<B, et_off>&&>(a);
}
template <class V, class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<(is_compatible_arithmetic_type<V, number<B, et_off> >::value && is_signed_number<B>::value), number<B, et_off> >::type
operator-(const V& a, number<B, et_off>&& b)
{
using default_ops::eval_subtract;
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a, b);
eval_subtract(b.backend(), number<B, et_off>::canonical_value(a));
b.backend().negate();
return static_cast<number<B, et_off>&&>(b);
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR number<B, et_off> operator*(number<B, et_off>&& a, const number<B, et_off>& b)
{
using default_ops::eval_multiply;
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a, b);
eval_multiply(a.backend(), b.backend());
return static_cast<number<B, et_off>&&>(a);
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR number<B, et_off> operator*(const number<B, et_off>& a, number<B, et_off>&& b)
{
using default_ops::eval_multiply;
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a, b);
eval_multiply(b.backend(), a.backend());
return static_cast<number<B, et_off>&&>(b);
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR number<B, et_off> operator*(number<B, et_off>&& a, number<B, et_off>&& b)
{
using default_ops::eval_multiply;
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a, b);
eval_multiply(a.backend(), b.backend());
return static_cast<number<B, et_off>&&>(a);
}
template <class B, class V>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if<is_compatible_arithmetic_type<V, number<B, et_off> >, number<B, et_off> >::type
operator*(number<B, et_off>&& a, const V& b)
{
using default_ops::eval_multiply;
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a, b);
eval_multiply(a.backend(), number<B, et_off>::canonical_value(b));
return static_cast<number<B, et_off>&&>(a);
}
template <class V, class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if<is_compatible_arithmetic_type<V, number<B, et_off> >, number<B, et_off> >::type
operator*(const V& a, number<B, et_off>&& b)
{
using default_ops::eval_multiply;
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a, b);
eval_multiply(b.backend(), number<B, et_off>::canonical_value(a));
return static_cast<number<B, et_off>&&>(b);
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR number<B, et_off> operator/(number<B, et_off>&& a, const number<B, et_off>& b)
{
using default_ops::eval_divide;
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a, b);
eval_divide(a.backend(), b.backend());
return static_cast<number<B, et_off>&&>(a);
}
template <class B, class V>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if<is_compatible_arithmetic_type<V, number<B, et_off> >, number<B, et_off> >::type
operator/(number<B, et_off>&& a, const V& b)
{
using default_ops::eval_divide;
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a, b);
eval_divide(a.backend(), number<B, et_off>::canonical_value(b));
return static_cast<number<B, et_off>&&>(a);
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<number_category<B>::value == number_kind_integer, number<B, et_off> >::type operator%(number<B, et_off>&& a, const number<B, et_off>& b)
{
using default_ops::eval_modulus;
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a, b);
eval_modulus(a.backend(), b.backend());
return static_cast<number<B, et_off>&&>(a);
}
template <class B, class V>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<is_compatible_arithmetic_type<V, number<B, et_off> >::value && (number_category<B>::value == number_kind_integer), number<B, et_off> >::type
operator%(number<B, et_off>&& a, const V& b)
{
using default_ops::eval_modulus;
detail::scoped_default_precision<multiprecision::number<B, et_off> > precision_guard(a, b);
eval_modulus(a.backend(), number<B, et_off>::canonical_value(b));
return static_cast<number<B, et_off>&&>(a);
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<number_category<B>::value == number_kind_integer, number<B, et_off> >::type operator|(number<B, et_off>&& a, const number<B, et_off>& b)
{
using default_ops::eval_bitwise_or;
eval_bitwise_or(a.backend(), b.backend());
return static_cast<number<B, et_off>&&>(a);
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<number_category<B>::value == number_kind_integer, number<B, et_off> >::type operator|(const number<B, et_off>& a, number<B, et_off>&& b)
{
using default_ops::eval_bitwise_or;
eval_bitwise_or(b.backend(), a.backend());
return static_cast<number<B, et_off>&&>(b);
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<number_category<B>::value == number_kind_integer, number<B, et_off> >::type operator|(number<B, et_off>&& a, number<B, et_off>&& b)
{
using default_ops::eval_bitwise_or;
eval_bitwise_or(a.backend(), b.backend());
return static_cast<number<B, et_off>&&>(a);
}
template <class B, class V>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<is_compatible_arithmetic_type<V, number<B, et_off> >::value && (number_category<B>::value == number_kind_integer), number<B, et_off> >::type
operator|(number<B, et_off>&& a, const V& b)
{
using default_ops::eval_bitwise_or;
eval_bitwise_or(a.backend(), number<B, et_off>::canonical_value(b));
return static_cast<number<B, et_off>&&>(a);
}
template <class V, class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<is_compatible_arithmetic_type<V, number<B, et_off> >::value && (number_category<B>::value == number_kind_integer), number<B, et_off> >::type
operator|(const V& a, number<B, et_off>&& b)
{
using default_ops::eval_bitwise_or;
eval_bitwise_or(b.backend(), number<B, et_off>::canonical_value(a));
return static_cast<number<B, et_off>&&>(b);
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<number_category<B>::value == number_kind_integer, number<B, et_off> >::type operator^(number<B, et_off>&& a, const number<B, et_off>& b)
{
using default_ops::eval_bitwise_xor;
eval_bitwise_xor(a.backend(), b.backend());
return static_cast<number<B, et_off>&&>(a);
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<number_category<B>::value == number_kind_integer, number<B, et_off> >::type operator^(const number<B, et_off>& a, number<B, et_off>&& b)
{
using default_ops::eval_bitwise_xor;
eval_bitwise_xor(b.backend(), a.backend());
return static_cast<number<B, et_off>&&>(b);
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<number_category<B>::value == number_kind_integer, number<B, et_off> >::type operator^(number<B, et_off>&& a, number<B, et_off>&& b)
{
using default_ops::eval_bitwise_xor;
eval_bitwise_xor(a.backend(), b.backend());
return static_cast<number<B, et_off>&&>(a);
}
template <class B, class V>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<is_compatible_arithmetic_type<V, number<B, et_off> >::value && (number_category<B>::value == number_kind_integer), number<B, et_off> >::type
operator^(number<B, et_off>&& a, const V& b)
{
using default_ops::eval_bitwise_xor;
eval_bitwise_xor(a.backend(), number<B, et_off>::canonical_value(b));
return static_cast<number<B, et_off>&&>(a);
}
template <class V, class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<is_compatible_arithmetic_type<V, number<B, et_off> >::value && (number_category<B>::value == number_kind_integer), number<B, et_off> >::type
operator^(const V& a, number<B, et_off>&& b)
{
using default_ops::eval_bitwise_xor;
eval_bitwise_xor(b.backend(), number<B, et_off>::canonical_value(a));
return static_cast<number<B, et_off>&&>(b);
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<number_category<B>::value == number_kind_integer, number<B, et_off> >::type operator&(number<B, et_off>&& a, const number<B, et_off>& b)
{
using default_ops::eval_bitwise_and;
eval_bitwise_and(a.backend(), b.backend());
return static_cast<number<B, et_off>&&>(a);
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<number_category<B>::value == number_kind_integer, number<B, et_off> >::type operator&(const number<B, et_off>& a, number<B, et_off>&& b)
{
using default_ops::eval_bitwise_and;
eval_bitwise_and(b.backend(), a.backend());
return static_cast<number<B, et_off>&&>(b);
}
template <class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<number_category<B>::value == number_kind_integer, number<B, et_off> >::type operator&(number<B, et_off>&& a, number<B, et_off>&& b)
{
using default_ops::eval_bitwise_and;
eval_bitwise_and(a.backend(), b.backend());
return static_cast<number<B, et_off>&&>(a);
}
template <class B, class V>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<is_compatible_arithmetic_type<V, number<B, et_off> >::value && (number_category<B>::value == number_kind_integer), number<B, et_off> >::type
operator&(number<B, et_off>&& a, const V& b)
{
using default_ops::eval_bitwise_and;
eval_bitwise_and(a.backend(), number<B, et_off>::canonical_value(b));
return static_cast<number<B, et_off>&&>(a);
}
template <class V, class B>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<is_compatible_arithmetic_type<V, number<B, et_off> >::value && (number_category<B>::value == number_kind_integer), number<B, et_off> >::type
operator&(const V& a, number<B, et_off>&& b)
{
using default_ops::eval_bitwise_and;
eval_bitwise_and(b.backend(), number<B, et_off>::canonical_value(a));
return static_cast<number<B, et_off>&&>(b);
}
template <class B, class I>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<is_integral<I>::value && (number_category<B>::value == number_kind_integer), number<B, et_off> >::type
operator<<(number<B, et_off>&& a, const I& b)
{
using default_ops::eval_left_shift;
eval_left_shift(a.backend(), b);
return static_cast<number<B, et_off>&&>(a);
}
template <class B, class I>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<is_integral<I>::value && (number_category<B>::value == number_kind_integer), number<B, et_off> >::type
operator>>(number<B, et_off>&& a, const I& b)
{
using default_ops::eval_right_shift;
eval_right_shift(a.backend(), b);
return static_cast<number<B, et_off>&&>(a);
}

#endif

}
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif 

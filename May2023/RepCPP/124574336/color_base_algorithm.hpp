#ifndef BOOST_GIL_COLOR_BASE_ALGORITHM_HPP
#define BOOST_GIL_COLOR_BASE_ALGORITHM_HPP

#include <boost/gil/concepts.hpp>
#include <boost/gil/utilities.hpp>
#include <boost/gil/detail/mp11.hpp>

#include <boost/config.hpp>

#include <algorithm>
#include <type_traits>

namespace boost { namespace gil {




template <typename ColorBase>
struct size : public mp11::mp_size<typename ColorBase::layout_t::color_space_t> {};



template <typename ColorBase, int K>
struct kth_semantic_element_type
{
using channel_mapping_t = typename ColorBase::layout_t::channel_mapping_t;
static_assert(K < mp11::mp_size<channel_mapping_t>::value,
"K index should be less than size of channel_mapping_t sequence");

static constexpr int semantic_index = mp11::mp_at_c<channel_mapping_t, K>::type::value;
using type = typename kth_element_type<ColorBase, semantic_index>::type;
};

template <typename ColorBase, int K>
struct kth_semantic_element_reference_type
{
using channel_mapping_t = typename ColorBase::layout_t::channel_mapping_t;
static_assert(K < mp11::mp_size<channel_mapping_t>::value,
"K index should be less than size of channel_mapping_t sequence");

static constexpr int semantic_index = mp11::mp_at_c<channel_mapping_t, K>::type::value;
using type = typename kth_element_reference_type<ColorBase, semantic_index>::type;
static type get(ColorBase& cb) { return gil::at_c<semantic_index>(cb); }
};

template <typename ColorBase, int K>
struct kth_semantic_element_const_reference_type
{
using channel_mapping_t = typename ColorBase::layout_t::channel_mapping_t;
static_assert(K < mp11::mp_size<channel_mapping_t>::value,
"K index should be less than size of channel_mapping_t sequence");

static constexpr int semantic_index = mp11::mp_at_c<channel_mapping_t, K>::type::value;
using type = typename kth_element_const_reference_type<ColorBase,semantic_index>::type;
static type get(const ColorBase& cb) { return gil::at_c<semantic_index>(cb); }
};

template <int K, typename ColorBase>
inline
auto semantic_at_c(ColorBase& p)
-> typename std::enable_if
<
!std::is_const<ColorBase>::value,
typename kth_semantic_element_reference_type<ColorBase, K>::type
>::type
{
return kth_semantic_element_reference_type<ColorBase, K>::get(p);
}

template <int K, typename ColorBase>
inline
auto semantic_at_c(ColorBase const& p)
-> typename kth_semantic_element_const_reference_type<ColorBase, K>::type
{
return kth_semantic_element_const_reference_type<ColorBase, K>::get(p);
}




template <typename ColorBase, typename Color>
struct contains_color
: mp11::mp_contains<typename ColorBase::layout_t::color_space_t, Color>
{};

template <typename ColorBase, typename Color>
struct color_index_type : public detail::type_to_index<typename ColorBase::layout_t::color_space_t,Color> {};

template <typename ColorBase, typename Color>
struct color_element_type : public kth_semantic_element_type<ColorBase,color_index_type<ColorBase,Color>::value> {};

template <typename ColorBase, typename Color>
struct color_element_reference_type : public kth_semantic_element_reference_type<ColorBase,color_index_type<ColorBase,Color>::value> {};

template <typename ColorBase, typename Color>
struct color_element_const_reference_type : public kth_semantic_element_const_reference_type<ColorBase,color_index_type<ColorBase,Color>::value> {};

template <typename ColorBase, typename Color>
typename color_element_reference_type<ColorBase,Color>::type get_color(ColorBase& cb, Color=Color()) {
return color_element_reference_type<ColorBase,Color>::get(cb);
}

template <typename ColorBase, typename Color>
typename color_element_const_reference_type<ColorBase,Color>::type get_color(const ColorBase& cb, Color=Color()) {
return color_element_const_reference_type<ColorBase,Color>::get(cb);
}



template <typename ColorBase>
struct element_type : public kth_element_type<ColorBase, 0> {};

template <typename ColorBase>
struct element_reference_type : public kth_element_reference_type<ColorBase, 0> {};

template <typename ColorBase>
struct element_const_reference_type : public kth_element_const_reference_type<ColorBase, 0> {};


namespace detail {

template <int N>
struct element_recursion
{

#if defined(BOOST_GCC) && (BOOST_GCC >= 40900)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif

template <typename P1,typename P2>
static bool static_equal(const P1& p1, const P2& p2)
{
return element_recursion<N-1>::static_equal(p1,p2) &&
semantic_at_c<N-1>(p1)==semantic_at_c<N-1>(p2);
}

template <typename P1,typename P2>
static void static_copy(const P1& p1, P2& p2)
{
element_recursion<N-1>::static_copy(p1,p2);
semantic_at_c<N-1>(p2)=semantic_at_c<N-1>(p1);
}

template <typename P,typename T2>
static void static_fill(P& p, T2 v)
{
element_recursion<N-1>::static_fill(p,v);
semantic_at_c<N-1>(p)=v;
}

template <typename Dst,typename Op>
static void static_generate(Dst& dst, Op op)
{
element_recursion<N-1>::static_generate(dst,op);
semantic_at_c<N-1>(dst)=op();
}

#if defined(BOOST_GCC) && (BOOST_GCC >= 40900)
#pragma GCC diagnostic pop
#endif

template <typename P1,typename Op>
static Op static_for_each(P1& p1, Op op) {
Op op2(element_recursion<N-1>::static_for_each(p1,op));
op2(semantic_at_c<N-1>(p1));
return op2;
}
template <typename P1,typename Op>
static Op static_for_each(const P1& p1, Op op) {
Op op2(element_recursion<N-1>::static_for_each(p1,op));
op2(semantic_at_c<N-1>(p1));
return op2;
}
template <typename P1,typename P2,typename Op>
static Op static_for_each(P1& p1, P2& p2, Op op) {
Op op2(element_recursion<N-1>::static_for_each(p1,p2,op));
op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2));
return op2;
}
template <typename P1,typename P2,typename Op>
static Op static_for_each(P1& p1, const P2& p2, Op op) {
Op op2(element_recursion<N-1>::static_for_each(p1,p2,op));
op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2));
return op2;
}
template <typename P1,typename P2,typename Op>
static Op static_for_each(const P1& p1, P2& p2, Op op) {
Op op2(element_recursion<N-1>::static_for_each(p1,p2,op));
op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2));
return op2;
}
template <typename P1,typename P2,typename Op>
static Op static_for_each(const P1& p1, const P2& p2, Op op) {
Op op2(element_recursion<N-1>::static_for_each(p1,p2,op));
op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2));
return op2;
}
template <typename P1,typename P2,typename P3,typename Op>
static Op static_for_each(P1& p1, P2& p2, P3& p3, Op op) {
Op op2(element_recursion<N-1>::static_for_each(p1,p2,p3,op));
op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2), semantic_at_c<N-1>(p3));
return op2;
}
template <typename P1,typename P2,typename P3,typename Op>
static Op static_for_each(P1& p1, P2& p2, const P3& p3, Op op) {
Op op2(element_recursion<N-1>::static_for_each(p1,p2,p3,op));
op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2), semantic_at_c<N-1>(p3));
return op2;
}
template <typename P1,typename P2,typename P3,typename Op>
static Op static_for_each(P1& p1, const P2& p2, P3& p3, Op op) {
Op op2(element_recursion<N-1>::static_for_each(p1,p2,p3,op));
op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2), semantic_at_c<N-1>(p3));
return op2;
}
template <typename P1,typename P2,typename P3,typename Op>
static Op static_for_each(P1& p1, const P2& p2, const P3& p3, Op op) {
Op op2(element_recursion<N-1>::static_for_each(p1,p2,p3,op));
op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2), semantic_at_c<N-1>(p3));
return op2;
}
template <typename P1,typename P2,typename P3,typename Op>
static Op static_for_each(const P1& p1, P2& p2, P3& p3, Op op) {
Op op2(element_recursion<N-1>::static_for_each(p1,p2,p3,op));
op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2), semantic_at_c<N-1>(p3));
return op2;
}
template <typename P1,typename P2,typename P3,typename Op>
static Op static_for_each(const P1& p1, P2& p2, const P3& p3, Op op) {
Op op2(element_recursion<N-1>::static_for_each(p1,p2,p3,op));
op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2), semantic_at_c<N-1>(p3));
return op2;
}
template <typename P1,typename P2,typename P3,typename Op>
static Op static_for_each(const P1& p1, const P2& p2, P3& p3, Op op) {
Op op2(element_recursion<N-1>::static_for_each(p1,p2,p3,op));
op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2), semantic_at_c<N-1>(p3));
return op2;
}
template <typename P1,typename P2,typename P3,typename Op>
static Op static_for_each(const P1& p1, const P2& p2, const P3& p3, Op op) {
Op op2(element_recursion<N-1>::static_for_each(p1,p2,p3,op));
op2(semantic_at_c<N-1>(p1), semantic_at_c<N-1>(p2), semantic_at_c<N-1>(p3));
return op2;
}
template <typename P1,typename Dst,typename Op>
static Op static_transform(P1& src, Dst& dst, Op op) {
Op op2(element_recursion<N-1>::static_transform(src,dst,op));
semantic_at_c<N-1>(dst)=op2(semantic_at_c<N-1>(src));
return op2;
}
template <typename P1,typename Dst,typename Op>
static Op static_transform(const P1& src, Dst& dst, Op op) {
Op op2(element_recursion<N-1>::static_transform(src,dst,op));
semantic_at_c<N-1>(dst)=op2(semantic_at_c<N-1>(src));
return op2;
}
template <typename P1,typename P2,typename Dst,typename Op>
static Op static_transform(P1& src1, P2& src2, Dst& dst, Op op) {
Op op2(element_recursion<N-1>::static_transform(src1,src2,dst,op));
semantic_at_c<N-1>(dst)=op2(semantic_at_c<N-1>(src1), semantic_at_c<N-1>(src2));
return op2;
}
template <typename P1,typename P2,typename Dst,typename Op>
static Op static_transform(P1& src1, const P2& src2, Dst& dst, Op op) {
Op op2(element_recursion<N-1>::static_transform(src1,src2,dst,op));
semantic_at_c<N-1>(dst)=op2(semantic_at_c<N-1>(src1), semantic_at_c<N-1>(src2));
return op2;
}
template <typename P1,typename P2,typename Dst,typename Op>
static Op static_transform(const P1& src1, P2& src2, Dst& dst, Op op) {
Op op2(element_recursion<N-1>::static_transform(src1,src2,dst,op));
semantic_at_c<N-1>(dst)=op2(semantic_at_c<N-1>(src1), semantic_at_c<N-1>(src2));
return op2;
}
template <typename P1,typename P2,typename Dst,typename Op>
static Op static_transform(const P1& src1, const P2& src2, Dst& dst, Op op) {
Op op2(element_recursion<N-1>::static_transform(src1,src2,dst,op));
semantic_at_c<N-1>(dst)=op2(semantic_at_c<N-1>(src1), semantic_at_c<N-1>(src2));
return op2;
}
};

template<> struct element_recursion<0> {
template <typename P1,typename P2>
static bool static_equal(const P1&, const P2&) { return true; }
template <typename P1,typename P2>
static void static_copy(const P1&, const P2&) {}
template <typename P, typename T2>
static void static_fill(const P&, T2) {}
template <typename Dst,typename Op>
static void static_generate(const Dst&,Op){}
template <typename P1,typename Op>
static Op static_for_each(const P1&,Op op){return op;}
template <typename P1,typename P2,typename Op>
static Op static_for_each(const P1&,const P2&,Op op){return op;}
template <typename P1,typename P2,typename P3,typename Op>
static Op static_for_each(const P1&,const P2&,const P3&,Op op){return op;}
template <typename P1,typename Dst,typename Op>
static Op static_transform(const P1&,const Dst&,Op op){return op;}
template <typename P1,typename P2,typename Dst,typename Op>
static Op static_transform(const P1&,const P2&,const Dst&,Op op){return op;}
};

template <typename Q> inline const Q& mutable_min(const Q& x, const Q& y) { return x<y ? x : y; }
template <typename Q> inline       Q& mutable_min(      Q& x,       Q& y) { return x<y ? x : y; }
template <typename Q> inline const Q& mutable_max(const Q& x, const Q& y) { return x<y ? y : x; }
template <typename Q> inline       Q& mutable_max(      Q& x,       Q& y) { return x<y ? y : x; }


template <int N>
struct min_max_recur {
template <typename P> static typename element_const_reference_type<P>::type max_(const P& p) {
return mutable_max(min_max_recur<N-1>::max_(p),semantic_at_c<N-1>(p));
}
template <typename P> static typename element_reference_type<P>::type       max_(      P& p) {
return mutable_max(min_max_recur<N-1>::max_(p),semantic_at_c<N-1>(p));
}
template <typename P> static typename element_const_reference_type<P>::type min_(const P& p) {
return mutable_min(min_max_recur<N-1>::min_(p),semantic_at_c<N-1>(p));
}
template <typename P> static typename element_reference_type<P>::type       min_(      P& p) {
return mutable_min(min_max_recur<N-1>::min_(p),semantic_at_c<N-1>(p));
}
};

template <>
struct min_max_recur<1> {
template <typename P> static typename element_const_reference_type<P>::type max_(const P& p) { return semantic_at_c<0>(p); }
template <typename P> static typename element_reference_type<P>::type       max_(      P& p) { return semantic_at_c<0>(p); }
template <typename P> static typename element_const_reference_type<P>::type min_(const P& p) { return semantic_at_c<0>(p); }
template <typename P> static typename element_reference_type<P>::type       min_(      P& p) { return semantic_at_c<0>(p); }
};
}  


template <typename P>
BOOST_FORCEINLINE
typename element_const_reference_type<P>::type static_max(const P& p) { return detail::min_max_recur<size<P>::value>::max_(p); }

template <typename P>
BOOST_FORCEINLINE
typename element_reference_type<P>::type       static_max(      P& p) { return detail::min_max_recur<size<P>::value>::max_(p); }

template <typename P>
BOOST_FORCEINLINE
typename element_const_reference_type<P>::type static_min(const P& p) { return detail::min_max_recur<size<P>::value>::min_(p); }

template <typename P>
BOOST_FORCEINLINE
typename element_reference_type<P>::type       static_min(      P& p) { return detail::min_max_recur<size<P>::value>::min_(p); }


template <typename P1,typename P2>
BOOST_FORCEINLINE
bool static_equal(const P1& p1, const P2& p2) { return detail::element_recursion<size<P1>::value>::static_equal(p1,p2); }



template <typename Src,typename Dst>
BOOST_FORCEINLINE
void static_copy(const Src& src, Dst& dst)
{
detail::element_recursion<size<Dst>::value>::static_copy(src, dst);
}



template <typename P,typename V>
BOOST_FORCEINLINE
void static_fill(P& p, const V& v)
{
detail::element_recursion<size<P>::value>::static_fill(p,v);
}



template <typename P1,typename Op>
BOOST_FORCEINLINE
void static_generate(P1& dst,Op op)                      { detail::element_recursion<size<P1>::value>::static_generate(dst,op); }


template <typename Src,typename Dst,typename Op>
BOOST_FORCEINLINE
Op static_transform(Src& src,Dst& dst,Op op)              { return detail::element_recursion<size<Dst>::value>::static_transform(src,dst,op); }
template <typename Src,typename Dst,typename Op>
BOOST_FORCEINLINE
Op static_transform(const Src& src,Dst& dst,Op op)              { return detail::element_recursion<size<Dst>::value>::static_transform(src,dst,op); }
template <typename P2,typename P3,typename Dst,typename Op>
BOOST_FORCEINLINE
Op static_transform(P2& p2,P3& p3,Dst& dst,Op op) { return detail::element_recursion<size<Dst>::value>::static_transform(p2,p3,dst,op); }
template <typename P2,typename P3,typename Dst,typename Op>
BOOST_FORCEINLINE
Op static_transform(P2& p2,const P3& p3,Dst& dst,Op op) { return detail::element_recursion<size<Dst>::value>::static_transform(p2,p3,dst,op); }
template <typename P2,typename P3,typename Dst,typename Op>
BOOST_FORCEINLINE
Op static_transform(const P2& p2,P3& p3,Dst& dst,Op op) { return detail::element_recursion<size<Dst>::value>::static_transform(p2,p3,dst,op); }
template <typename P2,typename P3,typename Dst,typename Op>
BOOST_FORCEINLINE
Op static_transform(const P2& p2,const P3& p3,Dst& dst,Op op) { return detail::element_recursion<size<Dst>::value>::static_transform(p2,p3,dst,op); }


template <typename P1,typename Op>
BOOST_FORCEINLINE
Op static_for_each(      P1& p1, Op op)                          { return detail::element_recursion<size<P1>::value>::static_for_each(p1,op); }
template <typename P1,typename Op>
BOOST_FORCEINLINE
Op static_for_each(const P1& p1, Op op)                          { return detail::element_recursion<size<P1>::value>::static_for_each(p1,op); }
template <typename P1,typename P2,typename Op>
BOOST_FORCEINLINE
Op static_for_each(P1& p1,      P2& p2, Op op)             { return detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,op); }
template <typename P1,typename P2,typename Op>
BOOST_FORCEINLINE
Op static_for_each(P1& p1,const P2& p2, Op op)             { return detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,op); }
template <typename P1,typename P2,typename Op>
BOOST_FORCEINLINE
Op static_for_each(const P1& p1,      P2& p2, Op op)             { return detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,op); }
template <typename P1,typename P2,typename Op>
BOOST_FORCEINLINE
Op static_for_each(const P1& p1,const P2& p2, Op op)             { return detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,op); }
template <typename P1,typename P2,typename P3,typename Op>
BOOST_FORCEINLINE
Op static_for_each(P1& p1,P2& p2,P3& p3,Op op) { return detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,p3,op); }
template <typename P1,typename P2,typename P3,typename Op>
BOOST_FORCEINLINE
Op static_for_each(P1& p1,P2& p2,const P3& p3,Op op) { return detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,p3,op); }
template <typename P1,typename P2,typename P3,typename Op>
BOOST_FORCEINLINE
Op static_for_each(P1& p1,const P2& p2,P3& p3,Op op) { return detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,p3,op); }
template <typename P1,typename P2,typename P3,typename Op>
BOOST_FORCEINLINE
Op static_for_each(P1& p1,const P2& p2,const P3& p3,Op op) { return detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,p3,op); }
template <typename P1,typename P2,typename P3,typename Op>
BOOST_FORCEINLINE
Op static_for_each(const P1& p1,P2& p2,P3& p3,Op op) { return detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,p3,op); }
template <typename P1,typename P2,typename P3,typename Op>
BOOST_FORCEINLINE
Op static_for_each(const P1& p1,P2& p2,const P3& p3,Op op) { return detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,p3,op); }
template <typename P1,typename P2,typename P3,typename Op>
BOOST_FORCEINLINE
Op static_for_each(const P1& p1,const P2& p2,P3& p3,Op op) { return detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,p3,op); }
template <typename P1,typename P2,typename P3,typename Op>
BOOST_FORCEINLINE
Op static_for_each(const P1& p1,const P2& p2,const P3& p3,Op op) { return detail::element_recursion<size<P1>::value>::static_for_each(p1,p2,p3,op); }

} }  

#endif

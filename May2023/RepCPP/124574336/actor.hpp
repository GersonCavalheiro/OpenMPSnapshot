
#ifndef BOOST_SPIRIT_CLASSIC_PHOENIX_ACTOR_HPP
#define BOOST_SPIRIT_CLASSIC_PHOENIX_ACTOR_HPP

#include <boost/spirit/home/classic/phoenix/tuples.hpp>
#include <boost/type_traits/remove_reference.hpp>

namespace phoenix {


struct assign_op;
struct index_op;

namespace impl {

template <typename OperationT, typename BaseT, typename B>
struct make_binary1;
}

template <typename TupleT>
struct unpack_tuple : public TupleT {

typedef TupleT tuple_t;

unpack_tuple() {}
unpack_tuple(tuple_t const &tuple_) : TupleT(tuple_) {}
};

template <typename ActorT, typename TupleT>
struct actor_result {

typedef typename ActorT::template result<TupleT>::type type;
typedef typename boost::remove_reference<type>::type plain_type;
};

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
#pragma warning(push)
#pragma warning(disable:4512) 
#endif

template <typename BaseT>
struct actor : public BaseT {

actor();
actor(BaseT const& base);

typename actor_result<BaseT, tuple<> >::type
operator()() const;

template <typename A>
typename actor_result<BaseT, tuple<A&> >::type
operator()(A& a) const;

template <typename A, typename B>
typename actor_result<BaseT, tuple<A&, B&> >::type
operator()(A& a, B& b) const;

template <typename A, typename B, typename C>
typename actor_result<BaseT, tuple<A&, B&, C&> >::type
operator()(A& a, B& b, C& c) const;

#if PHOENIX_LIMIT > 3
template <typename A, typename B, typename C, typename D>
typename actor_result<BaseT, tuple<A&, B&, C&, D&> >::type
operator()(A& a, B& b, C& c, D& d) const;

template <typename A, typename B, typename C, typename D, typename E>
typename actor_result<BaseT, tuple<A&, B&, C&, D&, E&> >::type
operator()(A& a, B& b, C& c, D& d, E& e) const;

template <
typename A, typename B, typename C, typename D, typename E,
typename F>
typename actor_result<BaseT, tuple<A&, B&, C&, D&, E&, F&> >::type
operator()(A& a, B& b, C& c, D& d, E& e, F& f) const;

#if PHOENIX_LIMIT > 6

template <
typename A, typename B, typename C, typename D, typename E,
typename F, typename G>
typename actor_result<BaseT, tuple<A&, B&, C&, D&, E&, F&, G&> >::type
operator()(A& a, B& b, C& c, D& d, E& e, F& f, G& g) const;

template <
typename A, typename B, typename C, typename D, typename E,
typename F, typename G, typename H>
typename actor_result<BaseT,
tuple<A&, B&, C&, D&, E&, F&, G&, H&>
>::type
operator()(A& a, B& b, C& c, D& d, E& e, F& f, G& g, H& h) const;

template <
typename A, typename B, typename C, typename D, typename E,
typename F, typename G, typename H, typename I>
typename actor_result<BaseT,
tuple<A&, B&, C&, D&, E&, F&, G&, H&, I&>
>::type
operator()(A& a, B& b, C& c, D& d, E& e, F& f, G& g, H& h, I& i) const;

#if PHOENIX_LIMIT > 9

template <
typename A, typename B, typename C, typename D, typename E,
typename F, typename G, typename H, typename I, typename J>
typename actor_result<BaseT,
tuple<A&, B&, C&, D&, E&, F&, G&, H&, I&, J&>
>::type
operator()(
A& a, B& b, C& c, D& d, E& e, F& f, G& g, H& h, I& i, J& j) const;

template <
typename A, typename B, typename C, typename D, typename E,
typename F, typename G, typename H, typename I, typename J,
typename K>
typename actor_result<BaseT,
tuple<A&, B&, C&, D&, E&, F&, G&, H&, I&, J&, K&>
>::type
operator()(
A& a, B& b, C& c, D& d, E& e, F& f, G& g, H& h, I& i, J& j,
K& k) const;

template <
typename A, typename B, typename C, typename D, typename E,
typename F, typename G, typename H, typename I, typename J,
typename K, typename L>
typename actor_result<BaseT,
tuple<A&, B&, C&, D&, E&, F&, G&, H&, I&, J&, K&, L&>
>::type
operator()(
A& a, B& b, C& c, D& d, E& e, F& f, G& g, H& h, I& i, J& j,
K& k, L& l) const;

#if PHOENIX_LIMIT > 12

template <
typename A, typename B, typename C, typename D, typename E,
typename F, typename G, typename H, typename I, typename J,
typename K, typename L, typename M>
typename actor_result<BaseT,
tuple<A&, B&, C&, D&, E&, F&, G&, H&, I&, J&, K&, L&, M&>
>::type
operator()(
A& a, B& b, C& c, D& d, E& e, F& f, G& g, H& h, I& i, J& j,
K& k, L& l, M& m) const;

template <
typename A, typename B, typename C, typename D, typename E,
typename F, typename G, typename H, typename I, typename J,
typename K, typename L, typename M, typename N>
typename actor_result<BaseT,
tuple<A&, B&, C&, D&, E&, F&, G&, H&, I&, J&, K&, L&, M&, N&>
>::type
operator()(
A& a, B& b, C& c, D& d, E& e, F& f, G& g, H& h, I& i, J& j,
K& k, L& l, M& m, N& n) const;

template <
typename A, typename B, typename C, typename D, typename E,
typename F, typename G, typename H, typename I, typename J,
typename K, typename L, typename M, typename N, typename O>
typename actor_result<BaseT,
tuple<A&, B&, C&, D&, E&, F&, G&, H&, I&, J&, K&, L&, M&, N&, O&>
>::type
operator()(
A& a, B& b, C& c, D& d, E& e, F& f, G& g, H& h, I& i, J& j,
K& k, L& l, M& m, N& n, O& o) const;

#endif
#endif
#endif
#endif

template <typename TupleT>
typename actor_result<BaseT, unpack_tuple<TupleT> >::type
operator()(unpack_tuple<TupleT> const &t) const;

template <typename B>
typename impl::make_binary1<assign_op, BaseT, B>::type
operator=(B const& b) const;

template <typename B>
typename impl::make_binary1<index_op, BaseT, B>::type
operator[](B const& b) const;
};

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
#pragma warning(pop)
#endif

template <typename T>
struct as_actor;

template <typename BaseT>
struct as_actor<actor<BaseT> > {

typedef actor<BaseT> type;
static type convert(actor<BaseT> const& x) { return x; }
};

template <>
struct as_actor<nil_t> {

typedef nil_t type;
static nil_t convert(nil_t )
{ return nil_t(); }
};

template <>
struct as_actor<void> {

typedef void type;
};

template <typename BaseT>
actor<BaseT>::actor()
:   BaseT() {}

template <typename BaseT>
actor<BaseT>::actor(BaseT const& base)
:   BaseT(base) {}

template <typename BaseT>
inline typename actor_result<BaseT, tuple<> >::type
actor<BaseT>::operator()() const
{
return BaseT::eval(tuple<>());
}

template <typename BaseT>
template <typename A>
inline typename actor_result<BaseT, tuple<A&> >::type
actor<BaseT>::operator()(A& a_) const
{
return BaseT::eval(tuple<A&>(a_));
}

template <typename BaseT>
template <typename A, typename B>
inline typename actor_result<BaseT, tuple<A&, B&> >::type
actor<BaseT>::operator()(A& a_, B& b_) const
{
return BaseT::eval(tuple<A&, B&>(a_, b_));
}

template <typename BaseT>
template <typename A, typename B, typename C>
inline typename actor_result<BaseT, tuple<A&, B&, C&> >::type
actor<BaseT>::operator()(A& a_, B& b_, C& c_) const
{
return BaseT::eval(tuple<A&, B&, C&>(a_, b_, c_));
}

#if PHOENIX_LIMIT > 3
template <typename BaseT>
template <typename A, typename B, typename C, typename D>
inline typename actor_result<BaseT, tuple<A&, B&, C&, D&> >::type
actor<BaseT>::operator()(A& a_, B& b_, C& c_, D& d_) const
{
return BaseT::eval(tuple<A&, B&, C&, D&>(a_, b_, c_, d_));
}

template <typename BaseT>
template <typename A, typename B, typename C, typename D, typename E>
inline typename actor_result<BaseT, tuple<A&, B&, C&, D&, E&> >::type
actor<BaseT>::operator()(A& a_, B& b_, C& c_, D& d_, E& e_) const
{
return BaseT::eval(tuple<A&, B&, C&, D&, E&>(a_, b_, c_, d_, e_));
}

template <typename BaseT>
template <
typename A, typename B, typename C, typename D, typename E,
typename F>
inline typename actor_result<BaseT,
tuple<A&, B&, C&, D&, E&, F&>
>::type
actor<BaseT>::operator()(
A& a_, B& b_, C& c_, D& d_, E& e_, F& f_
) const
{
return BaseT::eval(
tuple<A&, B&, C&, D&, E&, F&>
(a_, b_, c_, d_, e_, f_)
);
}

#if PHOENIX_LIMIT > 6
template <typename BaseT>
template <
typename A, typename B, typename C, typename D, typename E,
typename F, typename G>
inline typename actor_result<BaseT,
tuple<A&, B&, C&, D&, E&, F&, G&>
>::type
actor<BaseT>::operator()(
A& a_, B& b_, C& c_, D& d_, E& e_, F& f_, G& g_
) const
{
return BaseT::eval(
tuple<A&, B&, C&, D&, E&, F&, G&>
(a_, b_, c_, d_, e_, f_, g_)
);
}

template <typename BaseT>
template <
typename A, typename B, typename C, typename D, typename E,
typename F, typename G, typename H>
inline typename actor_result<BaseT,
tuple<A&, B&, C&, D&, E&, F&, G&, H&>
>::type
actor<BaseT>::operator()(
A& a_, B& b_, C& c_, D& d_, E& e_, F& f_, G& g_, H& h_
) const
{
return BaseT::eval(
tuple<A&, B&, C&, D&, E&, F&, G&, H&>
(a_, b_, c_, d_, e_, f_, g_, h_)
);
}

template <typename BaseT>
template <
typename A, typename B, typename C, typename D, typename E,
typename F, typename G, typename H, typename I>
inline typename actor_result<BaseT,
tuple<A&, B&, C&, D&, E&, F&, G&, H&, I&>
>::type
actor<BaseT>::operator()(
A& a_, B& b_, C& c_, D& d_, E& e_, F& f_, G& g_, H& h_, I& i_
) const
{
return BaseT::eval(
tuple<A&, B&, C&, D&, E&, F&, G&, H&, I&>
(a_, b_, c_, d_, e_, f_, g_, h_, i_)
);
}

#if PHOENIX_LIMIT > 9
template <typename BaseT>
template <
typename A, typename B, typename C, typename D, typename E,
typename F, typename G, typename H, typename I, typename J>
inline typename actor_result<BaseT,
tuple<A&, B&, C&, D&, E&, F&, G&, H&, I&, J&>
>::type
actor<BaseT>::operator()(
A& a_, B& b_, C& c_, D& d_, E& e_, F& f_, G& g_, H& h_, I& i_, J& j_
) const
{
return BaseT::eval(
tuple<A&, B&, C&, D&, E&, F&, G&, H&, I&, J&>
(a_, b_, c_, d_, e_, f_, g_, h_, i_, j_)
);
}

template <typename BaseT>
template <
typename A, typename B, typename C, typename D, typename E,
typename F, typename G, typename H, typename I, typename J,
typename K>
inline typename actor_result<BaseT,
tuple<A&, B&, C&, D&, E&, F&, G&, H&, I&, J&, K&>
>::type
actor<BaseT>::operator()(
A& a_, B& b_, C& c_, D& d_, E& e_, F& f_, G& g_, H& h_, I& i_, J& j_,
K& k_
) const
{
return BaseT::eval(
tuple<A&, B&, C&, D&, E&, F&, G&, H&, I&, J&, K&>
(a_, b_, c_, d_, e_, f_, g_, h_, i_, j_, k_)
);
}

template <typename BaseT>
template <
typename A, typename B, typename C, typename D, typename E,
typename F, typename G, typename H, typename I, typename J,
typename K, typename L>
inline typename actor_result<BaseT,
tuple<A&, B&, C&, D&, E&, F&, G&, H&, I&, J&, K&, L&>
>::type
actor<BaseT>::operator()(
A& a_, B& b_, C& c_, D& d_, E& e_, F& f_, G& g_, H& h_, I& i_, J& j_,
K& k_, L& l_
) const
{
return BaseT::eval(
tuple<A&, B&, C&, D&, E&, F&, G&, H&, I&, J&, K&, L&>
(a_, b_, c_, d_, e_, f_, g_, h_, i_, j_, k_, l_)
);
}

#if PHOENIX_LIMIT > 12
template <typename BaseT>
template <
typename A, typename B, typename C, typename D, typename E,
typename F, typename G, typename H, typename I, typename J,
typename K, typename L, typename M>
inline typename actor_result<BaseT,
tuple<A&, B&, C&, D&, E&, F&, G&, H&, I&, J&, K&, L&, M&>
>::type
actor<BaseT>::operator()(
A& a_, B& b_, C& c_, D& d_, E& e_, F& f_, G& g_, H& h_, I& i_, J& j_,
K& k_, L& l_, M& m_
) const
{
return BaseT::eval(
tuple<A&, B&, C&, D&, E&, F&, G&, H&, I&, J&, K&, L&, M&>
(a_, b_, c_, d_, e_, f_, g_, h_, i_, j_, k_, l_, m_)
);
}

template <typename BaseT>
template <
typename A, typename B, typename C, typename D, typename E,
typename F, typename G, typename H, typename I, typename J,
typename K, typename L, typename M, typename N>
inline typename actor_result<BaseT,
tuple<A&, B&, C&, D&, E&, F&, G&, H&, I&, J&, K&, L&, M&, N&>
>::type
actor<BaseT>::operator()(
A& a_, B& b_, C& c_, D& d_, E& e_, F& f_, G& g_, H& h_, I& i_, J& j_,
K& k_, L& l_, M& m_, N& n_
) const
{
return BaseT::eval(
tuple<A&, B&, C&, D&, E&, F&, G&, H&, I&, J&, K&, L&, M&, N&>
(a_, b_, c_, d_, e_, f_, g_, h_, i_, j_, k_, l_, m_, n_)
);
}

template <typename BaseT>
template <
typename A, typename B, typename C, typename D, typename E,
typename F, typename G, typename H, typename I, typename J,
typename K, typename L, typename M, typename N, typename O>
inline typename actor_result<BaseT,
tuple<A&, B&, C&, D&, E&, F&, G&, H&, I&, J&, K&, L&, M&, N&, O&>
>::type
actor<BaseT>::operator()(
A& a_, B& b_, C& c_, D& d_, E& e_, F& f_, G& g_, H& h_, I& i_, J& j_,
K& k_, L& l_, M& m_, N& n_, O& o_
) const
{
return BaseT::eval(
tuple<A&, B&, C&, D&, E&, F&, G&, H&, I&, J&, K&, L&, M&, N&, O&>
(a_, b_, c_, d_, e_, f_, g_, h_, i_, j_, k_, l_, m_, n_, o_)
);
}

#endif
#endif
#endif
#endif

template <typename BaseT>
template <typename TupleT>
typename actor_result<BaseT, unpack_tuple<TupleT> >::type
actor<BaseT>::operator()(unpack_tuple<TupleT> const &t) const
{
return BaseT::eval(t);
}

}   

#endif

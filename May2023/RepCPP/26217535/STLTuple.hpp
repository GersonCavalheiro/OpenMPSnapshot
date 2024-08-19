

#pragma once

#include <alpaka/core/BoostPredef.hpp>

#if BOOST_COMP_CLANG
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wdocumentation"
#   pragma clang diagnostic ignored "-Wdocumentation-unknown-command"
#endif
#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#    pragma warning(push)
#    pragma warning(disable : 4003) 
#endif

namespace utility {
namespace tuple {
template <bool, typename T = void>
struct StaticIf;
template <typename T>
struct StaticIf<true, T> {
typedef T type;
};

template <class... Ts>
struct Tuple {};

template <class T, class... Ts>
struct Tuple<T, Ts...> {
Tuple(T t, Ts... ts) : head(t), tail(ts...) {}
T head;
Tuple<Ts...> tail;
};

template <typename, typename>
struct concat_tuple {};

template <typename... Ts, typename... Us>
struct concat_tuple<Tuple<Ts...>, Tuple<Us...>> {
using type = Tuple<Ts..., Us...>;
};

template <typename T>
struct remove_last_type;

template <typename T>
struct remove_last_type<Tuple<T>> {
using type = Tuple<>;
};

template <typename T, typename... Args>
struct remove_last_type<Tuple<T, Args...>> {
using type = typename concat_tuple<
Tuple<T>, typename remove_last_type<Tuple<Args...>>::type>::type;
};

template <typename T>
struct remove_first_type {};

template <typename T, typename... Ts>
struct remove_first_type<Tuple<T, Ts...>> {
typedef Tuple<Ts...> type;
};

template <size_t, class>
struct ElemTypeHolder;

template <class T, class... Ts>
struct ElemTypeHolder<0, Tuple<T, Ts...>> {
typedef T type;
};

template <size_t k, class T, class... Ts>
struct ElemTypeHolder<k, Tuple<T, Ts...>> {
typedef typename ElemTypeHolder<k - 1, Tuple<Ts...>>::type type;
};


#define TERMINATE_CONDS_TUPLE_GET(CVQual)                                      \
template <size_t k, class... Ts>                                             \
typename StaticIf<k == 0, CVQual                                             \
typename ElemTypeHolder<0, Tuple<Ts...>>::type&>::type     \
get(CVQual Tuple<Ts...>& t) {                                                \
static_assert(sizeof...(Ts) != 0,                                          \
"The requseted value is bigger than the size of the tuple"); \
return t.head;                                                             \
}

TERMINATE_CONDS_TUPLE_GET(const)
TERMINATE_CONDS_TUPLE_GET()
#undef TERMINATE_CONDS_TUPLE_GET
#define RECURSIVE_TUPLE_GET(CVQual)                                           \
template <size_t k, class T, class... Ts>                                   \
typename StaticIf<k != 0, CVQual                                            \
typename ElemTypeHolder<k, Tuple<T, Ts...>>::type&>::type \
get(CVQual Tuple<T, Ts...>& t) {                                            \
return utility::tuple::get<k - 1>(t.tail);                                \
}
RECURSIVE_TUPLE_GET(const)
RECURSIVE_TUPLE_GET()
#undef RECURSIVE_TUPLE_GET

template <typename... Args>
Tuple<Args...> make_tuple(Args... args) {
return Tuple<Args...>(args...);
}

template <typename... Args>
static constexpr size_t size(Tuple<Args...>&) {
return sizeof...(Args);
}

template <size_t... Is>
struct IndexList {};

template <size_t MIN, size_t N, size_t... Is>
struct RangeBuilder;

template <size_t MIN, size_t... Is>
struct RangeBuilder<MIN, MIN, Is...> {
typedef IndexList<Is...> type;
};

template <size_t MIN, size_t N, size_t... Is>
struct RangeBuilder : public RangeBuilder<MIN, N - 1, N - 1, Is...> {};

template <size_t MIN, size_t MAX>
struct IndexRange : RangeBuilder<MIN, MAX>::type {};

template <typename... Args, typename T, size_t... I>
Tuple<Args..., T> append_base(Tuple<Args...> t, T a, IndexList<I...>) {
return utility::tuple::make_tuple(get<I>(t)..., a);
}

template <typename... Args, typename T>
Tuple<Args..., T> append(Tuple<Args...> t, T a) {
return utility::tuple::append_base(t, a, IndexRange<0, sizeof...(Args)>());
}

template <typename... Args1, typename... Args2, size_t... I1, size_t... I2>
Tuple<Args1..., Args2...> append_base(Tuple<Args1...> t1, Tuple<Args2...> t2,
IndexList<I1...>, IndexList<I2...>) {
return utility::tuple::make_tuple(get<I1>(t1)..., get<I2>(t2)...);
}

template <typename... Args1, typename... Args2>
Tuple<Args1..., Args2...> append(Tuple<Args1...> t1, Tuple<Args2...> t2) {
return utility::tuple::append_base(t1, t2, IndexRange<0, sizeof...(Args1)>(),
IndexRange<0, sizeof...(Args2)>());
}

}  
}  

#if BOOST_COMP_CLANG
#   pragma clang diagnostic pop
#endif
#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#    pragma warning(pop)
#endif

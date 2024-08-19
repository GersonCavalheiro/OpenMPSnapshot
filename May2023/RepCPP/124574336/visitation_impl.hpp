
#ifndef BOOST_VARIANT_DETAIL_VISITATION_IMPL_HPP
#define BOOST_VARIANT_DETAIL_VISITATION_IMPL_HPP

#include <boost/config.hpp>

#include <boost/variant/detail/backup_holder.hpp>
#include <boost/variant/detail/cast_storage.hpp>
#include <boost/variant/detail/forced_return.hpp>
#include <boost/variant/variant_fwd.hpp> 

#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/next.hpp>
#include <boost/mpl/deref.hpp>
#include <boost/mpl/or.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/inc.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/has_nothrow_copy.hpp>
#include <boost/type_traits/is_nothrow_move_constructible.hpp>

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
# pragma warning (push)
# pragma warning (disable : 4702) 
#endif

#if !defined(BOOST_VARIANT_VISITATION_UNROLLING_LIMIT)

#ifndef BOOST_VARIANT_DO_NOT_USE_VARIADIC_TEMPLATES
#   include <boost/mpl/limits/list.hpp>
#   define BOOST_VARIANT_VISITATION_UNROLLING_LIMIT   \
BOOST_MPL_LIMIT_LIST_SIZE
#else
#   define BOOST_VARIANT_VISITATION_UNROLLING_LIMIT   \
BOOST_VARIANT_LIMIT_TYPES
#endif

#endif

#if defined(BOOST_NO_NULLPTR)
#define BOOST_VARIANT_NULL 0
#else
#define BOOST_VARIANT_NULL nullptr
#endif

namespace boost {
namespace detail { namespace variant {

struct apply_visitor_unrolled {};



template <typename Iter, typename LastIter>
struct visitation_impl_step
{
typedef typename mpl::deref<Iter>::type type;

typedef typename mpl::next<Iter>::type next_iter;
typedef visitation_impl_step<
next_iter, LastIter
> next;
};

template <typename LastIter>
struct visitation_impl_step< LastIter,LastIter >
{
typedef apply_visitor_unrolled type;
typedef visitation_impl_step next;
};



template <typename Visitor, typename VoidPtrCV, typename T>
inline typename Visitor::result_type
visitation_impl_invoke_impl(
int, Visitor& visitor, VoidPtrCV storage, T*
, mpl::true_
)
{
return visitor.internal_visit(
cast_storage<T>(storage), 1L
);
}

template <typename Visitor, typename VoidPtrCV, typename T>
inline typename Visitor::result_type
visitation_impl_invoke_impl(
int internal_which, Visitor& visitor, VoidPtrCV storage, T*
, mpl::false_
)
{
if (internal_which >= 0)
{
return visitor.internal_visit(
cast_storage<T>(storage), 1L
);
}
else
{
return visitor.internal_visit(
cast_storage< backup_holder<T> >(storage), 1L
);
}
}

template <typename Visitor, typename VoidPtrCV, typename T, typename NoBackupFlag>
inline typename Visitor::result_type
visitation_impl_invoke(
int internal_which, Visitor& visitor, VoidPtrCV storage, T* t
, NoBackupFlag
, int
)
{
typedef typename mpl::or_<
NoBackupFlag
, is_nothrow_move_constructible<T>
, has_nothrow_copy<T>
>::type never_uses_backup;

return (visitation_impl_invoke_impl)(
internal_which, visitor, storage, t
, never_uses_backup()
);
}

template <typename Visitor, typename VoidPtrCV, typename NBF>
inline typename Visitor::result_type
visitation_impl_invoke(int, Visitor&, VoidPtrCV, apply_visitor_unrolled*, NBF, long)
{
typedef typename Visitor::result_type result_type;
return ::boost::detail::variant::forced_return< result_type >();
}


template <
typename W, typename S
, typename Visitor, typename VPCV
, typename NBF
>
inline typename Visitor::result_type
visitation_impl(
int, int, Visitor&, VPCV
, mpl::true_ 
, NBF, W* = BOOST_VARIANT_NULL, S* = BOOST_VARIANT_NULL
)
{
typedef typename Visitor::result_type result_type;
return ::boost::detail::variant::forced_return< result_type >();
}

template <
typename Which, typename step0
, typename Visitor, typename VoidPtrCV
, typename NoBackupFlag
>
BOOST_FORCEINLINE typename Visitor::result_type
visitation_impl(
const int internal_which, const int logical_which
, Visitor& visitor, VoidPtrCV storage
, mpl::false_ 
, NoBackupFlag no_backup_flag
, Which* = BOOST_VARIANT_NULL, step0* = BOOST_VARIANT_NULL
)
{
#   define BOOST_VARIANT_AUX_APPLY_VISITOR_STEP_TYPEDEF(z, N, _) \
typedef typename BOOST_PP_CAT(step,N)::type BOOST_PP_CAT(T,N); \
typedef typename BOOST_PP_CAT(step,N)::next \
BOOST_PP_CAT(step, BOOST_PP_INC(N)); \


BOOST_PP_REPEAT(
BOOST_VARIANT_VISITATION_UNROLLING_LIMIT
, BOOST_VARIANT_AUX_APPLY_VISITOR_STEP_TYPEDEF
, _
)

#   undef BOOST_VARIANT_AUX_APPLY_VISITOR_STEP_TYPEDEF

switch (logical_which)
{

#   define BOOST_VARIANT_AUX_APPLY_VISITOR_STEP_CASE(z, N, _) \
case (Which::value + (N)): \
return (visitation_impl_invoke)( \
internal_which, visitor, storage \
, static_cast<BOOST_PP_CAT(T,N)*>(0) \
, no_backup_flag, 1L \
); \


BOOST_PP_REPEAT(
BOOST_VARIANT_VISITATION_UNROLLING_LIMIT
, BOOST_VARIANT_AUX_APPLY_VISITOR_STEP_CASE
, _
)

#   undef BOOST_VARIANT_AUX_APPLY_VISITOR_STEP_CASE

default: break;
}

typedef mpl::int_<
Which::value + (BOOST_VARIANT_VISITATION_UNROLLING_LIMIT)
> next_which;

typedef BOOST_PP_CAT(step, BOOST_VARIANT_VISITATION_UNROLLING_LIMIT)
next_step;

typedef typename next_step::type next_type;
typedef typename is_same< next_type,apply_visitor_unrolled >::type
is_apply_visitor_unrolled;

return detail::variant::visitation_impl(
internal_which, logical_which
, visitor, storage
, is_apply_visitor_unrolled()
, no_backup_flag
, static_cast<next_which*>(0), static_cast<next_step*>(0)
);
}

}} 
} 

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
# pragma warning(pop)
#endif

#endif 


#ifndef BOOST_CONTAINER_ALLOCATOR_SCOPED_ALLOCATOR_HPP
#define BOOST_CONTAINER_ALLOCATOR_SCOPED_ALLOCATOR_HPP

#if defined (_MSC_VER)
#  pragma once 
#endif

#include <boost/container/detail/config_begin.hpp>
#include <boost/container/detail/workaround.hpp>

#include <boost/container/allocator_traits.hpp>
#include <boost/container/scoped_allocator_fwd.hpp>
#include <boost/container/detail/dispatch_uses_allocator.hpp>

#include <boost/container/detail/mpl.hpp>
#include <boost/container/detail/pair.hpp>
#include <boost/container/detail/type_traits.hpp>

#include <boost/move/adl_move_swap.hpp>
#if defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES)
#include <boost/move/detail/fwd_macros.hpp>
#endif
#include <boost/move/utility_core.hpp>

#include <boost/core/no_exceptions_support.hpp>

namespace boost { namespace container {

#ifndef BOOST_CONTAINER_DOXYGEN_INVOKED

namespace dtl {

template <typename Allocator>
struct is_scoped_allocator_imp
{
typedef char yes_type;
struct no_type{ char dummy[2]; };

template <typename T>
static yes_type test(typename T::outer_allocator_type*);

template <typename T>
static int test(...);

static const bool value = (sizeof(yes_type) == sizeof(test<Allocator>(0)));
};

template<class MaybeScopedAlloc, bool = is_scoped_allocator_imp<MaybeScopedAlloc>::value >
struct outermost_allocator_type_impl
{
typedef typename MaybeScopedAlloc::outer_allocator_type outer_type;
typedef typename outermost_allocator_type_impl<outer_type>::type type;
};

template<class MaybeScopedAlloc>
struct outermost_allocator_type_impl<MaybeScopedAlloc, false>
{
typedef MaybeScopedAlloc type;
};

template<class MaybeScopedAlloc, bool = is_scoped_allocator_imp<MaybeScopedAlloc>::value >
struct outermost_allocator_imp
{
typedef MaybeScopedAlloc type;

BOOST_CONTAINER_FORCEINLINE static type &get(MaybeScopedAlloc &a)
{  return a;  }

BOOST_CONTAINER_FORCEINLINE static const type &get(const MaybeScopedAlloc &a)
{  return a;  }
};

template<class MaybeScopedAlloc>
struct outermost_allocator_imp<MaybeScopedAlloc, true>
{
typedef typename MaybeScopedAlloc::outer_allocator_type outer_type;
typedef typename outermost_allocator_type_impl<outer_type>::type type;

BOOST_CONTAINER_FORCEINLINE static type &get(MaybeScopedAlloc &a)
{  return outermost_allocator_imp<outer_type>::get(a.outer_allocator());  }

BOOST_CONTAINER_FORCEINLINE static const type &get(const MaybeScopedAlloc &a)
{  return outermost_allocator_imp<outer_type>::get(a.outer_allocator());  }
};

}  

template <typename Allocator>
struct is_scoped_allocator
: dtl::is_scoped_allocator_imp<Allocator>
{};

template <typename Allocator>
struct outermost_allocator
: dtl::outermost_allocator_imp<Allocator>
{};

template <typename Allocator>
BOOST_CONTAINER_FORCEINLINE typename outermost_allocator<Allocator>::type &
get_outermost_allocator(Allocator &a)
{  return outermost_allocator<Allocator>::get(a);   }

template <typename Allocator>
BOOST_CONTAINER_FORCEINLINE const typename outermost_allocator<Allocator>::type &
get_outermost_allocator(const Allocator &a)
{  return outermost_allocator<Allocator>::get(a);   }

namespace dtl {

#if !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES)

template <typename OuterAlloc, class ...InnerAllocs>
class scoped_allocator_adaptor_base
: public OuterAlloc
{
typedef allocator_traits<OuterAlloc> outer_traits_type;
BOOST_COPYABLE_AND_MOVABLE(scoped_allocator_adaptor_base)

public:
template <class OuterA2>
struct rebind_base
{
typedef scoped_allocator_adaptor_base<OuterA2, InnerAllocs...> other;
};

typedef OuterAlloc outer_allocator_type;
typedef scoped_allocator_adaptor<InnerAllocs...>   inner_allocator_type;
typedef allocator_traits<inner_allocator_type>     inner_traits_type;
typedef scoped_allocator_adaptor
<OuterAlloc, InnerAllocs...>                    scoped_allocator_type;
typedef dtl::bool_<
outer_traits_type::propagate_on_container_copy_assignment::value ||
inner_allocator_type::propagate_on_container_copy_assignment::value
> propagate_on_container_copy_assignment;
typedef dtl::bool_<
outer_traits_type::propagate_on_container_move_assignment::value ||
inner_allocator_type::propagate_on_container_move_assignment::value
> propagate_on_container_move_assignment;
typedef dtl::bool_<
outer_traits_type::propagate_on_container_swap::value ||
inner_allocator_type::propagate_on_container_swap::value
> propagate_on_container_swap;
typedef dtl::bool_<
outer_traits_type::is_always_equal::value &&
inner_allocator_type::is_always_equal::value
> is_always_equal;

BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base()
{}

template <class OuterA2>
BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base(BOOST_FWD_REF(OuterA2) outerAlloc, const InnerAllocs &...args)
: outer_allocator_type(::boost::forward<OuterA2>(outerAlloc))
, m_inner(args...)
{}

BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base(const scoped_allocator_adaptor_base& other)
: outer_allocator_type(other.outer_allocator())
, m_inner(other.inner_allocator())
{}

BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base(BOOST_RV_REF(scoped_allocator_adaptor_base) other)
: outer_allocator_type(::boost::move(other.outer_allocator()))
, m_inner(::boost::move(other.inner_allocator()))
{}

template <class OuterA2>
BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base
(const scoped_allocator_adaptor_base<OuterA2, InnerAllocs...>& other)
: outer_allocator_type(other.outer_allocator())
, m_inner(other.inner_allocator())
{}

template <class OuterA2>
BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base
(BOOST_RV_REF_BEG scoped_allocator_adaptor_base
<OuterA2, InnerAllocs...> BOOST_RV_REF_END other)
: outer_allocator_type(other.outer_allocator())
, m_inner(other.inner_allocator())
{}

public:
struct internal_type_t{};

template <class OuterA2>
BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base
( internal_type_t
, BOOST_FWD_REF(OuterA2) outerAlloc
, const inner_allocator_type &inner)
: outer_allocator_type(::boost::forward<OuterA2>(outerAlloc))
, m_inner(inner)
{}

public:

BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base &operator=
(BOOST_COPY_ASSIGN_REF(scoped_allocator_adaptor_base) other)
{
outer_allocator_type::operator=(other.outer_allocator());
m_inner = other.inner_allocator();
return *this;
}

BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base &operator=(BOOST_RV_REF(scoped_allocator_adaptor_base) other)
{
outer_allocator_type::operator=(boost::move(other.outer_allocator()));
m_inner = ::boost::move(other.inner_allocator());
return *this;
}

BOOST_CONTAINER_FORCEINLINE void swap(scoped_allocator_adaptor_base &r)
{
boost::adl_move_swap(this->outer_allocator(), r.outer_allocator());
boost::adl_move_swap(this->m_inner, r.inner_allocator());
}

BOOST_CONTAINER_FORCEINLINE friend void swap(scoped_allocator_adaptor_base &l, scoped_allocator_adaptor_base &r)
{  l.swap(r);  }

BOOST_CONTAINER_FORCEINLINE inner_allocator_type&       inner_allocator() BOOST_NOEXCEPT_OR_NOTHROW
{ return m_inner; }

BOOST_CONTAINER_FORCEINLINE inner_allocator_type const& inner_allocator() const BOOST_NOEXCEPT_OR_NOTHROW
{ return m_inner; }

BOOST_CONTAINER_FORCEINLINE outer_allocator_type      & outer_allocator() BOOST_NOEXCEPT_OR_NOTHROW
{ return static_cast<outer_allocator_type&>(*this); }

BOOST_CONTAINER_FORCEINLINE const outer_allocator_type &outer_allocator() const BOOST_NOEXCEPT_OR_NOTHROW
{ return static_cast<const outer_allocator_type&>(*this); }

BOOST_CONTAINER_FORCEINLINE scoped_allocator_type select_on_container_copy_construction() const
{
return scoped_allocator_type
(internal_type_t()
,outer_traits_type::select_on_container_copy_construction(this->outer_allocator())
,inner_traits_type::select_on_container_copy_construction(this->inner_allocator())
);
}

private:
inner_allocator_type m_inner;
};

#else 

template <typename OuterAlloc, bool Dummy, BOOST_MOVE_CLASSDFLT9>
class scoped_allocator_adaptor_base;


#define BOOST_CONTAINER_SCOPED_ALLOCATOR_ADAPTOR_BASE_CODE(N)\
template <typename OuterAlloc BOOST_MOVE_I##N BOOST_MOVE_CLASS##N>\
class scoped_allocator_adaptor_base<OuterAlloc, true, BOOST_MOVE_TARG##N>\
: public OuterAlloc\
{\
typedef allocator_traits<OuterAlloc> outer_traits_type;\
BOOST_COPYABLE_AND_MOVABLE(scoped_allocator_adaptor_base)\
\
public:\
template <class OuterA2>\
struct rebind_base\
{\
typedef scoped_allocator_adaptor_base<OuterA2, true, BOOST_MOVE_TARG##N> other;\
};\
\
typedef OuterAlloc outer_allocator_type;\
typedef scoped_allocator_adaptor<BOOST_MOVE_TARG##N> inner_allocator_type;\
typedef scoped_allocator_adaptor<OuterAlloc, BOOST_MOVE_TARG##N> scoped_allocator_type;\
typedef allocator_traits<inner_allocator_type> inner_traits_type;\
typedef dtl::bool_<\
outer_traits_type::propagate_on_container_copy_assignment::value ||\
inner_allocator_type::propagate_on_container_copy_assignment::value\
> propagate_on_container_copy_assignment;\
typedef dtl::bool_<\
outer_traits_type::propagate_on_container_move_assignment::value ||\
inner_allocator_type::propagate_on_container_move_assignment::value\
> propagate_on_container_move_assignment;\
typedef dtl::bool_<\
outer_traits_type::propagate_on_container_swap::value ||\
inner_allocator_type::propagate_on_container_swap::value\
> propagate_on_container_swap;\
\
typedef dtl::bool_<\
outer_traits_type::is_always_equal::value &&\
inner_allocator_type::is_always_equal::value\
> is_always_equal;\
\
BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base(){}\
\
template <class OuterA2>\
BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base(BOOST_FWD_REF(OuterA2) outerAlloc, BOOST_MOVE_CREF##N)\
: outer_allocator_type(::boost::forward<OuterA2>(outerAlloc))\
, m_inner(BOOST_MOVE_ARG##N)\
{}\
\
BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base(const scoped_allocator_adaptor_base& other)\
: outer_allocator_type(other.outer_allocator())\
, m_inner(other.inner_allocator())\
{}\
\
BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base(BOOST_RV_REF(scoped_allocator_adaptor_base) other)\
: outer_allocator_type(::boost::move(other.outer_allocator()))\
, m_inner(::boost::move(other.inner_allocator()))\
{}\
\
template <class OuterA2>\
BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base\
(const scoped_allocator_adaptor_base<OuterA2, true, BOOST_MOVE_TARG##N>& other)\
: outer_allocator_type(other.outer_allocator())\
, m_inner(other.inner_allocator())\
{}\
\
template <class OuterA2>\
BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base\
(BOOST_RV_REF_BEG scoped_allocator_adaptor_base<OuterA2, true, BOOST_MOVE_TARG##N> BOOST_RV_REF_END other)\
: outer_allocator_type(other.outer_allocator())\
, m_inner(other.inner_allocator())\
{}\
\
public:\
struct internal_type_t{};\
\
template <class OuterA2>\
BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base\
( internal_type_t, BOOST_FWD_REF(OuterA2) outerAlloc, const inner_allocator_type &inner)\
: outer_allocator_type(::boost::forward<OuterA2>(outerAlloc))\
, m_inner(inner)\
{}\
\
public:\
BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base &operator=\
(BOOST_COPY_ASSIGN_REF(scoped_allocator_adaptor_base) other)\
{\
outer_allocator_type::operator=(other.outer_allocator());\
m_inner = other.inner_allocator();\
return *this;\
}\
\
BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base &operator=(BOOST_RV_REF(scoped_allocator_adaptor_base) other)\
{\
outer_allocator_type::operator=(boost::move(other.outer_allocator()));\
m_inner = ::boost::move(other.inner_allocator());\
return *this;\
}\
\
BOOST_CONTAINER_FORCEINLINE void swap(scoped_allocator_adaptor_base &r)\
{\
boost::adl_move_swap(this->outer_allocator(), r.outer_allocator());\
boost::adl_move_swap(this->m_inner, r.inner_allocator());\
}\
\
BOOST_CONTAINER_FORCEINLINE friend void swap(scoped_allocator_adaptor_base &l, scoped_allocator_adaptor_base &r)\
{  l.swap(r);  }\
\
BOOST_CONTAINER_FORCEINLINE inner_allocator_type&       inner_allocator()\
{ return m_inner; }\
\
BOOST_CONTAINER_FORCEINLINE inner_allocator_type const& inner_allocator() const\
{ return m_inner; }\
\
BOOST_CONTAINER_FORCEINLINE outer_allocator_type      & outer_allocator()\
{ return static_cast<outer_allocator_type&>(*this); }\
\
BOOST_CONTAINER_FORCEINLINE const outer_allocator_type &outer_allocator() const\
{ return static_cast<const outer_allocator_type&>(*this); }\
\
BOOST_CONTAINER_FORCEINLINE scoped_allocator_type select_on_container_copy_construction() const\
{\
return scoped_allocator_type\
(internal_type_t()\
,outer_traits_type::select_on_container_copy_construction(this->outer_allocator())\
,inner_traits_type::select_on_container_copy_construction(this->inner_allocator())\
);\
}\
private:\
inner_allocator_type m_inner;\
};\
BOOST_MOVE_ITERATE_1TO9(BOOST_CONTAINER_SCOPED_ALLOCATOR_ADAPTOR_BASE_CODE)
#undef BOOST_CONTAINER_SCOPED_ALLOCATOR_ADAPTOR_BASE_CODE

#endif   

#if defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) && !defined(BOOST_CONTAINER_DOXYGEN_INVOKED)
#define BOOST_CONTAINER_SCOPEDALLOC_DUMMYTRUE      ,true
#define BOOST_CONTAINER_SCOPEDALLOC_ALLINNER       BOOST_MOVE_TARG9
#define BOOST_CONTAINER_SCOPEDALLOC_ALLINNERCLASS  BOOST_MOVE_CLASS9
#else
#define BOOST_CONTAINER_SCOPEDALLOC_DUMMYTRUE
#define BOOST_CONTAINER_SCOPEDALLOC_ALLINNER       InnerAllocs...
#define BOOST_CONTAINER_SCOPEDALLOC_ALLINNERCLASS  typename... InnerAllocs
#endif

template <typename OuterAlloc>
class scoped_allocator_adaptor_base< OuterAlloc BOOST_CONTAINER_SCOPEDALLOC_DUMMYTRUE>
: public OuterAlloc
{
BOOST_COPYABLE_AND_MOVABLE(scoped_allocator_adaptor_base)
public:

template <class U>
struct rebind_base
{
typedef scoped_allocator_adaptor_base
<typename allocator_traits<OuterAlloc>::template portable_rebind_alloc<U>::type
BOOST_CONTAINER_SCOPEDALLOC_DUMMYTRUE > other;
};

typedef OuterAlloc                           outer_allocator_type;
typedef allocator_traits<OuterAlloc>         outer_traits_type;
typedef scoped_allocator_adaptor<OuterAlloc> inner_allocator_type;
typedef inner_allocator_type                 scoped_allocator_type;
typedef allocator_traits<inner_allocator_type>   inner_traits_type;
typedef typename outer_traits_type::
propagate_on_container_copy_assignment    propagate_on_container_copy_assignment;
typedef typename outer_traits_type::
propagate_on_container_move_assignment    propagate_on_container_move_assignment;
typedef typename outer_traits_type::
propagate_on_container_swap               propagate_on_container_swap;
typedef typename outer_traits_type::
is_always_equal                           is_always_equal;

BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base()
{}

template <class OuterA2>
BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base(BOOST_FWD_REF(OuterA2) outerAlloc)
: outer_allocator_type(::boost::forward<OuterA2>(outerAlloc))
{}

BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base(const scoped_allocator_adaptor_base& other)
: outer_allocator_type(other.outer_allocator())
{}

BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base(BOOST_RV_REF(scoped_allocator_adaptor_base) other)
: outer_allocator_type(::boost::move(other.outer_allocator()))
{}

template <class OuterA2>
BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base
(const scoped_allocator_adaptor_base<OuterA2 BOOST_CONTAINER_SCOPEDALLOC_DUMMYTRUE>& other)
: outer_allocator_type(other.outer_allocator())
{}

template <class OuterA2>
BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base
(BOOST_RV_REF_BEG scoped_allocator_adaptor_base<OuterA2 BOOST_CONTAINER_SCOPEDALLOC_DUMMYTRUE> BOOST_RV_REF_END other)
: outer_allocator_type(other.outer_allocator())
{}

public:
struct internal_type_t{};

template <class OuterA2>
BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base(internal_type_t, BOOST_FWD_REF(OuterA2) outerAlloc, const inner_allocator_type &)
: outer_allocator_type(::boost::forward<OuterA2>(outerAlloc))
{}

public:
BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base &operator=(BOOST_COPY_ASSIGN_REF(scoped_allocator_adaptor_base) other)
{
outer_allocator_type::operator=(other.outer_allocator());
return *this;
}

BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor_base &operator=(BOOST_RV_REF(scoped_allocator_adaptor_base) other)
{
outer_allocator_type::operator=(boost::move(other.outer_allocator()));
return *this;
}

BOOST_CONTAINER_FORCEINLINE void swap(scoped_allocator_adaptor_base &r)
{
boost::adl_move_swap(this->outer_allocator(), r.outer_allocator());
}

BOOST_CONTAINER_FORCEINLINE friend void swap(scoped_allocator_adaptor_base &l, scoped_allocator_adaptor_base &r)
{  l.swap(r);  }

BOOST_CONTAINER_FORCEINLINE inner_allocator_type&       inner_allocator()
{ return static_cast<inner_allocator_type&>(*this); }

BOOST_CONTAINER_FORCEINLINE inner_allocator_type const& inner_allocator() const
{ return static_cast<const inner_allocator_type&>(*this); }

BOOST_CONTAINER_FORCEINLINE outer_allocator_type      & outer_allocator()
{ return static_cast<outer_allocator_type&>(*this); }

BOOST_CONTAINER_FORCEINLINE const outer_allocator_type &outer_allocator() const
{ return static_cast<const outer_allocator_type&>(*this); }

BOOST_CONTAINER_FORCEINLINE scoped_allocator_type select_on_container_copy_construction() const
{
return scoped_allocator_type
(internal_type_t()
,outer_traits_type::select_on_container_copy_construction(this->outer_allocator())
, this->inner_allocator()
);
}
};

}  

#endif   

#if !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) || defined(BOOST_CONTAINER_DOXYGEN_INVOKED)

#if !defined(BOOST_CONTAINER_UNIMPLEMENTED_PACK_EXPANSION_TO_FIXED_LIST)

template <typename OuterAlloc, typename ...InnerAllocs>
class scoped_allocator_adaptor

#else 

template <typename OuterAlloc, typename ...InnerAllocs>
class scoped_allocator_adaptor<OuterAlloc, InnerAllocs...>

#endif   

#else 

template <typename OuterAlloc, BOOST_MOVE_CLASS9>
class scoped_allocator_adaptor
#endif

: public dtl::scoped_allocator_adaptor_base
<OuterAlloc BOOST_CONTAINER_SCOPEDALLOC_DUMMYTRUE, BOOST_CONTAINER_SCOPEDALLOC_ALLINNER>
{
BOOST_COPYABLE_AND_MOVABLE(scoped_allocator_adaptor)

public:
#ifndef BOOST_CONTAINER_DOXYGEN_INVOKED
typedef dtl::scoped_allocator_adaptor_base
<OuterAlloc BOOST_CONTAINER_SCOPEDALLOC_DUMMYTRUE, BOOST_CONTAINER_SCOPEDALLOC_ALLINNER> base_type;
typedef typename base_type::internal_type_t              internal_type_t;
#endif   
typedef OuterAlloc                                       outer_allocator_type;
typedef allocator_traits<OuterAlloc>                     outer_traits_type;
typedef typename base_type::inner_allocator_type         inner_allocator_type;
typedef allocator_traits<inner_allocator_type>           inner_traits_type;
typedef typename outer_traits_type::value_type           value_type;
typedef typename outer_traits_type::size_type            size_type;
typedef typename outer_traits_type::difference_type      difference_type;
typedef typename outer_traits_type::pointer              pointer;
typedef typename outer_traits_type::const_pointer        const_pointer;
typedef typename outer_traits_type::void_pointer         void_pointer;
typedef typename outer_traits_type::const_void_pointer   const_void_pointer;
typedef typename base_type::
propagate_on_container_copy_assignment                propagate_on_container_copy_assignment;
typedef typename base_type::
propagate_on_container_move_assignment                propagate_on_container_move_assignment;

typedef typename base_type::
propagate_on_container_swap                           propagate_on_container_swap;

typedef typename base_type::
is_always_equal                           is_always_equal;

template <class U>
struct rebind
{
typedef scoped_allocator_adaptor
< typename outer_traits_type::template portable_rebind_alloc<U>::type
, BOOST_CONTAINER_SCOPEDALLOC_ALLINNER> other;
};

BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor()
{}

BOOST_CONTAINER_FORCEINLINE ~scoped_allocator_adaptor()
{}

BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor(const scoped_allocator_adaptor& other)
: base_type(other.base())
{}

BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor(BOOST_RV_REF(scoped_allocator_adaptor) other)
: base_type(::boost::move(other.base()))
{}

#if !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) || defined(BOOST_CONTAINER_DOXYGEN_INVOKED)

template <class OuterA2>
BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor(BOOST_FWD_REF(OuterA2) outerAlloc, const InnerAllocs & ...innerAllocs)
: base_type(::boost::forward<OuterA2>(outerAlloc), innerAllocs...)
{}
#else 

#define BOOST_CONTAINER_SCOPED_ALLOCATOR_ADAPTOR_RELATED_ALLOCATOR_CONSTRUCTOR_CODE(N)\
template <class OuterA2>\
BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor(BOOST_FWD_REF(OuterA2) outerAlloc BOOST_MOVE_I##N BOOST_MOVE_CREF##N)\
: base_type(::boost::forward<OuterA2>(outerAlloc) BOOST_MOVE_I##N BOOST_MOVE_ARG##N)\
{}\
BOOST_MOVE_ITERATE_0TO9(BOOST_CONTAINER_SCOPED_ALLOCATOR_ADAPTOR_RELATED_ALLOCATOR_CONSTRUCTOR_CODE)
#undef BOOST_CONTAINER_SCOPED_ALLOCATOR_ADAPTOR_RELATED_ALLOCATOR_CONSTRUCTOR_CODE

#endif   

template <class OuterA2>
BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor(const scoped_allocator_adaptor<OuterA2, BOOST_CONTAINER_SCOPEDALLOC_ALLINNER> &other)
: base_type(other.base())
{}

template <class OuterA2>
BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor(BOOST_RV_REF_BEG scoped_allocator_adaptor
<OuterA2, BOOST_CONTAINER_SCOPEDALLOC_ALLINNER> BOOST_RV_REF_END other)
: base_type(::boost::move(other.base()))
{}

BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor &operator=(BOOST_COPY_ASSIGN_REF(scoped_allocator_adaptor) other)
{  return static_cast<scoped_allocator_adaptor&>(base_type::operator=(static_cast<const base_type &>(other))); }

BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor &operator=(BOOST_RV_REF(scoped_allocator_adaptor) other)
{  return static_cast<scoped_allocator_adaptor&>(base_type::operator=(boost::move(other.base()))); }

#ifdef BOOST_CONTAINER_DOXYGEN_INVOKED
void swap(scoped_allocator_adaptor &r);

friend void swap(scoped_allocator_adaptor &l, scoped_allocator_adaptor &r);

outer_allocator_type      & outer_allocator() BOOST_NOEXCEPT_OR_NOTHROW;

const outer_allocator_type &outer_allocator() const BOOST_NOEXCEPT_OR_NOTHROW;

inner_allocator_type&       inner_allocator() BOOST_NOEXCEPT_OR_NOTHROW;

inner_allocator_type const& inner_allocator() const BOOST_NOEXCEPT_OR_NOTHROW;

#endif   

BOOST_CONTAINER_FORCEINLINE size_type max_size() const BOOST_NOEXCEPT_OR_NOTHROW
{  return outer_traits_type::max_size(this->outer_allocator());   }

template <class T>
BOOST_CONTAINER_FORCEINLINE void destroy(T* p) BOOST_NOEXCEPT_OR_NOTHROW
{
allocator_traits<typename outermost_allocator<OuterAlloc>::type>
::destroy(get_outermost_allocator(this->outer_allocator()), p);
}

BOOST_CONTAINER_FORCEINLINE pointer allocate(size_type n)
{  return outer_traits_type::allocate(this->outer_allocator(), n);   }

BOOST_CONTAINER_FORCEINLINE pointer allocate(size_type n, const_void_pointer hint)
{  return outer_traits_type::allocate(this->outer_allocator(), n, hint);   }

BOOST_CONTAINER_FORCEINLINE void deallocate(pointer p, size_type n)
{  outer_traits_type::deallocate(this->outer_allocator(), p, n);  }

#ifdef BOOST_CONTAINER_DOXYGEN_INVOKED
scoped_allocator_adaptor select_on_container_copy_construction() const;
#endif   

#ifndef BOOST_CONTAINER_DOXYGEN_INVOKED
BOOST_CONTAINER_FORCEINLINE base_type &base()             { return *this; }

BOOST_CONTAINER_FORCEINLINE const base_type &base() const { return *this; }
#endif   

#if !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) || defined(BOOST_CONTAINER_DOXYGEN_INVOKED)

template < typename T, class ...Args>
BOOST_CONTAINER_FORCEINLINE void construct(T* p, BOOST_FWD_REF(Args)...args)
{
dtl::dispatch_uses_allocator
( (get_outermost_allocator)(this->outer_allocator())
, this->inner_allocator(), p, ::boost::forward<Args>(args)...);
}

#else 

#define BOOST_CONTAINER_SCOPED_ALLOCATOR_CONSTRUCT_CODE(N) \
template < typename T BOOST_MOVE_I##N BOOST_MOVE_CLASSQ##N >\
BOOST_CONTAINER_FORCEINLINE void construct(T* p BOOST_MOVE_I##N BOOST_MOVE_UREFQ##N)\
{\
dtl::dispatch_uses_allocator\
( (get_outermost_allocator)(this->outer_allocator())\
, this->inner_allocator(), p BOOST_MOVE_I##N BOOST_MOVE_FWDQ##N);\
}\
BOOST_MOVE_ITERATE_0TO9(BOOST_CONTAINER_SCOPED_ALLOCATOR_CONSTRUCT_CODE)
#undef BOOST_CONTAINER_SCOPED_ALLOCATOR_CONSTRUCT_CODE

#endif   

#ifndef BOOST_CONTAINER_DOXYGEN_INVOKED

public:
template <class OuterA2>
BOOST_CONTAINER_FORCEINLINE scoped_allocator_adaptor(internal_type_t, BOOST_FWD_REF(OuterA2) outer, const inner_allocator_type& inner)
: base_type(internal_type_t(), ::boost::forward<OuterA2>(outer), inner)
{}

#endif   
};


template<bool ZeroInner>
struct scoped_allocator_operator_equal
{
template<class IA>
BOOST_CONTAINER_FORCEINLINE static bool equal_outer(const IA &l, const IA &r)
{  return allocator_traits<IA>::equal(l, r);  }

template<class IA1, class IA2>
BOOST_CONTAINER_FORCEINLINE static bool equal_outer(const IA1 &l, const IA2 &r)
{  return l == r;  }

template<class IA>
BOOST_CONTAINER_FORCEINLINE static bool equal_inner(const IA &l, const IA &r)
{  return allocator_traits<IA>::equal(l, r);  }
};

template<>
struct scoped_allocator_operator_equal<true>
: scoped_allocator_operator_equal<false>
{
template<class IA1, class IA2>
BOOST_CONTAINER_FORCEINLINE static bool equal_inner(const IA1 &, const IA2 &)
{  return true;  }
};


template <typename OuterA1, typename OuterA2, BOOST_CONTAINER_SCOPEDALLOC_ALLINNERCLASS>
BOOST_CONTAINER_FORCEINLINE bool operator==(const scoped_allocator_adaptor<OuterA1, BOOST_CONTAINER_SCOPEDALLOC_ALLINNER>& a
,const scoped_allocator_adaptor<OuterA2, BOOST_CONTAINER_SCOPEDALLOC_ALLINNER>& b)
{
#if !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) || defined(BOOST_CONTAINER_DOXYGEN_INVOKED)
const bool has_zero_inner = sizeof...(InnerAllocs) == 0u;
#else
const bool has_zero_inner = boost::container::dtl::is_same<P0, void>::value;
#endif
typedef scoped_allocator_operator_equal<has_zero_inner> equal_t;
return equal_t::equal_outer(a.outer_allocator(), b.outer_allocator()) &&
equal_t::equal_inner(a.inner_allocator(), b.inner_allocator());
}

template <typename OuterA1, typename OuterA2, BOOST_CONTAINER_SCOPEDALLOC_ALLINNERCLASS>
BOOST_CONTAINER_FORCEINLINE bool operator!=(const scoped_allocator_adaptor<OuterA1, BOOST_CONTAINER_SCOPEDALLOC_ALLINNER>& a
,const scoped_allocator_adaptor<OuterA2, BOOST_CONTAINER_SCOPEDALLOC_ALLINNER>& b)
{  return !(a == b);   }

}} 

#include <boost/container/detail/config_end.hpp>

#endif 

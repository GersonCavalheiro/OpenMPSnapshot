#ifndef BOOST_CONCEPT_CHECK_MSVC_DWA2006429_HPP
# define BOOST_CONCEPT_CHECK_MSVC_DWA2006429_HPP

# include <boost/preprocessor/cat.hpp>
# include <boost/concept/detail/backward_compatibility.hpp>
# include <boost/config.hpp>

# ifdef BOOST_OLD_CONCEPT_SUPPORT
#  include <boost/concept/detail/has_constraints.hpp>
#  include <boost/type_traits/conditional.hpp>
# endif

# ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable:4100)
# endif

namespace boost { namespace concepts {


template <class Model>
struct check
{
virtual void failed(Model* x)
{
x->~Model();
}
};

# ifndef BOOST_NO_PARTIAL_SPECIALIZATION
struct failed {};
template <class Model>
struct check<failed ************ Model::************>
{
virtual void failed(Model* x)
{
x->~Model();
}
};
# endif

# ifdef BOOST_OLD_CONCEPT_SUPPORT

namespace detail
{
struct constraint {};
}

template <class Model>
struct require
: boost::conditional<
not_satisfied<Model>::value
, detail::constraint
# ifndef BOOST_NO_PARTIAL_SPECIALIZATION
, check<Model>
# else
, check<failed ************ Model::************>
# endif 
>::type
{};

# else

template <class Model>
struct require
# ifndef BOOST_NO_PARTIAL_SPECIALIZATION
: check<Model>
# else
: check<failed ************ Model::************>
# endif 
{};

# endif

# if BOOST_WORKAROUND(BOOST_MSVC, == 1310)

template <class Model>
struct require<void(*)(Model)>
{
virtual void failed(Model*)
{
require<Model>();
}
};

# define BOOST_CONCEPT_ASSERT_FN( ModelFnPtr )      \
enum                                                \
{                                                   \
BOOST_PP_CAT(boost_concept_check,__LINE__) =    \
sizeof(::boost::concepts::require<ModelFnPtr>)    \
}

# else 

template <class Model>
require<Model>
require_(void(*)(Model));

# define BOOST_CONCEPT_ASSERT_FN( ModelFnPtr )          \
enum                                                    \
{                                                       \
BOOST_PP_CAT(boost_concept_check,__LINE__) =        \
sizeof(::boost::concepts::require_((ModelFnPtr)0)) \
}

# endif
}}

# ifdef BOOST_MSVC
#  pragma warning(pop)
# endif

#endif 


#ifndef BOOST_MPL_BITAND_HPP_INCLUDED
#define BOOST_MPL_BITAND_HPP_INCLUDED



#if defined(_MSC_VER) && !defined(__clang__)
#ifndef __GCCXML__
#if defined(bitand)
#   pragma push_macro("bitand")
#   undef bitand
#   define bitand(x)
#endif
#endif
#endif

#define AUX778076_OP_NAME   bitand_
#define AUX778076_OP_PREFIX bitand
#define AUX778076_OP_TOKEN  &
#include <boost/mpl/aux_/arithmetic_op.hpp>

#if defined(_MSC_VER) && !defined(__clang__)
#ifndef __GCCXML__
#if defined(bitand)
#   pragma pop_macro("bitand")
#endif
#endif
#endif

#endif 

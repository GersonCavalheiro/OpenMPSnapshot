


#ifndef BOOST_ATOMIC_DETAIL_OPS_GCC_AARCH32_COMMON_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_OPS_GCC_AARCH32_COMMON_HPP_INCLUDED_

#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/capabilities.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#define BOOST_ATOMIC_DETAIL_AARCH32_MO_SWITCH(mo)\
switch (mo)\
{\
case memory_order_relaxed:\
BOOST_ATOMIC_DETAIL_AARCH32_MO_INSN("r", "r")\
break;\
\
case memory_order_consume:\
case memory_order_acquire:\
BOOST_ATOMIC_DETAIL_AARCH32_MO_INSN("a", "r")\
break;\
\
case memory_order_release:\
BOOST_ATOMIC_DETAIL_AARCH32_MO_INSN("r", "l")\
break;\
\
default:\
BOOST_ATOMIC_DETAIL_AARCH32_MO_INSN("a", "l")\
break;\
}

#if defined(BOOST_ATOMIC_DETAIL_AARCH32_LITTLE_ENDIAN)
#define BOOST_ATOMIC_DETAIL_AARCH32_ASM_ARG_LO(arg) "%" BOOST_STRINGIZE(arg)
#define BOOST_ATOMIC_DETAIL_AARCH32_ASM_ARG_HI(arg) "%H" BOOST_STRINGIZE(arg)
#else
#define BOOST_ATOMIC_DETAIL_AARCH32_ASM_ARG_LO(arg) "%H" BOOST_STRINGIZE(arg)
#define BOOST_ATOMIC_DETAIL_AARCH32_ASM_ARG_HI(arg) "%" BOOST_STRINGIZE(arg)
#endif

#endif 

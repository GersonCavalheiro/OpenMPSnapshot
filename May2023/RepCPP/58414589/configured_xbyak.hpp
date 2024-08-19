

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_CONFIGURED_XBYAK_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_CONFIGURED_XBYAK_HPP

#ifdef XBYAK_XBYAK_H_
#error "Don't #include xbyak.h directly! #include this file instead."
#endif





#define XBYAK64
#define XBYAK_NO_OP_NAMES


#define XBYAK_USE_MMAP_ALLOCATOR

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)

#pragma warning(disable : 4267)
#endif

#include <common/compiler_workarounds.hpp>

#if defined(__GNUG__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

#if defined(__GNUG__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#endif

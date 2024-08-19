#ifndef IGL_EMBREE_EMBREE_CONVENIENCE_H
#define IGL_EMBREE_EMBREE_CONVENIENCE_H

#undef interface
#undef near
#undef far
#ifdef __GNUC__
#  if __GNUC__ >= 4
#    if __GNUC_MINOR__ >= 6
#      pragma GCC diagnostic push
#      pragma GCC diagnostic ignored "-Weffc++"
#    endif
#  endif
#  pragma GCC system_header
#endif
#include <embree/include/embree.h>
#include <embree/include/intersector1.h>
#include <embree/common/ray.h>
#ifdef __GNUC__
#  if __GNUC__ >= 4
#    if __GNUC_MINOR__ >= 6
#      pragma GCC diagnostic pop
#    endif
#  endif
#endif

#endif

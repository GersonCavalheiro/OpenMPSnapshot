#ifndef TL_OBJECT_HPP
#define TL_OBJECT_HPP
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif
#include "tl-common.hpp"
#include <iostream>
#include <string>
#include <typeinfo>
#include "cxx-tltype.h"
#if !defined(HAVE_CXX11)
#include <tr1/memory>
namespace std
{
using std::tr1::shared_ptr;
using std::tr1::static_pointer_cast;
}
#endif
namespace TL
{
class LIBTL_CLASS Object
{
public:
Object() { }
virtual ~Object() { }
};
class LIBTL_CLASS Undefined : public Object
{
public :
virtual ~Undefined() { }
};
}
#endif 

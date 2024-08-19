
#ifndef BOOST_INTRUSIVE_DETAIL_FUNCTION_DETECTOR_HPP
#define BOOST_INTRUSIVE_DETAIL_FUNCTION_DETECTOR_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace intrusive {
namespace function_detector {

typedef char NotFoundType;
struct StaticFunctionType { NotFoundType x [2]; };
struct NonStaticFunctionType { NotFoundType x [3]; };

enum
{ NotFound          = 0,
StaticFunction    = sizeof( StaticFunctionType )    - sizeof( NotFoundType ),
NonStaticFunction = sizeof( NonStaticFunctionType ) - sizeof( NotFoundType )
};

}  
}  
}  

#define BOOST_INTRUSIVE_CREATE_FUNCTION_DETECTOR(Identifier, InstantiationKey) \
namespace boost { \
namespace intrusive { \
namespace function_detector { \
template < class T, \
class NonStaticType, \
class NonStaticConstType, \
class StaticType > \
class DetectMember_##InstantiationKey_##Identifier { \
template < NonStaticType > \
struct TestNonStaticNonConst ; \
\
template < NonStaticConstType > \
struct TestNonStaticConst ; \
\
template < StaticType > \
struct TestStatic ; \
\
template <class U > \
static NonStaticFunctionType Test( TestNonStaticNonConst<&U::Identifier>*, int ); \
\
template <class U > \
static NonStaticFunctionType Test( TestNonStaticConst<&U::Identifier>*, int ); \
\
template <class U> \
static StaticFunctionType Test( TestStatic<&U::Identifier>*, int ); \
\
template <class U> \
static NotFoundType Test( ... ); \
public : \
static const int check = NotFound + (sizeof(Test<T>(0, 0)) - sizeof(NotFoundType));\
};\
}}} 

#define BOOST_INTRUSIVE_DETECT_FUNCTION(Class, InstantiationKey, ReturnType, Identifier, Params) \
::boost::intrusive::function_detector::DetectMember_##InstantiationKey_##Identifier< Class,\
ReturnType (Class::*)Params,\
ReturnType (Class::*)Params const,\
ReturnType (*)Params \
>::check

#endif   

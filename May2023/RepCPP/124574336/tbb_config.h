

#ifndef __TBB_tbb_config_H
#define __TBB_tbb_config_H




#define __TBB_TODO 0



#if !defined(__TBB_SYMBOL) && !__TBB_CONFIG_PREPROC_ONLY
#include <cstddef>
#endif

#define __TBB_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

#if defined(TBB_USE_GLIBCXX_VERSION) && !defined(_GLIBCXX_RELEASE)
#define __TBB_GLIBCXX_VERSION TBB_USE_GLIBCXX_VERSION
#elif _GLIBCXX_RELEASE && _GLIBCXX_RELEASE != __GNUC__
#define __TBB_GLIBCXX_VERSION (_GLIBCXX_RELEASE*10000)
#elif __GLIBCPP__ || __GLIBCXX__
#define __TBB_GLIBCXX_VERSION __TBB_GCC_VERSION
#endif

#if __clang__
#define __TBB_CLANG_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#endif


#if __ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__
#define __TBB_IOS 1
#endif

#if __APPLE__
#if __INTEL_COMPILER && __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ > 1099 \
&& __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ < 101000
#define __TBB_MACOS_TARGET_VERSION  (100000 + 10*(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ - 1000))
#else
#define __TBB_MACOS_TARGET_VERSION  __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__
#endif
#endif



#if _WIN32||_WIN64
#   if defined(_M_X64)||defined(__x86_64__)  
#       define __TBB_x86_64 1
#   elif defined(_M_IA64)
#       define __TBB_ipf 1
#   elif defined(_M_IX86)||defined(__i386__) 
#       define __TBB_x86_32 1
#   else
#       define __TBB_generic_arch 1
#   endif
#else 
#   if !__linux__ && !__APPLE__
#       define __TBB_generic_os 1
#   endif
#   if __TBB_IOS
#       define __TBB_generic_arch 1
#   elif __x86_64__
#       define __TBB_x86_64 1
#   elif __ia64__
#       define __TBB_ipf 1
#   elif __i386__||__i386  
#       define __TBB_x86_32 1
#   else
#       define __TBB_generic_arch 1
#   endif
#endif

#if __MIC__ || __MIC2__
#define __TBB_DEFINE_MIC 1
#endif

#define __TBB_TSX_AVAILABLE  ((__TBB_x86_32 || __TBB_x86_64) && !__TBB_DEFINE_MIC)



#if __INTEL_COMPILER == 9999 && __INTEL_COMPILER_BUILD_DATE == 20110811

#undef __INTEL_COMPILER
#define __INTEL_COMPILER 1210
#endif

#if __clang__ && !__INTEL_COMPILER
#define __TBB_USE_OPTIONAL_RTTI __has_feature(cxx_rtti)
#elif defined(_CPPRTTI)
#define __TBB_USE_OPTIONAL_RTTI 1
#else
#define __TBB_USE_OPTIONAL_RTTI (__GXX_RTTI || __RTTI || __INTEL_RTTI__)
#endif

#if __TBB_GCC_VERSION >= 40400 && !defined(__INTEL_COMPILER)

#define __TBB_GCC_WARNING_SUPPRESSION_PRESENT 1
#endif




#define __TBB_CPP11_PRESENT (__cplusplus >= 201103L || _MSC_VER >= 1900)

#define __TBB_CPP17_FALLTHROUGH_PRESENT (__cplusplus >= 201703L)
#define __TBB_FALLTHROUGH_PRESENT       (__TBB_GCC_VERSION >= 70000 && !__INTEL_COMPILER)


#if __INTEL_COMPILER &&  !__INTEL_CXX11_MODE__
#define __INTEL_CXX11_MODE__ (__GXX_EXPERIMENTAL_CXX0X__ || (_MSC_VER && __STDC_HOSTED__))
#endif

#if __INTEL_COMPILER && (!_MSC_VER || __INTEL_CXX11_MODE__)

#define __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT          (__INTEL_CXX11_MODE__ && __VARIADIC_TEMPLATES)
#define __TBB_CPP11_RVALUE_REF_PRESENT                  ((_MSC_VER >= 1700 || __GXX_EXPERIMENTAL_CXX0X__ && (__TBB_GLIBCXX_VERSION >= 40500 || _LIBCPP_VERSION)) && __INTEL_COMPILER >= 1400)
#define __TBB_IMPLICIT_MOVE_PRESENT                     (__INTEL_CXX11_MODE__ && __INTEL_COMPILER >= 1400 && (_MSC_VER >= 1900 || __TBB_GCC_VERSION >= 40600 || __clang__))
#if  _MSC_VER >= 1600
#define __TBB_EXCEPTION_PTR_PRESENT                 ( __INTEL_COMPILER > 1300                                                \
\
|| (__INTEL_COMPILER == 1300 && __INTEL_COMPILER_BUILD_DATE >= 20120530) \
|| (__INTEL_COMPILER == 1210 && __INTEL_COMPILER_BUILD_DATE >= 20120410) )

#elif __TBB_GLIBCXX_VERSION >= 40404 && __TBB_GLIBCXX_VERSION < 40600
#define __TBB_EXCEPTION_PTR_PRESENT                 (__GXX_EXPERIMENTAL_CXX0X__ && __INTEL_COMPILER >= 1200)
#elif __TBB_GLIBCXX_VERSION >= 40600
#define __TBB_EXCEPTION_PTR_PRESENT                 (__GXX_EXPERIMENTAL_CXX0X__ && __INTEL_COMPILER >= 1300)
#elif _LIBCPP_VERSION
#define __TBB_EXCEPTION_PTR_PRESENT                 __GXX_EXPERIMENTAL_CXX0X__
#else
#define __TBB_EXCEPTION_PTR_PRESENT                 0
#endif
#define __TBB_STATIC_ASSERT_PRESENT                     (__INTEL_CXX11_MODE__ || _MSC_VER >= 1600)
#define __TBB_CPP11_TUPLE_PRESENT                       (_MSC_VER >= 1600 || __GXX_EXPERIMENTAL_CXX0X__ && (__TBB_GLIBCXX_VERSION >= 40300 || _LIBCPP_VERSION))
#define __TBB_INITIALIZER_LISTS_PRESENT                 (__INTEL_CXX11_MODE__ && __INTEL_COMPILER >= 1400 && (_MSC_VER >= 1800 || __TBB_GLIBCXX_VERSION >= 40400 || _LIBCPP_VERSION))
#define __TBB_CONSTEXPR_PRESENT                         (__INTEL_CXX11_MODE__ && __INTEL_COMPILER >= 1400)
#define __TBB_DEFAULTED_AND_DELETED_FUNC_PRESENT        (__INTEL_CXX11_MODE__ && __INTEL_COMPILER >= 1200)

#define __TBB_NOEXCEPT_PRESENT                          (__INTEL_CXX11_MODE__ && __INTEL_COMPILER >= 1300 && (__TBB_GLIBCXX_VERSION >= 40600 || _LIBCPP_VERSION || _MSC_VER))
#define __TBB_CPP11_STD_BEGIN_END_PRESENT               (_MSC_VER >= 1700 || __GXX_EXPERIMENTAL_CXX0X__ && __INTEL_COMPILER >= 1310 && (__TBB_GLIBCXX_VERSION >= 40600 || _LIBCPP_VERSION))
#define __TBB_CPP11_AUTO_PRESENT                        (_MSC_VER >= 1600 || __GXX_EXPERIMENTAL_CXX0X__ && __INTEL_COMPILER >= 1210)
#define __TBB_CPP11_DECLTYPE_PRESENT                    (_MSC_VER >= 1600 || __GXX_EXPERIMENTAL_CXX0X__ && __INTEL_COMPILER >= 1210)
#define __TBB_CPP11_LAMBDAS_PRESENT                     (__INTEL_CXX11_MODE__ && __INTEL_COMPILER >= 1200)
#define __TBB_CPP11_DEFAULT_FUNC_TEMPLATE_ARGS_PRESENT  (_MSC_VER >= 1800 || __GXX_EXPERIMENTAL_CXX0X__ && __INTEL_COMPILER >= 1210)
#define __TBB_OVERRIDE_PRESENT                          (__INTEL_CXX11_MODE__ && __INTEL_COMPILER >= 1400)
#define __TBB_ALIGNAS_PRESENT                           (__INTEL_CXX11_MODE__ && __INTEL_COMPILER >= 1500)
#define __TBB_CPP11_TEMPLATE_ALIASES_PRESENT            (__INTEL_CXX11_MODE__ && __INTEL_COMPILER >= 1210)
#define __TBB_CPP14_INTEGER_SEQUENCE_PRESENT            (__cplusplus >= 201402L)
#define __TBB_CPP14_VARIABLE_TEMPLATES_PRESENT          (__cplusplus >= 201402L)
#define __TBB_CPP17_DEDUCTION_GUIDES_PRESENT            (__INTEL_COMPILER > 1910) 
#define __TBB_CPP17_INVOKE_RESULT_PRESENT               (__cplusplus >= 201703L)
#elif __clang__

#define __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT          __has_feature(__cxx_variadic_templates__)
#define __TBB_CPP11_RVALUE_REF_PRESENT                  (__has_feature(__cxx_rvalue_references__) && (_LIBCPP_VERSION || __TBB_GLIBCXX_VERSION >= 40500))
#define __TBB_IMPLICIT_MOVE_PRESENT                     __has_feature(cxx_implicit_moves)

#define __TBB_EXCEPTION_PTR_PRESENT                     (__cplusplus >= 201103L && (_LIBCPP_VERSION || __TBB_GLIBCXX_VERSION >= 40600))
#define __TBB_STATIC_ASSERT_PRESENT                     __has_feature(__cxx_static_assert__)
#if (__cplusplus >= 201103L && __has_include(<tuple>))
#define __TBB_CPP11_TUPLE_PRESENT                   1
#endif
#if (__has_feature(__cxx_generalized_initializers__) && __has_include(<initializer_list>))
#define __TBB_INITIALIZER_LISTS_PRESENT             1
#endif
#define __TBB_CONSTEXPR_PRESENT                         __has_feature(__cxx_constexpr__)
#define __TBB_DEFAULTED_AND_DELETED_FUNC_PRESENT        (__has_feature(__cxx_defaulted_functions__) && __has_feature(__cxx_deleted_functions__))

#define __TBB_NOEXCEPT_PRESENT                          (__cplusplus >= 201103L)
#define __TBB_CPP11_STD_BEGIN_END_PRESENT               (__has_feature(__cxx_range_for__) && (_LIBCPP_VERSION || __TBB_GLIBCXX_VERSION >= 40600))
#define __TBB_CPP11_AUTO_PRESENT                        __has_feature(__cxx_auto_type__)
#define __TBB_CPP11_DECLTYPE_PRESENT                    __has_feature(__cxx_decltype__)
#define __TBB_CPP11_LAMBDAS_PRESENT                     __has_feature(cxx_lambdas)
#define __TBB_CPP11_DEFAULT_FUNC_TEMPLATE_ARGS_PRESENT  __has_feature(cxx_default_function_template_args)
#define __TBB_OVERRIDE_PRESENT                          __has_feature(cxx_override_control)
#define __TBB_ALIGNAS_PRESENT                           __has_feature(cxx_alignas)
#define __TBB_CPP11_TEMPLATE_ALIASES_PRESENT            __has_feature(cxx_alias_templates)
#define __TBB_CPP14_INTEGER_SEQUENCE_PRESENT            (__cplusplus >= 201402L)
#define __TBB_CPP14_VARIABLE_TEMPLATES_PRESENT          (__has_feature(cxx_variable_templates))
#define __TBB_CPP17_DEDUCTION_GUIDES_PRESENT            (__has_feature(__cpp_deduction_guides))
#define __TBB_CPP17_INVOKE_RESULT_PRESENT               (__has_feature(__cpp_lib_is_invocable))
#elif __GNUC__
#define __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT          __GXX_EXPERIMENTAL_CXX0X__
#define __TBB_CPP11_VARIADIC_FIXED_LENGTH_EXP_PRESENT   (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40700)
#define __TBB_CPP11_RVALUE_REF_PRESENT                  (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40500)
#define __TBB_IMPLICIT_MOVE_PRESENT                     (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40600)

#define __TBB_EXCEPTION_PTR_PRESENT                     (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40404 && __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4)
#define __TBB_STATIC_ASSERT_PRESENT                     (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40300)
#define __TBB_CPP11_TUPLE_PRESENT                       (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40300)
#define __TBB_INITIALIZER_LISTS_PRESENT                 (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40400)

#define __TBB_CONSTEXPR_PRESENT                         (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40400)
#define __TBB_DEFAULTED_AND_DELETED_FUNC_PRESENT        (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40400)
#define __TBB_NOEXCEPT_PRESENT                          (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40600)
#define __TBB_CPP11_STD_BEGIN_END_PRESENT               (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40600)
#define __TBB_CPP11_AUTO_PRESENT                        (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40400)
#define __TBB_CPP11_DECLTYPE_PRESENT                    (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40400)
#define __TBB_CPP11_LAMBDAS_PRESENT                     (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40500)
#define __TBB_CPP11_DEFAULT_FUNC_TEMPLATE_ARGS_PRESENT  (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40300)
#define __TBB_OVERRIDE_PRESENT                          (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40700)
#define __TBB_ALIGNAS_PRESENT                           (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40800)
#define __TBB_CPP11_TEMPLATE_ALIASES_PRESENT            (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40700)
#define __TBB_CPP14_INTEGER_SEQUENCE_PRESENT            (__cplusplus >= 201402L     && __TBB_GCC_VERSION >= 50000)
#define __TBB_CPP14_VARIABLE_TEMPLATES_PRESENT          (__cplusplus >= 201402L     && __TBB_GCC_VERSION >= 50000)
#define __TBB_CPP17_DEDUCTION_GUIDES_PRESENT            (__cpp_deduction_guides >= 201606L)
#define __TBB_CPP17_INVOKE_RESULT_PRESENT               (__cplusplus >= 201703L     && __TBB_GCC_VERSION >= 70000)
#elif _MSC_VER

#define __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT          (_MSC_VER >= 1800)
#define __TBB_CPP11_RVALUE_REF_PRESENT                  (_MSC_VER >= 1700 && (!__INTEL_COMPILER || __INTEL_COMPILER >= 1400))
#define __TBB_IMPLICIT_MOVE_PRESENT                     (_MSC_VER >= 1900)
#define __TBB_EXCEPTION_PTR_PRESENT                     (_MSC_VER >= 1600)
#define __TBB_STATIC_ASSERT_PRESENT                     (_MSC_VER >= 1600)
#define __TBB_CPP11_TUPLE_PRESENT                       (_MSC_VER >= 1600)
#define __TBB_INITIALIZER_LISTS_PRESENT                 (_MSC_VER >= 1800)
#define __TBB_CONSTEXPR_PRESENT                         (_MSC_VER >= 1900)
#define __TBB_DEFAULTED_AND_DELETED_FUNC_PRESENT        (_MSC_VER >= 1800)
#define __TBB_NOEXCEPT_PRESENT                          (_MSC_VER >= 1900)
#define __TBB_CPP11_STD_BEGIN_END_PRESENT               (_MSC_VER >= 1700)
#define __TBB_CPP11_AUTO_PRESENT                        (_MSC_VER >= 1600)
#define __TBB_CPP11_DECLTYPE_PRESENT                    (_MSC_VER >= 1600)
#define __TBB_CPP11_LAMBDAS_PRESENT                     (_MSC_VER >= 1600)
#define __TBB_CPP11_DEFAULT_FUNC_TEMPLATE_ARGS_PRESENT  (_MSC_VER >= 1800)
#define __TBB_OVERRIDE_PRESENT                          (_MSC_VER >= 1700)
#define __TBB_ALIGNAS_PRESENT                           (_MSC_VER >= 1900)
#define __TBB_CPP11_TEMPLATE_ALIASES_PRESENT            (_MSC_VER >= 1800)
#define __TBB_CPP14_INTEGER_SEQUENCE_PRESENT            (_MSC_VER >= 1900)

#define __TBB_CPP14_VARIABLE_TEMPLATES_PRESENT          (_MSC_FULL_VER >= 190023918 && (!__INTEL_COMPILER || __INTEL_COMPILER >= 1700))
#define __TBB_CPP17_DEDUCTION_GUIDES_PRESENT            (_MSVC_LANG >= 201703L && _MSC_VER >= 1914)
#define __TBB_CPP17_INVOKE_RESULT_PRESENT               (_MSVC_LANG >= 201703L && _MSC_VER >= 1911)
#else
#define __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT          __TBB_CPP11_PRESENT
#define __TBB_CPP11_RVALUE_REF_PRESENT                  __TBB_CPP11_PRESENT
#define __TBB_IMPLICIT_MOVE_PRESENT                     __TBB_CPP11_PRESENT
#define __TBB_EXCEPTION_PTR_PRESENT                     __TBB_CPP11_PRESENT
#define __TBB_STATIC_ASSERT_PRESENT                     __TBB_CPP11_PRESENT
#define __TBB_CPP11_TUPLE_PRESENT                       __TBB_CPP11_PRESENT
#define __TBB_INITIALIZER_LISTS_PRESENT                 __TBB_CPP11_PRESENT
#define __TBB_CONSTEXPR_PRESENT                         __TBB_CPP11_PRESENT
#define __TBB_DEFAULTED_AND_DELETED_FUNC_PRESENT        __TBB_CPP11_PRESENT
#define __TBB_NOEXCEPT_PRESENT                          __TBB_CPP11_PRESENT
#define __TBB_CPP11_STD_BEGIN_END_PRESENT               __TBB_CPP11_PRESENT
#define __TBB_CPP11_AUTO_PRESENT                        __TBB_CPP11_PRESENT
#define __TBB_CPP11_DECLTYPE_PRESENT                    __TBB_CPP11_PRESENT
#define __TBB_CPP11_LAMBDAS_PRESENT                     __TBB_CPP11_PRESENT
#define __TBB_CPP11_DEFAULT_FUNC_TEMPLATE_ARGS_PRESENT  __TBB_CPP11_PRESENT
#define __TBB_OVERRIDE_PRESENT                          __TBB_CPP11_PRESENT
#define __TBB_ALIGNAS_PRESENT                           __TBB_CPP11_PRESENT
#define __TBB_CPP11_TEMPLATE_ALIASES_PRESENT            __TBB_CPP11_PRESENT
#define __TBB_CPP14_INTEGER_SEQUENCE_PRESENT            (__cplusplus >= 201402L)
#define __TBB_CPP14_VARIABLE_TEMPLATES_PRESENT          (__cplusplus >= 201402L)
#define __TBB_CPP17_DEDUCTION_GUIDES_PRESENT            (__cplusplus >= 201703L)
#define __TBB_CPP17_INVOKE_RESULT_PRESENT               (__cplusplus >= 201703L)
#endif


#define __TBB_CPP11_ARRAY_PRESENT                           (_MSC_VER >= 1700 || _LIBCPP_VERSION || __GXX_EXPERIMENTAL_CXX0X__ && __TBB_GLIBCXX_VERSION >= 40300)

#ifndef __TBB_CPP11_VARIADIC_FIXED_LENGTH_EXP_PRESENT
#define __TBB_CPP11_VARIADIC_FIXED_LENGTH_EXP_PRESENT       __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
#endif
#define __TBB_CPP11_VARIADIC_TUPLE_PRESENT                  (!_MSC_VER || _MSC_VER >= 1800)

#define __TBB_CPP11_TYPE_PROPERTIES_PRESENT                 (_LIBCPP_VERSION || _MSC_VER >= 1700 || (__TBB_GLIBCXX_VERSION >= 50000 && __GXX_EXPERIMENTAL_CXX0X__))
#define __TBB_CPP11_IS_COPY_CONSTRUCTIBLE_PRESENT           (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GLIBCXX_VERSION >= 40700 || __TBB_CPP11_TYPE_PROPERTIES_PRESENT)

#define __TBB_MOVE_IF_NOEXCEPT_PRESENT                      (__TBB_NOEXCEPT_PRESENT && (__TBB_GLIBCXX_VERSION >= 40700 || _MSC_VER >= 1900 || _LIBCPP_VERSION))
#define __TBB_ALLOCATOR_TRAITS_PRESENT                      (__cplusplus >= 201103L && _LIBCPP_VERSION  || _MSC_VER >= 1800 ||  \
__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GLIBCXX_VERSION >= 40700 && !(__TBB_GLIBCXX_VERSION == 40700 && __TBB_DEFINE_MIC))
#define __TBB_MAKE_EXCEPTION_PTR_PRESENT                    (__TBB_EXCEPTION_PTR_PRESENT && (_MSC_VER >= 1700 || __TBB_GLIBCXX_VERSION >= 40600 || _LIBCPP_VERSION || __SUNPRO_CC))

#define __TBB_CPP11_SMART_POINTERS_PRESENT                  ( _MSC_VER >= 1600 || _LIBCPP_VERSION   \
|| ((__cplusplus >= 201103L || __GXX_EXPERIMENTAL_CXX0X__)  \
&& (__TBB_GLIBCXX_VERSION >= 40500 || __TBB_GLIBCXX_VERSION >= 40400 && __TBB_USE_OPTIONAL_RTTI)) )

#define __TBB_CPP11_FUTURE_PRESENT                          (_MSC_VER >= 1700 || __TBB_GLIBCXX_VERSION >= 40600 && __GXX_EXPERIMENTAL_CXX0X__ || _LIBCPP_VERSION)

#define __TBB_CPP11_GET_NEW_HANDLER_PRESENT                 (_MSC_VER >= 1900 || __TBB_GLIBCXX_VERSION >= 40900 && __GXX_EXPERIMENTAL_CXX0X__ || _LIBCPP_VERSION)

#define __TBB_CPP17_UNCAUGHT_EXCEPTIONS_PRESENT             (_MSC_VER >= 1900 || __GLIBCXX__ && __cpp_lib_uncaught_exceptions \
|| _LIBCPP_VERSION >= 3700 && (!__TBB_MACOS_TARGET_VERSION || __TBB_MACOS_TARGET_VERSION >= 101200))
#define __TBB_CPP17_MEMORY_RESOURCE_PRESENT                 (_MSC_VER >= 1913 && (_MSVC_LANG > 201402L || __cplusplus > 201402L) || \
__GLIBCXX__ && __cpp_lib_memory_resource >= 201603)
#define __TBB_CPP17_HW_INTERFERENCE_SIZE_PRESENT            (_MSC_VER >= 1911)
#if _MSC_VER>=1400 || _LIBCPP_VERSION || __GXX_EXPERIMENTAL_CXX0X__
#define __TBB_STD_SWAP_HEADER <utility>
#else
#define __TBB_STD_SWAP_HEADER <algorithm>
#endif

#if __INTEL_COMPILER && __GNUC__ && __TBB_EXCEPTION_PTR_PRESENT && !defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4)
#define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 1
#endif

#if __MINGW32__ && __TBB_EXCEPTION_PTR_PRESENT && !defined(_GLIBCXX_ATOMIC_BUILTINS_4)
#define _GLIBCXX_ATOMIC_BUILTINS_4
#endif

#if __GNUC__ || __SUNPRO_CC || __IBMCPP__

#define __TBB_ATTRIBUTE_ALIGNED_PRESENT 1
#elif _MSC_VER && (_MSC_VER >= 1300 || __INTEL_COMPILER)
#define __TBB_DECLSPEC_ALIGN_PRESENT 1
#endif




#if __TBB_GCC_VERSION >= 40306 || __INTEL_COMPILER >= 1200 || __clang__

#define __TBB_GCC_BUILTIN_ATOMICS_PRESENT 1
#endif

#if __TBB_GCC_VERSION >= 70000 && !__INTEL_COMPILER && !__clang__
#define TBB_USE_GCC_BUILTINS 1
#endif

#if __INTEL_COMPILER >= 1200

#define __TBB_ICC_BUILTIN_ATOMICS_PRESENT 1
#endif

#if _MSC_VER>=1600 && (!__INTEL_COMPILER || __INTEL_COMPILER>=1310)
#define __TBB_MSVC_PART_WORD_INTERLOCKED_INTRINSICS_PRESENT 1
#endif

#define __TBB_TSX_INTRINSICS_PRESENT ((__RTM__ || _MSC_VER>=1700 || __INTEL_COMPILER>=1300) && !__TBB_DEFINE_MIC && !__ANDROID__)


#define __TBB_CONCAT_AUX(A,B) A##B
#define __TBB_CONCAT(A,B) __TBB_CONCAT_AUX(A,B)
#define __TBB_IS_MACRO_EMPTY(A,IGNORED) __TBB_CONCAT_AUX(__TBB_MACRO_EMPTY,A)
#define __TBB_MACRO_EMPTY 1


#ifndef TBB_USE_DEBUG

#ifdef _DEBUG
#define __TBB_IS__DEBUG_EMPTY (__TBB_IS_MACRO_EMPTY(_DEBUG,IGNORED)==__TBB_MACRO_EMPTY)
#if __TBB_IS__DEBUG_EMPTY
#define TBB_USE_DEBUG 1
#else
#define TBB_USE_DEBUG _DEBUG
#endif 
#else
#define TBB_USE_DEBUG 0
#endif
#endif 

#ifndef TBB_USE_ASSERT
#define TBB_USE_ASSERT TBB_USE_DEBUG
#endif 

#ifndef TBB_USE_THREADING_TOOLS
#define TBB_USE_THREADING_TOOLS TBB_USE_DEBUG
#endif 

#ifndef TBB_USE_PERFORMANCE_WARNINGS
#ifdef TBB_PERFORMANCE_WARNINGS
#define TBB_USE_PERFORMANCE_WARNINGS TBB_PERFORMANCE_WARNINGS
#else
#define TBB_USE_PERFORMANCE_WARNINGS TBB_USE_DEBUG
#endif 
#endif 

#if __TBB_DEFINE_MIC
#if TBB_USE_EXCEPTIONS
#error The platform does not properly support exception handling. Please do not set TBB_USE_EXCEPTIONS macro or set it to 0.
#elif !defined(TBB_USE_EXCEPTIONS)
#define TBB_USE_EXCEPTIONS 0
#endif
#elif !(__EXCEPTIONS || defined(_CPPUNWIND) || __SUNPRO_CC)
#if TBB_USE_EXCEPTIONS
#error Compilation settings do not support exception handling. Please do not set TBB_USE_EXCEPTIONS macro or set it to 0.
#elif !defined(TBB_USE_EXCEPTIONS)
#define TBB_USE_EXCEPTIONS 0
#endif
#elif !defined(TBB_USE_EXCEPTIONS)
#define TBB_USE_EXCEPTIONS 1
#endif

#ifndef TBB_IMPLEMENT_CPP0X

#if __clang__

#if (__INTEL_COMPILER && (__INTEL_COMPILER < 1500 || __INTEL_COMPILER == 1500 && __INTEL_COMPILER_UPDATE <= 1))
#define TBB_IMPLEMENT_CPP0X (__cplusplus < 201103L || !_LIBCPP_VERSION)
#else
#define TBB_IMPLEMENT_CPP0X (__cplusplus < 201103L || (!__has_include(<thread>) && !__has_include(<condition_variable>)))
#endif
#elif __GNUC__
#define TBB_IMPLEMENT_CPP0X (__TBB_GCC_VERSION < 40400 || !__GXX_EXPERIMENTAL_CXX0X__)
#elif _MSC_VER
#define TBB_IMPLEMENT_CPP0X (_MSC_VER < 1700)
#else
#define TBB_IMPLEMENT_CPP0X (!__STDCPP_THREADS__)
#endif
#endif 


#ifndef TBB_USE_CAPTURED_EXCEPTION

#if __TBB_EXCEPTION_PTR_PRESENT && !defined(__ia64__)
#define TBB_USE_CAPTURED_EXCEPTION 0
#else
#define TBB_USE_CAPTURED_EXCEPTION 1
#endif
#else 
#if !TBB_USE_CAPTURED_EXCEPTION && !__TBB_EXCEPTION_PTR_PRESENT
#error Current runtime does not support std::exception_ptr. Set TBB_USE_CAPTURED_EXCEPTION and make sure that your code is ready to catch tbb::captured_exception.
#endif
#endif 


#if TBB_USE_GCC_BUILTINS && !__TBB_GCC_BUILTIN_ATOMICS_PRESENT
#error "GCC atomic built-ins are not supported."
#endif




#ifndef __TBB_CONCURRENT_ORDERED_CONTAINERS_PRESENT
#define __TBB_CONCURRENT_ORDERED_CONTAINERS_PRESENT ( __TBB_CPP11_RVALUE_REF_PRESENT && __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT \
&& __TBB_IMPLICIT_MOVE_PRESENT  && __TBB_CPP11_AUTO_PRESENT && __TBB_CPP11_LAMBDAS_PRESENT && __TBB_CPP11_ARRAY_PRESENT \
&& __TBB_INITIALIZER_LISTS_PRESENT )
#endif


#ifndef __TBB_WEAK_SYMBOLS_PRESENT
#define __TBB_WEAK_SYMBOLS_PRESENT ( !_WIN32 && !__APPLE__ && !__sun && (__TBB_GCC_VERSION >= 40000 || __INTEL_COMPILER ) )
#endif


#ifndef __TBB_DYNAMIC_LOAD_ENABLED
#define __TBB_DYNAMIC_LOAD_ENABLED 1
#endif


#if (_WIN32||_WIN64) && (__TBB_SOURCE_DIRECTLY_INCLUDED || TBB_USE_PREVIEW_BINARY)
#define __TBB_NO_IMPLICIT_LINKAGE 1
#define __TBBMALLOC_NO_IMPLICIT_LINKAGE 1
#endif

#ifndef __TBB_COUNT_TASK_NODES
#define __TBB_COUNT_TASK_NODES TBB_USE_ASSERT
#endif

#ifndef __TBB_TASK_GROUP_CONTEXT
#define __TBB_TASK_GROUP_CONTEXT 1
#endif 

#ifndef __TBB_SCHEDULER_OBSERVER
#define __TBB_SCHEDULER_OBSERVER 1
#endif 

#ifndef __TBB_FP_CONTEXT
#define __TBB_FP_CONTEXT __TBB_TASK_GROUP_CONTEXT
#endif 

#if __TBB_FP_CONTEXT && !__TBB_TASK_GROUP_CONTEXT
#error __TBB_FP_CONTEXT requires __TBB_TASK_GROUP_CONTEXT to be enabled
#endif

#define __TBB_RECYCLE_TO_ENQUEUE __TBB_BUILD 

#ifndef __TBB_ARENA_OBSERVER
#define __TBB_ARENA_OBSERVER __TBB_SCHEDULER_OBSERVER
#endif 

#ifndef __TBB_TASK_ISOLATION
#define __TBB_TASK_ISOLATION 1
#endif 

#if TBB_USE_EXCEPTIONS && !__TBB_TASK_GROUP_CONTEXT
#error TBB_USE_EXCEPTIONS requires __TBB_TASK_GROUP_CONTEXT to be enabled
#endif

#ifndef __TBB_TASK_PRIORITY
#define __TBB_TASK_PRIORITY (__TBB_TASK_GROUP_CONTEXT)
#endif 

#if __TBB_TASK_PRIORITY && !__TBB_TASK_GROUP_CONTEXT
#error __TBB_TASK_PRIORITY requires __TBB_TASK_GROUP_CONTEXT to be enabled
#endif

#if TBB_PREVIEW_NUMA_SUPPORT || __TBB_BUILD
#define __TBB_NUMA_SUPPORT 1
#endif

#if TBB_PREVIEW_WAITING_FOR_WORKERS || __TBB_BUILD
#define __TBB_SUPPORTS_WORKERS_WAITING_IN_TERMINATE 1
#endif

#ifndef __TBB_ENQUEUE_ENFORCED_CONCURRENCY
#define __TBB_ENQUEUE_ENFORCED_CONCURRENCY 1
#endif

#if !defined(__TBB_SURVIVE_THREAD_SWITCH) && \
(_WIN32 || _WIN64 || __APPLE__ || (__linux__ && !__ANDROID__))
#define __TBB_SURVIVE_THREAD_SWITCH 1
#endif 

#ifndef __TBB_DEFAULT_PARTITIONER
#define __TBB_DEFAULT_PARTITIONER tbb::auto_partitioner
#endif

#ifndef __TBB_USE_PROPORTIONAL_SPLIT_IN_BLOCKED_RANGES
#define __TBB_USE_PROPORTIONAL_SPLIT_IN_BLOCKED_RANGES 1
#endif

#ifndef __TBB_ENABLE_RANGE_FEEDBACK
#define __TBB_ENABLE_RANGE_FEEDBACK 0
#endif

#ifdef _VARIADIC_MAX
#define __TBB_VARIADIC_MAX _VARIADIC_MAX
#else
#if _MSC_VER == 1700
#define __TBB_VARIADIC_MAX 5 
#elif _MSC_VER == 1600
#define __TBB_VARIADIC_MAX 10 
#else
#define __TBB_VARIADIC_MAX 15
#endif
#endif

#if !defined(__INTEL_COMPILER) && (!defined(TBB_SUPPRESS_DEPRECATED_MESSAGES) || (TBB_SUPPRESS_DEPRECATED_MESSAGES == 0))
#if (__cplusplus >= 201402L)
#define __TBB_DEPRECATED [[deprecated]]
#define __TBB_DEPRECATED_MSG(msg) [[deprecated(msg)]]
#elif _MSC_VER
#define __TBB_DEPRECATED __declspec(deprecated)
#define __TBB_DEPRECATED_MSG(msg) __declspec(deprecated(msg))
#elif (__GNUC__ && __TBB_GCC_VERSION >= 40805) || __clang__
#define __TBB_DEPRECATED __attribute__((deprecated))
#define __TBB_DEPRECATED_MSG(msg) __attribute__((deprecated(msg)))
#endif
#endif  

#if !defined(__TBB_DEPRECATED)
#define __TBB_DEPRECATED
#define __TBB_DEPRECATED_MSG(msg)
#elif !defined(__TBB_SUPPRESS_INTERNAL_DEPRECATED_MESSAGES)
#define __TBB_SUPPRESS_INTERNAL_DEPRECATED_MESSAGES 1
#endif

#if defined(TBB_SUPPRESS_DEPRECATED_MESSAGES) && (TBB_SUPPRESS_DEPRECATED_MESSAGES == 0)
#define __TBB_DEPRECATED_IN_VERBOSE_MODE __TBB_DEPRECATED
#define __TBB_DEPRECATED_IN_VERBOSE_MODE_MSG(msg) __TBB_DEPRECATED_MSG(msg)
#else
#define __TBB_DEPRECATED_IN_VERBOSE_MODE
#define __TBB_DEPRECATED_IN_VERBOSE_MODE_MSG(msg)
#endif 

#if (!defined(TBB_SUPPRESS_DEPRECATED_MESSAGES) || (TBB_SUPPRESS_DEPRECATED_MESSAGES == 0)) && !__TBB_CPP11_PRESENT
#pragma message("TBB Warning: Support for C++98/03 is deprecated. Please use the compiler that supports C++11 features at least.")
#endif


#if defined(WINAPI_FAMILY) && WINAPI_FAMILY == WINAPI_FAMILY_APP
#define __TBB_WIN8UI_SUPPORT 1
#else
#define __TBB_WIN8UI_SUPPORT 0
#endif



#if __SIZEOF_POINTER__ < 8 && __ANDROID__ && __TBB_GCC_VERSION <= 40403 && !__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8

#define __TBB_GCC_64BIT_ATOMIC_BUILTINS_BROKEN 1
#elif __TBB_x86_32 && __TBB_GCC_VERSION == 40102 && ! __GNUC_RH_RELEASE__

#define __TBB_GCC_64BIT_ATOMIC_BUILTINS_BROKEN 1
#endif

#if __GNUC__ && __TBB_x86_64 && __INTEL_COMPILER == 1200
#define __TBB_ICC_12_0_INL_ASM_FSTCW_BROKEN 1
#endif

#if _MSC_VER && __INTEL_COMPILER && (__INTEL_COMPILER<1110 || __INTEL_COMPILER==1110 && __INTEL_COMPILER_BUILD_DATE < 20091012)

#define __TBB_DEFAULT_DTOR_THROW_SPEC_BROKEN 1
#endif

#if !__INTEL_COMPILER && (_MSC_VER && _MSC_VER < 1500 || __GNUC__ && __TBB_GCC_VERSION < 40102)

#define __TBB_TEMPLATE_FRIENDS_BROKEN 1
#endif

#if __GLIBC__==2 && __GLIBC_MINOR__==3 ||  (__APPLE__ && ( __INTEL_COMPILER==1200 && !TBB_USE_DEBUG))

#define __TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN 1
#endif

#if (_WIN32||_WIN64) && __INTEL_COMPILER == 1110

#define __TBB_ICL_11_1_CODE_GEN_BROKEN 1
#endif

#if __clang__ || (__GNUC__==3 && __GNUC_MINOR__==3 && !defined(__INTEL_COMPILER))

#define __TBB_PROTECTED_NESTED_CLASS_BROKEN 1
#endif

#if __MINGW32__ && __TBB_GCC_VERSION < 40200

#define __TBB_SSE_STACK_ALIGNMENT_BROKEN 1
#endif

#if __TBB_GCC_VERSION==40300 && !__INTEL_COMPILER && !__clang__

#define __TBB_GCC_OPTIMIZER_ORDERING_BROKEN 1
#endif

#if __FreeBSD__

#define __TBB_PRIO_INHERIT_BROKEN 1


#define __TBB_PLACEMENT_NEW_EXCEPTION_SAFETY_BROKEN 1
#endif 

#if (__linux__ || __APPLE__) && __i386__ && defined(__INTEL_COMPILER)

#define __TBB_ICC_ASM_VOLATILE_BROKEN 1
#endif

#if !__INTEL_COMPILER && (_MSC_VER && _MSC_VER < 1700 || __GNUC__==3 && __GNUC_MINOR__<=2)

#define __TBB_ALIGNOF_NOT_INSTANTIATED_TYPES_BROKEN 1
#endif

#if __TBB_DEFINE_MIC

#define __TBB_MAIN_THREAD_AFFINITY_BROKEN 1
#endif

#if __GXX_EXPERIMENTAL_CXX0X__ && !defined(__EXCEPTIONS) && \
((!__INTEL_COMPILER && !__clang__ && (__TBB_GCC_VERSION>=40400 && __TBB_GCC_VERSION<40600)) || \
(__INTEL_COMPILER<=1400 && (__TBB_GLIBCXX_VERSION>=40400 && __TBB_GLIBCXX_VERSION<=40801)))

#define __TBB_LIBSTDCPP_EXCEPTION_HEADERS_BROKEN 1
#endif

#if __INTEL_COMPILER==1300 && __TBB_GLIBCXX_VERSION>=40700 && defined(__GXX_EXPERIMENTAL_CXX0X__)

#define __TBB_ICC_13_0_CPP11_STDLIB_SUPPORT_BROKEN 1
#endif

#if (__GNUC__==4 && __GNUC_MINOR__==4 ) && !defined(__INTEL_COMPILER) && !defined(__clang__)

#define __TBB_GCC_STRICT_ALIASING_BROKEN 1

#if !__TBB_GCC_WARNING_SUPPRESSION_PRESENT
#error Warning suppression is not supported, while should.
#endif
#endif


#if __TBB_GCC_VERSION == 40102 && __PIC__ && !defined(__INTEL_COMPILER) && !defined(__clang__)
#define __TBB_GCC_CAS8_BUILTIN_INLINING_BROKEN 1
#endif

#if __TBB_x86_32 && ( __INTEL_COMPILER || (__GNUC__==5 && __GNUC_MINOR__>=2 && __GXX_EXPERIMENTAL_CXX0X__) \
|| (__GNUC__==3 && __GNUC_MINOR__==3) || (__MINGW32__ && __GNUC__==4 && __GNUC_MINOR__==5) || __SUNPRO_CC )
#define __TBB_FORCE_64BIT_ALIGNMENT_BROKEN 1
#else
#define __TBB_FORCE_64BIT_ALIGNMENT_BROKEN 0
#endif

#if __GNUC__ && !__INTEL_COMPILER && !__clang__ && __TBB_DEFAULTED_AND_DELETED_FUNC_PRESENT && __TBB_GCC_VERSION < 40700
#define __TBB_ZERO_INIT_WITH_DEFAULTED_CTOR_BROKEN 1
#endif

#if _MSC_VER && _MSC_VER <= 1800 && !__INTEL_COMPILER
#define __TBB_CONST_REF_TO_ARRAY_TEMPLATE_PARAM_BROKEN 1
#endif

#define __TBB_IF_NO_COPY_CTOR_MOVE_SEMANTICS_BROKEN (_MSC_VER && (__INTEL_COMPILER >= 1300 && __INTEL_COMPILER <= 1310) && !__INTEL_CXX11_MODE__)

#define __TBB_CPP11_DECLVAL_BROKEN (_MSC_VER == 1600 || (__GNUC__ && __TBB_GCC_VERSION < 40500) )
#define __TBB_COPY_FROM_NON_CONST_REF_BROKEN (_MSC_VER == 1700 && __INTEL_COMPILER && __INTEL_COMPILER < 1600)

#if __GXX_EXPERIMENTAL_CXX0X__ && __GLIBCXX__ && ((__INTEL_COMPILER >=1300 && __INTEL_COMPILER <=1310 && __TBB_GLIBCXX_VERSION>=40700) || (__TBB_GLIBCXX_VERSION < 40500))
#define __TBB_UPCAST_OF_TUPLE_OF_REF_BROKEN 1
#endif

#define __TBB_CPP11_DECLTYPE_OF_FUNCTION_RETURN_TYPE_BROKEN (_MSC_VER == 1600 && !__INTEL_COMPILER)

#if _MSC_VER && _MSC_VER <= 1800
#define __TBB_IMPLICIT_COPY_DELETION_BROKEN 1
#endif



#if defined(_MSC_VER) && _MSC_VER>=1500 && !defined(__INTEL_COMPILER)
#define __TBB_MSVC_UNREACHABLE_CODE_IGNORED 1
#endif

#define __TBB_ATOMIC_CTORS     (__TBB_CONSTEXPR_PRESENT && __TBB_DEFAULTED_AND_DELETED_FUNC_PRESENT && (!__TBB_ZERO_INIT_WITH_DEFAULTED_CTOR_BROKEN))

#if __ANDROID__
#include <android/api-level.h>
#define __TBB_USE_DLOPEN_REENTRANCY_WORKAROUND  (__ANDROID_API__ < 19)
#endif

#define __TBB_ALLOCATOR_CONSTRUCT_VARIADIC      (__TBB_CPP11_VARIADIC_TEMPLATES_PRESENT && __TBB_CPP11_RVALUE_REF_PRESENT)

#define __TBB_VARIADIC_PARALLEL_INVOKE          (TBB_PREVIEW_VARIADIC_PARALLEL_INVOKE && __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT && __TBB_CPP11_RVALUE_REF_PRESENT)
#define __TBB_FLOW_GRAPH_CPP11_FEATURES         (__TBB_CPP11_VARIADIC_TEMPLATES_PRESENT \
&& __TBB_CPP11_SMART_POINTERS_PRESENT && __TBB_CPP11_RVALUE_REF_PRESENT && __TBB_CPP11_AUTO_PRESENT) \
&& __TBB_CPP11_VARIADIC_TUPLE_PRESENT && __TBB_CPP11_DEFAULT_FUNC_TEMPLATE_ARGS_PRESENT \
&& !__TBB_UPCAST_OF_TUPLE_OF_REF_BROKEN
#define __TBB_PREVIEW_STREAMING_NODE            (__TBB_CPP11_VARIADIC_FIXED_LENGTH_EXP_PRESENT && __TBB_FLOW_GRAPH_CPP11_FEATURES \
&& TBB_PREVIEW_FLOW_GRAPH_NODES && !TBB_IMPLEMENT_CPP0X && !__TBB_UPCAST_OF_TUPLE_OF_REF_BROKEN)
#define __TBB_PREVIEW_OPENCL_NODE               (__TBB_PREVIEW_STREAMING_NODE && __TBB_CPP11_TEMPLATE_ALIASES_PRESENT)
#define __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING (TBB_PREVIEW_FLOW_GRAPH_FEATURES || __TBB_PREVIEW_OPENCL_NODE)
#define __TBB_PREVIEW_ASYNC_MSG                 (TBB_PREVIEW_FLOW_GRAPH_FEATURES && __TBB_FLOW_GRAPH_CPP11_FEATURES)


#ifndef __TBB_PREVIEW_FLOW_GRAPH_PRIORITIES
#define __TBB_PREVIEW_FLOW_GRAPH_PRIORITIES     TBB_PREVIEW_FLOW_GRAPH_FEATURES
#endif

#ifndef __TBB_PREVIEW_RESUMABLE_TASKS
#define __TBB_PREVIEW_RESUMABLE_TASKS           ((__TBB_CPF_BUILD || TBB_PREVIEW_RESUMABLE_TASKS) && !__TBB_WIN8UI_SUPPORT && !__ANDROID__ && !__TBB_ipf)
#endif

#ifndef __TBB_PREVIEW_CRITICAL_TASKS
#define __TBB_PREVIEW_CRITICAL_TASKS            (__TBB_CPF_BUILD || __TBB_PREVIEW_FLOW_GRAPH_PRIORITIES || __TBB_PREVIEW_RESUMABLE_TASKS)
#endif

#ifndef __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
#define __TBB_PREVIEW_FLOW_GRAPH_NODE_SET       (TBB_PREVIEW_FLOW_GRAPH_FEATURES && __TBB_CPP11_PRESENT && __TBB_FLOW_GRAPH_CPP11_FEATURES)
#endif

#endif 

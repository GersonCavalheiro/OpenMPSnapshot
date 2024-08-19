
#include <cstdlib>
#include <cerrno>
#include <ctime>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <typeinfo>

#include <limits>
#include <algorithm>
#include <complex>
#include <deque>
#include <queue>
#include <cassert>
#include <list>
#if __cplusplus >= 201103L
#include <random>
#ifdef EIGEN_USE_THREADS
#include <future>
#endif
#endif

#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 9)
#define EIGEN_TEST_CUDACC_VER  ((__CUDACC_VER_MAJOR__ * 10000) + (__CUDACC_VER_MINOR__ * 100))
#elif defined(__CUDACC_VER__)
#define EIGEN_TEST_CUDACC_VER __CUDACC_VER__
#else
#define EIGEN_TEST_CUDACC_VER 0
#endif

#if EIGEN_TEST_CUDACC_VER >= 70500
#include <cuda_fp16.h>
#endif

#define min(A,B) please_protect_your_min_with_parentheses
#define max(A,B) please_protect_your_max_with_parentheses
#define isnan(X) please_protect_your_isnan_with_parentheses
#define isinf(X) please_protect_your_isinf_with_parentheses
#define isfinite(X) please_protect_your_isfinite_with_parentheses

struct real {};
struct imag {};

#ifdef M_PI
#undef M_PI
#endif
#define M_PI please_use_EIGEN_PI_instead_of_M_PI

#define FORBIDDEN_IDENTIFIER (this_identifier_is_forbidden_to_avoid_clashes) this_identifier_is_forbidden_to_avoid_clashes
#define B0 FORBIDDEN_IDENTIFIER

#ifndef EIGEN_NO_DEBUG_SMALL_PRODUCT_BLOCKS
#define EIGEN_DEBUG_SMALL_PRODUCT_BLOCKS
#endif

#define TEST_SET_BUT_UNUSED_VARIABLE(X) EIGEN_UNUSED_VARIABLE(X)

#ifdef TEST_ENABLE_TEMPORARY_TRACKING

static long int nb_temporaries;
static long int nb_temporaries_on_assert = -1;

inline void on_temporary_creation(long int size) {
if(size!=0) nb_temporaries++;
if(nb_temporaries_on_assert>0) assert(nb_temporaries<nb_temporaries_on_assert);
}

#define EIGEN_DENSE_STORAGE_CTOR_PLUGIN { on_temporary_creation(size); }

#define VERIFY_EVALUATION_COUNT(XPR,N) {\
nb_temporaries = 0; \
XPR; \
if(nb_temporaries!=N) { std::cerr << "nb_temporaries == " << nb_temporaries << "\n"; }\
VERIFY( (#XPR) && nb_temporaries==N ); \
}

#endif

#include "split_test_helper.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#ifndef DEBUG
#define DEBUG
#endif

#if defined(__ALTIVEC__) || defined(__VSX__)
#define EIGEN_MAKING_DOCS
#endif

#ifndef EIGEN_TEST_FUNC
#error EIGEN_TEST_FUNC must be defined
#endif

#define DEFAULT_REPEAT 10

namespace Eigen
{
static std::vector<std::string> g_test_stack;
static int g_test_level = 0;
static int g_repeat;
static unsigned int g_seed;
static bool g_has_set_repeat, g_has_set_seed;
}

#define TRACK std::cerr << __FILE__ << " " << __LINE__ << std::endl

#define EI_PP_MAKE_STRING2(S) #S
#define EI_PP_MAKE_STRING(S) EI_PP_MAKE_STRING2(S)

#define EIGEN_DEFAULT_IO_FORMAT IOFormat(4, 0, "  ", "\n", "", "", "", "")

#if (defined(_CPPUNWIND) || defined(__EXCEPTIONS)) && !defined(__CUDA_ARCH__)
#define EIGEN_EXCEPTIONS
#endif

#ifndef EIGEN_NO_ASSERTION_CHECKING

namespace Eigen
{
static const bool should_raise_an_assert = false;

static bool no_more_assert = false;
static bool report_on_cerr_on_assert_failure = true;

struct eigen_assert_exception
{
eigen_assert_exception(void) {}
~eigen_assert_exception() { Eigen::no_more_assert = false; }
};

struct eigen_static_assert_exception
{
eigen_static_assert_exception(void) {}
~eigen_static_assert_exception() { Eigen::no_more_assert = false; }
};
}
#ifdef EIGEN_DEBUG_ASSERTS

namespace Eigen
{
namespace internal
{
static bool push_assert = false;
}
static std::vector<std::string> eigen_assert_list;
}
#define eigen_assert(a)                       \
if( (!(a)) && (!no_more_assert) )     \
{ \
if(report_on_cerr_on_assert_failure) \
std::cerr <<  #a << " " __FILE__ << "(" << __LINE__ << ")\n"; \
Eigen::no_more_assert = true;       \
EIGEN_THROW_X(Eigen::eigen_assert_exception()); \
}                                     \
else if (Eigen::internal::push_assert)       \
{                                     \
eigen_assert_list.push_back(std::string(EI_PP_MAKE_STRING(__FILE__) " (" EI_PP_MAKE_STRING(__LINE__) ") : " #a) ); \
}

#ifdef EIGEN_EXCEPTIONS
#define VERIFY_RAISES_ASSERT(a)                                                   \
{                                                                               \
Eigen::no_more_assert = false;                                                \
Eigen::eigen_assert_list.clear();                                             \
Eigen::internal::push_assert = true;                                          \
Eigen::report_on_cerr_on_assert_failure = false;                              \
try {                                                                         \
a;                                                                          \
std::cerr << "One of the following asserts should have been triggered:\n";  \
for (uint ai=0 ; ai<eigen_assert_list.size() ; ++ai)                        \
std::cerr << "  " << eigen_assert_list[ai] << "\n";                       \
VERIFY(Eigen::should_raise_an_assert && # a);                               \
} catch (Eigen::eigen_assert_exception) {                                     \
Eigen::internal::push_assert = false; VERIFY(true);                         \
}                                                                             \
Eigen::report_on_cerr_on_assert_failure = true;                               \
Eigen::internal::push_assert = false;                                         \
}
#endif 

#elif !defined(__CUDACC__) 
#define eigen_assert(a) \
if( (!Eigen::internal::copy_bool(a)) && (!no_more_assert) )\
{                                       \
Eigen::no_more_assert = true;         \
if(report_on_cerr_on_assert_failure)  \
eigen_plain_assert(a);              \
else                                  \
EIGEN_THROW_X(Eigen::eigen_assert_exception()); \
}

#ifdef EIGEN_EXCEPTIONS
#define VERIFY_RAISES_ASSERT(a) {                           \
Eigen::no_more_assert = false;                            \
Eigen::report_on_cerr_on_assert_failure = false;          \
try {                                                     \
a;                                                      \
VERIFY(Eigen::should_raise_an_assert && # a);           \
}                                                         \
catch (Eigen::eigen_assert_exception&) { VERIFY(true); }  \
Eigen::report_on_cerr_on_assert_failure = true;           \
}
#endif 
#endif 

#if defined(TEST_CHECK_STATIC_ASSERTIONS) && defined(EIGEN_EXCEPTIONS)
#define EIGEN_STATIC_ASSERT(a,MSG) \
if( (!Eigen::internal::copy_bool(a)) && (!no_more_assert) )\
{                                       \
Eigen::no_more_assert = true;         \
if(report_on_cerr_on_assert_failure)  \
eigen_plain_assert((a) && #MSG);      \
else                                  \
EIGEN_THROW_X(Eigen::eigen_static_assert_exception()); \
}
#define VERIFY_RAISES_STATIC_ASSERT(a) {                    \
Eigen::no_more_assert = false;                            \
Eigen::report_on_cerr_on_assert_failure = false;          \
try {                                                     \
a;                                                      \
VERIFY(Eigen::should_raise_an_assert && # a);           \
}                                                         \
catch (Eigen::eigen_static_assert_exception&) { VERIFY(true); }  \
Eigen::report_on_cerr_on_assert_failure = true;           \
}
#endif 

#ifndef VERIFY_RAISES_ASSERT
#define VERIFY_RAISES_ASSERT(a) \
std::cout << "Can't VERIFY_RAISES_ASSERT( " #a " ) with exceptions disabled\n";
#endif
#ifndef VERIFY_RAISES_STATIC_ASSERT
#define VERIFY_RAISES_STATIC_ASSERT(a) \
std::cout << "Can't VERIFY_RAISES_STATIC_ASSERT( " #a " ) with exceptions disabled\n";
#endif

#if !defined(__CUDACC__)
#define EIGEN_USE_CUSTOM_ASSERT
#endif

#else 

#define VERIFY_RAISES_ASSERT(a) {}
#define VERIFY_RAISES_STATIC_ASSERT(a) {}

#endif 

#define EIGEN_INTERNAL_DEBUGGING
#include <Eigen/QR> 

inline void verify_impl(bool condition, const char *testname, const char *file, int line, const char *condition_as_string)
{
if (!condition)
{
if(Eigen::g_test_level>0)
std::cerr << "WARNING: ";
std::cerr << "Test " << testname << " failed in " << file << " (" << line << ")"
<< std::endl << "    " << condition_as_string << std::endl;
std::cerr << "Stack:\n";
const int test_stack_size = static_cast<int>(Eigen::g_test_stack.size());
for(int i=test_stack_size-1; i>=0; --i)
std::cerr << "  - " << Eigen::g_test_stack[i] << "\n";
std::cerr << "\n";
if(Eigen::g_test_level==0)
abort();
}
}

#define VERIFY(a) ::verify_impl(a, g_test_stack.back().c_str(), __FILE__, __LINE__, EI_PP_MAKE_STRING(a))

#define VERIFY_GE(a, b) ::verify_impl(a >= b, g_test_stack.back().c_str(), __FILE__, __LINE__, EI_PP_MAKE_STRING(a >= b))
#define VERIFY_LE(a, b) ::verify_impl(a <= b, g_test_stack.back().c_str(), __FILE__, __LINE__, EI_PP_MAKE_STRING(a <= b))


#define VERIFY_IS_EQUAL(a, b) VERIFY(test_is_equal(a, b, true))
#define VERIFY_IS_NOT_EQUAL(a, b) VERIFY(test_is_equal(a, b, false))
#define VERIFY_IS_APPROX(a, b) VERIFY(verifyIsApprox(a, b))
#define VERIFY_IS_NOT_APPROX(a, b) VERIFY(!test_isApprox(a, b))
#define VERIFY_IS_MUCH_SMALLER_THAN(a, b) VERIFY(test_isMuchSmallerThan(a, b))
#define VERIFY_IS_NOT_MUCH_SMALLER_THAN(a, b) VERIFY(!test_isMuchSmallerThan(a, b))
#define VERIFY_IS_APPROX_OR_LESS_THAN(a, b) VERIFY(test_isApproxOrLessThan(a, b))
#define VERIFY_IS_NOT_APPROX_OR_LESS_THAN(a, b) VERIFY(!test_isApproxOrLessThan(a, b))

#define VERIFY_IS_UNITARY(a) VERIFY(test_isUnitary(a))

#define CALL_SUBTEST(FUNC) do { \
g_test_stack.push_back(EI_PP_MAKE_STRING(FUNC)); \
FUNC; \
g_test_stack.pop_back(); \
} while (0)


namespace Eigen {

template<typename T> inline typename NumTraits<T>::Real test_precision() { return NumTraits<T>::dummy_precision(); }
template<> inline float test_precision<float>() { return 1e-3f; }
template<> inline double test_precision<double>() { return 1e-6; }
template<> inline long double test_precision<long double>() { return 1e-6l; }
template<> inline float test_precision<std::complex<float> >() { return test_precision<float>(); }
template<> inline double test_precision<std::complex<double> >() { return test_precision<double>(); }
template<> inline long double test_precision<std::complex<long double> >() { return test_precision<long double>(); }

inline bool test_isApprox(const short& a, const short& b)
{ return internal::isApprox(a, b, test_precision<short>()); }
inline bool test_isApprox(const unsigned short& a, const unsigned short& b)
{ return internal::isApprox(a, b, test_precision<unsigned short>()); }
inline bool test_isApprox(const unsigned int& a, const unsigned int& b)
{ return internal::isApprox(a, b, test_precision<unsigned int>()); }
inline bool test_isApprox(const long& a, const long& b)
{ return internal::isApprox(a, b, test_precision<long>()); }
inline bool test_isApprox(const unsigned long& a, const unsigned long& b)
{ return internal::isApprox(a, b, test_precision<unsigned long>()); }

inline bool test_isApprox(const int& a, const int& b)
{ return internal::isApprox(a, b, test_precision<int>()); }
inline bool test_isMuchSmallerThan(const int& a, const int& b)
{ return internal::isMuchSmallerThan(a, b, test_precision<int>()); }
inline bool test_isApproxOrLessThan(const int& a, const int& b)
{ return internal::isApproxOrLessThan(a, b, test_precision<int>()); }

inline bool test_isApprox(const float& a, const float& b)
{ return internal::isApprox(a, b, test_precision<float>()); }
inline bool test_isMuchSmallerThan(const float& a, const float& b)
{ return internal::isMuchSmallerThan(a, b, test_precision<float>()); }
inline bool test_isApproxOrLessThan(const float& a, const float& b)
{ return internal::isApproxOrLessThan(a, b, test_precision<float>()); }

inline bool test_isApprox(const double& a, const double& b)
{ return internal::isApprox(a, b, test_precision<double>()); }
inline bool test_isMuchSmallerThan(const double& a, const double& b)
{ return internal::isMuchSmallerThan(a, b, test_precision<double>()); }
inline bool test_isApproxOrLessThan(const double& a, const double& b)
{ return internal::isApproxOrLessThan(a, b, test_precision<double>()); }

#ifndef EIGEN_TEST_NO_COMPLEX
inline bool test_isApprox(const std::complex<float>& a, const std::complex<float>& b)
{ return internal::isApprox(a, b, test_precision<std::complex<float> >()); }
inline bool test_isMuchSmallerThan(const std::complex<float>& a, const std::complex<float>& b)
{ return internal::isMuchSmallerThan(a, b, test_precision<std::complex<float> >()); }

inline bool test_isApprox(const std::complex<double>& a, const std::complex<double>& b)
{ return internal::isApprox(a, b, test_precision<std::complex<double> >()); }
inline bool test_isMuchSmallerThan(const std::complex<double>& a, const std::complex<double>& b)
{ return internal::isMuchSmallerThan(a, b, test_precision<std::complex<double> >()); }

#ifndef EIGEN_TEST_NO_LONGDOUBLE
inline bool test_isApprox(const std::complex<long double>& a, const std::complex<long double>& b)
{ return internal::isApprox(a, b, test_precision<std::complex<long double> >()); }
inline bool test_isMuchSmallerThan(const std::complex<long double>& a, const std::complex<long double>& b)
{ return internal::isMuchSmallerThan(a, b, test_precision<std::complex<long double> >()); }
#endif
#endif

#ifndef EIGEN_TEST_NO_LONGDOUBLE
inline bool test_isApprox(const long double& a, const long double& b)
{
bool ret = internal::isApprox(a, b, test_precision<long double>());
if (!ret) std::cerr
<< std::endl << "    actual   = " << a
<< std::endl << "    expected = " << b << std::endl << std::endl;
return ret;
}

inline bool test_isMuchSmallerThan(const long double& a, const long double& b)
{ return internal::isMuchSmallerThan(a, b, test_precision<long double>()); }
inline bool test_isApproxOrLessThan(const long double& a, const long double& b)
{ return internal::isApproxOrLessThan(a, b, test_precision<long double>()); }
#endif 

inline bool test_isApprox(const half& a, const half& b)
{ return internal::isApprox(a, b, test_precision<half>()); }
inline bool test_isMuchSmallerThan(const half& a, const half& b)
{ return internal::isMuchSmallerThan(a, b, test_precision<half>()); }
inline bool test_isApproxOrLessThan(const half& a, const half& b)
{ return internal::isApproxOrLessThan(a, b, test_precision<half>()); }

template<typename T1,typename T2>
typename NumTraits<typename T1::RealScalar>::NonInteger test_relative_error(const EigenBase<T1> &a, const EigenBase<T2> &b)
{
using std::sqrt;
typedef typename NumTraits<typename T1::RealScalar>::NonInteger RealScalar;
typename internal::nested_eval<T1,2>::type ea(a.derived());
typename internal::nested_eval<T2,2>::type eb(b.derived());
return sqrt(RealScalar((ea-eb).cwiseAbs2().sum()) / RealScalar((std::min)(eb.cwiseAbs2().sum(),ea.cwiseAbs2().sum())));
}

template<typename T1,typename T2>
typename T1::RealScalar test_relative_error(const T1 &a, const T2 &b, const typename T1::Coefficients* = 0)
{
return test_relative_error(a.coeffs(), b.coeffs());
}

template<typename T1,typename T2>
typename T1::Scalar test_relative_error(const T1 &a, const T2 &b, const typename T1::MatrixType* = 0)
{
return test_relative_error(a.matrix(), b.matrix());
}

template<typename S, int D>
S test_relative_error(const Translation<S,D> &a, const Translation<S,D> &b)
{
return test_relative_error(a.vector(), b.vector());
}

template <typename S, int D, int O>
S test_relative_error(const ParametrizedLine<S,D,O> &a, const ParametrizedLine<S,D,O> &b)
{
return (std::max)(test_relative_error(a.origin(), b.origin()), test_relative_error(a.origin(), b.origin()));
}

template <typename S, int D>
S test_relative_error(const AlignedBox<S,D> &a, const AlignedBox<S,D> &b)
{
return (std::max)(test_relative_error((a.min)(), (b.min)()), test_relative_error((a.max)(), (b.max)()));
}

template<typename Derived> class SparseMatrixBase;
template<typename T1,typename T2>
typename T1::RealScalar test_relative_error(const MatrixBase<T1> &a, const SparseMatrixBase<T2> &b)
{
return test_relative_error(a,b.toDense());
}

template<typename Derived> class SparseMatrixBase;
template<typename T1,typename T2>
typename T1::RealScalar test_relative_error(const SparseMatrixBase<T1> &a, const MatrixBase<T2> &b)
{
return test_relative_error(a.toDense(),b);
}

template<typename Derived> class SparseMatrixBase;
template<typename T1,typename T2>
typename T1::RealScalar test_relative_error(const SparseMatrixBase<T1> &a, const SparseMatrixBase<T2> &b)
{
return test_relative_error(a.toDense(),b.toDense());
}

template<typename T1,typename T2>
typename NumTraits<typename NumTraits<T1>::Real>::NonInteger test_relative_error(const T1 &a, const T2 &b, typename internal::enable_if<internal::is_arithmetic<typename NumTraits<T1>::Real>::value, T1>::type* = 0)
{
typedef typename NumTraits<typename NumTraits<T1>::Real>::NonInteger RealScalar;
return numext::sqrt(RealScalar(numext::abs2(a-b))/RealScalar((numext::mini)(numext::abs2(a),numext::abs2(b))));
}

template<typename T>
T test_relative_error(const Rotation2D<T> &a, const Rotation2D<T> &b)
{
return test_relative_error(a.angle(), b.angle());
}

template<typename T>
T test_relative_error(const AngleAxis<T> &a, const AngleAxis<T> &b)
{
return (std::max)(test_relative_error(a.angle(), b.angle()), test_relative_error(a.axis(), b.axis()));
}

template<typename Type1, typename Type2>
inline bool test_isApprox(const Type1& a, const Type2& b, typename Type1::Scalar* = 0) 
{
return a.isApprox(b, test_precision<typename Type1::Scalar>());
}

template<typename T>
typename NumTraits<typename T::Scalar>::Real get_test_precision(const T&, const typename T::Scalar* = 0)
{
return test_precision<typename NumTraits<typename T::Scalar>::Real>();
}

template<typename T>
typename NumTraits<T>::Real get_test_precision(const T&,typename internal::enable_if<internal::is_arithmetic<typename NumTraits<T>::Real>::value, T>::type* = 0)
{
return test_precision<typename NumTraits<T>::Real>();
}

template<typename Type1, typename Type2>
inline bool verifyIsApprox(const Type1& a, const Type2& b)
{
bool ret = test_isApprox(a,b);
if(!ret)
{
std::cerr << "Difference too large wrt tolerance " << get_test_precision(a)  << ", relative error is: " << test_relative_error(a,b) << std::endl;
}
return ret;
}

template<typename Scalar,typename ScalarRef>
inline bool test_isApproxWithRef(const Scalar& a, const Scalar& b, const ScalarRef& ref)
{
return test_isApprox(a+ref, b+ref);
}

template<typename Derived1, typename Derived2>
inline bool test_isMuchSmallerThan(const MatrixBase<Derived1>& m1,
const MatrixBase<Derived2>& m2)
{
return m1.isMuchSmallerThan(m2, test_precision<typename internal::traits<Derived1>::Scalar>());
}

template<typename Derived>
inline bool test_isMuchSmallerThan(const MatrixBase<Derived>& m,
const typename NumTraits<typename internal::traits<Derived>::Scalar>::Real& s)
{
return m.isMuchSmallerThan(s, test_precision<typename internal::traits<Derived>::Scalar>());
}

template<typename Derived>
inline bool test_isUnitary(const MatrixBase<Derived>& m)
{
return m.isUnitary(test_precision<typename internal::traits<Derived>::Scalar>());
}

template<typename T, typename U>
bool test_is_equal(const T& actual, const U& expected, bool expect_equal=true);

template<typename T, typename U>
bool test_is_equal(const T& actual, const U& expected, bool expect_equal)
{
if ((actual==expected) == expect_equal)
return true;
std::cerr
<< "\n    actual   = " << actual
<< "\n    expected " << (expect_equal ? "= " : "!=") << expected << "\n\n";
return false;
}


template<typename MatrixType>
void createRandomPIMatrixOfRank(Index desired_rank, Index rows, Index cols, MatrixType& m);
template<typename MatrixType>
void createRandomPIMatrixOfRank(Index desired_rank, Index rows, Index cols, MatrixType& m)
{
typedef typename internal::traits<MatrixType>::Scalar Scalar;
enum { Rows = MatrixType::RowsAtCompileTime, Cols = MatrixType::ColsAtCompileTime };

typedef Matrix<Scalar, Dynamic, 1> VectorType;
typedef Matrix<Scalar, Rows, Rows> MatrixAType;
typedef Matrix<Scalar, Cols, Cols> MatrixBType;

if(desired_rank == 0)
{
m.setZero(rows,cols);
return;
}

if(desired_rank == 1)
{
m = VectorType::Random(rows).normalized() * VectorType::Random(cols).normalized().transpose();
return;
}

MatrixAType a = MatrixAType::Random(rows,rows);
MatrixType d = MatrixType::Identity(rows,cols);
MatrixBType  b = MatrixBType::Random(cols,cols);

const Index diag_size = (std::min)(d.rows(),d.cols());
if(diag_size != desired_rank)
d.diagonal().segment(desired_rank, diag_size-desired_rank) = VectorType::Zero(diag_size-desired_rank);

HouseholderQR<MatrixAType> qra(a);
HouseholderQR<MatrixBType> qrb(b);
m = qra.householderQ() * d * qrb.householderQ();
}

template<typename PermutationVectorType>
void randomPermutationVector(PermutationVectorType& v, Index size);
template<typename PermutationVectorType>
void randomPermutationVector(PermutationVectorType& v, Index size)
{
typedef typename PermutationVectorType::Scalar Scalar;
v.resize(size);
for(Index i = 0; i < size; ++i) v(i) = Scalar(i);
if(size == 1) return;
for(Index n = 0; n < 3 * size; ++n)
{
Index i = internal::random<Index>(0, size-1);
Index j;
do j = internal::random<Index>(0, size-1); while(j==i);
std::swap(v(i), v(j));
}
}

template<typename T> bool isNotNaN(const T& x)
{
return x==x;
}

template<typename T> bool isPlusInf(const T& x)
{
return x > NumTraits<T>::highest();
}

template<typename T> bool isMinusInf(const T& x)
{
return x < NumTraits<T>::lowest();
}

} 

template<typename T> struct GetDifferentType;

template<> struct GetDifferentType<float> { typedef double type; };
template<> struct GetDifferentType<double> { typedef float type; };
template<typename T> struct GetDifferentType<std::complex<T> >
{ typedef std::complex<typename GetDifferentType<T>::type> type; };

template<typename T> std::string type_name();
template<typename T> std::string type_name()                    { return "other"; }
template<> std::string type_name<float>()                       { return "float"; }
template<> std::string type_name<double>()                      { return "double"; }
template<> std::string type_name<long double>()                 { return "long double"; }
template<> std::string type_name<int>()                         { return "int"; }
template<> std::string type_name<std::complex<float> >()        { return "complex<float>"; }
template<> std::string type_name<std::complex<double> >()       { return "complex<double>"; }
template<> std::string type_name<std::complex<long double> >()  { return "complex<long double>"; }
template<> std::string type_name<std::complex<int> >()          { return "complex<int>"; }

void EIGEN_CAT(test_,EIGEN_TEST_FUNC)();

using namespace Eigen;

inline void set_repeat_from_string(const char *str)
{
errno = 0;
g_repeat = int(strtoul(str, 0, 10));
if(errno || g_repeat <= 0)
{
std::cout << "Invalid repeat value " << str << std::endl;
exit(EXIT_FAILURE);
}
g_has_set_repeat = true;
}

inline void set_seed_from_string(const char *str)
{
errno = 0;
g_seed = int(strtoul(str, 0, 10));
if(errno || g_seed == 0)
{
std::cout << "Invalid seed value " << str << std::endl;
exit(EXIT_FAILURE);
}
g_has_set_seed = true;
}

int main(int argc, char *argv[])
{
g_has_set_repeat = false;
g_has_set_seed = false;
bool need_help = false;

for(int i = 1; i < argc; i++)
{
if(argv[i][0] == 'r')
{
if(g_has_set_repeat)
{
std::cout << "Argument " << argv[i] << " conflicting with a former argument" << std::endl;
return 1;
}
set_repeat_from_string(argv[i]+1);
}
else if(argv[i][0] == 's')
{
if(g_has_set_seed)
{
std::cout << "Argument " << argv[i] << " conflicting with a former argument" << std::endl;
return 1;
}
set_seed_from_string(argv[i]+1);
}
else
{
need_help = true;
}
}

if(need_help)
{
std::cout << "This test application takes the following optional arguments:" << std::endl;
std::cout << "  rN     Repeat each test N times (default: " << DEFAULT_REPEAT << ")" << std::endl;
std::cout << "  sN     Use N as seed for random numbers (default: based on current time)" << std::endl;
std::cout << std::endl;
std::cout << "If defined, the environment variables EIGEN_REPEAT and EIGEN_SEED" << std::endl;
std::cout << "will be used as default values for these parameters." << std::endl;
return 1;
}

char *env_EIGEN_REPEAT = getenv("EIGEN_REPEAT");
if(!g_has_set_repeat && env_EIGEN_REPEAT)
set_repeat_from_string(env_EIGEN_REPEAT);
char *env_EIGEN_SEED = getenv("EIGEN_SEED");
if(!g_has_set_seed && env_EIGEN_SEED)
set_seed_from_string(env_EIGEN_SEED);

if(!g_has_set_seed) g_seed = (unsigned int) time(NULL);
if(!g_has_set_repeat) g_repeat = DEFAULT_REPEAT;

std::cout << "Initializing random number generator with seed " << g_seed << std::endl;
std::stringstream ss;
ss << "Seed: " << g_seed;
g_test_stack.push_back(ss.str());
srand(g_seed);
std::cout << "Repeating each test " << g_repeat << " times" << std::endl;

Eigen::g_test_stack.push_back(std::string(EI_PP_MAKE_STRING(EIGEN_TEST_FUNC)));

EIGEN_CAT(test_,EIGEN_TEST_FUNC)();
return 0;
}

#if defined __INTEL_COMPILER
#pragma warning disable 279 383 1418 1572
#endif

#ifdef _MSC_VER
#pragma warning( disable : 4503)
#endif

#include<iostream>
#include<iomanip>
#include<string>
#include<typeinfo>
#include<vector>
#include<algorithm>

#include "boost/numeric/conversion/converter.hpp"

#ifdef BOOST_BORLANDC
#pragma hdrstop
#endif

#include "test_helpers.cpp"
#include "test_helpers2.cpp"
#include "test_helpers3.cpp"

using namespace std ;
using namespace boost ;
using namespace numeric ;
using namespace MyUDT ;




namespace MyUDT {

typedef conversion_traits<double , MyFloat> MyFloat_to_double_Traits;
typedef conversion_traits<int    , MyFloat> MyFloat_to_int_Traits;
typedef conversion_traits<MyInt  , MyFloat> MyFloat_to_MyInt_Traits;
typedef conversion_traits<int    , MyInt  > MyInt_to_int_Traits;
typedef conversion_traits<MyFloat, MyInt  > MyInt_to_MyFloat_Traits;
typedef conversion_traits<MyInt  , double > double_to_MyInt_Traits;

} 


namespace boost {

namespace numeric {

template<>
struct raw_converter<MyUDT::MyFloat_to_double_Traits>
{
static double low_level_convert ( MyUDT::MyFloat const&  s )
{ return s.to_builtin() ; }
} ;

template<>
struct raw_converter<MyUDT::MyFloat_to_int_Traits>
{
static int low_level_convert ( MyUDT::MyFloat const& s )
{ return static_cast<int>( s.to_builtin() ) ; }
} ;

template<>
struct raw_converter<MyUDT::MyFloat_to_MyInt_Traits>
{
static MyUDT::MyInt low_level_convert ( MyUDT::MyFloat const& s )
{ return MyUDT::MyInt( static_cast<int>(s.to_builtin()) ) ; }
} ;

template<>
struct raw_converter<MyUDT::MyInt_to_int_Traits>
{
static int low_level_convert ( MyUDT::MyInt const& s ) { return s.to_builtin() ; }
} ;

template<>
struct raw_converter<MyUDT::MyInt_to_MyFloat_Traits>
{
static MyUDT::MyFloat low_level_convert ( MyUDT::MyInt const& s )
{
return MyUDT::MyFloat( static_cast<double>(s.to_builtin()) ) ;
}
} ;

template<>
struct raw_converter<MyUDT::double_to_MyInt_Traits>
{
static MyUDT::MyInt low_level_convert ( double s )
{ return MyUDT::MyInt( static_cast<int>(s) ) ; }
} ;

} 

} 




namespace MyUDT {


template<class N> struct get_builtin_type { typedef N type ; } ;
template<> struct get_builtin_type<MyInt>   { typedef int type ; } ;
template<> struct get_builtin_type<MyFloat> { typedef double type ; } ;

template<class N>
struct extract_builtin
{
static N apply ( N n ) { return n ; }
} ;
template<>
struct extract_builtin<MyInt>
{
static int apply ( MyInt const& n ) { return n.to_builtin() ; }
} ;
template<>
struct extract_builtin<MyFloat>
{
static double apply ( MyFloat const& n ) { return n.to_builtin() ; }
} ;

template<class Traits>
struct MyCustomRangeChecker
{
typedef typename Traits::argument_type argument_type ;

typedef typename Traits::source_type S ;
typedef typename Traits::target_type T ;


typedef typename get_builtin_type<S>::type builtinS ;
typedef typename get_builtin_type<T>::type builtinT ;

typedef boost::numeric::converter<builtinT,builtinS> InternalConverter ;

static range_check_result out_of_range ( argument_type s )
{
return InternalConverter::out_of_range( extract_builtin<S>::apply(s) );
}

static void validate_range ( argument_type s )
{
return InternalConverter::validate_range( extract_builtin<S>::apply(s) );
}
} ;

} 









void test_udt_conversions_with_defaults()
{
cout << "Testing UDT conversion with default policies\n" ;


int mibv = rand();
MyInt miv(mibv);
TEST_SUCCEEDING_CONVERSION_DEF(MyInt,int,miv,mibv);
TEST_SUCCEEDING_CONVERSION_DEF(int,MyInt,mibv,miv);


double mfbv = static_cast<double>(rand()) / 3.0 ;
MyFloat mfv (mfbv);
TEST_SUCCEEDING_CONVERSION_DEF(MyFloat,double,mfv,mfbv);
TEST_SUCCEEDING_CONVERSION_DEF(double,MyFloat,mfbv,mfv);


MyInt   miv2  ( static_cast<int>(mfbv) );
MyFloat miv2F ( static_cast<int>(mfbv) );
MyFloat mfv2  ( static_cast<double>(mibv) );
MyInt   mfv2I ( static_cast<double>(mibv) );
TEST_SUCCEEDING_CONVERSION_DEF(MyFloat,MyInt,miv2F,miv2);
TEST_SUCCEEDING_CONVERSION_DEF(MyInt,MyFloat,mfv2I,mfv2);
}

template<class T, class S>
struct GenerateCustomConverter
{
typedef conversion_traits<T,S> Traits;

typedef def_overflow_handler         OverflowHandler ;
typedef Trunc<S>                     Float2IntRounder ;
typedef raw_converter<Traits>        RawConverter ;
typedef MyCustomRangeChecker<Traits> RangeChecker ;

typedef converter<T,S,Traits,OverflowHandler,Float2IntRounder,RawConverter,RangeChecker> type ;
} ;

void test_udt_conversions_with_custom_range_checking()
{
cout << "Testing UDT conversions with custom range checker\n" ;

int mibv = rand();
MyFloat mfv ( static_cast<double>(mibv) );

typedef GenerateCustomConverter<MyFloat,int>::type int_to_MyFloat_Conv ;

TEST_SUCCEEDING_CONVERSION( int_to_MyFloat_Conv, MyFloat, int, mfv, mibv );

int mibv2 = rand();
MyInt miv (mibv2);
MyFloat mfv2 ( static_cast<double>(mibv2) );

typedef GenerateCustomConverter<MyFloat,MyInt>::type MyInt_to_MyFloat_Conv ;

TEST_SUCCEEDING_CONVERSION( MyInt_to_MyFloat_Conv, MyFloat, MyInt, mfv2, miv );

double mfbv = bounds<double>::highest();
typedef GenerateCustomConverter<MyInt,double>::type double_to_MyInt_Conv ;

TEST_POS_OVERFLOW_CONVERSION( double_to_MyInt_Conv, MyInt, double, mfbv );

MyFloat mfv3 ( bounds<double>::lowest() ) ;
typedef GenerateCustomConverter<int,MyFloat>::type MyFloat_to_int_Conv ;

TEST_NEG_OVERFLOW_CONVERSION( MyFloat_to_int_Conv, int, MyFloat, mfv3 );
}


int test_main( int, char* [] )
{
cout << setprecision( numeric_limits<long double>::digits10 ) ;

test_udt_conversions_with_defaults();
test_udt_conversions_with_custom_range_checking();

return 0;
}







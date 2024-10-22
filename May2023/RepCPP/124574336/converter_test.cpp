
#include<cstdlib>
#include<iostream>
#include<iomanip>
#include<string>
#include<typeinfo>
#include<vector>
#include<algorithm>

#include "boost/config.hpp"
#include "boost/cstdint.hpp"
#include "boost/utility.hpp"

#if BOOST_WORKAROUND(BOOST_BORLANDC, <= 0x551)
namespace std
{

inline float       ceil  (float       x) { return std::ceil  ( static_cast<double>(x)); }
inline float       floor (float       x) { return std::floor ( static_cast<double>(x)); }
inline long double ceil  (long double x) { return std::ceill (x); }
inline long double floor (long double x) { return std::floorl(x); }

} 
#endif

#include "boost/numeric/conversion/converter.hpp"
#include "boost/numeric/conversion/cast.hpp"

#ifdef BOOST_BORLANDC
#pragma hdrstop
#endif

#include "test_helpers.cpp"
#include "test_helpers2.cpp"
#include "test_helpers3.cpp"

#include "boost/mpl/alias.hpp"

using std::cout ;

template<class N> inline N absG ( N v )
{
return v < static_cast<N>(0) ? static_cast<N>(-v) : v ;
}
template<> inline unsigned char  absG<unsigned char>  ( unsigned char  v ) { return v ; }
template<> inline unsigned short absG<unsigned short> ( unsigned short v ) { return v ; }
template<> inline unsigned int   absG<unsigned int>   ( unsigned int   v ) { return v ; }
template<> inline unsigned long  absG<unsigned long>  ( unsigned long  v ) { return v ; }

template<class T> inline void unused_variable ( T const& ) {}
void test_conversions()
{
using namespace boost ;
using namespace numeric ;

#if !defined(BOOST_NO_STDC_NAMESPACE)
using std::rand ;
#endif

boost::int16_t v16 ;
boost::uint16_t uv16 ;
boost::int32_t v32 ;
boost::uint32_t uv32 ;

volatile float  fv ; 
volatile double dv ; 

cout << "Testing representative conversions\n";



v16 = static_cast<boost::int16_t>(rand());
TEST_SUCCEEDING_CONVERSION_DEF(boost::int32_t,boost::int16_t,v16,v16);

v16 = static_cast<boost::int16_t>(rand());
TEST_SUCCEEDING_CONVERSION_DEF(boost::int16_t,boost::int32_t,v16,v16);
TEST_POS_OVERFLOW_CONVERSION_DEF(boost::int16_t,boost::int32_t,bounds<boost::int16_t>::highest() + boost::int32_t(1) ) ;
TEST_NEG_OVERFLOW_CONVERSION_DEF(boost::int16_t,boost::int32_t,bounds<boost::int16_t>::lowest()  - boost::int32_t(1) ) ;


v32 = absG(static_cast<boost::int32_t>(rand()));
v16 = absG(static_cast<boost::int16_t>(rand()));
TEST_SUCCEEDING_CONVERSION_DEF(boost::uint32_t,boost::int32_t,v32,v32);
TEST_SUCCEEDING_CONVERSION_DEF(boost::uint16_t,boost::int32_t,v16,v16);
TEST_POS_OVERFLOW_CONVERSION_DEF(boost::uint16_t,boost::int32_t,bounds<boost::uint16_t>::highest() + boost::int32_t(1) ) ;
TEST_NEG_OVERFLOW_CONVERSION_DEF(boost::uint32_t,boost::int32_t,boost::int32_t(-1) ) ;


v32 = absG(static_cast<boost::int32_t>(rand()));
TEST_SUCCEEDING_CONVERSION_DEF(boost::int32_t,boost::uint32_t,v32,v32);

v16 = absG(static_cast<boost::int16_t>(rand()));
TEST_SUCCEEDING_CONVERSION_DEF(boost::int16_t,boost::uint32_t,v16,v16);
TEST_POS_OVERFLOW_CONVERSION_DEF(boost::int32_t,boost::uint32_t,bounds<boost::uint32_t>::highest() ) ;
TEST_POS_OVERFLOW_CONVERSION_DEF(boost::int16_t,boost::uint32_t,bounds<boost::uint32_t>::highest() ) ;


uv16 = static_cast<boost::uint16_t>(rand());
TEST_SUCCEEDING_CONVERSION_DEF(boost::uint32_t,boost::uint16_t,uv16,uv16);

uv16 = static_cast<boost::uint16_t>(rand());
TEST_SUCCEEDING_CONVERSION_DEF(boost::uint16_t,boost::uint32_t,uv16,uv16);
TEST_POS_OVERFLOW_CONVERSION_DEF(boost::uint16_t,boost::uint32_t,bounds<boost::uint32_t>::highest() ) ;


v32 = static_cast<boost::int32_t>(rand());
TEST_SUCCEEDING_CONVERSION_DEF(double,boost::int32_t,v32,v32);

uv32 = static_cast<boost::uint32_t>(rand());
TEST_SUCCEEDING_CONVERSION_DEF(double,boost::uint32_t,uv32,uv32);


v32 =  static_cast<boost::int32_t>(rand());
TEST_SUCCEEDING_CONVERSION_DEF(boost::int32_t,double,v32,v32);

dv = static_cast<double>(bounds<boost::uint32_t>::highest()) + 1.0 ;
TEST_POS_OVERFLOW_CONVERSION_DEF(boost::int32_t,double,dv) ;
TEST_NEG_OVERFLOW_CONVERSION_DEF(boost::int32_t,double,-dv) ;


fv = static_cast<float>(rand()) / static_cast<float>(3) ;
TEST_SUCCEEDING_CONVERSION_DEF(double,float,fv,fv);


fv = static_cast<float>(rand()) / static_cast<float>(3) ;
TEST_SUCCEEDING_CONVERSION_DEF(float,double,fv,fv);
TEST_POS_OVERFLOW_CONVERSION_DEF(float,double,bounds<double>::highest()) ;
TEST_NEG_OVERFLOW_CONVERSION_DEF(float,double,bounds<double>::lowest ()) ;
}

struct custom_overflow_handler
{
void operator() ( boost::numeric::range_check_result r )
{
if ( r == boost::numeric::cNegOverflow )
cout << "negative_overflow detected!\n" ;
else if ( r == boost::numeric::cPosOverflow )
cout << "positive_overflow detected!\n" ;
}
} ;

template<class T, class S,class OverflowHandler>
void test_overflow_handler( MATCH_FNTPL_ARG(T), MATCH_FNTPL_ARG(S), MATCH_FNTPL_ARG(OverflowHandler),
PostCondition pos,
PostCondition neg
)
{
typedef boost::numeric::conversion_traits<T,S> traits ;
typedef boost::numeric::converter<T,S,traits,OverflowHandler> converter ;

static const S psrc = boost::numeric::bounds<S>::highest();
static const S nsrc = boost::numeric::bounds<S>::lowest ();

static const T pres = static_cast<T>(psrc);
static const T nres = static_cast<T>(nsrc);

test_conv_base ( ConversionInstance<converter>(pres,psrc,pos) ) ;
test_conv_base ( ConversionInstance<converter>(nres,nsrc,neg) ) ;
}

template<class T, class S>
void test_overflow_handlers( MATCH_FNTPL_ARG(T), MATCH_FNTPL_ARG(S) )
{
cout << "Testing Silent Overflow Handler policy\n";

test_overflow_handler( SET_FNTPL_ARG(T),
SET_FNTPL_ARG(S),
SET_FNTPL_ARG(boost::numeric::silent_overflow_handler),
c_converted,
c_converted
) ;

cout << "Testing Default Overflow Handler policy\n";

test_overflow_handler( SET_FNTPL_ARG(T),
SET_FNTPL_ARG(S),
SET_FNTPL_ARG(boost::numeric::def_overflow_handler),
c_pos_overflow,
c_neg_overflow
) ;

cout << "Testing Custom (User-Defined) Overflow Handler policy\n";

test_overflow_handler( SET_FNTPL_ARG(T),
SET_FNTPL_ARG(S),
SET_FNTPL_ARG(custom_overflow_handler),
c_converted,
c_converted
) ;
}

template<class T, class S, class Float2IntRounder>
void test_rounding_conversion ( MATCH_FNTPL_ARG(T), MATCH_FNTPL_ARG(Float2IntRounder),
S s,
PostCondition resl1,
PostCondition resl0,
PostCondition res,
PostCondition resr0,
PostCondition resr1
)
{
typedef boost::numeric::conversion_traits<T,S> Traits ;

typedef boost::numeric::converter<T,S, Traits, boost::numeric::def_overflow_handler,Float2IntRounder>
Converter ;

S sl1 = s - static_cast<S>(1);
S sl0 = s - static_cast<S>(0.5);
S sr0 = s + static_cast<S>(0.5);
S sr1 = s + static_cast<S>(1);

T tl1 = static_cast<T>( Converter::nearbyint(sl1) );
T tl0 = static_cast<T>( Converter::nearbyint(sl0) );
T t   = static_cast<T>( Converter::nearbyint(s)   );
T tr0 = static_cast<T>( Converter::nearbyint(sr0) );
T tr1 = static_cast<T>( Converter::nearbyint(sr1) );

test_conv_base ( ConversionInstance<Converter>(tl1,sl1,resl1) ) ;
test_conv_base ( ConversionInstance<Converter>(tl0,sl0,resl0) ) ;
test_conv_base ( ConversionInstance<Converter>(t,s,res) ) ;
test_conv_base ( ConversionInstance<Converter>(tr0,sr0,resr0) ) ;
test_conv_base ( ConversionInstance<Converter>(tr1,sr1,resr1) ) ;
}


template<class T,class S>
void test_round_style( MATCH_FNTPL_ARG(T), MATCH_FNTPL_ARG(S) )
{
S min = boost::numeric::bounds<T>::lowest();
S max = boost::numeric::bounds<T>::highest();

cout << "Testing 'Trunc' Float2IntRounder policy\n";

test_rounding_conversion(SET_FNTPL_ARG(T),
SET_FNTPL_ARG(boost::numeric::Trunc<S>),
min,
c_neg_overflow,
c_converted,
c_converted,
c_converted,
c_converted
) ;

test_rounding_conversion(SET_FNTPL_ARG(T),
SET_FNTPL_ARG(boost::numeric::Trunc<S>),
max,
c_converted,
c_converted,
c_converted,
c_converted,
c_pos_overflow
) ;

cout << "Testing 'RoundEven' Float2IntRounder policy\n";

test_rounding_conversion(SET_FNTPL_ARG(T),
SET_FNTPL_ARG(boost::numeric::RoundEven<S>),
min,
c_neg_overflow,
c_converted,
c_converted,
c_converted,
c_converted
) ;

test_rounding_conversion(SET_FNTPL_ARG(T),
SET_FNTPL_ARG(boost::numeric::RoundEven<S>),
max,
c_converted,
c_converted,
c_converted,
c_pos_overflow,
c_pos_overflow
) ;

cout << "Testing 'Ceil' Float2IntRounder policy\n";

test_rounding_conversion(SET_FNTPL_ARG(T),
SET_FNTPL_ARG(boost::numeric::Ceil<S>),
min,
c_neg_overflow,
c_converted,
c_converted,
c_converted,
c_converted
) ;

test_rounding_conversion(SET_FNTPL_ARG(T),
SET_FNTPL_ARG(boost::numeric::Ceil<S>),
max,
c_converted,
c_converted,
c_converted,
c_pos_overflow,
c_pos_overflow
) ;

cout << "Testing 'Floor' Float2IntRounder policy\n" ;

test_rounding_conversion(SET_FNTPL_ARG(T),
SET_FNTPL_ARG(boost::numeric::Floor<S>),
min,
c_neg_overflow,
c_neg_overflow,
c_converted,
c_converted,
c_converted
) ;

test_rounding_conversion(SET_FNTPL_ARG(T),
SET_FNTPL_ARG(boost::numeric::Floor<S>),
max,
c_converted,
c_converted,
c_converted,
c_converted,
c_pos_overflow
) ;

}

void test_round_even( double n, double x )
{
double r = boost::numeric::RoundEven<double>::nearbyint(n);
BOOST_CHECK( r == x ) ;
}

void test_round_even()
{
cout << "Testing 'RoundEven' tie-breaking\n";

double min = boost::numeric::bounds<double>::lowest();
double max = boost::numeric::bounds<double>::highest();

#if !defined(BOOST_NO_STDC_NAMESPACE)
using std::floor ;
using std::ceil ;
#endif
test_round_even(min, floor(min));
test_round_even(max, ceil (max));
test_round_even(2.0, 2.0);
test_round_even(2.3, 2.0);
test_round_even(2.5, 2.0);
test_round_even(2.7, 3.0);
test_round_even(3.0, 3.0);
test_round_even(3.3, 3.0);
test_round_even(3.5, 4.0);
test_round_even(3.7, 4.0);
}

int double_to_int ( double n ) { return static_cast<int>(n) ; }

void test_converter_as_function_object()
{
cout << "Testing converter as function object.\n";

std::vector<double> S ;
for ( int i = 0 ; i < 10 ; ++ i )
S.push_back( i * ( 18.0 / 19.0 ) );

std::vector<int> W ;
std::transform(S.begin(),S.end(),std::back_inserter(W),double_to_int);

std::vector<int> I ;
std::transform(S.begin(),
S.end(),
std::back_inserter(I),
boost::numeric::converter<int,double>()
) ;

bool double_to_int_OK = std::equal(W.begin(),W.end(),I.begin()) ;
BOOST_CHECK_MESSAGE(double_to_int_OK, "converter (int,double) as function object");

std::vector<double> D ;
std::transform(S.begin(),
S.end(),
std::back_inserter(D),
boost::numeric::converter<double,double>()
) ;

bool double_to_double_OK = std::equal(S.begin(),S.end(),D.begin()) ;
BOOST_CHECK_MESSAGE(double_to_double_OK, "converter (double,double) as function object");
}

#if BOOST_WORKAROUND(__IBMCPP__, <= 600 ) 
#  define UNOPTIMIZED
#else
#  define UNOPTIMIZED volatile
#endif

void test_optimizations()
{
using namespace boost;
using namespace numeric;

float fv0 = 18.0f / 19.0f ;


UNOPTIMIZED float fv1a = fv0 ;

float fv1b = numeric_cast<float>(fv0);
unused_variable(fv1a);
unused_variable(fv1b);

UNOPTIMIZED double dv1a = static_cast<double>(fv0);

double dv1b = numeric_cast<double>(fv0);
unused_variable(dv1a);
unused_variable(dv1b);



{
double const& s = dv1b ;
range_check_result r =  s < static_cast<double>(bounds<float>::lowest())
? cNegOverflow : cInRange ;
if ( r == cInRange )
{
r = s > static_cast<double>(bounds<float>::highest()) ? cPosOverflow : cInRange ;
}
if ( r == cNegOverflow )
throw negative_overflow() ;
else if ( r == cPosOverflow )
throw positive_overflow() ;
UNOPTIMIZED float fv2a = static_cast<float>(s);
unused_variable(fv2a);
}

float fv2b = numeric_cast<float>(dv1b);
unused_variable(fv2b);



{
double const& s = dv1b ;
range_check_result r = s <= static_cast<double>(bounds<int>::lowest()) - static_cast<double>(1.0)
? cNegOverflow : cInRange ;
if ( r == cInRange )
{
r = s >= static_cast<double>(bounds<int>::highest()) + static_cast<double>(1.0)
? cPosOverflow : cInRange ;
}
if ( r == cNegOverflow )
throw negative_overflow() ;
else if ( r == cPosOverflow )
throw positive_overflow() ;

#if !defined(BOOST_NO_STDC_NAMESPACE)
using std::floor ;
#endif

double s1 = floor(dv1b + 0.5);

UNOPTIMIZED int iv1a = static_cast<int>(s1);
unused_variable(iv1a);
}

int iv1b = numeric_cast<int>(dv1b);
unused_variable(iv1b);
}

int test_main( int, char* argv[] )
{
std::cout << std::setprecision( std::numeric_limits<long double>::digits10 ) ;

test_conversions();
test_overflow_handlers( SET_FNTPL_ARG(boost::int16_t), SET_FNTPL_ARG(boost::int32_t));
test_round_style(SET_FNTPL_ARG(boost::int32_t), SET_FNTPL_ARG(double) ) ;
test_round_even() ;
test_converter_as_function_object();
test_optimizations() ;

return 0;
}


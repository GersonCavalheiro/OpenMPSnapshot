#ifndef _DATE_TIME_INT_ADAPTER_HPP__
#define _DATE_TIME_INT_ADAPTER_HPP__




#include "boost/config.hpp"
#include "boost/limits.hpp" 
#include "boost/date_time/special_defs.hpp"
#include "boost/date_time/locale_config.hpp"
#ifndef BOOST_DATE_TIME_NO_LOCALE
#  include <ostream>
#endif

#if defined(BOOST_MSVC)
#pragma warning(push)
#pragma warning(disable: 4127)
#endif

namespace boost {
namespace date_time {



template<typename int_type_>
class int_adapter {
public:
typedef int_type_ int_type;
BOOST_CXX14_CONSTEXPR int_adapter(int_type v) :
value_(v)
{}
static BOOST_CONSTEXPR bool has_infinity()
{
return  true;
}
static BOOST_CONSTEXPR int_adapter  pos_infinity()
{
return (::std::numeric_limits<int_type>::max)();
}
static BOOST_CONSTEXPR int_adapter  neg_infinity()
{
return (::std::numeric_limits<int_type>::min)();
}
static BOOST_CONSTEXPR int_adapter  not_a_number()
{
return (::std::numeric_limits<int_type>::max)()-1;
}
static BOOST_CONSTEXPR int_adapter max BOOST_PREVENT_MACRO_SUBSTITUTION ()
{
return (::std::numeric_limits<int_type>::max)()-2;
}
static BOOST_CONSTEXPR int_adapter min BOOST_PREVENT_MACRO_SUBSTITUTION ()
{
return (::std::numeric_limits<int_type>::min)()+1;
}
static BOOST_CXX14_CONSTEXPR int_adapter from_special(special_values sv)
{
switch (sv) {
case not_a_date_time: return not_a_number();
case neg_infin:       return neg_infinity();
case pos_infin:       return pos_infinity();
case max_date_time:   return (max)();
case min_date_time:   return (min)();
default:              return not_a_number();
}
}
static BOOST_CONSTEXPR bool is_inf(int_type v)
{
return (v == neg_infinity().as_number() ||
v == pos_infinity().as_number());
}
static BOOST_CXX14_CONSTEXPR bool is_neg_inf(int_type v)
{
return (v == neg_infinity().as_number());
}
static BOOST_CXX14_CONSTEXPR bool is_pos_inf(int_type v)
{
return (v == pos_infinity().as_number());
}
static BOOST_CXX14_CONSTEXPR bool is_not_a_number(int_type v)
{
return (v == not_a_number().as_number());
}
static BOOST_CXX14_CONSTEXPR special_values to_special(int_type v)
{
if (is_not_a_number(v)) return not_a_date_time;
if (is_neg_inf(v)) return neg_infin;
if (is_pos_inf(v)) return pos_infin;
return not_special;
}

static BOOST_CONSTEXPR int_type maxcount()
{
return (::std::numeric_limits<int_type>::max)()-3;
}
BOOST_CONSTEXPR bool is_infinity() const
{
return (value_ == neg_infinity().as_number() ||
value_ == pos_infinity().as_number());
}
BOOST_CONSTEXPR bool is_pos_infinity()const
{
return(value_ == pos_infinity().as_number());
}
BOOST_CONSTEXPR bool is_neg_infinity()const
{
return(value_ == neg_infinity().as_number());
}
BOOST_CONSTEXPR bool is_nan() const
{
return (value_ == not_a_number().as_number());
}
BOOST_CONSTEXPR bool is_special() const
{
return(is_infinity() || is_nan()); 
}
BOOST_CONSTEXPR bool operator==(const int_adapter& rhs) const
{
return (compare(rhs) == 0);
}
BOOST_CXX14_CONSTEXPR bool operator==(const int& rhs) const
{
if(!std::numeric_limits<int_type>::is_signed)
{
if(is_neg_inf(value_) && rhs == 0)
{
return false;
}
}
return (compare(rhs) == 0);
}
BOOST_CONSTEXPR bool operator!=(const int_adapter& rhs) const
{
return (compare(rhs) != 0);
}
BOOST_CXX14_CONSTEXPR bool operator!=(const int& rhs) const
{
if(!std::numeric_limits<int_type>::is_signed)
{
if(is_neg_inf(value_) && rhs == 0)
{
return true;
}
}
return (compare(rhs) != 0);
}
BOOST_CONSTEXPR bool operator<(const int_adapter& rhs) const
{
return (compare(rhs) == -1);
}
BOOST_CXX14_CONSTEXPR bool operator<(const int& rhs) const
{
if(!std::numeric_limits<int_type>::is_signed)
{
if(is_neg_inf(value_) && rhs == 0)
{
return true;
}
}
return (compare(rhs) == -1);
}
BOOST_CONSTEXPR bool operator>(const int_adapter& rhs) const
{
return (compare(rhs) == 1);
}
BOOST_CONSTEXPR int_type as_number() const
{
return value_;
}
BOOST_CONSTEXPR special_values as_special() const
{
return int_adapter::to_special(value_);
}


template<class rhs_type>
BOOST_CXX14_CONSTEXPR
int_adapter operator+(const int_adapter<rhs_type>& rhs) const
{
if(is_special() || rhs.is_special())
{
if (is_nan() || rhs.is_nan()) 
{
return int_adapter::not_a_number();
}
if((is_pos_inf(value_) && rhs.is_neg_inf(rhs.as_number())) ||
(is_neg_inf(value_) && rhs.is_pos_inf(rhs.as_number())) )
{
return int_adapter::not_a_number();
}
if (is_infinity()) 
{
return *this;
}
if (rhs.is_pos_inf(rhs.as_number())) 
{
return int_adapter::pos_infinity();
}
if (rhs.is_neg_inf(rhs.as_number())) 
{
return int_adapter::neg_infinity();
}
}
return int_adapter<int_type>(value_ + static_cast<int_type>(rhs.as_number()));
}

BOOST_CXX14_CONSTEXPR
int_adapter operator+(const int_type rhs) const
{
if(is_special())
{
if (is_nan()) 
{
return int_adapter<int_type>(not_a_number());
}
if (is_infinity()) 
{
return *this;
}
}
return int_adapter<int_type>(value_ + rhs);
}


template<class rhs_type>
BOOST_CXX14_CONSTEXPR
int_adapter operator-(const int_adapter<rhs_type>& rhs)const
{
if(is_special() || rhs.is_special())
{
if (is_nan() || rhs.is_nan()) 
{
return int_adapter::not_a_number();
}
if((is_pos_inf(value_) && rhs.is_pos_inf(rhs.as_number())) ||
(is_neg_inf(value_) && rhs.is_neg_inf(rhs.as_number())) )
{
return int_adapter::not_a_number();
}
if (is_infinity()) 
{
return *this;
}
if (rhs.is_pos_inf(rhs.as_number())) 
{
return int_adapter::neg_infinity();
}
if (rhs.is_neg_inf(rhs.as_number())) 
{
return int_adapter::pos_infinity();
}
}
return int_adapter<int_type>(value_ - static_cast<int_type>(rhs.as_number()));
}

BOOST_CXX14_CONSTEXPR
int_adapter operator-(const int_type rhs) const
{
if(is_special())
{
if (is_nan()) 
{
return int_adapter<int_type>(not_a_number());
}
if (is_infinity()) 
{
return *this;
}
}
return int_adapter<int_type>(value_ - rhs);
}

BOOST_CXX14_CONSTEXPR
int_adapter operator*(const int_adapter& rhs)const
{
if(this->is_special() || rhs.is_special())
{
return mult_div_specials(rhs);
}
return int_adapter<int_type>(value_ * rhs.value_);
}


BOOST_CXX14_CONSTEXPR
int_adapter operator*(const int rhs) const
{
if(is_special())
{
return mult_div_specials(rhs);
}
return int_adapter<int_type>(value_ * rhs);
}

BOOST_CXX14_CONSTEXPR
int_adapter operator/(const int_adapter& rhs)const
{
if(this->is_special() || rhs.is_special())
{
if(is_infinity() && rhs.is_infinity())
{
return int_adapter<int_type>(not_a_number());
}
if(rhs != 0)
{
return mult_div_specials(rhs);
}
else { 
return int_adapter<int_type>(value_ / rhs.value_); 
}
}
return int_adapter<int_type>(value_ / rhs.value_);
}


BOOST_CXX14_CONSTEXPR
int_adapter operator/(const int rhs) const
{
if(is_special() && rhs != 0)
{
return mult_div_specials(rhs);
}
return int_adapter<int_type>(value_ / rhs); 
}

BOOST_CXX14_CONSTEXPR
int_adapter operator%(const int_adapter& rhs)const
{
if(this->is_special() || rhs.is_special())
{
if(is_infinity() && rhs.is_infinity())
{
return int_adapter<int_type>(not_a_number());
}
if(rhs != 0)
{
return mult_div_specials(rhs);
}
else { 
return int_adapter<int_type>(value_ % rhs.value_); 
}
}
return int_adapter<int_type>(value_ % rhs.value_);
}


BOOST_CXX14_CONSTEXPR
int_adapter operator%(const int rhs) const
{
if(is_special() && rhs != 0)
{
return mult_div_specials(rhs);
}
return int_adapter<int_type>(value_ % rhs); 
}

private:
int_type value_;

BOOST_CXX14_CONSTEXPR
int compare( const int_adapter& rhs ) const
{
if(this->is_special() || rhs.is_special())
{
if(this->is_nan() || rhs.is_nan()) {
if(this->is_nan() && rhs.is_nan()) {
return 0; 
}
else {
return 2; 
}
}
if((is_neg_inf(value_) && !is_neg_inf(rhs.value_)) ||
(is_pos_inf(rhs.value_) && !is_pos_inf(value_)) )
{
return -1; 
}
if((is_pos_inf(value_) && !is_pos_inf(rhs.value_)) ||
(is_neg_inf(rhs.value_) && !is_neg_inf(value_)) ) {
return 1; 
}
}
if(value_ < rhs.value_) return -1;
if(value_ > rhs.value_) return 1;
return 0;
}


BOOST_CXX14_CONSTEXPR
int_adapter mult_div_specials(const int_adapter& rhs) const
{
if(this->is_nan() || rhs.is_nan()) {
return int_adapter<int_type>(not_a_number());
}
BOOST_CONSTEXPR_OR_CONST int min_value = std::numeric_limits<int_type>::is_signed ? 0 : 1; 
if((*this > 0 && rhs > 0) || (*this < min_value && rhs < min_value)) {
return int_adapter<int_type>(pos_infinity());
}
if((*this > 0 && rhs < min_value) || (*this < min_value && rhs > 0)) {
return int_adapter<int_type>(neg_infinity());
}
return int_adapter<int_type>(not_a_number());
}


BOOST_CXX14_CONSTEXPR
int_adapter mult_div_specials(const int& rhs) const
{
if(this->is_nan()) {
return int_adapter<int_type>(not_a_number());
}
BOOST_CONSTEXPR_OR_CONST int min_value = std::numeric_limits<int_type>::is_signed ? 0 : 1; 
if((*this > 0 && rhs > 0) || (*this < min_value && rhs < 0)) {
return int_adapter<int_type>(pos_infinity());
}
if((*this > 0 && rhs < 0) || (*this < min_value && rhs > 0)) {
return int_adapter<int_type>(neg_infinity());
}
return int_adapter<int_type>(not_a_number());
}

};

#ifndef BOOST_DATE_TIME_NO_LOCALE

template<class charT, class traits, typename int_type>
inline
std::basic_ostream<charT, traits>& 
operator<<(std::basic_ostream<charT, traits>& os, const int_adapter<int_type>& ia)
{
if(ia.is_special()) {
switch(ia.as_special())
{
case not_a_date_time:
os << "not-a-number";
break;
case pos_infin:
os << "+infinity";
break;
case neg_infin:
os << "-infinity";
break;
default:
os << "";
}
}
else {
os << ia.as_number(); 
}
return os;
}
#endif


} } 

#if defined(BOOST_MSVC)
#pragma warning(pop)
#endif

#endif

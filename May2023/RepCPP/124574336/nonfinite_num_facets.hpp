#ifndef BOOST_MATH_NONFINITE_NUM_FACETS_HPP
#define BOOST_MATH_NONFINITE_NUM_FACETS_HPP





#include <cstring>
#include <ios>
#include <limits>
#include <locale>

#include <boost/version.hpp>
#include <boost/throw_exception.hpp>

#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/math/special_functions/sign.hpp>

#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable : 4127) 
#  pragma warning(disable : 4706) 
#endif

namespace boost {
namespace math {


const int legacy = 0x1; 
const int signed_zero = 0x2; 
const int trap_infinity = 0x4; 
const int trap_nan = 0x8; 


template<
class CharType,
class OutputIterator = std::ostreambuf_iterator<CharType>
>
class nonfinite_num_put : public std::num_put<CharType, OutputIterator>
{
public:
explicit nonfinite_num_put(int flags = 0) : flags_(flags) {}

protected:
virtual OutputIterator do_put(
OutputIterator it, std::ios_base& iosb, CharType fill, double val) const
{
put_and_reset_width(it, iosb, fill, val);
return it;
}

virtual OutputIterator do_put(
OutputIterator it, std::ios_base& iosb,  CharType fill, long double val) const
{
put_and_reset_width(it, iosb, fill, val);
return it;
}

private:
template<class ValType> void put_and_reset_width(
OutputIterator& it, std::ios_base& iosb,
CharType fill, ValType val) const
{
put_impl(it, iosb, fill, val);
iosb.width(0);
}

template<class ValType> void put_impl(
OutputIterator& it, std::ios_base& iosb,
CharType fill, ValType val) const
{
static const CharType prefix_plus[2] = { '+', '\0' };
static const CharType prefix_minus[2] = { '-', '\0' };
static const CharType body_inf[4] = { 'i', 'n', 'f', '\0' };
static const CharType body_nan[4] = { 'n', 'a', 'n', '\0' };
static const CharType* null_string = 0;

switch((boost::math::fpclassify)(val))
{

case FP_INFINITE:
if(flags_ & trap_infinity)
{
BOOST_THROW_EXCEPTION(std::ios_base::failure("Infinity"));
}
else if((boost::math::signbit)(val))
{ 
put_num_and_fill(it, iosb, prefix_minus, body_inf, fill, val);
}
else if(iosb.flags() & std::ios_base::showpos)
{ 
put_num_and_fill(it, iosb, prefix_plus, body_inf, fill, val);
}
else
{ 
put_num_and_fill(it, iosb, null_string, body_inf, fill, val);
}
break;

case FP_NAN:
if(flags_ & trap_nan)
{
BOOST_THROW_EXCEPTION(std::ios_base::failure("NaN"));
}
else if((boost::math::signbit)(val))
{ 
put_num_and_fill(it, iosb, prefix_minus, body_nan, fill, val);
}
else if(iosb.flags() & std::ios_base::showpos)
{ 
put_num_and_fill(it, iosb, prefix_plus, body_nan, fill, val);
}
else
{ 
put_num_and_fill(it, iosb, null_string, body_nan, fill, val);
}
break;

case FP_ZERO:
if((flags_ & signed_zero) && ((boost::math::signbit)(val)))
{ 

std::basic_ostringstream<CharType> zeros; 

zeros.flags(iosb.flags());
zeros.unsetf(std::ios::showpos); 
zeros.precision(iosb.precision());
zeros.fill(static_cast<char>(fill));
zeros << ValType(0);
put_num_and_fill(it, iosb, prefix_minus, zeros.str().c_str(), fill, val);
}
else
{ 
put_num_and_fill(it, iosb, null_string, null_string, fill, val);
}
break;

default:  
it = std::num_put<CharType, OutputIterator>::do_put(it, iosb, fill, val);
break;
}
}

template<class ValType>
void put_num_and_fill(
OutputIterator& it, std::ios_base& iosb, const CharType* prefix,
const CharType* body, CharType fill, ValType val) const
{
int prefix_length = prefix ? (int)std::char_traits<CharType>::length(prefix) : 0;
int body_length = body ? (int)std::char_traits<CharType>::length(body) : 0;
int width = prefix_length + body_length;
std::ios_base::fmtflags adjust = iosb.flags() & std::ios_base::adjustfield;
const std::ctype<CharType>& ct
= std::use_facet<std::ctype<CharType> >(iosb.getloc());

if(body || prefix)
{ 
if(adjust != std::ios_base::internal && adjust != std::ios_base::left)
put_fill(it, iosb, fill, width);
}

if(prefix)
{ 
while(*prefix)
*it = *(prefix++);
iosb.width( iosb.width() - prefix_length );
width -= prefix_length;
}

if(body)
{ 
if(adjust == std::ios_base::internal)
{ 
put_fill(it, iosb, fill, width);
}
if(iosb.flags() & std::ios_base::uppercase)
{
while(*body)
*it = ct.toupper(*(body++));
}
else
{
while(*body)
*it = *(body++);
}

if(adjust == std::ios_base::left)
put_fill(it, iosb, fill, width);
}
else
{
it = std::num_put<CharType, OutputIterator>::do_put(it, iosb, fill, val);
}
}

void put_fill(
OutputIterator& it, std::ios_base& iosb, CharType fill, int width) const
{ 
for(std::streamsize i = iosb.width() - static_cast<std::streamsize>(width); i > 0; --i)
*it = fill;
}

private:
const int flags_;
};



template<
class CharType,
class InputIterator = std::istreambuf_iterator<CharType>
>
class nonfinite_num_get : public std::num_get<CharType, InputIterator>
{

public:
explicit nonfinite_num_get(int flags = 0) : flags_(flags)
{}

protected:  
virtual InputIterator do_get(
InputIterator it, InputIterator end, std::ios_base& iosb,
std::ios_base::iostate& state, float& val) const
{
get_and_check_eof(it, end, iosb, state, val);
return it;
}

virtual InputIterator do_get(
InputIterator it, InputIterator end, std::ios_base& iosb,
std::ios_base::iostate& state, double& val) const
{
get_and_check_eof(it, end, iosb, state, val);
return it;
}

virtual InputIterator do_get(
InputIterator it, InputIterator end, std::ios_base& iosb,
std::ios_base::iostate& state, long double& val) const
{
get_and_check_eof(it, end, iosb, state, val);
return it;
}


private:
template<class ValType> static ValType positive_nan()
{
return (boost::math::copysign)(
std::numeric_limits<ValType>::quiet_NaN(), static_cast<ValType>(1)
);
}

template<class ValType> void get_and_check_eof
(
InputIterator& it, InputIterator end, std::ios_base& iosb,
std::ios_base::iostate& state, ValType& val
) const
{
get_signed(it, end, iosb, state, val);
if(it == end)
state |= std::ios_base::eofbit;
}

template<class ValType> void get_signed
(
InputIterator& it, InputIterator end, std::ios_base& iosb,
std::ios_base::iostate& state, ValType& val
) const
{
const std::ctype<CharType>& ct
= std::use_facet<std::ctype<CharType> >(iosb.getloc());

char c = peek_char(it, end, ct);

bool negative = (c == '-');

if(negative || c == '+')
{
++it;
c = peek_char(it, end, ct);
if(c == '-' || c == '+')
{ 
state |= std::ios_base::failbit;
return;
}
}

get_unsigned(it, end, iosb, ct, state, val);

if(negative)
{
val = (boost::math::changesign)(val);
}
} 

template<class ValType> void get_unsigned
( 
InputIterator& it, InputIterator end, std::ios_base& iosb,
const std::ctype<CharType>& ct,
std::ios_base::iostate& state, ValType& val
) const
{
switch(peek_char(it, end, ct))
{
case 'i':
get_i(it, end, ct, state, val);
break;

case 'n':
get_n(it, end, ct, state, val);
break;

case 'q':
case 's':
get_q(it, end, ct, state, val);
break;

default: 
it = std::num_get<CharType, InputIterator>::do_get(
it, end, iosb, state, val);
if((flags_ & legacy) && val == static_cast<ValType>(1)
&& peek_char(it, end, ct) == '#')
get_one_hash(it, end, ct, state, val);
break;
}
} 


template<class ValType> void get_i
( 
InputIterator& it, InputIterator end, const std::ctype<CharType>& ct,
std::ios_base::iostate& state, ValType& val
) const
{
if(!std::numeric_limits<ValType>::has_infinity
|| (flags_ & trap_infinity))
{
state |= std::ios_base::failbit;
return;
}

++it;
if(!match_string(it, end, ct, "nf"))
{
state |= std::ios_base::failbit;
return;
}

if(peek_char(it, end, ct) != 'i')
{
val = std::numeric_limits<ValType>::infinity();  
return;
}

++it;
if(!match_string(it, end, ct, "nity"))
{ 
state |= std::ios_base::failbit;
return;
}

val = std::numeric_limits<ValType>::infinity(); 
} 

template<class ValType> void get_n
( 
InputIterator& it, InputIterator end, const std::ctype<CharType>& ct,
std::ios_base::iostate& state, ValType& val
) const
{
if(!std::numeric_limits<ValType>::has_quiet_NaN
|| (flags_ & trap_nan)) {
state |= std::ios_base::failbit;
return;
}

++it;
if(!match_string(it, end, ct, "an"))
{
state |= std::ios_base::failbit;
return;
}

switch(peek_char(it, end, ct)) {
case 'q':
case 's':
if(flags_ && legacy)
++it;
break;  

case '(':   
{
++it;
char c;
while((c = peek_char(it, end, ct))
&& c != ')' && c != ' ' && c != '\n' && c != '\t')
++it;
if(c != ')')
{ 
state |= std::ios_base::failbit;
return;
}
++it;
break;  
}

default:
break;  
}

val = positive_nan<ValType>();
} 

template<class ValType> void get_q
( 
InputIterator& it, InputIterator end, const std::ctype<CharType>& ct,
std::ios_base::iostate& state, ValType& val
) const
{
if(!std::numeric_limits<ValType>::has_quiet_NaN
|| (flags_ & trap_nan) || !(flags_ & legacy))
{
state |= std::ios_base::failbit;
return;
}

++it;
if(!match_string(it, end, ct, "nan"))
{
state |= std::ios_base::failbit;
return;
}

val = positive_nan<ValType>(); 
} 

template<class ValType> void get_one_hash
( 
InputIterator& it, InputIterator end, const std::ctype<CharType>& ct,
std::ios_base::iostate& state, ValType& val
) const
{

++it;
switch(peek_char(it, end, ct))
{
case 'i': 
get_one_hash_i(it, end, ct, state, val); 
return;

case 'q': 
case 's': 
if(std::numeric_limits<ValType>::has_quiet_NaN
&& !(flags_ & trap_nan))
{
++it;
if(match_string(it, end, ct, "nan"))
{ 
val = positive_nan<ValType>(); 
return;
}
}
break;

default:
break;
}

state |= std::ios_base::failbit;
} 

template<class ValType> void get_one_hash_i
( 
InputIterator& it, InputIterator end, const std::ctype<CharType>& ct,
std::ios_base::iostate& state, ValType& val
) const
{
++it;

if(peek_char(it, end, ct) == 'n')
{
++it;
switch(peek_char(it, end, ct))
{
case 'f':  
if(std::numeric_limits<ValType>::has_infinity
&& !(flags_ & trap_infinity))
{
++it;
val = std::numeric_limits<ValType>::infinity();
return;
}
break;

case 'd':   
if(std::numeric_limits<ValType>::has_quiet_NaN
&& !(flags_ & trap_nan))
{
++it;
val = positive_nan<ValType>();
return;
}
break;

default:
break;
}
}

state |= std::ios_base::failbit;
} 


char peek_char
( 
InputIterator& it, InputIterator end,
const std::ctype<CharType>& ct
) const
{
if(it == end) return 0;
return ct.narrow(ct.tolower(*it), 0); 
}

bool match_string
( 
InputIterator& it, InputIterator end,
const std::ctype<CharType>& ct,
const char* s
) const
{
while(it != end && *s && *s == ct.narrow(ct.tolower(*it), 0))
{
++s;
++it; 
}
return !*s;
} 

private:
const int flags_;
}; 


}   
}   

#ifdef _MSC_VER
#   pragma warning(pop)
#endif

#endif 


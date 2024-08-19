
#ifndef BOOST_UTF8_CODECVT_FACET_HPP
#define BOOST_UTF8_CODECVT_FACET_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif






#include <locale>
#include <cwchar>   
#include <cstddef>  

#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>

#if defined(BOOST_NO_STDC_NAMESPACE)
namespace std {
using ::mbstate_t;
using ::size_t;
}
#endif

#define MB_LENGTH_MAX 8

BOOST_UTF8_BEGIN_NAMESPACE


#ifndef BOOST_UTF8_DECL
#define BOOST_UTF8_DECL
#endif

struct BOOST_UTF8_DECL utf8_codecvt_facet :
public std::codecvt<wchar_t, char, std::mbstate_t>  
{
public:
explicit utf8_codecvt_facet(std::size_t no_locale_manage=0);
virtual ~utf8_codecvt_facet();
protected:
virtual std::codecvt_base::result do_in(
std::mbstate_t& state, 
const char * from,
const char * from_end, 
const char * & from_next,
wchar_t * to, 
wchar_t * to_end, 
wchar_t*& to_next
) const;

virtual std::codecvt_base::result do_out(
std::mbstate_t & state,
const wchar_t * from,
const wchar_t * from_end,
const wchar_t*  & from_next,
char * to,
char * to_end,
char * & to_next
) const;

bool invalid_continuing_octet(unsigned char octet_1) const {
return (octet_1 < 0x80|| 0xbf< octet_1);
}

bool invalid_leading_octet(unsigned char octet_1)   const {
return (0x7f < octet_1 && octet_1 < 0xc0) ||
(octet_1 > 0xfd);
}

static unsigned int get_cont_octet_count(unsigned char lead_octet) {
return get_octet_count(lead_octet) - 1;
}

static unsigned int get_octet_count(unsigned char lead_octet);

int get_cont_octet_out_count(wchar_t word) const ;

virtual bool do_always_noconv() const BOOST_NOEXCEPT_OR_NOTHROW {
return false;
}

virtual std::codecvt_base::result do_unshift(
std::mbstate_t&,
char * from,
char * ,
char * & next
) const {
next = from;
return ok;
}

virtual int do_encoding() const BOOST_NOEXCEPT_OR_NOTHROW {
const int variable_byte_external_encoding=0;
return variable_byte_external_encoding;
}

virtual int do_length(
std::mbstate_t &,
const char * from,
const char * from_end, 
std::size_t max_limit
) const
#if BOOST_WORKAROUND(__IBMCPP__, BOOST_TESTED_AT(600))
throw()
#endif
;

virtual int do_length(
const std::mbstate_t & s,
const char * from,
const char * from_end, 
std::size_t max_limit
) const
#if BOOST_WORKAROUND(__IBMCPP__, BOOST_TESTED_AT(600))
throw()
#endif
{
return do_length(
const_cast<std::mbstate_t &>(s),
from,
from_end,
max_limit
);
}

virtual int do_max_length() const BOOST_NOEXCEPT_OR_NOTHROW {
return 6; 
}
};

BOOST_UTF8_END_NAMESPACE

#endif 

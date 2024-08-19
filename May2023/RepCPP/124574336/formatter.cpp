#define BOOST_LOCALE_SOURCE
#include <boost/locale/formatting.hpp>
#include "formatter.hpp"
#include <boost/locale/info.hpp>
#include "uconv.hpp"


#include <unicode/numfmt.h>
#include <unicode/rbnf.h>
#include <unicode/datefmt.h>
#include <unicode/smpdtfmt.h>
#include <unicode/decimfmt.h>

#include <limits>

#include <iostream>

#include "predefined_formatters.hpp"
#include "time_zone.hpp"

#ifdef BOOST_MSVC
#  pragma warning(disable : 4244) 
#endif


namespace boost {
namespace locale {
namespace impl_icu {


std::locale::id icu_formatters_cache::id;

namespace {
struct init { init() { std::has_facet<icu_formatters_cache>(std::locale::classic()); } } instance;
}


template<typename CharType>
class number_format : public formatter<CharType> {
public:
typedef CharType char_type;
typedef std::basic_string<CharType> string_type;

virtual string_type format(double value,size_t &code_points) const
{
icu::UnicodeString tmp;
icu_fmt_->format(value,tmp);
code_points=tmp.countChar32();
return cvt_.std(tmp);
}
virtual string_type format(int64_t value,size_t &code_points) const
{
icu::UnicodeString tmp;
icu_fmt_->format(static_cast< ::int64_t>(value),tmp);
code_points=tmp.countChar32();
return cvt_.std(tmp);
}

virtual string_type format(int32_t value,size_t &code_points) const
{
icu::UnicodeString tmp;
#ifdef __SUNPRO_CC 
icu_fmt_->format(static_cast<int>(value),tmp);
#else
icu_fmt_->format(::int32_t(value),tmp);
#endif
code_points=tmp.countChar32();
return cvt_.std(tmp);
}

virtual size_t parse(string_type const &str,double &value) const 
{
return do_parse(str,value);
}

virtual size_t parse(string_type const &str,int64_t &value) const 
{
return do_parse(str,value);
}
virtual size_t parse(string_type const &str,int32_t &value) const
{
return do_parse(str,value);
}

number_format(icu::NumberFormat *fmt,std::string codepage) :
cvt_(codepage),
icu_fmt_(fmt)
{
}

private:

bool get_value(double &v,icu::Formattable &fmt) const
{
UErrorCode err=U_ZERO_ERROR;
v=fmt.getDouble(err);
if(U_FAILURE(err))
return false;
return true;
}

bool get_value(int64_t &v,icu::Formattable &fmt) const
{
UErrorCode err=U_ZERO_ERROR;
v=fmt.getInt64(err);
if(U_FAILURE(err))
return false;
return true;
}

bool get_value(int32_t &v,icu::Formattable &fmt) const
{
UErrorCode err=U_ZERO_ERROR;
v=fmt.getLong(err);
if(U_FAILURE(err))
return false;
return true;
}

template<typename ValueType>
size_t do_parse(string_type const &str,ValueType &v) const
{
icu::Formattable val;
icu::ParsePosition pp;
icu::UnicodeString tmp = cvt_.icu(str.data(),str.data()+str.size());

icu_fmt_->parse(tmp,val,pp);

ValueType tmp_v;

if(pp.getIndex() == 0 || !get_value(tmp_v,val))
return 0;
size_t cut = cvt_.cut(tmp,str.data(),str.data()+str.size(),pp.getIndex());
if(cut==0)
return 0;
v=tmp_v;
return cut;
}

icu_std_converter<CharType> cvt_;
icu::NumberFormat *icu_fmt_;
};


template<typename CharType>
class date_format : public formatter<CharType> {
public:
typedef CharType char_type;
typedef std::basic_string<CharType> string_type;

virtual string_type format(double value,size_t &code_points) const
{
return do_format(value,code_points);
}
virtual string_type format(int64_t value,size_t &code_points) const
{
return do_format(value,code_points);
}

virtual string_type format(int32_t value,size_t &code_points) const
{
return do_format(value,code_points);
}

virtual size_t parse(string_type const &str,double &value) const 
{
return do_parse(str,value);
}
virtual size_t parse(string_type const &str,int64_t &value) const 
{
return do_parse(str,value);
}
virtual size_t parse(string_type const &str,int32_t &value) const
{
return do_parse(str,value);
}

date_format(icu::DateFormat *fmt,bool transfer_owneship,std::string codepage) :
cvt_(codepage)
{
if(transfer_owneship) {
aicu_fmt_.reset(fmt);
icu_fmt_ = aicu_fmt_.get();
}
else {
icu_fmt_ = fmt;
}
}

private:

template<typename ValueType>
size_t do_parse(string_type const &str,ValueType &value) const
{
icu::ParsePosition pp;
icu::UnicodeString tmp = cvt_.icu(str.data(),str.data() + str.size());

UDate udate = icu_fmt_->parse(tmp,pp);
if(pp.getIndex() == 0)
return 0;
double date = udate / 1000.0;
typedef std::numeric_limits<ValueType> limits_type;
if(date > limits_type::max() || date < limits_type::min())
return 0;
size_t cut = cvt_.cut(tmp,str.data(),str.data()+str.size(),pp.getIndex());
if(cut==0)
return 0;
value=static_cast<ValueType>(date);
return cut;

}

string_type do_format(double value,size_t &codepoints) const 
{
UDate date = value * 1000.0; 
icu::UnicodeString tmp;
icu_fmt_->format(date,tmp);
codepoints=tmp.countChar32();
return cvt_.std(tmp);
}

icu_std_converter<CharType> cvt_;
hold_ptr<icu::DateFormat> aicu_fmt_;
icu::DateFormat *icu_fmt_;
};

icu::UnicodeString strftime_to_icu_full(icu::DateFormat *dfin,char const *alt)
{
hold_ptr<icu::DateFormat> df(dfin);
icu::SimpleDateFormat *sdf=dynamic_cast<icu::SimpleDateFormat *>(df.get());
icu::UnicodeString tmp;
if(sdf) {
sdf->toPattern(tmp);
}
else {
tmp=alt;
}
return tmp;

}

icu::UnicodeString strftime_to_icu_symbol(char c,icu::Locale const &locale,icu_formatters_cache const *cache=0)
{
switch(c) {
case 'a': 
return "EE";
case 'A': 
return "EEEE";
case 'b': 
return "MMM";
case 'B': 
return "MMMM";
case 'c': 
{
if(cache)
return cache->date_time_format_[1][1];
return strftime_to_icu_full(
icu::DateFormat::createDateTimeInstance(icu::DateFormat::kFull,icu::DateFormat::kFull,locale),
"yyyy-MM-dd HH:mm:ss"
);
}
case 'd': 
return "dd";
case 'D': 
return "MM/dd/yy";
case 'e': 
return "d";
case 'h': 
return "MMM";
case 'H': 
return "HH";
case 'I': 
return "hh";
case 'j': 
return "D";
case 'm': 
return "MM";
case 'M': 
return "mm";
case 'n': 
return "\n";
case 'p': 
return "a";
case 'r': 
return "hh:mm:ss a";
case 'R': 
return "HH:mm";
case 'S': 
return "ss";
case 't': 
return "\t";
case 'T': 
return "HH:mm:ss";

case 'x': 
{
if(cache)
return cache->date_format_[1];
return strftime_to_icu_full(
icu::DateFormat::createDateInstance(icu::DateFormat::kMedium,locale),
"yyyy-MM-dd"
);
}
case 'X': 
{
if(cache)
return cache->time_format_[1];
return strftime_to_icu_full(
icu::DateFormat::createTimeInstance(icu::DateFormat::kMedium,locale),
"HH:mm:ss"
);
}
case 'y': 
return "yy";
case 'Y': 
return "yyyy";
case 'Z': 
return "vvvv";
case '%': 
return "%";
default:
return "";
}
}

icu::UnicodeString strftime_to_icu(icu::UnicodeString const &ftime,icu::Locale const &locale)
{
unsigned len=ftime.length();
icu::UnicodeString result;
bool escaped=false;
for(unsigned i=0;i<len;i++) {
UChar c=ftime[i];
if(c=='%') {
i++;
c=ftime[i];
if(c=='E' || c=='O') {
i++;
c=ftime[i];
}
if(escaped) {
result+="'";
escaped=false;
}
result+=strftime_to_icu_symbol(c,locale);
}
else if(c=='\'') {
result+="''";
}
else {
if(!escaped) {
result+="'";
escaped=true;
}
result+=c;
}
}
if(escaped)
result+="'";
return result;
}

template<typename CharType>
formatter<CharType> *generate_formatter(
std::ios_base &ios,
icu::Locale const &locale,
std::string const &encoding)
{
using namespace boost::locale::flags;

hold_ptr<formatter<CharType> > fmt;
ios_info &info=ios_info::get(ios);
uint64_t disp = info.display_flags();

icu_formatters_cache const &cache = std::use_facet<icu_formatters_cache>(ios.getloc());


if(disp == posix)
return fmt.release();

UErrorCode err=U_ZERO_ERROR;

switch(disp) {
case number:
{
std::ios_base::fmtflags how = (ios.flags() & std::ios_base::floatfield);
icu::NumberFormat *nf = 0;

if(how == std::ios_base::scientific)
nf = cache.number_format(icu_formatters_cache::fmt_sci);
else
nf = cache.number_format(icu_formatters_cache::fmt_number);

nf->setMaximumFractionDigits(ios.precision());
if(how == std::ios_base::scientific || how == std::ios_base::fixed ) {
nf->setMinimumFractionDigits(ios.precision());
}
else {
nf->setMinimumFractionDigits(0);
}
fmt.reset(new number_format<CharType>(nf,encoding));
}
break;
case currency:
{
icu::NumberFormat *nf;

uint64_t curr = info.currency_flags();

if(curr == currency_default || curr == currency_national)
nf = cache.number_format(icu_formatters_cache::fmt_curr_nat);
else
nf = cache.number_format(icu_formatters_cache::fmt_curr_iso);

fmt.reset(new number_format<CharType>(nf,encoding));
}
break;
case percent:
{
icu::NumberFormat *nf = cache.number_format(icu_formatters_cache::fmt_per);
nf->setMaximumFractionDigits(ios.precision());
std::ios_base::fmtflags how = (ios.flags() & std::ios_base::floatfield);
if(how == std::ios_base::scientific || how == std::ios_base::fixed ) {
nf->setMinimumFractionDigits(ios.precision());
}
else {
nf->setMinimumFractionDigits(0);
}
fmt.reset(new number_format<CharType>(nf,encoding));

}
break;
case spellout:
fmt.reset(new number_format<CharType>(cache.number_format(icu_formatters_cache::fmt_spell),encoding));
break;
case ordinal:
fmt.reset(new number_format<CharType>(cache.number_format(icu_formatters_cache::fmt_ord),encoding));
break;
case date:
case time:
case datetime:
case strftime:
{
using namespace flags;
hold_ptr<icu::DateFormat> adf;
icu::DateFormat *df = 0;
icu::SimpleDateFormat *sdf = cache.date_formatter();
if(sdf) {
int tmf=info.time_flags();
switch(tmf) {
case time_short:
tmf=0;
break;
case time_long:
tmf=2;
break;
case time_full:
tmf=3;
break;
case time_default:
case time_medium:
default:
tmf=1;
}
int dtf=info.date_flags();
switch(dtf) {
case date_short:
dtf=0;
break;
case date_long:
dtf=2;
break;
case date_full:
dtf=3;
break;
case date_default:
case date_medium:
default:
dtf=1;
}

icu::UnicodeString pattern;
switch(disp) {
case date:
pattern = cache.date_format_[dtf];
break;
case time:
pattern = cache.time_format_[tmf];
break;
case datetime:
pattern = cache.date_time_format_[dtf][tmf];
break;
case strftime:
{
if( !cache.date_format_[1].isEmpty() 
&& !cache.time_format_[1].isEmpty()
&& !cache.date_time_format_[1][1].isEmpty())
{
icu_std_converter<CharType> cvt_(encoding);
std::basic_string<CharType> const &f=info.date_time_pattern<CharType>();
pattern = strftime_to_icu(cvt_.icu(f.c_str(),f.c_str()+f.size()),locale);
}
}
break;
}
if(!pattern.isEmpty()) {
sdf->applyPattern(pattern);
df = sdf;
sdf = 0;
}
sdf = 0;
}

if(!df) {
icu::DateFormat::EStyle dstyle = icu::DateFormat::kDefault;
icu::DateFormat::EStyle tstyle = icu::DateFormat::kDefault;

switch(info.time_flags()) {
case time_short:    tstyle=icu::DateFormat::kShort; break;
case time_medium:   tstyle=icu::DateFormat::kMedium; break;
case time_long:     tstyle=icu::DateFormat::kLong; break;
case time_full:     tstyle=icu::DateFormat::kFull; break;
}
switch(info.date_flags()) {
case date_short:    dstyle=icu::DateFormat::kShort; break;
case date_medium:   dstyle=icu::DateFormat::kMedium; break;
case date_long:     dstyle=icu::DateFormat::kLong; break;
case date_full:     dstyle=icu::DateFormat::kFull; break;
}

if(disp==date)
adf.reset(icu::DateFormat::createDateInstance(dstyle,locale));
else if(disp==time)
adf.reset(icu::DateFormat::createTimeInstance(tstyle,locale));
else if(disp==datetime)
adf.reset(icu::DateFormat::createDateTimeInstance(dstyle,tstyle,locale));
else {
icu_std_converter<CharType> cvt_(encoding);
std::basic_string<CharType> const &f=info.date_time_pattern<CharType>();
icu::UnicodeString fmt = strftime_to_icu(cvt_.icu(f.data(),f.data()+f.size()),locale);
adf.reset(new icu::SimpleDateFormat(fmt,locale,err));
}
if(U_FAILURE(err)) 
return fmt.release();
df = adf.get();
}

df->adoptTimeZone(get_time_zone(info.time_zone()));

if(adf.get())
fmt.reset(new date_format<CharType>(adf.release(),true,encoding));
else
fmt.reset(new date_format<CharType>(df,false,encoding));
}
break;
}

return fmt.release();
}



template<>
formatter<char> *formatter<char>::create(std::ios_base &ios,icu::Locale const &l,std::string const &e)
{
return generate_formatter<char>(ios,l,e);
}

template<>
formatter<wchar_t> *formatter<wchar_t>::create(std::ios_base &ios,icu::Locale const &l,std::string const &e)
{
return generate_formatter<wchar_t>(ios,l,e);
}


#ifdef BOOST_LOCALE_ENABLE_CHAR16_T
template<>
formatter<char16_t> *formatter<char16_t>::create(std::ios_base &ios,icu::Locale const &l,std::string const &e)
{
return generate_formatter<char16_t>(ios,l,e);
}

#endif

#ifdef BOOST_LOCALE_ENABLE_CHAR32_T
template<>
formatter<char32_t> *formatter<char32_t>::create(std::ios_base &ios,icu::Locale const &l,std::string const &e)
{
return generate_formatter<char32_t>(ios,l,e);
}

#endif

} 

} 
} 






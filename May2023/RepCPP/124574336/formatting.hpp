#ifndef BOOST_LOCALE_FORMATTING_HPP_INCLUDED
#define BOOST_LOCALE_FORMATTING_HPP_INCLUDED

#include <boost/locale/config.hpp>
#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4275 4251 4231 4660)
#endif
#include <boost/cstdint.hpp>
#include <boost/locale/time_zone.hpp>
#include <ostream>
#include <istream>
#include <string>
#include <string.h>
#include <typeinfo>

namespace boost {
namespace locale {
namespace flags {
typedef enum {
posix               = 0,
number              = 1,
currency            = 2,
percent             = 3,
date                = 4,
time                = 5,
datetime            = 6,
strftime            = 7,
spellout            = 8,
ordinal             = 9,

display_flags_mask  = 31,

currency_default    = 0 << 5,
currency_iso        = 1 << 5,
currency_national   = 2 << 5,

currency_flags_mask = 3 << 5,

time_default        = 0 << 7,
time_short          = 1 << 7,
time_medium         = 2 << 7,
time_long           = 3 << 7,
time_full           = 4 << 7,
time_flags_mask     = 7 << 7,

date_default        = 0 << 10,
date_short          = 1 << 10,
date_medium         = 2 << 10,
date_long           = 3 << 10,
date_full           = 4 << 10,
date_flags_mask     = 7 << 10,

datetime_flags_mask = date_flags_mask | time_flags_mask

} display_flags_type;

typedef enum {
datetime_pattern,   
time_zone_id        
} pattern_type;

typedef enum {
domain_id           
} value_type;


} 

class BOOST_LOCALE_DECL ios_info {
public:


ios_info();
ios_info(ios_info const &);
ios_info const &operator=(ios_info const &);
~ios_info();


static ios_info &get(std::ios_base &ios);

void display_flags(uint64_t flags);

void currency_flags(uint64_t flags);

void date_flags(uint64_t flags);

void time_flags(uint64_t flags);

void datetime_flags(uint64_t flags);

void domain_id(int);

void time_zone(std::string const &);


template<typename CharType>
void date_time_pattern(std::basic_string<CharType> const &str)
{
string_set &s = date_time_pattern_set();
s.set<CharType>(str.c_str());
}


uint64_t display_flags() const;

uint64_t currency_flags() const;


uint64_t date_flags() const;

uint64_t time_flags() const;

uint64_t datetime_flags() const;

int domain_id() const;

std::string time_zone() const;

template<typename CharType>
std::basic_string<CharType> date_time_pattern() const
{
string_set const &s = date_time_pattern_set();
return s.get<CharType>();
}

void on_imbue();

private:

class string_set;

string_set const &date_time_pattern_set() const;
string_set &date_time_pattern_set();

class BOOST_LOCALE_DECL string_set {
public:
string_set(); 
~string_set();
string_set(string_set const &other);
string_set const &operator=(string_set const &other);
void swap(string_set &other);

template<typename Char>
void set(Char const *s)
{
delete [] ptr;
ptr = 0;
type=&typeid(Char);
Char const *end = s;
while(*end!=0) end++;
size = sizeof(Char)*(end - s+1);
ptr = new char[size];
memcpy(ptr,s,size);
}

template<typename Char>
std::basic_string<Char> get() const
{
if(type==0 || *type!=typeid(Char))
throw std::bad_cast();
std::basic_string<Char> result = reinterpret_cast<Char const *>(ptr);
return result;
}

private:
std::type_info const *type;
size_t size;
char *ptr;
};

uint64_t flags_;
int domain_id_;
std::string time_zone_;
string_set datetime_;

struct data;
data *d;

};


namespace as {


inline std::ios_base & posix(std::ios_base & ios)
{
ios_info::get(ios).display_flags(flags::posix);
return ios;
}

inline std::ios_base & number(std::ios_base & ios)
{
ios_info::get(ios).display_flags(flags::number);
return ios;
}

inline std::ios_base & currency(std::ios_base & ios)
{
ios_info::get(ios).display_flags(flags::currency);
return ios;
}

inline std::ios_base & percent(std::ios_base & ios)
{
ios_info::get(ios).display_flags(flags::percent);
return ios;
}

inline std::ios_base & date(std::ios_base & ios)
{
ios_info::get(ios).display_flags(flags::date);
return ios;
}

inline std::ios_base & time(std::ios_base & ios)
{
ios_info::get(ios).display_flags(flags::time);
return ios;
}

inline std::ios_base & datetime(std::ios_base & ios)
{
ios_info::get(ios).display_flags(flags::datetime);
return ios;
}

inline std::ios_base & strftime(std::ios_base & ios)
{
ios_info::get(ios).display_flags(flags::strftime);
return ios;
}

inline std::ios_base & spellout(std::ios_base & ios)
{
ios_info::get(ios).display_flags(flags::spellout);
return ios;
}

inline std::ios_base & ordinal(std::ios_base & ios)
{
ios_info::get(ios).display_flags(flags::ordinal);
return ios;
}

inline std::ios_base & currency_default(std::ios_base & ios)
{
ios_info::get(ios).currency_flags(flags::currency_default);
return ios;
}

inline std::ios_base & currency_iso(std::ios_base & ios)
{
ios_info::get(ios).currency_flags(flags::currency_iso);
return ios;
}

inline std::ios_base & currency_national(std::ios_base & ios)
{
ios_info::get(ios).currency_flags(flags::currency_national);
return ios;
}

inline std::ios_base & time_default(std::ios_base & ios)
{
ios_info::get(ios).time_flags(flags::time_default);
return ios;
}

inline std::ios_base & time_short(std::ios_base & ios)
{
ios_info::get(ios).time_flags(flags::time_short);
return ios;
}

inline std::ios_base & time_medium(std::ios_base & ios)
{
ios_info::get(ios).time_flags(flags::time_medium);
return ios;
}

inline std::ios_base & time_long(std::ios_base & ios)
{
ios_info::get(ios).time_flags(flags::time_long);
return ios;
}

inline std::ios_base & time_full(std::ios_base & ios)
{
ios_info::get(ios).time_flags(flags::time_full);
return ios;
}

inline std::ios_base & date_default(std::ios_base & ios)
{
ios_info::get(ios).date_flags(flags::date_default);
return ios;
}

inline std::ios_base & date_short(std::ios_base & ios)
{
ios_info::get(ios).date_flags(flags::date_short);
return ios;
}

inline std::ios_base & date_medium(std::ios_base & ios)
{
ios_info::get(ios).date_flags(flags::date_medium);
return ios;
}

inline std::ios_base & date_long(std::ios_base & ios)
{
ios_info::get(ios).date_flags(flags::date_long);
return ios;
}

inline std::ios_base & date_full(std::ios_base & ios)
{
ios_info::get(ios).date_flags(flags::date_full);
return ios;
}            


namespace details {
template<typename CharType>
struct add_ftime {

std::basic_string<CharType> ftime;

void apply(std::basic_ios<CharType> &ios) const
{
ios_info::get(ios).date_time_pattern(ftime);
as::strftime(ios);
}

};

template<typename CharType>
std::basic_ostream<CharType> &operator<<(std::basic_ostream<CharType> &out,add_ftime<CharType> const &fmt)
{
fmt.apply(out);
return out;
}

template<typename CharType>
std::basic_istream<CharType> &operator>>(std::basic_istream<CharType> &in,add_ftime<CharType> const &fmt)
{
fmt.apply(in);
return in;
}

}



template<typename CharType>
#ifdef BOOST_LOCALE_DOXYGEN
unspecified_type
#else
details::add_ftime<CharType> 
#endif
ftime(std::basic_string<CharType> const &format)
{
details::add_ftime<CharType> fmt;
fmt.ftime=format;
return fmt;
}

template<typename CharType>
#ifdef BOOST_LOCALE_DOXYGEN
unspecified_type
#else
details::add_ftime<CharType> 
#endif
ftime(CharType const *format)
{
details::add_ftime<CharType> fmt;
fmt.ftime=format;
return fmt;
}

namespace details {
struct set_timezone {
std::string id;
};
template<typename CharType>
std::basic_ostream<CharType> &operator<<(std::basic_ostream<CharType> &out,set_timezone const &fmt)
{
ios_info::get(out).time_zone(fmt.id);
return out;
}

template<typename CharType>
std::basic_istream<CharType> &operator>>(std::basic_istream<CharType> &in,set_timezone const &fmt)
{
ios_info::get(in).time_zone(fmt.id);
return in;
}
}

inline std::ios_base &gmt(std::ios_base &ios)
{
ios_info::get(ios).time_zone("GMT");
return ios;
}

inline std::ios_base &local_time(std::ios_base &ios)
{
ios_info::get(ios).time_zone(time_zone::global());
return ios;
}

inline 
#ifdef BOOST_LOCALE_DOXYGEN
unspecified_type
#else
details::set_timezone 
#endif
time_zone(char const *id) 
{
details::set_timezone tz;
tz.id=id;
return tz;
}

inline 
#ifdef BOOST_LOCALE_DOXYGEN
unspecified_type
#else
details::set_timezone 
#endif            
time_zone(std::string const &id) 
{
details::set_timezone tz;
tz.id=id;
return tz;
}



} 

} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif


#endif

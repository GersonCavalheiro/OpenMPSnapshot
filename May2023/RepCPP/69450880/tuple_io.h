






#pragma once

#include <istream>
#include <ostream>
#include <locale> 
#include <hydra/detail/external/hydra_thrust/detail/tuple/index_sequence.h>
#include <hydra/detail/external/hydra_thrust/tuple.h>

namespace hydra_thrust {
namespace detail {
namespace tuple_detail {

class format_info {
public:   

enum manipulator_type { open, close, delimiter };
enum { number_of_manipulators = delimiter + 1 };
private:

static int get_stream_index (int m)
{
static const int stream_index[number_of_manipulators]
= { std::ios::xalloc(), std::ios::xalloc(), std::ios::xalloc() };

return stream_index[m];
}

format_info(const format_info&);
format_info();   


public:

template<class CharType, class CharTrait>
static CharType get_manipulator(std::basic_ios<CharType, CharTrait>& i, 
manipulator_type m) {
CharType c = static_cast<CharType>(i.iword(get_stream_index(m)) ); 
if (!c) {
switch(m) {
case detail::tuple_detail::format_info::open :  c = i.widen('('); break;
case detail::tuple_detail::format_info::close : c = i.widen(')'); break;
case detail::tuple_detail::format_info::delimiter : c = i.widen(' '); break;
}
}
return c;
}


template<class CharType, class CharTrait>
static void set_manipulator(std::basic_ios<CharType, CharTrait>& i, 
manipulator_type m, CharType c) {
i.iword(get_stream_index(m)) = static_cast<long>(c);
}
};


template<class CharType>
class tuple_manipulator {
const format_info::manipulator_type mt;
CharType f_c;
public:
explicit tuple_manipulator(format_info::manipulator_type m, 
const char c = 0)
: mt(m), f_c(c) {}

template<class CharTrait>
void set(std::basic_ios<CharType, CharTrait> &io) const {
format_info::set_manipulator(io, mt, f_c);
}
};

template<class CharType, class CharTrait>
inline std::basic_ostream<CharType, CharTrait>&
operator<<(std::basic_ostream<CharType, CharTrait>& o, const tuple_manipulator<CharType>& m) {
m.set(o);
return o;
}


template<class CharType, class CharTrait>
inline std::basic_istream<CharType, CharTrait>&
operator>>(std::basic_istream<CharType, CharTrait>& i, const tuple_manipulator<CharType>& m) {
m.set(i);
return i;
}


} 
} 


template<class CharType>
inline detail::tuple_detail::tuple_manipulator<CharType> set_open(const CharType c) {
return detail::tuple_detail::tuple_manipulator<CharType>(detail::tuple_detail::format_info::open, c);
}


template<class CharType>
inline detail::tuple_detail::tuple_manipulator<CharType> set_close(const CharType c) {
return detail::tuple_detail::tuple_manipulator<CharType>(detail::tuple_detail::format_info::close, c);
}


template<class CharType>
inline detail::tuple_detail::tuple_manipulator<CharType> set_delimiter(const CharType c) {
return detail::tuple_detail::tuple_manipulator<CharType>(detail::tuple_detail::format_info::delimiter, c);
}


namespace detail {
namespace tuple_detail {



template<class CharType, class CharTrait>
void
print_helper(CharType d, std::basic_ostream<CharType, CharTrait>& o)
{
}

template<class CharType, class CharTrait, class T>
void
print_helper(CharType d, std::basic_ostream<CharType, CharTrait>& o, const T& t)
{
o << t;
}

template<class CharType, class CharTrait, class T, class... Types>
void
print_helper(CharType d, std::basic_ostream<CharType, CharTrait>& o, const T& t, const Types&... ts)
{
o << t << d;

print_helper(d, o, ts...);
}


template<class CharType, class CharTrait, class... Types, size_t... I>
inline std::basic_ostream<CharType, CharTrait>&
print(std::basic_ostream<CharType, CharTrait>& o, const hydra_thrust::tuple<Types...>& t, hydra_thrust::__index_sequence<I...>)
{
const CharType d = detail::tuple_detail::format_info::get_manipulator(o, detail::tuple_detail::format_info::delimiter);

print_helper(d, o, hydra_thrust::get<I>(t)...);

return o;
}


template<class CharT, class Traits, class T>
inline bool handle_width(std::basic_ostream<CharT, Traits>& o, const T& t) {
std::streamsize width = o.width();
if(width == 0) return false;

std::basic_ostringstream<CharT, Traits> ss;

ss.copyfmt(o);
ss.tie(0);
ss.width(0);

ss << t;
o << ss.str();

return true;
}


} 
} 


template<class CharType, class CharTrait, class... Types>
inline std::basic_ostream<CharType, CharTrait>& 
operator<<(std::basic_ostream<CharType, CharTrait>& o, 
const hydra_thrust::tuple<Types...>& t) {
if (!o.good() ) return o;
if (detail::tuple_detail::handle_width(o, t)) return o;

const CharType l = 
detail::tuple_detail::format_info::get_manipulator(o, detail::tuple_detail::format_info::open);
const CharType r = 
detail::tuple_detail::format_info::get_manipulator(o, detail::tuple_detail::format_info::close);

o << l;

detail::tuple_detail::print(o, t, hydra_thrust::__make_index_sequence<hydra_thrust::tuple_size<hydra_thrust::tuple<Types...>>::value>{});   

o << r;

return o;
}



namespace detail {
namespace tuple_detail {

template<class CharType, class CharTrait>
inline std::basic_istream<CharType, CharTrait>& 
extract_and_check_delimiter(
std::basic_istream<CharType, CharTrait> &is, format_info::manipulator_type del)
{
const CharType d = format_info::get_manipulator(is, del);

const bool is_delimiter = (!std::isspace(d, is.getloc()) );

CharType c;
if (is_delimiter) { 
is >> c;
if (is.good() && c!=d) { 
is.setstate(std::ios::failbit);
}
} else {
is >> std::ws;
}
return is;
}

template<class CharType, class CharTrait, class T>
void
read_helper(std::basic_istream<CharType, CharTrait> &is)
{
}

template<class CharType, class CharTrait, class T>
void
read_helper(std::basic_istream<CharType, CharTrait> &is, T& t)
{
is >> t;
}

template<class CharType, class CharTrait, class T, class... Types>
void
read_helper(std::basic_istream<CharType, CharTrait> &is, T& t, Types&... ts)
{
is >> t;
extract_and_check_delimiter(is, format_info::delimiter);
read_helper(is, ts...);
}

template<class CharType, class CharTrait, class... Types, size_t... I>
inline std::basic_istream<CharType, CharTrait>& 
read(std::basic_istream<CharType, CharTrait> &is, hydra_thrust::tuple<Types...>& t, hydra_thrust::__index_sequence<I...>)
{
if (!is.good()) return is;
read_helper(is, hydra_thrust::get<I>(t)...);
return is;
}


} 
} 


template<class CharType, class CharTrait, class... Types>
inline std::basic_istream<CharType, CharTrait>& 
operator>>(std::basic_istream<CharType, CharTrait>& is, hydra_thrust::tuple<Types...>& t) {

if (!is.good() ) return is;

detail::tuple_detail::extract_and_check_delimiter(is, detail::tuple_detail::format_info::open);

detail::tuple_detail::read(is, t, hydra_thrust::__make_index_sequence<hydra_thrust::tuple_size<hydra_thrust::tuple<Types...>>::value>{});   

detail::tuple_detail::extract_and_check_delimiter(is, detail::tuple_detail::format_info::close);

return is;
}


} 


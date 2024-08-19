




#ifndef BOOST_FILESYSTEM_PATH_HPP
#define BOOST_FILESYSTEM_PATH_HPP

#include <boost/config.hpp>

# if defined( BOOST_NO_STD_WSTRING )
#   error Configuration not supported: Boost.Filesystem V3 and later requires std::wstring support
# endif

#include <boost/assert.hpp>
#include <boost/filesystem/config.hpp>
#include <boost/filesystem/path_traits.hpp>  
#include <boost/system/error_code.hpp>
#include <boost/system/system_error.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/iterator_categories.hpp>
#include <boost/core/enable_if.hpp>
#include <boost/io/quoted.hpp>
#include <boost/functional/hash_fwd.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <string>
#include <iterator>
#include <cstring>
#include <iosfwd>
#include <stdexcept>
#include <cassert>
#include <locale>
#include <algorithm>

#include <boost/config/abi_prefix.hpp> 

namespace boost
{
namespace filesystem
{
namespace path_detail 
{

template< typename Char, Char Separator, Char PreferredSeparator, Char Dot >
struct path_constants
{
typedef path_constants< Char, Separator, PreferredSeparator, Dot > path_constants_base;
typedef Char                                    value_type;
static BOOST_CONSTEXPR_OR_CONST value_type      separator = Separator;
static BOOST_CONSTEXPR_OR_CONST value_type      preferred_separator = PreferredSeparator;
static BOOST_CONSTEXPR_OR_CONST value_type      dot = Dot;
};

#if defined(BOOST_NO_CXX17_INLINE_VARIABLES)
template< typename Char, Char Separator, Char PreferredSeparator, Char Dot >
BOOST_CONSTEXPR_OR_CONST typename path_constants< Char, Separator, PreferredSeparator, Dot >::value_type
path_constants< Char, Separator, PreferredSeparator, Dot >::separator;
template< typename Char, Char Separator, Char PreferredSeparator, Char Dot >
BOOST_CONSTEXPR_OR_CONST typename path_constants< Char, Separator, PreferredSeparator, Dot >::value_type
path_constants< Char, Separator, PreferredSeparator, Dot >::preferred_separator;
template< typename Char, Char Separator, Char PreferredSeparator, Char Dot >
BOOST_CONSTEXPR_OR_CONST typename path_constants< Char, Separator, PreferredSeparator, Dot >::value_type
path_constants< Char, Separator, PreferredSeparator, Dot >::dot;
#endif

} 


class path :
public filesystem::path_detail::path_constants<
#ifdef BOOST_WINDOWS_API
wchar_t, L'/', L'\\', L'.'
#else
char, '/', '/', '.'
#endif
>
{
public:


typedef path_constants_base::value_type value_type;
typedef std::basic_string<value_type>  string_type;
typedef std::codecvt<wchar_t, char,
std::mbstate_t>   codecvt_type;









path() BOOST_NOEXCEPT {}
path(const path& p) : m_pathname(p.m_pathname) {}

template <class Source>
path(Source const& source,
typename boost::enable_if<path_traits::is_pathable<
typename boost::decay<Source>::type> >::type* =0)
{
path_traits::dispatch(source, m_pathname);
}

path(const value_type* s) : m_pathname(s) {}
path(value_type* s) : m_pathname(s) {}
path(const string_type& s) : m_pathname(s) {}
path(string_type& s) : m_pathname(s) {}


# if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
path(path&& p) BOOST_NOEXCEPT : m_pathname(std::move(p.m_pathname)) {}
path& operator=(path&& p) BOOST_NOEXCEPT
{ m_pathname = std::move(p.m_pathname); return *this; }
# endif

template <class Source>
path(Source const& source, const codecvt_type& cvt)
{
path_traits::dispatch(source, m_pathname, cvt);
}

template <class InputIterator>
path(InputIterator begin, InputIterator end)
{
if (begin != end)
{
std::basic_string<typename std::iterator_traits<InputIterator>::value_type>
seq(begin, end);
path_traits::convert(seq.c_str(), seq.c_str()+seq.size(), m_pathname);
}
}

template <class InputIterator>
path(InputIterator begin, InputIterator end, const codecvt_type& cvt)
{
if (begin != end)
{
std::basic_string<typename std::iterator_traits<InputIterator>::value_type>
seq(begin, end);
path_traits::convert(seq.c_str(), seq.c_str()+seq.size(), m_pathname, cvt);
}
}


path& operator=(const path& p)
{
m_pathname = p.m_pathname;
return *this;
}

template <class Source>
typename boost::enable_if<path_traits::is_pathable<
typename boost::decay<Source>::type>, path&>::type
operator=(Source const& source)
{
m_pathname.clear();
path_traits::dispatch(source, m_pathname);
return *this;
}


path& operator=(const value_type* ptr)  
{m_pathname = ptr; return *this;}
path& operator=(value_type* ptr)  
{m_pathname = ptr; return *this;}
path& operator=(const string_type& s) {m_pathname = s; return *this;}
path& operator=(string_type& s)       {m_pathname = s; return *this;}

path& assign(const value_type* ptr, const codecvt_type&)  
{m_pathname = ptr; return *this;}
template <class Source>
path& assign(Source const& source, const codecvt_type& cvt)
{
m_pathname.clear();
path_traits::dispatch(source, m_pathname, cvt);
return *this;
}

template <class InputIterator>
path& assign(InputIterator begin, InputIterator end)
{
m_pathname.clear();
if (begin != end)
{
std::basic_string<typename std::iterator_traits<InputIterator>::value_type>
seq(begin, end);
path_traits::convert(seq.c_str(), seq.c_str()+seq.size(), m_pathname);
}
return *this;
}

template <class InputIterator>
path& assign(InputIterator begin, InputIterator end, const codecvt_type& cvt)
{
m_pathname.clear();
if (begin != end)
{
std::basic_string<typename std::iterator_traits<InputIterator>::value_type>
seq(begin, end);
path_traits::convert(seq.c_str(), seq.c_str()+seq.size(), m_pathname, cvt);
}
return *this;
}


template <class Source>
typename boost::enable_if<path_traits::is_pathable<
typename boost::decay<Source>::type>, path&>::type
operator+=(Source const& source)
{
return concat(source);
}

path& operator+=(const path& p)         { m_pathname += p.m_pathname; return *this; }
path& operator+=(const value_type* ptr) { m_pathname += ptr; return *this; }
path& operator+=(value_type* ptr)       { m_pathname += ptr; return *this; }
path& operator+=(const string_type& s)  { m_pathname += s; return *this; }
path& operator+=(string_type& s)        { m_pathname += s; return *this; }
path& operator+=(value_type c)          { m_pathname += c; return *this; }

template <class CharT>
typename boost::enable_if<boost::is_integral<CharT>, path&>::type
operator+=(CharT c)
{
CharT tmp[2];
tmp[0] = c;
tmp[1] = 0;
return concat(tmp);
}

template <class Source>
path& concat(Source const& source)
{
path_traits::dispatch(source, m_pathname);
return *this;
}

template <class Source>
path& concat(Source const& source, const codecvt_type& cvt)
{
path_traits::dispatch(source, m_pathname, cvt);
return *this;
}

template <class InputIterator>
path& concat(InputIterator begin, InputIterator end)
{
if (begin == end)
return *this;
std::basic_string<typename std::iterator_traits<InputIterator>::value_type>
seq(begin, end);
path_traits::convert(seq.c_str(), seq.c_str()+seq.size(), m_pathname);
return *this;
}

template <class InputIterator>
path& concat(InputIterator begin, InputIterator end, const codecvt_type& cvt)
{
if (begin == end)
return *this;
std::basic_string<typename std::iterator_traits<InputIterator>::value_type>
seq(begin, end);
path_traits::convert(seq.c_str(), seq.c_str()+seq.size(), m_pathname, cvt);
return *this;
}



BOOST_FILESYSTEM_DECL path& operator/=(const path& p);

template <class Source>
typename boost::enable_if<path_traits::is_pathable<
typename boost::decay<Source>::type>, path&>::type
operator/=(Source const& source)
{
return append(source);
}

BOOST_FILESYSTEM_DECL path& operator/=(const value_type* ptr);
path& operator/=(value_type* ptr)
{
return this->operator/=(const_cast<const value_type*>(ptr));
}
path& operator/=(const string_type& s) { return this->operator/=(path(s)); }
path& operator/=(string_type& s)       { return this->operator/=(path(s)); }

path& append(const value_type* ptr)  
{
this->operator/=(ptr);
return *this;
}

path& append(const value_type* ptr, const codecvt_type&)  
{
this->operator/=(ptr);
return *this;
}

template <class Source>
path& append(Source const& source);

template <class Source>
path& append(Source const& source, const codecvt_type& cvt);

template <class InputIterator>
path& append(InputIterator begin, InputIterator end);

template <class InputIterator>
path& append(InputIterator begin, InputIterator end, const codecvt_type& cvt);


void clear() BOOST_NOEXCEPT { m_pathname.clear(); }
#   ifdef BOOST_POSIX_API
path& make_preferred() { return *this; }  
#   else 
BOOST_FILESYSTEM_DECL path& make_preferred();  
#   endif
BOOST_FILESYSTEM_DECL path& remove_filename();
BOOST_FILESYSTEM_DECL path& remove_trailing_separator();
BOOST_FILESYSTEM_DECL path& replace_extension(const path& new_extension = path());
void swap(path& rhs) BOOST_NOEXCEPT { m_pathname.swap(rhs.m_pathname); }






const string_type&  native() const BOOST_NOEXCEPT  { return m_pathname; }
const value_type*   c_str() const BOOST_NOEXCEPT   { return m_pathname.c_str(); }
string_type::size_type size() const BOOST_NOEXCEPT { return m_pathname.size(); }

template <class String>
String string() const;

template <class String>
String string(const codecvt_type& cvt) const;

#   ifdef BOOST_WINDOWS_API
std::string string() const
{
std::string tmp;
if (!m_pathname.empty())
path_traits::convert(m_pathname.c_str(), m_pathname.c_str()+m_pathname.size(),
tmp);
return tmp;
}
std::string string(const codecvt_type& cvt) const
{
std::string tmp;
if (!m_pathname.empty())
path_traits::convert(m_pathname.c_str(), m_pathname.c_str()+m_pathname.size(),
tmp, cvt);
return tmp;
}

const std::wstring& wstring() const { return m_pathname; }
const std::wstring& wstring(const codecvt_type&) const { return m_pathname; }
#   else   
const std::string& string() const { return m_pathname; }
const std::string& string(const codecvt_type&) const { return m_pathname; }

std::wstring wstring() const
{
std::wstring tmp;
if (!m_pathname.empty())
path_traits::convert(m_pathname.c_str(), m_pathname.c_str()+m_pathname.size(),
tmp);
return tmp;
}
std::wstring wstring(const codecvt_type& cvt) const
{
std::wstring tmp;
if (!m_pathname.empty())
path_traits::convert(m_pathname.c_str(), m_pathname.c_str()+m_pathname.size(),
tmp, cvt);
return tmp;
}
#   endif


#   ifdef BOOST_WINDOWS_API
BOOST_FILESYSTEM_DECL path generic_path() const;
#   else
path generic_path() const { return path(*this); }
#   endif

template <class String>
String generic_string() const;

template <class String>
String generic_string(const codecvt_type& cvt) const;

#   ifdef BOOST_WINDOWS_API
std::string   generic_string() const { return generic_path().string(); }
std::string   generic_string(const codecvt_type& cvt) const { return generic_path().string(cvt); }
std::wstring  generic_wstring() const { return generic_path().wstring(); }
std::wstring  generic_wstring(const codecvt_type&) const { return generic_wstring(); }
#   else 
const std::string&  generic_string() const  { return m_pathname; }
const std::string&  generic_string(const codecvt_type&) const  { return m_pathname; }
std::wstring  generic_wstring() const { return this->wstring(); }
std::wstring  generic_wstring(const codecvt_type& cvt) const { return this->wstring(cvt); }
#   endif


BOOST_FILESYSTEM_DECL int compare(const path& p) const BOOST_NOEXCEPT;  
int compare(const std::string& s) const { return compare(path(s)); }
int compare(const value_type* s) const  { return compare(path(s)); }


BOOST_FILESYSTEM_DECL path  root_path() const;
BOOST_FILESYSTEM_DECL path  root_name() const;         
BOOST_FILESYSTEM_DECL path  root_directory() const;    
BOOST_FILESYSTEM_DECL path  relative_path() const;
BOOST_FILESYSTEM_DECL path  parent_path() const;
BOOST_FILESYSTEM_DECL path  filename() const;          
BOOST_FILESYSTEM_DECL path  stem() const;              
BOOST_FILESYSTEM_DECL path  extension() const;         


bool empty() const BOOST_NOEXCEPT { return m_pathname.empty(); }
bool filename_is_dot() const;
bool filename_is_dot_dot() const;
bool has_root_path() const       { return has_root_directory() || has_root_name(); }
bool has_root_name() const       { return !root_name().empty(); }
bool has_root_directory() const  { return !root_directory().empty(); }
bool has_relative_path() const   { return !relative_path().empty(); }
bool has_parent_path() const     { return !parent_path().empty(); }
bool has_filename() const        { return !m_pathname.empty(); }
bool has_stem() const            { return !stem().empty(); }
bool has_extension() const       { return !extension().empty(); }
bool is_relative() const         { return !is_absolute(); }
bool is_absolute() const
{
#     if defined(BOOST_WINDOWS_API) && !defined(UNDER_CE)
return has_root_name() && has_root_directory();
#     else
return has_root_directory();
#     endif
}


BOOST_FILESYSTEM_DECL path lexically_normal() const;
BOOST_FILESYSTEM_DECL path lexically_relative(const path& base) const;
path lexically_proximate(const path& base) const
{
path tmp(lexically_relative(base));
return tmp.empty() ? *this : tmp;
}


class iterator;
typedef iterator const_iterator;
class reverse_iterator;
typedef reverse_iterator const_reverse_iterator;

BOOST_FILESYSTEM_DECL iterator begin() const;
BOOST_FILESYSTEM_DECL iterator end() const;
reverse_iterator rbegin() const;
reverse_iterator rend() const;


static BOOST_FILESYSTEM_DECL std::locale imbue(const std::locale& loc);
static BOOST_FILESYSTEM_DECL const codecvt_type&  codecvt();


# if defined(BOOST_FILESYSTEM_DEPRECATED) && defined(BOOST_FILESYSTEM_NO_DEPRECATED)
#   error both BOOST_FILESYSTEM_DEPRECATED and BOOST_FILESYSTEM_NO_DEPRECATED are defined
# endif

# if !defined(BOOST_FILESYSTEM_NO_DEPRECATED)
path&  normalize()              {
path tmp(lexically_normal());
m_pathname.swap(tmp.m_pathname);
return *this;
}
path&  remove_leaf()            { return remove_filename(); }
path   leaf() const             { return filename(); }
path   branch_path() const      { return parent_path(); }
path   generic() const          { return generic_path(); }
bool   has_leaf() const         { return !m_pathname.empty(); }
bool   has_branch_path() const  { return !parent_path().empty(); }
bool   is_complete() const      { return is_absolute(); }
# endif

# if defined(BOOST_FILESYSTEM_DEPRECATED)
std::string file_string() const               { return string(); }
std::string directory_string() const          { return string(); }
std::string native_file_string() const        { return string(); }
std::string native_directory_string() const   { return string(); }
string_type external_file_string() const      { return native(); }
string_type external_directory_string() const { return native(); }

# endif


private:

#   if defined(_MSC_VER)
#     pragma warning(push) 
#     pragma warning(disable : 4251) 
#   endif                            

string_type  m_pathname;  
#   if defined(_MSC_VER)
#     pragma warning(pop) 
#   endif

BOOST_FILESYSTEM_DECL string_type::size_type m_append_separator_if_needed();

BOOST_FILESYSTEM_DECL void m_erase_redundant_separator(string_type::size_type sep_pos);
BOOST_FILESYSTEM_DECL string_type::size_type m_parent_path_end() const;

friend class iterator;
friend bool operator<(const path& lhs, const path& rhs);

static BOOST_FILESYSTEM_DECL void m_path_iterator_increment(path::iterator& it);
static BOOST_FILESYSTEM_DECL void m_path_iterator_decrement(path::iterator& it);

};  

namespace detail
{
BOOST_FILESYSTEM_DECL
int lex_compare(path::iterator first1, path::iterator last1,
path::iterator first2, path::iterator last2);
BOOST_FILESYSTEM_DECL
const path&  dot_path();
BOOST_FILESYSTEM_DECL
const path&  dot_dot_path();
}

# ifndef BOOST_FILESYSTEM_NO_DEPRECATED
typedef path wpath;
# endif


class path::iterator
: public boost::iterator_facade<
path::iterator,
path const,
boost::bidirectional_traversal_tag >
{
private:
friend class boost::iterator_core_access;
friend class boost::filesystem::path;
friend class boost::filesystem::path::reverse_iterator;
friend void m_path_iterator_increment(path::iterator & it);
friend void m_path_iterator_decrement(path::iterator & it);

const path& dereference() const { return m_element; }

bool equal(const iterator & rhs) const
{
return m_path_ptr == rhs.m_path_ptr && m_pos == rhs.m_pos;
}

void increment() { m_path_iterator_increment(*this); }
void decrement() { m_path_iterator_decrement(*this); }

path                    m_element;   
const path*             m_path_ptr;  
string_type::size_type  m_pos;       
}; 


class path::reverse_iterator
: public boost::iterator_facade<
path::reverse_iterator,
path const,
boost::bidirectional_traversal_tag >
{
public:
explicit reverse_iterator(iterator itr) : m_itr(itr)
{
if (itr != itr.m_path_ptr->begin())
m_element = *--itr;
}

private:
friend class boost::iterator_core_access;
friend class boost::filesystem::path;

const path& dereference() const { return m_element; }
bool equal(const reverse_iterator& rhs) const { return m_itr == rhs.m_itr; }
void increment()
{
--m_itr;
if (m_itr != m_itr.m_path_ptr->begin())
{
iterator tmp = m_itr;
m_element = *--tmp;
}
}
void decrement()
{
m_element = *m_itr;
++m_itr;
}

iterator m_itr;
path     m_element;

}; 


inline bool lexicographical_compare(path::iterator first1, path::iterator last1,
path::iterator first2, path::iterator last2)
{ return detail::lex_compare(first1, last1, first2, last2) < 0; }

inline bool operator==(const path& lhs, const path& rhs)              {return lhs.compare(rhs) == 0;}
inline bool operator==(const path& lhs, const path::string_type& rhs) {return lhs.compare(rhs) == 0;}
inline bool operator==(const path::string_type& lhs, const path& rhs) {return rhs.compare(lhs) == 0;}
inline bool operator==(const path& lhs, const path::value_type* rhs)  {return lhs.compare(rhs) == 0;}
inline bool operator==(const path::value_type* lhs, const path& rhs)  {return rhs.compare(lhs) == 0;}

inline bool operator!=(const path& lhs, const path& rhs)              {return lhs.compare(rhs) != 0;}
inline bool operator!=(const path& lhs, const path::string_type& rhs) {return lhs.compare(rhs) != 0;}
inline bool operator!=(const path::string_type& lhs, const path& rhs) {return rhs.compare(lhs) != 0;}
inline bool operator!=(const path& lhs, const path::value_type* rhs)  {return lhs.compare(rhs) != 0;}
inline bool operator!=(const path::value_type* lhs, const path& rhs)  {return rhs.compare(lhs) != 0;}


inline bool operator<(const path& lhs, const path& rhs)  {return lhs.compare(rhs) < 0;}
inline bool operator<=(const path& lhs, const path& rhs) {return !(rhs < lhs);}
inline bool operator> (const path& lhs, const path& rhs) {return rhs < lhs;}
inline bool operator>=(const path& lhs, const path& rhs) {return !(lhs < rhs);}

inline std::size_t hash_value(const path& x) BOOST_NOEXCEPT
{
# ifdef BOOST_WINDOWS_API
std::size_t seed = 0;
for(const path::value_type* it = x.c_str(); *it; ++it)
hash_combine(seed, *it == L'/' ? L'\\' : *it);
return seed;
# else   
return hash_range(x.native().begin(), x.native().end());
# endif
}

inline void swap(path& lhs, path& rhs) BOOST_NOEXCEPT { lhs.swap(rhs); }

inline path operator/(const path& lhs, const path& rhs)
{
path p = lhs;
p /= rhs;
return p;
}
# if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
inline path operator/(path&& lhs, const path& rhs)
{
lhs /= rhs;
return std::move(lhs);
}
# endif


template <class Char, class Traits>
inline std::basic_ostream<Char, Traits>&
operator<<(std::basic_ostream<Char, Traits>& os, const path& p)
{
return os
<< boost::io::quoted(p.template string<std::basic_string<Char> >(), static_cast<Char>('&'));
}

template <class Char, class Traits>
inline std::basic_istream<Char, Traits>&
operator>>(std::basic_istream<Char, Traits>& is, path& p)
{
std::basic_string<Char> str;
is >> boost::io::quoted(str, static_cast<Char>('&'));
p = str;
return is;
}



BOOST_FILESYSTEM_DECL bool portable_posix_name(const std::string & name);
BOOST_FILESYSTEM_DECL bool windows_name(const std::string & name);
BOOST_FILESYSTEM_DECL bool portable_name(const std::string & name);
BOOST_FILESYSTEM_DECL bool portable_directory_name(const std::string & name);
BOOST_FILESYSTEM_DECL bool portable_file_name(const std::string & name);
BOOST_FILESYSTEM_DECL bool native(const std::string & name);

namespace detail
{

inline bool is_directory_separator(path::value_type c) BOOST_NOEXCEPT
{
return c == path::separator
#     ifdef BOOST_WINDOWS_API
|| c == path::preferred_separator
#     endif
;
}
inline bool is_element_separator(path::value_type c) BOOST_NOEXCEPT
{
return c == path::separator
#     ifdef BOOST_WINDOWS_API
|| c == path::preferred_separator || c == L':'
#     endif
;
}
}  


inline path::reverse_iterator path::rbegin() const { return reverse_iterator(end()); }
inline path::reverse_iterator path::rend() const   { return reverse_iterator(begin()); }

inline bool path::filename_is_dot() const
{
path p(filename());
return p.size() == 1 && *p.c_str() == dot;
}

inline bool path::filename_is_dot_dot() const
{
return size() >= 2 && m_pathname[size()-1] == dot && m_pathname[size()-2] == dot
&& (m_pathname.size() == 2 || detail::is_element_separator(m_pathname[size()-3]));
}


template <class InputIterator>
path& path::append(InputIterator begin, InputIterator end)
{
if (begin == end)
return *this;
string_type::size_type sep_pos(m_append_separator_if_needed());
std::basic_string<typename std::iterator_traits<InputIterator>::value_type>
seq(begin, end);
path_traits::convert(seq.c_str(), seq.c_str()+seq.size(), m_pathname);
if (sep_pos)
m_erase_redundant_separator(sep_pos);
return *this;
}

template <class InputIterator>
path& path::append(InputIterator begin, InputIterator end, const codecvt_type& cvt)
{
if (begin == end)
return *this;
string_type::size_type sep_pos(m_append_separator_if_needed());
std::basic_string<typename std::iterator_traits<InputIterator>::value_type>
seq(begin, end);
path_traits::convert(seq.c_str(), seq.c_str()+seq.size(), m_pathname, cvt);
if (sep_pos)
m_erase_redundant_separator(sep_pos);
return *this;
}

template <class Source>
path& path::append(Source const& source)
{
if (path_traits::empty(source))
return *this;
string_type::size_type sep_pos(m_append_separator_if_needed());
path_traits::dispatch(source, m_pathname);
if (sep_pos)
m_erase_redundant_separator(sep_pos);
return *this;
}

template <class Source>
path& path::append(Source const& source, const codecvt_type& cvt)
{
if (path_traits::empty(source))
return *this;
string_type::size_type sep_pos(m_append_separator_if_needed());
path_traits::dispatch(source, m_pathname, cvt);
if (sep_pos)
m_erase_redundant_separator(sep_pos);
return *this;
}


template <> inline
std::string path::string<std::string>() const
{ return string(); }

template <> inline
std::wstring path::string<std::wstring>() const
{ return wstring(); }

template <> inline
std::string path::string<std::string>(const codecvt_type& cvt) const
{ return string(cvt); }

template <> inline
std::wstring path::string<std::wstring>(const codecvt_type& cvt) const
{ return wstring(cvt); }

template <> inline
std::string path::generic_string<std::string>() const
{ return generic_string(); }

template <> inline
std::wstring path::generic_string<std::wstring>() const
{ return generic_wstring(); }

template <> inline
std::string path::generic_string<std::string>(const codecvt_type& cvt) const
{ return generic_string(cvt); }

template <> inline
std::wstring path::generic_string<std::wstring>(const codecvt_type& cvt) const
{ return generic_wstring(cvt); }


namespace path_traits
{  

inline void convert(const char* from,
const char* from_end,    
std::wstring & to)
{
convert(from, from_end, to, path::codecvt());
}

inline void convert(const wchar_t* from,
const wchar_t* from_end,  
std::string & to)
{
convert(from, from_end, to, path::codecvt());
}

inline void convert(const char* from,
std::wstring & to)
{
BOOST_ASSERT(!!from);
convert(from, 0, to, path::codecvt());
}

inline void convert(const wchar_t* from,
std::string & to)
{
BOOST_ASSERT(!!from);
convert(from, 0, to, path::codecvt());
}
}  
}  
}  


#include <boost/config/abi_suffix.hpp> 

#endif  

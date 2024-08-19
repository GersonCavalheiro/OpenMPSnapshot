



#ifndef BOOST_REGEX_FORMAT_HPP
#define BOOST_REGEX_FORMAT_HPP

#include <boost/type_traits/is_pointer.hpp>
#include <boost/type_traits/is_function.hpp>
#include <boost/type_traits/is_class.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/type_traits/remove_pointer.hpp>
#include <boost/type_traits/remove_cv.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/not.hpp>
#ifndef BOOST_NO_SFINAE
#include <boost/mpl/has_xxx.hpp>
#endif
#include <boost/ref.hpp>

namespace boost{

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable: 4103)
#endif
#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_PREFIX
#endif
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

template <class BidiIterator, class Allocator = BOOST_DEDUCED_TYPENAME std::vector<sub_match<BidiIterator> >::allocator_type >
class match_results;

namespace BOOST_REGEX_DETAIL_NS{

template <class charT>
struct trivial_format_traits
{
typedef charT char_type;

static std::ptrdiff_t length(const charT* p)
{
return global_length(p);
}
static charT tolower(charT c)
{
return ::boost::BOOST_REGEX_DETAIL_NS::global_lower(c);
}
static charT toupper(charT c)
{
return ::boost::BOOST_REGEX_DETAIL_NS::global_upper(c);
}
static int value(const charT c, int radix)
{
int result = global_value(c);
return result >= radix ? -1 : result;
}
int toi(const charT*& p1, const charT* p2, int radix)const
{
return (int)global_toi(p1, p2, radix, *this);
}
};

#ifdef BOOST_MSVC
#  pragma warning(push)
#pragma warning(disable:26812)
#endif
template <class OutputIterator, class Results, class traits, class ForwardIter>
class basic_regex_formatter
{
public:
typedef typename traits::char_type char_type;
basic_regex_formatter(OutputIterator o, const Results& r, const traits& t)
: m_traits(t), m_results(r), m_out(o), m_position(), m_end(), m_flags(), m_state(output_copy), m_restore_state(output_copy), m_have_conditional(false) {}
OutputIterator format(ForwardIter p1, ForwardIter p2, match_flag_type f);
OutputIterator format(ForwardIter p1, match_flag_type f)
{
return format(p1, p1 + m_traits.length(p1), f);
}
private:
typedef typename Results::value_type sub_match_type;
enum output_state
{
output_copy,
output_next_lower,
output_next_upper,
output_lower,
output_upper,
output_none
};

void put(char_type c);
void put(const sub_match_type& sub);
void format_all();
void format_perl();
void format_escape();
void format_conditional();
void format_until_scope_end();
bool handle_perl_verb(bool have_brace);

inline typename Results::value_type const& get_named_sub(ForwardIter i, ForwardIter j, const mpl::false_&)
{
std::vector<char_type> v(i, j);
return (i != j) ? this->m_results.named_subexpression(&v[0], &v[0] + v.size())
: this->m_results.named_subexpression(static_cast<const char_type*>(0), static_cast<const char_type*>(0));
}
inline typename Results::value_type const& get_named_sub(ForwardIter i, ForwardIter j, const mpl::true_&)
{
return this->m_results.named_subexpression(i, j);
}
inline typename Results::value_type const& get_named_sub(ForwardIter i, ForwardIter j)
{
typedef typename boost::is_convertible<ForwardIter, const char_type*>::type tag_type;
return get_named_sub(i, j, tag_type());
}
inline int get_named_sub_index(ForwardIter i, ForwardIter j, const mpl::false_&)
{
std::vector<char_type> v(i, j);
return (i != j) ? this->m_results.named_subexpression_index(&v[0], &v[0] + v.size())
: this->m_results.named_subexpression_index(static_cast<const char_type*>(0), static_cast<const char_type*>(0));
}
inline int get_named_sub_index(ForwardIter i, ForwardIter j, const mpl::true_&)
{
return this->m_results.named_subexpression_index(i, j);
}
inline int get_named_sub_index(ForwardIter i, ForwardIter j)
{
typedef typename boost::is_convertible<ForwardIter, const char_type*>::type tag_type;
return get_named_sub_index(i, j, tag_type());
}
#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable:4244)
#endif
inline int toi(ForwardIter& i, ForwardIter j, int base, const boost::mpl::false_&)
{
if(i != j)
{
std::vector<char_type> v(i, j);
const char_type* start = &v[0];
const char_type* pos = start;
int r = (int)m_traits.toi(pos, &v[0] + v.size(), base);
std::advance(i, pos - start);
return r;
}
return -1;
}
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif
inline int toi(ForwardIter& i, ForwardIter j, int base, const boost::mpl::true_&)
{
return m_traits.toi(i, j, base);
}
inline int toi(ForwardIter& i, ForwardIter j, int base)
{
#if defined(_MSC_VER) && defined(__INTEL_COMPILER) && ((__INTEL_COMPILER == 9999) || (__INTEL_COMPILER == 1210))
return toi(i, j, base, mpl::false_());
#else
typedef typename boost::is_convertible<ForwardIter, const char_type*&>::type tag_type;
return toi(i, j, base, tag_type());
#endif
}

const traits&    m_traits;       
const Results&   m_results;     
OutputIterator   m_out;         
ForwardIter      m_position;  
ForwardIter      m_end;       
match_flag_type  m_flags;      
output_state     m_state;      
output_state     m_restore_state;  
bool             m_have_conditional; 
private:
basic_regex_formatter(const basic_regex_formatter&);
basic_regex_formatter& operator=(const basic_regex_formatter&);
};
#ifdef BOOST_MSVC
#  pragma warning(pop)
#endif

template <class OutputIterator, class Results, class traits, class ForwardIter>
OutputIterator basic_regex_formatter<OutputIterator, Results, traits, ForwardIter>::format(ForwardIter p1, ForwardIter p2, match_flag_type f)
{
m_position = p1;
m_end = p2;
m_flags = f;
format_all();
return m_out;
}

template <class OutputIterator, class Results, class traits, class ForwardIter>
void basic_regex_formatter<OutputIterator, Results, traits, ForwardIter>::format_all()
{
while(m_position != m_end)
{
switch(*m_position)
{
case '&':
if(m_flags & ::boost::regex_constants::format_sed)
{
++m_position;
put(m_results[0]);
break;
}
put(*m_position++);
break;
case '\\':
format_escape();
break;
case '(':
if(m_flags & boost::regex_constants::format_all)
{
++m_position;
bool have_conditional = m_have_conditional;
m_have_conditional = false;
format_until_scope_end();
m_have_conditional = have_conditional;
if(m_position == m_end)
return;
BOOST_ASSERT(*m_position == static_cast<char_type>(')'));
++m_position;  
break;
}
put(*m_position);
++m_position;
break;
case ')':
if(m_flags & boost::regex_constants::format_all)
{
return;
}
put(*m_position);
++m_position;
break;
case ':':
if((m_flags & boost::regex_constants::format_all) && m_have_conditional)
{
return;
}
put(*m_position);
++m_position;
break;
case '?':
if(m_flags & boost::regex_constants::format_all)
{
++m_position;
format_conditional();
break;
}
put(*m_position);
++m_position;
break;
case '$':
if((m_flags & format_sed) == 0)
{
format_perl();
break;
}
BOOST_FALLTHROUGH;
default:
put(*m_position);
++m_position;
break;
}
}
}

template <class OutputIterator, class Results, class traits, class ForwardIter>
void basic_regex_formatter<OutputIterator, Results, traits, ForwardIter>::format_perl()
{
BOOST_ASSERT(*m_position == '$');
if(++m_position == m_end)
{
--m_position;
put(*m_position);
++m_position;
return;
}
bool have_brace = false;
ForwardIter save_position = m_position;
switch(*m_position)
{
case '&':
++m_position;
put(this->m_results[0]);
break;
case '`':
++m_position;
put(this->m_results.prefix());
break;
case '\'':
++m_position;
put(this->m_results.suffix());
break;
case '$':
put(*m_position++);
break;
case '+':
if((++m_position != m_end) && (*m_position == '{'))
{
ForwardIter base = ++m_position;
while((m_position != m_end) && (*m_position != '}')) ++m_position;
if(m_position != m_end)
{
put(get_named_sub(base, m_position));
++m_position;
break;
}
else
{
m_position = --base;
}
}
put((this->m_results)[this->m_results.size() > 1 ? static_cast<int>(this->m_results.size() - 1) : 1]);
break;
case '{':
have_brace = true;
++m_position;
BOOST_FALLTHROUGH;
default:
{
std::ptrdiff_t len = ::boost::BOOST_REGEX_DETAIL_NS::distance(m_position, m_end);
int v = this->toi(m_position, m_position + len, 10);
if((v < 0) || (have_brace && ((m_position == m_end) || (*m_position != '}'))))
{
if(!handle_perl_verb(have_brace))
{
m_position = --save_position;
put(*m_position);
++m_position;
}
break;
}
put(this->m_results[v]);
if(have_brace)
++m_position;
}
}
}

template <class OutputIterator, class Results, class traits, class ForwardIter>
bool basic_regex_formatter<OutputIterator, Results, traits, ForwardIter>::handle_perl_verb(bool have_brace)
{
static const char_type MATCH[] = { 'M', 'A', 'T', 'C', 'H' };
static const char_type PREMATCH[] = { 'P', 'R', 'E', 'M', 'A', 'T', 'C', 'H' };
static const char_type POSTMATCH[] = { 'P', 'O', 'S', 'T', 'M', 'A', 'T', 'C', 'H' };
static const char_type LAST_PAREN_MATCH[] = { 'L', 'A', 'S', 'T', '_', 'P', 'A', 'R', 'E', 'N', '_', 'M', 'A', 'T', 'C', 'H' };
static const char_type LAST_SUBMATCH_RESULT[] = { 'L', 'A', 'S', 'T', '_', 'S', 'U', 'B', 'M', 'A', 'T', 'C', 'H', '_', 'R', 'E', 'S', 'U', 'L', 'T' };
static const char_type LAST_SUBMATCH_RESULT_ALT[] = { '^', 'N' };

if(m_position == m_end)
return false;
if(have_brace && (*m_position == '^'))
++m_position;

std::ptrdiff_t max_len = m_end - m_position;

if((max_len >= 5) && std::equal(m_position, m_position + 5, MATCH))
{
m_position += 5;
if(have_brace)
{
if((m_position != m_end) && (*m_position == '}'))
++m_position;
else
{
m_position -= 5;
return false;
}
}
put(this->m_results[0]);
return true;
}
if((max_len >= 8) && std::equal(m_position, m_position + 8, PREMATCH))
{
m_position += 8;
if(have_brace)
{
if((m_position != m_end) && (*m_position == '}'))
++m_position;
else
{
m_position -= 8;
return false;
}
}
put(this->m_results.prefix());
return true;
}
if((max_len >= 9) && std::equal(m_position, m_position + 9, POSTMATCH))
{
m_position += 9;
if(have_brace)
{
if((m_position != m_end) && (*m_position == '}'))
++m_position;
else
{
m_position -= 9;
return false;
}
}
put(this->m_results.suffix());
return true;
}
if((max_len >= 16) && std::equal(m_position, m_position + 16, LAST_PAREN_MATCH))
{
m_position += 16;
if(have_brace)
{
if((m_position != m_end) && (*m_position == '}'))
++m_position;
else
{
m_position -= 16;
return false;
}
}
put((this->m_results)[this->m_results.size() > 1 ? static_cast<int>(this->m_results.size() - 1) : 1]);
return true;
}
if((max_len >= 20) && std::equal(m_position, m_position + 20, LAST_SUBMATCH_RESULT))
{
m_position += 20;
if(have_brace)
{
if((m_position != m_end) && (*m_position == '}'))
++m_position;
else
{
m_position -= 20;
return false;
}
}
put(this->m_results.get_last_closed_paren());
return true;
}
if((max_len >= 2) && std::equal(m_position, m_position + 2, LAST_SUBMATCH_RESULT_ALT))
{
m_position += 2;
if(have_brace)
{
if((m_position != m_end) && (*m_position == '}'))
++m_position;
else
{
m_position -= 2;
return false;
}
}
put(this->m_results.get_last_closed_paren());
return true;
}
return false;
}

template <class OutputIterator, class Results, class traits, class ForwardIter>
void basic_regex_formatter<OutputIterator, Results, traits, ForwardIter>::format_escape()
{
if(++m_position == m_end)
{
put(static_cast<char_type>('\\'));
return;
}
switch(*m_position)
{
case 'a':
put(static_cast<char_type>('\a'));
++m_position;
break;
case 'f':
put(static_cast<char_type>('\f'));
++m_position;
break;
case 'n':
put(static_cast<char_type>('\n'));
++m_position;
break;
case 'r':
put(static_cast<char_type>('\r'));
++m_position;
break;
case 't':
put(static_cast<char_type>('\t'));
++m_position;
break;
case 'v':
put(static_cast<char_type>('\v'));
++m_position;
break;
case 'x':
if(++m_position == m_end)
{
put(static_cast<char_type>('x'));
return;
}
if(*m_position == static_cast<char_type>('{'))
{
++m_position;
int val = this->toi(m_position, m_end, 16);
if(val < 0)
{
put(static_cast<char_type>('x'));
put(static_cast<char_type>('{'));
return;
}
if((m_position == m_end) || (*m_position != static_cast<char_type>('}')))
{
--m_position;
while(*m_position != static_cast<char_type>('\\'))
--m_position;
++m_position;
put(*m_position++);
return;
}
++m_position;
put(static_cast<char_type>(val));
return;
}
else
{
std::ptrdiff_t len = ::boost::BOOST_REGEX_DETAIL_NS::distance(m_position, m_end);
len = (std::min)(static_cast<std::ptrdiff_t>(2), len);
int val = this->toi(m_position, m_position + len, 16);
if(val < 0)
{
--m_position;
put(*m_position++);
return;
}
put(static_cast<char_type>(val));
}
break;
case 'c':
if(++m_position == m_end)
{
--m_position;
put(*m_position++);
return;
}
put(static_cast<char_type>(*m_position++ % 32));
break;
case 'e':
put(static_cast<char_type>(27));
++m_position;
break;
default:
if((m_flags & boost::regex_constants::format_sed) == 0)
{
bool breakout = false;
switch(*m_position)
{
case 'l':
++m_position;
m_restore_state = m_state;
m_state = output_next_lower;
breakout = true;
break;
case 'L':
++m_position;
m_state = output_lower;
breakout = true;
break;
case 'u':
++m_position;
m_restore_state = m_state;
m_state = output_next_upper;
breakout = true;
break;
case 'U':
++m_position;
m_state = output_upper;
breakout = true;
break;
case 'E':
++m_position;
m_state = output_copy;
breakout = true;
break;
}
if(breakout)
break;
}
std::ptrdiff_t len = ::boost::BOOST_REGEX_DETAIL_NS::distance(m_position, m_end);
len = (std::min)(static_cast<std::ptrdiff_t>(1), len);
int v = this->toi(m_position, m_position+len, 10);
if((v > 0) || ((v == 0) && (m_flags & ::boost::regex_constants::format_sed)))
{
put(m_results[v]);
break;
}
else if(v == 0)
{
--m_position;
len = ::boost::BOOST_REGEX_DETAIL_NS::distance(m_position, m_end);
len = (std::min)(static_cast<std::ptrdiff_t>(4), len);
v = this->toi(m_position, m_position + len, 8);
BOOST_ASSERT(v >= 0);
put(static_cast<char_type>(v));
break;
}
put(*m_position++);
break;
}
}

template <class OutputIterator, class Results, class traits, class ForwardIter>
void basic_regex_formatter<OutputIterator, Results, traits, ForwardIter>::format_conditional()
{
if(m_position == m_end)
{
put(static_cast<char_type>('?'));
return;
}
int v;
if(*m_position == '{')
{
ForwardIter base = m_position;
++m_position;
v = this->toi(m_position, m_end, 10);
if(v < 0)
{
while((m_position != m_end) && (*m_position != '}'))
++m_position;
v = this->get_named_sub_index(base + 1, m_position);
}
if((v < 0) || (*m_position != '}'))
{
m_position = base;
put(static_cast<char_type>('?'));
return;
}
++m_position;
}
else
{
std::ptrdiff_t len = ::boost::BOOST_REGEX_DETAIL_NS::distance(m_position, m_end);
len = (std::min)(static_cast<std::ptrdiff_t>(2), len);
v = this->toi(m_position, m_position + len, 10);
}
if(v < 0)
{
put(static_cast<char_type>('?'));
return;
}

if(m_results[v].matched)
{
m_have_conditional = true;
format_all();
m_have_conditional = false;
if((m_position != m_end) && (*m_position == static_cast<char_type>(':')))
{
++m_position;
output_state saved_state = m_state;
m_state = output_none;
format_until_scope_end();
m_state = saved_state;
}
}
else
{
output_state saved_state = m_state;
m_state = output_none;
m_have_conditional = true;
format_all();
m_have_conditional = false;
m_state = saved_state;
if((m_position != m_end) && (*m_position == static_cast<char_type>(':')))
{
++m_position;
format_until_scope_end();
}
}
}

template <class OutputIterator, class Results, class traits, class ForwardIter>
void basic_regex_formatter<OutputIterator, Results, traits, ForwardIter>::format_until_scope_end()
{
do
{
format_all();
if((m_position == m_end) || (*m_position == static_cast<char_type>(')')))
return;
put(*m_position++);
}while(m_position != m_end);
}

template <class OutputIterator, class Results, class traits, class ForwardIter>
void basic_regex_formatter<OutputIterator, Results, traits, ForwardIter>::put(char_type c)
{
switch(this->m_state)
{
case output_none:
return;
case output_next_lower:
c = m_traits.tolower(c);
this->m_state = m_restore_state;
break;
case output_next_upper:
c = m_traits.toupper(c);
this->m_state = m_restore_state;
break;
case output_lower:
c = m_traits.tolower(c);
break;
case output_upper:
c = m_traits.toupper(c);
break;
default:
break;
}
*m_out = c;
++m_out;
}

template <class OutputIterator, class Results, class traits, class ForwardIter>
void basic_regex_formatter<OutputIterator, Results, traits, ForwardIter>::put(const sub_match_type& sub)
{
typedef typename sub_match_type::iterator iterator_type;
iterator_type i = sub.first;
while(i != sub.second)
{
put(*i);
++i;
}
}

template <class S>
class string_out_iterator
{
S* out;
public:
string_out_iterator(S& s) : out(&s) {}
string_out_iterator& operator++() { return *this; }
string_out_iterator& operator++(int) { return *this; }
string_out_iterator& operator*() { return *this; }
string_out_iterator& operator=(typename S::value_type v) 
{ 
out->append(1, v); 
return *this; 
}

typedef std::ptrdiff_t difference_type;
typedef typename S::value_type value_type;
typedef value_type* pointer;
typedef value_type& reference;
typedef std::output_iterator_tag iterator_category;
};

template <class OutputIterator, class Iterator, class Alloc, class ForwardIter, class traits>
OutputIterator regex_format_imp(OutputIterator out,
const match_results<Iterator, Alloc>& m,
ForwardIter p1, ForwardIter p2,
match_flag_type flags,
const traits& t
)
{
if(flags & regex_constants::format_literal)
{
return BOOST_REGEX_DETAIL_NS::copy(p1, p2, out);
}

BOOST_REGEX_DETAIL_NS::basic_regex_formatter<
OutputIterator, 
match_results<Iterator, Alloc>, 
traits, ForwardIter> f(out, m, t);
return f.format(p1, p2, flags);
}

#ifndef BOOST_NO_SFINAE

BOOST_MPL_HAS_XXX_TRAIT_DEF(const_iterator)

struct any_type 
{
template <class T>
any_type(const T&); 
template <class T, class U>
any_type(const T&, const U&); 
template <class T, class U, class V>
any_type(const T&, const U&, const V&); 
};
typedef char no_type;
typedef char (&unary_type)[2];
typedef char (&binary_type)[3];
typedef char (&ternary_type)[4];

no_type check_is_formatter(unary_type, binary_type, ternary_type);
template<typename T>
unary_type check_is_formatter(T const &, binary_type, ternary_type);
template<typename T>
binary_type check_is_formatter(unary_type, T const &, ternary_type);
template<typename T, typename U>
binary_type check_is_formatter(T const &, U const &, ternary_type);
template<typename T>
ternary_type check_is_formatter(unary_type, binary_type, T const &);
template<typename T, typename U>
ternary_type check_is_formatter(T const &, binary_type, U const &);
template<typename T, typename U>
ternary_type check_is_formatter(unary_type, T const &, U const &);
template<typename T, typename U, typename V>
ternary_type check_is_formatter(T const &, U const &, V const &);

struct unary_binary_ternary
{
typedef unary_type (*unary_fun)(any_type);
typedef binary_type (*binary_fun)(any_type, any_type);
typedef ternary_type (*ternary_fun)(any_type, any_type, any_type);
operator unary_fun();
operator binary_fun();
operator ternary_fun();
};

template<typename Formatter, bool IsFunction = boost::is_function<Formatter>::value>
struct formatter_wrapper
: Formatter
, unary_binary_ternary
{
formatter_wrapper(){}
};

template<typename Formatter>
struct formatter_wrapper<Formatter, true>
: unary_binary_ternary
{
operator Formatter *();
};

template<typename Formatter>
struct formatter_wrapper<Formatter *, false>
: unary_binary_ternary
{
operator Formatter *();
};

template <class F, class M, class O>
struct format_traits_imp
{
private:
BOOST_STATIC_ASSERT((::boost::is_pointer<F>::value || ::boost::is_function<F>::value || ::boost::is_class<F>::value));
static formatter_wrapper<typename unwrap_reference<F>::type> f;
static M m;
static O out;
static boost::regex_constants::match_flag_type flags;
public:
BOOST_STATIC_CONSTANT(int, value = sizeof(check_is_formatter(f(m), f(m, out), f(m, out, flags))));
};

template <class F, class M, class O>
struct format_traits
{
public:
typedef typename boost::mpl::if_<
boost::mpl::and_<boost::is_pointer<F>, boost::mpl::not_<boost::is_function<typename boost::remove_pointer<F>::type> > >,
boost::mpl::int_<0>,
typename boost::mpl::if_<
has_const_iterator<F>,
boost::mpl::int_<1>,
boost::mpl::int_<format_traits_imp<F, M, O>::value>
>::type
>::type type;
BOOST_STATIC_ASSERT( boost::is_class<F>::value && !has_const_iterator<F>::value ? (type::value > 1) : true);
};

#else 

template <class F, class M, class O>
struct format_traits
{
public:
typedef typename boost::mpl::if_<
boost::is_pointer<F>,
boost::mpl::int_<0>,
boost::mpl::int_<1>
>::type type;
};

#endif 

template <class Base, class Match>
struct format_functor3
{
format_functor3(Base b) : func(b) {}
template <class OutputIter>
OutputIter operator()(const Match& m, OutputIter i, boost::regex_constants::match_flag_type f)
{
return boost::unwrap_ref(func)(m, i, f);
}
template <class OutputIter, class Traits>
OutputIter operator()(const Match& m, OutputIter i, boost::regex_constants::match_flag_type f, const Traits&)
{
return (*this)(m, i, f);
}
private:
Base func;
format_functor3(const format_functor3&);
format_functor3& operator=(const format_functor3&);
};

template <class Base, class Match>
struct format_functor2
{
format_functor2(Base b) : func(b) {}
template <class OutputIter>
OutputIter operator()(const Match& m, OutputIter i, boost::regex_constants::match_flag_type )
{
return boost::unwrap_ref(func)(m, i);
}
template <class OutputIter, class Traits>
OutputIter operator()(const Match& m, OutputIter i, boost::regex_constants::match_flag_type f, const Traits&)
{
return (*this)(m, i, f);
}
private:
Base func;
format_functor2(const format_functor2&);
format_functor2& operator=(const format_functor2&);
};

template <class Base, class Match>
struct format_functor1
{
format_functor1(Base b) : func(b) {}

template <class S, class OutputIter>
OutputIter do_format_string(const S& s, OutputIter i)
{
return BOOST_REGEX_DETAIL_NS::copy(s.begin(), s.end(), i);
}
template <class S, class OutputIter>
inline OutputIter do_format_string(const S* s, OutputIter i)
{
while(s && *s)
{
*i = *s;
++i;
++s;
}
return i;
}
template <class OutputIter>
OutputIter operator()(const Match& m, OutputIter i, boost::regex_constants::match_flag_type )
{
return do_format_string(boost::unwrap_ref(func)(m), i);
}
template <class OutputIter, class Traits>
OutputIter operator()(const Match& m, OutputIter i, boost::regex_constants::match_flag_type f, const Traits&)
{
return (*this)(m, i, f);
}
private:
Base func;
format_functor1(const format_functor1&);
format_functor1& operator=(const format_functor1&);
};

template <class charT, class Match, class Traits>
struct format_functor_c_string
{
format_functor_c_string(const charT* ps) : func(ps) {}

template <class OutputIter>
OutputIter operator()(const Match& m, OutputIter i, boost::regex_constants::match_flag_type f, const Traits& t = Traits())
{
const charT* end = func;
while(*end) ++end;
return regex_format_imp(i, m, func, end, f, t);
}
private:
const charT* func;
format_functor_c_string(const format_functor_c_string&);
format_functor_c_string& operator=(const format_functor_c_string&);
};

template <class Container, class Match, class Traits>
struct format_functor_container
{
format_functor_container(const Container& c) : func(c) {}

template <class OutputIter>
OutputIter operator()(const Match& m, OutputIter i, boost::regex_constants::match_flag_type f, const Traits& t = Traits())
{
return BOOST_REGEX_DETAIL_NS::regex_format_imp(i, m, func.begin(), func.end(), f, t);
}
private:
const Container& func;
format_functor_container(const format_functor_container&);
format_functor_container& operator=(const format_functor_container&);
};

template <class Func, class Match, class OutputIterator, class Traits = BOOST_REGEX_DETAIL_NS::trivial_format_traits<typename Match::char_type> >
struct compute_functor_type
{
typedef typename format_traits<Func, Match, OutputIterator>::type tag;
typedef typename boost::remove_cv< typename boost::remove_pointer<Func>::type>::type maybe_char_type;

typedef typename mpl::if_<
::boost::is_same<tag, mpl::int_<0> >, format_functor_c_string<maybe_char_type, Match, Traits>,
typename mpl::if_<
::boost::is_same<tag, mpl::int_<1> >, format_functor_container<Func, Match, Traits>,
typename mpl::if_<
::boost::is_same<tag, mpl::int_<2> >, format_functor1<Func, Match>,
typename mpl::if_<
::boost::is_same<tag, mpl::int_<3> >, format_functor2<Func, Match>, 
format_functor3<Func, Match>
>::type
>::type
>::type
>::type type;
};

} 

template <class OutputIterator, class Iterator, class Allocator, class Functor>
inline OutputIterator regex_format(OutputIterator out,
const match_results<Iterator, Allocator>& m,
Functor fmt,
match_flag_type flags = format_all
)
{
return m.format(out, fmt, flags);
}

template <class Iterator, class Allocator, class Functor>
inline std::basic_string<typename match_results<Iterator, Allocator>::char_type> regex_format(const match_results<Iterator, Allocator>& m, 
Functor fmt, 
match_flag_type flags = format_all)
{
return m.format(fmt, flags);
}

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable: 4103)
#endif
#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_SUFFIX
#endif
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

} 

#endif  







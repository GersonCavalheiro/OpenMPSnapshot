

#ifndef BOOST_REGEX_MATCHER_HPP
#define BOOST_REGEX_MATCHER_HPP

#include <boost/regex/v4/iterator_category.hpp>

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

#ifdef BOOST_MSVC
#  pragma warning(push)
#pragma warning(disable : 4251)
#if BOOST_MSVC < 1700
#     pragma warning(disable : 4231)
#endif
#  if BOOST_MSVC < 1600
#     pragma warning(disable : 4660)
#  endif
#if BOOST_MSVC < 1910
#pragma warning(disable:4800)
#endif
#endif

namespace boost{
namespace BOOST_REGEX_DETAIL_NS{

BOOST_REGEX_DECL void BOOST_REGEX_CALL verify_options(boost::regex_constants::syntax_option_type ef, match_flag_type mf);
template <class charT>
inline bool can_start(charT c, const unsigned char* map, unsigned char mask)
{
return ((c < static_cast<charT>(0)) ? true : ((c >= static_cast<charT>(1 << CHAR_BIT)) ? true : map[c] & mask));
}
inline bool can_start(char c, const unsigned char* map, unsigned char mask)
{
return map[(unsigned char)c] & mask;
}
inline bool can_start(signed char c, const unsigned char* map, unsigned char mask)
{
return map[(unsigned char)c] & mask;
}
inline bool can_start(unsigned char c, const unsigned char* map, unsigned char mask)
{
return map[c] & mask;
}
inline bool can_start(unsigned short c, const unsigned char* map, unsigned char mask)
{
return ((c >= (1 << CHAR_BIT)) ? true : map[c] & mask);
}
#if !defined(__hpux) && !defined(__WINSCW__)
#if defined(WCHAR_MIN) && (WCHAR_MIN == 0) && !defined(BOOST_NO_INTRINSIC_WCHAR_T)
inline bool can_start(wchar_t c, const unsigned char* map, unsigned char mask)
{
return ((c >= static_cast<wchar_t>(1u << CHAR_BIT)) ? true : map[c] & mask);
}
#endif
#endif
#if !defined(BOOST_NO_INTRINSIC_WCHAR_T)
inline bool can_start(unsigned int c, const unsigned char* map, unsigned char mask)
{
return (((c >= static_cast<unsigned int>(1u << CHAR_BIT)) ? true : map[c] & mask));
}
#endif


#ifndef _RWSTD_VER
template <class C, class T, class A>
inline int string_compare(const std::basic_string<C,T,A>& s, const C* p)
{ 
if(0 == *p)
{
if(s.empty() || ((s.size() == 1) && (s[0] == 0)))
return 0;
}
return s.compare(p); 
}
#else
template <class C, class T, class A>
inline int string_compare(const std::basic_string<C,T,A>& s, const C* p)
{ 
if(0 == *p)
{
if(s.empty() || ((s.size() == 1) && (s[0] == 0)))
return 0;
}
return s.compare(p); 
}
inline int string_compare(const std::string& s, const char* p)
{ return std::strcmp(s.c_str(), p); }
# ifndef BOOST_NO_WREGEX
inline int string_compare(const std::wstring& s, const wchar_t* p)
{ return std::wcscmp(s.c_str(), p); }
#endif
#endif
template <class Seq, class C>
inline int string_compare(const Seq& s, const C* p)
{
std::size_t i = 0;
while((i < s.size()) && (p[i] == s[i]))
{
++i;
}
return (i == s.size()) ? -(int)p[i] : (int)s[i] - (int)p[i];
}
# define STR_COMP(s,p) string_compare(s,p)

template<class charT>
inline const charT* re_skip_past_null(const charT* p)
{
while (*p != static_cast<charT>(0)) ++p;
return ++p;
}

template <class iterator, class charT, class traits_type, class char_classT>
iterator BOOST_REGEX_CALL re_is_set_member(iterator next, 
iterator last, 
const re_set_long<char_classT>* set_, 
const regex_data<charT, traits_type>& e, bool icase)
{   
const charT* p = reinterpret_cast<const charT*>(set_+1);
iterator ptr;
unsigned int i;

if(next == last) return next;

typedef typename traits_type::string_type traits_string_type;
const ::boost::regex_traits_wrapper<traits_type>& traits_inst = *(e.m_ptraits);

(void)traits_inst;

for(i = 0; i < set_->csingles; ++i)
{
ptr = next;
if(*p == static_cast<charT>(0))
{
if(traits_inst.translate(*ptr, icase))
{
++p;
continue;
}
return set_->isnot ? next : (ptr == next) ? ++next : ptr;
}
else
{
while(*p && (ptr != last))
{
if(traits_inst.translate(*ptr, icase) != *p)
break;
++p;
++ptr;
}

if(*p == static_cast<charT>(0)) 
return set_->isnot ? next : (ptr == next) ? ++next : ptr;

p = re_skip_past_null(p);     
}
}

charT col = traits_inst.translate(*next, icase);


if(set_->cranges || set_->cequivalents)
{
traits_string_type s1;
if(set_->cranges)
{
if((e.m_flags & regex_constants::collate) == 0)
s1.assign(1, col);
else
{
charT a[2] = { col, charT(0), };
s1 = traits_inst.transform(a, a + 1);
}
for(i = 0; i < set_->cranges; ++i)
{
if(STR_COMP(s1, p) >= 0)
{
do{ ++p; }while(*p);
++p;
if(STR_COMP(s1, p) <= 0)
return set_->isnot ? next : ++next;
}
else
{
do{ ++p; }while(*p);
++p;
}
do{ ++p; }while(*p);
++p;
}
}
if(set_->cequivalents)
{
charT a[2] = { col, charT(0), };
s1 = traits_inst.transform_primary(a, a +1);
for(i = 0; i < set_->cequivalents; ++i)
{
if(STR_COMP(s1, p) == 0)
return set_->isnot ? next : ++next;
do{ ++p; }while(*p);
++p;
}
}
}
if(traits_inst.isctype(col, set_->cclasses) == true)
return set_->isnot ? next : ++next;
if((set_->cnclasses != 0) && (traits_inst.isctype(col, set_->cnclasses) == false))
return set_->isnot ? next : ++next;
return set_->isnot ? ++next : next;
}

template <class BidiIterator>
class repeater_count
{
repeater_count** stack;
repeater_count* next;
int state_id;
std::size_t count;        
BidiIterator start_pos;   

repeater_count* unwind_until(int n, repeater_count* p, int current_recursion_id)
{ 
while(p && (p->state_id != n))
{
if(-2 - current_recursion_id == p->state_id)
return 0;
p = p->next;
if(p && (p->state_id < 0))
{
p = unwind_until(p->state_id, p, current_recursion_id);
if(!p)
return p;
p = p->next;
}
}
return p;
}
public:
repeater_count(repeater_count** s) : stack(s), next(0), state_id(-1), count(0), start_pos() {}

repeater_count(int i, repeater_count** s, BidiIterator start, int current_recursion_id)
: start_pos(start)
{
state_id = i;
stack = s;
next = *stack;
*stack = this;
if((state_id > next->state_id) && (next->state_id >= 0))
count = 0;
else
{
repeater_count* p = next;
p = unwind_until(state_id, p, current_recursion_id);
if(p)
{
count = p->count;
start_pos = p->start_pos;
}
else
count = 0;
}
}
~repeater_count()
{
if(next)
*stack = next;
}
std::size_t get_count() { return count; }
int get_id() { return state_id; }
std::size_t operator++() { return ++count; }
bool check_null_repeat(const BidiIterator& pos, std::size_t max)
{
bool result = (count == 0) ? false : (pos == start_pos);
if(result)
count = max;
else
start_pos = pos;
return result;
}
};

struct saved_state;

enum saved_state_type
{
saved_type_end = 0,
saved_type_paren = 1,
saved_type_recurse = 2,
saved_type_assertion = 3,
saved_state_alt = 4,
saved_state_repeater_count = 5,
saved_state_extra_block = 6,
saved_state_greedy_single_repeat = 7,
saved_state_rep_slow_dot = 8,
saved_state_rep_fast_dot = 9,
saved_state_rep_char = 10,
saved_state_rep_short_set = 11,
saved_state_rep_long_set = 12,
saved_state_non_greedy_long_repeat = 13, 
saved_state_count = 14
};

#ifdef BOOST_MSVC
#  pragma warning(push)
#if BOOST_MSVC >= 1800
#pragma warning(disable:26495)
#endif
#endif
template <class Results>
struct recursion_info
{
typedef typename Results::value_type value_type;
typedef typename value_type::iterator iterator;
int idx;
const re_syntax_base* preturn_address;
Results results;
repeater_count<iterator>* repeater_stack;
iterator location_of_start;
};
#ifdef BOOST_MSVC
#  pragma warning(pop)
#endif

template <class BidiIterator, class Allocator, class traits>
class perl_matcher
{
public:
typedef typename traits::char_type char_type;
typedef perl_matcher<BidiIterator, Allocator, traits> self_type;
typedef bool (self_type::*matcher_proc_type)(void);
typedef std::size_t traits_size_type;
typedef typename is_byte<char_type>::width_type width_type;
typedef typename regex_iterator_traits<BidiIterator>::difference_type difference_type;
typedef match_results<BidiIterator, Allocator> results_type;

perl_matcher(BidiIterator first, BidiIterator end, 
match_results<BidiIterator, Allocator>& what, 
const basic_regex<char_type, traits>& e,
match_flag_type f,
BidiIterator l_base)
:  m_result(what), base(first), last(end), 
position(first), backstop(l_base), re(e), traits_inst(e.get_traits()), 
m_independent(false), next_count(&rep_obj), rep_obj(&next_count)
#ifdef BOOST_REGEX_NON_RECURSIVE
, m_recursions(0)
#endif
{
construct_init(e, f);
}

bool match();
bool find();

void setf(match_flag_type f)
{ m_match_flags |= f; }
void unsetf(match_flag_type f)
{ m_match_flags &= ~f; }

private:
void construct_init(const basic_regex<char_type, traits>& e, match_flag_type f);

bool find_imp();
bool match_imp();
#ifdef BOOST_REGEX_HAS_MS_STACK_GUARD
typedef bool (perl_matcher::*protected_proc_type)();
bool protected_call(protected_proc_type);
#endif
void estimate_max_state_count(std::random_access_iterator_tag*);
void estimate_max_state_count(void*);
bool match_prefix();
bool match_all_states();

bool match_startmark();
bool match_endmark();
bool match_literal();
bool match_start_line();
bool match_end_line();
bool match_wild();
bool match_match();
bool match_word_boundary();
bool match_within_word();
bool match_word_start();
bool match_word_end();
bool match_buffer_start();
bool match_buffer_end();
bool match_backref();
bool match_long_set();
bool match_set();
bool match_jump();
bool match_alt();
bool match_rep();
bool match_combining();
bool match_soft_buffer_end();
bool match_restart_continue();
bool match_long_set_repeat();
bool match_set_repeat();
bool match_char_repeat();
bool match_dot_repeat_fast();
bool match_dot_repeat_slow();
bool match_dot_repeat_dispatch()
{
return ::boost::is_random_access_iterator<BidiIterator>::value ? match_dot_repeat_fast() : match_dot_repeat_slow();
}
bool match_backstep();
bool match_assert_backref();
bool match_toggle_case();
#ifdef BOOST_REGEX_RECURSIVE
bool backtrack_till_match(std::size_t count);
#endif
bool match_recursion();
bool match_fail();
bool match_accept();
bool match_commit();
bool match_then();
bool skip_until_paren(int index, bool match = true);

bool find_restart_any();
bool find_restart_word();
bool find_restart_line();
bool find_restart_buf();
bool find_restart_lit();

private:
match_results<BidiIterator, Allocator>& m_result;
scoped_ptr<match_results<BidiIterator, Allocator> > m_temp_match;
match_results<BidiIterator, Allocator>* m_presult;
BidiIterator base;
BidiIterator last; 
BidiIterator position;
BidiIterator restart;
BidiIterator search_base;
BidiIterator backstop;
const basic_regex<char_type, traits>& re;
const ::boost::regex_traits_wrapper<traits>& traits_inst;
const re_syntax_base* pstate;
match_flag_type m_match_flags;
std::ptrdiff_t state_count;
std::ptrdiff_t max_state_count;
bool icase;
bool m_has_partial_match;
bool m_has_found_match;
bool m_independent;
repeater_count<BidiIterator>* next_count;
repeater_count<BidiIterator> rep_obj;
typename traits::char_class_type m_word_mask;
unsigned char match_any_mask;
std::vector<recursion_info<results_type> > recursion_stack;
#ifdef BOOST_REGEX_RECURSIVE
bool m_can_backtrack;
bool m_have_accept;
bool m_have_then;
#endif
#ifdef BOOST_REGEX_NON_RECURSIVE
typedef bool (self_type::*unwind_proc_type)(bool);

void extend_stack();
bool unwind(bool);
bool unwind_end(bool);
bool unwind_paren(bool);
bool unwind_recursion_stopper(bool);
bool unwind_assertion(bool);
bool unwind_alt(bool);
bool unwind_repeater_counter(bool);
bool unwind_extra_block(bool);
bool unwind_greedy_single_repeat(bool);
bool unwind_slow_dot_repeat(bool);
bool unwind_fast_dot_repeat(bool);
bool unwind_char_repeat(bool);
bool unwind_short_set_repeat(bool);
bool unwind_long_set_repeat(bool);
bool unwind_non_greedy_repeat(bool);
bool unwind_recursion(bool);
bool unwind_recursion_pop(bool);
bool unwind_commit(bool);
bool unwind_then(bool);
bool unwind_case(bool);
void destroy_single_repeat();
void push_matched_paren(int index, const sub_match<BidiIterator>& sub);
void push_recursion_stopper();
void push_assertion(const re_syntax_base* ps, bool positive);
void push_alt(const re_syntax_base* ps);
void push_repeater_count(int i, repeater_count<BidiIterator>** s);
void push_single_repeat(std::size_t c, const re_repeat* r, BidiIterator last_position, int state_id);
void push_non_greedy_repeat(const re_syntax_base* ps);
void push_recursion(int idx, const re_syntax_base* p, results_type* presults, results_type* presults2);
void push_recursion_pop();
void push_case_change(bool);

saved_state* m_stack_base;
saved_state* m_backup_state;
unsigned used_block_count;
bool m_recursive_result;
bool m_unwound_lookahead;
bool m_unwound_alt;
unsigned m_recursions;
#endif

#ifdef BOOST_MSVC
#  pragma warning(push)
#if BOOST_MSVC >= 1800
#pragma warning(disable:26495)
#endif
#endif
perl_matcher& operator=(const perl_matcher&)
{
return *this;
}
perl_matcher(const perl_matcher& that)
: m_result(that.m_result), re(that.re), traits_inst(that.traits_inst), rep_obj(0) {}
#ifdef BOOST_MSVC
#  pragma warning(pop)
#endif
};

} 

#ifdef BOOST_MSVC
#  pragma warning(pop)
#endif

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

#ifdef BOOST_REGEX_RECURSIVE
#include <boost/regex/v4/perl_matcher_recursive.hpp>
#else
#include <boost/regex/v4/perl_matcher_non_recursive.hpp>
#endif
#include <boost/regex/v4/perl_matcher_common.hpp>

#endif


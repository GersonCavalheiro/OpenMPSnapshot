



#ifndef BOOST_REGEX_V4_PERL_MATCHER_COMMON_HPP
#define BOOST_REGEX_V4_PERL_MATCHER_COMMON_HPP

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable: 4103)
#if BOOST_MSVC >= 1800
#pragma warning(disable: 26812)
#endif
#endif
#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_PREFIX
#endif
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#ifdef BOOST_BORLANDC
#  pragma option push -w-8008 -w-8066
#endif
#ifdef BOOST_MSVC
#  pragma warning(push)
#if BOOST_MSVC < 1910
#pragma warning(disable:4800)
#endif
#endif

namespace boost{
namespace BOOST_REGEX_DETAIL_NS{

#ifdef BOOST_MSVC
#  pragma warning(push)
#pragma warning(disable:26812)
#endif
template <class BidiIterator, class Allocator, class traits>
void perl_matcher<BidiIterator, Allocator, traits>::construct_init(const basic_regex<char_type, traits>& e, match_flag_type f)
{ 
typedef typename regex_iterator_traits<BidiIterator>::iterator_category category;
typedef typename basic_regex<char_type, traits>::flag_type expression_flag_type;

if(e.empty())
{
std::invalid_argument ex("Invalid regular expression object");
boost::throw_exception(ex);
}
pstate = 0;
m_match_flags = f;
estimate_max_state_count(static_cast<category*>(0));
expression_flag_type re_f = re.flags();
icase = re_f & regex_constants::icase;
if(!(m_match_flags & (match_perl|match_posix)))
{
if((re_f & (regbase::main_option_type|regbase::no_perl_ex)) == 0)
m_match_flags |= match_perl;
else if((re_f & (regbase::main_option_type|regbase::emacs_ex)) == (regbase::basic_syntax_group|regbase::emacs_ex))
m_match_flags |= match_perl;
else if((re_f & (regbase::main_option_type|regbase::literal)) == (regbase::literal))
m_match_flags |= match_perl;
else
m_match_flags |= match_posix;
}
if(m_match_flags & match_posix)
{
m_temp_match.reset(new match_results<BidiIterator, Allocator>());
m_presult = m_temp_match.get();
}
else
m_presult = &m_result;
#ifdef BOOST_REGEX_NON_RECURSIVE
m_stack_base = 0;
m_backup_state = 0;
#elif defined(BOOST_REGEX_RECURSIVE)
m_can_backtrack = true;
m_have_accept = false;
#endif
m_word_mask = re.get_data().m_word_mask; 
match_any_mask = static_cast<unsigned char>((f & match_not_dot_newline) ? BOOST_REGEX_DETAIL_NS::test_not_newline : BOOST_REGEX_DETAIL_NS::test_newline);
if(e.get_data().m_disable_match_any)
m_match_flags &= regex_constants::match_not_any;
}
#ifdef BOOST_MSVC
#  pragma warning(pop)
#endif

template <class BidiIterator, class Allocator, class traits>
void perl_matcher<BidiIterator, Allocator, traits>::estimate_max_state_count(std::random_access_iterator_tag*)
{
static const std::ptrdiff_t k = 100000;
std::ptrdiff_t dist = boost::BOOST_REGEX_DETAIL_NS::distance(base, last);
if(dist == 0)
dist = 1;
std::ptrdiff_t states = re.size();
if(states == 0)
states = 1;
if ((std::numeric_limits<std::ptrdiff_t>::max)() / states < states)
{
max_state_count = (std::min)((std::ptrdiff_t)BOOST_REGEX_MAX_STATE_COUNT, (std::numeric_limits<std::ptrdiff_t>::max)() - 2);
return;
}
states *= states;
if((std::numeric_limits<std::ptrdiff_t>::max)() / dist < states)
{
max_state_count = (std::min)((std::ptrdiff_t)BOOST_REGEX_MAX_STATE_COUNT, (std::numeric_limits<std::ptrdiff_t>::max)() - 2);
return;
}
states *= dist;
if((std::numeric_limits<std::ptrdiff_t>::max)() - k < states)
{
max_state_count = (std::min)((std::ptrdiff_t)BOOST_REGEX_MAX_STATE_COUNT, (std::numeric_limits<std::ptrdiff_t>::max)() - 2);
return;
}
states += k;

max_state_count = states;

states = dist;
if((std::numeric_limits<std::ptrdiff_t>::max)() / dist < states)
{
max_state_count = (std::min)((std::ptrdiff_t)BOOST_REGEX_MAX_STATE_COUNT, (std::numeric_limits<std::ptrdiff_t>::max)() - 2);
return;
}
states *= dist;
if((std::numeric_limits<std::ptrdiff_t>::max)() - k < states)
{
max_state_count = (std::min)((std::ptrdiff_t)BOOST_REGEX_MAX_STATE_COUNT, (std::numeric_limits<std::ptrdiff_t>::max)() - 2);
return;
}
states += k;
if(states > BOOST_REGEX_MAX_STATE_COUNT)
states = BOOST_REGEX_MAX_STATE_COUNT;
if(states > max_state_count)
max_state_count = states;
}

template <class BidiIterator, class Allocator, class traits>
inline void perl_matcher<BidiIterator, Allocator, traits>::estimate_max_state_count(void*)
{
max_state_count = BOOST_REGEX_MAX_STATE_COUNT;
}

#ifdef BOOST_REGEX_HAS_MS_STACK_GUARD
template <class BidiIterator, class Allocator, class traits>
inline bool perl_matcher<BidiIterator, Allocator, traits>::protected_call(
protected_proc_type proc)
{
::boost::BOOST_REGEX_DETAIL_NS::concrete_protected_call
<perl_matcher<BidiIterator, Allocator, traits> >
obj(this, proc);
return obj.execute();

}
#endif

template <class BidiIterator, class Allocator, class traits>
inline bool perl_matcher<BidiIterator, Allocator, traits>::match()
{
#ifdef BOOST_REGEX_HAS_MS_STACK_GUARD
return protected_call(&perl_matcher<BidiIterator, Allocator, traits>::match_imp);
#else
return match_imp();
#endif
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::match_imp()
{
#ifdef BOOST_REGEX_NON_RECURSIVE
save_state_init init(&m_stack_base, &m_backup_state);
used_block_count = BOOST_REGEX_MAX_BLOCKS;
#if !defined(BOOST_NO_EXCEPTIONS)
try{
#endif
#endif

position = base;
search_base = base;
state_count = 0;
m_match_flags |= regex_constants::match_all;
m_presult->set_size((m_match_flags & match_nosubs) ? 1u : static_cast<typename results_type::size_type>(1u + re.mark_count()), search_base, last);
m_presult->set_base(base);
m_presult->set_named_subs(this->re.get_named_subs());
if(m_match_flags & match_posix)
m_result = *m_presult;
verify_options(re.flags(), m_match_flags);
if(0 == match_prefix())
return false;
return (m_result[0].second == last) && (m_result[0].first == base);

#if defined(BOOST_REGEX_NON_RECURSIVE) && !defined(BOOST_NO_EXCEPTIONS)
}
catch(...)
{
while(unwind(true)){}
throw;
}
#endif
}

template <class BidiIterator, class Allocator, class traits>
inline bool perl_matcher<BidiIterator, Allocator, traits>::find()
{
#ifdef BOOST_REGEX_HAS_MS_STACK_GUARD
return protected_call(&perl_matcher<BidiIterator, Allocator, traits>::find_imp);
#else
return find_imp();
#endif
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::find_imp()
{
static matcher_proc_type const s_find_vtable[7] = 
{
&perl_matcher<BidiIterator, Allocator, traits>::find_restart_any,
&perl_matcher<BidiIterator, Allocator, traits>::find_restart_word,
&perl_matcher<BidiIterator, Allocator, traits>::find_restart_line,
&perl_matcher<BidiIterator, Allocator, traits>::find_restart_buf,
&perl_matcher<BidiIterator, Allocator, traits>::match_prefix,
&perl_matcher<BidiIterator, Allocator, traits>::find_restart_lit,
&perl_matcher<BidiIterator, Allocator, traits>::find_restart_lit,
};

#ifdef BOOST_REGEX_NON_RECURSIVE
save_state_init init(&m_stack_base, &m_backup_state);
used_block_count = BOOST_REGEX_MAX_BLOCKS;
#if !defined(BOOST_NO_EXCEPTIONS)
try{
#endif
#endif

state_count = 0;
if((m_match_flags & regex_constants::match_init) == 0)
{
search_base = position = base;
pstate = re.get_first_state();
m_presult->set_size((m_match_flags & match_nosubs) ? 1u : static_cast<typename results_type::size_type>(1u + re.mark_count()), base, last);
m_presult->set_base(base);
m_presult->set_named_subs(this->re.get_named_subs());
m_match_flags |= regex_constants::match_init;
}
else
{
search_base = position = m_result[0].second;
if(((m_match_flags & match_not_null) == 0) && (m_result.length() == 0))
{
if(position == last)
return false;
else 
++position;
}
m_presult->set_size((m_match_flags & match_nosubs) ? 1u : static_cast<typename results_type::size_type>(1u + re.mark_count()), search_base, last);
}
if(m_match_flags & match_posix)
{
m_result.set_size(static_cast<typename results_type::size_type>(1u + re.mark_count()), base, last);
m_result.set_base(base);
}

verify_options(re.flags(), m_match_flags);
unsigned type = (m_match_flags & match_continuous) ? 
static_cast<unsigned int>(regbase::restart_continue) 
: static_cast<unsigned int>(re.get_restart_type());

matcher_proc_type proc = s_find_vtable[type];
return (this->*proc)();

#if defined(BOOST_REGEX_NON_RECURSIVE) && !defined(BOOST_NO_EXCEPTIONS)
}
catch(...)
{
while(unwind(true)){}
throw;
}
#endif
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::match_prefix()
{
m_has_partial_match = false;
m_has_found_match = false;
pstate = re.get_first_state();
m_presult->set_first(position);
restart = position;
match_all_states();
if(!m_has_found_match && m_has_partial_match && (m_match_flags & match_partial))
{
m_has_found_match = true;
m_presult->set_second(last, 0, false);
position = last;
if((m_match_flags & match_posix) == match_posix)
{
m_result.maybe_assign(*m_presult);
}
}
#ifdef BOOST_REGEX_MATCH_EXTRA
if(m_has_found_match && (match_extra & m_match_flags))
{
for(unsigned i = 0; i < m_presult->size(); ++i)
{
typename sub_match<BidiIterator>::capture_sequence_type & seq = ((*m_presult)[i]).get_captures();
std::reverse(seq.begin(), seq.end());
}
}
#endif
if(!m_has_found_match)
position = restart; 
#ifdef BOOST_REGEX_RECURSIVE
m_can_backtrack = true; 
#endif
return m_has_found_match;
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::match_literal()
{
unsigned int len = static_cast<const re_literal*>(pstate)->length;
const char_type* what = reinterpret_cast<const char_type*>(static_cast<const re_literal*>(pstate) + 1);
for(unsigned int i = 0; i < len; ++i, ++position)
{
if((position == last) || (traits_inst.translate(*position, icase) != what[i]))
return false;
}
pstate = pstate->next.p;
return true;
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::match_start_line()
{
if(position == backstop)
{
if((m_match_flags & match_prev_avail) == 0)
{
if((m_match_flags & match_not_bol) == 0)
{
pstate = pstate->next.p;
return true;
}
return false;
}
}
else if(m_match_flags & match_single_line)
return false;

BidiIterator t(position);
--t;
if(position != last)
{
if(is_separator(*t) && !((*t == static_cast<char_type>('\r')) && (*position == static_cast<char_type>('\n'))) )
{
pstate = pstate->next.p;
return true;
}
}
else if(is_separator(*t))
{
pstate = pstate->next.p;
return true;
}
return false;
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::match_end_line()
{
if(position != last)
{
if(m_match_flags & match_single_line)
return false;
if(is_separator(*position))
{
if((position != backstop) || (m_match_flags & match_prev_avail))
{
BidiIterator t(position);
--t;
if((*t == static_cast<char_type>('\r')) && (*position == static_cast<char_type>('\n')))
{
return false;
}
}
pstate = pstate->next.p;
return true;
}
}
else if((m_match_flags & match_not_eol) == 0)
{
pstate = pstate->next.p;
return true;
}
return false;
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::match_wild()
{
if(position == last) 
return false;
if(is_separator(*position) && ((match_any_mask & static_cast<const re_dot*>(pstate)->mask) == 0))
return false;
if((*position == char_type(0)) && (m_match_flags & match_not_dot_null))
return false;
pstate = pstate->next.p;
++position;
return true;
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::match_word_boundary()
{
bool b; 
if(position != last)
{
b = traits_inst.isctype(*position, m_word_mask);
}
else
{
if (m_match_flags & match_not_eow)
return false;
b = false;
}
if((position == backstop) && ((m_match_flags & match_prev_avail) == 0))
{
if(m_match_flags & match_not_bow)
return false;
else
b ^= false;
}
else
{
--position;
b ^= traits_inst.isctype(*position, m_word_mask);
++position;
}
if(b)
{
pstate = pstate->next.p;
return true;
}
return false; 
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::match_within_word()
{
if(position == last)
return false;
bool prev = traits_inst.isctype(*position, m_word_mask);
{
bool b;
if((position == backstop) && ((m_match_flags & match_prev_avail) == 0)) 
return false;
else
{
--position;
b = traits_inst.isctype(*position, m_word_mask);
++position;
}
if(b == prev)
{
pstate = pstate->next.p;
return true;
}
}
return false;
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::match_word_start()
{
if(position == last)
return false; 
if(!traits_inst.isctype(*position, m_word_mask))
return false; 
if((position == backstop) && ((m_match_flags & match_prev_avail) == 0))
{
if(m_match_flags & match_not_bow)
return false; 
}
else
{
BidiIterator t(position);
--t;
if(traits_inst.isctype(*t, m_word_mask))
return false; 
}
pstate = pstate->next.p;
return true;
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::match_word_end()
{
if((position == backstop) && ((m_match_flags & match_prev_avail) == 0))
return false;  
BidiIterator t(position);
--t;
if(traits_inst.isctype(*t, m_word_mask) == false)
return false;  

if(position == last)
{
if(m_match_flags & match_not_eow)
return false; 
}
else
{
if(traits_inst.isctype(*position, m_word_mask))
return false; 
}
pstate = pstate->next.p;
return true;      
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::match_buffer_start()
{
if((position != backstop) || (m_match_flags & match_not_bob))
return false;
pstate = pstate->next.p;
return true;
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::match_buffer_end()
{
if((position != last) || (m_match_flags & match_not_eob))
return false;
pstate = pstate->next.p;
return true;
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::match_backref()
{
int index = static_cast<const re_brace*>(pstate)->index;
if(index >= hash_value_mask)
{
named_subexpressions::range_type r = re.get_data().equal_range(index);
BOOST_ASSERT(r.first != r.second);
do
{
index = r.first->index;
++r.first;
}while((r.first != r.second) && ((*m_presult)[index].matched != true));
}

if((m_match_flags & match_perl) && !(*m_presult)[index].matched)
return false;

BidiIterator i = (*m_presult)[index].first;
BidiIterator j = (*m_presult)[index].second;
while(i != j)
{
if((position == last) || (traits_inst.translate(*position, icase) != traits_inst.translate(*i, icase)))
return false;
++i;
++position;
}
pstate = pstate->next.p;
return true;
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::match_long_set()
{
typedef typename traits::char_class_type char_class_type;
if(position == last)
return false;
BidiIterator t = re_is_set_member(position, last, static_cast<const re_set_long<char_class_type>*>(pstate), re.get_data(), icase);
if(t != position)
{
pstate = pstate->next.p;
position = t;
return true;
}
return false;
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::match_set()
{
if(position == last)
return false;
if(static_cast<const re_set*>(pstate)->_map[static_cast<unsigned char>(traits_inst.translate(*position, icase))])
{
pstate = pstate->next.p;
++position;
return true;
}
return false;
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::match_jump()
{
pstate = static_cast<const re_jump*>(pstate)->alt.p;
return true;
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::match_combining()
{
if(position == last)
return false;
if(is_combining(traits_inst.translate(*position, icase)))
return false;
++position;
while((position != last) && is_combining(traits_inst.translate(*position, icase)))
++position;
pstate = pstate->next.p;
return true;
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::match_soft_buffer_end()
{
if(m_match_flags & match_not_eob)
return false;
BidiIterator p(position);
while((p != last) && is_separator(traits_inst.translate(*p, icase)))++p;
if(p != last)
return false;
pstate = pstate->next.p;
return true;
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::match_restart_continue()
{
if(position == search_base)
{
pstate = pstate->next.p;
return true;
}
return false;
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::match_backstep()
{
#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable:4127)
#endif
if( ::boost::is_random_access_iterator<BidiIterator>::value)
{
std::ptrdiff_t maxlen = ::boost::BOOST_REGEX_DETAIL_NS::distance(backstop, position);
if(maxlen < static_cast<const re_brace*>(pstate)->index)
return false;
std::advance(position, -static_cast<const re_brace*>(pstate)->index);
}
else
{
int c = static_cast<const re_brace*>(pstate)->index;
while(c--)
{
if(position == backstop)
return false;
--position;
}
}
pstate = pstate->next.p;
return true;
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif
}

template <class BidiIterator, class Allocator, class traits>
inline bool perl_matcher<BidiIterator, Allocator, traits>::match_assert_backref()
{
int index = static_cast<const re_brace*>(pstate)->index;
bool result = false;
if(index == 9999)
{
return false;
}
else if(index > 0)
{
if(index >= hash_value_mask)
{
named_subexpressions::range_type r = re.get_data().equal_range(index);
while(r.first != r.second)
{
if((*m_presult)[r.first->index].matched)
{
result = true;
break;
}
++r.first;
}
}
else
{
result = (*m_presult)[index].matched;
}
pstate = pstate->next.p;
}
else
{
int idx = -(index+1);
if(idx >= hash_value_mask)
{
named_subexpressions::range_type r = re.get_data().equal_range(idx);
int stack_index = recursion_stack.empty() ? -1 : recursion_stack.back().idx;
while(r.first != r.second)
{
result |= (stack_index == r.first->index);
if(result)break;
++r.first;
}
}
else
{
result = !recursion_stack.empty() && ((recursion_stack.back().idx == idx) || (index == 0));
}
pstate = pstate->next.p;
}
return result;
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::match_fail()
{
return false;
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::match_accept()
{
if(!recursion_stack.empty())
{
return skip_until_paren(recursion_stack.back().idx);
}
else
{
return skip_until_paren(INT_MAX);
}
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::find_restart_any()
{
#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable:4127)
#endif
const unsigned char* _map = re.get_map();
while(true)
{
while((position != last) && !can_start(*position, _map, (unsigned char)mask_any) )
++position;
if(position == last)
{
if(re.can_be_null())
return match_prefix();
break;
}
if(match_prefix())
return true;
if(position == last)
return false;
++position;
}
return false;
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::find_restart_word()
{
#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable:4127)
#endif
const unsigned char* _map = re.get_map();
if((m_match_flags & match_prev_avail) || (position != base))
--position;
else if(match_prefix())
return true;
do
{
while((position != last) && traits_inst.isctype(*position, m_word_mask))
++position;
while((position != last) && !traits_inst.isctype(*position, m_word_mask))
++position;
if(position == last)
break;

if(can_start(*position, _map, (unsigned char)mask_any) )
{
if(match_prefix())
return true;
}
if(position == last)
break;
} while(true);
return false;
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::find_restart_line()
{
const unsigned char* _map = re.get_map();
if(match_prefix())
return true;
while(position != last)
{
while((position != last) && !is_separator(*position))
++position;
if(position == last)
return false;
++position;
if(position == last)
{
if(re.can_be_null() && match_prefix())
return true;
return false;
}

if( can_start(*position, _map, (unsigned char)mask_any) )
{
if(match_prefix())
return true;
}
if(position == last)
return false;
}
return false;
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::find_restart_buf()
{
if((position == base) && ((m_match_flags & match_not_bob) == 0))
return match_prefix();
return false;
}

template <class BidiIterator, class Allocator, class traits>
bool perl_matcher<BidiIterator, Allocator, traits>::find_restart_lit()
{
#if 0
if(position == last)
return false; 

unsigned type = (m_match_flags & match_continuous) ? 
static_cast<unsigned int>(regbase::restart_continue) 
: static_cast<unsigned int>(re.get_restart_type());

const kmp_info<char_type>* info = access::get_kmp(re);
int len = info->len;
const char_type* x = info->pstr;
int j = 0; 
while (position != last) 
{
while((j > -1) && (x[j] != traits_inst.translate(*position, icase))) 
j = info->kmp_next[j];
++position;
++j;
if(j >= len) 
{
if(type == regbase::restart_fixed_lit)
{
std::advance(position, -j);
restart = position;
std::advance(restart, len);
m_result.set_first(position);
m_result.set_second(restart);
position = restart;
return true;
}
else
{
restart = position;
std::advance(position, -j);
if(match_prefix())
return true;
else
{
for(int k = 0; (restart != position) && (k < j); ++k, --restart)
{} 
if(restart != last)
++restart;
position = restart;
j = 0;  
}
}
}
}
if((m_match_flags & match_partial) && (position == last) && j)
{
restart = position;
std::advance(position, -j);
return match_prefix();
}
#endif
return false;
}

} 

} 

#ifdef BOOST_MSVC
#  pragma warning(pop)
#endif

#ifdef BOOST_BORLANDC
#  pragma option pop
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

#endif


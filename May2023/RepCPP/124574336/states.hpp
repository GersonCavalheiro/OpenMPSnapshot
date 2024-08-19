



#ifndef BOOST_REGEX_V4_STATES_HPP
#define BOOST_REGEX_V4_STATES_HPP

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

namespace boost{
namespace BOOST_REGEX_DETAIL_NS{


enum mask_type
{
mask_take = 1,
mask_skip = 2,
mask_init = 4,
mask_any = mask_skip | mask_take,
mask_all = mask_any
};


struct _narrow_type{};
struct _wide_type{};
template <class charT> struct is_byte;
template<>             struct is_byte<char>         { typedef _narrow_type width_type; };
template<>             struct is_byte<unsigned char>{ typedef _narrow_type width_type; };
template<>             struct is_byte<signed char>  { typedef _narrow_type width_type; };
template <class charT> struct is_byte               { typedef _wide_type width_type; };


enum syntax_element_type
{
syntax_element_startmark = 0,
syntax_element_endmark = syntax_element_startmark + 1,
syntax_element_literal = syntax_element_endmark + 1,
syntax_element_start_line = syntax_element_literal + 1,
syntax_element_end_line = syntax_element_start_line + 1,
syntax_element_wild = syntax_element_end_line + 1,
syntax_element_match = syntax_element_wild + 1,
syntax_element_word_boundary = syntax_element_match + 1,
syntax_element_within_word = syntax_element_word_boundary + 1,
syntax_element_word_start = syntax_element_within_word + 1,
syntax_element_word_end = syntax_element_word_start + 1,
syntax_element_buffer_start = syntax_element_word_end + 1,
syntax_element_buffer_end = syntax_element_buffer_start + 1,
syntax_element_backref = syntax_element_buffer_end + 1,
syntax_element_long_set = syntax_element_backref + 1,
syntax_element_set = syntax_element_long_set + 1,
syntax_element_jump = syntax_element_set + 1,
syntax_element_alt = syntax_element_jump + 1,
syntax_element_rep = syntax_element_alt + 1,
syntax_element_combining = syntax_element_rep + 1,
syntax_element_soft_buffer_end = syntax_element_combining + 1,
syntax_element_restart_continue = syntax_element_soft_buffer_end + 1,
syntax_element_dot_rep = syntax_element_restart_continue + 1,
syntax_element_char_rep = syntax_element_dot_rep + 1,
syntax_element_short_set_rep = syntax_element_char_rep + 1,
syntax_element_long_set_rep = syntax_element_short_set_rep + 1,
syntax_element_backstep = syntax_element_long_set_rep + 1,
syntax_element_assert_backref = syntax_element_backstep + 1,
syntax_element_toggle_case = syntax_element_assert_backref + 1,
syntax_element_recurse = syntax_element_toggle_case + 1,
syntax_element_fail = syntax_element_recurse + 1,
syntax_element_accept = syntax_element_fail + 1,
syntax_element_commit = syntax_element_accept + 1,
syntax_element_then = syntax_element_commit + 1
};

#ifdef BOOST_REGEX_DEBUG
std::ostream& operator<<(std::ostream&, syntax_element_type);
#endif

struct re_syntax_base;


union offset_type
{
re_syntax_base*   p;
std::ptrdiff_t    i;
};


struct re_syntax_base
{
syntax_element_type   type;         
offset_type           next;         
};


struct re_brace : public re_syntax_base
{
int index;
bool icase;
};


enum
{
dont_care = 1,
force_not_newline = 0,
force_newline = 2,

test_not_newline = 2,
test_newline = 3
};
struct re_dot : public re_syntax_base
{
unsigned char mask;
};


struct re_literal : public re_syntax_base
{
unsigned int length;
};


struct re_case : public re_syntax_base
{
bool icase;
};


template <class mask_type>
struct re_set_long : public re_syntax_base
{
unsigned int            csingles, cranges, cequivalents;
mask_type               cclasses;
mask_type               cnclasses;
bool                    isnot;
bool                    singleton;
};


struct re_set : public re_syntax_base
{
unsigned char _map[1 << CHAR_BIT];
};


struct re_jump : public re_syntax_base
{
offset_type     alt;                 
};


struct re_alt : public re_jump
{
unsigned char   _map[1 << CHAR_BIT]; 
unsigned int    can_be_null;         
};


struct re_repeat : public re_alt
{
std::size_t   min, max;  
int           state_id;        
bool          leading;   
bool          greedy;    
};


struct re_recurse : public re_jump
{
int state_id;             
};


enum commit_type
{
commit_prune,
commit_skip,
commit_commit
};
struct re_commit : public re_syntax_base
{
commit_type action;
};


enum re_jump_size_type
{
re_jump_size = (sizeof(re_jump) + padding_mask) & ~(padding_mask),
re_repeater_size = (sizeof(re_repeat) + padding_mask) & ~(padding_mask),
re_alt_size = (sizeof(re_alt) + padding_mask) & ~(padding_mask)
};



template<class charT, class traits>
struct regex_data;

template <class iterator, class charT, class traits_type, class char_classT>
iterator BOOST_REGEX_CALL re_is_set_member(iterator next, 
iterator last, 
const re_set_long<char_classT>* set_, 
const regex_data<charT, traits_type>& e, bool icase);

} 

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

#endif



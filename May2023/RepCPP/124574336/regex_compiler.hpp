
#ifndef BOOST_XPRESSIVE_REGEX_COMPILER_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_REGEX_COMPILER_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <map>
#include <boost/config.hpp>
#include <boost/assert.hpp>
#include <boost/next_prior.hpp>
#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/throw_exception.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/iterator/iterator_traits.hpp>
#include <boost/xpressive/basic_regex.hpp>
#include <boost/xpressive/detail/dynamic/parser.hpp>
#include <boost/xpressive/detail/dynamic/parse_charset.hpp>
#include <boost/xpressive/detail/dynamic/parser_enum.hpp>
#include <boost/xpressive/detail/dynamic/parser_traits.hpp>
#include <boost/xpressive/detail/core/linker.hpp>
#include <boost/xpressive/detail/core/optimize.hpp>

namespace boost { namespace xpressive
{

template<typename BidiIter, typename RegexTraits, typename CompilerTraits>
struct regex_compiler
{
typedef BidiIter iterator_type;
typedef typename iterator_value<BidiIter>::type char_type;
typedef regex_constants::syntax_option_type flag_type;
typedef RegexTraits traits_type;
typedef typename traits_type::string_type string_type;
typedef typename traits_type::locale_type locale_type;
typedef typename traits_type::char_class_type char_class_type;

explicit regex_compiler(RegexTraits const &traits = RegexTraits())
: mark_count_(0)
, hidden_mark_count_(0)
, traits_(traits)
, upper_(0)
, self_()
, rules_()
{
this->upper_ = lookup_classname(this->rxtraits(), "upper");
}

locale_type imbue(locale_type loc)
{
locale_type oldloc = this->traits_.imbue(loc);
this->upper_ = lookup_classname(this->rxtraits(), "upper");
return oldloc;
}

locale_type getloc() const
{
return this->traits_.getloc();
}

template<typename InputIter>
basic_regex<BidiIter>
compile(InputIter begin, InputIter end, flag_type flags = regex_constants::ECMAScript)
{
typedef typename iterator_category<InputIter>::type category;
return this->compile_(begin, end, flags, category());
}

template<typename InputRange>
typename disable_if<is_pointer<InputRange>, basic_regex<BidiIter> >::type
compile(InputRange const &pat, flag_type flags = regex_constants::ECMAScript)
{
return this->compile(boost::begin(pat), boost::end(pat), flags);
}

basic_regex<BidiIter>
compile(char_type const *begin, flag_type flags = regex_constants::ECMAScript)
{
BOOST_ASSERT(0 != begin);
char_type const *end = begin + std::char_traits<char_type>::length(begin);
return this->compile(begin, end, flags);
}

basic_regex<BidiIter> compile(char_type const *begin, std::size_t size, flag_type flags)
{
BOOST_ASSERT(0 != begin);
char_type const *end = begin + size;
return this->compile(begin, end, flags);
}

basic_regex<BidiIter> &operator [](string_type const &name)
{
BOOST_ASSERT(!name.empty());
return this->rules_[name];
}

basic_regex<BidiIter> const &operator [](string_type const &name) const
{
BOOST_ASSERT(!name.empty());
return this->rules_[name];
}

private:

typedef detail::escape_value<char_type, char_class_type> escape_value;
typedef detail::alternate_matcher<detail::alternates_vector<BidiIter>, RegexTraits> alternate_matcher;

template<typename FwdIter>
basic_regex<BidiIter> compile_(FwdIter begin, FwdIter end, flag_type flags, std::forward_iterator_tag)
{
BOOST_MPL_ASSERT((is_same<char_type, typename iterator_value<FwdIter>::type>));
using namespace regex_constants;
this->reset();
this->traits_.flags(flags);

basic_regex<BidiIter> rextmp, *prex = &rextmp;
FwdIter tmp = begin;

string_type name;
if(token_group_begin == this->traits_.get_token(tmp, end) &&
BOOST_XPR_ENSURE_(tmp != end, error_paren, "mismatched parenthesis") &&
token_rule_assign == this->traits_.get_group_type(tmp, end, name))
{
begin = tmp;
BOOST_XPR_ENSURE_
(
begin != end && token_group_end == this->traits_.get_token(begin, end)
, error_paren
, "mismatched parenthesis"
);
prex = &this->rules_[name];
}

this->self_ = detail::core_access<BidiIter>::get_regex_impl(*prex);

detail::sequence<BidiIter> seq = this->parse_alternates(begin, end);
BOOST_XPR_ENSURE_(begin == end, error_paren, "mismatched parenthesis");

seq += detail::make_dynamic<BidiIter>(detail::end_matcher());

detail::common_compile(seq.xpr().matchable(), *this->self_, this->rxtraits());

this->self_->traits_ = new detail::traits_holder<RegexTraits>(this->rxtraits());
this->self_->mark_count_ = this->mark_count_;
this->self_->hidden_mark_count_ = this->hidden_mark_count_;

this->self_->tracking_update();
this->self_.reset();
return *prex;
}

template<typename InputIter>
basic_regex<BidiIter> compile_(InputIter begin, InputIter end, flag_type flags, std::input_iterator_tag)
{
string_type pat(begin, end);
return this->compile_(boost::begin(pat), boost::end(pat), flags, std::forward_iterator_tag());
}

void reset()
{
this->mark_count_ = 0;
this->hidden_mark_count_ = 0;
this->traits_.flags(regex_constants::ECMAScript);
}

traits_type &rxtraits()
{
return this->traits_.traits();
}

traits_type const &rxtraits() const
{
return this->traits_.traits();
}

template<typename FwdIter>
detail::sequence<BidiIter> parse_alternates(FwdIter &begin, FwdIter end)
{
using namespace regex_constants;
int count = 0;
FwdIter tmp = begin;
detail::sequence<BidiIter> seq;

do switch(++count)
{
case 1:
seq = this->parse_sequence(tmp, end);
break;
case 2:
seq = detail::make_dynamic<BidiIter>(alternate_matcher()) | seq;
BOOST_FALLTHROUGH;
default:
seq |= this->parse_sequence(tmp, end);
}
while((begin = tmp) != end && token_alternate == this->traits_.get_token(tmp, end));

return seq;
}

template<typename FwdIter>
detail::sequence<BidiIter> parse_group(FwdIter &begin, FwdIter end)
{
using namespace regex_constants;
int mark_nbr = 0;
bool keeper = false;
bool lookahead = false;
bool lookbehind = false;
bool negative = false;
string_type name;

detail::sequence<BidiIter> seq, seq_end;
FwdIter tmp = FwdIter();

syntax_option_type old_flags = this->traits_.flags();

switch(this->traits_.get_group_type(begin, end, name))
{
case token_no_mark:
if(token_group_end == this->traits_.get_token(tmp = begin, end))
{
return this->parse_atom(begin = tmp, end);
}
break;

case token_negative_lookahead:
negative = true;
BOOST_FALLTHROUGH;
case token_positive_lookahead:
lookahead = true;
break;

case token_negative_lookbehind:
negative = true;
BOOST_FALLTHROUGH;
case token_positive_lookbehind:
lookbehind = true;
break;

case token_independent_sub_expression:
keeper = true;
break;

case token_comment:
while(BOOST_XPR_ENSURE_(begin != end, error_paren, "mismatched parenthesis"))
{
switch(this->traits_.get_token(begin, end))
{
case token_group_end:
return this->parse_atom(begin, end);
case token_escape:
BOOST_XPR_ENSURE_(begin != end, error_escape, "incomplete escape sequence");
BOOST_FALLTHROUGH;
case token_literal:
++begin;
break;
default:
break;
}
}
break;

case token_recurse:
BOOST_XPR_ENSURE_
(
begin != end && token_group_end == this->traits_.get_token(begin, end)
, error_paren
, "mismatched parenthesis"
);
return detail::make_dynamic<BidiIter>(detail::regex_byref_matcher<BidiIter>(this->self_));

case token_rule_assign:
BOOST_THROW_EXCEPTION(
regex_error(error_badrule, "rule assignments must be at the front of the regex")
);
break;

case token_rule_ref:
{
typedef detail::core_access<BidiIter> access;
BOOST_XPR_ENSURE_
(
begin != end && token_group_end == this->traits_.get_token(begin, end)
, error_paren
, "mismatched parenthesis"
);
basic_regex<BidiIter> &rex = this->rules_[name];
shared_ptr<detail::regex_impl<BidiIter> > impl = access::get_regex_impl(rex);
this->self_->track_reference(*impl);
return detail::make_dynamic<BidiIter>(detail::regex_byref_matcher<BidiIter>(impl));
}

case token_named_mark:
mark_nbr = static_cast<int>(++this->mark_count_);
for(std::size_t i = 0; i < this->self_->named_marks_.size(); ++i)
{
BOOST_XPR_ENSURE_(this->self_->named_marks_[i].name_ != name, error_badmark, "named mark already exists");
}
this->self_->named_marks_.push_back(detail::named_mark<char_type>(name, this->mark_count_));
seq = detail::make_dynamic<BidiIter>(detail::mark_begin_matcher(mark_nbr));
seq_end = detail::make_dynamic<BidiIter>(detail::mark_end_matcher(mark_nbr));
break;

case token_named_mark_ref:
BOOST_XPR_ENSURE_
(
begin != end && token_group_end == this->traits_.get_token(begin, end)
, error_paren
, "mismatched parenthesis"
);
for(std::size_t i = 0; i < this->self_->named_marks_.size(); ++i)
{
if(this->self_->named_marks_[i].name_ == name)
{
mark_nbr = static_cast<int>(this->self_->named_marks_[i].mark_nbr_);
return detail::make_backref_xpression<BidiIter>
(
mark_nbr, this->traits_.flags(), this->rxtraits()
);
}
}
BOOST_THROW_EXCEPTION(regex_error(error_badmark, "invalid named back-reference"));
break;

default:
mark_nbr = static_cast<int>(++this->mark_count_);
seq = detail::make_dynamic<BidiIter>(detail::mark_begin_matcher(mark_nbr));
seq_end = detail::make_dynamic<BidiIter>(detail::mark_end_matcher(mark_nbr));
break;
}

seq += this->parse_alternates(begin, end);
seq += seq_end;
BOOST_XPR_ENSURE_
(
begin != end && token_group_end == this->traits_.get_token(begin, end)
, error_paren
, "mismatched parenthesis"
);

typedef detail::shared_matchable<BidiIter> xpr_type;
if(lookahead)
{
seq += detail::make_independent_end_xpression<BidiIter>(seq.pure());
detail::lookahead_matcher<xpr_type> lam(seq.xpr(), negative, seq.pure());
seq = detail::make_dynamic<BidiIter>(lam);
}
else if(lookbehind)
{
seq += detail::make_independent_end_xpression<BidiIter>(seq.pure());
detail::lookbehind_matcher<xpr_type> lbm(seq.xpr(), seq.width().value(), negative, seq.pure());
seq = detail::make_dynamic<BidiIter>(lbm);
}
else if(keeper) 
{
seq += detail::make_independent_end_xpression<BidiIter>(seq.pure());
detail::keeper_matcher<xpr_type> km(seq.xpr(), seq.pure());
seq = detail::make_dynamic<BidiIter>(km);
}

this->traits_.flags(old_flags);
return seq;
}

template<typename FwdIter>
detail::sequence<BidiIter> parse_charset(FwdIter &begin, FwdIter end)
{
detail::compound_charset<traits_type> chset;

detail::parse_charset(begin, end, chset, this->traits_);

return detail::make_charset_xpression<BidiIter>
(
chset
, this->rxtraits()
, this->traits_.flags()
);
}

template<typename FwdIter>
detail::sequence<BidiIter> parse_atom(FwdIter &begin, FwdIter end)
{
using namespace regex_constants;
escape_value esc = { 0, 0, 0, detail::escape_char };
FwdIter old_begin = begin;

switch(this->traits_.get_token(begin, end))
{
case token_literal:
return detail::make_literal_xpression<BidiIter>
(
this->parse_literal(begin, end), this->traits_.flags(), this->rxtraits()
);

case token_any:
return detail::make_any_xpression<BidiIter>(this->traits_.flags(), this->rxtraits());

case token_assert_begin_sequence:
return detail::make_dynamic<BidiIter>(detail::assert_bos_matcher());

case token_assert_end_sequence:
return detail::make_dynamic<BidiIter>(detail::assert_eos_matcher());

case token_assert_begin_line:
return detail::make_assert_begin_line<BidiIter>(this->traits_.flags(), this->rxtraits());

case token_assert_end_line:
return detail::make_assert_end_line<BidiIter>(this->traits_.flags(), this->rxtraits());

case token_assert_word_boundary:
return detail::make_assert_word<BidiIter>(detail::word_boundary<mpl::true_>(), this->rxtraits());

case token_assert_not_word_boundary:
return detail::make_assert_word<BidiIter>(detail::word_boundary<mpl::false_>(), this->rxtraits());

case token_assert_word_begin:
return detail::make_assert_word<BidiIter>(detail::word_begin(), this->rxtraits());

case token_assert_word_end:
return detail::make_assert_word<BidiIter>(detail::word_end(), this->rxtraits());

case token_escape:
esc = this->parse_escape(begin, end);
switch(esc.type_)
{
case detail::escape_mark:
return detail::make_backref_xpression<BidiIter>
(
esc.mark_nbr_, this->traits_.flags(), this->rxtraits()
);
case detail::escape_char:
return detail::make_char_xpression<BidiIter>
(
esc.ch_, this->traits_.flags(), this->rxtraits()
);
case detail::escape_class:
return detail::make_posix_charset_xpression<BidiIter>
(
esc.class_
, this->is_upper_(*begin++)
, this->traits_.flags()
, this->rxtraits()
);
}

case token_group_begin:
return this->parse_group(begin, end);

case token_charset_begin:
return this->parse_charset(begin, end);

case token_invalid_quantifier:
BOOST_THROW_EXCEPTION(regex_error(error_badrepeat, "quantifier not expected"));
break;

case token_quote_meta_begin:
return detail::make_literal_xpression<BidiIter>
(
this->parse_quote_meta(begin, end), this->traits_.flags(), this->rxtraits()
);

case token_quote_meta_end:
BOOST_THROW_EXCEPTION(
regex_error(
error_escape
, "found quote-meta end without corresponding quote-meta begin"
)
);
break;

case token_end_of_pattern:
break;

default:
begin = old_begin;
break;
}

return detail::sequence<BidiIter>();
}

template<typename FwdIter>
detail::sequence<BidiIter> parse_quant(FwdIter &begin, FwdIter end)
{
BOOST_ASSERT(begin != end);
detail::quant_spec spec = { 0, 0, false, &this->hidden_mark_count_ };
detail::sequence<BidiIter> seq = this->parse_atom(begin, end);

if(!seq.empty() && begin != end && detail::quant_none != seq.quant())
{
if(this->traits_.get_quant_spec(begin, end, spec))
{
BOOST_ASSERT(spec.min_ <= spec.max_);

if(0 == spec.max_) 
{
seq = this->parse_quant(begin, end);
}
else
{
seq.repeat(spec);
}
}
}

return seq;
}

template<typename FwdIter>
detail::sequence<BidiIter> parse_sequence(FwdIter &begin, FwdIter end)
{
detail::sequence<BidiIter> seq;

while(begin != end)
{
detail::sequence<BidiIter> seq_quant = this->parse_quant(begin, end);

if(seq_quant.empty())
break;

seq += seq_quant;
}

return seq;
}

template<typename FwdIter>
string_type parse_literal(FwdIter &begin, FwdIter end)
{
using namespace regex_constants;
BOOST_ASSERT(begin != end);
BOOST_ASSERT(token_literal == this->traits_.get_token(begin, end));
escape_value esc = { 0, 0, 0, detail::escape_char };
string_type literal(1, *begin);

for(FwdIter prev = begin, tmp = ++begin; begin != end; prev = begin, begin = tmp)
{
detail::quant_spec spec = { 0, 0, false, &this->hidden_mark_count_ };
if(this->traits_.get_quant_spec(tmp, end, spec))
{
if(literal.size() != 1)
{
begin = prev;
literal.erase(boost::prior(literal.end()));
}
return literal;
}
else switch(this->traits_.get_token(tmp, end))
{
case token_escape:
esc = this->parse_escape(tmp, end);
if(detail::escape_char != esc.type_) return literal;
literal.insert(literal.end(), esc.ch_);
break;
case token_literal:
literal.insert(literal.end(), *tmp++);
break;
default:
return literal;
}
}

return literal;
}

template<typename FwdIter>
string_type parse_quote_meta(FwdIter &begin, FwdIter end)
{
using namespace regex_constants;
FwdIter old_begin = begin, old_end;
while(end != (old_end = begin))
{
switch(this->traits_.get_token(begin, end))
{
case token_quote_meta_end:
return string_type(old_begin, old_end);
case token_escape:
BOOST_XPR_ENSURE_(begin != end, error_escape, "incomplete escape sequence");
BOOST_FALLTHROUGH;
case token_invalid_quantifier:
case token_literal:
++begin;
break;
default:
break;
}
}
return string_type(old_begin, begin);
}

template<typename FwdIter>
escape_value parse_escape(FwdIter &begin, FwdIter end)
{
BOOST_XPR_ENSURE_(begin != end, regex_constants::error_escape, "incomplete escape sequence");

if(0 < this->rxtraits().value(*begin, 10))
{
FwdIter tmp = begin;
int mark_nbr = detail::toi(tmp, end, this->rxtraits(), 10, 999);

if(10 > mark_nbr || mark_nbr <= static_cast<int>(this->mark_count_))
{
begin = tmp;
escape_value esc = {0, mark_nbr, 0, detail::escape_mark};
return esc;
}
}

return detail::parse_escape(begin, end, this->traits_);
}

bool is_upper_(char_type ch) const
{
return 0 != this->upper_ && this->rxtraits().isctype(ch, this->upper_);
}

std::size_t mark_count_;
std::size_t hidden_mark_count_;
CompilerTraits traits_;
typename RegexTraits::char_class_type upper_;
shared_ptr<detail::regex_impl<BidiIter> > self_;
std::map<string_type, basic_regex<BidiIter> > rules_;
};

}} 

#endif

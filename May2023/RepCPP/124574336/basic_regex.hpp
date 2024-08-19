
#ifndef BOOST_XPRESSIVE_BASIC_REGEX_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_BASIC_REGEX_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/config.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/xpressive/xpressive_fwd.hpp>
#include <boost/xpressive/regex_constants.hpp>
#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/core/regex_impl.hpp>
#include <boost/xpressive/detail/core/regex_domain.hpp>

#ifndef BOOST_XPRESSIVE_DOXYGEN_INVOKED
# include <boost/xpressive/detail/static/grammar.hpp>
# include <boost/proto/extends.hpp>
#endif

#if BOOST_XPRESSIVE_HAS_MS_STACK_GUARD
# include <excpt.h>     
# include <malloc.h>    
#endif

namespace boost { namespace xpressive
{

namespace detail
{
inline void throw_on_stack_error(bool stack_error)
{
BOOST_XPR_ENSURE_(!stack_error, regex_constants::error_stack, "Regex stack space exhausted");
}
}

template<typename BidiIter>
struct basic_regex
: proto::extends<
proto::expr<proto::tag::terminal, proto::term<detail::tracking_ptr<detail::regex_impl<BidiIter> > >, 0>
, basic_regex<BidiIter>
, detail::regex_domain
>
{
private:
typedef proto::expr<proto::tag::terminal, proto::term<detail::tracking_ptr<detail::regex_impl<BidiIter> > >, 0> pimpl_type;
typedef proto::extends<pimpl_type, basic_regex<BidiIter>, detail::regex_domain> base_type;

public:
typedef BidiIter iterator_type;
typedef typename iterator_value<BidiIter>::type char_type;
typedef typename iterator_value<BidiIter>::type value_type;
typedef typename detail::string_type<char_type>::type string_type;
typedef regex_constants::syntax_option_type flag_type;

BOOST_STATIC_CONSTANT(regex_constants::syntax_option_type, ECMAScript         = regex_constants::ECMAScript);
BOOST_STATIC_CONSTANT(regex_constants::syntax_option_type, icase              = regex_constants::icase_);
BOOST_STATIC_CONSTANT(regex_constants::syntax_option_type, nosubs             = regex_constants::nosubs);
BOOST_STATIC_CONSTANT(regex_constants::syntax_option_type, optimize           = regex_constants::optimize);
BOOST_STATIC_CONSTANT(regex_constants::syntax_option_type, collate            = regex_constants::collate);
BOOST_STATIC_CONSTANT(regex_constants::syntax_option_type, single_line        = regex_constants::single_line);
BOOST_STATIC_CONSTANT(regex_constants::syntax_option_type, not_dot_null       = regex_constants::not_dot_null);
BOOST_STATIC_CONSTANT(regex_constants::syntax_option_type, not_dot_newline    = regex_constants::not_dot_newline);
BOOST_STATIC_CONSTANT(regex_constants::syntax_option_type, ignore_white_space = regex_constants::ignore_white_space);

basic_regex()
: base_type()
{
}

basic_regex(basic_regex<BidiIter> const &that)
: base_type(that)
{
}

basic_regex<BidiIter> &operator =(basic_regex<BidiIter> const &that)
{
proto::value(*this) = proto::value(that);
return *this;
}

template<typename Expr>
basic_regex(Expr const &expr)
: base_type()
{
BOOST_XPRESSIVE_CHECK_REGEX(Expr, char_type);
this->compile_(expr, is_valid_regex<Expr, char_type>());
}

template<typename Expr>
basic_regex<BidiIter> &operator =(Expr const &expr)
{
BOOST_XPRESSIVE_CHECK_REGEX(Expr, char_type);
this->compile_(expr, is_valid_regex<Expr, char_type>());
return *this;
}

std::size_t mark_count() const
{
return proto::value(*this) ? proto::value(*this)->mark_count_ : 0;
}

regex_id_type regex_id() const
{
return proto::value(*this) ? proto::value(*this)->xpr_.get() : 0;
}

void swap(basic_regex<BidiIter> &that) 
{
proto::value(*this).swap(proto::value(that));
}

template<typename InputIter>
static basic_regex<BidiIter> compile(InputIter begin, InputIter end, flag_type flags = regex_constants::ECMAScript)
{
return regex_compiler<BidiIter>().compile(begin, end, flags);
}

template<typename InputRange>
static basic_regex<BidiIter> compile(InputRange const &pat, flag_type flags = regex_constants::ECMAScript)
{
return regex_compiler<BidiIter>().compile(pat, flags);
}

static basic_regex<BidiIter> compile(char_type const *begin, flag_type flags = regex_constants::ECMAScript)
{
return regex_compiler<BidiIter>().compile(begin, flags);
}

static basic_regex<BidiIter> compile(char_type const *begin, std::size_t len, flag_type flags)
{
return regex_compiler<BidiIter>().compile(begin, len, flags);
}

private:
friend struct detail::core_access<BidiIter>;


basic_regex(char_type const *);
basic_regex(string_type const &);

bool match_(detail::match_state<BidiIter> &state) const
{
#if BOOST_XPRESSIVE_HAS_MS_STACK_GUARD
bool success = false, stack_error = false;
__try
{
success = proto::value(*this)->xpr_->match(state);
}
__except(_exception_code() == 0xC00000FDUL)
{
stack_error = true;
_resetstkoflw();
}
detail::throw_on_stack_error(stack_error);
return success;
#else
return proto::value(*this)->xpr_->match(state);
#endif
}

template<typename Expr>
void compile_(Expr const &expr, mpl::true_)
{
detail::static_compile(expr, proto::value(*this).get());
}

template<typename Expr>
void compile_(Expr const &, mpl::false_)
{
}
};

#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
template<typename BidiIter> regex_constants::syntax_option_type const basic_regex<BidiIter>::ECMAScript;
template<typename BidiIter> regex_constants::syntax_option_type const basic_regex<BidiIter>::icase;
template<typename BidiIter> regex_constants::syntax_option_type const basic_regex<BidiIter>::nosubs;
template<typename BidiIter> regex_constants::syntax_option_type const basic_regex<BidiIter>::optimize;
template<typename BidiIter> regex_constants::syntax_option_type const basic_regex<BidiIter>::collate;
template<typename BidiIter> regex_constants::syntax_option_type const basic_regex<BidiIter>::single_line;
template<typename BidiIter> regex_constants::syntax_option_type const basic_regex<BidiIter>::not_dot_null;
template<typename BidiIter> regex_constants::syntax_option_type const basic_regex<BidiIter>::not_dot_newline;
template<typename BidiIter> regex_constants::syntax_option_type const basic_regex<BidiIter>::ignore_white_space;
#endif

template<typename BidiIter>
inline void swap(basic_regex<BidiIter> &left, basic_regex<BidiIter> &right) 
{
left.swap(right);
}

}} 

#endif 

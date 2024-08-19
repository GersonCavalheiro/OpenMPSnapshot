
#ifndef BOOST_SPIRIT_EPSILON_HPP
#define BOOST_SPIRIT_EPSILON_HPP

#include <boost/spirit/home/classic/namespace.hpp>
#include <boost/spirit/home/classic/core/parser.hpp>
#include <boost/spirit/home/classic/meta/parser_traits.hpp>
#include <boost/spirit/home/classic/core/composite/composite.hpp>
#include <boost/spirit/home/classic/core/composite/no_actions.hpp>

#if defined(BOOST_MSVC)
# pragma warning(push)
# pragma warning(disable: 4800) 
#endif

namespace boost { namespace spirit {

BOOST_SPIRIT_CLASSIC_NAMESPACE_BEGIN

template <typename CondT, bool positive_ = true>
struct condition_parser : parser<condition_parser<CondT, positive_> >
{
typedef condition_parser<CondT, positive_> self_t;

condition_parser(CondT const& cond_) : cond(cond_) {}

template <typename ScannerT>
typename parser_result<self_t, ScannerT>::type
parse(ScannerT const& scan) const
{
if (positive_ == bool(cond())) 
return scan.empty_match();
else
return scan.no_match();
}

condition_parser<CondT, !positive_>
negate() const
{ return condition_parser<CondT, !positive_>(cond); }

private:

CondT cond;
};

#if BOOST_WORKAROUND(BOOST_MSVC, == 1310) || \
BOOST_WORKAROUND(BOOST_MSVC, == 1400) || \
BOOST_WORKAROUND(__SUNPRO_CC, <= 0x580)
template <typename CondT>
inline condition_parser<CondT, false>
operator~(condition_parser<CondT, true> const& p)
{ return p.negate(); }

template <typename CondT>
inline condition_parser<CondT, true>
operator~(condition_parser<CondT, false> const& p)
{ return p.negate(); }
#else 
template <typename CondT, bool positive>
inline condition_parser<CondT, !positive>
operator~(condition_parser<CondT, positive> const& p)
{ return p.negate(); }
#endif 

struct empty_match_parser_gen;
struct negated_empty_match_parser_gen;

template <typename SubjectT>
struct negated_empty_match_parser; 

template<typename SubjectT>
struct empty_match_parser
: unary<SubjectT, parser<empty_match_parser<SubjectT> > >
{
typedef empty_match_parser<SubjectT>        self_t;
typedef unary<SubjectT, parser<self_t> >    base_t;
typedef unary_parser_category               parser_category_t;
typedef empty_match_parser_gen              parser_genererator_t;
typedef self_t embed_t;

explicit empty_match_parser(SubjectT const& p) : base_t(p) {}

template <typename ScannerT>
struct result
{ typedef typename match_result<ScannerT, nil_t>::type type; };

template <typename ScannerT>
typename parser_result<self_t, ScannerT>::type
parse(ScannerT const& scan) const
{
typename ScannerT::iterator_t save(scan.first);

typedef typename no_actions_scanner<ScannerT>::policies_t
policies_t;

bool matches = this->subject().parse(
scan.change_policies(policies_t(scan)));
if (matches)
{
scan.first = save; 
return scan.empty_match();
}
else
{
return scan.no_match();
}
}

negated_empty_match_parser<SubjectT>
negate() const
{ return negated_empty_match_parser<SubjectT>(this->subject()); }
};

template<typename SubjectT>
struct negated_empty_match_parser
: public unary<SubjectT, parser<negated_empty_match_parser<SubjectT> > >
{
typedef negated_empty_match_parser<SubjectT>    self_t;
typedef unary<SubjectT, parser<self_t> >        base_t;
typedef unary_parser_category                   parser_category_t;
typedef negated_empty_match_parser_gen          parser_genererator_t;

explicit negated_empty_match_parser(SubjectT const& p) : base_t(p) {}

template <typename ScannerT>
struct result
{ typedef typename match_result<ScannerT, nil_t>::type type; };

template <typename ScannerT>
typename parser_result<self_t, ScannerT>::type
parse(ScannerT const& scan) const
{
typename ScannerT::iterator_t save(scan.first);

typedef typename no_actions_scanner<ScannerT>::policies_t
policies_t;

bool matches = this->subject().parse(
scan.change_policies(policies_t(scan)));
if (!matches)
{
scan.first = save; 
return scan.empty_match();
}
else
{
return scan.no_match();
}
}

empty_match_parser<SubjectT>
negate() const
{ return empty_match_parser<SubjectT>(this->subject()); }
};

struct empty_match_parser_gen
{
template <typename SubjectT>
struct result
{ typedef empty_match_parser<SubjectT> type; };

template <typename SubjectT>
static empty_match_parser<SubjectT>
generate(parser<SubjectT> const& subject)
{ return empty_match_parser<SubjectT>(subject.derived()); }
};

struct negated_empty_match_parser_gen
{
template <typename SubjectT>
struct result
{ typedef negated_empty_match_parser<SubjectT> type; };

template <typename SubjectT>
static negated_empty_match_parser<SubjectT>
generate(parser<SubjectT> const& subject)
{ return negated_empty_match_parser<SubjectT>(subject.derived()); }
};

template <typename SubjectT>
inline negated_empty_match_parser<SubjectT>
operator~(empty_match_parser<SubjectT> const& p)
{ return p.negate(); }

template <typename SubjectT>
inline empty_match_parser<SubjectT>
operator~(negated_empty_match_parser<SubjectT> const& p)
{ return p.negate(); }

namespace impl
{
template <typename SubjectT>
struct epsilon_selector
{
typedef typename as_parser<SubjectT>::type subject_t;
typedef typename
mpl::if_<
is_parser<subject_t>
,empty_match_parser<subject_t>
,condition_parser<subject_t>
>::type type;
};
}

struct epsilon_parser : public parser<epsilon_parser>
{
typedef epsilon_parser self_t;

epsilon_parser() {}

template <typename ScannerT>
typename parser_result<self_t, ScannerT>::type
parse(ScannerT const& scan) const
{ return scan.empty_match(); }

template <typename SubjectT>
typename impl::epsilon_selector<SubjectT>::type
operator()(SubjectT const& subject) const
{
typedef typename impl::epsilon_selector<SubjectT>::type result_t;
return result_t(subject);
}
};

epsilon_parser const epsilon_p = epsilon_parser();
epsilon_parser const eps_p = epsilon_parser();

BOOST_SPIRIT_CLASSIC_NAMESPACE_END

}} 

#ifdef BOOST_MSVC
# pragma warning (pop)
#endif

#endif

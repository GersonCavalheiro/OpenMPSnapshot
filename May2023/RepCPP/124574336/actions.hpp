
#ifndef BOOST_SPIRIT_ACTIONS_HPP
#define BOOST_SPIRIT_ACTIONS_HPP

#include <boost/spirit/home/classic/namespace.hpp>
#include <boost/spirit/home/classic/core/parser.hpp>
#include <boost/spirit/home/classic/core/composite/composite.hpp>
#include <boost/core/ignore_unused.hpp>

namespace boost { namespace spirit {

BOOST_SPIRIT_CLASSIC_NAMESPACE_BEGIN

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
#pragma warning(push)
#pragma warning(disable:4512) 
#endif

template <typename ParserT, typename ActionT>
class action : public unary<ParserT, parser<action<ParserT, ActionT> > >
{
public:

typedef action<ParserT, ActionT>        self_t;
typedef action_parser_category          parser_category_t;
typedef unary<ParserT, parser<self_t> > base_t;
typedef ActionT                         predicate_t;

template <typename ScannerT>
struct result
{
typedef typename parser_result<ParserT, ScannerT>::type type;
};

action(ParserT const& p, ActionT const& a)
: base_t(p)
, actor(a) {}

template <typename ScannerT>
typename parser_result<self_t, ScannerT>::type
parse(ScannerT const& scan) const
{
typedef typename ScannerT::iterator_t iterator_t;
typedef typename parser_result<self_t, ScannerT>::type result_t;

ignore_unused(scan.at_end()); 
iterator_t save = scan.first;
result_t hit = this->subject().parse(scan);
if (hit)
{
typename result_t::return_t val = hit.value();
scan.do_action(actor, val, save, scan.first);
}
return hit;
}

ActionT const& predicate() const { return actor; }

private:

ActionT actor;
};

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
#pragma warning(pop)
#endif

BOOST_SPIRIT_CLASSIC_NAMESPACE_END

}} 

#endif

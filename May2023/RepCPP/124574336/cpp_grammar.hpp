

#if !defined(BOOST_CPP_GRAMMAR_HPP_FEAEBC2E_2734_428B_A7CA_85E5A415E23E_INCLUDED)
#define BOOST_CPP_GRAMMAR_HPP_FEAEBC2E_2734_428B_A7CA_85E5A415E23E_INCLUDED

#include <boost/spirit/include/classic_core.hpp>
#include <boost/spirit/include/classic_parse_tree.hpp>
#include <boost/spirit/include/classic_parse_tree_utils.hpp>
#include <boost/spirit/include/classic_confix.hpp>
#include <boost/spirit/include/classic_lists.hpp>

#include <boost/wave/wave_config.hpp>
#include <boost/pool/pool_alloc.hpp>

#if BOOST_WAVE_DUMP_PARSE_TREE != 0
#include <map>
#include <boost/spirit/include/classic_tree_to_xml.hpp>
#endif

#include <boost/wave/token_ids.hpp>
#include <boost/wave/grammars/cpp_grammar_gen.hpp>
#include <boost/wave/util/pattern_parser.hpp>

#include <boost/wave/cpp_exceptions.hpp>

#ifdef BOOST_HAS_ABI_HEADERS
#include BOOST_ABI_PREFIX
#endif

namespace boost {
namespace wave {
namespace grammars {

namespace impl {


struct store_found_eof {

store_found_eof(bool &found_eof_) : found_eof(found_eof_) {}

template <typename TokenT>
void operator()(TokenT const &) const
{
found_eof = true;
}

bool &found_eof;
};


template <typename TokenT>
struct store_found_directive {

store_found_directive(TokenT &found_directive_)
:   found_directive(found_directive_) {}

void operator()(TokenT const &token) const
{
found_directive = token;
}

TokenT &found_directive;
};


template <typename ContainerT>
struct store_found_eoltokens {

store_found_eoltokens(ContainerT &found_eoltokens_)
:   found_eoltokens(found_eoltokens_) {}

template <typename IteratorT>
void operator()(IteratorT const &first, IteratorT const& last) const
{
std::copy(first, last,
std::inserter(found_eoltokens, found_eoltokens.end()));
}

ContainerT &found_eoltokens;
};

struct flush_underlying_parser
:   public boost::spirit::classic::parser<flush_underlying_parser>
{
typedef flush_underlying_parser this_t;

template <typename ScannerT>
typename boost::spirit::classic::parser_result<this_t, ScannerT>::type
parse(ScannerT const& scan) const
{
scan.first.clear_queue();
return scan.empty_match();
}
};

flush_underlying_parser const
flush_underlying_parser_p = flush_underlying_parser();

}   

#define TRACE_CPP_GRAMMAR \
bool(BOOST_SPIRIT_DEBUG_FLAGS_CPP & BOOST_SPIRIT_DEBUG_FLAGS_CPP_GRAMMAR) \


template <typename TokenT, typename ContainerT>
struct cpp_grammar :
public boost::spirit::classic::grammar<cpp_grammar<TokenT, ContainerT> >
{
typedef typename TokenT::position_type  position_type;
typedef cpp_grammar<TokenT, ContainerT> grammar_type;
typedef impl::store_found_eof           store_found_eof_type;
typedef impl::store_found_directive<TokenT>     store_found_directive_type;
typedef impl::store_found_eoltokens<ContainerT> store_found_eoltokens_type;

template <typename ScannerT>
struct definition
{
typedef typename ScannerT::iteration_policy_t iteration_policy_t;
typedef boost::spirit::classic::match_policy match_policy_t;
typedef typename ScannerT::action_policy_t action_policy_t;
typedef
boost::spirit::classic::scanner_policies<
iteration_policy_t, match_policy_t, action_policy_t>
policies_t;
typedef
boost::spirit::classic::scanner<typename ScannerT::iterator_t, policies_t>
non_tree_scanner_t;
typedef
boost::spirit::classic::rule<
non_tree_scanner_t, boost::spirit::classic::dynamic_parser_tag>
no_tree_rule_type;

typedef
boost::spirit::classic::rule<
ScannerT, boost::spirit::classic::dynamic_parser_tag>
rule_type;

rule_type pp_statement, macro_include_file;
rule_type plain_define, macro_definition, macro_parameters;
rule_type undefine;
rule_type ppifdef, ppifndef, ppif, ppelif;
rule_type ppline;
rule_type pperror;
rule_type ppwarning;
rule_type pppragma;
rule_type illformed;
rule_type ppqualifiedname;
rule_type eol_tokens;
no_tree_rule_type ppsp;
#if BOOST_WAVE_SUPPORT_MS_EXTENSIONS != 0
rule_type ppregion;
rule_type ppendregion;
#endif

definition(cpp_grammar const &self)
{
using namespace boost::spirit::classic;
using namespace boost::wave;
using namespace boost::wave::util;

pp_statement.set_id(BOOST_WAVE_PP_STATEMENT_ID);
macro_include_file.set_id(BOOST_WAVE_MACROINCLUDE_FILE_ID);
plain_define.set_id(BOOST_WAVE_PLAIN_DEFINE_ID);
macro_parameters.set_id(BOOST_WAVE_MACRO_PARAMETERS_ID);
macro_definition.set_id(BOOST_WAVE_MACRO_DEFINITION_ID);
undefine.set_id(BOOST_WAVE_UNDEFINE_ID);
ppifdef.set_id(BOOST_WAVE_IFDEF_ID);
ppifndef.set_id(BOOST_WAVE_IFNDEF_ID);
ppif.set_id(BOOST_WAVE_IF_ID);
ppelif.set_id(BOOST_WAVE_ELIF_ID);
ppline.set_id(BOOST_WAVE_LINE_ID);
pperror.set_id(BOOST_WAVE_ERROR_ID);
ppwarning.set_id(BOOST_WAVE_WARNING_ID);
pppragma.set_id(BOOST_WAVE_PRAGMA_ID);
illformed.set_id(BOOST_WAVE_ILLFORMED_ID);
ppsp.set_id(BOOST_WAVE_PPSPACE_ID);
ppqualifiedname.set_id(BOOST_WAVE_PPQUALIFIEDNAME_ID);
#if BOOST_WAVE_SUPPORT_MS_EXTENSIONS != 0
ppregion.set_id(BOOST_WAVE_REGION_ID);
ppendregion.set_id(BOOST_WAVE_ENDREGION_ID);
#endif

#if BOOST_WAVE_DUMP_PARSE_TREE != 0
self.map_rule_id_to_name.init_rule_id_to_name_map(self);
#endif



pp_statement
=   (   plain_define
|   ppif
|   ppelif
|   ppifndef
|   ppifdef
|   undefine
|   macro_include_file
|   ppline
|   pppragma
|   pperror
|   ppwarning
#if BOOST_WAVE_SUPPORT_MS_EXTENSIONS != 0
|   ppregion
|   ppendregion
#endif
|   illformed
)
>> eol_tokens
[ store_found_eoltokens_type(self.found_eoltokens) ]
#if !(defined(BOOST_SPIRIT_DEBUG) && \
(BOOST_SPIRIT_DEBUG_FLAGS_CPP & BOOST_SPIRIT_DEBUG_FLAGS_CPP_GRAMMAR) \
)
>>  impl::flush_underlying_parser_p
#endif 
;



macro_include_file      
=   no_node_d
[
ch_p(T_PP_INCLUDE)
[ store_found_directive_type(self.found_directive) ]
#if BOOST_WAVE_SUPPORT_INCLUDE_NEXT != 0
|   ch_p(T_PP_INCLUDE_NEXT)
[ store_found_directive_type(self.found_directive) ]
#endif
]
>> *(   anychar_p -
(ch_p(T_NEWLINE) | ch_p(T_CPPCOMMENT) | ch_p(T_EOF))
)
;

plain_define
=   no_node_d
[
ch_p(T_PP_DEFINE)
[ store_found_directive_type(self.found_directive) ]
>> +ppsp
]
>>  (   ch_p(T_IDENTIFIER)
|   pattern_p(KeywordTokenType,
TokenTypeMask|PPTokenFlag)
|   pattern_p(OperatorTokenType|AltExtTokenType,
ExtTokenTypeMask|PPTokenFlag)   
|   pattern_p(BoolLiteralTokenType,
TokenTypeMask|PPTokenFlag)  
)
>>  (   (   no_node_d[eps_p(ch_p(T_LEFTPAREN))]
>>  macro_parameters
>> !macro_definition
)
|  !(   no_node_d[+ppsp]
>>  macro_definition
)
)
;

macro_parameters
=   confix_p(
no_node_d[ch_p(T_LEFTPAREN) >> *ppsp],
!list_p(
(   ch_p(T_IDENTIFIER)
|   pattern_p(KeywordTokenType,
TokenTypeMask|PPTokenFlag)
|   pattern_p(OperatorTokenType|AltExtTokenType,
ExtTokenTypeMask|PPTokenFlag)   
|   pattern_p(BoolLiteralTokenType,
TokenTypeMask|PPTokenFlag)  
#if BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS != 0
|   ch_p(T_ELLIPSIS)
#endif
),
no_node_d[*ppsp >> ch_p(T_COMMA) >> *ppsp]
),
no_node_d[*ppsp >> ch_p(T_RIGHTPAREN)]
)
;

macro_definition
=   no_node_d[*ppsp]
>> *(   anychar_p -
(ch_p(T_NEWLINE) | ch_p(T_CPPCOMMENT) | ch_p(T_EOF))
)
;

undefine
=   no_node_d
[
ch_p(T_PP_UNDEF)
[ store_found_directive_type(self.found_directive) ]
>> +ppsp
]
>>  (   ch_p(T_IDENTIFIER)
|   pattern_p(KeywordTokenType,
TokenTypeMask|PPTokenFlag)
|   pattern_p(OperatorTokenType|AltExtTokenType,
ExtTokenTypeMask|PPTokenFlag)   
|   pattern_p(BoolLiteralTokenType,
TokenTypeMask|PPTokenFlag)  
)
;

ppifdef
=   no_node_d
[
ch_p(T_PP_IFDEF)
[ store_found_directive_type(self.found_directive) ]
>> +ppsp
]
>>  ppqualifiedname
;

ppifndef
=   no_node_d
[
ch_p(T_PP_IFNDEF)
[ store_found_directive_type(self.found_directive) ]
>> +ppsp
]
>>  ppqualifiedname
;

ppif
=   no_node_d
[
ch_p(T_PP_IF)
[ store_found_directive_type(self.found_directive) ]
]
>> +(   anychar_p -
(ch_p(T_NEWLINE) | ch_p(T_CPPCOMMENT) | ch_p(T_EOF))
)
;


ppelif
=   no_node_d
[
ch_p(T_PP_ELIF)
[ store_found_directive_type(self.found_directive) ]
]
>> +(   anychar_p -
(ch_p(T_NEWLINE) | ch_p(T_CPPCOMMENT) | ch_p(T_EOF))
)
;


ppline
=   no_node_d
[
ch_p(T_PP_LINE)
[ store_found_directive_type(self.found_directive) ]
>> *ppsp
]
>> +(   anychar_p -
(ch_p(T_NEWLINE) | ch_p(T_CPPCOMMENT) | ch_p(T_EOF))
)
;

#if BOOST_WAVE_SUPPORT_MS_EXTENSIONS != 0
ppregion
=   no_node_d
[
ch_p(T_MSEXT_PP_REGION)
[ store_found_directive_type(self.found_directive) ]
>> +ppsp
]
>>  ppqualifiedname
;

ppendregion
=   no_node_d
[
ch_p(T_MSEXT_PP_ENDREGION)
[ store_found_directive_type(self.found_directive) ]
]
;
#endif

illformed           
=   no_node_d
[
pattern_p(T_POUND, MainTokenMask)
>> *ppsp
]
>>  (   anychar_p -
(ch_p(T_NEWLINE) | ch_p(T_CPPCOMMENT) | ch_p(T_EOF))
)
>>  no_node_d
[
*(   anychar_p -
(ch_p(T_NEWLINE) | ch_p(T_CPPCOMMENT) | ch_p(T_EOF))
)
]
;

pperror
=   no_node_d
[
ch_p(T_PP_ERROR)
[ store_found_directive_type(self.found_directive) ]
>> *ppsp
]
>> *(   anychar_p -
(ch_p(T_NEWLINE) | ch_p(T_CPPCOMMENT) | ch_p(T_EOF))
)
;

ppwarning
=   no_node_d
[
ch_p(T_PP_WARNING)
[ store_found_directive_type(self.found_directive) ]
>> *ppsp
]
>> *(   anychar_p -
(ch_p(T_NEWLINE) | ch_p(T_CPPCOMMENT) | ch_p(T_EOF))
)
;

pppragma
=   no_node_d
[
ch_p(T_PP_PRAGMA)
[ store_found_directive_type(self.found_directive) ]
]
>> *(   anychar_p -
(ch_p(T_NEWLINE) | ch_p(T_CPPCOMMENT) | ch_p(T_EOF))
)
;

ppqualifiedname
=   no_node_d[*ppsp]
>>  (   ch_p(T_IDENTIFIER)
|   pattern_p(KeywordTokenType,
TokenTypeMask|PPTokenFlag)
|   pattern_p(OperatorTokenType|AltExtTokenType,
ExtTokenTypeMask|PPTokenFlag)   
|   pattern_p(BoolLiteralTokenType,
TokenTypeMask|PPTokenFlag)  
)
;

ppsp     
=   ch_p(T_SPACE) | ch_p(T_CCOMMENT)
;

eol_tokens
=   no_node_d
[
*(   ch_p(T_SPACE)
|   ch_p(T_CCOMMENT)
)
>>  (   ch_p(T_NEWLINE)
|   ch_p(T_CPPCOMMENT)
|   ch_p(T_EOF)
[ store_found_eof_type(self.found_eof) ]
)
]
;

BOOST_SPIRIT_DEBUG_TRACE_RULE(pp_statement, TRACE_CPP_GRAMMAR);
BOOST_SPIRIT_DEBUG_TRACE_RULE(macro_include_file, TRACE_CPP_GRAMMAR);
BOOST_SPIRIT_DEBUG_TRACE_RULE(plain_define, TRACE_CPP_GRAMMAR);
BOOST_SPIRIT_DEBUG_TRACE_RULE(macro_definition, TRACE_CPP_GRAMMAR);
BOOST_SPIRIT_DEBUG_TRACE_RULE(macro_parameters, TRACE_CPP_GRAMMAR);
BOOST_SPIRIT_DEBUG_TRACE_RULE(undefine, TRACE_CPP_GRAMMAR);
BOOST_SPIRIT_DEBUG_TRACE_RULE(ppifdef, TRACE_CPP_GRAMMAR);
BOOST_SPIRIT_DEBUG_TRACE_RULE(ppifndef, TRACE_CPP_GRAMMAR);
BOOST_SPIRIT_DEBUG_TRACE_RULE(ppif, TRACE_CPP_GRAMMAR);
BOOST_SPIRIT_DEBUG_TRACE_RULE(ppendif, TRACE_CPP_GRAMMAR);
BOOST_SPIRIT_DEBUG_TRACE_RULE(ppline, TRACE_CPP_GRAMMAR);
BOOST_SPIRIT_DEBUG_TRACE_RULE(pperror, TRACE_CPP_GRAMMAR);
BOOST_SPIRIT_DEBUG_TRACE_RULE(ppwarning, TRACE_CPP_GRAMMAR);
BOOST_SPIRIT_DEBUG_TRACE_RULE(illformed, TRACE_CPP_GRAMMAR);
BOOST_SPIRIT_DEBUG_TRACE_RULE(ppsp, TRACE_CPP_GRAMMAR);
BOOST_SPIRIT_DEBUG_TRACE_RULE(ppqualifiedname, TRACE_CPP_GRAMMAR);
#if BOOST_WAVE_SUPPORT_MS_EXTENSIONS != 0
BOOST_SPIRIT_DEBUG_TRACE_RULE(ppregion, TRACE_CPP_GRAMMAR);
BOOST_SPIRIT_DEBUG_TRACE_RULE(ppendregion, TRACE_CPP_GRAMMAR);
#endif
}

rule_type const& start() const
{ return pp_statement; }
};

bool &found_eof;
TokenT &found_directive;
ContainerT &found_eoltokens;

cpp_grammar(bool &found_eof_, TokenT &found_directive_,
ContainerT &found_eoltokens_)
:   found_eof(found_eof_),
found_directive(found_directive_),
found_eoltokens(found_eoltokens_)
{
BOOST_SPIRIT_DEBUG_TRACE_GRAMMAR_NAME(*this, "cpp_grammar",
TRACE_CPP_GRAMMAR);
}

#if BOOST_WAVE_DUMP_PARSE_TREE != 0
struct map_ruleid_to_name :
public std::map<boost::spirit::classic::parser_id, std::string>
{
typedef std::map<boost::spirit::classic::parser_id, std::string> base_type;

void init_rule_id_to_name_map(cpp_grammar const &self)
{
struct {
int parser_id;
char const *rule_name;
}
init_ruleid_name_map[] = {
{ BOOST_WAVE_PP_STATEMENT_ID, "pp_statement" },
{ BOOST_WAVE_MACROINCLUDE_FILE_ID, "macro_include_file" },
{ BOOST_WAVE_PLAIN_DEFINE_ID, "plain_define" },
{ BOOST_WAVE_MACRO_PARAMETERS_ID, "macro_parameters" },
{ BOOST_WAVE_MACRO_DEFINITION_ID, "macro_definition" },
{ BOOST_WAVE_UNDEFINE_ID, "undefine" },
{ BOOST_WAVE_IFDEF_ID, "ppifdef" },
{ BOOST_WAVE_IFNDEF_ID, "ppifndef" },
{ BOOST_WAVE_IF_ID, "ppif" },
{ BOOST_WAVE_ELIF_ID, "ppelif" },
{ BOOST_WAVE_LINE_ID, "ppline" },
{ BOOST_WAVE_ERROR_ID, "pperror" },
{ BOOST_WAVE_WARNING_ID, "ppwarning" },
{ BOOST_WAVE_PRAGMA_ID, "pppragma" },
{ BOOST_WAVE_ILLFORMED_ID, "illformed" },
{ BOOST_WAVE_PPSPACE_ID, "ppspace" },
{ BOOST_WAVE_PPQUALIFIEDNAME_ID, "ppqualifiedname" },
#if BOOST_WAVE_SUPPORT_MS_EXTENSIONS != 0
{ BOOST_WAVE_REGION_ID, "ppregion" },
{ BOOST_WAVE_ENDREGION_ID, "ppendregion" },
#endif
{ 0 }
};

for (int i = 0; 0 != init_ruleid_name_map[i].parser_id; ++i)
base_type::insert(base_type::value_type(
boost::spirit::classic::parser_id(init_ruleid_name_map[i].parser_id),
std::string(init_ruleid_name_map[i].rule_name))
);
}
};
mutable map_ruleid_to_name map_rule_id_to_name;
#endif 
};

#undef TRACE_CPP_GRAMMAR

template <typename NodeFactoryT, typename IteratorT, typename ParserT>
inline boost::spirit::classic::tree_parse_info<IteratorT, NodeFactoryT>
parsetree_parse(IteratorT const& first_, IteratorT const& last,
boost::spirit::classic::parser<ParserT> const& p)
{
using namespace boost::spirit::classic;

typedef pt_match_policy<IteratorT, NodeFactoryT> pt_match_policy_type;
typedef scanner_policies<iteration_policy, pt_match_policy_type>
scanner_policies_type;
typedef scanner<IteratorT, scanner_policies_type> scanner_type;

scanner_policies_type policies;
IteratorT first = first_;
scanner_type scan(first, last, policies);
tree_match<IteratorT, NodeFactoryT> hit = p.derived().parse(scan);
return tree_parse_info<IteratorT, NodeFactoryT>(
first, hit, hit && (first == last), hit.length(), hit.trees);
}


#if BOOST_WAVE_SEPARATE_GRAMMAR_INSTANTIATION != 0
#define BOOST_WAVE_GRAMMAR_GEN_INLINE
#else
#define BOOST_WAVE_GRAMMAR_GEN_INLINE inline
#endif

template <typename LexIteratorT, typename TokenContainerT>
BOOST_WAVE_GRAMMAR_GEN_INLINE
boost::spirit::classic::tree_parse_info<
LexIteratorT,
typename cpp_grammar_gen<LexIteratorT, TokenContainerT>::node_factory_type
>
cpp_grammar_gen<LexIteratorT, TokenContainerT>::parse_cpp_grammar (
LexIteratorT const &first, LexIteratorT const &last,
position_type const &act_pos, bool &found_eof,
token_type &found_directive, token_container_type &found_eoltokens)
{
using namespace boost::spirit::classic;
using namespace boost::wave;

cpp_grammar<token_type, TokenContainerT> g(found_eof, found_directive, found_eoltokens);
tree_parse_info<LexIteratorT, node_factory_type> hit =
parsetree_parse<node_factory_type>(first, last, g);

#if BOOST_WAVE_DUMP_PARSE_TREE != 0
if (hit.match) {
tree_to_xml (BOOST_WAVE_DUMP_PARSE_TREE_OUT, hit.trees, "",
g.map_rule_id_to_name, &token_type::get_token_id,
&token_type::get_token_value);
}
#endif

return hit;
}

#undef BOOST_WAVE_GRAMMAR_GEN_INLINE

}   
}   
}   

#ifdef BOOST_HAS_ABI_HEADERS
#include BOOST_ABI_SUFFIX
#endif

#endif 

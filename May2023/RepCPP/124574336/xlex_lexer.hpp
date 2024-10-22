

#if !defined(BOOST_XLEX_LEXER_HPP)
#define BOOST_XLEX_LEXER_HPP

#include <string>
#include <cstdio>
#include <cstdarg>
#if defined(BOOST_SPIRIT_DEBUG)
#include <iostream>
#endif 

#include <boost/concept_check.hpp>
#include <boost/assert.hpp>
#include <boost/spirit/include/classic_core.hpp>

#include <boost/wave/token_ids.hpp>
#include <boost/wave/language_support.hpp>
#include <boost/wave/util/file_position.hpp>
#include <boost/wave/cpplexer/validate_universal_char.hpp>
#include <boost/wave/cpplexer/cpplexer_exceptions.hpp>
#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
#include <boost/wave/cpplexer/detect_include_guards.hpp>
#endif
#include <boost/wave/cpplexer/cpp_lex_interface.hpp>

#include "../xlex_interface.hpp"

#include "xpressive_lexer.hpp"

namespace boost {
namespace wave {
namespace cpplexer {
namespace xlex {
namespace lexer {


template <
typename Iterator, 
typename Position = boost::wave::util::file_position_type
>
class lexer 
{
public:
typedef char                                        char_type;
typedef boost::wave::cpplexer::lex_token<Position>  token_type;
typedef typename token_type::string_type            string_type;

lexer(Iterator const &first, Iterator const &last, 
Position const &pos, boost::wave::language_support language);
~lexer() {}

token_type& get(token_type& t);
void set_position(Position const &pos)
{
filename = pos.get_file();
line = pos.get_line();
column = pos.get_column();
}

#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
bool has_include_guards(std::string& guard_name) const 
{ return guards.detected(guard_name); }
#endif

private:
typedef xpressive_lexer<Iterator, token_id> lexer_type;
typedef typename lexer_type::callback_type callback_type;

lexer_type xlexer;
Iterator first;
Iterator last;

string_type filename;
int line;
int column;
bool at_eof;
boost::wave::language_support language;

struct lexer_data {
token_id tokenid;                 
char_type const *tokenregex;      
callback_type tokencb;            
};

static lexer_data const init_data[];        
static lexer_data const init_data_cpp[];    
static lexer_data const init_data_cpp0x[];  
static lexer_data const init_data_cpp2a[];  

#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
boost::wave::cpplexer::include_guards<token_type> guards;
#endif
};

#define TOKEN_DATA(id, regex) \
{ id, regex, 0 }

#define TOKEN_DATA_EX(id, regex, callback) \
{ id, regex, callback }

#define OR                  "|"
#define Q(c)                "\\" c
#define TRI(c)              Q("?") Q("?") c

#define BLANK               "[ \t\v\f]"
#define CCOMMENT            Q("/") Q("*") ".*?" Q("*") Q("/")

#define PPSPACE             "(" BLANK OR CCOMMENT ")*"

#define OCTALDIGIT          "[0-7]"
#define DIGIT               "[0-9]"
#define HEXDIGIT            "[0-9a-fA-F]"
#define SIGN                "[-+]?"
#define EXPONENT            "(" "[eE]" SIGN "[0-9]+" ")"

#define INTEGER             "(" \
"(0x|0X)" HEXDIGIT "+" OR \
"0" OCTALDIGIT "*" OR \
"[1-9]" DIGIT "*" \
")"

#define INTEGER_SUFFIX      "(" "[uU][lL]?|[lL][uU]?" ")"
#if BOOST_WAVE_SUPPORT_MS_EXTENSIONS != 0
#define LONGINTEGER_SUFFIX  "(" "[uU]" "(" "ll" OR "LL" ")" OR \
"(" "ll" OR "LL" ")" "[uU]" "?" OR \
"i64" \
")" 
#else
#define LONGINTEGER_SUFFIX  "(" "[uU]" "(" "ll" OR "LL" ")" OR \
"(" "ll" OR "LL" ")" "[uU]" "?" ")"
#endif
#define FLOAT_SUFFIX        "(" "[fF][lL]?|[lL][fF]?" ")"
#define CHAR_SPEC           "L?"
#define EXTCHAR_SPEC        "(" "[uU]" OR "u8" ")"

#define BACKSLASH           "(" Q("\\") OR TRI(Q("/")) ")"
#define ESCAPESEQ           BACKSLASH "(" \
"[abfnrtv?'\"]" OR \
BACKSLASH OR \
"x" HEXDIGIT "+" OR \
OCTALDIGIT OCTALDIGIT "?" OCTALDIGIT "?" \
")"
#define HEXQUAD             HEXDIGIT HEXDIGIT HEXDIGIT HEXDIGIT 
#define UNIVERSALCHAR       BACKSLASH "(" \
"u" HEXQUAD OR \
"U" HEXQUAD HEXQUAD \
")" 

#define POUNDDEF            "(" "#" OR TRI("=") OR Q("%:") ")"
#define NEWLINEDEF          "(" "\n" OR "\r\n" OR "\r" ")"

#if BOOST_WAVE_SUPPORT_INCLUDE_NEXT != 0
#define INCLUDEDEF          "(include_next|include)"
#else
#define INCLUDEDEF          "include"
#endif

template <typename Iterator, typename Position>
typename lexer<Iterator, Position>::lexer_data const 
lexer<Iterator, Position>::init_data[] = 
{
TOKEN_DATA(T_CCOMMENT, CCOMMENT),
TOKEN_DATA(T_CPPCOMMENT, Q("/") Q("/.*?") NEWLINEDEF ),
TOKEN_DATA(T_CHARLIT, CHAR_SPEC "'" 
"(" ESCAPESEQ OR "[^\n\r']" OR UNIVERSALCHAR ")+" "'"),
TOKEN_DATA(T_STRINGLIT, CHAR_SPEC Q("\"") 
"(" ESCAPESEQ OR "[^\n\r\"]" OR UNIVERSALCHAR ")*" Q("\"")),
TOKEN_DATA(T_ANDAND, "&&"),
TOKEN_DATA(T_ANDASSIGN, "&="),
TOKEN_DATA(T_AND, "&"),
TOKEN_DATA(T_EQUAL, "=="),
TOKEN_DATA(T_ASSIGN, "="),
TOKEN_DATA(T_ORASSIGN, Q("|=")),
TOKEN_DATA(T_ORASSIGN_TRIGRAPH, TRI("!=")),
TOKEN_DATA(T_OROR, Q("|") Q("|")),
TOKEN_DATA(T_OROR_TRIGRAPH, TRI("!") Q("|") OR Q("|") TRI("!") OR TRI("!") TRI("!")),
TOKEN_DATA(T_OR, Q("|")),
TOKEN_DATA(T_OR_TRIGRAPH, TRI("!")),
TOKEN_DATA(T_XORASSIGN, Q("^=")),
TOKEN_DATA(T_XORASSIGN_TRIGRAPH, TRI("'=")),
TOKEN_DATA(T_XOR, Q("^")),
TOKEN_DATA(T_XOR_TRIGRAPH, TRI("'")),
TOKEN_DATA(T_COMMA, ","),
TOKEN_DATA(T_RIGHTBRACKET_ALT, ":>"),
TOKEN_DATA(T_COLON, ":"),
TOKEN_DATA(T_DIVIDEASSIGN, Q("/=")),
TOKEN_DATA(T_DIVIDE, Q("/")),
TOKEN_DATA(T_ELLIPSIS, Q(".") Q(".") Q(".")),
TOKEN_DATA(T_SHIFTRIGHTASSIGN, ">>="),
TOKEN_DATA(T_SHIFTRIGHT, ">>"),
TOKEN_DATA(T_GREATEREQUAL, ">="),
TOKEN_DATA(T_GREATER, ">"),
TOKEN_DATA(T_LEFTBRACE, Q("{")),
TOKEN_DATA(T_SHIFTLEFTASSIGN, "<<="),
TOKEN_DATA(T_SHIFTLEFT, "<<"),
TOKEN_DATA(T_LEFTBRACE_ALT, "<" Q("%")),
TOKEN_DATA(T_LESSEQUAL, "<="),
TOKEN_DATA(T_LEFTBRACKET_ALT, "<:"),
TOKEN_DATA(T_LESS, "<"),
TOKEN_DATA(T_LEFTBRACE_TRIGRAPH, TRI("<")),
TOKEN_DATA(T_LEFTPAREN, Q("(")),
TOKEN_DATA(T_LEFTBRACKET, Q("[")),
TOKEN_DATA(T_LEFTBRACKET_TRIGRAPH, TRI(Q("("))),
TOKEN_DATA(T_MINUSMINUS, Q("-") Q("-")),
TOKEN_DATA(T_MINUSASSIGN, Q("-=")),
TOKEN_DATA(T_ARROW, Q("->")),
TOKEN_DATA(T_MINUS, Q("-")),
TOKEN_DATA(T_POUND_POUND_ALT, Q("%:") Q("%:")),
TOKEN_DATA(T_PERCENTASSIGN, Q("%=")),
TOKEN_DATA(T_RIGHTBRACE_ALT, Q("%>")),
TOKEN_DATA(T_POUND_ALT, Q("%:")),
TOKEN_DATA(T_PERCENT, Q("%")),
TOKEN_DATA(T_NOTEQUAL, "!="),
TOKEN_DATA(T_NOT, "!"),
TOKEN_DATA(T_PLUSASSIGN, Q("+=")),
TOKEN_DATA(T_PLUSPLUS, Q("+") Q("+")),
TOKEN_DATA(T_PLUS, Q("+")),
TOKEN_DATA(T_RIGHTBRACE, Q("}")),
TOKEN_DATA(T_RIGHTBRACE_TRIGRAPH, TRI(">")),
TOKEN_DATA(T_RIGHTPAREN, Q(")")),
TOKEN_DATA(T_RIGHTBRACKET, Q("]")),
TOKEN_DATA(T_RIGHTBRACKET_TRIGRAPH, TRI(Q(")"))),
TOKEN_DATA(T_SEMICOLON, ";"),
TOKEN_DATA(T_STARASSIGN, Q("*=")),
TOKEN_DATA(T_STAR, Q("*")),
TOKEN_DATA(T_COMPL, Q("~")),
TOKEN_DATA(T_COMPL_TRIGRAPH, TRI("-")),
TOKEN_DATA(T_ASM, "asm"),
TOKEN_DATA(T_AUTO, "auto"),
TOKEN_DATA(T_BOOL, "bool"),
TOKEN_DATA(T_FALSE, "false"),
TOKEN_DATA(T_TRUE, "true"),
TOKEN_DATA(T_BREAK, "break"),
TOKEN_DATA(T_CASE, "case"),
TOKEN_DATA(T_CATCH, "catch"),
TOKEN_DATA(T_CHAR, "char"),
TOKEN_DATA(T_CLASS, "class"),
TOKEN_DATA(T_CONSTCAST, "const_cast"),
TOKEN_DATA(T_CONST, "const"),
TOKEN_DATA(T_CONTINUE, "continue"),
TOKEN_DATA(T_DEFAULT, "default"),
TOKEN_DATA(T_DELETE, "delete"),
TOKEN_DATA(T_DOUBLE, "double"),
TOKEN_DATA(T_DO, "do"),
TOKEN_DATA(T_DYNAMICCAST, "dynamic_cast"),
TOKEN_DATA(T_ELSE, "else"),
TOKEN_DATA(T_ENUM, "enum"),
TOKEN_DATA(T_EXPLICIT, "explicit"),
TOKEN_DATA(T_EXPORT, "export"),
TOKEN_DATA(T_EXTERN, "extern"),
TOKEN_DATA(T_FLOAT, "float"),
TOKEN_DATA(T_FOR, "for"),
TOKEN_DATA(T_FRIEND, "friend"),
TOKEN_DATA(T_GOTO, "goto"),
TOKEN_DATA(T_IF, "if"),
TOKEN_DATA(T_INLINE, "inline"),
TOKEN_DATA(T_INT, "int"),
TOKEN_DATA(T_LONG, "long"),
TOKEN_DATA(T_MUTABLE, "mutable"),
TOKEN_DATA(T_NAMESPACE, "namespace"),
TOKEN_DATA(T_NEW, "new"),
TOKEN_DATA(T_OPERATOR, "operator"),
TOKEN_DATA(T_PRIVATE, "private"),
TOKEN_DATA(T_PROTECTED, "protected"),
TOKEN_DATA(T_PUBLIC, "public"),
TOKEN_DATA(T_REGISTER, "register"),
TOKEN_DATA(T_REINTERPRETCAST, "reinterpret_cast"),
TOKEN_DATA(T_RETURN, "return"),
TOKEN_DATA(T_SHORT, "short"),
TOKEN_DATA(T_SIGNED, "signed"),
TOKEN_DATA(T_SIZEOF, "sizeof"),
TOKEN_DATA(T_STATICCAST, "static_cast"),
TOKEN_DATA(T_STATIC, "static"),
TOKEN_DATA(T_STRUCT, "struct"),
TOKEN_DATA(T_SWITCH, "switch"),
TOKEN_DATA(T_TEMPLATE, "template"),
TOKEN_DATA(T_THIS, "this"),
TOKEN_DATA(T_THROW, "throw"),
TOKEN_DATA(T_TRY, "try"),
TOKEN_DATA(T_TYPEDEF, "typedef"),
TOKEN_DATA(T_TYPEID, "typeid"),
TOKEN_DATA(T_TYPENAME, "typename"),
TOKEN_DATA(T_UNION, "union"),
TOKEN_DATA(T_UNSIGNED, "unsigned"),
TOKEN_DATA(T_USING, "using"),
TOKEN_DATA(T_VIRTUAL, "virtual"),
TOKEN_DATA(T_VOID, "void"),
TOKEN_DATA(T_VOLATILE, "volatile"),
TOKEN_DATA(T_WCHART, "wchar_t"),
TOKEN_DATA(T_WHILE, "while"),
#if BOOST_WAVE_SUPPORT_MS_EXTENSIONS != 0
TOKEN_DATA(T_MSEXT_INT8, "__int8"),
TOKEN_DATA(T_MSEXT_INT16, "__int16"),
TOKEN_DATA(T_MSEXT_INT32, "__int32"),
TOKEN_DATA(T_MSEXT_INT64, "__int64"),
TOKEN_DATA(T_MSEXT_BASED, "_?" "_based"),
TOKEN_DATA(T_MSEXT_DECLSPEC, "_?" "_declspec"),
TOKEN_DATA(T_MSEXT_CDECL, "_?" "_cdecl"),
TOKEN_DATA(T_MSEXT_FASTCALL, "_?" "_fastcall"),
TOKEN_DATA(T_MSEXT_STDCALL, "_?" "_stdcall"),
TOKEN_DATA(T_MSEXT_TRY , "__try"),
TOKEN_DATA(T_MSEXT_EXCEPT, "__except"),
TOKEN_DATA(T_MSEXT_FINALLY, "__finally"),
TOKEN_DATA(T_MSEXT_LEAVE, "__leave"),
TOKEN_DATA(T_MSEXT_INLINE, "_?" "_inline"),
TOKEN_DATA(T_MSEXT_ASM, "_?" "_asm"),
TOKEN_DATA(T_MSEXT_PP_REGION, POUNDDEF PPSPACE "region"),
TOKEN_DATA(T_MSEXT_PP_ENDREGION, POUNDDEF PPSPACE "endregion"),
#endif 
TOKEN_DATA(T_PP_DEFINE, POUNDDEF PPSPACE "define"),
TOKEN_DATA(T_PP_IFDEF, POUNDDEF PPSPACE "ifdef"),
TOKEN_DATA(T_PP_IFNDEF, POUNDDEF PPSPACE "ifndef"),
TOKEN_DATA(T_PP_IF, POUNDDEF PPSPACE "if"),
TOKEN_DATA(T_PP_ELSE, POUNDDEF PPSPACE "else"),
TOKEN_DATA(T_PP_ELIF, POUNDDEF PPSPACE "elif"),
TOKEN_DATA(T_PP_ENDIF, POUNDDEF PPSPACE "endif"),
TOKEN_DATA(T_PP_ERROR, POUNDDEF PPSPACE "error"),
TOKEN_DATA(T_PP_QHEADER, POUNDDEF PPSPACE \
INCLUDEDEF PPSPACE Q("\"") "[^\n\r\"]+" Q("\"")),
TOKEN_DATA(T_PP_HHEADER, POUNDDEF PPSPACE \
INCLUDEDEF PPSPACE "<" "[^\n\r>]+" ">"),
TOKEN_DATA(T_PP_INCLUDE, POUNDDEF PPSPACE \
INCLUDEDEF PPSPACE),
TOKEN_DATA(T_PP_LINE, POUNDDEF PPSPACE "line"),
TOKEN_DATA(T_PP_PRAGMA, POUNDDEF PPSPACE "pragma"),
TOKEN_DATA(T_PP_UNDEF, POUNDDEF PPSPACE "undef"),
TOKEN_DATA(T_PP_WARNING, POUNDDEF PPSPACE "warning"),
TOKEN_DATA(T_FLOATLIT, 
"(" DIGIT "*" Q(".") DIGIT "+" OR DIGIT "+" Q(".") ")" 
EXPONENT "?" FLOAT_SUFFIX "?" OR
DIGIT "+" EXPONENT FLOAT_SUFFIX "?"),
TOKEN_DATA(T_LONGINTLIT, INTEGER LONGINTEGER_SUFFIX),
TOKEN_DATA(T_INTLIT, INTEGER INTEGER_SUFFIX "?"),
#if BOOST_WAVE_USE_STRICT_LEXER != 0
TOKEN_DATA(T_IDENTIFIER, "([a-zA-Z_]" OR UNIVERSALCHAR ")([a-zA-Z0-9_]" OR UNIVERSALCHAR ")*"),
#else
TOKEN_DATA(T_IDENTIFIER, "([a-zA-Z_$]" OR UNIVERSALCHAR ")([a-zA-Z0-9_$]" OR UNIVERSALCHAR ")*"),
#endif
TOKEN_DATA(T_SPACE, BLANK "+"),
TOKEN_DATA(T_CONTLINE, Q("\\") "\n"), 
TOKEN_DATA(T_NEWLINE, NEWLINEDEF),
TOKEN_DATA(T_POUND_POUND, "##"),
TOKEN_DATA(T_POUND_POUND_TRIGRAPH, TRI("=") TRI("=")),
TOKEN_DATA(T_POUND, "#"),
TOKEN_DATA(T_POUND_TRIGRAPH, TRI("=")),
TOKEN_DATA(T_ANY_TRIGRAPH, TRI(Q("/"))),
TOKEN_DATA(T_QUESTION_MARK, Q("?")),
TOKEN_DATA(T_DOT, Q(".")),
TOKEN_DATA(T_ANY, "."),
{ token_id(0) }       
};

template <typename Iterator, typename Position>
typename lexer<Iterator, Position>::lexer_data const 
lexer<Iterator, Position>::init_data_cpp[] = 
{
TOKEN_DATA(T_AND_ALT, "bitand"),
TOKEN_DATA(T_ANDASSIGN_ALT, "and_eq"),
TOKEN_DATA(T_ANDAND_ALT, "and"),
TOKEN_DATA(T_OR_ALT, "bitor"),
TOKEN_DATA(T_ORASSIGN_ALT, "or_eq"),
TOKEN_DATA(T_OROR_ALT, "or"),
TOKEN_DATA(T_XORASSIGN_ALT, "xor_eq"),
TOKEN_DATA(T_XOR_ALT, "xor"),
TOKEN_DATA(T_NOTEQUAL_ALT, "not_eq"),
TOKEN_DATA(T_NOT_ALT, "not"),
TOKEN_DATA(T_COMPL_ALT, "compl"),
TOKEN_DATA(T_ARROWSTAR, Q("->") Q("*")),
TOKEN_DATA(T_DOTSTAR, Q(".") Q("*")),
TOKEN_DATA(T_COLON_COLON, "::"),
{ token_id(0) }       
};

#define T_EXTCHARLIT      token_id(T_CHARLIT|AltTokenType)
#define T_EXTSTRINGLIT    token_id(T_STRINGLIT|AltTokenType)
#define T_EXTRAWSTRINGLIT token_id(T_RAWSTRINGLIT|AltTokenType)

template <typename Iterator, typename Position>
typename lexer<Iterator, Position>::lexer_data const
lexer<Iterator, Position>::init_data_cpp0x[] =
{
TOKEN_DATA(T_EXTCHARLIT, EXTCHAR_SPEC "'"
"(" ESCAPESEQ OR UNIVERSALCHAR OR "[^\\n\\r\\\\']" ")+" "'"),
TOKEN_DATA(T_EXTSTRINGLIT, EXTCHAR_SPEC Q("\"")
"(" ESCAPESEQ OR UNIVERSALCHAR OR "[^\\n\\r\\\\\"]" ")*" Q("\"")),
TOKEN_DATA(T_RAWSTRINGLIT, CHAR_SPEC "R" Q("\"")
"(" ESCAPESEQ OR UNIVERSALCHAR OR "[^\\\\\"]" ")*" Q("\"")),
TOKEN_DATA(T_EXTRAWSTRINGLIT, EXTCHAR_SPEC "R" Q("\"")
"(" ESCAPESEQ OR UNIVERSALCHAR OR "[^\\\\\"]" ")*" Q("\"")),
TOKEN_DATA(T_ALIGNAS, "alignas"),
TOKEN_DATA(T_ALIGNOF, "alignof"),
TOKEN_DATA(T_CHAR16_T, "char16_t"),
TOKEN_DATA(T_CHAR32_T, "char32_t"),
TOKEN_DATA(T_CONSTEXPR, "constexpr"),
TOKEN_DATA(T_DECLTYPE, "decltype"),
TOKEN_DATA(T_NOEXCEPT, "noexcept"),
TOKEN_DATA(T_NULLPTR, "nullptr"),
TOKEN_DATA(T_STATICASSERT, "static_assert"),
TOKEN_DATA(T_THREADLOCAL, "thread_local"),
{ token_id(0) }       
};


template <typename Iterator, typename Position>
typename lexer<Iterator, Position>::lexer_data const
lexer<Iterator, Position>::init_data_cpp2a[] =
{
TOKEN_DATA(T_CHAR8_T, "char8_t"),
TOKEN_DATA(T_CONCEPT, "concept"),
TOKEN_DATA(T_CONSTEVAL, "consteval"),
TOKEN_DATA(T_CONSTINIT, "constinit"),
TOKEN_DATA(T_CO_AWAIT, "co_await"),
TOKEN_DATA(T_CO_RETURN, "co_return"),
TOKEN_DATA(T_CO_YIELD, "co_yield"),
TOKEN_DATA(T_REQUIRES, "requires"),
TOKEN_DATA(T_SPACESHIP, "<=>"),
{ token_id(0) }       
};

#undef INCLUDEDEF
#undef POUNDDEF
#undef CCOMMENT
#undef PPSPACE
#undef DIGIT
#undef OCTALDIGIT
#undef HEXDIGIT
#undef SIGN
#undef EXPONENT
#undef LONGINTEGER_SUFFIX
#undef INTEGER_SUFFIX
#undef INTEGER
#undef FLOAT_SUFFIX
#undef CHAR_SPEC
#undef EXTCHAR_SPEC
#undef BACKSLASH    
#undef ESCAPESEQ    
#undef HEXQUAD      
#undef UNIVERSALCHAR

#undef Q
#undef TRI
#undef OR

#undef TOKEN_DATA
#undef TOKEN_DATA_EX

#undef T_EXTCHARLIT
#undef T_EXTSTRINGLIT
#undef T_EXTRAWSTRINGLIT
template <typename Iterator, typename Position>
inline
lexer<Iterator, Position>::lexer(Iterator const &first, 
Iterator const &last, Position const &pos, 
boost::wave::language_support language) 
:   first(first), last(last), 
filename(pos.get_file()), line(pos.get_line()), column(pos.get_column()),
at_eof(false), language(language)
{
if (!boost::wave::need_c99(language)) {
for (int j = 0; 0 != init_data_cpp[j].tokenid; ++j) {
xlexer.register_regex(init_data_cpp[j].tokenregex, 
init_data_cpp[j].tokenid, init_data_cpp[j].tokencb);
}
}

#if BOOST_WAVE_SUPPORT_CPP0X != 0
if (boost::wave::need_cpp0x(language) || boost::wave::need_cpp2a(language)) {
for (int j = 0; 0 != init_data_cpp0x[j].tokenid; ++j) {
xlexer.register_regex(init_data_cpp0x[j].tokenregex,
init_data_cpp0x[j].tokenid, init_data_cpp[j].tokencb);
}
}
#endif

#if BOOST_WAVE_SUPPORT_CPP2A != 0
if (boost::wave::need_cpp2a(language) || boost::wave::need_cpp2a(language)) {
for (int j = 0; 0 != init_data_cpp2a[j].tokenid; ++j) {
xlexer.register_regex(init_data_cpp2a[j].tokenregex,
init_data_cpp2a[j].tokenid, init_data_cpp[j].tokencb);
}
}
#endif

for (int i = 0; 0 != init_data[i].tokenid; ++i) {
xlexer.register_regex(init_data[i].tokenregex, init_data[i].tokenid, 
init_data[i].tokencb);
}
}

template <typename Iterator, typename Position>
inline boost::wave::cpplexer::lex_token<Position>& 
lexer<Iterator, Position>::get(boost::wave::cpplexer::lex_token<Position>& t)
{
using namespace boost::wave;    

if (at_eof) 
return t = cpplexer::lex_token<Position>();  

std::string tokval;
token_id id = xlexer.next_token(first, last, tokval);
string_type value = tokval.c_str();

if ((token_id)(-1) == id)
id = T_EOF;     

if (T_IDENTIFIER == id) {
if (!boost::wave::need_no_character_validation(language)) {
cpplexer::impl::validate_identifier_name(value, line, -1, filename); 
}
}
else if (T_STRINGLIT == id || T_CHARLIT == id) {
if (!boost::wave::need_no_character_validation(language)) {
cpplexer::impl::validate_literal(value, line, -1, filename); 
}
}
else if (T_EOF == id) {
at_eof = true;
value.clear();
}
else if (T_NEWLINE == id) {
++line;
column = 1;
} else {
column += value.size();
}

#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
cpplexer::lex_token<Position> tok(id, value, Position(filename, line, column));
return t = guards.detect_guard(tok);
#else
return t = cpplexer::lex_token<Position>(id, value, 
Position(filename, line, column));
#endif
}

template <
typename Iterator, 
typename Position = boost::wave::util::file_position_type
>
class xlex_functor 
:   public xlex_input_interface<typename lexer<Iterator, Position>::token_type>
{    
public:

typedef typename lexer<Iterator, Position>::token_type   token_type;

xlex_functor(Iterator const &first, Iterator const &last, 
Position const &pos, boost::wave::language_support language)
:   lexer_(first, last, pos, language)
{}
virtual ~xlex_functor() {}

token_type& get(token_type& t) BOOST_OVERRIDE { return lexer_.get(t); }
void set_position(Position const &pos) BOOST_OVERRIDE { lexer_.set_position(pos); }

#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
bool has_include_guards(std::string& guard_name) const BOOST_OVERRIDE
{ return lexer_.has_include_guards(guard_name); }
#endif    

private:
lexer<Iterator, Position> lexer_;
};

}   


#if BOOST_WAVE_SEPARATE_LEXER_INSTANTIATION != 0
#define BOOST_WAVE_XLEX_NEW_LEXER_INLINE
#else
#define BOOST_WAVE_XLEX_NEW_LEXER_INLINE inline
#endif 


template <typename Iterator, typename Position>
BOOST_WAVE_XLEX_NEW_LEXER_INLINE
lex_input_interface<boost::wave::cpplexer::lex_token<Position> > *
new_lexer_gen<Iterator, Position>::new_lexer(Iterator const &first,
Iterator const &last, Position const &pos, 
wave::language_support language)
{
return new lexer::xlex_functor<Iterator, Position>(
first, last, pos, language);
}

#undef BOOST_WAVE_XLEX_NEW_LEXER_INLINE

}   
}   
}   
}   

#endif 

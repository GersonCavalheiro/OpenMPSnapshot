
#include <string>
#include <iostream>

#include <boost/config.hpp>

#if defined(BOOST_MSVC)
#   pragma warning(disable:4355) 
#endif

#include <boost/typeof/typeof.hpp>

#include <boost/spirit/include/classic_core.hpp>
#include <boost/spirit/include/classic_typeof.hpp>

#include <boost/spirit/include/classic_rule_parser.hpp>


#include BOOST_TYPEOF_INCREMENT_REGISTRATION_GROUP()

using namespace BOOST_SPIRIT_CLASSIC_NS;

namespace 
{
void do_int(int v)       { std::cout << "PUSH(" << v << ')' << std::endl; }
void do_add(char const*, char const*)  { std::cout << "ADD" << std::endl; }
void do_sub(char const*, char const*)  { std::cout << "SUB" << std::endl; }
void do_mul(char const*, char const*)  { std::cout << "MUL" << std::endl; }
void do_div(char const*, char const*)  { std::cout << "DIV" << std::endl; }
void do_neg(char const*, char const*)  { std::cout << "NEG" << std::endl; }
}

#define BOOST_SPIRIT__NAMESPACE -


BOOST_SPIRIT_RULE_PARSER(expression,
(1,(term)),
-,
-,

term 
>> *( ('+' >> term)[ &do_add ]
| ('-' >> term)[ &do_sub ]
)
)

BOOST_SPIRIT_RULE_PARSER(term,
(1,(factor)),
-,
-,

factor
>> *( ('*' >> factor)[ &do_mul ]  
| ('/' >> factor)[ &do_div ]  
)
)

BOOST_SPIRIT_RULE_PARSER(factor,
(1,(expression)),
-,
(1,( ((parser_reference<factor_t>),factor,(*this)) )),

(   int_p[& do_int]           
|   ('(' >> expression >> ')')
|   ('-' >> factor)[&do_neg]
|   ('+' >> factor)
)
) 


BOOST_SPIRIT_RULE_PARSER( calc,
-,
-,
(3,( ((subrule<0>),sr_expression,()),
((subrule<1>),sr_term,()),
((subrule<2>),sr_factor,() )) ),

(
sr_expression = expression(sr_term),
sr_term       = term(sr_factor),
sr_factor     = factor(sr_expression)
)
)

int main()
{
std::cout 
<< "
<< "\t\tA ruleless calculator using rule parsers and subrules...\n"
<< "
<< "Type an expression...or an empty line to quit\n" 
<< std::endl;

std::string str;
while (std::getline(std::cin, str))
{
if (str.empty()) break;

parse_info<> info = parse(str.c_str(), calc, space_p);

if (info.full)
std::cout 
<< "OK." 
<< std::endl;
else
std::cout 
<< "ERROR.\n"
<< "Stopped at: \": " << info.stop << "\".\n"
<< std::endl;
}
return 0;
}


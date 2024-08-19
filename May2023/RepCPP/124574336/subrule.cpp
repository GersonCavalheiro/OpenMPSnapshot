
#include <boost/detail/lightweight_test.hpp>
#include <boost/spirit/include/qi_operator.hpp>
#include <boost/spirit/include/qi_char.hpp>
#include <boost/spirit/include/qi_string.hpp>
#include <boost/spirit/include/qi_numeric.hpp>
#include <boost/spirit/include/qi_auxiliary.hpp>
#include <boost/spirit/include/qi_nonterminal.hpp>
#include <boost/spirit/include/qi_action.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_object.hpp>
#include <boost/spirit/include/phoenix_bind.hpp>
#include <boost/fusion/include/std_pair.hpp>

#include <boost/spirit/repository/include/qi_subrule.hpp>

#include <string>
#include <cstring>
#include <iostream>
#include "test.hpp"

int
main()
{
using spirit_test::test_attr;
using spirit_test::test;

using namespace boost::spirit::ascii;
using namespace boost::spirit::qi::labels;
using boost::spirit::qi::locals;
using boost::spirit::qi::rule;
using boost::spirit::qi::int_;
using boost::spirit::qi::fail;
using boost::spirit::qi::on_error;
using boost::spirit::qi::debug;
using boost::spirit::repository::qi::subrule;

namespace phx = boost::phoenix;


{ 

subrule<99> entry;
subrule<42> a;
subrule<48> b;
subrule<16> c;
rule<char const*> start;

entry.name("entry");
a.name("a");
b.name("b");
c.name("c");
start.name("start");

debug(start);

BOOST_TEST(test("abcabcacb", (
entry = *(a | b | c)
, a = 'a'
, b = 'b'
, c = 'c'
)));

BOOST_TEST(test("xabcabcacb", 'x' >> (
entry = *(a | b | c)
, a = 'a'
, b = 'b'
, c = 'c'
)));

start = (
entry = *(a | b | c)
, a = 'a'
, b = 'b'
, c = 'c'
);
BOOST_TEST(test("abcabcacb", start));

start = (
entry = (a | b) >> (start | b)
, a = 'a'
, b = 'b'
);
BOOST_TEST(test("aaaabababaaabbb", start));
BOOST_TEST(test("aaaabababaaabba", start, false));

start = (
entry = (a | b) >> (entry | b)
, a = 'a'
, b = 'b'
);
BOOST_TEST(test("aaaabababaaabbb", start));
BOOST_TEST(test("aaaabababaaabba", start, false));

#if defined(BOOST_CLANG) && defined(__has_warning)
# pragma clang diagnostic push
# if __has_warning("-Wself-assign-overloaded")
#  pragma clang diagnostic ignored "-Wself-assign-overloaded"
# endif
#endif
a = a;
#if defined(BOOST_CLANG) && defined(__has_warning)
# pragma clang diagnostic pop
#endif
subrule<42> aa(a);
}

{ 

subrule<0> const entry("entry");
subrule<1> const a("a");
subrule<2> const b("b");
subrule<3> const c("c");
rule<char const*, space_type> start("start");

debug(start);

start = (
entry = *(a | b | c)
, a = 'a'
, b = 'b'
, c = 'c'
);
BOOST_TEST(test(" a b c a b c a c b ", start, space));

start = (
entry = (a | b) >> (entry | b)
, a = 'a'
, b = 'b'
);
BOOST_TEST(test(" a a a a b a b a b a a a b b b ", start, space));
BOOST_TEST(test(" a a a a b a b a b a a a b b a ", start, space, false));

#if defined(BOOST_CLANG) && defined(__has_warning)
# pragma clang diagnostic push
# if __has_warning("-Wself-assign-overloaded")
#  pragma clang diagnostic ignored "-Wself-assign-overloaded"
# endif
#endif
a = a;
#if defined(BOOST_CLANG) && defined(__has_warning)
# pragma clang diagnostic pop
#endif
subrule<1> aa(a);
}

{ 

char ch;
rule<char const*, char()> a;
subrule<0, char()> entry;

a = (entry = alpha[_val = _1])[_val = _1];
BOOST_TEST(test("x", a[phx::ref(ch) = _1]));
BOOST_TEST(ch == 'x');

a %= (entry = alpha[_val = _1]);
BOOST_TEST(test_attr("z", a, ch)); 
BOOST_TEST(ch == 'z');
}

{ 

char ch;
rule<char const*, char()> a;
subrule<0, char()> entry;

a = (entry %= alpha)[_val = _1];
BOOST_TEST(test("x", a[phx::ref(ch) = _1]));
BOOST_TEST(ch == 'x');

a %= (entry %= alpha);
BOOST_TEST(test_attr("z", a, ch)); 
BOOST_TEST(ch == 'z');
}

{ 

std::string s;
rule<char const*, std::string()> r;
subrule<0, std::string()> entry;

r %= (entry %= char_ >> *(',' >> char_));
BOOST_TEST(test("a,b,c,d,e,f", r[phx::ref(s) = _1]));
BOOST_TEST(s == "abcdef");

BOOST_TEST(test("abcdef", (
entry %= char_ >> char_ >> char_ >> char_ >> char_ >> char_
)[phx::ref(s) = _1]));
BOOST_TEST(s == "abcdef");
}

{ 

std::string s;
subrule<0, char()> sr;
BOOST_TEST(test_attr("abcdef", +(sr = alpha[_val += _1]), s));
BOOST_TEST(s == "abcdef");
}

{ 

char ch;
rule<char const*, char()> r;
subrule<0, char()> a;
subrule<1, char()> b;
r %= (
a %= b
, b %= alpha
);

BOOST_TEST(test("x", r[phx::ref(ch) = _1]));
BOOST_TEST(ch == 'x');

BOOST_TEST(test_attr("z", r, ch)); 
BOOST_TEST(ch == 'z');
}

{ 

char ch;

rule<char const*, char(int)> a;
subrule<1, char(int)> sr1;
a %= (
sr1 = alpha[_val = _1 + _r1]
)(_r1);
BOOST_TEST(test("x", a(phx::val(1))[phx::ref(ch) = _1]));
BOOST_TEST(ch == 'x' + 1);

subrule<0, char()> sr0;
a %= (
sr0 %= sr1(1)
, sr1 = alpha[_val = _1 + _r1]
);

rule<char const*, char()> b;
b %= (
sr1 = alpha[_val = _1 + _r1]
)(1);
BOOST_TEST(test_attr("b", b, ch));
BOOST_TEST(ch == 'b' + 1);

subrule<2, char(int, int)> sr2;
BOOST_TEST(test_attr("a", (
sr2 = alpha[_val = _1 + _r1 + _r2]
)(1, 2), ch));
BOOST_TEST(ch == 'a' + 1 + 2);

BOOST_TEST(test_attr("ba", (
sr2 = alpha[_val = _1 + _r1 + _r2] >> sr1(3)[_val -= _1]
, sr1 = alpha[_val = _1 + _r1]
)(1, 2), ch));
BOOST_TEST(ch == ('b' + 1 + 2) - ('a' + 3));
}

{ 

char ch;
subrule<0, void(char&)> sr; 
BOOST_TEST(test("x", (sr = alpha[_r1 = _1])(phx::ref(ch))));
BOOST_TEST(ch == 'x');

rule<char const*, void(char&)> a; 
a = (sr = alpha[_r1 = _1])(_r1);
BOOST_TEST(test("y", a(phx::ref(ch))));
BOOST_TEST(ch == 'y');
}

{ 

rule<char const*> r;
subrule<0, locals<char> > a; 
r = (
a = alpha[_a = _1] >> char_(_a)
);
BOOST_TEST(test("aa", r));
BOOST_TEST(!test("ax", r));
}

{ 

rule<char const*, void(int)> a;
subrule<0, void(int), locals<char> > sr; 
a = (
sr = alpha[_a = _1 + _r1] >> char_(_a)
)(_r1);
BOOST_TEST(test("ab", a(phx::val(1))));
BOOST_TEST(test("xy", a(phx::val(1))));
BOOST_TEST(!test("ax", a(phx::val(1))));
}

{ 

std::pair<int, char> attr;
subrule<0, void()> sr;
BOOST_TEST(test_attr("123ax", int_ >> char_ >> (sr = char_), attr));
BOOST_TEST(attr.first == 123);
BOOST_TEST(attr.second == 'a');
}

{ 

rule<char const*> r;
subrule<0, char(int)> sr;

r = (
sr = char_(_r1)[_val = _1]
)(42);
}

{ 
subrule<0, int()> sra;
subrule<1, int()> srb;
int attr;

BOOST_TEST(test_attr("123", (sra %= int_), attr));
BOOST_TEST(attr == 123);

BOOST_TEST(test_attr("123", (srb %= sra, sra %= int_), attr));
BOOST_TEST(attr == 123);

BOOST_TEST(test_attr("123", (srb = sra, sra %= int_), attr));
BOOST_TEST(attr == 123);
}

{ 

subrule<0, std::string()> text;
std::string attr;
BOOST_TEST(test_attr("x", (
text %= +(!char_(')') >> !char_('>') >> char_)
), attr));
BOOST_TEST(attr == "x");
}


return boost::report_errors();
}


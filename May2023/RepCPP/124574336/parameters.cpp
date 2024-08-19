

#include <string>
#include <iostream>
#include <cassert>

#if defined(_MSC_VER) 
#pragma warning(disable: 4244)
#pragma warning(disable: 4355)
#endif 

#include <boost/spirit/include/classic_core.hpp>
#include <boost/spirit/include/classic_symbols.hpp>
#include <boost/spirit/include/classic_closure.hpp>

#include <boost/spirit/include/phoenix1_tuples.hpp>
#include <boost/spirit/include/phoenix1_tuple_helpers.hpp>
#include <boost/spirit/include/phoenix1_primitives.hpp>

using namespace BOOST_SPIRIT_CLASSIC_NS;
using namespace phoenix;
using namespace std;

class declaration_type
{
public:
enum vartype {
vartype_unknown = 0,        
vartype_int = 1,            
vartype_real = 2            
};

declaration_type() : type(vartype_unknown)
{
}
template <typename ItT>
declaration_type(ItT const &first, ItT const &last)
{
init(string(first, last-first-1));
}
declaration_type(declaration_type const &type_) : type(type_.type)
{
}
declaration_type(string const &type_) : type(vartype_unknown)
{
init(type_);
}

operator vartype const &() const { return type; }
operator string ()
{
switch(type) {
default:
case vartype_unknown:   break;
case vartype_int:       return string("int");
case vartype_real:      return string("real");
}
return string ("unknown");
}

void swap(declaration_type &s) { std::swap(type, s.type); }

protected:
void init (string const &type_)
{
if (type_ == "int")
type = vartype_int;
else if (type_ == "real")
type = vartype_real;
else
type = vartype_unknown;
}

private:
vartype type;
};

struct var_decl_closure : BOOST_SPIRIT_CLASSIC_NS::closure<var_decl_closure, declaration_type>
{
member1 val;
};

template <typename T, typename InitT>
class symbols_with_data
{
public:
typedef
symbol_inserter<T, BOOST_SPIRIT_CLASSIC_NS::impl::tst<T, char> >
symbol_inserter_t;

symbols_with_data(symbol_inserter_t const &add_, InitT const &data_) :
add(add_), data(as_actor<InitT>::convert(data_))
{
}

template <typename IteratorT>
symbol_inserter_t const &
operator()(IteratorT const &first_, IteratorT const &last) const
{
IteratorT first = first_;
return add(first, last, data());
}

private:
symbol_inserter_t const &add;
typename as_actor<InitT>::type data;
};

template <typename T, typename CharT, typename InitT>
inline
symbols_with_data<T, InitT>
symbols_gen(symbol_inserter<T, BOOST_SPIRIT_CLASSIC_NS::impl::tst<T, CharT> > const &add_,
InitT const &data_)
{
return symbols_with_data<T, InitT>(add_, data_);
}


struct var_decl_list :
public grammar<var_decl_list, var_decl_closure::context_t>
{
template <typename ScannerT>
struct definition
{
definition(var_decl_list const &self)
{
decl = type[self.val = arg1] >> +space_p >> list(self.val);

list = ident(list.val) >> *(',' >> ident(list.val));

ident = (*alnum_p)[symbols_gen(symtab.add, ident.val)];

type =
str_p("int")[type.val = construct_<string>(arg1, arg2)]
|   str_p("real")[type.val = construct_<string>(arg1, arg2)]
;

BOOST_SPIRIT_DEBUG_RULE(decl);
BOOST_SPIRIT_DEBUG_RULE(list);
BOOST_SPIRIT_DEBUG_RULE(ident);
BOOST_SPIRIT_DEBUG_RULE(type);
}

rule<ScannerT> const&
start() const { return decl; }

private:
typedef rule<ScannerT, var_decl_closure::context_t> rule_t;
rule_t type;
rule_t list;
rule_t ident;
symbols<declaration_type> symtab;

rule<ScannerT> decl;        
};
};

int main()
{
var_decl_list decl;
declaration_type type;
char const *pbegin = "int  var1";

if (parse (pbegin, decl[assign(type)]).full) {
cout << endl
<< "Parsed variable declarations successfully!" << endl
<< "Detected type: " << declaration_type::vartype(type)
<< " (" << string(type) << ")"
<< endl;
} else {
cout << endl
<< "Parsing the input stream failed!"
<< endl;
}
return 0;
}


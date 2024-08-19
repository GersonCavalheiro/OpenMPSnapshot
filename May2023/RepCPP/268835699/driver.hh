#ifndef __DRIVER_HH__
#define __DRIVER_HH__

#include "ast/ast.hh"
#include "parser.hh"

# define YY_DECL \
yy::parser::symbol_type yylex (Driver& drv)
YY_DECL;

class Driver {
public:
std::string file="";
bool trace_parsing=false;
bool trace_scanning=false;
yy::location location;
pragma_ast_t ast;
Driver();
~Driver();
int parse(const std::string& f);
void scanBegin();
void scanEnd();
};
#endif

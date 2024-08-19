#ifndef __PRAGMA_PARSER__
#define __PRAGMA_PARSER__

#include <string>
#include <utility>

#include "ast/ast.hh"
#include "parser.hh"
#include "driver.hh"

pragma_ast_t pragma_parser(std::string filename,bool tracep=false,bool traces=false);

#endif

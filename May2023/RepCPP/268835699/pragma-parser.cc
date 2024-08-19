#include "pragma-parser.hh"

pragma_ast_t pragma_parser(std::string filename,bool tracep,bool traces) {
Driver drv;
drv.trace_parsing=tracep;
drv.trace_scanning=traces;
if(!drv.parse(filename)) 
return std::move(drv.ast);
else
return std::move(pragma_ast_t());
}

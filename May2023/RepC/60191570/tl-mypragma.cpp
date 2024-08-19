#include "tl-mypragma.hpp"
#include "tl-pragmasupport.hpp"
#include "tl-langconstruct.hpp"
#include <vector>
#include <stack>
#include <sstream>
#include <fstream>
namespace TL
{
class MyPragmaPhase : public PragmaCustomCompilerPhase
{
public:
MyPragmaPhase()
: PragmaCustomCompilerPhase("mypragma")
{
register_construct("test");
on_directive_post["test"].connect(functor(&MyPragmaPhase::construct_post, *this));
}
virtual void run(DTO& dto)
{
std::cerr << " --> RUNNING MYPRAGMA <-- " << std::endl;
PragmaCustomCompilerPhase::run(dto);
}
void construct_post(PragmaCustomConstruct construct)
{
std::cerr << " --> RUNNING CONSTRUCT POST <-- " << std::endl;
std::cerr << "Getting enclosing function def" << std::endl;
FunctionDefinition function_def = construct.get_enclosing_function();
Statement fun_body = function_def.get_function_body();
std::cerr << "BODY -->" << fun_body << "<--" << std::endl;
std::cerr << "BODY is compound statement? " << fun_body.is_compound_statement() << std::endl;
}
};
}
EXPORT_PHASE(TL::MyPragmaPhase);

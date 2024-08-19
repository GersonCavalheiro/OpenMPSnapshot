
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include "ast_vertex.h"

#if 0
void 
unparser_PARALLEL_REGION_OMP_original(void* argv)  
{
DEFINE_UNPARSER;
MAKE_INDENT;
unparser->ofs << "#pragma parallel\n";
MAKE_INDENT;
unparser->ofs << "{\n";
INC_INDENT;
UNPARSE_FIRST_TEST(VT_SEQ_DECLARATIONS);
UNPARSE_SECOND_TEST(VT_S); 
DEC_INDENT;
MAKE_INDENT;
unparser->ofs << "}\n";
}

void 
unparser_PARALLEL_FOR_REGION_OMP_original(void* argv)  
{
DEFINE_UNPARSER;
MAKE_INDENT;
unparser->ofs << "#pragma parallel for\n";
MAKE_INDENT;
unparser->ofs << "for(";   UNPARSE_FIRST_TEST(VT_FOR_LOOP_HEADER);   unparser->ofs << ")\n"; 
MAKE_INDENT; unparser->ofs << "{\n";
INC_INDENT;
UNPARSE_SECOND_TEST(VT_SEQ_DECLARATIONS); 
UNPARSE_THIRD_TEST(VT_S);
DEC_INDENT;
MAKE_INDENT;
unparser->ofs << "}\n";
}	

void 
unparser_SEQ_DECLARATIONS_original(void* argv)  
{
DEFINE_UNPARSER;
UNPARSE_ALL; 
}

void 
unparser_S_original(void* argv) 
{ 
DEFINE_UNPARSER;
MAKE_INDENT; unparser->ofs << "
for(list<ast_vertex*>::const_iterator it=unparser->v->children.begin();it!=unparser->v->children.end();it++) { 
assert(*it); 
unparser->v = *it; 
MAKE_INDENT;
(*it)->unparse(argv); 
unparser->v=v; 
if( (*it)->type!=VT_SPL_cond_if  &&  (*it)->type!=VT_SPL_cond_while ) unparser->ofs << ";";
unparser->ofs << endl;
}
MAKE_INDENT; unparser->ofs << "
}

#endif



#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include "symbol_table.h"
#include "ast_vertex.h"
#include "AST.h"

using namespace std;

string  STACKc_push   =  "STACKc.push";
string  STACKi_push   =  "STACKi.push";
string  STACKf_push   =  "STACKf.push";
string  STACKc_pop    =  "STACKc.pop";
string  STACKi_pop    =  "STACKi.pop";
string  STACKf_pop    =  "STACKf.pop";
string  STACKc_top    =  "STACKc.top";
string  STACKi_top    =  "STACKi.top";
string  STACKf_top    =  "STACKf.top";
string  STACKc_empty  =  "STACKc.empty";
const string BEFORE_REVERSE_MODE_HOOK = "before_reverse_mode";

string  EXCLUSIVE_READ_FAILURE  = "exclusive read failure";

void unparser_spl_code(void* argv) { DEFINE_UNPARSER; UNPARSE_ALL; } 

void 
unparser_global_declarations(void* argv) { 
DEFINE_UNPARSER; 
if(!option_suppress_global_region) { 
PAPI_SHARED_VARIABLES
UNPARSE_ALL; 
} 
}

void unparser_spl_STACKf(void* argv){ DEFINE_UNPARSER; unparser->ofs << v->get_str() << "STACKf"; }  
void unparser_spl_STACKi(void* argv){ DEFINE_UNPARSER; unparser->ofs << v->get_str() << "STACKi"; }
void unparser_spl_STACKc(void* argv){ DEFINE_UNPARSER; unparser->ofs << v->get_str() << "STACKc"; }

void unparser_spl_STACKfpush(void* argv) { DEFINE_UNPARSER; unparser->ofs << v->get_str() << STACKf_push<<"("; UNPARSE_FIRST; unparser->ofs << ")"; }
void unparser_spl_STACKipush(void* argv) { DEFINE_UNPARSER; unparser->ofs << v->get_str() << STACKi_push<<"("; UNPARSE_FIRST; unparser->ofs << ")"; }
void unparser_spl_STACKcpush(void* argv) { DEFINE_UNPARSER; unparser->ofs << v->get_str() << STACKc_push<<"("; UNPARSE_FIRST; unparser->ofs << ")"; }

void unparser_spl_STACKcempty(void* argv) { DEFINE_UNPARSER; unparser->ofs << v->get_str() << STACKc_empty<<"()"; }

void unparser_spl_STACKftop(void* argv) { DEFINE_UNPARSER; unparser->ofs << v->get_str() << STACKf_top<<"()"; }
void unparser_spl_STACKctop(void* argv) { DEFINE_UNPARSER; unparser->ofs << v->get_str() << STACKc_top<<"()"; }
void unparser_spl_STACKitop(void* argv) { DEFINE_UNPARSER; unparser->ofs << v->get_str() << STACKi_top<<"()"; }


void unparser_spl_STACKfpop(void* argv)  { DEFINE_UNPARSER; unparser->ofs << v->get_str() << STACKf_pop<<"()"; }
void unparser_spl_STACKipop(void* argv)  { DEFINE_UNPARSER; unparser->ofs << v->get_str() << STACKi_pop<<"()"; }
void unparser_spl_STACKcpop(void* argv)  { DEFINE_UNPARSER; unparser->ofs << v->get_str() << STACKc_pop<<"()"; }

void 
unparser_spl_int_assign(void* argv)  
{
DEFINE_UNPARSER;
UNPARSE_FIRST_TEST(VT_SPL_identifier);
unparser->ofs << "="; 
UNPARSE_SECOND;
}

void 
unparser_spl_float_assign(void* argv) 
{
DEFINE_UNPARSER;
UNPARSE_FIRST_TEST(VT_SPL_identifier);
unparser->ofs << "="; 
UNPARSE_SECOND;
}

void 
unparser_spl_float_plus_assign(void* argv)  
{
DEFINE_UNPARSER;
UNPARSE_FIRST_TEST(VT_SPL_identifier);
unparser->ofs << "+="; 
UNPARSE_SECOND;
}


void 
unparser_spl_cond_if(void* argv) 
{
DEFINE_UNPARSER;
unparser->ofs << "if("; UNPARSE_FIRST; unparser->ofs << ") {\n"; 
INC_INDENT;
UNPARSE_SECOND;
DEC_INDENT;
MAKE_INDENT; unparser->ofs << "}"; 
}

void 
unparser_spl_cond_while(void* argv) 
{
DEFINE_UNPARSER;
unparser->ofs << "while("; UNPARSE_FIRST; unparser->ofs << ") {\n"; 
INC_INDENT;
UNPARSE_SECOND;
DEC_INDENT;
MAKE_INDENT; unparser->ofs << "}"; 
}

void unparser_spl_identifier(void* argv) 
{ 
DEFINE_UNPARSER; 
unparser->ofs << v->get_str(); 
if(unparser->v->get_children().size()>0) {
UNPARSE_FIRST;
}
}

void 
unparser_spl_array_index(void* argv)
{ 
DEFINE_UNPARSER; 
unparser->ofs << "[";
UNPARSE_FIRST;
unparser->ofs << "]";
}

void unparser_spl_expr_negative(void* argv) { DEFINE_UNPARSER; unparser->ofs << "-"; UNPARSE_FIRST; }
void unparser_spl_expr_in_brackets(void* argv) { DEFINE_UNPARSER; unparser->ofs << "("; UNPARSE_FIRST; unparser->ofs << ")"; }

void unparser_spl_float_const(void* argv){ DEFINE_UNPARSER; unparser->ofs << v->get_str(); }
void unparser_spl_float_sin(void* argv)
{
DEFINE_UNPARSER;
unparser->ofs << "sin("; 
UNPARSE_FIRST;
unparser->ofs << ")"; 
}

void unparser_spl_float_cos(void* argv)
{
DEFINE_UNPARSER;
unparser->ofs << "cos("; 
UNPARSE_FIRST;
unparser->ofs << ")"; 
}

void unparser_spl_float_exp(void* argv)
{
DEFINE_UNPARSER;
unparser->ofs << "exp("; 
UNPARSE_FIRST;
unparser->ofs << ")"; 
}

void unparser_spl_float_mul(void* argv) { DEFINE_UNPARSER; UNPARSE_FIRST; unparser->ofs << "*"; UNPARSE_SECOND; }
void unparser_spl_float_div(void* argv) { DEFINE_UNPARSER; UNPARSE_FIRST; unparser->ofs << "/"; UNPARSE_SECOND; }
void unparser_spl_float_add(void* argv) { DEFINE_UNPARSER; UNPARSE_FIRST; unparser->ofs << "+"; UNPARSE_SECOND; }
void unparser_spl_float_sub(void* argv) { DEFINE_UNPARSER; UNPARSE_FIRST; unparser->ofs << "-"; UNPARSE_SECOND; }

void unparser_spl_int_add(void* argv){ DEFINE_UNPARSER; UNPARSE_FIRST; unparser->ofs << "+"; UNPARSE_SECOND; }
void unparser_spl_int_sub(void* argv){ DEFINE_UNPARSER; UNPARSE_FIRST; unparser->ofs << "-"; UNPARSE_SECOND; }
void unparser_spl_int_mul(void* argv){ DEFINE_UNPARSER; UNPARSE_FIRST; unparser->ofs << "*"; UNPARSE_SECOND; }
void unparser_spl_int_div(void* argv){ DEFINE_UNPARSER; UNPARSE_FIRST; unparser->ofs << "/"; UNPARSE_SECOND; }
void unparser_spl_int_mudolo(void* argv){ DEFINE_UNPARSER; UNPARSE_FIRST; unparser->ofs << "%"; UNPARSE_SECOND; }
void unparser_spl_int_const(void* argv){ DEFINE_UNPARSER; unparser->ofs << v->get_str(); }

void unparser_spl_boolean_or(void* argv) { DEFINE_UNPARSER; UNPARSE_FIRST; unparser->ofs << " || "; UNPARSE_SECOND; }
void unparser_spl_boolean_and(void* argv){ DEFINE_UNPARSER; UNPARSE_FIRST; unparser->ofs << " && "; UNPARSE_SECOND; }
void unparser_spl_eq(void* argv){ DEFINE_UNPARSER; UNPARSE_FIRST; unparser->ofs << "=="; UNPARSE_SECOND; }
void unparser_spl_neq(void* argv){ DEFINE_UNPARSER; UNPARSE_FIRST; unparser->ofs << "!="; UNPARSE_SECOND; }
void unparser_spl_leq(void* argv){ DEFINE_UNPARSER; UNPARSE_FIRST; unparser->ofs << "<="; UNPARSE_SECOND; }
void unparser_spl_geq(void* argv){ DEFINE_UNPARSER; UNPARSE_FIRST; unparser->ofs << ">="; UNPARSE_SECOND; }
void unparser_spl_lower(void* argv){ DEFINE_UNPARSER; UNPARSE_FIRST; unparser->ofs << "<"; UNPARSE_SECOND; }
void unparser_spl_greater(void* argv){ DEFINE_UNPARSER; UNPARSE_FIRST; unparser->ofs << ">"; UNPARSE_SECOND; }
void unparser_spl_not(void* argv){ DEFINE_UNPARSER; unparser->ofs << "!"; UNPARSE_FIRST;}


void 
unparser_PARALLEL_REGION(void* argv)  
{
DEFINE_UNPARSER;
OMP_GET_WTIME_START ;
MAKE_INDENT; unparser->ofs << "#pragma omp parallel\n";
MAKE_INDENT; unparser->ofs << "{\n";
INC_INDENT;
PAPI_HEADER
UNPARSE_FIRST_TEST(VT_SEQ_DECLARATIONS);
UNPARSE_SECOND_TEST(VT_S); 
PAPI_FOOTER
DEC_INDENT;
MAKE_INDENT; unparser->ofs << "}";
if(!unparser->suppress_linefeed) unparser->ofs << "\n";
OMP_GET_WTIME_END;
PAPI_PRINT_MFLOPS
}

void 
unparser_PARALLEL_FOR_REGION(void* argv)  
{
DEFINE_UNPARSER;
OMP_GET_WTIME_START;
if(!option_papi) {
MAKE_INDENT; unparser->ofs << "#pragma omp parallel for\n";
}
else {
MAKE_INDENT; unparser->ofs << "#pragma omp parallel\n";
MAKE_INDENT; unparser->ofs << "{\n";
INC_INDENT;
PAPI_HEADER
MAKE_INDENT; unparser->ofs << "#pragma omp for nowait\n";
}
MAKE_INDENT;
unparser->ofs << "for(";   UNPARSE_FIRST_TEST(VT_FOR_LOOP_HEADER);   unparser->ofs << ")\n"; 
MAKE_INDENT; unparser->ofs << "{\n";
INC_INDENT;
UNPARSE_SECOND_TEST(VT_SEQ_DECLARATIONS); 
UNPARSE_THIRD_TEST(VT_S);
DEC_INDENT;
if(!option_papi) {
MAKE_INDENT; unparser->ofs << "}";
if(!unparser->suppress_linefeed) unparser->ofs << "\n";
OMP_GET_WTIME_END;
}
else {
MAKE_INDENT; unparser->ofs << "}\n";
PAPI_FOOTER
DEC_INDENT;
MAKE_INDENT; unparser->ofs << "}\n";
OMP_GET_WTIME_END;
PAPI_PRINT_MFLOPS
}
}

void 
unparser_SEQ_DECLARATIONS(void* argv)  { DEFINE_UNPARSER; UNPARSE_ALL; unparser->ofs << "\n"; }

void 
unparser_INT_POINTER_DECLARATION(void* argv)
{
DEFINE_UNPARSER;
MAKE_INDENT; unparser->ofs << "int* " ;
UNPARSE_FIRST_TEST(VT_SPL_identifier);
unparser->ofs << ";";
if(!unparser->suppress_linefeed) unparser->ofs << "\n";
}

void 
unparser_FLOAT_POINTER_DECLARATION(void* argv)
{
DEFINE_UNPARSER;
MAKE_INDENT; unparser->ofs << unparser->float_type << "* " ;
UNPARSE_FIRST_TEST(VT_SPL_identifier);
unparser->ofs << ";";
if(!unparser->suppress_linefeed) unparser->ofs << "\n";
}

void 
unparser_UINT_DECLARATION(void* argv)  
{
DEFINE_UNPARSER;
if(FIRST->get_type()==VT_SPL_identifier) { MAKE_INDENT; unparser->ofs << "unsigned int " << FIRST->get_str() ; }
else if(FIRST->get_type()==VT_SPL_int_assign) { 
MAKE_INDENT; 
unparser->ofs << "unsigned int ";
UNPARSE_FIRST; 
}
unparser->ofs << ";";
if(!unparser->suppress_linefeed) unparser->ofs << "\n";
}

void 
unparser_INT_DECLARATION(void* argv)  
{
DEFINE_UNPARSER;
if(FIRST->get_type()==VT_SPL_identifier) { MAKE_INDENT; unparser->ofs << "int " << FIRST->get_str() ; }
else if(FIRST->get_type()==VT_SPL_int_assign) { 
MAKE_INDENT; 
unparser->ofs << "int ";
UNPARSE_FIRST; 
}
unparser->ofs << ";";
if(!unparser->suppress_linefeed) unparser->ofs << "\n";
}

void unparser_FLOAT_DECLARATION(void* argv) 
{
DEFINE_UNPARSER;
if(FIRST->get_type()==VT_SPL_identifier) { 
MAKE_INDENT; 
unparser->ofs << unparser->float_type << " " ; 
if(option_mpfr) {
ast_vertex* assign_lhs=*(*FIRST->get_children().begin())->get_children().begin();
ASSUME_TYPE(assign_lhs,VT_SPL_identifier);
unparser->ofs << assign_lhs->get_str() << "; ";
unparser->ofs << "mpfr_init2("<<assign_lhs->get_str()<<","<<MPFR_FLOAT_PRECISION<<");";
}
else {
unparser->ofs << FIRST->get_str() ; 
}
}
else if(FIRST->get_type()==VT_SPL_float_assign) { 
MAKE_INDENT; 
unparser->ofs << unparser->float_type << " ";
if(option_mpfr) {
ast_vertex* assign_lhs=*(*FIRST->get_children().begin())->get_children().begin();
ast_vertex* assign_rhs=*++(*FIRST->get_children().begin())->get_children().begin();
ASSUME_TYPE(assign_lhs,VT_SPL_identifier);
unparser->ofs << assign_lhs->get_str() << "; ";
unparser->ofs << "mpfr_init2("<<assign_lhs->get_str()<<","<<MPFR_FLOAT_PRECISION<<");";
unparser->ofs << "mpfr_set_d("<<assign_rhs->get_str()<<","<<MPFR_FLOAT_PRECISION<<")";
}
else { UNPARSE_FIRST; }
}
unparser->ofs << ";";
if(!unparser->suppress_linefeed) unparser->ofs << "\n";
}

void 
unparser_STACKc_DECLARATION(void* argv) { 
DEFINE_UNPARSER; 
unparser->ofs << "Stackc " << v->get_str() << "STACKc;"; 
if(!unparser->suppress_linefeed) unparser->ofs << "\n";
}

void 
unparser_STACKi_DECLARATION(void* argv) { 
DEFINE_UNPARSER; 
unparser->ofs << "Stacki " << v->get_str() << "STACKi;"; 
if(!unparser->suppress_linefeed) unparser->ofs << "\n";
}

void 
unparser_STACKf_DECLARATION(void* argv) { 
DEFINE_UNPARSER; 
if(option_long_double) unparser->ofs << "ld"; 
unparser->ofs << "Stackf " << v->get_str() << "STACKf;"; 
if(!unparser->suppress_linefeed) unparser->ofs << "\n";
}

void 
unparser_S(void* argv) 
{ 
DEFINE_UNPARSER;
for(list<ast_vertex*>::const_iterator it=unparser->v->get_children().begin();it!=unparser->v->get_children().end();it++) { 
assert(*it); 
unparser->v = *it; 
MAKE_INDENT;
(*it)->unparse(argv); 
unparser->v=v; 
(*it)->print_statement_tail(unparser->ofs);
if(!unparser->suppress_linefeed) unparser->ofs << "\n";
}
}


void unparser_omp_ATOMIC(void* argv)   
{
DEFINE_UNPARSER;
ASSUME_TYPE(v, VT_ATOMIC);
ASSUME_TYPE(FIRST, VT_SPL_float_plus_assign);
unparser->ofs << "#pragma omp atomic\n";
MAKE_INDENT; 
UNPARSE_FIRST; 
}

void unparser_spl_omp_threadprivate(void* argv) 
{
DEFINE_UNPARSER;
unparser->ofs << "#pragma omp threadprivate ("; UNPARSE_FIRST; unparser->ofs << ")";
if(!unparser->suppress_linefeed) unparser->ofs << "\n";
}

void unparser_spl_omp_for(void* argv) 
{ 
DEFINE_UNPARSER; 
unparser->ofs << "#pragma omp for\n"; 
MAKE_INDENT; UNPARSE_FIRST; unparser->ofs << ";"; unparser->ofs << "\n";
MAKE_INDENT; UNPARSE_SECOND; 
}

void unparser_FOR_LOOP_HEADER(void* argv){ DEFINE_UNPARSER; UNPARSE_FIRST; unparser->ofs << ";"; UNPARSE_SECOND; unparser->ofs << ";"; UNPARSE_THIRD; }
void unparser_FOR_LOOP_HEADER_INIT(void* argv){ DEFINE_UNPARSER; UNPARSE_FIRST; }
void unparser_FOR_LOOP_HEADER_TEST(void* argv){ DEFINE_UNPARSER; UNPARSE_FIRST; }
void unparser_FOR_LOOP_HEADER_UPDATE(void* argv){ DEFINE_UNPARSER; UNPARSE_FIRST; unparser->ofs << "++";}

void 
unparser_FOR_LOOP_HEADER_VAR_DEF(void* argv)
{
DEFINE_UNPARSER;
MAKE_INDENT; unparser->ofs << "int ";
UNPARSE_FIRST_TEST(VT_SPL_int_assign);
}


void unparser_clauses(void* argv){}


void unparser_list_of_vars(void* argv) 
{
DEFINE_UNPARSER; 
list<ast_vertex*>::const_iterator begin_iterator=unparser->v->get_children().begin();
for(list<ast_vertex*>::const_iterator it=begin_iterator;it!=unparser->v->get_children().end();it++) 
{ 
assert(*it); 
unparser->v = *it; 
if(it!=begin_iterator) unparser->ofs << ", ";
(*it)->unparse(argv); 
unparser->v=v; 
}
}

void unparser_cfg_entry(void* argv){}
void unparser_cfg_exit(void* argv){}

void unparser_dummy(void* argv)
{ 
DEFINE_UNPARSER; 
ast_vertex* list_of_vars=FIRST;
unparser->ofs << "dummy(\"\", ";
list<ast_vertex*>::const_iterator begin_iterator=list_of_vars->get_children().begin();
for(list<ast_vertex*>::const_iterator it=begin_iterator;it!=list_of_vars->get_children().end();it++) 
{ 
assert(*it); 
unparser->v = *it; 
if(it!=begin_iterator) unparser->ofs << ", ";
unparser->ofs << "(void*)";
assert( sym_tbl.is_float(unparser->v->get_str()) );
if ( !sym_tbl.is_float_pointer(unparser->v->get_str()) ){ unparser->ofs << "&"; }
(*it)->unparse(argv); 
unparser->v=v; 
}
unparser->ofs <<      ");";
}

void unparser_OMP_RUNTIME_ROUTINE(void* argv){ DEFINE_UNPARSER; unparser->ofs << v->get_str(); }
void unparser_barrier(void* argv) { DEFINE_UNPARSER; unparser->ofs << "#pragma omp barrier"; }

void unparser_master(void* argv) 
{ 
DEFINE_UNPARSER; 
ASSUME_TYPE(v, VT_MASTER);
unparser->ofs << "#pragma omp master\n"; 
UNPARSE_FIRST_AS_BLOCK_TEST(VT_S);
}


void unparser_ad_exclusive_read_failure(void* argv)
{
DEFINE_UNPARSER;
ASSUME_TYPE(v, VT_EXCLUSIVE_READ_FAILURE);
switch(FIRST->get_type()) {
case VT_SPL_float_plus_assign:
case VT_SPL_float_assign:
break;
default:
assert(0);
}
unparser->ofs << "#pragma ad " << EXCLUSIVE_READ_FAILURE << "\n";
MAKE_INDENT; UNPARSE_FIRST; 
}



void unparser_critical(void* argv) 
{ 
DEFINE_UNPARSER; 
ASSUME_TYPE(v, VT_CRITICAL);
unparser->ofs << "#pragma omp critical\n"; 
UNPARSE_FIRST_AS_BLOCK_TEST(VT_S);
}


void unparser_assert(void* argv) { DEFINE_UNPARSER; ASSUME_TYPE(v, VT_ASSERT); unparser->ofs << "assert("; UNPARSE_FIRST; unparser->ofs << ")"; }
void unparser_before_reverse_mode_hook(void* argv) { DEFINE_UNPARSER; ASSUME_TYPE(v, VT_BEFORE_REVERSE_MODE_HOOK); unparser->ofs << BEFORE_REVERSE_MODE_HOOK << "()"; }



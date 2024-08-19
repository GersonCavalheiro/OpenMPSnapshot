

#include <cassert>
#include "symbol_table.h"
#include "ast_vertex.h"
#include "AST.h"
#include "exclusive-read.h"

static bool inside_critical_construct=false;

ast_vertex* 
ast_vertex::rho()
{
switch(this->type) {
case VT_SEQ_DECLARATIONS:
assert(0);
case VT_S:
return rho_seq_of_stmts();
case VT_SPL_cond_if:
case VT_SPL_cond_while:
return rho_conditional();
break;
case VT_SPL_STACKfpush:
case VT_SPL_STACKipush:
case VT_SPL_STACKcpush:
case VT_SPL_STACKfpop:
case VT_SPL_STACKipop:
case VT_SPL_STACKcpop:
return rho_stack();
break;
case VT_SPL_int_assign:
case VT_SPL_float_assign:
case VT_SPL_float_plus_assign:
return rho_assign();
break;
case VT_OMP_FOR: 
return rho_omp_for(); break;
case VT_BARRIER:  
return rho_barrier();  break;
case VT_ATOMIC:  
return rho_atomic(); break;
case VT_EXCLUSIVE_READ_FAILURE:  
return rho_exclusive_read(); break;
case VT_DUMMY: 
return rho_dummy(); break;
case VT_MASTER:  
return rho_master(); break;
case VT_CRITICAL:  
return rho_critical(); break;
case VT_ASSERT: 
return rho_assert(); break;
default:
UNKNOWN_TYPE(this);
}
}

ast_vertex*
ast_vertex::rho_omp_for()
{
ast_vertex* v=this;
ast_vertex* loop_init=FIRST;
ast_vertex* loop=SECOND;
ast_vertex* S=NULL;
ast_vertex* rho_omp_for=NULL;
ast_vertex* rho_loop=NULL;
ast_vertex* rho_loop_init=NULL;

ASSUME_TYPE(FIRST, VT_SPL_int_assign);
ASSUME_TYPE(SECOND, VT_SPL_cond_while);
S=createS();
rho_omp_for = new ast_vertex(0, VT_OMP_FOR, "");
rho_loop_init = loop_init->rho() ;
rho_loop      = loop->rho() ;

add_child( rho_omp_for, (*rho_loop_init->children.begin()) );
add_child( rho_omp_for, rho_loop );
add_child( S, rho_omp_for );

return S;
}

ast_vertex*
ast_vertex::rho_stack()
{
switch(type) {
case VT_SPL_STACKfpush:
case VT_SPL_STACKipush:
case VT_SPL_STACKcpush:
case VT_SPL_STACKfpop:
case VT_SPL_STACKipop:
case VT_SPL_STACKcpop:
break;
default:
assert(0);
}
ast_vertex* S=createS();

assert(S->type==VT_S);
return S;
}

ast_vertex* 
ast_vertex::rho_conditional()
{
ast_vertex* S=createS();
switch(type) {
case VT_SPL_cond_if:
case VT_SPL_cond_while:
break;
default:
assert(0);
}
S = (*++children.begin())->rho();  
assert(S->type==VT_S);
return S;
}

ast_vertex* 
ast_vertex::rho_seq_of_stmts()
{
ASSUME_TYPE(this,VT_S);
ast_vertex* S;
list<ast_vertex*> seqs;
set_slc(); 
if(!slc) { 
partition_seq_into_slc_and_cfstmts(seqs);
S=createS();
for(list<ast_vertex*>::const_iterator it=seqs.begin();it!=seqs.end();it++) {
ast_vertex* resultfromRho = (*it)->rho(); ASSUME_TYPE(resultfromRho, VT_S); 		
S->appendS( resultfromRho );  		
}
}
else { 
S=createS();
ast_vertex* subseq=createS();
ostringstream oss;
oss << (*children.begin())->id;
ast_vertex* label=new ast_vertex( 0, VT_SPL_int_const, oss.str() );

ast_vertex* STACKcpop=new ast_vertex(0, VT_SPL_STACKcpop, adj_prefix);
add_child(subseq, STACKcpop);
for(list<ast_vertex*>::const_reverse_iterator it=children.rbegin();it!=children.rend();it++) {
ast_vertex* resultfromRho = (*it)->rho(); ASSUME_TYPE(resultfromRho, VT_S); 
subseq->appendS( resultfromRho ); 
}

ast_vertex* testexpr = new ast_vertex(0, VT_SPL_eq, "");
ast_vertex* STACKctop = new ast_vertex(0, VT_SPL_STACKctop, adj_prefix);
add_child(testexpr, STACKctop);
add_child(testexpr, label);
ast_vertex* branch = new ast_vertex(0, VT_SPL_cond_if, "");
add_child(branch, testexpr);
add_child(branch, subseq);

add_child(S, branch);
}
return S;
}

ast_vertex* 
ast_vertex::rho_assign()
{
string var, der_var;
ast_vertex* v=this;
ast_vertex* lhs=NULL;
ast_vertex* rhs=NULL;
ast_vertex* assign=NULL;
ast_vertex* STACKpop=NULL;
ast_vertex* STACKtop=NULL;
ast_vertex* stack_restore=NULL;

switch(type) {
case VT_SPL_int_assign:
STACKpop = new ast_vertex(0, VT_SPL_STACKipop, adj_prefix);
STACKtop = new ast_vertex(0, VT_SPL_STACKitop, adj_prefix);
stack_restore = new ast_vertex(0, VT_SPL_int_assign, "");
break;
case VT_SPL_float_assign:
case VT_SPL_float_plus_assign:
STACKpop = new ast_vertex(0, VT_SPL_STACKfpop, adj_prefix);
STACKtop = new ast_vertex(0, VT_SPL_STACKftop, adj_prefix);
stack_restore = new ast_vertex(0, VT_SPL_float_assign, "");
break;
default:
assert(0);
}
ast_vertex* S=createS();
lhs = FIRST->clone();
add_child(stack_restore, lhs);
add_child(stack_restore, STACKtop);
add_child(S, stack_restore);
add_child(S, STACKpop);
if(type==VT_SPL_float_assign || type==VT_SPL_float_plus_assign) {
if( SECOND->type!=VT_SPL_STACKftop ) {  
S->appendS( build_adj_rhs_forward(var, der_var) );

lhs = new ast_vertex(0, VT_SPL_identifier, der_var);
rhs = FIRST->clone(); rhs->str = adj_prefix + rhs->str;
assign = new ast_vertex(0, VT_SPL_float_assign, "");
add_child(assign, lhs);
add_child(assign, rhs);
add_child(S, assign);
if(type==VT_SPL_float_assign) {
lhs = rhs;
rhs = new ast_vertex(0, VT_SPL_float_const, "0.");
assign = new ast_vertex(0, VT_SPL_float_assign, "");
add_child(assign, lhs);   add_child(assign, rhs);
add_child(S, assign);
}

S->appendS( build_adj_rhs_reverse(var, der_var) );
}
else { 
if(type==VT_SPL_float_assign) {
lhs = FIRST->clone(); lhs->str = adj_prefix + lhs->str;
rhs = new ast_vertex(0, VT_SPL_float_const, "0.");
assign = new ast_vertex(0, VT_SPL_float_assign, "");
add_child(assign, lhs);   add_child(assign, rhs);
add_child(S, assign);
}
}
}
assert(S->type==VT_S);
return S;
}

ast_vertex* 
ast_vertex::build_adj_rhs_forward(string& var, string& der_var)
{
ast_vertex* S=NULL;
ast_vertex* v=this;
ast_vertex* assign=NULL;
ast_vertex* lhs=NULL;
ast_vertex* rhs=NULL;
string l,r;
string der_l,der_r;

S=createS();
if(type==VT_SPL_float_assign || type==VT_SPL_float_plus_assign) {
set_intermediate_name_counter_to_zero();
S->appendS( SECOND->build_adj_rhs_forward(l,der_l) );
var=l; der_var=der_l;
}
else {
switch(type) {
case VT_SPL_float_add:
S->appendS( FIRST->build_adj_rhs_forward(l,der_l) );
S->appendS( SECOND->build_adj_rhs_forward(r,der_r) );
get_intermediate_name(adjoint, var, der_var);

assign = new ast_vertex(0, VT_SPL_float_assign, "");
lhs = new ast_vertex(0, VT_SPL_identifier, var);
add_child(assign, lhs);
rhs = new ast_vertex(0, VT_SPL_float_add, "");
add_child(rhs, new ast_vertex(0, VT_SPL_identifier, l));
add_child(rhs, new ast_vertex(0, VT_SPL_identifier, r));
add_child(assign, rhs);

add_child(S, assign);
break;
case VT_SPL_float_sub:
S->appendS( FIRST->build_adj_rhs_forward(l,der_l) );
S->appendS( SECOND->build_adj_rhs_forward(r,der_r) );
get_intermediate_name(adjoint, var, der_var);

assign = new ast_vertex(0, VT_SPL_float_assign, "");
lhs = new ast_vertex(0, VT_SPL_identifier, var);
add_child(assign, lhs);
rhs = new ast_vertex(0, VT_SPL_float_sub, "");
add_child(rhs, new ast_vertex(0, VT_SPL_identifier, l));
add_child(rhs, new ast_vertex(0, VT_SPL_identifier, r));
add_child(assign, rhs);

add_child(S, assign);
break;
case VT_SPL_float_mul:
S->appendS( FIRST->build_adj_rhs_forward(l,der_l) );
S->appendS( SECOND->build_adj_rhs_forward(r,der_r) );
get_intermediate_name(adjoint, var, der_var);

assign = new ast_vertex(0, VT_SPL_float_assign, "");
lhs = new ast_vertex(0, VT_SPL_identifier, var);
add_child(assign, lhs);
rhs = new ast_vertex(0, VT_SPL_float_mul, "");
add_child(rhs, new ast_vertex(0, VT_SPL_identifier, l));
add_child(rhs, new ast_vertex(0, VT_SPL_identifier, r));
add_child(assign, rhs);

add_child(S, assign);
break;
case VT_SPL_float_sin:
S->appendS( FIRST->build_adj_rhs_forward(l,der_l) );
get_intermediate_name(adjoint, var, der_var);

assign = new ast_vertex(0, VT_SPL_float_assign, "");
lhs = new ast_vertex(0, VT_SPL_identifier, var);
add_child(assign, lhs);
rhs = new ast_vertex(0, VT_SPL_float_sin, "");
add_child(rhs, new ast_vertex(0, VT_SPL_identifier, l));
add_child(assign, rhs);

add_child(S, assign);
break;
case VT_SPL_float_cos:
S->appendS( FIRST->build_adj_rhs_forward(l,der_l) );
get_intermediate_name(adjoint, var, der_var);

assign = new ast_vertex(0, VT_SPL_float_assign, "");
lhs = new ast_vertex(0, VT_SPL_identifier, var);
add_child(assign, lhs);
rhs = new ast_vertex(0, VT_SPL_float_cos, "");
add_child(rhs, new ast_vertex(0, VT_SPL_identifier, l));
add_child(assign, rhs);

add_child(S, assign);
break;
case VT_SPL_float_exp:
S->appendS( FIRST->build_adj_rhs_forward(l,der_l) );
get_intermediate_name(adjoint, var, der_var);

assign = new ast_vertex(0, VT_SPL_float_assign, "");
lhs = new ast_vertex(0, VT_SPL_identifier, var);
add_child(assign, lhs);
rhs = new ast_vertex(0, VT_SPL_float_exp, "");
add_child(rhs, new ast_vertex(0, VT_SPL_identifier, l));
add_child(assign, rhs);

add_child(S, assign);
break;
case VT_SPL_identifier: 
get_intermediate_name(adjoint, var, der_var);

assign = new ast_vertex(0, VT_SPL_float_assign, "");
lhs = new ast_vertex(0, VT_SPL_identifier, var);
add_child(assign, lhs);
rhs = clone();
add_child(assign, rhs);

add_child(S, assign);
break;
case VT_SPL_float_const:
get_intermediate_name(adjoint, var, der_var);

assign = new ast_vertex(0, VT_SPL_float_assign, "");
lhs = new ast_vertex(0, VT_SPL_identifier, var);
add_child(assign, lhs);
rhs = clone();
add_child(assign, rhs);

add_child(S, assign);
break;
case VT_SPL_expr_in_brackets:
S->appendS( FIRST->build_adj_rhs_forward(var, der_var) );
break;
case VT_SPL_STACKftop:
break;
case VT_SPL_array_index:
assert(0);
break;
default: 
UNKNOWN_TYPE(this);
}
}
assert(S->type==VT_S);
return S;
}


ast_vertex*  
ast_vertex::build_adj_rhs_reverse(string& var, string& der_var)
{
ast_vertex* S=NULL;
ast_vertex* v=this;
ast_vertex* assign=NULL;
ast_vertex* lhs=NULL;
ast_vertex* rhs=NULL;
ast_vertex* tmp=NULL;
ast_vertex* tmp2=NULL;
ast_vertex* S_first=NULL;
ast_vertex* S_second=NULL;
string l,r;
string der_l,der_r;

S=createS();
if(type==VT_SPL_float_assign || type==VT_SPL_float_plus_assign) {
set_intermediate_name_counter_to_zero();
S->appendS( SECOND->build_adj_rhs_reverse(l, der_l) );
var=l;
}
else {
switch(type) {
case VT_SPL_float_add:
S_first = FIRST->build_adj_rhs_reverse(l, der_l);
S_second = SECOND->build_adj_rhs_reverse(r, der_r);
get_intermediate_name(adjoint, var, der_var);

assign = new ast_vertex(0, VT_SPL_float_assign, "");
lhs = new ast_vertex(0, VT_SPL_identifier, der_l);
rhs = new ast_vertex(0, VT_SPL_identifier, der_var);
add_child(assign, lhs);   add_child(assign, rhs);
add_child(S, assign);
assign = new ast_vertex(0, VT_SPL_float_assign, "");
lhs = new ast_vertex(0, VT_SPL_identifier, der_r);
rhs = new ast_vertex(0, VT_SPL_identifier, der_var);
add_child(assign, lhs);   add_child(assign, rhs);
add_child(S, assign);

S->appendS( S_first );
S->appendS( S_second );
break;
case VT_SPL_float_sub:
S_first = FIRST->build_adj_rhs_reverse(l, der_l);
S_second = SECOND->build_adj_rhs_reverse(r, der_r);
get_intermediate_name(adjoint, var, der_var);

assign = new ast_vertex(0, VT_SPL_float_assign, "");
lhs = new ast_vertex(0, VT_SPL_identifier, der_l);
rhs = new ast_vertex(0, VT_SPL_identifier, der_var);
add_child(assign, lhs);   add_child(assign, rhs);
add_child(S, assign);

assign = new ast_vertex(0, VT_SPL_float_assign, "");
lhs = new ast_vertex(0, VT_SPL_identifier, der_r);
add_child(assign, lhs);
rhs = new ast_vertex(0, VT_SPL_float_sub, "");
add_child(rhs, new ast_vertex(0, VT_SPL_float_const, "0."));
add_child(rhs, new ast_vertex(0, VT_SPL_identifier, der_var));
add_child(assign, rhs);
add_child(S, assign);

S->appendS( S_first );
S->appendS( S_second );
break;
case VT_SPL_float_mul:
S_first = FIRST->build_adj_rhs_reverse(l, der_l);
S_second = SECOND->build_adj_rhs_reverse(r, der_r);
get_intermediate_name(adjoint, var, der_var);

assign = new ast_vertex(0, VT_SPL_float_assign, "");
lhs = new ast_vertex(0, VT_SPL_identifier, der_l);
add_child(assign, lhs);
rhs = new ast_vertex(0, VT_SPL_float_mul, "");
add_child(rhs, new ast_vertex(0, VT_SPL_identifier, der_var));
add_child(rhs, new ast_vertex(0, VT_SPL_identifier, r));
add_child(assign, rhs);
add_child(S, assign);

assign = new ast_vertex(0, VT_SPL_float_assign, "");
lhs = new ast_vertex(0, VT_SPL_identifier, der_r);
add_child(assign, lhs);
rhs = new ast_vertex(0, VT_SPL_float_mul, "");
add_child(rhs, new ast_vertex(0, VT_SPL_identifier, der_var));
add_child(rhs, new ast_vertex(0, VT_SPL_identifier, l));
add_child(assign, rhs);
add_child(S, assign);

S->appendS( S_first );
S->appendS( S_second );
break;
case VT_SPL_float_sin:
S_first = FIRST->build_adj_rhs_reverse(l, der_l);
get_intermediate_name(adjoint, var, der_var);

assign = new ast_vertex(0, VT_SPL_float_assign, "");
lhs = new ast_vertex(0, VT_SPL_identifier, der_l);
add_child(assign, lhs);
rhs = new ast_vertex(0, VT_SPL_float_mul, "");
add_child(rhs, new ast_vertex(0, VT_SPL_identifier, der_var));
tmp = new ast_vertex(0, VT_SPL_float_cos, "");
add_child(tmp, new ast_vertex(0, VT_SPL_identifier, l));
add_child(rhs, tmp);
add_child(assign, rhs);
add_child(S, assign);

S->appendS( S_first );
break;
case VT_SPL_float_cos:
S_first = FIRST->build_adj_rhs_reverse(l,der_l) ;
get_intermediate_name(adjoint, var, der_var);

tmp = new ast_vertex(0, VT_SPL_float_sin, "");
add_child(tmp, new ast_vertex(0, VT_SPL_identifier, l));
tmp2 = new ast_vertex(0, VT_SPL_float_sub, "");
add_child(tmp2, new ast_vertex(0, VT_SPL_float_const, "0."));
add_child(tmp2, tmp);
tmp = new ast_vertex(0, VT_SPL_expr_in_brackets, "");
add_child(tmp, tmp2);

assign = new ast_vertex(0, VT_SPL_float_assign, "");
lhs = new ast_vertex(0, VT_SPL_identifier, der_l);
add_child(assign, lhs);
rhs = new ast_vertex(0, VT_SPL_float_mul, "");
add_child(rhs, new ast_vertex(0, VT_SPL_identifier, der_var));
add_child(rhs, tmp);
add_child(assign, rhs);
add_child(S, assign);

S->appendS( S_first );
break;
case VT_SPL_float_exp:
S_first = FIRST->build_adj_rhs_reverse(l, der_l);
get_intermediate_name(adjoint, var, der_var);

assign = new ast_vertex(0, VT_SPL_float_assign, "");
lhs = new ast_vertex(0, VT_SPL_identifier, der_l);
add_child(assign, lhs);
rhs = new ast_vertex(0, VT_SPL_float_mul, "");
add_child(rhs, new ast_vertex(0, VT_SPL_identifier, der_var));
tmp = new ast_vertex(0, VT_SPL_float_exp, "");
add_child(tmp, new ast_vertex(0, VT_SPL_identifier, l));
add_child(rhs, tmp);
add_child(assign, rhs);
add_child(S, assign);

S->appendS( S_first );
break;
case VT_SPL_identifier: 
get_intermediate_name(adjoint, var, der_var);




if( option_suppress_atomic || inside_critical_construct ) {
assign = new ast_vertex(0, VT_SPL_float_plus_assign, "");
lhs = clone();     lhs->str = adj_prefix+str;
add_child(assign, lhs);
rhs = new ast_vertex(0, VT_SPL_identifier, der_var);
add_child(assign, rhs);
add_child(S, assign);
}
else {
ast_vertex* atomic_stmt = NULL;
bool exclusive_read_property = false;
if( sym_tbl.is_private(get_str()) ) {
exclusive_read_property = true;
}
else if(    exclusive_read::is_memory_read_exclusive.size()>0 
&& exclusive_read::is_memory_read_exclusive.count(this)>0  ) {
exclusive_read_property = exclusive_read::is_memory_read_exclusive[this];
}
if( !exclusive_read_property )
atomic_stmt = new ast_vertex(0, VT_ATOMIC, "");
assign = new ast_vertex(0, VT_SPL_float_plus_assign, "");
lhs = clone();     lhs->str = adj_prefix+str;
add_child(assign, lhs);
rhs = new ast_vertex(0, VT_SPL_identifier, der_var);
add_child(assign, rhs);
if( atomic_stmt ) {
add_child(atomic_stmt, assign);
add_child(S, atomic_stmt);
}
else {
add_child(S, assign);
}
}
break;
case VT_SPL_float_const:
get_intermediate_name(adjoint, var, der_var);

break;
case VT_SPL_expr_in_brackets:
S->appendS( FIRST->build_adj_rhs_reverse(var, der_var) );
break;
case VT_SPL_STACKftop:
case VT_SPL_array_index:
assert(0);
break;
default: 
UNKNOWN_TYPE(this);
}
}
assert(S->type==VT_S);
return S;
}

ast_vertex*
ast_vertex::rho_barrier()
{
ast_vertex* S=NULL;
ASSUME_TYPE(this, VT_BARRIER);
S=createS();
add_child(S, clone());
ASSUME_TYPE(S, VT_S);
return S;
}



ast_vertex*
ast_vertex::rho_dummy()
{
ast_vertex* v=this;
ast_vertex* S=NULL;
list<ast_vertex*> l;
ast_vertex* list_of_vars=FIRST;
ast_vertex* result_rho_dummy=clone();

list<ast_vertex*>::const_iterator begin_iterator=list_of_vars->get_children().begin();
for(list<ast_vertex*>::const_iterator it=begin_iterator;it!=list_of_vars->get_children().end();it++) 
{ 
ast_vertex* adj_pendant = (*it)->clone();
adj_pendant->str = adj_prefix + adj_pendant->str;
l.push_back( adj_pendant );
}
v=result_rho_dummy;
list_of_vars=FIRST;
for(list<ast_vertex*>::const_iterator it=l.begin();it!=l.end();it++) {
add_child(list_of_vars, *it);
}
S=createS();
add_child(S, result_rho_dummy);
ASSUME_TYPE(S, VT_S);
return S;
}


ast_vertex*
ast_vertex::rho_master()
{
ast_vertex* v=this;
ast_vertex* S=NULL;
ASSUME_TYPE(this, VT_MASTER);
S=createS();
ast_vertex* result_rho=FIRST->rho(); ASSUME_TYPE(result_rho, VT_S);
S->appendS(result_rho);
ASSUME_TYPE(S, VT_S);
return S;
}


ast_vertex*
ast_vertex::rho_atomic()
{ 
ast_vertex* S=NULL;
ASSUME_TYPE(this, VT_ATOMIC);
S=createS();

ast_vertex *critical_region = new ast_vertex(line_number, VT_CRITICAL, "");
ast_vertex *critical_region_subseq = createS();
add_child(critical_region, critical_region_subseq);

ast_vertex* if_stmt = new ast_vertex(0, VT_SPL_cond_if, "");
ast_vertex* if_stmt_test = new ast_vertex(0, VT_SPL_neq, ""); 
ast_vertex* two = new ast_vertex(0, VT_SPL_int_const, "2");
ast_vertex* lhs = new ast_vertex(line_number, VT_SPL_identifier, ast.atomic_flag_names[id] );
add_child(if_stmt_test, lhs);
add_child(if_stmt_test, two);
ast_vertex* if_stmt_S = createS(); 
add_child(if_stmt, if_stmt_test);
add_child(if_stmt, if_stmt_S);

ast_vertex* atomic_flag_assign = new ast_vertex(0, VT_SPL_int_assign, "");
add_child( atomic_flag_assign, lhs );
add_child( atomic_flag_assign, two );
add_child( if_stmt_S, atomic_flag_assign );
add_child( critical_region_subseq, if_stmt );

ast_vertex* atomic_storage_assign = new ast_vertex(0, VT_SPL_float_assign, "");
ast_vertex* atomic_storage_identifier = new ast_vertex(line_number, VT_SPL_identifier, ast.atomic_storage_names[id] );
add_child( atomic_storage_assign, FIRSTOF(FIRSTOF(this)) );
add_child( atomic_storage_assign, atomic_storage_identifier);
add_child( if_stmt_S, atomic_storage_assign );


ast_vertex *incremental_assign = FIRSTOF(this);  ASSUME_TYPE(incremental_assign, VT_SPL_float_plus_assign);
ast_vertex* result_rho=incremental_assign->rho(); ASSUME_TYPE(result_rho, VT_S);
result_rho->children.pop_front();
result_rho->children.pop_front();
S->appendS( result_rho );
add_child(S, critical_region);

ASSUME_TYPE(S, VT_S);
return S;
}


ast_vertex*
ast_vertex::rho_exclusive_read()
{ 
ast_vertex* v=this;
ast_vertex* S=NULL;
ASSUME_TYPE(this, VT_EXCLUSIVE_READ_FAILURE);
ast_vertex* result_rho=FIRST->rho(); ASSUME_TYPE(result_rho, VT_S);
list<ast_vertex*>::iterator it=result_rho->children.begin();
list<ast_vertex*>::iterator it2;
while( it!=result_rho->children.end()  &&  (*it)->get_type()!=VT_SPL_STACKfpop )
it++;
assert(it!=result_rho->children.end());
it++;
for(;it!=result_rho->children.end();it++) {
if( (*it)->get_type()!=VT_SPL_float_plus_assign ) continue;
if( sym_tbl.get_sym( FIRSTOF(*it)->get_str() )->is_sym_intermediate() ) continue;

assert( !sym_tbl.get_sym( FIRSTOF(*it)->get_str() )->is_sym_intermediate() );
assert( (*it)->get_type()==VT_SPL_float_plus_assign ) ;
ast_vertex* atomic= new ast_vertex(line_number, VT_ATOMIC, "");
add_child(atomic, *it);
ast_vertex *check_address=*it;
result_rho->children.insert(it, atomic);

for(it2=result_rho->children.begin(); it2!=result_rho->children.end() && *it2!=check_address;it2++) 
;
assert( *it2==check_address );
result_rho->children.erase(it2);
for(it=result_rho->children.begin(); it!=result_rho->children.end() && *it!=atomic;it++) 
;
}
S=result_rho;
ASSUME_TYPE(S, VT_S);
return S;
}



ast_vertex*
ast_vertex::rho_critical()
{
ast_vertex* v=this; 
ast_vertex* S=NULL; 
ASSUME_TYPE(this, VT_CRITICAL); 
inside_critical_construct=true;
S=createS(); 

ast_vertex* result_trans=FIRST->rho(); ASSUME_TYPE(result_trans, VT_S); 
ostringstream oss; oss << id;
ast_vertex* astv_label=new ast_vertex( 0, VT_SPL_int_const, oss.str() );
ast_vertex* testexpr = new ast_vertex(0, VT_SPL_neq, "");
ast_vertex* STACKctop = new ast_vertex(0, VT_SPL_STACKctop, adj_prefix);
add_child(testexpr, STACKctop);
add_child(testexpr, astv_label);
ast_vertex* while_loop = new ast_vertex(0, VT_SPL_cond_while, "");
add_child(while_loop, testexpr);
add_child(while_loop, result_trans);


ast_vertex* subseq=createS();
add_child( subseq, new ast_vertex( line_number, VT_SPL_STACKcpop, adj_prefix ) ) ;
ast_vertex* lhs = new ast_vertex(line_number, VT_SPL_identifier, ast.critical_counter_names[id].first );
ast_vertex* expr_sub = new ast_vertex(line_number, VT_SPL_int_sub, ""); 
add_child( expr_sub, new ast_vertex(line_number, VT_SPL_identifier, ast.critical_counter_names[id].first ) );
add_child( expr_sub, new ast_vertex(line_number, VT_SPL_int_const, "1") );
ast_vertex* decrement_for_a_l = new ast_vertex(line_number, VT_SPL_int_assign, ""); 
add_child( decrement_for_a_l, lhs );
add_child( decrement_for_a_l, expr_sub );
add_child( subseq, decrement_for_a_l );
add_child(subseq, while_loop);
expr_sub = new ast_vertex(line_number, VT_SPL_int_sub, ""); 
add_child( expr_sub, new ast_vertex(line_number, VT_SPL_identifier, ast.critical_counter_names[id].first ) );
add_child( expr_sub, new ast_vertex(line_number, VT_SPL_int_const, "1") );
testexpr = new ast_vertex(line_number, VT_SPL_eq, "") ;
add_child( testexpr, new ast_vertex( line_number, VT_SPL_STACKctop, adj_prefix ) ) ;
add_child( testexpr, expr_sub ) ;
ast_vertex* branch = new ast_vertex(0, VT_SPL_cond_if, "");
add_child(branch, testexpr);
add_child(branch, subseq);



subseq=createS();
add_child(subseq, branch);
ast_vertex* critical_region=clone_node();  add_child(critical_region, subseq);  

subseq=createS();
add_child(subseq, critical_region);
astv_label=new ast_vertex( 0, VT_SPL_int_const, oss.str() );
testexpr = new ast_vertex(0, VT_SPL_neq, "");
STACKctop = new ast_vertex(0, VT_SPL_STACKctop, adj_prefix);
add_child(testexpr, STACKctop);
add_child(testexpr, astv_label);
while_loop = new ast_vertex(0, VT_SPL_cond_while, "");
add_child(while_loop, testexpr);
add_child(while_loop, subseq);

subseq=createS();
add_child(subseq, new ast_vertex(0, VT_SPL_STACKcpop, adj_prefix));
add_child(subseq, while_loop);
add_child(subseq, new ast_vertex(0, VT_SPL_STACKcpop, adj_prefix));

ast_vertex* branch_for_asking_for_label_from_critical_construct = createBranchSTACKcTop_equal_label(id, adj_prefix, subseq);
add_child( S, branch_for_asking_for_label_from_critical_construct );
inside_critical_construct=false;
ASSUME_TYPE(S, VT_S);
return S;
}


ast_vertex*
ast_vertex::rho_assert()
{
ast_vertex* S=NULL;

S=createS();
add_child(S, clone());
ASSUME_TYPE(S, VT_S);
return S;
}


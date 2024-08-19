#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "ompi.h"
#include "builder.h"
#include "dfa.h"
#include "ast_copy.h"
#include "ast_free.h"
#include "ast_print.h"
#include "ast_vars.h"
#include "ast_xform.h"
#include "x_clauses.h"
#include "x_parallel.h"
#include "x_arith.h"
#include "x_task.h"
#include "x_single.h"
#include "x_sections.h"
#include "x_for.h"
#include "x_thrpriv.h"
#include "x_shglob.h"
#include "x_target.h"
#include "x_decltarg.h"
#include "x_types.h"
#include "x_cars.h"
#include "x_teams.h"
#include "x_arrays.h"
#include "ox_xform.h"
int closest_parallel_scope = -1;
int target_data_scope = -1;
int inDeclTarget = 0;
static int cur_parallel_line = 0;
static int cur_taskgroup_line = 0;
void ast_stmt_iteration_xform(aststmt *t)
{
switch ((*t)->subtype)
{
case SFOR:
{
int newscope = false;
if ((*t)->u.iteration.init != NULL)
{
if ((*t)->u.iteration.init->type == DECLARATION)
{
newscope = true;
scope_start(stab);  
}
ast_stmt_xform(&((*t)->u.iteration.init));
}
if ((*t)->u.iteration.cond != NULL)
ast_expr_xform(&((*t)->u.iteration.cond));
if ((*t)->u.iteration.incr != NULL)
ast_expr_xform(&((*t)->u.iteration.incr));
ast_stmt_xform(&((*t)->body));
if (newscope)
scope_end(stab);
break;
}
case SWHILE:
if ((*t)->u.iteration.cond != NULL)
ast_expr_xform(&((*t)->u.iteration.cond));
ast_stmt_xform(&((*t)->body));
break;
case SDO:
if ((*t)->u.iteration.cond != NULL)
ast_expr_xform(&((*t)->u.iteration.cond));
ast_stmt_xform(&((*t)->body));
break;
}
}
void ast_stmt_selection_xform(aststmt *t)
{
switch ((*t)->subtype)
{
case SSWITCH:
ast_expr_xform(&((*t)->u.selection.cond));
ast_stmt_xform(&((*t)->body));
break;
case SIF:
ast_expr_xform(&((*t)->u.selection.cond));
ast_stmt_xform(&((*t)->body));
if ((*t)->u.selection.elsebody)
ast_stmt_xform(&((*t)->u.selection.elsebody));
break;
}
}
void ast_stmt_labeled_xform(aststmt *t)
{
ast_stmt_xform(&((*t)->body));
}
void ast_stmt_xform(aststmt *t)
{
if (t == NULL || *t == NULL) return;
switch ((*t)->type)
{
case EXPRESSION:
ast_expr_xform(&((*t)->u.expr));
break;
case ITERATION:
ast_stmt_iteration_xform(t);
break;
case JUMP:   
if ((*t)->subtype == SRETURN && (*t)->u.expr != NULL)
ast_expr_xform(&((*t)->u.expr));
break;
case SELECTION:
ast_stmt_selection_xform(t);
break;
case LABELED:
ast_stmt_labeled_xform(t);
break;
case COMPOUND:
if ((*t)->body)
{
scope_start(stab);
ast_stmt_xform(&((*t)->body));
scope_end(stab);
}
break;
case STATEMENTLIST:
ast_stmt_xform(&((*t)->u.next));
ast_stmt_xform(&((*t)->body));
break;
case DECLARATION:
xt_declaration_xform(t);   
break;
case FUNCDEF:
{
symbol  fsym = decl_getidentifier_symbol((*t)->u.declaration.decl);
stentry f    = symtab_get(stab, fsym, FUNCNAME);
assert(f == NULL || f->funcdef == NULL); 
if (f == NULL)
{
f = symtab_put(stab,fsym, FUNCNAME);
f->spec = (*t)->u.declaration.spec;
f->decl = (*t)->u.declaration.decl;
}
f->funcdef = *t;
if (decltarg_id_isknown(fsym)) 
{
inDeclTarget++;
decltarg_bind_id(f);  
}
scope_start(stab);         
if ((*t)->u.declaration.dlist) 
{
xt_dlist_array2pointer((*t)->u.declaration.dlist);  
ast_stmt_xform(&((*t)->u.declaration.dlist));  
}
else                           
{
xt_barebones_substitute(&((*t)->u.declaration.spec),
&((*t)->u.declaration.decl));
xt_decl_array2pointer((*t)->u.declaration.decl->decl->u.params);
ast_declare_function_params((*t)->u.declaration.decl);
}
ast_stmt_xform(&((*t)->body));
tp_fix_funcbody_gtpvars((*t)->body);    
if (processmode)
analyze_pointerize_sgl(*t);        
scope_end(stab);           
if (decltarg_id_isknown(fsym))  
inDeclTarget--;
break;
}
case ASMSTMT:
break;
case OMPSTMT:
if (enableOpenMP || testingmode) ast_omp_xform(t);
break;
case OX_STMT:
if (enableOmpix || testingmode) ast_ompix_xform(t);
break;
}
}
void ast_expr_xform(astexpr *t)
{
if (t == NULL || *t == NULL) return;
switch ((*t)->type)
{
case CONDEXPR:
ast_expr_xform(&((*t)->u.cond));
case FUNCCALL:
case BOP:
case ASS:
case DESIGNATED:
case COMMALIST:
case SPACELIST:
case ARRAYIDX:
if ((*t)->right)
ast_expr_xform(&((*t)->right));
case DOTFIELD:
case PTRFIELD:
case BRACEDINIT:
case PREOP:
case POSTOP:
case IDXDES:
ast_expr_xform(&((*t)->left));
break;
case CASTEXPR:
xt_barebones_substitute(&((*t)->u.dtype->spec), &((*t)->u.dtype->decl));
ast_expr_xform(&((*t)->left));
break;
case UOP:
if ((*t)->opid == UOP_sizeoftype || (*t)->opid == UOP_typetrick)
xt_barebones_substitute(&((*t)->u.dtype->spec), &((*t)->u.dtype->decl));
else
ast_expr_xform(&((*t)->left));
break;
}
}
void ast_ompcon_xform(ompcon *t)
{
if ((*t)->body)     
ast_stmt_xform(&((*t)->body));
}
static
aststmt linepragma(int line, symbol file)
{
return ((cppLineNo) ? verbit("# %d \"%s\"", line, file->name) : verbit(""));
}
aststmt ompdir_commented(ompdir d)
{
static str bf = NULL;
aststmt    st;
if (bf == NULL) bf = Strnew();
str_printf(bf, "");
st = BlockList(Verbatim(strdup(str_string(bf))),
linepragma(d->l, d->file));
str_truncate(bf);
return (st);
}
static
astdecl fix_array_decl_copy(astdecl copy, astdecl orig)
{
int ic = decl_initializer_cardinality(orig);
if (ic == 0)
exit_error(2, "(%s, line %d): cannot determine missing dimension "
"size of '%s'\n", orig->file->name, orig->l, 
decl_getidentifier_symbol(orig)->name);
arr_dimension_size(copy, 0, numConstant(ic));
return (copy);
}
astdecl xform_clone_declonly(stentry e)
{
astdecl decl = ast_decl_copy(e->decl);
if (e->isarray && arr_dimension_size(decl, 0, NULL) == NULL)
{
if (e->idecl)
fix_array_decl_copy(decl, e->idecl);
else
exit_error(2, "(%s, line %d): '%s' cannot be cloned; "
"unknown dimension size\n", 
e->decl->file->name, e->decl->l, e->key->name);
}
return (decl);
}
inline aststmt 
xform_clone_declaration(symbol s, astexpr initer, bool mkptr, symbol snew)
{
stentry e    = symtab_get(stab, s, IDNAME);
astdecl decl = xform_clone_declonly(e);
if (snew)     
{
astdecl id = IdentifierDecl(snew);
*(decl_getidentifier(decl)) = *id;
free(id);
}
if (mkptr)    
decl = decl_topointer(decl);
return Declaration(
ast_spec_copy_nosc(e->spec),
initer ? InitDecl(decl, initer) : decl
);
}
inline aststmt xform_clone_funcdecl(symbol funcname)
{
stentry func = symtab_get(stab, funcname, FUNCNAME);
assert(func != NULL);
return (Declaration(ast_spec_copy(func->spec), ast_decl_copy(func->decl)));
}
int xform_implicit_barrier_is_needed(ompcon t)
{
aststmt p = t->parent;    
for (; p->parent->type == STATEMENTLIST && p->parent->body == p; p = p->parent)
;
for (; p->parent->type == COMPOUND; p = p->parent)
;
if ((p = p->parent)->type == OMPSTMT)
{
t = p->u.omp;
if (t->type == DCPARSECTIONS || t->type == DCSECTIONS ||
t->type == DCPARALLEL || t->type == DCSINGLE)
if (xc_ompcon_get_clause(t, OCNOWAIT) == NULL)
return (0);  
}
return (1);        
}
void xform_master(aststmt *t)
{
aststmt s = (*t)->u.omp->body, parent = (*t)->parent, v;
bool    parisif;
v = ompdir_commented((*t)->u.omp->directive); 
parisif = (*t)->parent->type == SELECTION && (*t)->parent->subtype == SIF;
(*t)->u.omp->body = NULL;     
ast_free(*t);                 
*t = Block3(
v,
If( 
BinaryOperator(
BOP_eqeq,
Call0_expr("omp_get_thread_num"),
numConstant(0)
),
s,
NULL
),
linepragma(s->l + 1 - parisif, s->file)
);
if (parisif)
*t = Compound(*t);
ast_stmt_parent(parent, *t);
}
void xform_ordered_doacross(aststmt *t)
{
aststmt   s = (*t)->u.omp->body, parent = (*t)->parent, 
v = ompdir_commented((*t)->u.omp->directive);
ompclause cl;
ompcon    enclosing;
int       i, ordnum, ncla;
symbol    *indices;
astexpr   args, vec, initer;
enclosing = ast_get_enclosing_ompcon((*t)->parent, 0);
if ((enclosing->type != DCFOR && enclosing->type != DCFOR_P) || 
!(cl = xc_ompcon_get_clause(enclosing, OCORDEREDNUM)))
exit_error(1, "(%s, line %d) openmp error:\n\t"
"stand-alone ordered constructs should be closely nested in\n\t"
"doacross loops.\n",
(*t)->u.omp->directive->file->name, (*t)->u.omp->directive->l);
ordnum = cl->subtype; 
if (ordnum <= 0)
exit_error(1, "(%s, line %d) openmp error:\n\t"
"ordered loop nest depth must be > 0.\n",
(*t)->u.omp->directive->file->name, (*t)->u.omp->directive->l);
cl = xc_ompcon_get_clause((*t)->u.omp, OCDEPEND);
if (!cl)    
exit_error(1, "(%s, line %d) openmp error:\n\t"
"stand-alone ordered constructs should contain depend clause(s).\n",
(*t)->u.omp->directive->file->name, (*t)->u.omp->directive->l);
if (cl->subtype != OC_source && cl->subtype != OC_sink)   
exit_error(1, "(%s, line %d) openmp error:\n\t"         
"stand-alone ordered constructs should contain depend(source) or\n\t"
"depend(sink: ) clause(s).\n",
(*t)->u.omp->directive->file->name, (*t)->u.omp->directive->l);
indices = ompfor_get_indices(enclosing->body, ordnum);
if (cl->subtype == OC_source)
{
args = Identifier(indices[0]);
for (i = 1; i < ordnum; i++)
args = CommaList(args, Identifier(indices[i]));
v = BlockList(
v,
FuncCallStmt(
IdentName("ort_doacross_post"), 
Comma2(
IdentName(DOACCPARAMS),
CastedExpr(         
Casttypename(
Declspec(SPEC_long),
AbstractDeclarator(
NULL,
ArrayDecl(NULL, NULL, NULL)
)
),
BracedInitializer(args)
)
)
)
);
}
else   
{
initer = NULL;
ncla = 0;
cl = xc_ompcon_get_every_clause((*t)->u.omp, OCDEPEND);  
do
{
ncla++;
vec = (cl->type == OCLIST) ? cl->u.list.elem->u.expr : cl->u.expr;
args = NULL;
for (i = 0; i < ordnum; i++)
{
if ((i != ordnum-1 && vec->type != COMMALIST) ||   
(i == ordnum-1 && vec->type == COMMALIST))
exit_error(1, "(%s, line %d) openmp error:\n\t"
"sink indices (%d) are %s than the doacrros loop indices (%d).\n",
(*t)->u.omp->directive->file->name, (*t)->u.omp->directive->l,
i+1, (i == ordnum-1) ? "more" : "fewer", ordnum);
if (i != ordnum-1)
{
assert(vec->left->left->type == IDENT);
if (vec->left->left->u.sym != indices[i])
exit_error(1, "(%s, line %d) openmp error:\n\t"
"sink index '%s' does not correspond to loop index #%d (%s).\n",
(*t)->u.omp->directive->file->name, (*t)->u.omp->directive->l,
vec->left->left->u.sym->name, i+1, indices[i]->name); 
args = args ? CommaList(args, ast_expr_copy(vec->left)) : ast_expr_copy(vec->left);
vec = vec->right;
}
else
{
assert(vec->left->type == IDENT);
if (vec->left->u.sym != indices[i])
exit_error(1, "(%s, line %d) openmp error:\n\t"
"sink index '%s' does not correspond to loop index #%d (%s).\n",
(*t)->u.omp->directive->file->name, (*t)->u.omp->directive->l,
vec->left->u.sym->name, i+1, indices[i]->name); 
args = args ? CommaList(args, ast_expr_copy(vec)) : ast_expr_copy(vec);
}
}
initer = initer ? CommaList(initer, args) : args;
if (cl->type != OCLIST)
break;
else
cl = cl->u.list.next;
}
while (cl != NULL);
v = BlockList(
v,
FuncCallStmt(
IdentName("ort_doacross_wait"), 
Comma3(
IdentName(DOACCPARAMS),
numConstant(ncla), 
CastedExpr(         
Casttypename(
Declspec(SPEC_long),
AbstractDeclarator(
NULL,
ArrayDecl(NULL, NULL, NULL)
)
),
BracedInitializer(initer)
)
)
)
);
}
free(indices);
ast_free(*t);
*t = v;
ast_stmt_parent(parent, *t);
}
void xform_ordered(aststmt *t)
{
aststmt s = (*t)->u.omp->body, parent = (*t)->parent, v;
bool    stlist;   
if (OMPCON_IS_STANDALONE((*t)->u.omp))
{
xform_ordered_doacross(t);
return;
}
stlist = ((*t)->parent->type == STATEMENTLIST ||
(*t)->parent->type == COMPOUND);
v = ompdir_commented((*t)->u.omp->directive);
(*t)->u.omp->body = NULL;     
ast_free(*t);                 
*t = Block5(
v, 
Call0_stmt("ort_ordered_begin"),
s,
Call0_stmt("ort_ordered_end"),
linepragma(s->l + 1 - (!stlist), s->file)
);
if (!stlist)
*t = Compound(*t);
ast_stmt_parent(parent, *t);
}
void xform_critical(aststmt *t)
{
aststmt s = (*t)->u.omp->body, parent = (*t)->parent, v;
char    lock[128];
bool    stlist;   
v = ompdir_commented((*t)->u.omp->directive); 
stlist = ((*t)->parent->type == STATEMENTLIST ||
(*t)->parent->type == COMPOUND);
if ((*t)->u.omp->directive->u.region)
snprintf(lock, 127, "_ompi_crity_%s", (*t)->u.omp->directive->u.region->name);
else
strcpy(lock, "_ompi_crity");
(*t)->u.omp->body = NULL;    
ast_free(*t);                
if (!symbol_exists(lock))    
bld_globalvar_add(Declaration(Declspec(SPEC_void),   
InitDecl(
Declarator(
Pointer(),
IdentifierDecl(Symbol(lock))
),
NullExpr()
)));
*t = Block5(
v,
FuncCallStmt(   
IdentName("ort_critical_begin"),
UOAddress(IdentName(lock))
),
s,
FuncCallStmt(   
IdentName("ort_critical_end"),
UOAddress(IdentName(lock))
),
linepragma(s->l + 1 - (!stlist), s->file)
);
if (!stlist)
*t = Compound(*t);
ast_stmt_parent(parent, *t);
}
void xform_taskgroup(aststmt *t)
{
aststmt s = (*t)->u.omp->body, parent = (*t)->parent, v;
bool    stlist;   
char    clabel[23];
snprintf(clabel, 23, "CANCEL_taskgroup_%d", cur_taskgroup_line);
v = ompdir_commented((*t)->u.omp->directive); 
stlist = ((*t)->parent->type == STATEMENTLIST ||
(*t)->parent->type == COMPOUND);
(*t)->u.omp->body = NULL;    
ast_free(*t);                
*t = Block5(
v,
Call0_stmt("ort_entering_taskgroup"),
s,
Labeled(
Symbol(clabel), 
Call0_stmt("ort_leaving_taskgroup")
),
linepragma(s->l + 1 - (!stlist), s->file)
);
if (!stlist)
*t = Compound(*t);
ast_stmt_parent(parent, *t);
}
void xform_atomic(aststmt *t)
{
aststmt s = (*t)->u.omp->body, parent = (*t)->parent, v;
astexpr ex = s->u.expr;
bool    stlist;   
if ((s->type != EXPRESSION) ||
(ex->type != POSTOP && ex->type != PREOP && ex->type != ASS))
exit_error(1, "(%s, line %d) openmp error:\n\t"
"non-compliant ATOMIC expression.\n",
(*t)->u.omp->directive->file->name, (*t)->u.omp->directive->l);
v = ompdir_commented((*t)->u.omp->directive); 
stlist = ((*t)->parent->type == STATEMENTLIST ||
(*t)->parent->type == COMPOUND);
(*t)->u.omp->body = NULL;     
ast_free(*t);                 
if (ex->type == ASS &&
(ex->right->type != IDENT && !xar_expr_is_constant(ex->right)))
{
aststmt tmp;
tmp = Declaration(
(ex->left->type != IDENT ?
Declspec(SPEC_long) :
ast_spec_copy_nosc(
symtab_get(stab, ex->left->u.sym, IDNAME)->spec)
),
InitDecl(
Declarator(NULL, IdentifierDecl(Symbol("__tmp"))),
ex->right
)
);
ex->right = IdentName("__tmp");
*t = Compound(
Block6(
v, tmp, Call0_stmt("ort_atomic_begin"), s,
Call0_stmt("ort_atomic_end"),
linepragma(s->l + 1 - (!stlist), s->file)
)
);
}
else
{
*t = Block5(
v, Call0_stmt("ort_atomic_begin"), s, Call0_stmt("ort_atomic_end"),
linepragma(s->l + 1 - (!stlist), s->file)
);
if (!stlist)
*t = Compound(*t);
}
ast_stmt_parent(parent, *t);
}
void xform_flush(aststmt *t)
{
aststmt parent = (*t)->parent, v;
v = ompdir_commented((*t)->u.omp->directive);  
ast_free(*t);                                  
*t = BlockList(v, Call0_stmt("ort_fence"));
ast_stmt_parent(parent, *t);                   
}
aststmt BarrierCall()
{
char label[23], mabel[23];
if (!cur_parallel_line && !cur_taskgroup_line)
return Call0_stmt("ort_barrier_me");
if (cur_parallel_line && !cur_taskgroup_line)
{
snprintf(label, 23, "CANCEL_parallel_%d", cur_parallel_line);
return If(BinaryOperator(BOP_eqeq, 
Call0_expr("ort_barrier_me"), numConstant(1)), 
Goto(Symbol(label)), 
NULL);
}
if (!cur_parallel_line && cur_taskgroup_line)
{
snprintf(label, 23, "CANCEL_taskgroup_%d", cur_taskgroup_line);
return If(BinaryOperator(BOP_eqeq, 
Call0_expr("ort_barrier_me"), numConstant(2)), 
Goto(Symbol(label)), 
NULL);
}
snprintf(label, 23, "CANCEL_parallel_%d", cur_parallel_line);
snprintf(mabel, 23, "CANCEL_taskgroup_%d", cur_taskgroup_line);
Switch(Call0_expr("ort_barrier_me"),
Compound(
Block3(
Case(numConstant(0), Break()), 
Case(numConstant(1), Goto(Symbol(label))),
Case(numConstant(2), Goto(Symbol(mabel)))
)
)
);
}
void xform_barrier(aststmt *t)
{
aststmt parent = (*t)->parent, v;
v = ompdir_commented((*t)->u.omp->directive);
ast_free(*t);
*t = BlockList(v, BarrierCall());
ast_stmt_parent(parent, *t);
}
void xform_taskwait(aststmt *t)
{
aststmt parent = (*t)->parent, v;
v = ompdir_commented((*t)->u.omp->directive);
ast_free(*t);
*t = BlockList(v, FuncCallStmt(IdentName("ort_taskwait"),
numConstant(0)));
ast_stmt_parent(parent, *t);
}
void xform_taskyield(aststmt *t)
{
aststmt parent = (*t)->parent, v;
v = ompdir_commented((*t)->u.omp->directive);
ast_free(*t);
*t = v;            
ast_stmt_parent(parent, *t);
}
static void cancel_error(aststmt *t, char *ctype, char *etype)
{
exit_error(1, "(%s, line %d) openmp error:\n\t"
"\"cancel %s\" must be closely nested inside a \"%s\" construct\n",
(*t)->u.omp->directive->file->name, (*t)->u.omp->directive->l,
ctype, etype);
}
void xform_cancel_type(aststmt *t, int *type, char label[22])
{
ompclause c;
ompcon    enclosing_con = ast_get_enclosing_ompcon((*t)->parent, 0);
if ((c = xc_ompcon_get_clause((*t)->u.omp, OCPARALLEL)) != NULL)
{
if (enclosing_con->type != DCPARALLEL)
cancel_error(t, "parallel", "parallel");
*type = 0;
}
else
if ((c = xc_ompcon_get_clause((*t)->u.omp, OCTASKGROUP)) != NULL)
{
if (enclosing_con->type != DCTASK)
cancel_error(t, "taskgroup", "task");
*type = 1;
}
else
if ((c = xc_ompcon_get_clause((*t)->u.omp, OCFOR)) != NULL)
{
if (enclosing_con->type != DCFOR && enclosing_con->type != DCFOR_P)
cancel_error(t, "for", "for");
*type = 2;
}
else
{
c = xc_ompcon_get_clause((*t)->u.omp, OCSECTIONS);
if (enclosing_con->type != DCSECTIONS)
cancel_error(t, "sections", "sections");
*type = 3;
}
assert(c != NULL);
snprintf(label, 22, "CANCEL_%s_%d", ompdirnames[enclosing_con->type],
enclosing_con->l);
}
void xform_cancel(aststmt *t)
{
astexpr   ifexpr = NULL;
aststmt   parent = (*t)->parent, v;
int       type = -1;
char      label[22];
ompclause c;
if ((c = xc_ompcon_get_clause((*t)->u.omp, OCIF)) != NULL)
ifexpr = ast_expr_copy(c->u.expr);
xform_cancel_type(t, &type, label);
v = ompdir_commented((*t)->u.omp->directive);
ast_free(*t);
if (ifexpr == NULL)
*t = BlockList(v,
If(
FunctionCall(
IdentName("ort_enable_cancel"), numConstant(type)
),
Goto(Symbol(label)),
NULL
)
);
else
*t = BlockList(v,
If(ifexpr,
If(
FunctionCall(
IdentName("ort_enable_cancel"), numConstant(type)
),
Goto(Symbol(label)),
NULL
),
If(
FunctionCall(
IdentName("ort_check_cancel"), numConstant(type)
),
Goto(Symbol(label)),
NULL
)
)
);
ast_stmt_parent(parent, *t);
}
void xform_cancellationpoint(aststmt *t)
{
aststmt   parent = (*t)->parent, v;
int       type = -1;
char      label[22];
xform_cancel_type(t, &type, label);
v = ompdir_commented((*t)->u.omp->directive);
ast_free(*t);
*t = BlockList(v,
If(
FunctionCall(
IdentName("ort_check_cancel"), numConstant(type)
),
Goto(Symbol(label)),
NULL
)
);
ast_stmt_parent(parent, *t);
}
static
void declare_private_dataclause_vars(ompclause t)
{
if (t == NULL) return;
if (t->type == OCLIST)
{
if (t->u.list.next != NULL)
declare_private_dataclause_vars(t->u.list.next);
assert((t = t->u.list.elem) != NULL);
}
switch (t->type)
{
case OCPRIVATE:
case OCFIRSTPRIVATE:
case OCLASTPRIVATE:
case OCCOPYPRIVATE:
case OCCOPYIN:
ast_declare_varlist_vars(t->u.varlist, t->type, t->subtype);
break;
case OCREDUCTION:
case OCMAP:
ast_declare_xlist_vars(t);
break;
}
}
static
void xform_ompcon_body(ompcon t)
{
if (t->body->type == COMPOUND && t->body->body == NULL)
return;      
xc_validate_only_dataclause_vars(t->directive);  
scope_start(stab);
if (t->type == DCTARGETDATA || t->type == DCTARGET)
{
aststmt tmp = get_denv_var_decl(true);
ast_declare_decllist_vars(tmp->u.declaration.spec, tmp->u.declaration.decl);
}
declare_private_dataclause_vars(t->directive->clauses);
ast_stmt_xform(&(t->body));     
scope_end(stab);
}
int empty_bodied_omp(aststmt *t, char *conname)
{
if ((*t)->u.omp->body->type == COMPOUND && (*t)->u.omp->body->body == NULL)
{
ast_stmt_free(*t);
*t = verbit("", (*t)->l, conname);
return (1);
}
return (0);
}
void ast_omp_xform(aststmt *t)
{
int combined = 0;   
xc_validate_clauses((*t)->u.omp->type, (*t)->u.omp->directive->clauses);
switch ((*t)->u.omp->type)
{
case DCTHREADPRIVATE:
xform_threadprivate(t);
break;
case DCATOMIC:
ast_stmt_xform(&((*t)->u.omp->body));     
xform_atomic(t);
break;
case DCBARRIER:
xform_barrier(t);
break;
case DCTASKWAIT:
xform_taskwait(t);
break;
case DCTASKYIELD:
xform_taskyield(t);
break;
case DCCRITICAL:
ast_stmt_xform(&((*t)->u.omp->body));     
xform_critical(t);
break;
case DCFLUSH:
xform_flush(t);
break;
case DCMASTER:
ast_stmt_xform(&((*t)->u.omp->body));     
xform_master(t);
break;
case DCORDERED:
ast_stmt_xform(&((*t)->u.omp->body));     
xform_ordered(t);
break;
case DCTASK:
if (!empty_bodied_omp(t, "task"))
{
xform_ompcon_body((*t)->u.omp);
xform_task(t, taskoptLevel);
}
break;
case DCTASKGROUP:
if (!empty_bodied_omp(t, "taskgroup"))
{
int save_tgl = cur_taskgroup_line;
cur_taskgroup_line = (*t)->l;    
xform_ompcon_body((*t)->u.omp);
xform_taskgroup(t);
cur_taskgroup_line = save_tgl;   
}
break;
case DCSINGLE:
xform_ompcon_body((*t)->u.omp);
xform_single(t);
break;
case DCSECTIONS:
xform_ompcon_body((*t)->u.omp);
xform_sections(t);
break;
case DCSECTION:
ast_ompcon_xform(&((*t)->u.omp));
break;
case DCFOR:
case DCFOR_P:       
xform_ompcon_body((*t)->u.omp);
xform_for(t);
break;
case DCPARSECTIONS:
case DCPARFOR:
{
ompclause       pc = NULL, wc = NULL;
enum dircontype type = ((*t)->u.omp->type == DCPARFOR ?
DCFOR_P : DCSECTIONS);  
if (empty_bodied_omp(t, "combined parallel"))
break;
combined = 1;
if ((*t)->u.omp->directive->clauses)
xc_split_combined_clauses((*t)->u.omp->directive->clauses, &pc, &wc);
ast_ompclause_free((*t)->u.omp->directive->clauses);   
(*t)->u.omp->type = DCPARALLEL;                   
(*t)->u.omp->directive->type = DCPARALLEL;
(*t)->u.omp->directive->clauses = pc;
(*t)->u.omp->body = OmpStmt(                      
OmpConstruct(
type,
OmpDirective(type, wc),
(*t)->u.omp->body
)
);
(*t)->u.omp->body->file =                    
(*t)->u.omp->body->u.omp->file =
(*t)->u.omp->body->u.omp->directive->file =
(*t)->u.omp->directive->file;
(*t)->u.omp->body->l =
(*t)->u.omp->body->u.omp->l =
(*t)->u.omp->body->u.omp->directive->l =
(*t)->u.omp->directive->l;
(*t)->u.omp->body->c =
(*t)->u.omp->body->u.omp->c =
(*t)->u.omp->body->u.omp->directive->c =
(*t)->u.omp->directive->c;
ast_stmt_parent((*t)->parent, (*t));   
}
case DCPARALLEL:
{
int savescope = closest_parallel_scope;
int savecpl   = cur_parallel_line;
int savectgl  = cur_taskgroup_line;
if (empty_bodied_omp(t, "parallel"))
break;
cur_parallel_line = (*t)->l;
cur_taskgroup_line = 0; 
if (enableAutoscope)
if (dfa_parreg_get_results(cur_parallel_line) == NULL)
dfa_analyse(*t);
closest_parallel_scope = stab->scopelevel;
xform_ompcon_body((*t)->u.omp);
xform_parallel(t, combined);     
if (enableAutoscope)
dfa_parreg_remove(cur_parallel_line);
closest_parallel_scope = savescope;
cur_parallel_line = savecpl;
cur_taskgroup_line = savectgl;
break;
}
case DCTARGET:
{
int savecpl   = cur_parallel_line;
int savectgl  = cur_taskgroup_line;
targstats_t *ts = NULL; 
if (empty_bodied_omp(t, "target"))
break;
cur_parallel_line = 0;  
cur_taskgroup_line = 0; 
#define CHECKTARGET(TYPE) do {\
if (inTarget())\
exit_error(1, "(%s, line %d) openmp error:\n\t"\
TYPE " within a target leads to undefined behavior\n",\
(*t)->u.omp->directive->file->name,\
(*t)->u.omp->directive->l);\
\
if (inDeclTarget)\
exit_error(1, "(%s, line %d) openmp error:\n\t"\
TYPE " within a declare target leads to undefined behavior\n",\
(*t)->u.omp->directive->file->name,\
(*t)->u.omp->directive->l);\
} while(0)
CHECKTARGET("target");
targtree = verbit("");    
if (analyzeKernels)         
ts = cars_analyze_target((*t)->u.omp->body);  
xform_ompcon_body((*t)->u.omp);
xform_target(t, ts);
cur_parallel_line = savecpl;
cur_taskgroup_line = savectgl;
break;
}
case DCTARGETDATA:
{
int bak;
if (empty_bodied_omp(t, "target data"))
break;
CHECKTARGET("target data");
bak = target_data_scope;  
target_data_scope = stab->scopelevel;
xform_ompcon_body((*t)->u.omp);
xform_targetdata(t);
target_data_scope = bak;  
break;
}
case DCTARGETUPD:
{
CHECKTARGET("target update");
xform_targetupdate(t);
break;
}
case DCTARGENTERDATA:
CHECKTARGET("target enter data");
xform_targetenterdata(t);
break;
case DCTARGEXITDATA:
CHECKTARGET("target exit data");
#undef CHECKTARGET
xform_targetexitdata(t);
break;
case DCDECLTARGET:
{
if (!(*t)->u.omp->directive->clauses)     
{ 
if (empty_bodied_omp(t, "declare target"))
break;
inDeclTarget++;
ast_stmt_xform(&((*t)->u.omp->body));   
inDeclTarget--;
}
xform_declaretarget(t);
break;
}
case DCCANCEL:
xform_cancel(t);
break;
case DCCANCELLATIONPOINT:
xform_cancellationpoint(t);
break;
case DCTEAMS:
xform_teams(t);
break;
case DCTARGETTEAMS:
xform_targetteams(t);
break;
default:
fprintf(stderr, "WARNING: %s directive not implemented\n",
ompdirnames[(*t)->u.omp->type]);
ast_ompcon_xform(&((*t)->u.omp));
break;
}
}
void ast_xform(aststmt *tree)
{
if (__has_target && analyzeKernels)
cars_analyze_declared_funcs(*tree);   
decltarg_find_all_directives(*tree);    
ast_stmt_xform(tree);     
bld_outfuncs_xform();     
bld_outfuncs_place();     
sgl_fix_sglvars();        
produce_decl_var_code();  
bld_ctors_build();        
bld_headtail_place(tree); 
}

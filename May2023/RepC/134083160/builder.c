#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include "ompi.h"
#include "ast_xform.h"
#include "x_clauses.h"
#include "x_target.h"
#include "x_decltarg.h"
static aststmt build_head = NULL, 
build_tail = NULL; 
static void aststmt_add(aststmt *to, aststmt s)
{
if (*to == NULL)
*to = s;
else
{
aststmt t = *to;
*to = BlockList(t, s);
t->parent = *to;          
s->parent = *to;
}
}
void bld_head_add(aststmt s)
{
aststmt_add(&build_head, s);
}
void bld_tail_add(aststmt s)
{
aststmt_add(&build_tail, s);
}
stentry bld_globalvar_add(aststmt s)
{
astdecl decl;
stentry e;
bld_head_add(s);
assert(s->type == DECLARATION);    
decl = s->u.declaration.decl;
e = symtab_insert_global(stab, decl_getidentifier_symbol(decl), IDNAME);
if (decl->type == DINIT)
decl = (e->idecl = decl)->decl;
e->decl       = decl;
e->spec       = s->u.declaration.spec;
e->isarray    = (decl_getkind(decl) == DARRAY);
e->isthrpriv  = false;
e->pval       = NULL;
e->scopelevel = 0;
if (inTarget() || inDeclTarget)
{
decltarg_inject_newglobal(e->key);
e->isindevenv = due2DECLTARG; 
}
return (e);
}
static aststmt head_place(aststmt tree)
{
aststmt p;
if (testingmode)
{
for (p = tree; p->type == STATEMENTLIST; p = p->u.next)
;               
p = p->parent;    
p->u.next = BlockList(p->u.next, build_head);
p->u.next->parent       = p;           
p->u.next->body->parent = p->u.next;
build_head->parent      = p->u.next;
return (tree);
}
if (tree->type == STATEMENTLIST)
{
if ((p = head_place(tree->u.next)) != NULL)
return (p);
else
return (head_place(tree->body));
}
if (tree->type == DECLARATION &&
tree->u.declaration.decl != NULL &&
decl_getidentifier(tree->u.declaration.decl)->u.id ==
Symbol("__ompi_defs__"))
{
p = smalloc(sizeof(struct aststmt_));
*p = *tree;
if (cppLineNo)
*tree = *Block4(
p,
verbit("# 1 \"%s-newglobals\"", filename),
build_head,
verbit("# 1 \"%s\"", filename)
);
else
*tree = *BlockList(p, build_head);
tree->parent = p->parent;
p->parent = tree;
build_head->parent = tree;
return (tree);
}
return (NULL);
}
void bld_headtail_place(aststmt *tree)
{
if (tree != NULL && build_head != NULL)
head_place(*tree);
if (tree != NULL && build_tail != NULL)  
*tree = BlockList(*tree, build_tail);  
if (build_head != NULL)
build_head = NULL;
if (build_tail != NULL);
build_tail = NULL;
}
typedef struct funclist_ *funclist;
struct funclist_
{
aststmt  funcdef;        
aststmt  fromfunc;       
symbol   fname;          
funclist next;
};
static funclist outfuncs = NULL;
void bld_outfuncs_add(symbol name, aststmt fd, aststmt curfunc)
{
funclist e   = (funclist) smalloc(sizeof(struct funclist_));
e->fname     = name;
e->funcdef   = fd;
e->fromfunc  = curfunc;
e->next      = outfuncs;
outfuncs     = e;
}
static void outfuncs_xform(funclist l)
{
for (; l != NULL; l = l->next)
ast_stmt_xform(&(l->funcdef));
}
void bld_outfuncs_xform() 
{ 
outfuncs_xform(outfuncs); 
}
static void outfuncs_place(funclist l)
{
aststmt neu, bl;
funclist nl;
if (l == NULL) return;
neu = (aststmt) smalloc(sizeof(struct aststmt_));
*neu = *(bl = l->fromfunc);                     
*(bl) = *Block3(                                
Declaration(                          
Speclist_right(StClassSpec(SPEC_static), Declspec(SPEC_void)),
Declarator(
Pointer(),
FuncDecl(
IdentifierDecl(l->fname) ,
ParamDecl(
Declspec(SPEC_void),
AbstractDeclarator(
Pointer(),
NULL
)
)
)
)
),
neu, 
l->funcdef  
);
bl->parent         = neu->parent;   
neu->parent        = bl->body;      
l->funcdef->parent = bl->body;      
bl->u.next->parent = bl;            
bl->body->parent   = bl;            
bl->file           = NULL;
bl->u.next->file   = NULL;
bl->body->file     = NULL;
l->funcdef->file   = NULL;
symtab_get(stab, decl_getidentifier_symbol(neu->u.declaration.decl), 
FUNCNAME)->funcdef = neu;
for (nl=l; nl; nl = nl->next)
if (nl->fromfunc == bl)
nl->fromfunc = neu;
if (l->next != NULL)
outfuncs_place(l->next);
free(l);    
}
void bld_outfuncs_place()
{
outfuncs_place(outfuncs);
outfuncs = NULL;
}
static aststmt ortinits, autoinits;
void bld_ortinits_add(aststmt st)
{
ortinits = ortinits ? BlockList(ortinits, st) : st;
}
void bld_autoinits_add(aststmt st)
{
autoinits = autoinits ? BlockList(autoinits, st) : st;
}
void bld_ctors_build()
{
stentry e;
aststmt st, l = NULL;
astexpr initer;
struct timeval ts;
char    funcname[32];
if (!ortinits && !autoinits) return;
gettimeofday(&ts, NULL); 
if (ortinits)
{
sprintf(funcname,"_ompi_init_%X%X_",(unsigned)ts.tv_sec,(unsigned)ts.tv_usec);
ortinits = 
FuncDef(
Declspec(SPEC_void),
Declarator(
NULL,
FuncDecl(
IdentifierDecl(Symbol(funcname)),
ParamDecl(Declspec(SPEC_void), NULL)
)
),
NULL, 
Compound(ortinits)
);
st = FuncCallStmt(Identifier(Symbol("ort_initreqs_add")), 
Identifier(Symbol(funcname)));
autoinits = autoinits ? BlockList(st, autoinits) : st;
}
sprintf(funcname,"_ompi_ctor_%X%X_",(unsigned)ts.tv_sec,(unsigned)ts.tv_usec);
autoinits = 
FuncDef(
Declspec(SPEC_void),
Declarator(
NULL,
FuncDecl(
IdentifierDecl(Symbol(funcname)),
ParamDecl(Declspec(SPEC_void), NULL)
)
),
NULL, 
Compound(autoinits)
);
autoinits = 
BlockList(
verbit("#ifdef __SUNPRO_C\n"
"  #pragma init(%s)\n"
"#else \n"  
"  static void __attribute__ ((constructor)) %s(void);\n"
"#endif\n", funcname, funcname),
autoinits
);
if (ortinits)
autoinits = BlockList(ortinits, autoinits);
bld_tail_add(autoinits);   
}

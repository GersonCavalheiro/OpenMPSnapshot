#include "cxx-pragma.h"
#include "cxx-nodecl-output.h"
#include "cxx-ast.h"
#include "cxx-tltype.h"
#include "cxx-scope.h"
#include "string_utils.h"
#include <stdio.h>
static void common_build_scope_pragma_custom_clause_argument(AST a, 
const decl_context_t* decl_context UNUSED_PARAMETER,
nodecl_t *nodecl_output)
{
*nodecl_output = nodecl_make_pragma_clause_arg(ASTText(a), ast_get_locus(a));
}
static void common_build_scope_pragma_custom_clause(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
nodecl_t nodecl_argument = nodecl_null();
if (ASTSon0(a) != NULL)
{
common_build_scope_pragma_custom_clause_argument(ASTSon0(a), decl_context, &nodecl_argument);
nodecl_argument = nodecl_make_list_1(nodecl_argument);
}
*nodecl_output = nodecl_make_pragma_custom_clause(nodecl_argument, strtolower(ASTText(a)), ast_get_locus(a));
}
void common_build_scope_pragma_custom_line(
AST start_clauses, 
AST end_clauses,
const decl_context_t* decl_context, 
nodecl_t* nodecl_output)
{
nodecl_t nodecl_clauses = nodecl_null();
nodecl_t nodecl_parameter = nodecl_null();
if (start_clauses != NULL)
{
AST list, iter;
list = ASTSon0(start_clauses);
if (list != NULL)
{
for_each_element(list, iter)
{
AST pragma_clause = ASTSon1(iter);
nodecl_t nodecl_clause = nodecl_null();
common_build_scope_pragma_custom_clause(pragma_clause, decl_context, &nodecl_clause);
nodecl_clauses = nodecl_append_to_list(nodecl_clauses, nodecl_clause);
}
}
AST parameter = ASTSon1(start_clauses);
if (parameter != NULL)
{
common_build_scope_pragma_custom_clause_argument(parameter, decl_context, &nodecl_parameter);
nodecl_parameter = nodecl_make_list_1(nodecl_parameter);
}
}
nodecl_t nodecl_end_clauses = nodecl_null();
if(end_clauses != NULL) 
{    
AST list, iter;
list = ASTSon0(end_clauses);
if(list != NULL)
{
for_each_element(list, iter)
{
AST pragma_clause = ASTSon1(iter);
nodecl_t nodecl_end_clause = nodecl_null();
common_build_scope_pragma_custom_clause(pragma_clause, decl_context, &nodecl_end_clause);
nodecl_end_clauses = nodecl_append_to_list(nodecl_end_clauses, nodecl_end_clause);
}
}
}
*nodecl_output = nodecl_make_pragma_custom_line(nodecl_parameter, nodecl_clauses, nodecl_end_clauses, 
strtolower(ASTText(start_clauses)), 
ast_get_locus(start_clauses));
}
void common_build_scope_pragma_custom_statement(AST a, 
const decl_context_t* decl_context, 
nodecl_t* nodecl_output,
nodecl_t* nodecl_pragma_line,
void (*function_for_child)(AST, const decl_context_t* decl_context, nodecl_t*, void* info),
void* info)
{
common_build_scope_pragma_custom_line(ASTSon0(a), ASTSon2(a), decl_context, nodecl_pragma_line);
nodecl_t nodecl_child = nodecl_null();
function_for_child(ASTSon1(a), decl_context, &nodecl_child, info);
*nodecl_output = nodecl_make_pragma_custom_statement(*nodecl_pragma_line, nodecl_child, strtolower(ASTText(a)), ast_get_locus(a));
}
void common_build_scope_pragma_custom_directive(AST a, 
const decl_context_t* decl_context, 
nodecl_t* nodecl_output)
{
nodecl_t nodecl_pragma_line = nodecl_null();
common_build_scope_pragma_custom_line(ASTSon0(a),  NULL, decl_context, &nodecl_pragma_line);
nodecl_t nodecl_pragma_context = nodecl_make_pragma_context(decl_context, ast_get_locus(a));
*nodecl_output = nodecl_make_pragma_custom_directive(nodecl_pragma_line, nodecl_pragma_context, strtolower(ASTText(a)), ast_get_locus(a));
}

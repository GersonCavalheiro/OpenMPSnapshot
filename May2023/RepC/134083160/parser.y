%{
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <stdarg.h>
#include <ctype.h>
#include <assert.h>
#include "scanner.h"
#include "ompi.h"
#include "ast.h"
#include "symtab.h"
#include "ast_free.h"
#include "ast_copy.h"
#include "ast_vars.h"
#include "ast_print.h"
#include "x_arith.h"
#include "x_clauses.h"
#include "str.h"
void    check_uknown_var(char *name);
void    parse_error(int exitvalue, char *format, ...);
void    parse_warning(char *format, ...);
void    yyerror(const char *s);
void    check_for_main_and_declare(astspec s, astdecl d);
void    add_declaration_links(astspec s, astdecl d);
astdecl fix_known_typename(astspec s);
void    check_schedule(ompclsubt_e sched, ompclmod_e mod);
char    *strdupcat(char *first, char *second, int freethem);
aststmt pastree = NULL;       
aststmt pastree_stmt = NULL;  
astexpr pastree_expr = NULL;  
int     checkDecls = 1;       
int     tempsave;
int     isTypedef  = 0;       
char    *parsingstring;       
int     __has_target = 0;
int     errorOnReturn = 0;
%}
%union {
char      name[2048];  
int       type;        
char     *string;      
symbol    symb;        
astexpr   expr;        
astspec   spec;        
astdecl   decl;        
aststmt   stmt;        
asmop     asmo;        
ompcon    ocon;        
ompdir    odir;        
ompclause ocla;        
omparrdim oasd;        
ompxli    oxli;        
oxcon     xcon;        
oxdir     xdir;
oxclause  xcla;
}
%right NOELSE ELSE  
%token START_SYMBOL_EXPRESSION START_SYMBOL_BLOCKLIST START_SYMBOL_TRANSUNIT
%type <type> start_trick
%token <name> IDENTIFIER TYPE_NAME CONSTANT STRING_LITERAL
%token <name> PTR_OP INC_OP DEC_OP LEFT_OP RIGHT_OP LE_OP GE_OP EQ_OP NE_OP
%token <name> AND_OP OR_OP MUL_ASSIGN DIV_ASSIGN MOD_ASSIGN ADD_ASSIGN
%token <name> SUB_ASSIGN LEFT_ASSIGN RIGHT_ASSIGN AND_ASSIGN
%token <name> XOR_ASSIGN OR_ASSIGN SIZEOF
%token <name> TYPEDEF EXTERN STATIC AUTO REGISTER RESTRICT
%token <name> CHAR SHORT INT LONG SIGNED UNSIGNED FLOAT DOUBLE
%token <name> CONST VOLATILE VOID INLINE
%token <name> UBOOL UCOMPLEX UIMAGINARY
%token <name> STRUCT UNION ENUM ELLIPSIS
%token <name> CASE DEFAULT IF ELSE SWITCH WHILE DO FOR
%token <name> GOTO CONTINUE BREAK RETURN
%token <name> __BUILTIN_VA_ARG __BUILTIN_OFFSETOF __BUILTIN_TYPES_COMPATIBLE_P
__ATTRIBUTE__ __ASM__ PRAGMA_OTHER
%token <name> PRAGMA_OMP PRAGMA_OMP_THREADPRIVATE OMP_PARALLEL OMP_SECTIONS
%token <name> OMP_NOWAIT OMP_ORDERED OMP_SCHEDULE OMP_STATIC OMP_DYNAMIC
%token <name> OMP_GUIDED OMP_RUNTIME OMP_AUTO OMP_SECTION OMP_AFFINITY
%token <name> OMP_SINGLE OMP_MASTER OMP_CRITICAL OMP_BARRIER OMP_ATOMIC
%token <name> OMP_FLUSH OMP_PRIVATE OMP_FIRSTPRIVATE
%token <name> OMP_LASTPRIVATE OMP_SHARED OMP_DEFAULT OMP_NONE OMP_REDUCTION
%token <name> OMP_COPYIN OMP_NUMTHREADS OMP_COPYPRIVATE OMP_FOR OMP_IF
%token <name> OMP_TASK OMP_UNTIED OMP_TASKWAIT OMP_COLLAPSE
%token <name> OMP_FINAL OMP_MERGEABLE OMP_TASKYIELD OMP_READ OMP_WRITE
%token <name> OMP_CAPTURE OMP_UPDATE OMP_MIN OMP_MAX
%token <name> OMP_PROCBIND OMP_CLOSE OMP_SPREAD OMP_SIMD OMP_INBRANCH
%token <name> OMP_NOTINBRANCH OMP_UNIFORM OMP_LINEAR OMP_ALIGNED OMP_SIMDLEN
%token <name> OMP_SAFELEN OMP_DECLARE OMP_TARGET OMP_DATA OMP_DEVICE OMP_MAP
%token <name> OMP_ALLOC OMP_TO OMP_FROM OMP_TOFROM OMP_END OMP_TEAMS
%token <name> OMP_DISTRIBUTE OMP_NUMTEAMS OMP_THREADLIMIT OMP_DISTSCHEDULE
%token <name> OMP_DEPEND OMP_IN OMP_OUT OMP_INOUT OMP_TASKGROUP OMP_SEQ_CST
%token <name> OMP_CANCEL OMP_INITIALIZER PRAGMA_OMP_CANCELLATIONPOINT
%token <name> OMP_HINT OMP_SOURCE OMP_SINK OMP_RELEASE OMP_DELETE OMP_ALWAYS
%token <name> OMP_ENTER OMP_EXIT OMP_IS_DEVICE_PTR OMP_USE_DEVICE_PTR
%token <name> OMP_PRIORITY OMP_TASKLOOP OMP_THREADS OMP_LINK OMP_DEFAULTMAP
%token <name> OMP_SCALAR OMP_MONOTONIC OMP_NONMONOTONIC
%token <name> OMP_PRIMARY
%type <symb>   enumeration_constant
%type <string> string_literal
%type <expr>   primary_expression
%type <expr>   postfix_expression
%type <expr>   argument_expression_list
%type <expr>   unary_expression
%type <type>   unary_operator
%type <expr>   cast_expression
%type <expr>   multiplicative_expression
%type <expr>   additive_expression
%type <expr>   shift_expression
%type <expr>   relational_expression
%type <expr>   equality_expression
%type <expr>   AND_expression
%type <expr>   exclusive_OR_expression
%type <expr>   inclusive_OR_expression
%type <expr>   logical_AND_expression
%type <expr>   logical_OR_expression
%type <expr>   conditional_expression
%type <expr>   assignment_expression
%type <type>   assignment_operator
%type <expr>   expression
%type <expr>   constant_expression
%type <stmt>   declaration
%type <spec>   declaration_specifiers
%type <decl>   init_declarator_list
%type <decl>   init_declarator
%type <spec>   storage_class_specifier
%type <spec>   type_specifier
%type <spec>   struct_or_union_specifier
%type <type>   struct_or_union
%type <decl>   struct_declaration_list
%type <decl>   struct_declaration
%type <spec>   specifier_qualifier_list
%type <decl>   struct_declarator_list
%type <decl>   struct_declarator
%type <spec>   enum_specifier
%type <spec>   enumerator_list
%type <spec>   enumerator
%type <spec>   type_qualifier
%type <spec>   function_specifier
%type <decl>   declarator
%type <decl>   direct_declarator
%type <spec>   pointer
%type <spec>   type_qualifier_list
%type <decl>   parameter_type_list
%type <decl>   parameter_list
%type <decl>   parameter_declaration
%type <decl>   identifier_list
%type <decl>   type_name
%type <decl>   abstract_declarator
%type <decl>   direct_abstract_declarator
%type <symb>   typedef_name
%type <expr>   initializer
%type <expr>   initializer_list
%type <expr>   designation
%type <expr>   designator_list
%type <expr>   designator
%type <stmt>   statement
%type <stmt>   statement_for_labeled
%type <stmt>   labeled_statement
%type <stmt>   compound_statement
%type <stmt>   block_item_list
%type <stmt>   block_item
%type <stmt>   expression_statement
%type <stmt>   selection_statement
%type <stmt>   iteration_statement
%type <stmt>   iteration_statement_for
%type <stmt>   jump_statement
%type <stmt>   translation_unit
%type <stmt>   external_declaration
%type <stmt>   function_definition
%type <stmt>   normal_function_definition
%type <stmt>   oldstyle_function_definition
%type <stmt>   declaration_list
%type <expr>   labellist
%type <expr>   asm_clobbers
%type <asmo>   asm_inoperand
%type <asmo>   asm_input
%type <asmo>   asm_outoperand
%type <asmo>   asm_output
%type <spec>   asm_qualifier
%type <spec>   asm_qualifiers
%type <stmt>   asm_stmtrest
%type <stmt>   asm_statement
%type <spec>   attribute_optseq
%type <spec>   attribute_seq
%type <spec>   attribute
%type <string> attribute_name_list
%type <string> attribute_name
%type <string> attr_name
%type <ocon>   openmp_construct
%type <ocon>   openmp_directive
%type <stmt>   structured_block
%type <ocon>   parallel_construct
%type <odir>   parallel_directive
%type <ocla>   parallel_clause_optseq
%type <ocla>   parallel_clause
%type <ocla>   unique_parallel_clause
%type <ocon>   for_construct
%type <ocla>   for_clause_optseq
%type <odir>   for_directive
%type <ocla>   for_clause
%type <ocla>   unique_for_clause
%type <type>   schedule_kind
%type <ocon>   sections_construct
%type <ocla>   sections_clause_optseq
%type <odir>   sections_directive
%type <ocla>   sections_clause
%type <stmt>   section_scope
%type <stmt>   section_sequence
%type <odir>   section_directive
%type <ocon>   single_construct
%type <ocla>   single_clause_optseq
%type <odir>   single_directive
%type <ocla>   single_clause
%type <ocon>   parallel_for_construct
%type <ocla>   parallel_for_clause_optseq
%type <odir>   parallel_for_directive
%type <ocla>   parallel_for_clause
%type <ocon>   parallel_sections_construct
%type <ocla>   parallel_sections_clause_optseq
%type <odir>   parallel_sections_directive
%type <ocla>   parallel_sections_clause
%type <ocon>   master_construct
%type <odir>   master_directive
%type <ocon>   critical_construct
%type <odir>   critical_directive
%type <symb>   region_phrase
%type <odir>   barrier_directive
%type <ocon>   atomic_construct
%type <odir>   atomic_directive
%type <ocla>   atomic_clause_opt
%type <odir>   flush_directive
%type <decl>   flush_vars
%type <ocon>   ordered_construct
%type <odir>   ordered_directive_full
%type <odir>   threadprivate_directive
%type <ocla>   procbind_clause
%type <decl>   variable_list
%type <decl>   thrprv_variable_list
%type <ocon>   task_construct
%type <odir>   task_directive
%type <ocla>   task_clause_optseq
%type <ocla>   task_clause
%type <ocla>   unique_task_clause
%type <odir>   taskwait_directive
%type <odir>   taskyield_directive
%type <ocla>   data_default_clause
%type <ocla>   data_privatization_clause
%type <ocla>   data_privatization_in_clause
%type <ocla>   data_privatization_out_clause
%type <ocla>   data_sharing_clause
%type <ocla>   data_reduction_clause
%type <stmt>   declaration_definition
%type <stmt>   function_statement
%type <stmt>   declarations_definitions_seq
%type <ocon>   simd_construct
%type <odir>   simd_directive
%type <ocla>   simd_clause_optseq
%type <ocla>   simd_clause
%type <ocla>   unique_simd_clause
%type <ocon>   declare_simd_construct
%type <odir>   declare_simd_directive
%type <ocla>   declare_simd_clause_optseq
%type <ocla>   declare_simd_clause
%type <ocon>   for_simd_construct
%type <odir>   for_simd_directive
%type <ocla>   for_simd_clause_optseq
%type <ocla>   for_simd_clause
%type <ocon>   parallel_for_simd_construct
%type <odir>   parallel_for_simd_directive
%type <ocla>   parallel_for_simd_clause_optseq
%type <ocla>   parallel_for_simd_clause
%type <ocon>   target_data_construct
%type <odir>   target_data_directive
%type <ocla>   target_data_clause_optseq
%type <ocla>   target_data_clause
%type <ocla>   device_clause
%type <ocla>   map_clause
%type <type>   map_modifier
%type <type>   map_type
%type <ocon>   target_construct
%type <odir>   target_directive
%type <ocla>   target_clause_optseq
%type <ocla>   target_clause
%type <ocla>   unique_target_clause
%type <odir>   target_update_directive
%type <ocla>   target_update_clause_seq
%type <ocla>   target_update_clause
%type <ocla>   motion_clause
%type <ocon>   teams_construct
%type <odir>   teams_directive
%type <ocla>   teams_clause_optseq
%type <ocla>   teams_clause
%type <ocla>   unique_teams_clause
%type <ocon>   distribute_construct
%type <odir>   distribute_directive
%type <ocla>   distribute_clause_optseq
%type <ocla>   distribute_clause
%type <ocla>   unique_distribute_clause
%type <ocon>   distribute_simd_construct
%type <odir>   distribute_simd_directive
%type <ocla>   distribute_simd_clause_optseq
%type <ocla>   distribute_simd_clause
%type <ocon>   distribute_parallel_for_construct
%type <odir>   distribute_parallel_for_directive
%type <ocla>   distribute_parallel_for_clause_optseq
%type <ocla>   distribute_parallel_for_clause
%type <ocon>   distribute_parallel_for_simd_construct
%type <odir>   distribute_parallel_for_simd_directive
%type <ocla>   distribute_parallel_for_simd_clause_optseq
%type <ocla>   distribute_parallel_for_simd_clause
%type <ocon>   target_teams_construct
%type <odir>   target_teams_directive
%type <ocla>   target_teams_clause_optseq
%type <ocla>   target_teams_clause
%type <ocon>   teams_distribute_construct
%type <odir>   teams_distribute_directive
%type <ocla>   teams_distribute_clause_optseq
%type <ocla>   teams_distribute_clause
%type <ocon>   teams_distribute_simd_construct
%type <odir>   teams_distribute_simd_directive
%type <ocla>   teams_distribute_simd_clause_optseq
%type <ocla>   teams_distribute_simd_clause
%type <ocon>   target_teams_distribute_construct
%type <odir>   target_teams_distribute_directive
%type <ocla>   target_teams_distribute_clause_optseq
%type <ocla>   target_teams_distribute_clause
%type <ocon>   target_teams_distribute_simd_construct
%type <odir>   target_teams_distribute_simd_directive
%type <ocla>   target_teams_distribute_simd_clause_optseq
%type <ocla>   target_teams_distribute_simd_clause
%type <ocon>   teams_distribute_parallel_for_construct
%type <odir>   teams_distribute_parallel_for_directive
%type <ocla>   teams_distribute_parallel_for_clause_optseq
%type <ocla>   teams_distribute_parallel_for_clause
%type <ocon>   target_teams_distribute_parallel_for_construct
%type <odir>   target_teams_distribute_parallel_for_directive
%type <ocla>   target_teams_distribute_parallel_for_clause_optseq
%type <ocla>   target_teams_distribute_parallel_for_clause
%type <ocon>   teams_distribute_parallel_for_simd_construct
%type <odir>   teams_distribute_parallel_for_simd_directive
%type <ocla>   teams_distribute_parallel_for_simd_clause_optseq
%type <ocla>   teams_distribute_parallel_for_simd_clause
%type <ocon>   target_teams_distribute_parallel_for_simd_construct
%type <odir>   target_teams_distribute_parallel_for_simd_directive
%type <ocla>   target_teams_distribute_parallel_for_simd_clause_optseq
%type <ocla>   target_teams_distribute_parallel_for_simd_clause
%type <ocla>   unique_single_clause
%type <ocla>   aligned_clause
%type <ocla>   linear_clause
%type <expr>   optional_expression
%type <ocla>   uniform_clause
%type <ocla>   inbranch_clause
%type <ocon>   declare_target_construct
%type <odir>   declare_target_directive
%type <odir>   end_declare_target_directive
%type <type>   dependence_type
%type <ocon>   taskgroup_construct
%type <odir>   taskgroup_directive
%type <ocla>   seq_cst_clause_opt
%type <odir>   cancel_directive
%type <odir>   cancellation_point_directive
%type <ocla>   construct_type_clause
%type <type>   reduction_identifier
%type <type>   reduction_type_list 
%type <ocla>   initializer_clause_opt
%type <ocla>   depend_clause
%type <ocla>   if_clause
%type <ocla>   collapse_clause
%type <oxli>   variable_array_section_list
%type <oxli>   varid_or_array_section
%type <oasd>   array_section_slice_list
%type <oasd>   array_section_slice
%type <odir>   ordered_directive_standalone
%type <oxli>   funcname_variable_array_section_list
%type <oxli>   funcvarid_or_array_section
%type <odir>   declare_target_directive_v45
%type <ocla>   declare_target_clause_optseq
%type <ocla>   unique_declare_target_clause
%type <odir>   target_enter_data_directive
%type <ocla>   target_enter_data_clause_seq
%type <ocla>   target_enter_data_clause
%type <odir>   target_exit_data_directive
%type <ocla>   target_exit_data_clause_seq
%type <ocla>   target_exit_data_clause
%type <ocla>   defaultmap_clause
%type <ocla>   use_device_ptr_clause
%type <ocla>   is_device_ptr_clause
%type <type>   if_related_construct
%type <ocla>   hint_clause
%type <ocla>   ordered_clause_optseq_full
%type <ocla>   ordered_clause_type_full
%type <ocla>   ordered_clause_optseq_standalone
%type <ocla>   ordered_clause_depend_sink
%type <type>   schedule_mod
%type <expr>   sink_vec
%type <expr>   sink_vec_elem
%token <name> PRAGMA_OMPIX OMPIX_TASKDEF
%token <name> OMPIX_TASKSYNC OMPIX_UPONRETURN OMPIX_ATNODE OMPIX_DETACHED
%token <name> OMPIX_ATWORKER OMPIX_TASKSCHEDULE OMPIX_STRIDE OMPIX_START
%token <name> OMPIX_SCOPE OMPIX_NODES OMPIX_WORKERS OMPIX_LOCAL OMPIX_GLOBAL
%token <name> OMPIX_HERE OMPIX_REMOTE OMPIX_HINTS
%token <name> OMPIX_TIED
%type <xcon>   ompix_construct
%type <xcon>   ompix_directive
%type <xcon>   ox_taskdef_construct
%type <xdir>   ox_taskdef_directive
%type <xcla>   ox_taskdef_clause_optseq
%type <xcla>   ox_taskdef_clause
%type <decl>   ox_variable_size_list
%type <decl>   ox_variable_size_elem
%type <xcon>   ox_task_construct
%type <xdir>   ox_task_directive
%type <xcla>   ox_task_clause_optseq
%type <xcla>   ox_task_clause
%type <xdir>   ox_tasksync_directive
%type <xdir>   ox_taskschedule_directive
%type <xcla>   ox_taskschedule_clause_optseq
%type <xcla>   ox_taskschedule_clause
%type <type>   ox_scope_spec
%%
start_trick:
translation_unit                        {  }
| START_SYMBOL_EXPRESSION expression      { pastree_expr = $2; }
| START_SYMBOL_BLOCKLIST block_item_list  { pastree_stmt = $2; }
| START_SYMBOL_TRANSUNIT translation_unit { pastree_stmt = $2; }
;
enumeration_constant:
IDENTIFIER
{
symbol s = Symbol($1);
if (checkDecls)
{
if ( symtab_get(stab, s, LABELNAME) )  
parse_error(-1, "enum symbol '%s' is already in use.", $1);
symtab_put(stab, s, LABELNAME);
}
$$ = s;
}
;
string_literal:
STRING_LITERAL
{
$$ = strdup($1);
}
| string_literal STRING_LITERAL
{
if (($1 = realloc($1, strlen($1) + strlen($2))) == NULL)
parse_error(-1, "string out of memory\n");
strcpy(($1)+(strlen($1)-1),($2)+1);  
$$ = $1;
}
;
primary_expression:
IDENTIFIER
{
symbol  id = Symbol($1);
stentry e;
bool    chflag = false;
if (checkDecls)
{
check_uknown_var($1);
if ((e = symtab_get(stab, id, IDNAME)) != NULL) 
if (istp(e) && threadmode)
chflag = true;
}
$$ = chflag ? Parenthesis(Deref(Identifier(id)))
: Identifier(id);
}
| CONSTANT
{
$$ = Constant( strdup($1) );
}
| string_literal
{
$$ = String($1);
}
| '(' expression ')'
{
$$ = Parenthesis($2);
}
;
postfix_expression:
primary_expression
{
$$ = $1;
}
| postfix_expression '[' expression ']'
{
$$ = ArrayIndex($1, $3);
}
| IDENTIFIER '(' argument_expression_list ')'
{
$$ = strcmp($1, "main") ?
FunctionCall(IdentName($1), $3) :
FunctionCall(IdentName(MAIN_NEWNAME), $3);
}
| postfix_expression '(' argument_expression_list ')'
{
$$ = FunctionCall($1, $3);
}
| postfix_expression '.' IDENTIFIER
{
$$ = DotField($1, Symbol($3));
}
| postfix_expression PTR_OP IDENTIFIER
{
$$ = PtrField($1, Symbol($3));
}
| postfix_expression '.' typedef_name
{
$$ = DotField($1, $3);
}
| postfix_expression PTR_OP typedef_name
{
$$ = PtrField($1, $3);
}
| postfix_expression INC_OP
{
$$ = PostOperator($1, UOP_inc);
}
| postfix_expression DEC_OP
{
$$ = PostOperator($1, UOP_dec);
}
| '(' type_name ')' '{' initializer_list '}'
{
$$ = CastedExpr($2, BracedInitializer($5));
}
| '(' type_name ')' '{' initializer_list ',' '}'
{
$$ = CastedExpr($2, BracedInitializer($5));
}
;
argument_expression_list:
{
$$ = NULL;
}
| assignment_expression
{
$$ = $1;
}
| argument_expression_list ',' assignment_expression
{
$$ = CommaList($1, $3);
}
;
unary_expression:
postfix_expression
{
$$ = $1;
}
| INC_OP unary_expression
{
$$ = PreOperator($2, UOP_inc);
}
| DEC_OP unary_expression
{
$$ = PreOperator($2, UOP_dec);
}
| unary_operator cast_expression
{
if ($1 == -1)
$$ = $2;                    
else
$$ = UnaryOperator($1, $2);
}
| SIZEOF unary_expression
{
$$ = Sizeof($2);
}
| SIZEOF '(' type_name ')'
{
$$ = Sizeoftype($3);
}
| __BUILTIN_VA_ARG '(' assignment_expression ',' type_name ')'
{
$$ = FunctionCall(IdentName("__builtin_va_arg"),
CommaList($3, TypeTrick($5)));
}
| __BUILTIN_OFFSETOF '(' type_name ',' IDENTIFIER ')'
{
$$ = FunctionCall(IdentName("__builtin_offsetof"),
CommaList(TypeTrick($3), IdentName($5)));
}
| __BUILTIN_TYPES_COMPATIBLE_P '(' type_name ',' type_name ')'
{
$$ = FunctionCall(IdentName("__builtin_types_compatible_p"),
CommaList(TypeTrick($3), TypeTrick($5)));
}
;
unary_operator:
'&'
{
$$ = UOP_addr;
}
| '*'
{
$$ = UOP_star;
}
| '+'
{
$$ = -1;         
}
| '-'
{
$$ = UOP_neg;
}
| '~'
{
$$ = UOP_bnot;
}
| '!'
{
$$ = UOP_lnot;
}
;
cast_expression:
unary_expression
{
$$ = $1;
}
| '(' type_name ')' cast_expression
{
$$ = CastedExpr($2, $4);
}
;
multiplicative_expression:
cast_expression
{
$$ = $1;
}
| multiplicative_expression '*' cast_expression
{
$$ = BinaryOperator(BOP_mul, $1, $3);
}
| multiplicative_expression '/' cast_expression
{
$$ = BinaryOperator(BOP_div, $1, $3);
}
| multiplicative_expression '%' cast_expression
{
$$ = BinaryOperator(BOP_mod, $1, $3);
}
;
additive_expression:
multiplicative_expression
{
$$ = $1;
}
| additive_expression '+' multiplicative_expression
{
$$ = BinaryOperator(BOP_add, $1, $3);
}
| additive_expression '-' multiplicative_expression
{
$$ = BinaryOperator(BOP_sub, $1, $3);
}
;
shift_expression:
additive_expression
{
$$ = $1;
}
| shift_expression LEFT_OP additive_expression
{
$$ = BinaryOperator(BOP_shl, $1, $3);
}
| shift_expression RIGHT_OP additive_expression
{
$$ = BinaryOperator(BOP_shr, $1, $3);
}
;
relational_expression:
shift_expression
{
$$ = $1;
}
| relational_expression '<' shift_expression
{
$$ = BinaryOperator(BOP_lt, $1, $3);
}
| relational_expression '>' shift_expression
{
$$ = BinaryOperator(BOP_gt, $1, $3);
}
| relational_expression LE_OP shift_expression
{
$$ = BinaryOperator(BOP_leq, $1, $3);
}
| relational_expression GE_OP shift_expression
{
$$ = BinaryOperator(BOP_geq, $1, $3);
}
;
equality_expression:
relational_expression
{
$$ = $1;
}
| equality_expression EQ_OP relational_expression
{
$$ = BinaryOperator(BOP_eqeq, $1, $3);
}
| equality_expression NE_OP relational_expression
{
$$ = BinaryOperator(BOP_neq, $1, $3);
}
;
AND_expression:
equality_expression
{
$$ = $1;
}
| AND_expression '&' equality_expression
{
$$ = BinaryOperator(BOP_band, $1, $3);
}
;
exclusive_OR_expression:
AND_expression
{
$$ = $1;
}
| exclusive_OR_expression '^' AND_expression
{
$$ = BinaryOperator(BOP_xor, $1, $3);
}
;
inclusive_OR_expression:
exclusive_OR_expression
{
$$ = $1;
}
| inclusive_OR_expression '|' exclusive_OR_expression
{
$$ = BinaryOperator(BOP_bor, $1, $3);
}
;
logical_AND_expression:
inclusive_OR_expression
{
$$ = $1;
}
| logical_AND_expression AND_OP inclusive_OR_expression
{
$$ = BinaryOperator(BOP_land, $1, $3);
}
;
logical_OR_expression:
logical_AND_expression
{
$$ = $1;
}
| logical_OR_expression OR_OP logical_AND_expression
{
$$ = BinaryOperator(BOP_lor, $1, $3);
}
;
conditional_expression:
logical_OR_expression
{
$$ = $1;
}
| logical_OR_expression '?' expression ':' conditional_expression
{
$$ = ConditionalExpr($1, $3, $5);
}
;
assignment_expression:
conditional_expression
{
$$ = $1;
}
| unary_expression assignment_operator assignment_expression
{
$$ = Assignment($1, $2, $3);
}
;
assignment_operator:
'='
{
$$ = ASS_eq;  
}
| MUL_ASSIGN
{
$$ = ASS_mul;
}
| DIV_ASSIGN
{
$$ = ASS_div;
}
| MOD_ASSIGN
{
$$ = ASS_mod;
}
| ADD_ASSIGN
{
$$ = ASS_add;
}
| SUB_ASSIGN
{
$$ = ASS_sub;
}
| LEFT_ASSIGN
{
$$ = ASS_shl;
}
| RIGHT_ASSIGN
{
$$ = ASS_shr;
}
| AND_ASSIGN
{
$$ = ASS_and;
}
| XOR_ASSIGN
{
$$ = ASS_xor;
}
| OR_ASSIGN
{
$$ = ASS_or;
}
;
expression:
assignment_expression
{
$$ = $1;
}
| expression ',' assignment_expression
{
$$ = CommaList($1, $3);
}
;
constant_expression:
conditional_expression
{
$$ = $1;
}
;
declaration:
declaration_specifiers ';'
{
if (isTypedef && $1->type == SPECLIST)
$$ = Declaration($1, fix_known_typename($1));
else
$$ = Declaration($1, NULL);
isTypedef = 0;
}
| declaration_specifiers init_declarator_list ';'
{
$$ = Declaration($1, $2);
if (checkDecls) add_declaration_links($1, $2);
isTypedef = 0;
}
| threadprivate_directive 
{
$$ = OmpStmt(OmpConstruct(DCTHREADPRIVATE, $1, NULL));
}
| 
declare_simd_construct
{
}
| declare_target_construct
{
$$ = OmpStmt($1);
}
| declare_reduction_directive
{
}
;
declaration_specifiers:
storage_class_specifier
{
$$ = $1;
}
| storage_class_specifier declaration_specifiers
{
$$ = Speclist_right($1, $2);
}
| type_specifier
{
$$ = $1;
}
| type_specifier declaration_specifiers
{
$$ = Speclist_right($1, $2);
}
| type_qualifier
{
$$ = $1;
}
| type_qualifier declaration_specifiers
{
$$ = Speclist_right($1, $2);
}
| function_specifier
{
$$ = $1;
}
| function_specifier declaration_specifiers
{
$$ = Speclist_right($1, $2);
}
;
init_declarator_list:
init_declarator
{
$$ = $1;
}
| init_declarator_list ',' init_declarator
{
$$ = DeclList($1, $3);
}
;
init_declarator:
declarator
{
astdecl s = decl_getidentifier($1);
int     declkind = decl_getkind($1);
stentry e;
if (!isTypedef && declkind == DFUNC && strcmp(s->u.id->name, "main") == 0)
s->u.id = Symbol(MAIN_NEWNAME);       
if (checkDecls)
{
e = symtab_put(stab, s->u.id, (isTypedef) ? TYPENAME :
(declkind == DFUNC) ? FUNCNAME : IDNAME);
e->isarray = (declkind == DARRAY);
}
$$ = $1;
}
| declarator '='
{
astdecl s = decl_getidentifier($1);
int     declkind = decl_getkind($1);
stentry e;
if (!isTypedef && declkind == DFUNC && strcmp(s->u.id->name, "main") == 0)
s->u.id = Symbol(MAIN_NEWNAME);         
if (checkDecls)
{
e = symtab_put(stab, s->u.id, (isTypedef) ? TYPENAME :
(declkind == DFUNC) ? FUNCNAME : IDNAME);
e->isarray = (declkind == DARRAY);
}
}
initializer
{
$$ = InitDecl($1, $4);
}
;
storage_class_specifier:
TYPEDEF
{
$$ = StClassSpec(SPEC_typedef);    
isTypedef = 1;
}
| EXTERN
{
$$ = StClassSpec(SPEC_extern);
}
| STATIC
{
$$ = StClassSpec(SPEC_static);
}
| AUTO
{
$$ = StClassSpec(SPEC_auto);
}
| REGISTER
{
$$ = StClassSpec(SPEC_register);
}
;
type_specifier:
VOID
{
$$ = Declspec(SPEC_void);
}
| CHAR
{
$$ = Declspec(SPEC_char);
}
| SHORT
{
$$ = Declspec(SPEC_short);
}
| INT
{
$$ = Declspec(SPEC_int);
}
| LONG
{
$$ = Declspec(SPEC_long);
}
| FLOAT
{
$$ = Declspec(SPEC_float);
}
| DOUBLE
{
$$ = Declspec(SPEC_double);
}
| SIGNED
{
$$ = Declspec(SPEC_signed);
}
| UNSIGNED
{
$$ = Declspec(SPEC_unsigned);
}
| UBOOL
{
$$ = Declspec(SPEC_ubool);
}
| UCOMPLEX
{
$$ = Declspec(SPEC_ucomplex);
}
| UIMAGINARY
{
$$ = Declspec(SPEC_uimaginary);
}
| struct_or_union_specifier
{
$$ = $1;
}
| enum_specifier
{
$$ = $1;
}
| typedef_name
{
$$ = Usertype($1);
}
;
struct_or_union_specifier:
struct_or_union attribute_optseq '{' struct_declaration_list '}'
{
$$ = SUdecl($1, NULL, $4, $2);
}
| struct_or_union attribute_optseq '{' '}' 
{
$$ = SUdecl($1, NULL, NULL, $2);
}
| struct_or_union attribute_optseq IDENTIFIER '{' struct_declaration_list '}'
{
symbol s = Symbol($3);
if (checkDecls)
symtab_put(stab, s, SUNAME);
$$ = SUdecl($1, s, $5, $2);
}
| struct_or_union attribute_optseq typedef_name '{' struct_declaration_list '}'
{
symbol s = $3;
if (checkDecls)
symtab_put(stab, s, SUNAME);
$$ = SUdecl($1, s, $5, $2);
}
| struct_or_union attribute_optseq IDENTIFIER
{
symbol s = Symbol($3);
if (checkDecls)
symtab_put(stab, s, SUNAME);
$$ = SUdecl($1, s, NULL, $2);
}
| struct_or_union attribute_optseq typedef_name       
{
symbol s = $3;
if (checkDecls)
symtab_put(stab, s, SUNAME);
$$ = SUdecl($1, s, NULL, $2);
}
;
struct_or_union:
STRUCT
{
$$ = SPEC_struct;
}
| UNION
{
$$ = SPEC_union;
}
;
struct_declaration_list:
struct_declaration
{
$$ = $1;
}
| struct_declaration_list struct_declaration
{
$$ = StructfieldList($1, $2);
}
;
struct_declaration:
specifier_qualifier_list struct_declarator_list ';'
{
$$ = StructfieldDecl($1, $2);
}
| specifier_qualifier_list ';'        
{
$$ = StructfieldDecl($1, NULL);
}
;
specifier_qualifier_list:
type_specifier
{
$$ = $1;
}
| type_specifier specifier_qualifier_list
{
$$ = Speclist_right($1, $2);
}
| type_qualifier
{
$$ = $1;
}
| type_qualifier specifier_qualifier_list
{
$$ = Speclist_right($1, $2);
}
;
struct_declarator_list:
struct_declarator
{
$$ = $1;
}
| struct_declarator_list ',' struct_declarator
{
$$ = DeclList($1, $3);
}
;
struct_declarator:
declarator
{
$$ = $1;
}
| declarator ':' constant_expression
{
$$ = BitDecl($1, $3);
}
| ':' constant_expression
{
$$ = BitDecl(NULL, $2);
}
;
enum_specifier:
ENUM attribute_optseq '{' enumerator_list '}'
{
$$ = Enumdecl(NULL, $4, $2);
}
| ENUM attribute_optseq IDENTIFIER '{' enumerator_list '}'
{
symbol s = Symbol($3);
if (checkDecls)
{
if (symtab_get(stab, s, ENUMNAME))
parse_error(-1, "enum name '%s' is already in use.", $3);
symtab_put(stab, s, ENUMNAME);
}
$$ = Enumdecl(s, $5, $2);
}
| ENUM attribute_optseq typedef_name '{' enumerator_list '}'
{
symbol s = $3;
if (checkDecls)
{
if (symtab_get(stab, s, ENUMNAME))
parse_error(-1, "enum name '%s' is already in use.", s->name);
symtab_put(stab, s, ENUMNAME);
}
$$ = Enumdecl(s, $5, $2);
}
| ENUM attribute_optseq '{' enumerator_list ',' '}'
{
$$ = Enumdecl(NULL, $4, $2);
}
| ENUM attribute_optseq IDENTIFIER '{' enumerator_list ',' '}'
{
symbol s = Symbol($3);
if (checkDecls)
{
if (symtab_get(stab, s, ENUMNAME))
parse_error(-1, "enum name '%s' is already in use.", $3);
symtab_put(stab, s, ENUMNAME);
}
$$ = Enumdecl(s, $5, $2);
}
| ENUM attribute_optseq typedef_name '{' enumerator_list ',' '}'
{
symbol s = $3;
if (checkDecls)
{
if (symtab_get(stab, s, ENUMNAME))
parse_error(-1, "enum name '%s' is already in use.", s->name);
symtab_put(stab, s, ENUMNAME);
}
$$ = Enumdecl(s, $5, $2);
}
| ENUM attribute_optseq IDENTIFIER
{
$$ = Enumdecl(Symbol($3), NULL, $2);
}
| ENUM attribute_optseq typedef_name
{
$$ = Enumdecl($3, NULL, $2);
}
;
enumerator_list:
enumerator
{
$$ = $1;
}
| enumerator_list ',' enumerator
{
$$ = Enumbodylist($1, $3);
}
;
enumerator:
enumeration_constant
{
$$ = Enumerator($1, NULL);
}
|  enumeration_constant '=' constant_expression
{
$$ = Enumerator($1, $3);
}
;
type_qualifier:
CONST
{
$$ = Declspec(SPEC_const);
}
| RESTRICT
{
$$ = Declspec(SPEC_restrict);
}
| VOLATILE
{
$$ = Declspec(SPEC_volatile);
}
| attribute
{
$$ = $1;
}
;
function_specifier:
INLINE
{
$$ = Declspec(SPEC_inline);
}
;
declarator:
direct_declarator
{
$$ = Declarator(NULL, $1);
}
| pointer direct_declarator
{
$$ = Declarator($1, $2);
}
;
direct_declarator:
IDENTIFIER
{
$$ = IdentifierDecl( Symbol($1) );
}
| '(' declarator ')'
{
if ($2->spec == NULL && $2->decl->type == DIDENT)
$$ = $2->decl;
else
$$ = ParenDecl($2);
}
| direct_declarator '[' ']'
{
$$ = ArrayDecl($1, NULL, NULL);
}
| direct_declarator '[' type_qualifier_list ']'
{
$$ = ArrayDecl($1, $3, NULL);
}
| direct_declarator '[' assignment_expression ']'
{
$$ = ArrayDecl($1, NULL, $3);
}
| direct_declarator '[' type_qualifier_list assignment_expression ']'
{
$$ = ArrayDecl($1, $3, $4);
}
| direct_declarator '[' STATIC assignment_expression ']'
{
$$ = ArrayDecl($1, StClassSpec(SPEC_static), $4);
}
| direct_declarator '[' STATIC type_qualifier_list assignment_expression ']'
{
$$ = ArrayDecl($1, Speclist_right( StClassSpec(SPEC_static), $4 ), $5);
}
| direct_declarator '[' type_qualifier_list STATIC assignment_expression ']'
{
$$ = ArrayDecl($1, Speclist_left( $3, StClassSpec(SPEC_static) ), $5);
}
| direct_declarator '[' '*' ']'
{
$$ = ArrayDecl($1, Declspec(SPEC_star), NULL);
}
| direct_declarator '[' type_qualifier_list '*' ']'
{
$$ = ArrayDecl($1, Speclist_left( $3, Declspec(SPEC_star) ), NULL);
}
| direct_declarator '(' parameter_type_list ')'
{
$$ = FuncDecl($1, $3);
}
| direct_declarator '(' ')'
{
$$ = FuncDecl($1, NULL);
}
| direct_declarator '(' identifier_list ')'
{
$$ = FuncDecl($1, $3);
}
;
pointer:
'*'
{
$$ = Pointer();
}
| '*' type_qualifier_list
{
$$ = Speclist_right(Pointer(), $2);
}
| '*' pointer
{
$$ = Speclist_right(Pointer(), $2);
}
| '*' type_qualifier_list pointer
{
$$ = Speclist_right( Pointer(), Speclist_left($2, $3) );
}
;
type_qualifier_list:
type_qualifier
{
$$ = $1;
}
| type_qualifier_list type_qualifier
{
$$ = Speclist_left($1, $2);
}
;
parameter_type_list:
parameter_list
{
$$ = $1;
}
| parameter_list ',' ELLIPSIS
{
$$ = ParamList($1, Ellipsis());
}
;
parameter_list:
parameter_declaration
{
$$ = $1;
}
| parameter_list ',' parameter_declaration
{
$$ = ParamList($1, $3);
}
;
parameter_declaration:
declaration_specifiers declarator
{
$$ = ParamDecl($1, $2);
}
| declaration_specifiers
{
$$ = ParamDecl($1, NULL);
}
| declaration_specifiers abstract_declarator
{
$$ = ParamDecl($1, $2);
}
;
identifier_list:
IDENTIFIER
{
$$ = IdentifierDecl( Symbol($1) );
}
| identifier_list ',' IDENTIFIER
{
$$ = IdList($1, IdentifierDecl( Symbol($3) ));
}
;
type_name:
specifier_qualifier_list
{
$$ = Casttypename($1, NULL);
}
| specifier_qualifier_list abstract_declarator
{
$$ = Casttypename($1, $2);
}
;
abstract_declarator:
pointer
{
$$ = AbstractDeclarator($1, NULL);
}
| direct_abstract_declarator
{
$$ = AbstractDeclarator(NULL, $1);
}
| pointer direct_abstract_declarator
{
$$ = AbstractDeclarator($1, $2);
}
;
direct_abstract_declarator:
'(' abstract_declarator ')'
{
$$ = ParenDecl($2);
}
| '[' ']'
{
$$ = ArrayDecl(NULL, NULL, NULL);
}
| direct_abstract_declarator '[' ']'
{
$$ = ArrayDecl($1, NULL, NULL);
}
| '[' assignment_expression ']'
{
$$ = ArrayDecl(NULL, NULL, $2);
}
| direct_abstract_declarator '[' assignment_expression ']'
{
$$ = ArrayDecl($1, NULL, $3);
}
| '[' '*' ']'
{
$$ = ArrayDecl(NULL, Declspec(SPEC_star), NULL);
}
| direct_abstract_declarator '[' '*' ']'
{
$$ = ArrayDecl($1, Declspec(SPEC_star), NULL);
}
| '(' ')'
{
$$ = FuncDecl(NULL, NULL);
}
| direct_abstract_declarator '(' ')'
{
$$ = FuncDecl($1, NULL);
}
| '(' parameter_type_list ')'
{
$$ = FuncDecl(NULL, $2);
}
| direct_abstract_declarator '(' parameter_type_list ')'
{
$$ = FuncDecl($1, $3);
}
;
typedef_name:
TYPE_NAME
{
$$ = Symbol($1);
}
;
initializer:
assignment_expression
{
$$ = $1;
}
| '{' initializer_list '}'
{
$$ = BracedInitializer($2);
}
| '{' initializer_list ',' '}'
{
$$ = BracedInitializer($2);
}
;
initializer_list:
initializer
{
$$ = $1;
}
| designation initializer
{
$$ = Designated($1, $2);
}
| initializer_list ',' initializer
{
$$ = CommaList($1, $3);
}
| initializer_list ',' designation initializer
{
$$ = CommaList($1, Designated($3, $4));
}
;
designation:
designator_list '='
{
$$ = $1;
}
;
designator_list:
designator
{
$$ = $1;
}
| designator_list designator
{
$$ = SpaceList($1, $2);
}
;
designator:
'[' constant_expression ']'
{
$$ = IdxDesignator($2);
}
| '.' IDENTIFIER
{
$$ = DotDesignator( Symbol($2) );
}
| '.' typedef_name     
{
$$ = DotDesignator($2);
}
;
statement:
labeled_statement
{
$$ = $1;
}
| compound_statement
{
$$ = $1;
}
| expression_statement
{
$$ = $1;
}
| selection_statement
{
$$ = $1;
}
| iteration_statement
{
$$ = $1;
}
| jump_statement
{
$$ = $1;
}
| asm_statement    
{
$$ = $1;
}
| openmp_construct 
{
$$ = OmpStmt($1);
$$->l = $1->l;
}
| ompix_construct 
{
$$ = OmpixStmt($1);
$$->l = $1->l;
}
| PRAGMA_OTHER
{
$$ = Verbatim(strdup($1));
}    
;
statement_for_labeled:
statement
{ 
$$ = $1; 
}
| openmp_directive 
{       
$$ = OmpStmt($1);
$$->l = $1->l;
}
;
labeled_statement:             
IDENTIFIER ':' attribute_optseq statement  
{
$$ = Labeled( Symbol($1), $4 );
}
| CASE constant_expression ':' statement_for_labeled
{
$$ = Case($2, $4);
}
| DEFAULT ':' statement_for_labeled
{
$$ = Default($3);
}
;
compound_statement:
'{' '}'
{
$$ = Compound(NULL);
}
| '{'  { $<type>$ = sc_original_line()-1; scope_start(stab); }
block_item_list '}'
{
$$ = Compound($3);
scope_end(stab);
$$->l = $<type>2;     
}
;
block_item_list:
block_item
{
$$ = $1;
}
| block_item_list block_item
{
$$ = BlockList($1, $2);
$$->l = $1->l;
}
;
block_item:
declaration
{
$$ = $1;
}
| statement
{
$$ = $1;
}
| openmp_directive 
{
$$ = OmpStmt($1);
$$->l = $1->l;
}
| ompix_directive 
{
$$ = OmpixStmt($1);
$$->l = $1->l;
}
;
expression_statement:
';'
{
$$ = Expression(NULL);
}
| expression ';'
{
$$ = Expression($1);
$$->l = $1->l;
}
;
selection_statement:
IF '(' expression ')' statement  %prec NOELSE 
{
$$ = If($3, $5, NULL);
}
| IF '(' expression ')' statement ELSE statement
{
$$ = If($3, $5, $7);
}
| SWITCH '(' expression ')' statement
{
$$ = Switch($3, $5);
}
;
iteration_statement:
WHILE '(' expression ')' statement
{
$$ = While($3, $5);
}
| DO statement WHILE '(' expression ')' ';'
{
$$ = Do($2, $5);
}
| iteration_statement_for
;
iteration_statement_for:
FOR '(' ';' ';' ')' statement
{
$$ = For(NULL, NULL, NULL, $6);
}
| FOR '(' expression ';' ';' ')' statement
{
$$ = For(Expression($3), NULL, NULL, $7);
}
| FOR '(' ';' expression ';' ')' statement
{
$$ = For(NULL, $4, NULL, $7);
}
| FOR '(' ';' ';' expression ')' statement
{
$$ = For(NULL, NULL, $5, $7);
}
| FOR '(' expression ';' expression ';' ')' statement
{
$$ = For(Expression($3), $5, NULL, $8);
}
| FOR '(' expression ';' ';' expression ')' statement
{
$$ = For(Expression($3), NULL, $6, $8);
}
| FOR '(' ';' expression ';' expression ')' statement
{
$$ = For(NULL, $4, $6, $8);
}
| FOR '(' expression ';' expression ';' expression ')' statement
{
$$ = For(Expression($3), $5, $7, $9);
}
| FOR '(' declaration ';' ')' statement
{
$$ = For($3, NULL, NULL, $6);
}
| FOR '(' declaration expression ';' ')' statement
{
$$ = For($3, $4, NULL, $7);
}
| FOR '(' declaration ';' expression ')' statement
{
$$ = For($3, NULL, $5, $7);
}
| FOR '(' declaration expression ';' expression ')' statement
{
$$ = For($3, $4, $6, $8);
}
;
jump_statement:
GOTO IDENTIFIER ';'
{
$$ = Goto( Symbol($2) );
}
| CONTINUE ';'
{
$$ = Continue();
}
| BREAK ';'
{
$$ = Break();
}
| RETURN ';'
{
if (errorOnReturn)
parse_error(1, "return statement not allowed in an outlined region\n");
$$ = Return(NULL);
}
| RETURN expression ';'
{
if (errorOnReturn)
parse_error(1, "return statement not allowed in an outlined region\n");
$$ = Return($2);
}
;
translation_unit:
external_declaration
{
$$ = pastree = $1;
}
| translation_unit external_declaration
{
$$ = pastree = BlockList($1, $2);
}
;
external_declaration:
function_definition
{
$$ = $1;
}
| declaration
{
$$ = $1;
}
| ox_taskdef_construct
{
$$ = OmpixStmt($1);
}
| PRAGMA_OTHER
{
$$ = Verbatim(strdup($1));
}
;
function_definition:
normal_function_definition   { $$ = $1; }
| oldstyle_function_definition { $$ = $1; }
;
normal_function_definition:
declaration_specifiers declarator
{
stentry f;
if (isTypedef || $2->decl->type != DFUNC)
parse_error(1, "function definition cannot be parsed.\n");
f = symtab_get(stab, decl_getidentifier_symbol($2), FUNCNAME);
if (f && f->funcdef)
parse_error(1, "function %s is multiply defined.\n", f->key->name);
if (f == NULL)
{
f = symtab_put(stab, decl_getidentifier_symbol($2), FUNCNAME);
f->spec = $1;
f->decl = $2;
}
scope_start(stab);
ast_declare_function_params($2);
}
compound_statement
{
scope_end(stab);
check_for_main_and_declare($1, $2);
$$ = FuncDef($1, $2, NULL, $4);
symtab_get(stab, decl_getidentifier_symbol($2), FUNCNAME)->funcdef = $$;
}
| declarator 
{
stentry f;
if (isTypedef || $1->decl->type != DFUNC)
parse_error(1, "function definition cannot be parsed.\n");
f = symtab_get(stab, decl_getidentifier_symbol($1), FUNCNAME);
if (f && f->funcdef)
parse_error(1, "function %s is multiply defined.\n", f->key->name);
if (f == NULL)
{
f = symtab_put(stab, decl_getidentifier_symbol($1), FUNCNAME);
f->spec = NULL;
f->decl = $1;
}
scope_start(stab);
ast_declare_function_params($1);
}
compound_statement
{
astspec s = Declspec(SPEC_int);  
stentry f;
scope_end(stab);
check_for_main_and_declare(s, $1);
$$ = FuncDef(s, $1, NULL, $3);
f = symtab_get(stab, decl_getidentifier_symbol($1), FUNCNAME);
if (!f->spec) f->spec = s;
f->funcdef = $$;
}
;
oldstyle_function_definition:
declaration_specifiers declarator 
{
stentry f;
if (isTypedef || $2->decl->type != DFUNC)
parse_error(1, "function definition cannot be parsed.\n");
f = symtab_get(stab, decl_getidentifier_symbol($2), FUNCNAME);
if (f && f->funcdef)
parse_error(1, "function %s is multiply defined.\n", f->key->name);
if (f == NULL)
{
f = symtab_put(stab, decl_getidentifier_symbol($2), FUNCNAME);
f->spec = $1;
f->decl = $2;
}
scope_start(stab);
}
declaration_list compound_statement
{
scope_end(stab);
check_for_main_and_declare($1, $2);
$$ = FuncDef($1, $2, $4, $5);
symtab_get(stab, decl_getidentifier_symbol($2), FUNCNAME)->funcdef = $$;
}
| declarator 
{
stentry f;
if (isTypedef || $1->decl->type != DFUNC)
parse_error(1, "function definition cannot be parsed.\n");
f = symtab_get(stab, decl_getidentifier_symbol($1), FUNCNAME);
if (f && f->funcdef)
parse_error(1, "function %s is multiply defined.\n", f->key->name);
if (f == NULL)
{
f = symtab_put(stab, decl_getidentifier_symbol($1), FUNCNAME);
f->spec = NULL;
f->decl = $1;
}
scope_start(stab);
}
declaration_list compound_statement
{
astspec s = Declspec(SPEC_int);  
stentry f;
scope_end(stab);
check_for_main_and_declare(s, $1);
$$ = FuncDef(s, $1, $3, $4);
f = symtab_get(stab, decl_getidentifier_symbol($1), FUNCNAME);
if (!f->spec) f->spec = s;
f->funcdef = $$;
}
;
declaration_list:
declaration
{
$$ = $1;
}
| declaration_list declaration
{
$$ = BlockList($1, $2);         
}
;
asm_statement:
__ASM__ asm_stmtrest 
{
($$ = $2)->u.assem->qualifiers = NULL;
}
| __ASM__ asm_qualifiers asm_stmtrest 
{
($$ = $3)->u.assem->qualifiers = $2;
}
;
asm_stmtrest:
'(' string_literal ')'
{
$$ = BasicAsm(NULL, $2);
}
| '(' string_literal ':' asm_output ')'
{
$$ = XtendAsm(NULL, $2, $4, NULL, NULL);
}
| '(' string_literal ':' asm_output ':' asm_input ')'
{
$$ = XtendAsm(NULL, $2, $4, $6, NULL);
}
| '(' string_literal ':' asm_output ':' asm_input ':' asm_clobbers ')'
{
$$ = XtendAsm(NULL, $2, $4, $6, $8);
}
| GOTO '(' string_literal ':' ':' asm_input ':' asm_clobbers ':' labellist ')'
{
$$ = XtendAsmGoto($3, $6, $8, $10);
}
;
asm_qualifiers:
asm_qualifier
{
$$ = $1;
}
| asm_qualifiers asm_qualifier
{
$$ = Speclist_right($1, $2);
}
;
asm_qualifier:
VOLATILE { $$ = Declspec(SPEC_volatile); }
| INLINE   { $$ = Declspec(SPEC_inline); }
;
asm_output:
{
$$ = NULL;
}
| asm_outoperand
{
$$ = $1;
}
| asm_output ',' asm_outoperand
{
$$ = XAsmOpList($1, $3);
}
;
asm_outoperand:
'[' IDENTIFIER ']' string_literal '(' unary_expression  ')'
{
$$ = XAsmOperand(IdentName($2), $4, $6);
}
| string_literal '(' unary_expression  ')'
{
$$ = XAsmOperand(NULL, $1, $3);
}
;
asm_input:
{
$$ = NULL;
}
| asm_inoperand
{
$$ = $1;
}
| asm_input ',' asm_inoperand
{
$$ = XAsmOpList($1, $3);
}
;
asm_inoperand:
'[' IDENTIFIER ']' string_literal '(' expression ')'
{
$$ = XAsmOperand(IdentName($2), $4, $6);
}
| string_literal '(' expression ')'
{
$$ = XAsmOperand(NULL, $1, $3);
}
;
asm_clobbers:
string_literal 
{
$$ = String($1);
}
| asm_clobbers ',' string_literal 
{
$$ = CommaList($1, String($3));
}
;
labellist:
IDENTIFIER 
{ 
$$ = IdentName($1);
}
| labellist ',' IDENTIFIER 
{
$$ = CommaList($1, IdentName($3));
}
;
attribute_optseq:
{
$$ = NULL;
}
|	attribute_seq
{
$$ = $1;
}
;
attribute_seq:
attribute
{
$$ = $1;
}
|	attribute_seq attribute
{
$$ = Speclist_left($1, $2);
}
;
attribute:
__ATTRIBUTE__ '(' '(' attribute_name_list ')' ')'
{
$$ = AttrSpec($4);
}
;
attribute_name_list:
attribute_name
{
$$ = $1;
}
| attribute_name_list ',' attribute_name
{
if ($1 == NULL && $3 == NULL)
$$ = strdup(",");
else
if ($1 == NULL)
$$ = strdupcat(strdup(", "), $3, 1);
else
if ($3 == NULL)
$$ = strdupcat($1, strdup(", "), 1);
else
$$ = strdupcat($1, strdupcat(strdup(", "), $3, 1), 1);
}
;
attribute_name:
{
$$ = NULL;
}
|	attr_name
{
$$ = $1;
}
|	attr_name '(' argument_expression_list ')'
{
static str xp = NULL;
if (xp == NULL) xp = Strnew();
str_printf(xp, "%s(", $1);
if ($3)
ast_expr_print(xp, $3);
str_printf(xp, ")");
$$ = strdup(str_string(xp));
str_truncate(xp);
free($1);
}
;
attr_name:
IDENTIFIER { $$ = strdup($1); }
|	TYPE_NAME  { $$ = strdup($1); }
|	CONSTANT   { $$ = strdup($1); } 
;
declaration_definition:
function_definition
{
$$ = $1;
}
| declaration
{
$$ = $1;
}
;
function_statement: 
function_definition
{
$$ = $1;
}
;
declarations_definitions_seq:
declaration_definition
{
$$ = $1;
}
| declarations_definitions_seq declaration_definition
{
$$ = pastree = BlockList($1, $2);
}
;
openmp_construct:
parallel_construct
{
$$ = $1;
}
| for_construct
{
$$ = $1;
}
| sections_construct
{
$$ = $1;
}
| single_construct
{
$$ = $1;
}
| parallel_for_construct
{
$$ = $1;
}
| parallel_sections_construct
{
$$ = $1;
}
| master_construct
{
$$ = $1;
}
| critical_construct
{
$$ = $1;
}
| atomic_construct
{
$$ = $1;
}
| ordered_construct
{
$$ = $1;
}
| 
task_construct
{
$$ = $1;
}
| 
simd_construct
{
$$ = $1;
}
| for_simd_construct
{
$$ = $1;
}
| parallel_for_simd_construct
{
$$ = $1;
}
| target_data_construct
{
$$ = $1;
}
| target_construct
{
$$ = $1;
}
| teams_construct
{
$$ = $1;
}
| distribute_construct
{
$$ = $1;
}
| distribute_simd_construct
{
$$ = $1;
}
| distribute_parallel_for_construct
{
$$ = $1;
}
| distribute_parallel_for_simd_construct
{
$$ = $1;
}
| target_teams_construct
{
$$ = $1;
}
| teams_distribute_construct
{
$$ = $1;
}
| teams_distribute_simd_construct
{
$$ = $1;
}
| target_teams_distribute_construct
{
$$ = $1;
}
| target_teams_distribute_simd_construct
{
$$ = $1;
}
| teams_distribute_parallel_for_construct
{
$$ = $1;
}
| target_teams_distribute_parallel_for_construct
{
$$ = $1;
}
| teams_distribute_parallel_for_simd_construct
{
$$ = $1;
}
| target_teams_distribute_parallel_for_simd_construct
{
$$ = $1;
}
| 
taskgroup_construct 
{
$$ = $1;
}
;
openmp_directive:
barrier_directive
{
$$ = OmpConstruct(DCBARRIER, $1, NULL);
}
| flush_directive
{
$$ = OmpConstruct(DCFLUSH, $1, NULL);
}
| 
taskwait_directive
{
$$ = OmpConstruct(DCTASKWAIT, $1, NULL);
}
| 
taskyield_directive
{
$$ = OmpConstruct(DCTASKYIELD, $1, NULL);
}
| 
cancel_directive
{
$$ = OmpConstruct(DCCANCEL, $1, NULL);
}
| 
cancellation_point_directive
{
$$ = OmpConstruct(DCCANCELLATIONPOINT, $1, NULL);
}
| 
target_update_directive
{
$$ = OmpConstruct(DCTARGETUPD, $1, NULL);
}
| 
target_enter_data_directive
{
$$ = OmpConstruct(DCTARGENTERDATA, $1, NULL);
}
| 
target_exit_data_directive
{
$$ = OmpConstruct(DCTARGEXITDATA, $1, NULL);
}
;
structured_block:
statement
{
$$ = $1;
}
;
parallel_construct:
parallel_directive structured_block
{
$$ = OmpConstruct(DCPARALLEL, $1, $2);
$$->l = $1->l;
}
;
parallel_directive:
PRAGMA_OMP OMP_PARALLEL parallel_clause_optseq '\n'
{
$$ = OmpDirective(DCPARALLEL, $3);
}
;
parallel_clause_optseq:
{
$$ = NULL;
}
| parallel_clause_optseq parallel_clause
{
$$ = OmpClauseList($1, $2);
}
| parallel_clause_optseq ',' parallel_clause
{
$$ = OmpClauseList($1, $3);
}
;
parallel_clause:
unique_parallel_clause
{
$$ = $1;
}
| data_default_clause
{
$$ = $1;
}
| data_privatization_clause
{
$$ = $1;
}
| data_privatization_in_clause
{
$$ = $1;
}
| data_sharing_clause
{
$$ = $1;
}
| data_reduction_clause
{
$$ = $1;
}
;
unique_parallel_clause:
if_clause
{
$$ = $1;
}
| OMP_NUMTHREADS '(' { sc_pause_openmp(); } expression ')'
{
sc_start_openmp();
$$ = NumthreadsClause($4);
}
| OMP_COPYIN { sc_pause_openmp(); } '(' variable_list ')'
{
sc_start_openmp();
$$ = VarlistClause(OCCOPYIN, $4);
}
|  
procbind_clause  
{
$$ = $1;
}
|  
OMP_AUTO { sc_pause_openmp(); } '(' variable_list ')'
{
sc_start_openmp();
$$ = VarlistClause(OCAUTO, $4);
}
;
for_construct:
for_directive iteration_statement_for
{
$$ = OmpConstruct(DCFOR, $1, $2);
}
;
for_directive:
PRAGMA_OMP OMP_FOR for_clause_optseq '\n'
{
$$ = OmpDirective(DCFOR, $3);
}
;
for_clause_optseq:
{
$$ = NULL;
}
| for_clause_optseq for_clause
{
$$ = OmpClauseList($1, $2);
}
| for_clause_optseq ',' for_clause
{
$$ = OmpClauseList($1, $3);
}
;
for_clause:
unique_for_clause
{
$$ = $1;
}
| data_privatization_clause
{
$$ = $1;
}
| data_privatization_in_clause
{
$$ = $1;
}
| data_privatization_out_clause
{
$$ = $1;
}
| data_reduction_clause
{
$$ = $1;
}
| OMP_NOWAIT
{
$$ = PlainClause(OCNOWAIT);
}
;
unique_for_clause:
OMP_ORDERED
{
$$ = PlainClause(OCORDERED);
}
| OMP_ORDERED '(' expression  ')'     
{
int n = 0, er = 0;
if (xar_expr_is_constant($3))
{
n = xar_calc_int_expr($3, &er);
if (er) n = 0;
}
if (n <= 0)
parse_error(1, "invalid number in ordered() clause.\n");
$$ = OrderedNumClause(n);
}
| OMP_SCHEDULE '(' schedule_mod schedule_kind ')'
{
check_schedule($4, $3);
$$ = ScheduleClause($4, $3, NULL);
}
| OMP_SCHEDULE '(' schedule_mod schedule_kind ',' 
{ sc_pause_openmp(); } expression ')'
{
sc_start_openmp();
if ($4 == OC_runtime)
parse_error(1, "\"runtime\" schedules may not have a chunksize.\n");
check_schedule($4, $3);
$$ = ScheduleClause($4, $3, $7);
}
| OMP_SCHEDULE '(' OMP_AFFINITY ','
{  
tempsave = checkDecls;
checkDecls = 0;   
sc_pause_openmp();
}
expression ')'
{
sc_start_openmp();
checkDecls = tempsave;
$$ = ScheduleClause(OC_affinity, OCM_none, $6);
}
| collapse_clause
{
$$ = $1;
}
;
schedule_kind:
OMP_STATIC
{
$$ = OC_static;
}
| OMP_DYNAMIC
{
$$ = OC_dynamic;
}
| OMP_GUIDED
{
$$ = OC_guided;
}
| OMP_RUNTIME
{
$$ = OC_runtime;
}
| OMP_AUTO      
{
$$ = OC_auto;
}
| error { parse_error(1, "invalid openmp schedule type.\n"); }
;
schedule_mod:
{
$$ = OCM_none;
}
| OMP_MONOTONIC ':'
{
$$ = OCM_monotonic;
}
| OMP_NONMONOTONIC ':'
{
$$ = OCM_nonmonotonic;
}
;
sections_construct:
sections_directive section_scope
{
$$ = OmpConstruct(DCSECTIONS, $1, $2);
}
;
sections_directive:
PRAGMA_OMP OMP_SECTIONS sections_clause_optseq '\n'
{
$$ = OmpDirective(DCSECTIONS, $3);
}
;
sections_clause_optseq:
{
$$ = NULL;
}
| sections_clause_optseq sections_clause
{
$$ = OmpClauseList($1, $2);
}
| sections_clause_optseq ',' sections_clause
{
$$ = OmpClauseList($1, $3);
}
;
sections_clause:
data_privatization_clause
{
$$ = $1;
}
| data_privatization_in_clause
{
$$ = $1;
}
| data_privatization_out_clause
{
$$ = $1;
}
| data_reduction_clause
{
$$ = $1;
}
| OMP_NOWAIT
{
$$ = PlainClause(OCNOWAIT);
}
;
section_scope:
'{' section_sequence '}'
{
$$ = Compound($2);
}
;
section_sequence:
structured_block  
{
$$ = OmpStmt( OmpConstruct(DCSECTION, OmpDirective(DCSECTION,NULL), $1) );
}
| section_directive structured_block
{
$$ = OmpStmt( OmpConstruct(DCSECTION, $1, $2) );
}
| section_sequence section_directive structured_block
{
$$ = BlockList($1, OmpStmt( OmpConstruct(DCSECTION, $2, $3) ));
}
;
section_directive:
PRAGMA_OMP OMP_SECTION '\n'
{
$$ = OmpDirective(DCSECTION, NULL);
}
;
single_construct:
single_directive structured_block
{
$$ = OmpConstruct(DCSINGLE, $1, $2);
}
;
single_directive:
PRAGMA_OMP OMP_SINGLE single_clause_optseq '\n'
{
$$ = OmpDirective(DCSINGLE, $3);
}
;
single_clause_optseq:
{
$$ = NULL;
}
| single_clause_optseq single_clause
{
$$ = OmpClauseList($1, $2);
}
| single_clause_optseq ',' single_clause
{
$$ = OmpClauseList($1, $3);
}
;
single_clause:
unique_single_clause
{
$$ = $1;
}
| data_privatization_clause
{
$$ = $1;
}
| data_privatization_in_clause
{
$$ = $1;
}
| OMP_NOWAIT
{
$$ = PlainClause(OCNOWAIT);
}
;
unique_single_clause:
OMP_COPYPRIVATE  { sc_pause_openmp(); } '(' variable_list ')'
{
sc_start_openmp();
$$ = VarlistClause(OCCOPYPRIVATE, $4);
}
;
simd_construct:
simd_directive iteration_statement_for
{
}
;
simd_directive:
PRAGMA_OMP OMP_SIMD simd_clause_optseq '\n'
{
}
;
simd_clause_optseq:
{
$$ = NULL;
}
| simd_clause_optseq simd_clause
{
$$ = OmpClauseList($1, $2);
}
| simd_clause_optseq ',' simd_clause
{
$$ = OmpClauseList($1, $3);
}
;
simd_clause:
unique_simd_clause
{
$$ = $1;
}
| data_privatization_clause
{
$$ = $1;
}
| data_privatization_out_clause
{
$$ = $1;
}
| data_reduction_clause
{
$$ = $1;
}
| collapse_clause
{
$$ = $1;
}
;
unique_simd_clause:
OMP_SAFELEN '(' expression  ')'
{
int n = 0, er = 0;
if (xar_expr_is_constant($3))
{
n = xar_calc_int_expr($3, &er);
if (er) n = 0;
}
if (n <= 0)
parse_error(1, "invalid number in simdlen() clause.\n");
}
| linear_clause
{
$$ = $1;
}
| aligned_clause
{
$$ = $1;
}
;
inbranch_clause:
OMP_INBRANCH
{
}
| OMP_NOTINBRANCH
{
}
;
uniform_clause:
OMP_UNIFORM { sc_pause_openmp(); } '(' variable_list ')'
{
sc_start_openmp();
}
;
linear_clause:
OMP_LINEAR { sc_pause_openmp(); } '(' variable_list optional_expression ')'
{
sc_start_openmp();
}
;
aligned_clause: 
OMP_ALIGNED { sc_pause_openmp(); } '(' variable_list optional_expression ')'
{
sc_start_openmp();
}
;
optional_expression:
{
$$ = NULL;
}
|  ':' expression
{
}
;
declare_simd_construct: 
declare_simd_directive_seq function_statement
{
}
;
declare_simd_directive_seq:
declare_simd_directive
{
}
| declare_simd_directive_seq declare_simd_directive
{
}
;
declare_simd_directive:
PRAGMA_OMP OMP_DECLARE OMP_SIMD declare_simd_clause_optseq '\n'
{
}
;
declare_simd_clause_optseq:
{
$$ = NULL;
}
| declare_simd_clause_optseq declare_simd_clause
{
$$ = OmpClauseList($1, $2);
}
| declare_simd_clause_optseq ',' declare_simd_clause
{
$$ = OmpClauseList($1, $3);
}
;
declare_simd_clause:
OMP_SIMDLEN '(' expression  ')'
{
int n = 0, er = 0;
if (xar_expr_is_constant($3))
{
n = xar_calc_int_expr($3, &er);
if (er) n = 0;
}
if (n <= 0)
parse_error(1, "invalid number in simdlen() clause.\n");
}
| linear_clause
{
$$ = $1;
}
| aligned_clause
{
$$ = $1;
}
| uniform_clause
{
$$ = $1;
}
| inbranch_clause
{
$$ = $1;
}
;
for_simd_construct:
for_simd_directive iteration_statement_for
{
}
;
for_simd_directive:
PRAGMA_OMP OMP_FOR OMP_SIMD for_simd_clause_optseq '\n'
{
}
;
for_simd_clause_optseq:
{
$$ = NULL;
}
| for_simd_clause_optseq for_simd_clause
{
$$ = OmpClauseList($1, $2);
}
| for_simd_clause_optseq ',' for_simd_clause
{
$$ = OmpClauseList($1, $3);
}
;
for_simd_clause:
for_clause
{
$$ = $1;
}
| unique_simd_clause
{
$$ = $1;
}
;
parallel_for_simd_construct:
parallel_for_simd_directive iteration_statement_for
{
}
;
parallel_for_simd_directive:
PRAGMA_OMP OMP_PARALLEL OMP_FOR OMP_SIMD parallel_for_simd_clause_optseq '\n'
{
}
;
parallel_for_simd_clause_optseq:
{
$$ = NULL;
}
| parallel_for_simd_clause_optseq parallel_for_simd_clause
{
$$ = OmpClauseList($1, $2);
}
| parallel_for_simd_clause_optseq ',' parallel_for_simd_clause
{
$$ = OmpClauseList($1, $3);
}
;
parallel_for_simd_clause:
parallel_for_clause
{
$$ = $1;
}
| unique_simd_clause
{
$$ = $1;
}
;
target_data_construct:
target_data_directive structured_block
{
$$ = OmpConstruct(DCTARGETDATA, $1, $2);
}
;
target_data_directive:
PRAGMA_OMP OMP_TARGET OMP_DATA target_data_clause_optseq '\n'
{
$$ = OmpDirective(DCTARGETDATA, $4);
}
;
target_data_clause_optseq:
{
$$ = NULL;
}
| target_data_clause_optseq target_data_clause
{
$$ = OmpClauseList($1, $2);
}
| target_data_clause_optseq ',' target_data_clause
{
$$ = OmpClauseList($1, $3);
}
;
target_data_clause:
device_clause
{
$$ = $1;
}
| map_clause
{
$$ = $1;
if ($$->subtype != OC_tofrom && $$->subtype != OC_to && 
$$->subtype != OC_from   && $$->subtype != OC_alloc)
parse_error(1, "expected a map type of 'to', 'from', 'tofrom' or 'alloc'\n");
}
| if_clause
{
$$ = $1;
}
| use_device_ptr_clause    
{
$$ = $1;
}
;
device_clause:
OMP_DEVICE '(' { sc_pause_openmp(); } expression ')'
{
sc_start_openmp();
$$ = DeviceClause($4);
}
;
map_clause:
OMP_MAP '(' map_modifier map_type ':' { sc_pause_openmp(); } variable_array_section_list ')'
{
sc_start_openmp();
$$ = MapClause($4, $3, $7);
}
| OMP_MAP '(' { sc_pause_openmp(); } variable_array_section_list ')'
{
sc_start_openmp();
$$ = MapClause(OC_tofrom, OCM_none, $4);
}
;
map_modifier:
{
$$ = OCM_none;
}
| OMP_ALWAYS 
{
$$ = OCM_always;
}
| OMP_ALWAYS ','
{
$$ = OCM_always;
}
;
map_type:
OMP_ALLOC
{
$$ = OC_alloc;
}
| OMP_TO
{
$$ = OC_to;
}
| OMP_FROM
{
$$ = OC_from;
}
| OMP_TOFROM
{
$$ = OC_tofrom;
}
| OMP_RELEASE   
{
$$ = OC_release; 
}
| OMP_DELETE    
{
$$ = OC_delete;
}
;
use_device_ptr_clause:
OMP_USE_DEVICE_PTR '(' { sc_pause_openmp(); } variable_list ')'
{
sc_start_openmp();
$$ = VarlistClause(OCUSEDEVPTR, $4);
}
;
defaultmap_clause:
OMP_DEFAULTMAP '(' OMP_TOFROM ':' OMP_SCALAR ')'
{
$$ = PlainClause(OCDEFAULTMAP);
}
;
is_device_ptr_clause:
OMP_IS_DEVICE_PTR '(' { sc_pause_openmp(); } variable_list ')'
{
sc_start_openmp();
$$ = VarlistClause(OCISDEVPTR, $4);
}
;
target_construct:
target_directive { $<type>$ = errorOnReturn;  errorOnReturn = 1; } 
structured_block
{
errorOnReturn = $<type>2;
$$ = OmpConstruct(DCTARGET, $1, $3);
__has_target = 1;
}
;
target_directive:
PRAGMA_OMP OMP_TARGET target_clause_optseq '\n'
{
$$ = OmpDirective(DCTARGET, $3);
}
;
target_clause_optseq:
{
$$ = NULL;
}
| target_clause_optseq target_clause
{
$$ = OmpClauseList($1, $2);
}
| target_clause_optseq ',' target_clause
{
$$ = OmpClauseList($1, $3);
}
;
target_clause:
unique_target_clause
{
$$ = $1;
}
| if_clause
{
$$ = $1;
}
| depend_clause
{
$$ = $1;
}
| OMP_NOWAIT
{
$$ = PlainClause(OCNOWAIT);
}
| data_privatization_clause
{
$$ = $1;
}
| data_privatization_in_clause
{
$$ = $1;
}
;
unique_target_clause:
device_clause
{
$$ = $1;
}
| map_clause
{
$$ = $1;
if ($$->subtype != OC_tofrom && $$->subtype != OC_to && 
$$->subtype != OC_from   && $$->subtype != OC_alloc)
parse_error(1, "expected a map type of 'to', 'from', 'tofrom' or 'alloc'\n");
}
| defaultmap_clause       
{
$$ = $1;
}
| is_device_ptr_clause    
{
$$ = $1;
}
;
target_enter_data_directive:
PRAGMA_OMP OMP_TARGET OMP_ENTER OMP_DATA target_enter_data_clause_seq '\n'
{
if (xc_clauselist_get_clause($5, OCMAP, 0) == NULL)
parse_error(1, "target enter data directives must contain at least 1 "
"map() clause");
$$ = OmpDirective(DCTARGENTERDATA, $5);
}
;
target_enter_data_clause_seq:
target_enter_data_clause
{
$$ = $1;
}
| target_enter_data_clause_seq target_enter_data_clause
{
$$ = OmpClauseList($1, $2);
}
| target_enter_data_clause_seq ',' target_enter_data_clause
{
$$ = OmpClauseList($1, $3);
}
;
target_enter_data_clause:
map_clause
{
$$ = $1;
if ($$->subtype != OC_to && $$->subtype != OC_alloc)
parse_error(1, "expected a map type of 'to' or 'alloc'\n");
}
| device_clause
{
$$ = $1;
}
| if_clause
{
$$ = $1;
}
| depend_clause
{
$$ = $1;
}
| OMP_NOWAIT
{
$$ = PlainClause(OCNOWAIT);
};
target_exit_data_directive:
PRAGMA_OMP OMP_TARGET OMP_EXIT OMP_DATA target_exit_data_clause_seq '\n'
{
if (xc_clauselist_get_clause($5, OCMAP, 0) == NULL)
parse_error(1, "target exit data directives must contain at least 1 "
"map() clause");
$$ = OmpDirective(DCTARGEXITDATA, $5);
}
;
target_exit_data_clause_seq:
target_exit_data_clause
{
$$ = $1;
}
| target_exit_data_clause_seq target_exit_data_clause
{
$$ = OmpClauseList($1, $2);
}
| target_exit_data_clause_seq ',' target_exit_data_clause
{
$$ = OmpClauseList($1, $3);
}
;
target_exit_data_clause:
map_clause
{
$$ = $1;
if ($$->subtype != OC_from && $$->subtype != OC_release &&
$$->subtype != OC_delete)
parse_error(1, "expected a map type of 'from', 'release' or 'delete'\n");
}
| device_clause
{
$$ = $1;
}
| if_clause
{
$$ = $1;
}
| depend_clause
{
$$ = $1;
}
| OMP_NOWAIT
{
$$ = PlainClause(OCNOWAIT);
}
;
target_update_directive:
PRAGMA_OMP OMP_TARGET OMP_UPDATE target_update_clause_seq '\n'
{
$$ = OmpDirective(DCTARGETUPD, $4);
}
;
target_update_clause_seq:
target_update_clause
{
$$ = $1;
}
| target_update_clause_seq target_update_clause
{
$$ = OmpClauseList($1, $2);
}
| target_update_clause_seq ',' target_update_clause
{
$$ = OmpClauseList($1, $3);
}
;
target_update_clause:
motion_clause
{
$$ = $1;
}
| device_clause
{
$$ = $1;
}
| if_clause
{
$$ = $1;
}
| depend_clause
{
$$ = $1;
}
| OMP_NOWAIT
{
$$ = PlainClause(OCNOWAIT);
}
;
motion_clause:
OMP_TO { sc_pause_openmp(); } '(' variable_array_section_list ')'
{
sc_start_openmp();
$$ = UpdateClause(OCTO, $4);
}
| OMP_FROM { sc_pause_openmp(); } '(' variable_array_section_list ')'
{
sc_start_openmp();
$$ = UpdateClause(OCFROM, $4);
}
;
declare_target_construct:
declare_target_directive declarations_definitions_seq 
end_declare_target_directive
{
$$ = OmpConstruct(DCDECLTARGET, $1, $2);
}
| declare_target_directive_v45
{
$$ = OmpConstruct(DCDECLTARGET, $1, NULL);
}
;
declare_target_directive:
PRAGMA_OMP OMP_DECLARE OMP_TARGET'\n'
{
$$ = OmpDirective(DCDECLTARGET, NULL);
}
;
end_declare_target_directive:
PRAGMA_OMP OMP_END OMP_DECLARE OMP_TARGET'\n'
{
}
;
declare_target_directive_v45:
PRAGMA_OMP OMP_DECLARE OMP_TARGET 
'(' funcname_variable_array_section_list ')' '\n'
{
$$ = OmpDirective(DCDECLTARGET, UpdateClause(OCTO, $5));
}
| PRAGMA_OMP OMP_DECLARE OMP_TARGET declare_target_clause_optseq '\n'
{
$$ = OmpDirective(DCDECLTARGET, $4);
}
;
declare_target_clause_optseq:
unique_declare_target_clause
{
$$ = $1;
}
| declare_target_clause_optseq unique_declare_target_clause
{
$$ = OmpClauseList($1, $2);
}
| declare_target_clause_optseq ',' unique_declare_target_clause
{
$$ = OmpClauseList($1, $3);
}
;
unique_declare_target_clause:
OMP_TO { sc_pause_openmp(); } '(' funcname_variable_array_section_list ')'
{
sc_start_openmp();
$$ = UpdateClause(OCTO, $4);
}
| OMP_LINK 
{ 
tempsave = checkDecls;   
checkDecls = 0; 
sc_pause_openmp(); 
} 
'(' variable_array_section_list ')'
{
sc_start_openmp();
checkDecls = tempsave;
$$ = UpdateClause(OCLINK, $4);
}
;
teams_construct:
teams_directive structured_block
{
$$ = OmpConstruct(DCTEAMS, $1, $2);
}
;
teams_directive:
PRAGMA_OMP OMP_TEAMS teams_clause_optseq '\n'
{
$$ = OmpDirective(DCTEAMS, $3);
}
;
teams_clause_optseq:
{
$$ = NULL;
}
| teams_clause_optseq teams_clause
{
$$ = OmpClauseList($1, $2);
}
| teams_clause_optseq ',' teams_clause
{
$$ = OmpClauseList($1, $3);
}
;
teams_clause:
unique_teams_clause
{
$$ = $1;
}
| data_default_clause
{
$$ = $1;
}
| data_privatization_clause
{
$$ = $1;
}
| data_privatization_in_clause
{
$$ = $1;
}
| data_sharing_clause
{
$$ = $1;
}
| data_reduction_clause
{
$$ = $1;
}
;
unique_teams_clause:
OMP_NUMTEAMS '(' { sc_pause_openmp(); } expression ')'
{
sc_start_openmp();
$$ = NumteamsClause($4);
}
| 
OMP_THREADLIMIT '(' { sc_pause_openmp(); } expression ')'
{
sc_start_openmp();
$$ = ThreadlimitClause($4);
}
;
distribute_construct:
distribute_directive iteration_statement_for
{
}
;
distribute_directive:
PRAGMA_OMP OMP_DISTRIBUTE distribute_clause_optseq '\n'
{
}
;
distribute_clause_optseq:
{
$$ = NULL;
}
| distribute_clause_optseq distribute_clause
{
$$ = OmpClauseList($1, $2);
}
| distribute_clause_optseq ',' distribute_clause
{
$$ = OmpClauseList($1, $3);
}
;
distribute_clause:
data_privatization_clause
{
$$ = $1;
}
| data_privatization_in_clause
{
$$ = $1;
}
| collapse_clause
{
$$ = $1;
}
| unique_distribute_clause
{
$$ = $1;
}
;
unique_distribute_clause:
OMP_DISTSCHEDULE '(' OMP_STATIC ')'
{
$$ = ScheduleClause(OC_static, OCM_none, NULL);
}
| OMP_DISTSCHEDULE '(' OMP_STATIC ',' { sc_pause_openmp(); } expression ')'
{
sc_start_openmp();
$$ = ScheduleClause(OC_static,OCM_none,  $6);
}
;
distribute_simd_construct:
distribute_simd_directive iteration_statement_for
{
}
;
distribute_simd_directive:
PRAGMA_OMP OMP_DISTRIBUTE OMP_SIMD distribute_simd_clause_optseq '\n'
{
}
;
distribute_simd_clause_optseq:
{
$$ = NULL;
}
| distribute_simd_clause_optseq distribute_simd_clause
{
$$ = OmpClauseList($1, $2);
}
| distribute_simd_clause_optseq ',' distribute_simd_clause
{
$$ = OmpClauseList($1, $3);
}
;
distribute_simd_clause:
unique_distribute_clause
{
$$ = $1;
}
| data_privatization_in_clause
{
$$ = $1;
}
| simd_clause
{
$$ = $1;
}
;
distribute_parallel_for_construct:
distribute_parallel_for_directive iteration_statement_for
{
}
;
distribute_parallel_for_directive:
PRAGMA_OMP OMP_DISTRIBUTE OMP_PARALLEL OMP_FOR distribute_parallel_for_clause_optseq '\n'
{
}
;
distribute_parallel_for_clause_optseq:
{
$$ = NULL;
}
| distribute_parallel_for_clause_optseq distribute_parallel_for_clause
{
$$ = OmpClauseList($1, $2);
}
| distribute_parallel_for_clause_optseq ',' distribute_parallel_for_clause
{
$$ = OmpClauseList($1, $3);
}
;
distribute_parallel_for_clause:
unique_distribute_clause
{
$$ = $1;
}
| parallel_for_clause
{
$$ = $1;
}
;
distribute_parallel_for_simd_construct:
distribute_parallel_for_simd_directive structured_block
{
}
;
distribute_parallel_for_simd_directive:
PRAGMA_OMP OMP_DISTRIBUTE OMP_PARALLEL OMP_FOR OMP_SIMD distribute_parallel_for_simd_clause_optseq '\n'
{
}
;
distribute_parallel_for_simd_clause_optseq:
{
$$ = NULL;
}
| distribute_parallel_for_simd_clause_optseq distribute_parallel_for_simd_clause
{
$$ = OmpClauseList($1, $2);
}
| distribute_parallel_for_simd_clause_optseq ',' distribute_parallel_for_simd_clause
{
$$ = OmpClauseList($1, $3);
}
;
distribute_parallel_for_simd_clause:
unique_distribute_clause
{
$$ = $1;
}
| parallel_for_simd_clause
{
$$ = $1;
}
;
target_teams_construct:
target_teams_directive structured_block 
{
$$ = OmpConstruct(DCTARGETTEAMS, $1, $2);
}
;
target_teams_directive:
PRAGMA_OMP OMP_TARGET OMP_TEAMS target_teams_clause_optseq '\n'
{
$$ = OmpDirective(DCTARGETTEAMS, $4);
}
;
target_teams_clause_optseq:
{
$$ = NULL;
}
| target_teams_clause_optseq target_teams_clause
{
$$ = OmpClauseList($1, $2);
}
| target_teams_clause_optseq ',' target_teams_clause
{
$$ = OmpClauseList($1, $3);
}
;
target_teams_clause:
teams_clause
{
$$ = $1;
}
| unique_target_clause
{
$$ = $1;
}
| if_clause
{
$$ = $1;
}
| depend_clause
{
$$ = $1;
}
| OMP_NOWAIT
{
$$ = PlainClause(OCNOWAIT);
}
;
teams_distribute_construct:
teams_distribute_directive iteration_statement_for
{
}
;
teams_distribute_directive:
PRAGMA_OMP OMP_TEAMS OMP_DISTRIBUTE teams_distribute_clause_optseq '\n'
{
}
;
teams_distribute_clause_optseq:
{
$$ = NULL;
}
| teams_distribute_clause_optseq teams_distribute_clause
{
$$ = OmpClauseList($1, $2);
}
| teams_distribute_clause_optseq ',' teams_distribute_clause
{
$$ = OmpClauseList($1, $3);
}
;
teams_distribute_clause:
teams_clause
{
$$ = $1;
}
| unique_distribute_clause
{
$$ = $1;
}
| collapse_clause
{
$$ = $1;
}
;
teams_distribute_simd_construct:
teams_distribute_simd_directive iteration_statement_for
{
}
;
teams_distribute_simd_directive:
PRAGMA_OMP OMP_TEAMS OMP_DISTRIBUTE OMP_SIMD teams_distribute_simd_clause_optseq '\n'
{
}
;
teams_distribute_simd_clause_optseq:
{
$$ = NULL;
}
| teams_distribute_simd_clause_optseq teams_distribute_simd_clause
{
$$ = OmpClauseList($1, $2);
}
| teams_distribute_simd_clause_optseq ',' teams_distribute_simd_clause
{
$$ = OmpClauseList($1, $3);
}
;
teams_distribute_simd_clause:
unique_teams_clause
{
$$ = $1;
}
| data_default_clause
{
$$ = $1;
}
| data_sharing_clause
{
$$ = $1;
}
| distribute_simd_clause
{
$$ = $1;
}
;
target_teams_distribute_construct:
target_teams_distribute_directive iteration_statement_for
{
}
;
target_teams_distribute_directive:
PRAGMA_OMP OMP_TARGET OMP_TEAMS OMP_DISTRIBUTE target_teams_distribute_clause_optseq '\n'
{
}
;
target_teams_distribute_clause_optseq:
{
$$ = NULL;
}
| target_teams_distribute_clause_optseq target_teams_distribute_clause
{
$$ = OmpClauseList($1, $2);
}
| target_teams_distribute_clause_optseq ',' target_teams_distribute_clause
{
$$ = OmpClauseList($1, $3);
}
;
target_teams_distribute_clause:
teams_distribute_clause
{
$$ = $1;
}
| unique_target_clause
{
$$ = $1;
}
| if_clause
{
$$ = $1;
}
| depend_clause
{
$$ = $1;
}
| OMP_NOWAIT
{
$$ = PlainClause(OCNOWAIT);
}
;
target_teams_distribute_simd_construct:
target_teams_distribute_simd_directive iteration_statement_for
{
}
;
target_teams_distribute_simd_directive:
PRAGMA_OMP OMP_TARGET OMP_TEAMS OMP_DISTRIBUTE OMP_SIMD
target_teams_distribute_simd_clause_optseq '\n'
{
}
;
target_teams_distribute_simd_clause_optseq:
{
$$ = NULL;
}
| target_teams_distribute_simd_clause_optseq
target_teams_distribute_simd_clause
{
$$ = OmpClauseList($1, $2);
}
| target_teams_distribute_simd_clause_optseq ','
target_teams_distribute_simd_clause
{
$$ = OmpClauseList($1, $3);
}
;
target_teams_distribute_simd_clause:
teams_distribute_simd_clause
{
$$ = $1;
}
| unique_target_clause
{
$$ = $1;
}
| if_clause
{
$$ = $1;
}
| depend_clause
{
$$ = $1;
}
| OMP_NOWAIT
{
$$ = PlainClause(OCNOWAIT);
}
;
teams_distribute_parallel_for_construct:
teams_distribute_parallel_for_directive iteration_statement_for
{
}
;
teams_distribute_parallel_for_directive:
PRAGMA_OMP OMP_TEAMS OMP_DISTRIBUTE OMP_PARALLEL OMP_FOR
teams_distribute_parallel_for_clause_optseq '\n'
{
}
;
teams_distribute_parallel_for_clause_optseq:
{
$$ = NULL;
}
| teams_distribute_parallel_for_clause_optseq
teams_distribute_parallel_for_clause
{
$$ = OmpClauseList($1, $2);
}
| teams_distribute_parallel_for_clause_optseq ','
teams_distribute_parallel_for_clause
{
$$ = OmpClauseList($1, $3);
}
;
teams_distribute_parallel_for_clause:
unique_teams_clause
{
$$ = $1;
}
| distribute_parallel_for_clause
{
$$ = $1;
}
;
target_teams_distribute_parallel_for_construct:
target_teams_distribute_parallel_for_directive iteration_statement_for
{
}
;
target_teams_distribute_parallel_for_directive:
PRAGMA_OMP OMP_TARGET OMP_TEAMS OMP_DISTRIBUTE OMP_PARALLEL OMP_FOR target_teams_distribute_parallel_for_clause_optseq '\n'
{
}
;
target_teams_distribute_parallel_for_clause_optseq:
{
$$ = NULL;
}
| target_teams_distribute_parallel_for_clause_optseq target_teams_distribute_parallel_for_clause
{
$$ = OmpClauseList($1, $2);
}
| target_teams_distribute_parallel_for_clause_optseq ',' target_teams_distribute_parallel_for_clause
{
$$ = OmpClauseList($1, $3);
}
;
target_teams_distribute_parallel_for_clause:
unique_target_clause
{
$$ = $1;
}
| teams_distribute_parallel_for_clause
{
$$ = $1;
}
;
teams_distribute_parallel_for_simd_construct:
teams_distribute_parallel_for_simd_directive iteration_statement_for
{
}
;
teams_distribute_parallel_for_simd_directive:
PRAGMA_OMP OMP_TEAMS OMP_DISTRIBUTE OMP_PARALLEL OMP_FOR OMP_SIMD teams_distribute_parallel_for_simd_clause_optseq '\n'
{
}
;
teams_distribute_parallel_for_simd_clause_optseq:
{
$$ = NULL;
}
| teams_distribute_parallel_for_simd_clause_optseq teams_distribute_parallel_for_simd_clause
{
$$ = OmpClauseList($1, $2);
}
| teams_distribute_parallel_for_simd_clause_optseq ',' teams_distribute_parallel_for_simd_clause
{
$$ = OmpClauseList($1, $3);
}
;
teams_distribute_parallel_for_simd_clause:
unique_teams_clause
{
$$ = $1;
}
| distribute_parallel_for_simd_clause
{
$$ = $1;
}
;
target_teams_distribute_parallel_for_simd_construct:
target_teams_distribute_parallel_for_simd_directive iteration_statement_for
{
}
;
target_teams_distribute_parallel_for_simd_directive:
PRAGMA_OMP OMP_TARGET OMP_TEAMS OMP_DISTRIBUTE OMP_PARALLEL OMP_FOR OMP_SIMD target_teams_distribute_parallel_for_simd_clause_optseq '\n'
{
}
;
target_teams_distribute_parallel_for_simd_clause_optseq:
{
$$ = NULL;
}
| target_teams_distribute_parallel_for_simd_clause_optseq target_teams_distribute_parallel_for_simd_clause
{
$$ = OmpClauseList($1, $2);
}
| target_teams_distribute_parallel_for_simd_clause_optseq ',' target_teams_distribute_parallel_for_simd_clause
{
$$ = OmpClauseList($1, $3);
}
;
target_teams_distribute_parallel_for_simd_clause:
unique_target_clause
{
$$ = $1;
}
| teams_distribute_parallel_for_simd_clause
{
$$ = $1;
}
;
task_construct:
task_directive structured_block
{
$$ = OmpConstruct(DCTASK, $1, $2);
$$->l = $1->l;
}
;
task_directive:
PRAGMA_OMP OMP_TASK task_clause_optseq '\n'
{
$$ = OmpDirective(DCTASK, $3);
}
;
task_clause_optseq:
{
$$ = NULL;
}
| task_clause_optseq task_clause
{
$$ = OmpClauseList($1, $2);
}
| task_clause_optseq ',' task_clause
{
$$ = OmpClauseList($1, $3);
}
;
task_clause:
unique_task_clause
{
$$ = $1;
}
| data_default_clause
{
$$ = $1;
}
| data_privatization_clause
{
$$ = $1;
}
| data_privatization_in_clause
{
$$ = $1;
}
| data_sharing_clause
{
$$ = $1;
}
| depend_clause
{
$$ = $1;
}
;
unique_task_clause:
if_clause
{
$$ = $1;
}
| OMP_FINAL '(' { sc_pause_openmp(); } expression ')'
{
sc_start_openmp();
$$ = FinalClause($4);
}
| OMP_UNTIED
{
$$ = PlainClause(OCUNTIED);
}
| OMP_MERGEABLE
{
$$ = PlainClause(OCMERGEABLE);
}
| 
OMP_PRIORITY '(' expression ')'
{
$$ = PriorityClause($3);
}
;
depend_clause:
OMP_DEPEND '(' dependence_type { sc_pause_openmp(); } ':' variable_array_section_list ')'
{
sc_start_openmp();
$$ = DependClause($3, $6);
}
;
dependence_type:
OMP_IN
{
$$ = OC_in;
}
| OMP_OUT
{
$$ = OC_out;
}
| OMP_INOUT
{
$$ = OC_inout;
}
;
parallel_for_construct:
parallel_for_directive iteration_statement_for
{
$$ = OmpConstruct(DCPARFOR, $1, $2);
$$->l = $1->l;
}
;
parallel_for_directive:
PRAGMA_OMP OMP_PARALLEL OMP_FOR parallel_for_clause_optseq '\n'
{
$$ = OmpDirective(DCPARFOR, $4);
}
;
parallel_for_clause_optseq:
{
$$ = NULL;
}
| parallel_for_clause_optseq parallel_for_clause
{
$$ = OmpClauseList($1, $2);
}
| parallel_for_clause_optseq ',' parallel_for_clause
{
$$ = OmpClauseList($1, $3);
}
;
parallel_for_clause:
unique_parallel_clause
{
$$ = $1;
}
| unique_for_clause
{
$$ = $1;
}
| data_default_clause
{
$$ = $1;
}
| data_privatization_clause
{
$$ = $1;
}
| data_privatization_in_clause
{
$$ = $1;
}
| data_privatization_out_clause
{
$$ = $1;
}
| data_sharing_clause
{
$$ = $1;
}
| data_reduction_clause
{
$$ = $1;
}
;
parallel_sections_construct:
parallel_sections_directive section_scope
{
$$ = OmpConstruct(DCPARSECTIONS, $1, $2);
$$->l = $1->l;
}
;
parallel_sections_directive:
PRAGMA_OMP OMP_PARALLEL OMP_SECTIONS parallel_sections_clause_optseq '\n'
{
$$ = OmpDirective(DCPARSECTIONS, $4);
}
;
parallel_sections_clause_optseq:
{
$$ = NULL;
}
| parallel_sections_clause_optseq parallel_sections_clause
{
$$ = OmpClauseList($1, $2);
}
| parallel_sections_clause_optseq ',' parallel_sections_clause
{
$$ = OmpClauseList($1, $3);
}
;
parallel_sections_clause:
unique_parallel_clause
{
$$ = $1;
}
| data_default_clause
{
$$ = $1;
}
| data_privatization_clause
{
$$ = $1;
}
| data_privatization_in_clause
{
$$ = $1;
}
| data_privatization_out_clause
{
$$ = $1;
}
| data_sharing_clause
{
$$ = $1;
}
| data_reduction_clause
{
$$ = $1;
}
;
master_construct:
master_directive structured_block
{
$$ = OmpConstruct(DCMASTER, $1, $2);
}
;
master_directive:
PRAGMA_OMP OMP_MASTER '\n'
{
$$ = OmpDirective(DCMASTER, NULL);
}
;
critical_construct:
critical_directive structured_block
{
$$ = OmpConstruct(DCCRITICAL, $1, $2);
}
;
critical_directive:
PRAGMA_OMP OMP_CRITICAL '\n'
{
$$ = OmpCriticalDirective(NULL, NULL);
}
| PRAGMA_OMP OMP_CRITICAL region_phrase '\n'
{
$$ = OmpCriticalDirective($3, NULL);
}
| PRAGMA_OMP OMP_CRITICAL region_phrase hint_clause '\n'
{
$$ = OmpCriticalDirective($3, $4);
}
;
region_phrase:
'(' IDENTIFIER ')'
{
$$ = Symbol($2);
}
;
hint_clause:
OMP_HINT '(' expression ')'
{
$$ = HintClause($3);
}
;
barrier_directive:
PRAGMA_OMP OMP_BARRIER '\n'
{
$$ = OmpDirective(DCBARRIER, NULL);
}
;
taskwait_directive:
PRAGMA_OMP OMP_TASKWAIT '\n'
{
$$ = OmpDirective(DCTASKWAIT, NULL);
}
;
taskgroup_construct: 
taskgroup_directive structured_block
{
$$ = OmpConstruct(DCTASKGROUP, $1, $2);
}
;
taskgroup_directive:
PRAGMA_OMP OMP_TASKGROUP'\n'
{
$$ = OmpDirective(DCTASKGROUP, NULL);
}
;
taskyield_directive:
PRAGMA_OMP OMP_TASKYIELD '\n'
{
$$ = OmpDirective(DCTASKYIELD, NULL);
}
;
atomic_construct:
atomic_directive expression_statement
{
$$ = OmpConstruct(DCATOMIC, $1, $2);
}
;
atomic_directive:
PRAGMA_OMP OMP_ATOMIC atomic_clause_opt seq_cst_clause_opt '\n' 
{
$$ = OmpDirective(DCATOMIC, NULL);  
}
;
atomic_clause_opt:
{
$$ = NULL;
}
| OMP_READ
{
}
| OMP_WRITE
{
}
| OMP_UPDATE
{
}
| OMP_CAPTURE
{
}
;
seq_cst_clause_opt:
{
$$ = NULL;
}
| OMP_SEQ_CST
{
}
;
flush_directive:
PRAGMA_OMP OMP_FLUSH '\n'
{
$$ = OmpFlushDirective(NULL);
}
| PRAGMA_OMP OMP_FLUSH flush_vars '\n'
{
$$ = OmpFlushDirective($3);
}
;
flush_vars:
'(' { sc_pause_openmp(); } variable_list ')'
{
sc_start_openmp();
$$ = $3;
}
;
ordered_construct:
ordered_directive_full structured_block
{
$$ = OmpConstruct(DCORDERED, $1, $2);
}
| ordered_directive_standalone
{
$$ = OmpConstruct(DCORDERED, $1, NULL);
}
;
ordered_directive_full:
PRAGMA_OMP OMP_ORDERED ordered_clause_optseq_full '\n'
{
$$ = OmpDirective(DCORDERED, $3);
}
;
ordered_directive_standalone:
PRAGMA_OMP OMP_ORDERED OMP_DEPEND '(' OMP_SOURCE ')' '\n'
{
$$ = OmpDirective(DCORDERED, DependClause(OC_source,NULL));
}
| PRAGMA_OMP OMP_ORDERED ordered_clause_optseq_standalone '\n'
{
$$ = OmpDirective(DCORDERED, $3);
}
;
ordered_clause_optseq_full:
{
$$ = NULL;
}
| ordered_clause_optseq_full ordered_clause_type_full
{
$$ = OmpClauseList($1, $2);
}
| ordered_clause_optseq_full ',' ordered_clause_type_full
{
$$ = OmpClauseList($1, $3);
}
;
ordered_clause_type_full:
OMP_THREADS
{
$$ = PlainClause(OCTHREADS);
}
| OMP_SIMD
{
}
;
ordered_clause_optseq_standalone:
ordered_clause_depend_sink
{
$$ = $1;
}
| ordered_clause_optseq_standalone ordered_clause_depend_sink
{
$$ = OmpClauseList($1, $2);
}
| ordered_clause_optseq_standalone ',' ordered_clause_depend_sink
{
$$ = OmpClauseList($1, $3);
}
;
ordered_clause_depend_sink:
OMP_DEPEND '(' OMP_SINK ':' sink_vec ')'
{
$$ = DependClause(OC_sink, NULL);
$$->u.expr = $5;
}
; 
sink_vec:
sink_vec_elem  
{
$$ = $1;
}
| sink_vec_elem ',' sink_vec
{ 
$$ = CommaList($1, $3);
}
;
sink_vec_elem:
IDENTIFIER
{
if (checkDecls)
check_uknown_var($1);
$$ = BinaryOperator(BOP_add, IdentName($1), numConstant(0));
}
| IDENTIFIER '+' multiplicative_expression
{
if (checkDecls)
check_uknown_var($1);
$$ = BinaryOperator(BOP_add, IdentName($1), $3);
}
| IDENTIFIER '-' multiplicative_expression
{
if (checkDecls)
check_uknown_var($1);
$$ = BinaryOperator(BOP_sub, IdentName($1), $3);
}
;
cancel_directive: 
PRAGMA_OMP OMP_CANCEL construct_type_clause '\n'
{
$$ = OmpDirective(DCCANCEL, $3);
}
| PRAGMA_OMP OMP_CANCEL construct_type_clause if_clause '\n'
{
$$ = OmpDirective(DCCANCEL, OmpClauseList($3, $4));
}
| PRAGMA_OMP OMP_CANCEL construct_type_clause ',' if_clause '\n'
{
$$ = OmpDirective(DCCANCEL, OmpClauseList($3, $5));
}
;
construct_type_clause:
OMP_PARALLEL
{
$$ = PlainClause(OCPARALLEL);
}
| OMP_SECTIONS
{
$$ = PlainClause(OCSECTIONS);
}
| OMP_FOR
{
$$ = PlainClause(OCFOR);
}
| OMP_TASKGROUP
{
$$ = PlainClause(OCTASKGROUP);
}
;
cancellation_point_directive: 
PRAGMA_OMP_CANCELLATIONPOINT construct_type_clause '\n'
{
$$ = OmpDirective(DCCANCELLATIONPOINT, $2);
}
;
threadprivate_directive:
PRAGMA_OMP_THREADPRIVATE { sc_pause_openmp(); } '(' thrprv_variable_list ')' { sc_start_openmp(); } '\n'
{
$$ = OmpThreadprivateDirective($4);
}
;
declare_reduction_directive:
PRAGMA_OMP OMP_DECLARE OMP_REDUCTION '(' reduction_identifier ':' reduction_type_list ':' { sc_pause_openmp(); } expression ')' { sc_start_openmp(); } initializer_clause_opt '\n'
{
}
;
reduction_identifier:
IDENTIFIER
{
parse_error(1, "user-defined reductions are not implemented yet.\n");
}
| 
'+'
{
$$ = OC_plus;
}
| '*'
{
$$ = OC_times;
}
| '-'
{
$$ = OC_minus;
}
| '&'
{
$$ = OC_band;
}
| '^'
{
$$ = OC_xor;
}
| '|'
{
$$ = OC_bor;
}
| AND_OP
{
$$ = OC_land;
}
| OR_OP
{
$$ = OC_lor;
}
| OMP_MIN
{
$$ = OC_min;
}
| OMP_MAX
{
$$ = OC_max;
}
;
reduction_type_list:
type_specifier
{
}
| reduction_type_list ',' type_specifier
{
}
;
initializer_clause_opt:
{
$$ = NULL;
}
| OMP_INITIALIZER '(' IDENTIFIER '=' conditional_expression ')'
{
}
| OMP_INITIALIZER '(' IDENTIFIER '(' argument_expression_list ')' ')'
{
}
;
data_default_clause:
OMP_DEFAULT '(' OMP_SHARED ')'
{
$$ = DefaultClause(OC_defshared);
}
| OMP_DEFAULT '(' OMP_NONE ')'
{
$$ = DefaultClause(OC_defnone);
}
| 
OMP_DEFAULT '(' OMP_AUTO ')'
{
$$ = DefaultClause(OC_auto); 
}
;
data_privatization_clause:
OMP_PRIVATE { sc_pause_openmp(); } '(' variable_list ')'
{
sc_start_openmp();
$$ = VarlistClause(OCPRIVATE, $4);
}
;
data_privatization_in_clause:
OMP_FIRSTPRIVATE { sc_pause_openmp(); } '(' variable_list ')'
{
sc_start_openmp();
$$ = VarlistClause(OCFIRSTPRIVATE, $4);
}
;
data_privatization_out_clause:
OMP_LASTPRIVATE { sc_pause_openmp(); } '(' variable_list ')'
{
sc_start_openmp();
$$ = VarlistClause(OCLASTPRIVATE, $4);
}
;
data_sharing_clause:
OMP_SHARED { sc_pause_openmp(); } '(' variable_list ')'
{
sc_start_openmp();
$$ = VarlistClause(OCSHARED, $4);
}
;
data_reduction_clause:
OMP_REDUCTION '(' reduction_identifier { sc_pause_openmp(); } 
':' variable_array_section_list ')'
{
sc_start_openmp();
$$ = ReductionClause($3, $6);
}
;
if_clause:
OMP_IF '(' { sc_pause_openmp(); } expression ')'
{
sc_start_openmp();
$$ = IfClause($4, OCM_none);
}
| OMP_IF '(' if_related_construct ':' { sc_pause_openmp(); } expression ')'
{
sc_start_openmp();
$$ = IfClause($6, $3);
}
;
if_related_construct:
OMP_PARALLEL { $$ = OCM_parallel; }
| OMP_TASK { $$ = OCM_task; }
| OMP_TARGET { $$ = OCM_target; }
| OMP_TARGET OMP_DATA { $$ = OCM_targetdata; }
| OMP_TARGET OMP_ENTER OMP_DATA { $$ = OCM_targetenterdata; }
| OMP_TARGET OMP_EXIT OMP_DATA { $$ = OCM_targetexitdata; }
| OMP_TARGET OMP_UPDATE { $$ = OCM_targetupdate; }
| OMP_CANCEL { $$ = OCM_cancel; }
;
collapse_clause:
OMP_COLLAPSE '(' expression  ')'   
{
int n = 0, er = 0;
if (xar_expr_is_constant($3))
{
n = xar_calc_int_expr($3, &er);
if (er) n = 0;
}
if (n <= 0)
parse_error(1, "invalid number in collapse() clause.\n");
$$ = CollapseClause(n);
}
;
variable_list:
IDENTIFIER
{
if (checkDecls)
if (symtab_get(stab, Symbol($1), IDNAME) == NULL)
parse_error(-1, "unknown identifier `%s'.\n", $1);
$$ = IdentifierDecl( Symbol($1) );
}
| variable_list ',' IDENTIFIER
{
if (checkDecls)
if (symtab_get(stab, Symbol($3), IDNAME) == NULL)
parse_error(-1, "unknown identifier `%s'.\n", $3);
$$ = IdList($1, IdentifierDecl( Symbol($3) ));
}
;
variable_array_section_list:
varid_or_array_section
{
$$ = $1;
}
| variable_array_section_list ',' varid_or_array_section
{
ompxli l = $1;
for (; l->next; l = l->next) ;  
l->next = $3;
$$ = $1;
}
;
varid_or_array_section:
IDENTIFIER
{
if (checkDecls)
if (symtab_get(stab, Symbol($1), IDNAME) == NULL)
parse_error(-1, "unknown identifier `%s'.\n", $1);
$$ = PlainXLI( Symbol($1) );
}
| IDENTIFIER array_section_slice_list
{
if (checkDecls)
if (symtab_get(stab, Symbol($1), IDNAME) == NULL)
parse_error(-1, "unknown identifier `%s'.\n", $1);
$$ = ArraySection( Symbol($1), $2 );
}
;
funcname_variable_array_section_list:
funcvarid_or_array_section
{
$$ = $1;
}
| funcname_variable_array_section_list ',' funcvarid_or_array_section
{
ompxli l = $1;
for (; l->next; l = l->next) ;  
l->next = $3;
$$ = $1;
}
;
funcvarid_or_array_section:
IDENTIFIER
{
$$ = PlainXLI( Symbol($1) );
}
| IDENTIFIER array_section_slice_list
{
$$ = ArraySection( Symbol($1), $2 );
}
;
array_section_slice_list:
array_section_slice_list '[' { sc_pause_openmp(); } array_section_slice ']'
{
omparrdim d = $1;
sc_start_openmp();
for (; d->next; d = d->next) ;  
d->next = $4;
$$ = $1;
}
| '[' { sc_pause_openmp(); } array_section_slice ']'
{
sc_start_openmp();
$$ = $3;
}
;
array_section_slice:
expression ':' expression
{
$$ = OmpArrDim($1, $3);
}
| expression ':'
{
$$ = OmpArrDim($1, NULL);
}
| expression
{
$$ = OmpArrDim($1, numConstant(1));
}
| ':' expression
{
$$ = OmpArrDim(numConstant(0), $2);
}
| ':' 
{
$$ = OmpArrDim(numConstant(0), NULL);
}
;
procbind_clause:
OMP_PROCBIND '(' OMP_MASTER ')'
{
$$ = ProcBindClause(OC_bindmaster);
}
| OMP_PROCBIND '(' OMP_PRIMARY ')'
{
$$ = ProcBindClause(OC_bindprimary);
}
| OMP_PROCBIND '(' OMP_CLOSE ')'
{
$$ = ProcBindClause(OC_bindclose);
}
| OMP_PROCBIND '(' OMP_SPREAD ')'
{
$$ = ProcBindClause(OC_bindspread);
}
;
thrprv_variable_list:
IDENTIFIER
{
if (checkDecls)
{
stentry e = symtab_get(stab, Symbol($1), IDNAME);
if (e == NULL)
parse_error(-1, "unknown identifier `%s'.\n", $1);
if (e->scopelevel != stab->scopelevel)
parse_error(-1, "threadprivate directive appears at different "
"scope level\nfrom the one `%s' was declared.\n", $1);
if (stab->scopelevel > 0)    
if (speclist_getspec(e->spec, STCLASSSPEC, SPEC_static) == NULL)
parse_error(-1, "threadprivate variable `%s' does not have static "
"storage type.\n", $1);
e->isthrpriv = true;   
}
$$ = IdentifierDecl( Symbol($1) );
}
| thrprv_variable_list ',' IDENTIFIER
{
if (checkDecls)
{
stentry e = symtab_get(stab, Symbol($3), IDNAME);
if (e == NULL)
parse_error(-1, "unknown identifier `%s'.\n", $3);
if (e->scopelevel != stab->scopelevel)
parse_error(-1, "threadprivate directive appears at different "
"scope level\nfrom the one `%s' was declared.\n", $3);
if (stab->scopelevel > 0)    
if (speclist_getspec(e->spec, STCLASSSPEC, SPEC_static) == NULL)
parse_error(-1, "threadprivate variable `%s' does not have static "
"storage type.\n", $3);
e->isthrpriv = true;   
}
$$ = IdList($1, IdentifierDecl( Symbol($3) ));
}
;
ompix_directive:
ox_tasksync_directive
{
$$ = OmpixConstruct(OX_DCTASKSYNC, $1, NULL);
}
| ox_taskschedule_directive
{
$$ = OmpixConstruct(OX_DCTASKSCHEDULE, $1, NULL);
}
;
ox_tasksync_directive:
PRAGMA_OMPIX OMPIX_TASKSYNC '\n'
{
$$ = OmpixDirective(OX_DCTASKSYNC, NULL);
}
;
ox_taskschedule_directive:
PRAGMA_OMPIX OMPIX_TASKSCHEDULE
{
scope_start(stab);
}
ox_taskschedule_clause_optseq '\n'
{
scope_end(stab);
$$ = OmpixDirective(OX_DCTASKSCHEDULE, $4);
}
;
ox_taskschedule_clause_optseq:
{
$$ = NULL;
}
| ox_taskschedule_clause_optseq ox_taskschedule_clause
{
$$ = OmpixClauseList($1, $2);
}
| ox_taskschedule_clause_optseq ',' ox_taskschedule_clause
{
$$ = OmpixClauseList($1, $3);
}
;
ox_taskschedule_clause:
OMPIX_STRIDE '(' assignment_expression')'
{
$$ = OmpixStrideClause($3);
}
| OMPIX_START '(' assignment_expression ')'
{
$$ = OmpixStartClause($3);
}
| OMPIX_SCOPE '(' ox_scope_spec ')'
{
$$ = OmpixScopeClause($3);
}
| OMPIX_TIED
{
$$ = OmpixPlainClause(OX_OCTIED);
}
| OMP_UNTIED
{
$$ = OmpixPlainClause(OX_OCUNTIED);
}
;
ox_scope_spec:
OMPIX_NODES
{
$$ = OX_SCOPE_NODES;
}
| OMPIX_WORKERS
{
$$ = OX_SCOPE_WGLOBAL;
}
| OMPIX_WORKERS ',' OMPIX_GLOBAL
{
$$ = OX_SCOPE_WGLOBAL;
}
| OMPIX_WORKERS ',' OMPIX_LOCAL
{
$$ = OX_SCOPE_WLOCAL;
}
;
ompix_construct:
ox_taskdef_construct
{
$$ = $1;
}
| ox_task_construct
{
$$ = $1;
}
;
ox_taskdef_construct:
ox_taskdef_directive normal_function_definition
{
scope_start(stab);   
ast_declare_function_params($2->u.declaration.decl);
}
compound_statement
{
scope_end(stab);
$$ = OmpixTaskdef($1, $2, $4);
$$->l = $1->l;
}
| ox_taskdef_directive normal_function_definition
{
$$ = OmpixTaskdef($1, $2, NULL);
$$->l = $1->l;
}
;
ox_taskdef_directive:
PRAGMA_OMPIX OMPIX_TASKDEF
{
scope_start(stab);
}
ox_taskdef_clause_optseq '\n'
{
scope_end(stab);
$$ = OmpixDirective(OX_DCTASKDEF, $4);
}
;
ox_taskdef_clause_optseq:
{
$$ = NULL;
}
| ox_taskdef_clause_optseq ox_taskdef_clause
{
$$ = OmpixClauseList($1, $2);
}
| ox_taskdef_clause_optseq ',' ox_taskdef_clause
{
$$ = OmpixClauseList($1, $3);
}
;
ox_taskdef_clause:
OMP_IN '(' ox_variable_size_list')'
{
$$ = OmpixVarlistClause(OX_OCIN, $3);
}
| OMP_OUT '(' ox_variable_size_list')'
{
$$ = OmpixVarlistClause(OX_OCOUT, $3);
}
| OMP_INOUT '(' ox_variable_size_list')'
{
$$ = OmpixVarlistClause(OX_OCINOUT, $3);
}
| OMP_REDUCTION '(' reduction_identifier ':' ox_variable_size_list ')'
{
$$ = OmpixReductionClause($3, $5);
}
;
ox_variable_size_list:
ox_variable_size_elem
{
$$ = $1;
}
| ox_variable_size_list ',' ox_variable_size_elem
{
$$ = IdList($1, $3);
}
;
ox_variable_size_elem:
IDENTIFIER
{
$$ = IdentifierDecl( Symbol($1) );
symtab_put(stab, Symbol($1), IDNAME);
}
| IDENTIFIER '[' '?' IDENTIFIER ']'
{
if (checkDecls) check_uknown_var($4);
$$ = ArrayDecl(IdentifierDecl( Symbol($1) ), StClassSpec(SPEC_extern),
IdentName($4));
symtab_put(stab, Symbol($1), IDNAME);
}
| IDENTIFIER '[' assignment_expression ']'
{
$$ = ArrayDecl(IdentifierDecl( Symbol($1) ), NULL, $3);
symtab_put(stab, Symbol($1), IDNAME);
}
;
ox_task_construct:
ox_task_directive IDENTIFIER '(' argument_expression_list ')' ';'
{
$$ = OmpixConstruct(OX_DCTASK, $1, 
FuncCallStmt(IdentName(strcmp($2,"main") ? $2 : MAIN_NEWNAME),$4));
$$->l = $1->l;
}
;
ox_task_directive:
PRAGMA_OMPIX OMP_TASK ox_task_clause_optseq '\n'
{
$$ = OmpixDirective(OX_DCTASK, $3);
}
;
ox_task_clause_optseq:
{
$$ = NULL;
}
| ox_task_clause_optseq ox_task_clause
{
$$ = OmpixClauseList($1, $2);
}
| ox_task_clause_optseq ',' ox_task_clause
{
$$ = OmpixClauseList($1, $3);
}
;
ox_task_clause:
OMPIX_ATNODE '(' '*' ')'
{
$$ = OmpixPlainClause(OX_OCATALL);
}
| OMPIX_ATNODE '(' assignment_expression ')'
{
$$ = OmpixAtnodeClause($3);
}
| OMPIX_ATNODE '(' OMPIX_HERE ')'
{
$$ = OmpixPlainClause(OX_OCLOCAL);
}
| OMPIX_ATNODE '(' OMPIX_REMOTE ')'
{
$$ = OmpixPlainClause(OX_OCREMOTE);
}
| OMPIX_ATWORKER '(' assignment_expression ')'
{
$$ = OmpixAtworkerClause($3);
}
| OMPIX_TIED
{
$$ = OmpixPlainClause(OX_OCTIED);
}
| OMP_UNTIED
{
$$ = OmpixPlainClause(OX_OCUNTIED);
}
| OMPIX_DETACHED
{
$$ = OmpixPlainClause(OX_OCDETACHED);
}
| OMP_HINT '(' expression ')'
{
$$ = OmpixHintsClause($3);
}
| IF '(' expression ')'
{
$$ = OmpixIfClause($3);
}
;
%%
void yyerror(const char *s)
{
fprintf(stderr, "(file %s, line %d, column %d):\n\t%s\n",
sc_original_file(), sc_original_line(), sc_column(), s);
}
char *strdupcat(char *first, char *second, int freethem)
{
char *res;
if (first == NULL && second == NULL)
return NULL;
if (first == NULL) 
return (freethem) ? second : strdup(second);
if (second == NULL) 
return (freethem) ? first : strdup(first);
if ((res = malloc(strlen(first)+strlen(second)+1)) == NULL)
parse_error(1, "strdupcat ran out of memory\n");
sprintf(res, "%s%s", first, second);
if (freethem)
{
free(first);
free(second);
}
return res;
}
void check_uknown_var(char *name)
{
symbol s = Symbol(name);
if (symtab_get(stab, s, IDNAME) == NULL &&
symtab_get(stab, s, LABELNAME) == NULL &&
symtab_get(stab, s, FUNCNAME) == NULL)
parse_error(-1, "unknown identifier `%s'.\n", name);
}
astdecl fix_known_typename(astspec s)
{
astspec prev;
astdecl d;
if (s->type != SPECLIST || s->u.next->type != SPECLIST) return (NULL);
for (; s->u.next->type == SPECLIST; prev = s, s = s->u.next)
;   
if (s->u.next->type != USERTYPE)         
return (NULL);
prev->u.next = s->body;
d = Declarator(NULL, IdentifierDecl(s->u.next->name));
if (checkDecls)
symtab_put(stab, s->u.next->name, TYPENAME);
free(s);
return (d);
}
void check_for_main_and_declare(astspec s, astdecl d)
{
astdecl n = decl_getidentifier(d);
assert(d->type == DECLARATOR);
assert(d->decl->type == DFUNC);
if (strcmp(n->u.id->name, "main") == 0)
{
n->u.id = Symbol(MAIN_NEWNAME);         
hasMainfunc = 1;
if (d->decl->u.params == NULL || d->decl->u.params->type != DLIST)
d->decl->u.params =
ParamList(
ParamDecl(
Declspec(SPEC_int),
Declarator( NULL, IdentifierDecl( Symbol("_argc_ignored") ) )
),
ParamDecl(
Declspec(SPEC_char),
Declarator(Speclist_right( Pointer(), Pointer() ),
IdentifierDecl( Symbol("_argv_ignored") ))
)
);
mainfuncRettype = 0; 
if (s != NULL)
{
for (; s->type == SPECLIST && s->subtype == SPEC_Rlist; s = s->u.next)
if (s->body->type == SPEC && s->body->subtype == SPEC_void)
{
s = s->body;
break;
};
if (s->type == SPEC && s->subtype == SPEC_void)
mainfuncRettype = 1; 
}
}
if (symtab_get(stab, n->u.id, FUNCNAME) == NULL)
symtab_put(stab, n->u.id, FUNCNAME);
}
void add_declaration_links(astspec s, astdecl d)
{
astdecl ini = NULL;
if (d->type == DLIST && d->subtype == DECL_decllist)
{
add_declaration_links(s, d->u.next);
d = d->decl;
}
if (d->type == DINIT) d = (ini = d)->decl;   
assert(d->type == DECLARATOR);
if (d->decl != NULL && d->decl->type != ABSDECLARATOR)
{
symbol  t = decl_getidentifier_symbol(d->decl);
stentry e = isTypedef ?
symtab_get(stab,t,TYPENAME) :
symtab_get(stab,t,(decl_getkind(d)==DFUNC) ? FUNCNAME : IDNAME);
e->spec  = s;
e->decl  = d;
e->idecl = ini;
}
}
void  check_schedule(ompclsubt_e sched, ompclmod_e mod)
{
if (mod == OCM_none) return;
if (mod == OCM_nonmonotonic && sched != OC_dynamic && sched != OC_guided)
parse_error(1, "nonmonotonic modifier is only allowed in dynamic or "
"guided schedules\n");
}
void parse_error(int exitvalue, char *format, ...)
{
va_list ap;
va_start(ap, format);
fprintf(stderr, "(%s, line %d)\n\t", sc_original_file(), sc_original_line());
vfprintf(stderr, format, ap);
va_end(ap);
if (strcmp(sc_original_file(), "injected_code") == 0)
fprintf(stderr, "\n>>>>>>>\n%s\n>>>>>>>\n", parsingstring);
_exit(exitvalue);
}
void parse_warning(char *format, ...)
{
va_list ap;
va_start(ap, format);
fprintf(stderr, "[warning] ");
vfprintf(stderr, format, ap);
va_end(ap);
}
aststmt parse_file(char *fname, int *error)
{
*error = 0;
if ( (yyin = fopen(fname, "r")) == NULL )
return (NULL);
sc_set_filename(fname);      
*error = yyparse();
fclose(yyin);                
return (pastree);
}
#define PARSE_STRING_SIZE 16384
astexpr parse_expression_string(char *format, ...)
{
static char s[PARSE_STRING_SIZE];
int    savecD;
va_list ap;
va_start(ap, format);
vsnprintf(s, PARSE_STRING_SIZE-1, format, ap);
va_end(ap);
parsingstring = s;
sc_scan_string(s);
sc_set_start_token(START_SYMBOL_EXPRESSION);
savecD = checkDecls;
checkDecls = 0;         
yyparse();
checkDecls = savecD;    
return ( pastree_expr );
}
aststmt parse_blocklist_string(char *format, ...)
{
static char s[PARSE_STRING_SIZE];
int    savecD;
va_list ap;
va_start(ap, format);
vsnprintf(s, PARSE_STRING_SIZE-1, format, ap);
va_end(ap);
parsingstring = s;
sc_scan_string(s);
sc_set_start_token(START_SYMBOL_BLOCKLIST);
savecD = checkDecls;
checkDecls = 0;         
yyparse();
checkDecls = savecD;    
return ( pastree_stmt );
}
aststmt parse_and_declare_blocklist_string(char *format, ...)
{
static char s[PARSE_STRING_SIZE];
int    savecD;
va_list ap;
va_start(ap, format);
vsnprintf(s, PARSE_STRING_SIZE-1, format, ap);
va_end(ap);
parsingstring = s;
sc_scan_string(s);
sc_set_start_token(START_SYMBOL_BLOCKLIST);
savecD = checkDecls;
checkDecls = 1;         
yyparse();
checkDecls = savecD;    
return ( pastree_stmt );
}
aststmt parse_transunit_string(char *format, ...)
{
static char s[PARSE_STRING_SIZE];
int    savecD;
va_list ap;
va_start(ap, format);
vsnprintf(s, PARSE_STRING_SIZE-1, format, ap);
va_end(ap);
parsingstring = s;
sc_scan_string(s);
sc_set_start_token(START_SYMBOL_TRANSUNIT);
savecD = checkDecls;
checkDecls = 0;         
yyparse();
checkDecls = savecD;    
return ( pastree_stmt );
}

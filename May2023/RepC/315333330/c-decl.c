#include "config.h"
#define INCLUDE_UNIQUE_PTR
#include "system.h"
#include "coretypes.h"
#include "target.h"
#include "function.h"
#include "c-tree.h"
#include "timevar.h"
#include "stringpool.h"
#include "cgraph.h"
#include "intl.h"
#include "print-tree.h"
#include "stor-layout.h"
#include "varasm.h"
#include "attribs.h"
#include "toplev.h"
#include "debug.h"
#include "c-family/c-objc.h"
#include "c-family/c-pragma.h"
#include "c-family/c-ubsan.h"
#include "c-lang.h"
#include "langhooks.h"
#include "tree-iterator.h"
#include "dumpfile.h"
#include "plugin.h"
#include "c-family/c-ada-spec.h"
#include "builtins.h"
#include "spellcheck-tree.h"
#include "gcc-rich-location.h"
#include "asan.h"
#include "c-family/name-hint.h"
#include "c-family/known-headers.h"
#include "c-family/c-spellcheck.h"
enum decl_context
{ NORMAL,			
FUNCDEF,			
PARM,				
FIELD,			
TYPENAME};			
enum deprecated_states {
DEPRECATED_NORMAL,
DEPRECATED_SUPPRESS
};

tree pending_invalid_xref;
location_t pending_invalid_xref_location;
static location_t current_function_prototype_locus;
static bool current_function_prototype_built_in;
static tree current_function_prototype_arg_types;
static struct c_arg_info *current_function_arg_info;
struct obstack parser_obstack;
static GTY(()) struct stmt_tree_s c_stmt_tree;
tree c_break_label;
tree c_cont_label;
static GTY(()) tree visible_builtins;
int current_function_returns_value;
int current_function_returns_null;
int current_function_returns_abnormally;
static int warn_about_return_type;
static bool undef_nested_function;
int current_omp_declare_target_attribute;

struct GTY((chain_next ("%h.prev"))) c_binding {
union GTY(()) {		
tree GTY((tag ("0"))) type; 
struct c_label_vars * GTY((tag ("1"))) label; 
} GTY((desc ("TREE_CODE (%0.decl) == LABEL_DECL"))) u;
tree decl;			
tree id;			
struct c_binding *prev;	
struct c_binding *shadowed;	
unsigned int depth : 28;      
BOOL_BITFIELD invisible : 1;  
BOOL_BITFIELD nested : 1;     
BOOL_BITFIELD inner_comp : 1; 
BOOL_BITFIELD in_struct : 1;	
location_t locus;		
};
#define B_IN_SCOPE(b1, b2) ((b1)->depth == (b2)->depth)
#define B_IN_CURRENT_SCOPE(b) ((b)->depth == current_scope->depth)
#define B_IN_FILE_SCOPE(b) ((b)->depth == 1 )
#define B_IN_EXTERNAL_SCOPE(b) ((b)->depth == 0 )
struct GTY(()) lang_identifier {
struct c_common_identifier common_id;
struct c_binding *symbol_binding; 
struct c_binding *tag_binding;    
struct c_binding *label_binding;  
};
extern char C_SIZEOF_STRUCT_LANG_IDENTIFIER_isnt_accurate
[(sizeof(struct lang_identifier) == C_SIZEOF_STRUCT_LANG_IDENTIFIER) ? 1 : -1];
void (*c_binding_oracle) (enum c_oracle_request, tree identifier);
#define I_SYMBOL_CHECKED(node) \
(TREE_LANG_FLAG_4 (IDENTIFIER_NODE_CHECK (node)))
static inline struct c_binding* *
i_symbol_binding (tree node)
{
struct lang_identifier *lid
= (struct lang_identifier *) IDENTIFIER_NODE_CHECK (node);
if (lid->symbol_binding == NULL
&& c_binding_oracle != NULL
&& !I_SYMBOL_CHECKED (node))
{
I_SYMBOL_CHECKED (node) = 1;
c_binding_oracle (C_ORACLE_SYMBOL, node);
}
return &lid->symbol_binding;
}
#define I_SYMBOL_BINDING(node) (*i_symbol_binding (node))
#define I_SYMBOL_DECL(node) \
(I_SYMBOL_BINDING(node) ? I_SYMBOL_BINDING(node)->decl : 0)
#define I_TAG_CHECKED(node) \
(TREE_LANG_FLAG_5 (IDENTIFIER_NODE_CHECK (node)))
static inline struct c_binding **
i_tag_binding (tree node)
{
struct lang_identifier *lid
= (struct lang_identifier *) IDENTIFIER_NODE_CHECK (node);
if (lid->tag_binding == NULL
&& c_binding_oracle != NULL
&& !I_TAG_CHECKED (node))
{
I_TAG_CHECKED (node) = 1;
c_binding_oracle (C_ORACLE_TAG, node);
}
return &lid->tag_binding;
}
#define I_TAG_BINDING(node) (*i_tag_binding (node))
#define I_TAG_DECL(node) \
(I_TAG_BINDING(node) ? I_TAG_BINDING(node)->decl : 0)
#define I_LABEL_CHECKED(node) \
(TREE_LANG_FLAG_6 (IDENTIFIER_NODE_CHECK (node)))
static inline struct c_binding **
i_label_binding (tree node)
{
struct lang_identifier *lid
= (struct lang_identifier *) IDENTIFIER_NODE_CHECK (node);
if (lid->label_binding == NULL
&& c_binding_oracle != NULL
&& !I_LABEL_CHECKED (node))
{
I_LABEL_CHECKED (node) = 1;
c_binding_oracle (C_ORACLE_LABEL, node);
}
return &lid->label_binding;
}
#define I_LABEL_BINDING(node) (*i_label_binding (node))
#define I_LABEL_DECL(node) \
(I_LABEL_BINDING(node) ? I_LABEL_BINDING(node)->decl : 0)
union GTY((desc ("TREE_CODE (&%h.generic) == IDENTIFIER_NODE"),
chain_next ("(union lang_tree_node *) c_tree_chain_next (&%h.generic)"))) lang_tree_node
{
union tree_node GTY ((tag ("0"),
desc ("tree_node_structure (&%h)")))
generic;
struct lang_identifier GTY ((tag ("1"))) identifier;
};
struct GTY(()) c_spot_bindings {
struct c_scope *scope;
struct c_binding *bindings_in_scope;
int stmt_exprs;
bool left_stmt_expr;
};
struct GTY(()) c_goto_bindings {
location_t loc;
struct c_spot_bindings goto_bindings;
};
typedef struct c_goto_bindings *c_goto_bindings_p;
struct GTY(()) c_label_vars {
struct c_label_vars *shadowed;
struct c_spot_bindings label_bindings;
vec<tree, va_gc> *decls_in_scope;
vec<c_goto_bindings_p, va_gc> *gotos;
};
struct GTY((chain_next ("%h.outer"))) c_scope {
struct c_scope *outer;
struct c_scope *outer_function;
struct c_binding *bindings;
tree blocks;
tree blocks_last;
unsigned int depth : 28;
BOOL_BITFIELD parm_flag : 1;
BOOL_BITFIELD had_vla_unspec : 1;
BOOL_BITFIELD warned_forward_parm_decls : 1;
BOOL_BITFIELD function_body : 1;
BOOL_BITFIELD keep : 1;
BOOL_BITFIELD float_const_decimal64 : 1;
BOOL_BITFIELD has_label_bindings : 1;
BOOL_BITFIELD has_jump_unsafe_decl : 1;
};
static GTY(()) struct c_scope *current_scope;
static GTY(()) struct c_scope *current_function_scope;
static GTY(()) struct c_scope *file_scope;
static GTY(()) struct c_scope *external_scope;
static GTY((deletable)) struct c_scope *scope_freelist;
static GTY((deletable)) struct c_binding *binding_freelist;
#define SCOPE_LIST_APPEND(scope, list, decl) do {	\
struct c_scope *s_ = (scope);				\
tree d_ = (decl);					\
if (s_->list##_last)					\
BLOCK_CHAIN (s_->list##_last) = d_;			\
else							\
s_->list = d_;					\
s_->list##_last = d_;					\
} while (0)
#define SCOPE_LIST_CONCAT(tscope, to, fscope, from) do {	\
struct c_scope *t_ = (tscope);				\
struct c_scope *f_ = (fscope);				\
if (t_->to##_last)						\
BLOCK_CHAIN (t_->to##_last) = f_->from;			\
else								\
t_->to = f_->from;						\
t_->to##_last = f_->from##_last;				\
} while (0)
struct GTY((chain_next ("%h.next"))) c_inline_static {
location_t location;
tree function;
tree static_decl;
enum c_inline_static_type type;
struct c_inline_static *next;
};
static GTY(()) struct c_inline_static *c_inline_statics;
static bool keep_next_level_flag;
static bool next_is_function_body;
typedef struct c_binding *c_binding_ptr;
struct c_struct_parse_info
{
auto_vec<tree> struct_types;
auto_vec<c_binding_ptr> fields;
auto_vec<tree> typedefs_seen;
};
static struct c_struct_parse_info *struct_parse_info;
static tree lookup_name_in_scope (tree, struct c_scope *);
static tree c_make_fname_decl (location_t, tree, int);
static tree grokdeclarator (const struct c_declarator *,
struct c_declspecs *,
enum decl_context, bool, tree *, tree *, tree *,
bool *, enum deprecated_states);
static tree grokparms (struct c_arg_info *, bool);
static void layout_array_type (tree);
static void warn_defaults_to (location_t, int, const char *, ...)
ATTRIBUTE_GCC_DIAG(3,4);

tree
add_stmt (tree t)
{
enum tree_code code = TREE_CODE (t);
if (CAN_HAVE_LOCATION_P (t) && code != LABEL_EXPR)
{
if (!EXPR_HAS_LOCATION (t))
SET_EXPR_LOCATION (t, input_location);
}
if (code == LABEL_EXPR || code == CASE_LABEL_EXPR)
STATEMENT_LIST_HAS_LABEL (cur_stmt_list) = 1;
if (!building_stmt_list_p ())
push_stmt_list ();
append_to_statement_list_force (t, &cur_stmt_list);
return t;
}

static tree
c_build_pointer_type (tree to_type)
{
addr_space_t as = to_type == error_mark_node? ADDR_SPACE_GENERIC
: TYPE_ADDR_SPACE (to_type);
machine_mode pointer_mode;
if (as != ADDR_SPACE_GENERIC || c_default_pointer_mode == VOIDmode)
pointer_mode = targetm.addr_space.pointer_mode (as);
else
pointer_mode = c_default_pointer_mode;
return build_pointer_type_for_mode (to_type, pointer_mode, false);
}

static bool
decl_jump_unsafe (tree decl)
{
if (error_operand_p (decl))
return false;
if ((VAR_P (decl) || TREE_CODE (decl) == TYPE_DECL)
&& variably_modified_type_p (TREE_TYPE (decl), NULL_TREE))
return true;
if (warn_jump_misses_init
&& VAR_P (decl)
&& !TREE_STATIC (decl)
&& DECL_INITIAL (decl) != NULL_TREE)
return true;
return false;
}

void
c_print_identifier (FILE *file, tree node, int indent)
{
void (*save) (enum c_oracle_request, tree identifier);
save = c_binding_oracle;
c_binding_oracle = NULL;
print_node (file, "symbol", I_SYMBOL_DECL (node), indent + 4);
print_node (file, "tag", I_TAG_DECL (node), indent + 4);
print_node (file, "label", I_LABEL_DECL (node), indent + 4);
if (C_IS_RESERVED_WORD (node) && C_RID_CODE (node) != RID_CXX_COMPAT_WARN)
{
tree rid = ridpointers[C_RID_CODE (node)];
indent_to (file, indent + 4);
fprintf (file, "rid " HOST_PTR_PRINTF " \"%s\"",
(void *) rid, IDENTIFIER_POINTER (rid));
}
c_binding_oracle = save;
}
static void
bind (tree name, tree decl, struct c_scope *scope, bool invisible,
bool nested, location_t locus)
{
struct c_binding *b, **here;
if (binding_freelist)
{
b = binding_freelist;
binding_freelist = b->prev;
}
else
b = ggc_alloc<c_binding> ();
b->shadowed = 0;
b->decl = decl;
b->id = name;
b->depth = scope->depth;
b->invisible = invisible;
b->nested = nested;
b->inner_comp = 0;
b->in_struct = 0;
b->locus = locus;
b->u.type = NULL;
b->prev = scope->bindings;
scope->bindings = b;
if (decl_jump_unsafe (decl))
scope->has_jump_unsafe_decl = 1;
if (!name)
return;
switch (TREE_CODE (decl))
{
case LABEL_DECL:     here = &I_LABEL_BINDING (name);   break;
case ENUMERAL_TYPE:
case UNION_TYPE:
case RECORD_TYPE:    here = &I_TAG_BINDING (name);     break;
case VAR_DECL:
case FUNCTION_DECL:
case TYPE_DECL:
case CONST_DECL:
case PARM_DECL:
case ERROR_MARK:     here = &I_SYMBOL_BINDING (name);  break;
default:
gcc_unreachable ();
}
while (*here && (*here)->depth > scope->depth)
here = &(*here)->shadowed;
b->shadowed = *here;
*here = b;
}
static struct c_binding *
free_binding_and_advance (struct c_binding *b)
{
struct c_binding *prev = b->prev;
memset (b, 0, sizeof (struct c_binding));
b->prev = binding_freelist;
binding_freelist = b;
return prev;
}
static void
bind_label (tree name, tree label, struct c_scope *scope,
struct c_label_vars *label_vars)
{
struct c_binding *b;
bind (name, label, scope, false, false,
UNKNOWN_LOCATION);
scope->has_label_bindings = true;
b = scope->bindings;
gcc_assert (b->decl == label);
label_vars->shadowed = b->u.label;
b->u.label = label_vars;
}

void
c_finish_incomplete_decl (tree decl)
{
if (VAR_P (decl))
{
tree type = TREE_TYPE (decl);
if (type != error_mark_node
&& TREE_CODE (type) == ARRAY_TYPE
&& !DECL_EXTERNAL (decl)
&& TYPE_DOMAIN (type) == NULL_TREE)
{
warning_at (DECL_SOURCE_LOCATION (decl),
0, "array %q+D assumed to have one element", decl);
complete_array_type (&TREE_TYPE (decl), NULL_TREE, true);
relayout_decl (decl);
}
}
}

void
record_inline_static (location_t loc, tree func, tree decl,
enum c_inline_static_type type)
{
c_inline_static *csi = ggc_alloc<c_inline_static> ();
csi->location = loc;
csi->function = func;
csi->static_decl = decl;
csi->type = type;
csi->next = c_inline_statics;
c_inline_statics = csi;
}
static void
check_inline_statics (void)
{
struct c_inline_static *csi;
for (csi = c_inline_statics; csi; csi = csi->next)
{
if (DECL_EXTERNAL (csi->function))
switch (csi->type)
{
case csi_internal:
pedwarn (csi->location, 0,
"%qD is static but used in inline function %qD "
"which is not static", csi->static_decl, csi->function);
break;
case csi_modifiable:
pedwarn (csi->location, 0,
"%q+D is static but declared in inline function %qD "
"which is not static", csi->static_decl, csi->function);
break;
default:
gcc_unreachable ();
}
}
c_inline_statics = NULL;
}

static void
set_spot_bindings (struct c_spot_bindings *p, bool defining)
{
if (defining)
{
p->scope = current_scope;
p->bindings_in_scope = current_scope->bindings;
}
else
{
p->scope = NULL;
p->bindings_in_scope = NULL;
}
p->stmt_exprs = 0;
p->left_stmt_expr = false;
}
static bool
update_spot_bindings (struct c_scope *scope, struct c_spot_bindings *p)
{
if (p->scope != scope)
{
return false;
}
p->scope = scope->outer;
p->bindings_in_scope = p->scope->bindings;
return true;
}

void *
objc_get_current_scope (void)
{
return current_scope;
}
void
objc_mark_locals_volatile (void *enclosing_blk)
{
struct c_scope *scope;
struct c_binding *b;
for (scope = current_scope;
scope && scope != enclosing_blk;
scope = scope->outer)
{
for (b = scope->bindings; b; b = b->prev)
objc_volatilize_decl (b->decl);
if (scope->function_body)
break;
}
}
bool
global_bindings_p (void)
{
return current_scope == file_scope;
}
void
keep_next_level (void)
{
keep_next_level_flag = true;
}
void
set_float_const_decimal64 (void)
{
current_scope->float_const_decimal64 = true;
}
void
clear_float_const_decimal64 (void)
{
current_scope->float_const_decimal64 = false;
}
bool
float_const_decimal64_p (void)
{
return current_scope->float_const_decimal64;
}
void
declare_parm_level (void)
{
current_scope->parm_flag = true;
}
void
push_scope (void)
{
if (next_is_function_body)
{
current_scope->parm_flag         = false;
current_scope->function_body     = true;
current_scope->keep              = true;
current_scope->outer_function    = current_function_scope;
current_function_scope           = current_scope;
keep_next_level_flag = false;
next_is_function_body = false;
if (current_scope->outer)
current_scope->float_const_decimal64
= current_scope->outer->float_const_decimal64;
else
current_scope->float_const_decimal64 = false;
}
else
{
struct c_scope *scope;
if (scope_freelist)
{
scope = scope_freelist;
scope_freelist = scope->outer;
}
else
scope = ggc_cleared_alloc<c_scope> ();
if (current_scope)
scope->float_const_decimal64 = current_scope->float_const_decimal64;
else
scope->float_const_decimal64 = false;
scope->keep          = keep_next_level_flag;
scope->outer         = current_scope;
scope->depth	   = current_scope ? (current_scope->depth + 1) : 0;
if (current_scope && scope->depth == 0)
{
scope->depth--;
sorry ("GCC supports only %u nested scopes", scope->depth);
}
current_scope        = scope;
keep_next_level_flag = false;
}
}
static void
update_label_decls (struct c_scope *scope)
{
struct c_scope *s;
s = scope;
while (s != NULL)
{
if (s->has_label_bindings)
{
struct c_binding *b;
for (b = s->bindings; b != NULL; b = b->prev)
{
struct c_label_vars *label_vars;
struct c_binding *b1;
bool hjud;
unsigned int ix;
struct c_goto_bindings *g;
if (TREE_CODE (b->decl) != LABEL_DECL)
continue;
label_vars = b->u.label;
b1 = label_vars->label_bindings.bindings_in_scope;
if (label_vars->label_bindings.scope == NULL)
hjud = false;
else
hjud = label_vars->label_bindings.scope->has_jump_unsafe_decl;
if (update_spot_bindings (scope, &label_vars->label_bindings))
{
if (hjud)
{
for (; b1 != NULL; b1 = b1->prev)
{
if (decl_jump_unsafe (b1->decl))
vec_safe_push(label_vars->decls_in_scope, b1->decl);
}
}
}
FOR_EACH_VEC_SAFE_ELT (label_vars->gotos, ix, g)
update_spot_bindings (scope, &g->goto_bindings);
}
}
if (s == current_function_scope)
break;
s = s->outer;
}
}
static void
set_type_context (tree type, tree context)
{
for (type = TYPE_MAIN_VARIANT (type); type;
type = TYPE_NEXT_VARIANT (type))
TYPE_CONTEXT (type) = context;
}
tree
pop_scope (void)
{
struct c_scope *scope = current_scope;
tree block, context, p;
struct c_binding *b;
bool functionbody = scope->function_body;
bool keep = functionbody || scope->keep || scope->bindings;
update_label_decls (scope);
block = NULL_TREE;
if (keep)
{
block = make_node (BLOCK);
BLOCK_SUBBLOCKS (block) = scope->blocks;
TREE_USED (block) = 1;
for (p = scope->blocks; p; p = BLOCK_CHAIN (p))
BLOCK_SUPERCONTEXT (p) = block;
BLOCK_VARS (block) = NULL_TREE;
}
if (scope->function_body)
context = current_function_decl;
else if (scope == file_scope)
{
tree file_decl
= build_translation_unit_decl (get_identifier (main_input_filename));
context = file_decl;
debug_hooks->register_main_translation_unit (file_decl);
}
else
context = block;
for (b = scope->bindings; b; b = free_binding_and_advance (b))
{
p = b->decl;
switch (TREE_CODE (p))
{
case LABEL_DECL:
if (TREE_USED (p) && !DECL_INITIAL (p))
{
error ("label %q+D used but not defined", p);
DECL_INITIAL (p) = error_mark_node;
}
else
warn_for_unused_label (p);
DECL_CHAIN (p) = BLOCK_VARS (block);
BLOCK_VARS (block) = p;
gcc_assert (I_LABEL_BINDING (b->id) == b);
I_LABEL_BINDING (b->id) = b->shadowed;
release_tree_vector (b->u.label->decls_in_scope);
b->u.label = b->u.label->shadowed;
break;
case ENUMERAL_TYPE:
case UNION_TYPE:
case RECORD_TYPE:
set_type_context (p, context);
if (b->id)
{
gcc_assert (I_TAG_BINDING (b->id) == b);
I_TAG_BINDING (b->id) = b->shadowed;
}
break;
case FUNCTION_DECL:
if (!TREE_ASM_WRITTEN (p)
&& DECL_INITIAL (p) != NULL_TREE
&& TREE_ADDRESSABLE (p)
&& DECL_ABSTRACT_ORIGIN (p) != NULL_TREE
&& DECL_ABSTRACT_ORIGIN (p) != p)
TREE_ADDRESSABLE (DECL_ABSTRACT_ORIGIN (p)) = 1;
if (!TREE_PUBLIC (p)
&& !DECL_INITIAL (p)
&& !b->nested
&& scope != file_scope
&& scope != external_scope)
{
error ("nested function %q+D declared but never defined", p);
undef_nested_function = true;
}
else if (DECL_DECLARED_INLINE_P (p)
&& TREE_PUBLIC (p)
&& !DECL_INITIAL (p))
{
if (!flag_gnu89_inline
&& !lookup_attribute ("gnu_inline", DECL_ATTRIBUTES (p))
&& scope == external_scope)
pedwarn (input_location, 0,
"inline function %q+D declared but never defined", p);
DECL_EXTERNAL (p) = 1;
}
goto common_symbol;
case VAR_DECL:
if ((!TREE_USED (p) || !DECL_READ_P (p))
&& !TREE_NO_WARNING (p)
&& !DECL_IN_SYSTEM_HEADER (p)
&& DECL_NAME (p)
&& !DECL_ARTIFICIAL (p)
&& scope != file_scope
&& scope != external_scope)
{
if (!TREE_USED (p))
warning (OPT_Wunused_variable, "unused variable %q+D", p);
else if (DECL_CONTEXT (p) == current_function_decl)
warning_at (DECL_SOURCE_LOCATION (p),
OPT_Wunused_but_set_variable,
"variable %qD set but not used", p);
}
if (b->inner_comp)
{
error ("type of array %q+D completed incompatibly with"
" implicit initialization", p);
}
case TYPE_DECL:
case CONST_DECL:
common_symbol:
if (!b->nested)
{
DECL_CHAIN (p) = BLOCK_VARS (block);
BLOCK_VARS (block) = p;
}
else if (VAR_OR_FUNCTION_DECL_P (p) && scope != file_scope)
{
tree extp = copy_node (p);
DECL_EXTERNAL (extp) = 1;
TREE_STATIC (extp) = 0;
TREE_PUBLIC (extp) = 1;
DECL_INITIAL (extp) = NULL_TREE;
DECL_LANG_SPECIFIC (extp) = NULL;
DECL_CONTEXT (extp) = current_function_decl;
if (TREE_CODE (p) == FUNCTION_DECL)
{
DECL_RESULT (extp) = NULL_TREE;
DECL_SAVED_TREE (extp) = NULL_TREE;
DECL_STRUCT_FUNCTION (extp) = NULL;
}
if (b->locus != UNKNOWN_LOCATION)
DECL_SOURCE_LOCATION (extp) = b->locus;
DECL_CHAIN (extp) = BLOCK_VARS (block);
BLOCK_VARS (block) = extp;
}
if (scope == file_scope)
{
DECL_CONTEXT (p) = context;
if (TREE_CODE (p) == TYPE_DECL
&& TREE_TYPE (p) != error_mark_node)
set_type_context (TREE_TYPE (p), context);
}
gcc_fallthrough ();
case PARM_DECL:
case ERROR_MARK:
if (b->id)
{
gcc_assert (I_SYMBOL_BINDING (b->id) == b);
I_SYMBOL_BINDING (b->id) = b->shadowed;
if (b->shadowed && b->shadowed->u.type)
TREE_TYPE (b->shadowed->decl) = b->shadowed->u.type;
}
break;
default:
gcc_unreachable ();
}
}
if ((scope->function_body || scope == file_scope) && context)
{
DECL_INITIAL (context) = block;
BLOCK_SUPERCONTEXT (block) = context;
}
else if (scope->outer)
{
if (block)
SCOPE_LIST_APPEND (scope->outer, blocks, block);
else if (scope->blocks)
SCOPE_LIST_CONCAT (scope->outer, blocks, scope, blocks);
}
current_scope = scope->outer;
if (scope->function_body)
current_function_scope = scope->outer_function;
memset (scope, 0, sizeof (struct c_scope));
scope->outer = scope_freelist;
scope_freelist = scope;
return block;
}
void
push_file_scope (void)
{
tree decl;
if (file_scope)
return;
push_scope ();
file_scope = current_scope;
start_fname_decls ();
for (decl = visible_builtins; decl; decl = DECL_CHAIN (decl))
bind (DECL_NAME (decl), decl, file_scope,
false, true, DECL_SOURCE_LOCATION (decl));
}
void
pop_file_scope (void)
{
while (current_scope != file_scope)
pop_scope ();
finish_fname_decls ();
check_inline_statics ();
if (pch_file)
{
c_common_write_pch ();
flag_syntax_only = 1;
return;
}
pop_scope ();
file_scope = 0;
maybe_apply_pending_pragma_weaks ();
}

void
c_bindings_start_stmt_expr (struct c_spot_bindings* switch_bindings)
{
struct c_scope *scope;
for (scope = current_scope; scope != NULL; scope = scope->outer)
{
struct c_binding *b;
if (!scope->has_label_bindings)
continue;
for (b = scope->bindings; b != NULL; b = b->prev)
{
struct c_label_vars *label_vars;
unsigned int ix;
struct c_goto_bindings *g;
if (TREE_CODE (b->decl) != LABEL_DECL)
continue;
label_vars = b->u.label;
++label_vars->label_bindings.stmt_exprs;
FOR_EACH_VEC_SAFE_ELT (label_vars->gotos, ix, g)
++g->goto_bindings.stmt_exprs;
}
}
if (switch_bindings != NULL)
++switch_bindings->stmt_exprs;
}
void
c_bindings_end_stmt_expr (struct c_spot_bindings *switch_bindings)
{
struct c_scope *scope;
for (scope = current_scope; scope != NULL; scope = scope->outer)
{
struct c_binding *b;
if (!scope->has_label_bindings)
continue;
for (b = scope->bindings; b != NULL; b = b->prev)
{
struct c_label_vars *label_vars;
unsigned int ix;
struct c_goto_bindings *g;
if (TREE_CODE (b->decl) != LABEL_DECL)
continue;
label_vars = b->u.label;
--label_vars->label_bindings.stmt_exprs;
if (label_vars->label_bindings.stmt_exprs < 0)
{
label_vars->label_bindings.left_stmt_expr = true;
label_vars->label_bindings.stmt_exprs = 0;
}
FOR_EACH_VEC_SAFE_ELT (label_vars->gotos, ix, g)
{
--g->goto_bindings.stmt_exprs;
if (g->goto_bindings.stmt_exprs < 0)
{
g->goto_bindings.left_stmt_expr = true;
g->goto_bindings.stmt_exprs = 0;
}
}
}
}
if (switch_bindings != NULL)
{
--switch_bindings->stmt_exprs;
gcc_assert (switch_bindings->stmt_exprs >= 0);
}
}

static void
pushtag (location_t loc, tree name, tree type)
{
if (name && !TYPE_NAME (type))
TYPE_NAME (type) = name;
bind (name, type, current_scope, false, false, loc);
TYPE_STUB_DECL (type) = pushdecl (build_decl (loc,
TYPE_DECL, NULL_TREE, type));
TYPE_CONTEXT (type) = DECL_CONTEXT (TYPE_STUB_DECL (type));
if (warn_cxx_compat && name != NULL_TREE)
{
struct c_binding *b = I_SYMBOL_BINDING (name);
if (b != NULL
&& b->decl != NULL_TREE
&& TREE_CODE (b->decl) == TYPE_DECL
&& (B_IN_CURRENT_SCOPE (b)
|| (current_scope == file_scope && B_IN_EXTERNAL_SCOPE (b)))
&& (TYPE_MAIN_VARIANT (TREE_TYPE (b->decl))
!= TYPE_MAIN_VARIANT (type)))
{
if (warning_at (loc, OPT_Wc___compat,
("using %qD as both a typedef and a tag is "
"invalid in C++"), b->decl)
&& b->locus != UNKNOWN_LOCATION)
inform (b->locus, "originally defined here");
}
}
}
void
c_pushtag (location_t loc, tree name, tree type)
{
pushtag (loc, name, type);
}
void
c_bind (location_t loc, tree decl, bool is_global)
{
struct c_scope *scope;
bool nested = false;
if (!VAR_P (decl) || current_function_scope == NULL)
{
scope = file_scope;
DECL_EXTERNAL (decl) = 1;
TREE_PUBLIC (decl) = 1;
}
else if (is_global)
{
bind (DECL_NAME (decl), decl, external_scope, true, false, loc);
nested = true;
scope = file_scope;
DECL_EXTERNAL (decl) = 1;
TREE_PUBLIC (decl) = 1;
}
else
{
DECL_CONTEXT (decl) = current_function_decl;
TREE_PUBLIC (decl) = 0;
scope = current_function_scope;
}
bind (DECL_NAME (decl), decl, scope, false, nested, loc);
}

static tree
match_builtin_function_types (tree newtype, tree oldtype)
{
tree newrettype, oldrettype;
tree newargs, oldargs;
tree trytype, tryargs;
oldrettype = TREE_TYPE (oldtype);
newrettype = TREE_TYPE (newtype);
if (TYPE_MODE (oldrettype) != TYPE_MODE (newrettype))
return NULL_TREE;
oldargs = TYPE_ARG_TYPES (oldtype);
newargs = TYPE_ARG_TYPES (newtype);
tryargs = newargs;
while (oldargs || newargs)
{
if (!oldargs
|| !newargs
|| !TREE_VALUE (oldargs)
|| !TREE_VALUE (newargs)
|| TYPE_MODE (TREE_VALUE (oldargs))
!= TYPE_MODE (TREE_VALUE (newargs)))
return NULL_TREE;
oldargs = TREE_CHAIN (oldargs);
newargs = TREE_CHAIN (newargs);
}
trytype = build_function_type (newrettype, tryargs);
tree oldattrs = TYPE_ATTRIBUTES (oldtype);
tree oldtsafe = lookup_attribute ("transaction_safe", oldattrs);
tree newattrs = TYPE_ATTRIBUTES (newtype);
tree newtsafe = lookup_attribute ("transaction_safe", newattrs);
if (oldtsafe && !newtsafe)
oldattrs = remove_attribute ("transaction_safe", oldattrs);
else if (newtsafe && !oldtsafe)
oldattrs = tree_cons (get_identifier ("transaction_safe"),
NULL_TREE, oldattrs);
return build_type_attribute_variant (trytype, oldattrs);
}
static void
diagnose_arglist_conflict (tree newdecl, tree olddecl,
tree newtype, tree oldtype)
{
tree t;
if (TREE_CODE (olddecl) != FUNCTION_DECL
|| !comptypes (TREE_TYPE (oldtype), TREE_TYPE (newtype))
|| !((!prototype_p (oldtype) && DECL_INITIAL (olddecl) == NULL_TREE)
|| (!prototype_p (newtype) && DECL_INITIAL (newdecl) == NULL_TREE)))
return;
t = TYPE_ARG_TYPES (oldtype);
if (t == NULL_TREE)
t = TYPE_ARG_TYPES (newtype);
for (; t; t = TREE_CHAIN (t))
{
tree type = TREE_VALUE (t);
if (TREE_CHAIN (t) == NULL_TREE
&& TYPE_MAIN_VARIANT (type) != void_type_node)
{
inform (input_location, "a parameter list with an ellipsis can%'t match "
"an empty parameter name list declaration");
break;
}
if (c_type_promotes_to (type) != type)
{
inform (input_location, "an argument type that has a default promotion can%'t match "
"an empty parameter name list declaration");
break;
}
}
}
static bool
validate_proto_after_old_defn (tree newdecl, tree newtype, tree oldtype)
{
tree newargs, oldargs;
int i;
#define END_OF_ARGLIST(t) ((t) == void_type_node)
oldargs = TYPE_ACTUAL_ARG_TYPES (oldtype);
newargs = TYPE_ARG_TYPES (newtype);
i = 1;
for (;;)
{
tree oldargtype = TREE_VALUE (oldargs);
tree newargtype = TREE_VALUE (newargs);
if (oldargtype == error_mark_node || newargtype == error_mark_node)
return false;
oldargtype = (TYPE_ATOMIC (oldargtype)
? c_build_qualified_type (TYPE_MAIN_VARIANT (oldargtype),
TYPE_QUAL_ATOMIC)
: TYPE_MAIN_VARIANT (oldargtype));
newargtype = (TYPE_ATOMIC (newargtype)
? c_build_qualified_type (TYPE_MAIN_VARIANT (newargtype),
TYPE_QUAL_ATOMIC)
: TYPE_MAIN_VARIANT (newargtype));
if (END_OF_ARGLIST (oldargtype) && END_OF_ARGLIST (newargtype))
break;
if (END_OF_ARGLIST (oldargtype))
{
error ("prototype for %q+D declares more arguments "
"than previous old-style definition", newdecl);
return false;
}
else if (END_OF_ARGLIST (newargtype))
{
error ("prototype for %q+D declares fewer arguments "
"than previous old-style definition", newdecl);
return false;
}
else if (!comptypes (oldargtype, newargtype))
{
error ("prototype for %q+D declares argument %d"
" with incompatible type",
newdecl, i);
return false;
}
oldargs = TREE_CHAIN (oldargs);
newargs = TREE_CHAIN (newargs);
i++;
}
warning (0, "prototype for %q+D follows non-prototype definition",
newdecl);
return true;
#undef END_OF_ARGLIST
}
static void
locate_old_decl (tree decl)
{
if (TREE_CODE (decl) == FUNCTION_DECL && DECL_BUILT_IN (decl)
&& !C_DECL_DECLARED_BUILTIN (decl))
;
else if (DECL_INITIAL (decl))
inform (input_location, "previous definition of %q+D was here", decl);
else if (C_DECL_IMPLICIT (decl))
inform (input_location, "previous implicit declaration of %q+D was here", decl);
else
inform (input_location, "previous declaration of %q+D was here", decl);
}
static bool
diagnose_mismatched_decls (tree newdecl, tree olddecl,
tree *newtypep, tree *oldtypep)
{
tree newtype, oldtype;
bool pedwarned = false;
bool warned = false;
bool retval = true;
#define DECL_EXTERN_INLINE(DECL) (DECL_DECLARED_INLINE_P (DECL)  \
&& DECL_EXTERNAL (DECL))
if (olddecl == error_mark_node || newdecl == error_mark_node)
return false;
*oldtypep = oldtype = TREE_TYPE (olddecl);
*newtypep = newtype = TREE_TYPE (newdecl);
if (oldtype == error_mark_node || newtype == error_mark_node)
return false;
if (TREE_CODE (olddecl) != TREE_CODE (newdecl))
{
if (!(TREE_CODE (olddecl) == FUNCTION_DECL
&& DECL_BUILT_IN (olddecl)
&& !C_DECL_DECLARED_BUILTIN (olddecl)))
{
error ("%q+D redeclared as different kind of symbol", newdecl);
locate_old_decl (olddecl);
}
else if (TREE_PUBLIC (newdecl))
warning (OPT_Wbuiltin_declaration_mismatch,
"built-in function %q+D declared as non-function",
newdecl);
else
warning (OPT_Wshadow, "declaration of %q+D shadows "
"a built-in function", newdecl);
return false;
}
if (TREE_CODE (olddecl) == CONST_DECL)
{
error ("redeclaration of enumerator %q+D", newdecl);
locate_old_decl (olddecl);
return false;
}
if (!comptypes (oldtype, newtype))
{
if (TREE_CODE (olddecl) == FUNCTION_DECL
&& DECL_BUILT_IN (olddecl) && !C_DECL_DECLARED_BUILTIN (olddecl))
{
tree trytype = match_builtin_function_types (newtype, oldtype);
if (trytype && comptypes (newtype, trytype))
*oldtypep = oldtype = trytype;
else
{
warning (OPT_Wbuiltin_declaration_mismatch,
"conflicting types for built-in function %q+D",
newdecl);
return false;
}
}
else if (TREE_CODE (olddecl) == FUNCTION_DECL
&& DECL_IS_BUILTIN (olddecl))
{
TREE_THIS_VOLATILE (newdecl) |= TREE_THIS_VOLATILE (olddecl);
return false;
}
else if (TREE_CODE (newdecl) == FUNCTION_DECL && DECL_INITIAL (newdecl)
&& TYPE_MAIN_VARIANT (TREE_TYPE (oldtype)) == void_type_node
&& TYPE_MAIN_VARIANT (TREE_TYPE (newtype)) == integer_type_node
&& C_FUNCTION_IMPLICIT_INT (newdecl) && !DECL_INITIAL (olddecl))
{
pedwarned = pedwarn (input_location, 0,
"conflicting types for %q+D", newdecl);
TREE_TYPE (newdecl) = *newtypep = newtype = oldtype;
C_FUNCTION_IMPLICIT_INT (newdecl) = 0;
}
else if (TREE_CODE (newdecl) == FUNCTION_DECL
&& TYPE_MAIN_VARIANT (TREE_TYPE (newtype)) == void_type_node
&& TYPE_MAIN_VARIANT (TREE_TYPE (oldtype)) == integer_type_node
&& C_DECL_IMPLICIT (olddecl) && !DECL_INITIAL (olddecl))
{
pedwarned = pedwarn (input_location, 0,
"conflicting types for %q+D", newdecl);
TREE_TYPE (olddecl) = *oldtypep = oldtype = newtype;
}
else
{
int new_quals = TYPE_QUALS (newtype);
int old_quals = TYPE_QUALS (oldtype);
if (new_quals != old_quals)
{
addr_space_t new_addr = DECODE_QUAL_ADDR_SPACE (new_quals);
addr_space_t old_addr = DECODE_QUAL_ADDR_SPACE (old_quals);
if (new_addr != old_addr)
{
if (ADDR_SPACE_GENERIC_P (new_addr))
error ("conflicting named address spaces (generic vs %s) "
"for %q+D",
c_addr_space_name (old_addr), newdecl);
else if (ADDR_SPACE_GENERIC_P (old_addr))
error ("conflicting named address spaces (%s vs generic) "
"for %q+D",
c_addr_space_name (new_addr), newdecl);
else
error ("conflicting named address spaces (%s vs %s) "
"for %q+D",
c_addr_space_name (new_addr),
c_addr_space_name (old_addr),
newdecl);
}
if (CLEAR_QUAL_ADDR_SPACE (new_quals)
!= CLEAR_QUAL_ADDR_SPACE (old_quals))
error ("conflicting type qualifiers for %q+D", newdecl);
}
else
error ("conflicting types for %q+D", newdecl);
diagnose_arglist_conflict (newdecl, olddecl, newtype, oldtype);
locate_old_decl (olddecl);
return false;
}
}
if (TREE_CODE (newdecl) == TYPE_DECL)
{
bool types_different = false;
int comptypes_result;
comptypes_result
= comptypes_check_different_types (oldtype, newtype, &types_different);
if (comptypes_result != 1 || types_different)
{
error ("redefinition of typedef %q+D with different type", newdecl);
locate_old_decl (olddecl);
return false;
}
if (DECL_IN_SYSTEM_HEADER (newdecl)
|| DECL_IN_SYSTEM_HEADER (olddecl)
|| TREE_NO_WARNING (newdecl)
|| TREE_NO_WARNING (olddecl))
return true;  
if (variably_modified_type_p (newtype, NULL))
{
error ("redefinition of typedef %q+D with variably modified type",
newdecl);
locate_old_decl (olddecl);
}
else if (pedwarn_c99 (input_location, OPT_Wpedantic,
"redefinition of typedef %q+D", newdecl))
locate_old_decl (olddecl);
return true;
}
else if (TREE_CODE (newdecl) == FUNCTION_DECL)
{
if (DECL_BUILT_IN (olddecl)
&& !C_DECL_DECLARED_BUILTIN (olddecl)
&& (!TREE_PUBLIC (newdecl)
|| (DECL_INITIAL (newdecl)
&& !prototype_p (TREE_TYPE (newdecl)))))
{
warning (OPT_Wshadow, "declaration of %q+D shadows "
"a built-in function", newdecl);
return false;
}
if (DECL_INITIAL (newdecl))
{
if (DECL_INITIAL (olddecl))
{
if ((!DECL_EXTERN_INLINE (olddecl)
|| DECL_EXTERN_INLINE (newdecl)
|| (!flag_gnu89_inline
&& (!DECL_DECLARED_INLINE_P (olddecl)
|| !lookup_attribute ("gnu_inline",
DECL_ATTRIBUTES (olddecl)))
&& (!DECL_DECLARED_INLINE_P (newdecl)
|| !lookup_attribute ("gnu_inline",
DECL_ATTRIBUTES (newdecl))))
)
&& same_translation_unit_p (newdecl, olddecl))
{
error ("redefinition of %q+D", newdecl);
locate_old_decl (olddecl);
return false;
}
}
}
else if (DECL_INITIAL (olddecl)
&& !prototype_p (oldtype) && prototype_p (newtype)
&& TYPE_ACTUAL_ARG_TYPES (oldtype)
&& !validate_proto_after_old_defn (newdecl, newtype, oldtype))
{
locate_old_decl (olddecl);
return false;
}
if (TREE_PUBLIC (olddecl) && !TREE_PUBLIC (newdecl))
{
if (!DECL_IS_BUILTIN (olddecl)
&& !DECL_EXTERN_INLINE (olddecl))
{
error ("static declaration of %q+D follows "
"non-static declaration", newdecl);
locate_old_decl (olddecl);
}
return false;
}
else if (TREE_PUBLIC (newdecl) && !TREE_PUBLIC (olddecl))
{
if (DECL_CONTEXT (olddecl))
{
error ("non-static declaration of %q+D follows "
"static declaration", newdecl);
locate_old_decl (olddecl);
return false;
}
else if (warn_traditional)
{
warned |= warning (OPT_Wtraditional,
"non-static declaration of %q+D "
"follows static declaration", newdecl);
}
}
if (DECL_DECLARED_INLINE_P (olddecl)
&& DECL_DECLARED_INLINE_P (newdecl))
{
bool newa = lookup_attribute ("gnu_inline",
DECL_ATTRIBUTES (newdecl)) != NULL;
bool olda = lookup_attribute ("gnu_inline",
DECL_ATTRIBUTES (olddecl)) != NULL;
if (newa != olda)
{
error_at (input_location, "%<gnu_inline%> attribute present on %q+D",
newa ? newdecl : olddecl);
error_at (DECL_SOURCE_LOCATION (newa ? olddecl : newdecl),
"but not here");
}
}
}
else if (VAR_P (newdecl))
{
if (C_DECL_THREADPRIVATE_P (olddecl) && !DECL_THREAD_LOCAL_P (newdecl))
{
;
}
else if (DECL_THREAD_LOCAL_P (newdecl) != DECL_THREAD_LOCAL_P (olddecl))
{
if (DECL_THREAD_LOCAL_P (newdecl))
error ("thread-local declaration of %q+D follows "
"non-thread-local declaration", newdecl);
else
error ("non-thread-local declaration of %q+D follows "
"thread-local declaration", newdecl);
locate_old_decl (olddecl);
return false;
}
if (DECL_INITIAL (newdecl) && DECL_INITIAL (olddecl))
{
error ("redefinition of %q+D", newdecl);
locate_old_decl (olddecl);
return false;
}
if (DECL_FILE_SCOPE_P (newdecl)
&& TREE_PUBLIC (newdecl) != TREE_PUBLIC (olddecl))
{
if (DECL_EXTERNAL (newdecl))
{
if (!DECL_FILE_SCOPE_P (olddecl))
{
error ("extern declaration of %q+D follows "
"declaration with no linkage", newdecl);
locate_old_decl (olddecl);
return false;
}
else if (warn_traditional)
{
warned |= warning (OPT_Wtraditional,
"non-static declaration of %q+D "
"follows static declaration", newdecl);
}
}
else
{
if (TREE_PUBLIC (newdecl))
error ("non-static declaration of %q+D follows "
"static declaration", newdecl);
else
error ("static declaration of %q+D follows "
"non-static declaration", newdecl);
locate_old_decl (olddecl);
return false;
}
}
else if (!DECL_FILE_SCOPE_P (newdecl))
{
if (DECL_EXTERNAL (newdecl))
{
}
else if (DECL_EXTERNAL (olddecl))
{
error ("declaration of %q+D with no linkage follows "
"extern declaration", newdecl);
locate_old_decl (olddecl);
}
else
{
error ("redeclaration of %q+D with no linkage", newdecl);
locate_old_decl (olddecl);
}
return false;
}
if (warn_cxx_compat
&& DECL_FILE_SCOPE_P (newdecl)
&& !DECL_EXTERNAL (newdecl)
&& !DECL_EXTERNAL (olddecl))
warned |= warning_at (DECL_SOURCE_LOCATION (newdecl),
OPT_Wc___compat,
("duplicate declaration of %qD is "
"invalid in C++"),
newdecl);
}
if (CODE_CONTAINS_STRUCT (TREE_CODE (newdecl), TS_DECL_WITH_VIS)
&& DECL_VISIBILITY_SPECIFIED (newdecl) && DECL_VISIBILITY_SPECIFIED (olddecl)
&& DECL_VISIBILITY (newdecl) != DECL_VISIBILITY (olddecl))
{
warned |= warning (0, "redeclaration of %q+D with different visibility "
"(old visibility preserved)", newdecl);
}
if (TREE_CODE (newdecl) == FUNCTION_DECL)
warned |= diagnose_mismatched_attributes (olddecl, newdecl);
else 
{
if (TREE_CODE (newdecl) == PARM_DECL
&& (!TREE_ASM_WRITTEN (olddecl) || TREE_ASM_WRITTEN (newdecl)))
{
error ("redefinition of parameter %q+D", newdecl);
locate_old_decl (olddecl);
return false;
}
}
if (!warned && !pedwarned
&& warn_redundant_decls
&& !(TREE_CODE (newdecl) == FUNCTION_DECL
&& DECL_INITIAL (newdecl) && !DECL_INITIAL (olddecl))
&& !(TREE_CODE (newdecl) == FUNCTION_DECL
&& !DECL_BUILT_IN (newdecl)
&& DECL_BUILT_IN (olddecl)
&& !C_DECL_DECLARED_BUILTIN (olddecl))
&& !(DECL_EXTERNAL (olddecl) && !DECL_EXTERNAL (newdecl))
&& !(TREE_CODE (newdecl) == PARM_DECL
&& TREE_ASM_WRITTEN (olddecl) && !TREE_ASM_WRITTEN (newdecl))
&& !(VAR_P (newdecl)
&& DECL_INITIAL (newdecl) && !DECL_INITIAL (olddecl)))
{
warned = warning (OPT_Wredundant_decls, "redundant redeclaration of %q+D",
newdecl);
}
if (warned || pedwarned)
locate_old_decl (olddecl);
#undef DECL_EXTERN_INLINE
return retval;
}
static void
merge_decls (tree newdecl, tree olddecl, tree newtype, tree oldtype)
{
bool new_is_definition = (TREE_CODE (newdecl) == FUNCTION_DECL
&& DECL_INITIAL (newdecl) != NULL_TREE);
bool new_is_prototype = (TREE_CODE (newdecl) == FUNCTION_DECL
&& prototype_p (TREE_TYPE (newdecl)));
bool old_is_prototype = (TREE_CODE (olddecl) == FUNCTION_DECL
&& prototype_p (TREE_TYPE (olddecl)));
if (TREE_CODE (newdecl) == PARM_DECL
&& TREE_ASM_WRITTEN (olddecl) && !TREE_ASM_WRITTEN (newdecl))
{
struct c_binding *b, **here;
for (here = &current_scope->bindings; *here; here = &(*here)->prev)
if ((*here)->decl == olddecl)
goto found;
gcc_unreachable ();
found:
b = *here;
*here = b->prev;
b->prev = current_scope->bindings;
current_scope->bindings = b;
TREE_ASM_WRITTEN (olddecl) = 0;
}
DECL_ATTRIBUTES (newdecl)
= targetm.merge_decl_attributes (olddecl, newdecl);
if (TREE_CODE (newdecl) == TYPE_DECL)
{
tree tem = newtype;
newtype = oldtype;
if (TYPE_USER_ALIGN (tem))
{
if (TYPE_ALIGN (tem) > TYPE_ALIGN (newtype))
SET_TYPE_ALIGN (newtype, TYPE_ALIGN (tem));
TYPE_USER_ALIGN (newtype) = true;
}
if (TYPE_NAME (TREE_TYPE (newdecl)) == newdecl)
{
tree remove = TREE_TYPE (newdecl);
for (tree t = TYPE_MAIN_VARIANT (remove); ;
t = TYPE_NEXT_VARIANT (t))
if (TYPE_NEXT_VARIANT (t) == remove)
{
TYPE_NEXT_VARIANT (t) = TYPE_NEXT_VARIANT (remove);
break;
}
}
}
TREE_TYPE (newdecl)
= TREE_TYPE (olddecl)
= composite_type (newtype, oldtype);
if (!comptypes (oldtype, TREE_TYPE (newdecl)))
{
if (TREE_TYPE (newdecl) != error_mark_node)
layout_type (TREE_TYPE (newdecl));
if (TREE_CODE (newdecl) != FUNCTION_DECL
&& TREE_CODE (newdecl) != TYPE_DECL
&& TREE_CODE (newdecl) != CONST_DECL)
layout_decl (newdecl, 0);
}
else
{
DECL_SIZE (newdecl) = DECL_SIZE (olddecl);
DECL_SIZE_UNIT (newdecl) = DECL_SIZE_UNIT (olddecl);
SET_DECL_MODE (newdecl, DECL_MODE (olddecl));
if (DECL_ALIGN (olddecl) > DECL_ALIGN (newdecl))
{
SET_DECL_ALIGN (newdecl, DECL_ALIGN (olddecl));
DECL_USER_ALIGN (newdecl) |= DECL_USER_ALIGN (olddecl);
}
if (DECL_WARN_IF_NOT_ALIGN (olddecl)
> DECL_WARN_IF_NOT_ALIGN (newdecl))
SET_DECL_WARN_IF_NOT_ALIGN (newdecl,
DECL_WARN_IF_NOT_ALIGN (olddecl));
}
if (HAS_RTL_P (olddecl))
COPY_DECL_RTL (olddecl, newdecl);
if (TREE_READONLY (newdecl))
TREE_READONLY (olddecl) = 1;
if (TREE_THIS_VOLATILE (newdecl))
TREE_THIS_VOLATILE (olddecl) = 1;
if (TREE_DEPRECATED (newdecl))
TREE_DEPRECATED (olddecl) = 1;
if (CODE_CONTAINS_STRUCT (TREE_CODE (olddecl), TS_DECL_WITH_VIS)
&& DECL_IN_SYSTEM_HEADER (olddecl)
&& !DECL_IN_SYSTEM_HEADER (newdecl) )
DECL_SOURCE_LOCATION (newdecl) = DECL_SOURCE_LOCATION (olddecl);
else if (CODE_CONTAINS_STRUCT (TREE_CODE (olddecl), TS_DECL_WITH_VIS)
&& DECL_IN_SYSTEM_HEADER (newdecl)
&& !DECL_IN_SYSTEM_HEADER (olddecl))
DECL_SOURCE_LOCATION (olddecl) = DECL_SOURCE_LOCATION (newdecl);
else if ((DECL_INITIAL (newdecl) == NULL_TREE
&& DECL_INITIAL (olddecl) != NULL_TREE)
|| (old_is_prototype && !new_is_prototype
&& !C_DECL_BUILTIN_PROTOTYPE (olddecl)))
DECL_SOURCE_LOCATION (newdecl) = DECL_SOURCE_LOCATION (olddecl);
if (DECL_INITIAL (newdecl) == NULL_TREE)
DECL_INITIAL (newdecl) = DECL_INITIAL (olddecl);
if (VAR_P (olddecl) && C_DECL_THREADPRIVATE_P (olddecl))
C_DECL_THREADPRIVATE_P (newdecl) = 1;
if (CODE_CONTAINS_STRUCT (TREE_CODE (olddecl), TS_DECL_WITH_VIS))
{
COPY_DECL_ASSEMBLER_NAME (olddecl, newdecl);
if (DECL_VISIBILITY_SPECIFIED (olddecl))
{
DECL_VISIBILITY (newdecl) = DECL_VISIBILITY (olddecl);
DECL_VISIBILITY_SPECIFIED (newdecl) = 1;
}
if (TREE_CODE (newdecl) == FUNCTION_DECL)
{
DECL_STATIC_CONSTRUCTOR(newdecl) |= DECL_STATIC_CONSTRUCTOR(olddecl);
DECL_STATIC_DESTRUCTOR (newdecl) |= DECL_STATIC_DESTRUCTOR (olddecl);
DECL_NO_LIMIT_STACK (newdecl) |= DECL_NO_LIMIT_STACK (olddecl);
DECL_NO_INSTRUMENT_FUNCTION_ENTRY_EXIT (newdecl)
|= DECL_NO_INSTRUMENT_FUNCTION_ENTRY_EXIT (olddecl);
TREE_THIS_VOLATILE (newdecl) |= TREE_THIS_VOLATILE (olddecl);
DECL_IS_MALLOC (newdecl) |= DECL_IS_MALLOC (olddecl);
DECL_IS_OPERATOR_NEW (newdecl) |= DECL_IS_OPERATOR_NEW (olddecl);
TREE_READONLY (newdecl) |= TREE_READONLY (olddecl);
DECL_PURE_P (newdecl) |= DECL_PURE_P (olddecl);
DECL_IS_NOVOPS (newdecl) |= DECL_IS_NOVOPS (olddecl);
}
merge_weak (newdecl, olddecl);
if (TREE_CODE (newdecl) == FUNCTION_DECL)
{
TREE_PUBLIC (newdecl) &= TREE_PUBLIC (olddecl);
TREE_PUBLIC (olddecl) = TREE_PUBLIC (newdecl);
if (!TREE_PUBLIC (olddecl))
TREE_PUBLIC (DECL_NAME (olddecl)) = 0;
}
}
if (TREE_CODE (newdecl) == FUNCTION_DECL
&& !flag_gnu89_inline
&& (DECL_DECLARED_INLINE_P (newdecl)
|| DECL_DECLARED_INLINE_P (olddecl))
&& (!DECL_DECLARED_INLINE_P (newdecl)
|| !DECL_DECLARED_INLINE_P (olddecl)
|| !DECL_EXTERNAL (olddecl))
&& DECL_EXTERNAL (newdecl)
&& !lookup_attribute ("gnu_inline", DECL_ATTRIBUTES (newdecl))
&& !current_function_decl)
DECL_EXTERNAL (newdecl) = 0;
if (new_is_definition
&& (DECL_DECLARED_INLINE_P (newdecl)
|| DECL_DECLARED_INLINE_P (olddecl))
&& !TREE_PUBLIC (olddecl))
DECL_EXTERNAL (newdecl) = 0;
if (DECL_EXTERNAL (newdecl))
{
TREE_STATIC (newdecl) = TREE_STATIC (olddecl);
DECL_EXTERNAL (newdecl) = DECL_EXTERNAL (olddecl);
TREE_PUBLIC (newdecl) = TREE_PUBLIC (olddecl);
if (!DECL_EXTERNAL (newdecl))
{
DECL_CONTEXT (newdecl) = DECL_CONTEXT (olddecl);
DECL_COMMON (newdecl) = DECL_COMMON (olddecl);
}
}
else
{
TREE_STATIC (olddecl) = TREE_STATIC (newdecl);
TREE_PUBLIC (olddecl) = TREE_PUBLIC (newdecl);
}
if (TREE_CODE (newdecl) == FUNCTION_DECL)
{
if (new_is_definition && DECL_INITIAL (olddecl))
DECL_UNINLINABLE (newdecl) = 1;
else
{
if (DECL_DECLARED_INLINE_P (newdecl)
|| DECL_DECLARED_INLINE_P (olddecl))
DECL_DECLARED_INLINE_P (newdecl) = 1;
DECL_UNINLINABLE (newdecl) = DECL_UNINLINABLE (olddecl)
= (DECL_UNINLINABLE (newdecl) || DECL_UNINLINABLE (olddecl));
DECL_DISREGARD_INLINE_LIMITS (newdecl)
= DECL_DISREGARD_INLINE_LIMITS (olddecl)
= (DECL_DISREGARD_INLINE_LIMITS (newdecl)
|| DECL_DISREGARD_INLINE_LIMITS (olddecl));
}
if (DECL_BUILT_IN (olddecl))
{
DECL_BUILT_IN_CLASS (newdecl) = DECL_BUILT_IN_CLASS (olddecl);
DECL_FUNCTION_CODE (newdecl) = DECL_FUNCTION_CODE (olddecl);
C_DECL_DECLARED_BUILTIN (newdecl) = 1;
if (new_is_prototype)
{
C_DECL_BUILTIN_PROTOTYPE (newdecl) = 0;
if (DECL_BUILT_IN_CLASS (newdecl) == BUILT_IN_NORMAL)
{
enum built_in_function fncode = DECL_FUNCTION_CODE (newdecl);
switch (fncode)
{
case BUILT_IN_STPCPY:
if (builtin_decl_explicit_p (fncode))
set_builtin_decl_implicit_p (fncode, true);
break;
default:
if (builtin_decl_explicit_p (fncode))
set_builtin_decl_declared_p (fncode, true);
break;
}
copy_attributes_to_builtin (newdecl);
}
}
else
C_DECL_BUILTIN_PROTOTYPE (newdecl)
= C_DECL_BUILTIN_PROTOTYPE (olddecl);
}
if (DECL_FUNCTION_SPECIFIC_TARGET (olddecl)
&& !DECL_FUNCTION_SPECIFIC_TARGET (newdecl))
DECL_FUNCTION_SPECIFIC_TARGET (newdecl)
= DECL_FUNCTION_SPECIFIC_TARGET (olddecl);
if (DECL_FUNCTION_SPECIFIC_OPTIMIZATION (olddecl)
&& !DECL_FUNCTION_SPECIFIC_OPTIMIZATION (newdecl))
DECL_FUNCTION_SPECIFIC_OPTIMIZATION (newdecl)
= DECL_FUNCTION_SPECIFIC_OPTIMIZATION (olddecl);
if (!new_is_definition)
{
tree t;
DECL_RESULT (newdecl) = DECL_RESULT (olddecl);
DECL_INITIAL (newdecl) = DECL_INITIAL (olddecl);
DECL_STRUCT_FUNCTION (newdecl) = DECL_STRUCT_FUNCTION (olddecl);
DECL_SAVED_TREE (newdecl) = DECL_SAVED_TREE (olddecl);
DECL_ARGUMENTS (newdecl) = copy_list (DECL_ARGUMENTS (olddecl));
for (t = DECL_ARGUMENTS (newdecl); t ; t = DECL_CHAIN (t))
DECL_CONTEXT (t) = newdecl;
if (DECL_SAVED_TREE (olddecl))
DECL_ABSTRACT_ORIGIN (newdecl)
= DECL_ABSTRACT_ORIGIN (olddecl);
}
}
if (TREE_USED (olddecl))
TREE_USED (newdecl) = 1;
else if (TREE_USED (newdecl))
TREE_USED (olddecl) = 1;
if (VAR_P (olddecl) || TREE_CODE (olddecl) == PARM_DECL)
DECL_READ_P (newdecl) |= DECL_READ_P (olddecl);
if (DECL_PRESERVE_P (olddecl))
DECL_PRESERVE_P (newdecl) = 1;
else if (DECL_PRESERVE_P (newdecl))
DECL_PRESERVE_P (olddecl) = 1;
if (VAR_P (olddecl) && VAR_P (newdecl)
&& !lookup_attribute ("common", DECL_ATTRIBUTES (newdecl))
&& !lookup_attribute ("nocommon", DECL_ATTRIBUTES (newdecl)))
DECL_COMMON (newdecl) = DECL_COMMON (newdecl) && DECL_COMMON (olddecl);
{
unsigned olddecl_uid = DECL_UID (olddecl);
tree olddecl_context = DECL_CONTEXT (olddecl);
tree olddecl_arguments = NULL;
if (TREE_CODE (olddecl) == FUNCTION_DECL)
olddecl_arguments = DECL_ARGUMENTS (olddecl);
memcpy ((char *) olddecl + sizeof (struct tree_common),
(char *) newdecl + sizeof (struct tree_common),
sizeof (struct tree_decl_common) - sizeof (struct tree_common));
DECL_USER_ALIGN (olddecl) = DECL_USER_ALIGN (newdecl);
switch (TREE_CODE (olddecl))
{
case FUNCTION_DECL:
case VAR_DECL:
{
struct symtab_node *snode = olddecl->decl_with_vis.symtab_node;
memcpy ((char *) olddecl + sizeof (struct tree_decl_common),
(char *) newdecl + sizeof (struct tree_decl_common),
tree_code_size (TREE_CODE (olddecl)) - sizeof (struct tree_decl_common));
olddecl->decl_with_vis.symtab_node = snode;
if ((DECL_EXTERNAL (olddecl)
|| TREE_PUBLIC (olddecl)
|| TREE_STATIC (olddecl))
&& DECL_SECTION_NAME (newdecl) != NULL)
set_decl_section_name (olddecl, DECL_SECTION_NAME (newdecl));
if (VAR_P (olddecl) && DECL_THREAD_LOCAL_P (newdecl))
set_decl_tls_model (olddecl, DECL_TLS_MODEL (newdecl));
break;
}
case FIELD_DECL:
case PARM_DECL:
case LABEL_DECL:
case RESULT_DECL:
case CONST_DECL:
case TYPE_DECL:
memcpy ((char *) olddecl + sizeof (struct tree_decl_common),
(char *) newdecl + sizeof (struct tree_decl_common),
tree_code_size (TREE_CODE (olddecl)) - sizeof (struct tree_decl_common));
break;
default:
memcpy ((char *) olddecl + sizeof (struct tree_decl_common),
(char *) newdecl + sizeof (struct tree_decl_common),
sizeof (struct tree_decl_non_common) - sizeof (struct tree_decl_common));
}
DECL_UID (olddecl) = olddecl_uid;
DECL_CONTEXT (olddecl) = olddecl_context;
if (TREE_CODE (olddecl) == FUNCTION_DECL)
DECL_ARGUMENTS (olddecl) = olddecl_arguments;
}
if (DECL_RTL_SET_P (olddecl)
&& (TREE_CODE (olddecl) == FUNCTION_DECL
|| (VAR_P (olddecl) && TREE_STATIC (olddecl))))
make_decl_rtl (olddecl);
}
static bool
duplicate_decls (tree newdecl, tree olddecl)
{
tree newtype = NULL, oldtype = NULL;
if (!diagnose_mismatched_decls (newdecl, olddecl, &newtype, &oldtype))
{
TREE_NO_WARNING (olddecl) = 1;
return false;
}
merge_decls (newdecl, olddecl, newtype, oldtype);
if (TREE_CODE (newdecl) == FUNCTION_DECL)
DECL_STRUCT_FUNCTION (newdecl) = NULL;
if (VAR_OR_FUNCTION_DECL_P (newdecl))
{
struct symtab_node *snode = symtab_node::get (newdecl);
if (snode)
snode->remove ();
}
ggc_free (newdecl);
return true;
}

static void
warn_if_shadowing (tree new_decl)
{
struct c_binding *b;
if (!(warn_shadow
|| warn_shadow_local
|| warn_shadow_compatible_local)
|| DECL_IS_BUILTIN (new_decl)
|| DECL_FROM_INLINE (new_decl))
return;
for (b = I_SYMBOL_BINDING (DECL_NAME (new_decl)); b; b = b->shadowed)
if (b->decl && b->decl != new_decl && !b->invisible
&& (b->decl == error_mark_node
|| diagnostic_report_warnings_p (global_dc,
DECL_SOURCE_LOCATION (b->decl))))
{
tree old_decl = b->decl;
bool warned = false;
if (old_decl == error_mark_node)
{
warning (OPT_Wshadow, "declaration of %q+D shadows previous "
"non-variable", new_decl);
break;
}
else if (TREE_CODE (old_decl) == PARM_DECL)
{
enum opt_code warning_code;
if (warn_shadow)
warning_code = OPT_Wshadow;
else if (comptypes (TREE_TYPE (old_decl), TREE_TYPE (new_decl)))
warning_code = OPT_Wshadow_compatible_local;
else
warning_code = OPT_Wshadow_local;
warned = warning_at (DECL_SOURCE_LOCATION (new_decl), warning_code,
"declaration of %qD shadows a parameter",
new_decl);
}
else if (DECL_FILE_SCOPE_P (old_decl))
{
if (TREE_CODE (old_decl) == FUNCTION_DECL
&& TREE_CODE (new_decl) != FUNCTION_DECL
&& !FUNCTION_POINTER_TYPE_P (TREE_TYPE (new_decl)))
continue;
warned = warning_at (DECL_SOURCE_LOCATION (new_decl), OPT_Wshadow,
"declaration of %qD shadows a global "
"declaration",
new_decl);
}
else if (TREE_CODE (old_decl) == FUNCTION_DECL
&& DECL_BUILT_IN (old_decl))
{
warning (OPT_Wshadow, "declaration of %q+D shadows "
"a built-in function", new_decl);
break;
}
else
{
enum opt_code warning_code;
if (warn_shadow)
warning_code = OPT_Wshadow;
else if (comptypes (TREE_TYPE (old_decl), TREE_TYPE (new_decl)))
warning_code = OPT_Wshadow_compatible_local;
else
warning_code = OPT_Wshadow_local;
warned = warning_at (DECL_SOURCE_LOCATION (new_decl), warning_code,
"declaration of %qD shadows a previous local",
new_decl);
}
if (warned)
inform (DECL_SOURCE_LOCATION (old_decl),
"shadowed declaration is here");
break;
}
}
tree
pushdecl (tree x)
{
tree name = DECL_NAME (x);
struct c_scope *scope = current_scope;
struct c_binding *b;
bool nested = false;
location_t locus = DECL_SOURCE_LOCATION (x);
if (current_function_decl
&& (!VAR_OR_FUNCTION_DECL_P (x)
|| DECL_INITIAL (x) || !DECL_EXTERNAL (x)))
DECL_CONTEXT (x) = current_function_decl;
if (!name)
{
bind (name, x, scope, false, false,
locus);
return x;
}
b = I_SYMBOL_BINDING (name);
if (b && B_IN_SCOPE (b, scope))
{
struct c_binding *b_ext, *b_use;
tree type = TREE_TYPE (x);
tree visdecl = b->decl;
tree vistype = TREE_TYPE (visdecl);
if (TREE_CODE (TREE_TYPE (x)) == ARRAY_TYPE
&& COMPLETE_TYPE_P (TREE_TYPE (x)))
b->inner_comp = false;
b_use = b;
b_ext = b;
if (TREE_PUBLIC (x) && TREE_PUBLIC (visdecl))
{
while (b_ext && !B_IN_EXTERNAL_SCOPE (b_ext))
b_ext = b_ext->shadowed;
if (b_ext)
{
b_use = b_ext;
if (b_use->u.type)
TREE_TYPE (b_use->decl) = b_use->u.type;
}
}
if (duplicate_decls (x, b_use->decl))
{
if (b_use != b)
{
tree thistype;
if (comptypes (vistype, type))
thistype = composite_type (vistype, type);
else
thistype = TREE_TYPE (b_use->decl);
b_use->u.type = TREE_TYPE (b_use->decl);
if (TREE_CODE (b_use->decl) == FUNCTION_DECL
&& DECL_BUILT_IN (b_use->decl))
thistype
= build_type_attribute_variant (thistype,
TYPE_ATTRIBUTES
(b_use->u.type));
TREE_TYPE (b_use->decl) = thistype;
}
return b_use->decl;
}
else
goto skip_external_and_shadow_checks;
}
if (DECL_EXTERNAL (x) || scope == file_scope)
{
tree type = TREE_TYPE (x);
tree vistype = NULL_TREE;
tree visdecl = NULL_TREE;
bool type_saved = false;
if (b && !B_IN_EXTERNAL_SCOPE (b)
&& VAR_OR_FUNCTION_DECL_P (b->decl)
&& DECL_FILE_SCOPE_P (b->decl))
{
visdecl = b->decl;
vistype = TREE_TYPE (visdecl);
}
if (scope != file_scope
&& !DECL_IN_SYSTEM_HEADER (x))
warning_at (locus, OPT_Wnested_externs,
"nested extern declaration of %qD", x);
while (b && !B_IN_EXTERNAL_SCOPE (b))
{
if (DECL_P (b->decl) && DECL_FILE_SCOPE_P (b->decl) && !type_saved)
{
b->u.type = TREE_TYPE (b->decl);
type_saved = true;
}
if (B_IN_FILE_SCOPE (b)
&& VAR_P (b->decl)
&& TREE_STATIC (b->decl)
&& TREE_CODE (TREE_TYPE (b->decl)) == ARRAY_TYPE
&& !TYPE_DOMAIN (TREE_TYPE (b->decl))
&& TREE_CODE (type) == ARRAY_TYPE
&& TYPE_DOMAIN (type)
&& TYPE_MAX_VALUE (TYPE_DOMAIN (type))
&& !integer_zerop (TYPE_MAX_VALUE (TYPE_DOMAIN (type))))
{
b->inner_comp = true;
}
b = b->shadowed;
}
if (b && (TREE_PUBLIC (x) || same_translation_unit_p (x, b->decl))
&& b->u.type)
TREE_TYPE (b->decl) = b->u.type;
if (b
&& (TREE_PUBLIC (x) || same_translation_unit_p (x, b->decl))
&& duplicate_decls (x, b->decl))
{
tree thistype;
if (vistype)
{
if (comptypes (vistype, type))
thistype = composite_type (vistype, type);
else
thistype = TREE_TYPE (b->decl);
}
else
thistype = type;
b->u.type = TREE_TYPE (b->decl);
if (TREE_CODE (b->decl) == FUNCTION_DECL && DECL_BUILT_IN (b->decl))
thistype
= build_type_attribute_variant (thistype,
TYPE_ATTRIBUTES (b->u.type));
TREE_TYPE (b->decl) = thistype;
bind (name, b->decl, scope, false, true,
locus);
return b->decl;
}
else if (TREE_PUBLIC (x))
{
if (visdecl && !b && duplicate_decls (x, visdecl))
{
nested = true;
x = visdecl;
}
else
{
bind (name, x, external_scope, true,
false, locus);
nested = true;
}
}
}
if (TREE_CODE (x) != PARM_DECL)
warn_if_shadowing (x);
skip_external_and_shadow_checks:
if (TREE_CODE (x) == TYPE_DECL)
{
set_underlying_type (x);
record_locally_defined_typedef (x);
}
bind (name, x, scope, false, nested, locus);
if (TREE_TYPE (x) != error_mark_node
&& !COMPLETE_TYPE_P (TREE_TYPE (x)))
{
tree element = TREE_TYPE (x);
while (TREE_CODE (element) == ARRAY_TYPE)
element = TREE_TYPE (element);
element = TYPE_MAIN_VARIANT (element);
if (RECORD_OR_UNION_TYPE_P (element)
&& (TREE_CODE (x) != TYPE_DECL
|| TREE_CODE (TREE_TYPE (x)) == ARRAY_TYPE)
&& !COMPLETE_TYPE_P (element))
C_TYPE_INCOMPLETE_VARS (element)
= tree_cons (NULL_TREE, x, C_TYPE_INCOMPLETE_VARS (element));
}
return x;
}

static void
implicit_decl_warning (location_t loc, tree id, tree olddecl)
{
if (!warn_implicit_function_declaration)
return;
bool warned;
name_hint hint;
if (!olddecl)
hint = lookup_name_fuzzy (id, FUZZY_LOOKUP_FUNCTION_NAME, loc);
if (flag_isoc99)
{
if (hint)
{
gcc_rich_location richloc (loc);
richloc.add_fixit_replace (hint.suggestion ());
warned = pedwarn (&richloc, OPT_Wimplicit_function_declaration,
"implicit declaration of function %qE;"
" did you mean %qs?",
id, hint.suggestion ());
}
else
warned = pedwarn (loc, OPT_Wimplicit_function_declaration,
"implicit declaration of function %qE", id);
}
else if (hint)
{
gcc_rich_location richloc (loc);
richloc.add_fixit_replace (hint.suggestion ());
warned = warning_at
(&richloc, OPT_Wimplicit_function_declaration,
G_("implicit declaration of function %qE; did you mean %qs?"),
id, hint.suggestion ());
}
else
warned = warning_at (loc, OPT_Wimplicit_function_declaration,
G_("implicit declaration of function %qE"), id);
if (olddecl && warned)
locate_old_decl (olddecl);
if (!warned)
hint.suppress ();
}
static const char *
header_for_builtin_fn (enum built_in_function fcode)
{
switch (fcode)
{
CASE_FLT_FN (BUILT_IN_ACOS):
CASE_FLT_FN (BUILT_IN_ACOSH):
CASE_FLT_FN (BUILT_IN_ASIN):
CASE_FLT_FN (BUILT_IN_ASINH):
CASE_FLT_FN (BUILT_IN_ATAN):
CASE_FLT_FN (BUILT_IN_ATANH):
CASE_FLT_FN (BUILT_IN_ATAN2):
CASE_FLT_FN (BUILT_IN_CBRT):
CASE_FLT_FN (BUILT_IN_CEIL):
CASE_FLT_FN_FLOATN_NX (BUILT_IN_CEIL):
CASE_FLT_FN (BUILT_IN_COPYSIGN):
CASE_FLT_FN_FLOATN_NX (BUILT_IN_COPYSIGN):
CASE_FLT_FN (BUILT_IN_COS):
CASE_FLT_FN (BUILT_IN_COSH):
CASE_FLT_FN (BUILT_IN_ERF):
CASE_FLT_FN (BUILT_IN_ERFC):
CASE_FLT_FN (BUILT_IN_EXP):
CASE_FLT_FN (BUILT_IN_EXP2):
CASE_FLT_FN (BUILT_IN_EXPM1):
CASE_FLT_FN (BUILT_IN_FABS):
CASE_FLT_FN_FLOATN_NX (BUILT_IN_FABS):
CASE_FLT_FN (BUILT_IN_FDIM):
CASE_FLT_FN (BUILT_IN_FLOOR):
CASE_FLT_FN_FLOATN_NX (BUILT_IN_FLOOR):
CASE_FLT_FN (BUILT_IN_FMA):
CASE_FLT_FN_FLOATN_NX (BUILT_IN_FMA):
CASE_FLT_FN (BUILT_IN_FMAX):
CASE_FLT_FN_FLOATN_NX (BUILT_IN_FMAX):
CASE_FLT_FN (BUILT_IN_FMIN):
CASE_FLT_FN_FLOATN_NX (BUILT_IN_FMIN):
CASE_FLT_FN (BUILT_IN_FMOD):
CASE_FLT_FN (BUILT_IN_FREXP):
CASE_FLT_FN (BUILT_IN_HYPOT):
CASE_FLT_FN (BUILT_IN_ILOGB):
CASE_FLT_FN (BUILT_IN_LDEXP):
CASE_FLT_FN (BUILT_IN_LGAMMA):
CASE_FLT_FN (BUILT_IN_LLRINT):
CASE_FLT_FN (BUILT_IN_LLROUND):
CASE_FLT_FN (BUILT_IN_LOG):
CASE_FLT_FN (BUILT_IN_LOG10):
CASE_FLT_FN (BUILT_IN_LOG1P):
CASE_FLT_FN (BUILT_IN_LOG2):
CASE_FLT_FN (BUILT_IN_LOGB):
CASE_FLT_FN (BUILT_IN_LRINT):
CASE_FLT_FN (BUILT_IN_LROUND):
CASE_FLT_FN (BUILT_IN_MODF):
CASE_FLT_FN (BUILT_IN_NAN):
CASE_FLT_FN (BUILT_IN_NEARBYINT):
CASE_FLT_FN_FLOATN_NX (BUILT_IN_NEARBYINT):
CASE_FLT_FN (BUILT_IN_NEXTAFTER):
CASE_FLT_FN (BUILT_IN_NEXTTOWARD):
CASE_FLT_FN (BUILT_IN_POW):
CASE_FLT_FN (BUILT_IN_REMAINDER):
CASE_FLT_FN (BUILT_IN_REMQUO):
CASE_FLT_FN (BUILT_IN_RINT):
CASE_FLT_FN_FLOATN_NX (BUILT_IN_RINT):
CASE_FLT_FN (BUILT_IN_ROUND):
CASE_FLT_FN_FLOATN_NX (BUILT_IN_ROUND):
CASE_FLT_FN (BUILT_IN_SCALBLN):
CASE_FLT_FN (BUILT_IN_SCALBN):
CASE_FLT_FN (BUILT_IN_SIN):
CASE_FLT_FN (BUILT_IN_SINH):
CASE_FLT_FN (BUILT_IN_SINCOS):
CASE_FLT_FN (BUILT_IN_SQRT):
CASE_FLT_FN_FLOATN_NX (BUILT_IN_SQRT):
CASE_FLT_FN (BUILT_IN_TAN):
CASE_FLT_FN (BUILT_IN_TANH):
CASE_FLT_FN (BUILT_IN_TGAMMA):
CASE_FLT_FN (BUILT_IN_TRUNC):
CASE_FLT_FN_FLOATN_NX (BUILT_IN_TRUNC):
case BUILT_IN_ISINF:
case BUILT_IN_ISNAN:
return "<math.h>";
CASE_FLT_FN (BUILT_IN_CABS):
CASE_FLT_FN (BUILT_IN_CACOS):
CASE_FLT_FN (BUILT_IN_CACOSH):
CASE_FLT_FN (BUILT_IN_CARG):
CASE_FLT_FN (BUILT_IN_CASIN):
CASE_FLT_FN (BUILT_IN_CASINH):
CASE_FLT_FN (BUILT_IN_CATAN):
CASE_FLT_FN (BUILT_IN_CATANH):
CASE_FLT_FN (BUILT_IN_CCOS):
CASE_FLT_FN (BUILT_IN_CCOSH):
CASE_FLT_FN (BUILT_IN_CEXP):
CASE_FLT_FN (BUILT_IN_CIMAG):
CASE_FLT_FN (BUILT_IN_CLOG):
CASE_FLT_FN (BUILT_IN_CONJ):
CASE_FLT_FN (BUILT_IN_CPOW):
CASE_FLT_FN (BUILT_IN_CPROJ):
CASE_FLT_FN (BUILT_IN_CREAL):
CASE_FLT_FN (BUILT_IN_CSIN):
CASE_FLT_FN (BUILT_IN_CSINH):
CASE_FLT_FN (BUILT_IN_CSQRT):
CASE_FLT_FN (BUILT_IN_CTAN):
CASE_FLT_FN (BUILT_IN_CTANH):
return "<complex.h>";
case BUILT_IN_MEMCHR:
case BUILT_IN_MEMCMP:
case BUILT_IN_MEMCPY:
case BUILT_IN_MEMMOVE:
case BUILT_IN_MEMSET:
case BUILT_IN_STRCAT:
case BUILT_IN_STRCHR:
case BUILT_IN_STRCMP:
case BUILT_IN_STRCPY:
case BUILT_IN_STRCSPN:
case BUILT_IN_STRLEN:
case BUILT_IN_STRNCAT:
case BUILT_IN_STRNCMP:
case BUILT_IN_STRNCPY:
case BUILT_IN_STRPBRK:
case BUILT_IN_STRRCHR:
case BUILT_IN_STRSPN:
case BUILT_IN_STRSTR:
return "<string.h>";
case BUILT_IN_FPRINTF:
case BUILT_IN_PUTC:
case BUILT_IN_FPUTC:
case BUILT_IN_FPUTS:
case BUILT_IN_FSCANF:
case BUILT_IN_FWRITE:
case BUILT_IN_PRINTF:
case BUILT_IN_PUTCHAR:
case BUILT_IN_PUTS:
case BUILT_IN_SCANF:
case BUILT_IN_SNPRINTF:
case BUILT_IN_SPRINTF:
case BUILT_IN_SSCANF:
case BUILT_IN_VFPRINTF:
case BUILT_IN_VFSCANF:
case BUILT_IN_VPRINTF:
case BUILT_IN_VSCANF:
case BUILT_IN_VSNPRINTF:
case BUILT_IN_VSPRINTF:
case BUILT_IN_VSSCANF:
return "<stdio.h>";
case BUILT_IN_ISALNUM:
case BUILT_IN_ISALPHA:
case BUILT_IN_ISBLANK:
case BUILT_IN_ISCNTRL:
case BUILT_IN_ISDIGIT:
case BUILT_IN_ISGRAPH:
case BUILT_IN_ISLOWER:
case BUILT_IN_ISPRINT:
case BUILT_IN_ISPUNCT:
case BUILT_IN_ISSPACE:
case BUILT_IN_ISUPPER:
case BUILT_IN_ISXDIGIT:
case BUILT_IN_TOLOWER:
case BUILT_IN_TOUPPER:
return "<ctype.h>";
case BUILT_IN_ISWALNUM:
case BUILT_IN_ISWALPHA:
case BUILT_IN_ISWBLANK:
case BUILT_IN_ISWCNTRL:
case BUILT_IN_ISWDIGIT:
case BUILT_IN_ISWGRAPH:
case BUILT_IN_ISWLOWER:
case BUILT_IN_ISWPRINT:
case BUILT_IN_ISWPUNCT:
case BUILT_IN_ISWSPACE:
case BUILT_IN_ISWUPPER:
case BUILT_IN_ISWXDIGIT:
case BUILT_IN_TOWLOWER:
case BUILT_IN_TOWUPPER:
return "<wctype.h>";
case BUILT_IN_ABORT:
case BUILT_IN_ABS:
case BUILT_IN_CALLOC:
case BUILT_IN_EXIT:
case BUILT_IN_FREE:
case BUILT_IN_LABS:
case BUILT_IN_LLABS:
case BUILT_IN_MALLOC:
case BUILT_IN_REALLOC:
case BUILT_IN__EXIT2:
case BUILT_IN_ALIGNED_ALLOC:
return "<stdlib.h>";
case BUILT_IN_IMAXABS:
return "<inttypes.h>";
case BUILT_IN_STRFTIME:
return "<time.h>";
default:
return NULL;
}
}
tree
implicitly_declare (location_t loc, tree functionid)
{
struct c_binding *b;
tree decl = NULL_TREE;
tree asmspec_tree;
for (b = I_SYMBOL_BINDING (functionid); b; b = b->shadowed)
{
if (B_IN_SCOPE (b, external_scope))
{
decl = b->decl;
break;
}
}
if (decl)
{
if (TREE_CODE (decl) != FUNCTION_DECL)
return decl;
if (!DECL_BUILT_IN (decl) && DECL_IS_BUILTIN (decl))
{
bind (functionid, decl, file_scope,
false, true,
DECL_SOURCE_LOCATION (decl));
return decl;
}
else
{
tree newtype = default_function_type;
if (b->u.type)
TREE_TYPE (decl) = b->u.type;
if (!C_DECL_IMPLICIT (decl))
{
implicit_decl_warning (loc, functionid, decl);
C_DECL_IMPLICIT (decl) = 1;
}
if (DECL_BUILT_IN (decl))
{
newtype = build_type_attribute_variant (newtype,
TYPE_ATTRIBUTES
(TREE_TYPE (decl)));
if (!comptypes (newtype, TREE_TYPE (decl)))
{
bool warned = warning_at (loc, 0, "incompatible implicit "
"declaration of built-in "
"function %qD", decl);
const char *header
= header_for_builtin_fn (DECL_FUNCTION_CODE (decl));
if (header != NULL && warned)
{
rich_location richloc (line_table, loc);
maybe_add_include_fixit (&richloc, header);
inform (&richloc,
"include %qs or provide a declaration of %qD",
header, decl);
}
newtype = TREE_TYPE (decl);
}
}
else
{
if (!comptypes (newtype, TREE_TYPE (decl)))
{
error_at (loc, "incompatible implicit declaration of "
"function %qD", decl);
locate_old_decl (decl);
}
}
b->u.type = TREE_TYPE (decl);
TREE_TYPE (decl) = newtype;
bind (functionid, decl, current_scope,
false, true,
DECL_SOURCE_LOCATION (decl));
return decl;
}
}
decl = build_decl (loc, FUNCTION_DECL, functionid, default_function_type);
DECL_EXTERNAL (decl) = 1;
TREE_PUBLIC (decl) = 1;
C_DECL_IMPLICIT (decl) = 1;
implicit_decl_warning (loc, functionid, 0);
asmspec_tree = maybe_apply_renaming_pragma (decl, NULL);
if (asmspec_tree)
set_user_assembler_name (decl, TREE_STRING_POINTER (asmspec_tree));
decl = pushdecl (decl);
rest_of_decl_compilation (decl, 0, 0);
gen_aux_info_record (decl, 0, 1, 0);
decl_attributes (&decl, NULL_TREE, 0);
return decl;
}
void
undeclared_variable (location_t loc, tree id)
{
static bool already = false;
struct c_scope *scope;
if (current_function_decl == NULL_TREE)
{
name_hint guessed_id = lookup_name_fuzzy (id, FUZZY_LOOKUP_NAME, loc);
if (guessed_id)
{
gcc_rich_location richloc (loc);
richloc.add_fixit_replace (guessed_id.suggestion ());
error_at (&richloc,
"%qE undeclared here (not in a function);"
" did you mean %qs?",
id, guessed_id.suggestion ());
}
else
error_at (loc, "%qE undeclared here (not in a function)", id);
scope = current_scope;
}
else
{
if (!objc_diagnose_private_ivar (id))
{
name_hint guessed_id = lookup_name_fuzzy (id, FUZZY_LOOKUP_NAME, loc);
if (guessed_id)
{
gcc_rich_location richloc (loc);
richloc.add_fixit_replace (guessed_id.suggestion ());
error_at (&richloc,
"%qE undeclared (first use in this function);"
" did you mean %qs?",
id, guessed_id.suggestion ());
}
else
error_at (loc, "%qE undeclared (first use in this function)", id);
}
if (!already)
{
inform (loc, "each undeclared identifier is reported only"
" once for each function it appears in");
already = true;
}
scope = current_function_scope ? current_function_scope : current_scope;
}
bind (id, error_mark_node, scope, false, false,
UNKNOWN_LOCATION);
}

static tree
make_label (location_t location, tree name, bool defining,
struct c_label_vars **p_label_vars)
{
tree label = build_decl (location, LABEL_DECL, name, void_type_node);
DECL_CONTEXT (label) = current_function_decl;
SET_DECL_MODE (label, VOIDmode);
c_label_vars *label_vars = ggc_alloc<c_label_vars> ();
label_vars->shadowed = NULL;
set_spot_bindings (&label_vars->label_bindings, defining);
label_vars->decls_in_scope = make_tree_vector ();
label_vars->gotos = NULL;
*p_label_vars = label_vars;
return label;
}
tree
lookup_label (tree name)
{
tree label;
struct c_label_vars *label_vars;
if (current_function_scope == 0)
{
error ("label %qE referenced outside of any function", name);
return NULL_TREE;
}
label = I_LABEL_DECL (name);
if (label && (DECL_CONTEXT (label) == current_function_decl
|| C_DECLARED_LABEL_FLAG (label)))
{
if (DECL_INITIAL (label) == NULL_TREE)
DECL_SOURCE_LOCATION (label) = input_location;
return label;
}
label = make_label (input_location, name, false, &label_vars);
bind_label (name, label, current_function_scope, label_vars);
return label;
}
static void
warn_about_goto (location_t goto_loc, tree label, tree decl)
{
if (variably_modified_type_p (TREE_TYPE (decl), NULL_TREE))
error_at (goto_loc,
"jump into scope of identifier with variably modified type");
else
warning_at (goto_loc, OPT_Wjump_misses_init,
"jump skips variable initialization");
inform (DECL_SOURCE_LOCATION (label), "label %qD defined here", label);
inform (DECL_SOURCE_LOCATION (decl), "%qD declared here", decl);
}
tree
lookup_label_for_goto (location_t loc, tree name)
{
tree label;
struct c_label_vars *label_vars;
unsigned int ix;
tree decl;
label = lookup_label (name);
if (label == NULL_TREE)
return NULL_TREE;
if (DECL_CONTEXT (label) != current_function_decl)
{
gcc_assert (C_DECLARED_LABEL_FLAG (label));
return label;
}
label_vars = I_LABEL_BINDING (name)->u.label;
if (label_vars->label_bindings.scope == NULL)
{
c_goto_bindings *g = ggc_alloc<c_goto_bindings> ();
g->loc = loc;
set_spot_bindings (&g->goto_bindings, true);
vec_safe_push (label_vars->gotos, g);
return label;
}
FOR_EACH_VEC_SAFE_ELT (label_vars->decls_in_scope, ix, decl)
warn_about_goto (loc, label, decl);
if (label_vars->label_bindings.left_stmt_expr)
{
error_at (loc, "jump into statement expression");
inform (DECL_SOURCE_LOCATION (label), "label %qD defined here", label);
}
return label;
}
tree
declare_label (tree name)
{
struct c_binding *b = I_LABEL_BINDING (name);
tree label;
struct c_label_vars *label_vars;
if (b && B_IN_CURRENT_SCOPE (b))
{
error ("duplicate label declaration %qE", name);
locate_old_decl (b->decl);
return b->decl;
}
label = make_label (input_location, name, false, &label_vars);
C_DECLARED_LABEL_FLAG (label) = 1;
bind_label (name, label, current_scope, label_vars);
return label;
}
static void
check_earlier_gotos (tree label, struct c_label_vars* label_vars)
{
unsigned int ix;
struct c_goto_bindings *g;
FOR_EACH_VEC_SAFE_ELT (label_vars->gotos, ix, g)
{
struct c_binding *b;
struct c_scope *scope;
if (g->goto_bindings.scope->has_jump_unsafe_decl)
{
for (b = g->goto_bindings.scope->bindings;
b != g->goto_bindings.bindings_in_scope;
b = b->prev)
{
if (decl_jump_unsafe (b->decl))
warn_about_goto (g->loc, label, b->decl);
}
}
for (scope = label_vars->label_bindings.scope;
scope != g->goto_bindings.scope;
scope = scope->outer)
{
gcc_assert (scope != NULL);
if (scope->has_jump_unsafe_decl)
{
if (scope == label_vars->label_bindings.scope)
b = label_vars->label_bindings.bindings_in_scope;
else
b = scope->bindings;
for (; b != NULL; b = b->prev)
{
if (decl_jump_unsafe (b->decl))
warn_about_goto (g->loc, label, b->decl);
}
}
}
if (g->goto_bindings.stmt_exprs > 0)
{
error_at (g->loc, "jump into statement expression");
inform (DECL_SOURCE_LOCATION (label), "label %qD defined here",
label);
}
}
vec_safe_truncate (label_vars->gotos, 0);
label_vars->gotos = NULL;
}
tree
define_label (location_t location, tree name)
{
tree label = I_LABEL_DECL (name);
if (label
&& ((DECL_CONTEXT (label) == current_function_decl
&& DECL_INITIAL (label) != NULL_TREE)
|| (DECL_CONTEXT (label) != current_function_decl
&& C_DECLARED_LABEL_FLAG (label))))
{
error_at (location, "duplicate label %qD", label);
locate_old_decl (label);
return NULL_TREE;
}
else if (label && DECL_CONTEXT (label) == current_function_decl)
{
struct c_label_vars *label_vars = I_LABEL_BINDING (name)->u.label;
DECL_SOURCE_LOCATION (label) = location;
set_spot_bindings (&label_vars->label_bindings, true);
check_earlier_gotos (label, label_vars);
}
else
{
struct c_label_vars *label_vars;
label = make_label (location, name, true, &label_vars);
bind_label (name, label, current_function_scope, label_vars);
}
if (!in_system_header_at (input_location) && lookup_name (name))
warning_at (location, OPT_Wtraditional,
"traditional C lacks a separate namespace "
"for labels, identifier %qE conflicts", name);
DECL_INITIAL (label) = error_mark_node;
return label;
}

struct c_spot_bindings *
c_get_switch_bindings (void)
{
struct c_spot_bindings *switch_bindings;
switch_bindings = XNEW (struct c_spot_bindings);
set_spot_bindings (switch_bindings, true);
return switch_bindings;
}
void
c_release_switch_bindings (struct c_spot_bindings *bindings)
{
gcc_assert (bindings->stmt_exprs == 0 && !bindings->left_stmt_expr);
XDELETE (bindings);
}
bool
c_check_switch_jump_warnings (struct c_spot_bindings *switch_bindings,
location_t switch_loc, location_t case_loc)
{
bool saw_error;
struct c_scope *scope;
saw_error = false;
for (scope = current_scope;
scope != switch_bindings->scope;
scope = scope->outer)
{
struct c_binding *b;
gcc_assert (scope != NULL);
if (!scope->has_jump_unsafe_decl)
continue;
for (b = scope->bindings; b != NULL; b = b->prev)
{
if (decl_jump_unsafe (b->decl))
{
if (variably_modified_type_p (TREE_TYPE (b->decl), NULL_TREE))
{
saw_error = true;
error_at (case_loc,
("switch jumps into scope of identifier with "
"variably modified type"));
}
else
warning_at (case_loc, OPT_Wjump_misses_init,
"switch jumps over variable initialization");
inform (switch_loc, "switch starts here");
inform (DECL_SOURCE_LOCATION (b->decl), "%qD declared here",
b->decl);
}
}
}
if (switch_bindings->stmt_exprs > 0)
{
saw_error = true;
error_at (case_loc, "switch jumps into statement expression");
inform (switch_loc, "switch starts here");
}
return saw_error;
}

static tree
lookup_tag (enum tree_code code, tree name, bool thislevel_only,
location_t *ploc)
{
struct c_binding *b = I_TAG_BINDING (name);
bool thislevel = false;
if (!b || !b->decl)
return NULL_TREE;
if (thislevel_only || TREE_CODE (b->decl) != code)
{
if (B_IN_CURRENT_SCOPE (b)
|| (current_scope == file_scope && B_IN_EXTERNAL_SCOPE (b)))
thislevel = true;
}
if (thislevel_only && !thislevel)
return NULL_TREE;
if (TREE_CODE (b->decl) != code)
{
pending_invalid_xref = name;
pending_invalid_xref_location = input_location;
if (thislevel)
pending_xref_error ();
}
if (ploc != NULL)
*ploc = b->locus;
return b->decl;
}
bool
tag_exists_p (enum tree_code code, tree name)
{
struct c_binding *b = I_TAG_BINDING (name);
if (b == NULL || b->decl == NULL_TREE)
return false;
return TREE_CODE (b->decl) == code;
}
void
pending_xref_error (void)
{
if (pending_invalid_xref != NULL_TREE)
error_at (pending_invalid_xref_location, "%qE defined as wrong kind of tag",
pending_invalid_xref);
pending_invalid_xref = NULL_TREE;
}

tree
lookup_name (tree name)
{
struct c_binding *b = I_SYMBOL_BINDING (name);
if (b && !b->invisible)
{
maybe_record_typedef_use (b->decl);
return b->decl;
}
return NULL_TREE;
}
static tree
lookup_name_in_scope (tree name, struct c_scope *scope)
{
struct c_binding *b;
for (b = I_SYMBOL_BINDING (name); b; b = b->shadowed)
if (B_IN_SCOPE (b, scope))
return b->decl;
return NULL_TREE;
}
name_hint
lookup_name_fuzzy (tree name, enum lookup_name_fuzzy_kind kind, location_t loc)
{
gcc_assert (TREE_CODE (name) == IDENTIFIER_NODE);
const char *header_hint
= get_c_stdlib_header_for_name (IDENTIFIER_POINTER (name));
if (header_hint)
return name_hint (NULL,
new suggest_missing_header (loc,
IDENTIFIER_POINTER (name),
header_hint));
bool consider_implementation_names = (IDENTIFIER_POINTER (name)[0] == '_');
best_match<tree, tree> bm (name);
for (c_scope *scope = current_scope; scope; scope = scope->outer)
for (c_binding *binding = scope->bindings; binding; binding = binding->prev)
{
if (!binding->id || binding->invisible)
continue;
if (binding->decl == error_mark_node)
continue;
if (TREE_CODE (binding->decl) == FUNCTION_DECL)
if (C_DECL_IMPLICIT (binding->decl))
continue;
if (!consider_implementation_names)
{
const char *suggestion_str = IDENTIFIER_POINTER (binding->id);
if (name_reserved_for_implementation_p (suggestion_str))
continue;
}
switch (kind)
{
case FUZZY_LOOKUP_TYPENAME:
if (TREE_CODE (binding->decl) != TYPE_DECL)
continue;
break;
case FUZZY_LOOKUP_FUNCTION_NAME:
if (TREE_CODE (binding->decl) != FUNCTION_DECL)
{
if ((VAR_P (binding->decl)
|| TREE_CODE (binding->decl) == PARM_DECL)
&& TREE_CODE (TREE_TYPE (binding->decl)) == POINTER_TYPE
&& (TREE_CODE (TREE_TYPE (TREE_TYPE (binding->decl)))
== FUNCTION_TYPE))
break;
continue;
}
break;
default:
break;
}
bm.consider (binding->id);
}
best_macro_match bmm (name, bm.get_best_distance (), parse_in);
cpp_hashnode *best_macro = bmm.get_best_meaningful_candidate ();
if (best_macro)
{
const char *id = (const char *)best_macro->ident.str;
tree macro_as_identifier
= get_identifier_with_length (id, best_macro->ident.len);
bm.set_best_so_far (macro_as_identifier,
bmm.get_best_distance (),
bmm.get_best_candidate_length ());
}
if (kind == FUZZY_LOOKUP_TYPENAME)
{
for (unsigned i = 0; i < num_c_common_reswords; i++)
{
const c_common_resword *resword = &c_common_reswords[i];
if (!c_keyword_starts_typename (resword->rid))
continue;
tree resword_identifier = ridpointers [resword->rid];
if (!resword_identifier)
continue;
gcc_assert (TREE_CODE (resword_identifier) == IDENTIFIER_NODE);
bm.consider (resword_identifier);
}
}
tree best = bm.get_best_meaningful_candidate ();
if (best)
return name_hint (IDENTIFIER_POINTER (best), NULL);
else
return name_hint (NULL, NULL);
}

void
c_init_decl_processing (void)
{
location_t save_loc = input_location;
c_parse_init ();
current_function_decl = NULL_TREE;
gcc_obstack_init (&parser_obstack);
push_scope ();
external_scope = current_scope;
input_location = BUILTINS_LOCATION;
c_common_nodes_and_builtins ();
truthvalue_type_node = integer_type_node;
truthvalue_true_node = integer_one_node;
truthvalue_false_node = integer_zero_node;
pushdecl (build_decl (UNKNOWN_LOCATION, TYPE_DECL, get_identifier ("_Bool"),
boolean_type_node));
input_location = save_loc;
make_fname_decl = c_make_fname_decl;
start_fname_decls ();
}
static tree
c_make_fname_decl (location_t loc, tree id, int type_dep)
{
const char *name = fname_as_string (type_dep);
tree decl, type, init;
size_t length = strlen (name);
type = build_array_type (char_type_node,
build_index_type (size_int (length)));
type = c_build_qualified_type (type, TYPE_QUAL_CONST);
decl = build_decl (loc, VAR_DECL, id, type);
TREE_STATIC (decl) = 1;
TREE_READONLY (decl) = 1;
DECL_ARTIFICIAL (decl) = 1;
init = build_string (length + 1, name);
free (CONST_CAST (char *, name));
TREE_TYPE (init) = type;
DECL_INITIAL (decl) = init;
TREE_USED (decl) = 1;
if (current_function_decl
&& current_function_scope)
{
DECL_CONTEXT (decl) = current_function_decl;
bind (id, decl, current_function_scope,
false, false, UNKNOWN_LOCATION);
}
finish_decl (decl, loc, init, NULL_TREE, NULL_TREE);
return decl;
}
tree
c_builtin_function (tree decl)
{
tree type = TREE_TYPE (decl);
tree   id = DECL_NAME (decl);
const char *name = IDENTIFIER_POINTER (id);
C_DECL_BUILTIN_PROTOTYPE (decl) = prototype_p (type);
gcc_assert (!I_SYMBOL_BINDING (id));
bind (id, decl, external_scope, true, false,
UNKNOWN_LOCATION);
if (name[0] == '_' && (name[1] == '_' || ISUPPER (name[1])))
{
DECL_CHAIN (decl) = visible_builtins;
visible_builtins = decl;
}
return decl;
}
tree
c_builtin_function_ext_scope (tree decl)
{
tree type = TREE_TYPE (decl);
tree   id = DECL_NAME (decl);
const char *name = IDENTIFIER_POINTER (id);
C_DECL_BUILTIN_PROTOTYPE (decl) = prototype_p (type);
if (external_scope)
bind (id, decl, external_scope, false, false,
UNKNOWN_LOCATION);
if (name[0] == '_' && (name[1] == '_' || ISUPPER (name[1])))
{
DECL_CHAIN (decl) = visible_builtins;
visible_builtins = decl;
}
return decl;
}

void
shadow_tag (const struct c_declspecs *declspecs)
{
shadow_tag_warned (declspecs, 0);
}
void
shadow_tag_warned (const struct c_declspecs *declspecs, int warned)
{
bool found_tag = false;
if (declspecs->type && !declspecs->default_int_p && !declspecs->typedef_p)
{
tree value = declspecs->type;
enum tree_code code = TREE_CODE (value);
if (code == RECORD_TYPE || code == UNION_TYPE || code == ENUMERAL_TYPE)
{
tree name = TYPE_NAME (value);
tree t;
found_tag = true;
if (declspecs->restrict_p)
{
error ("invalid use of %<restrict%>");
warned = 1;
}
if (name == NULL_TREE)
{
if (warned != 1 && code != ENUMERAL_TYPE)
{
pedwarn (input_location, 0,
"unnamed struct/union that defines no instances");
warned = 1;
}
}
else if (declspecs->typespec_kind != ctsk_tagdef
&& declspecs->typespec_kind != ctsk_tagfirstref
&& declspecs->storage_class != csc_none)
{
if (warned != 1)
pedwarn (input_location, 0,
"empty declaration with storage class specifier "
"does not redeclare tag");
warned = 1;
pending_xref_error ();
}
else if (declspecs->typespec_kind != ctsk_tagdef
&& declspecs->typespec_kind != ctsk_tagfirstref
&& (declspecs->const_p
|| declspecs->volatile_p
|| declspecs->atomic_p
|| declspecs->restrict_p
|| declspecs->address_space))
{
if (warned != 1)
pedwarn (input_location, 0,
"empty declaration with type qualifier "
"does not redeclare tag");
warned = 1;
pending_xref_error ();
}
else if (declspecs->typespec_kind != ctsk_tagdef
&& declspecs->typespec_kind != ctsk_tagfirstref
&& declspecs->alignas_p)
{
if (warned != 1)
pedwarn (input_location, 0,
"empty declaration with %<_Alignas%> "
"does not redeclare tag");
warned = 1;
pending_xref_error ();
}
else
{
pending_invalid_xref = NULL_TREE;
t = lookup_tag (code, name, true, NULL);
if (t == NULL_TREE)
{
t = make_node (code);
pushtag (input_location, name, t);
}
}
}
else
{
if (warned != 1 && !in_system_header_at (input_location))
{
pedwarn (input_location, 0,
"useless type name in empty declaration");
warned = 1;
}
}
}
else if (warned != 1 && !in_system_header_at (input_location)
&& declspecs->typedef_p)
{
pedwarn (input_location, 0, "useless type name in empty declaration");
warned = 1;
}
pending_invalid_xref = NULL_TREE;
if (declspecs->inline_p)
{
error ("%<inline%> in empty declaration");
warned = 1;
}
if (declspecs->noreturn_p)
{
error ("%<_Noreturn%> in empty declaration");
warned = 1;
}
if (current_scope == file_scope && declspecs->storage_class == csc_auto)
{
error ("%<auto%> in file-scope empty declaration");
warned = 1;
}
if (current_scope == file_scope && declspecs->storage_class == csc_register)
{
error ("%<register%> in file-scope empty declaration");
warned = 1;
}
if (!warned && !in_system_header_at (input_location)
&& declspecs->storage_class != csc_none)
{
warning (0, "useless storage class specifier in empty declaration");
warned = 2;
}
if (!warned && !in_system_header_at (input_location) && declspecs->thread_p)
{
warning (0, "useless %qs in empty declaration",
declspecs->thread_gnu_p ? "__thread" : "_Thread_local");
warned = 2;
}
if (!warned
&& !in_system_header_at (input_location)
&& (declspecs->const_p
|| declspecs->volatile_p
|| declspecs->atomic_p
|| declspecs->restrict_p
|| declspecs->address_space))
{
warning (0, "useless type qualifier in empty declaration");
warned = 2;
}
if (!warned && !in_system_header_at (input_location)
&& declspecs->alignas_p)
{
warning (0, "useless %<_Alignas%> in empty declaration");
warned = 2;
}
if (warned != 1)
{
if (!found_tag)
pedwarn (input_location, 0, "empty declaration");
}
}

int
quals_from_declspecs (const struct c_declspecs *specs)
{
int quals = ((specs->const_p ? TYPE_QUAL_CONST : 0)
| (specs->volatile_p ? TYPE_QUAL_VOLATILE : 0)
| (specs->restrict_p ? TYPE_QUAL_RESTRICT : 0)
| (specs->atomic_p ? TYPE_QUAL_ATOMIC : 0)
| (ENCODE_QUAL_ADDR_SPACE (specs->address_space)));
gcc_assert (!specs->type
&& !specs->decl_attr
&& specs->typespec_word == cts_none
&& specs->storage_class == csc_none
&& !specs->typedef_p
&& !specs->explicit_signed_p
&& !specs->deprecated_p
&& !specs->long_p
&& !specs->long_long_p
&& !specs->short_p
&& !specs->signed_p
&& !specs->unsigned_p
&& !specs->complex_p
&& !specs->inline_p
&& !specs->noreturn_p
&& !specs->thread_p);
return quals;
}
struct c_declarator *
build_array_declarator (location_t loc,
tree expr, struct c_declspecs *quals, bool static_p,
bool vla_unspec_p)
{
struct c_declarator *declarator = XOBNEW (&parser_obstack,
struct c_declarator);
declarator->id_loc = loc;
declarator->kind = cdk_array;
declarator->declarator = 0;
declarator->u.array.dimen = expr;
if (quals)
{
declarator->u.array.attrs = quals->attrs;
declarator->u.array.quals = quals_from_declspecs (quals);
}
else
{
declarator->u.array.attrs = NULL_TREE;
declarator->u.array.quals = 0;
}
declarator->u.array.static_p = static_p;
declarator->u.array.vla_unspec_p = vla_unspec_p;
if (static_p || quals != NULL)
pedwarn_c90 (loc, OPT_Wpedantic,
"ISO C90 does not support %<static%> or type "
"qualifiers in parameter array declarators");
if (vla_unspec_p)
pedwarn_c90 (loc, OPT_Wpedantic,
"ISO C90 does not support %<[*]%> array declarators");
if (vla_unspec_p)
{
if (!current_scope->parm_flag)
{
error_at (loc, "%<[*]%> not allowed in other than "
"function prototype scope");
declarator->u.array.vla_unspec_p = false;
return NULL;
}
current_scope->had_vla_unspec = true;
}
return declarator;
}
struct c_declarator *
set_array_declarator_inner (struct c_declarator *decl,
struct c_declarator *inner)
{
decl->declarator = inner;
return decl;
}
static void
add_flexible_array_elts_to_size (tree decl, tree init)
{
tree elt, type;
if (vec_safe_is_empty (CONSTRUCTOR_ELTS (init)))
return;
elt = CONSTRUCTOR_ELTS (init)->last ().value;
type = TREE_TYPE (elt);
if (TREE_CODE (type) == ARRAY_TYPE
&& TYPE_SIZE (type) == NULL_TREE
&& TYPE_DOMAIN (type) != NULL_TREE
&& TYPE_MAX_VALUE (TYPE_DOMAIN (type)) == NULL_TREE)
{
complete_array_type (&type, elt, false);
DECL_SIZE (decl)
= size_binop (PLUS_EXPR, DECL_SIZE (decl), TYPE_SIZE (type));
DECL_SIZE_UNIT (decl)
= size_binop (PLUS_EXPR, DECL_SIZE_UNIT (decl), TYPE_SIZE_UNIT (type));
}
}

tree
groktypename (struct c_type_name *type_name, tree *expr,
bool *expr_const_operands)
{
tree type;
tree attrs = type_name->specs->attrs;
type_name->specs->attrs = NULL_TREE;
type = grokdeclarator (type_name->declarator, type_name->specs, TYPENAME,
false, NULL, &attrs, expr, expr_const_operands,
DEPRECATED_NORMAL);
decl_attributes (&type, attrs, 0);
return type;
}
static tree
c_decl_attributes (tree *node, tree attributes, int flags)
{
if (current_omp_declare_target_attribute
&& ((VAR_P (*node) && is_global_var (*node))
|| TREE_CODE (*node) == FUNCTION_DECL))
{
if (VAR_P (*node)
&& !lang_hooks.types.omp_mappable_type (TREE_TYPE (*node)))
attributes = tree_cons (get_identifier ("omp declare target implicit"),
NULL_TREE, attributes);
else
attributes = tree_cons (get_identifier ("omp declare target"),
NULL_TREE, attributes);
}
tree last_decl = lookup_name (DECL_NAME (*node));
if (!last_decl)
last_decl = lookup_name_in_scope (DECL_NAME (*node), external_scope);
return decl_attributes (node, attributes, flags, last_decl);
}
tree
start_decl (struct c_declarator *declarator, struct c_declspecs *declspecs,
bool initialized, tree attributes)
{
tree decl;
tree tem;
tree expr = NULL_TREE;
enum deprecated_states deprecated_state = DEPRECATED_NORMAL;
if (lookup_attribute ("deprecated", attributes))
deprecated_state = DEPRECATED_SUPPRESS;
decl = grokdeclarator (declarator, declspecs,
NORMAL, initialized, NULL, &attributes, &expr, NULL,
deprecated_state);
if (!decl || decl == error_mark_node)
return NULL_TREE;
if (expr)
add_stmt (fold_convert (void_type_node, expr));
if (TREE_CODE (decl) != FUNCTION_DECL && MAIN_NAME_P (DECL_NAME (decl)))
warning (OPT_Wmain, "%q+D is usually a function", decl);
if (initialized)
switch (TREE_CODE (decl))
{
case TYPE_DECL:
error ("typedef %qD is initialized (use __typeof__ instead)", decl);
initialized = false;
break;
case FUNCTION_DECL:
error ("function %qD is initialized like a variable", decl);
initialized = false;
break;
case PARM_DECL:
error ("parameter %qD is initialized", decl);
initialized = false;
break;
default:
if (TREE_TYPE (decl) == error_mark_node)
initialized = false;
else if (COMPLETE_TYPE_P (TREE_TYPE (decl)))
{
if (TREE_CODE (TYPE_SIZE (TREE_TYPE (decl))) != INTEGER_CST
|| C_DECL_VARIABLE_SIZE (decl))
{
error ("variable-sized object may not be initialized");
initialized = false;
}
}
else if (TREE_CODE (TREE_TYPE (decl)) != ARRAY_TYPE)
{
error ("variable %qD has initializer but incomplete type", decl);
initialized = false;
}
else if (C_DECL_VARIABLE_SIZE (decl))
{
error ("variable-sized object may not be initialized");
initialized = false;
}
}
if (initialized)
{
if (current_scope == file_scope)
TREE_STATIC (decl) = 1;
DECL_INITIAL (decl) = error_mark_node;
}
if (TREE_CODE (decl) == FUNCTION_DECL)
gen_aux_info_record (decl, 0, 0, prototype_p (TREE_TYPE (decl)));
if (VAR_P (decl)
&& !initialized
&& TREE_PUBLIC (decl)
&& !DECL_THREAD_LOCAL_P (decl)
&& !flag_no_common)
DECL_COMMON (decl) = 1;
c_decl_attributes (&decl, attributes, 0);
if (declspecs->inline_p
&& !flag_gnu89_inline
&& TREE_CODE (decl) == FUNCTION_DECL
&& (lookup_attribute ("gnu_inline", DECL_ATTRIBUTES (decl))
|| current_function_decl))
{
if (declspecs->storage_class == csc_auto && current_scope != file_scope)
;
else if (declspecs->storage_class != csc_static)
DECL_EXTERNAL (decl) = !DECL_EXTERNAL (decl);
}
if (TREE_CODE (decl) == FUNCTION_DECL
&& targetm.calls.promote_prototypes (TREE_TYPE (decl)))
{
struct c_declarator *ce = declarator;
if (ce->kind == cdk_pointer)
ce = declarator->declarator;
if (ce->kind == cdk_function)
{
tree args = ce->u.arg_info->parms;
for (; args; args = DECL_CHAIN (args))
{
tree type = TREE_TYPE (args);
if (type && INTEGRAL_TYPE_P (type)
&& TYPE_PRECISION (type) < TYPE_PRECISION (integer_type_node))
DECL_ARG_TYPE (args) = c_type_promotes_to (type);
}
}
}
if (TREE_CODE (decl) == FUNCTION_DECL
&& DECL_DECLARED_INLINE_P (decl)
&& DECL_UNINLINABLE (decl)
&& lookup_attribute ("noinline", DECL_ATTRIBUTES (decl)))
warning (OPT_Wattributes, "inline function %q+D given attribute noinline",
decl);
if (VAR_P (decl)
&& current_scope != file_scope
&& TREE_STATIC (decl)
&& !TREE_READONLY (decl)
&& DECL_DECLARED_INLINE_P (current_function_decl)
&& DECL_EXTERNAL (current_function_decl))
record_inline_static (input_location, current_function_decl,
decl, csi_modifiable);
if (c_dialect_objc ()
&& VAR_OR_FUNCTION_DECL_P (decl))
objc_check_global_decl (decl);
tem = pushdecl (decl);
if (initialized && DECL_EXTERNAL (tem))
{
DECL_EXTERNAL (tem) = 0;
TREE_STATIC (tem) = 1;
}
return tem;
}
static void
diagnose_uninitialized_cst_member (tree decl, tree type)
{
tree field;
for (field = TYPE_FIELDS (type); field; field = TREE_CHAIN (field))
{
tree field_type;
if (TREE_CODE (field) != FIELD_DECL)
continue;
field_type = strip_array_types (TREE_TYPE (field));
if (TYPE_QUALS (field_type) & TYPE_QUAL_CONST)
{
warning_at (DECL_SOURCE_LOCATION (decl), OPT_Wc___compat,
"uninitialized const member in %qT is invalid in C++",
strip_array_types (TREE_TYPE (decl)));
inform (DECL_SOURCE_LOCATION (field), "%qD should be initialized", field);
}
if (RECORD_OR_UNION_TYPE_P (field_type))
diagnose_uninitialized_cst_member (decl, field_type);
}
}
void
finish_decl (tree decl, location_t init_loc, tree init,
tree origtype, tree asmspec_tree)
{
tree type;
bool was_incomplete = (DECL_SIZE (decl) == NULL_TREE);
const char *asmspec = 0;
if (VAR_OR_FUNCTION_DECL_P (decl)
&& DECL_FILE_SCOPE_P (decl))
asmspec_tree = maybe_apply_renaming_pragma (decl, asmspec_tree);
if (asmspec_tree)
asmspec = TREE_STRING_POINTER (asmspec_tree);
if (VAR_P (decl)
&& TREE_STATIC (decl)
&& global_bindings_p ())
record_types_used_by_current_var_decl (decl);
if (init != NULL_TREE && DECL_INITIAL (decl) == NULL_TREE)
init = NULL_TREE;
if (TREE_CODE (decl) == PARM_DECL)
init = NULL_TREE;
if (init)
store_init_value (init_loc, decl, init, origtype);
if (c_dialect_objc () && (VAR_OR_FUNCTION_DECL_P (decl)
|| TREE_CODE (decl) == FIELD_DECL))
objc_check_decl (decl);
type = TREE_TYPE (decl);
if (TREE_CODE (type) == ARRAY_TYPE
&& TYPE_DOMAIN (type) == NULL_TREE
&& TREE_CODE (decl) != TYPE_DECL)
{
bool do_default
= (TREE_STATIC (decl)
? pedantic && !TREE_PUBLIC (decl)
: !DECL_EXTERNAL (decl));
int failure
= complete_array_type (&TREE_TYPE (decl), DECL_INITIAL (decl),
do_default);
type = TREE_TYPE (decl);
switch (failure)
{
case 1:
error ("initializer fails to determine size of %q+D", decl);
break;
case 2:
if (do_default)
error ("array size missing in %q+D", decl);
else if (!pedantic && TREE_STATIC (decl) && !TREE_PUBLIC (decl))
DECL_EXTERNAL (decl) = 1;
break;
case 3:
error ("zero or negative size array %q+D", decl);
break;
case 0:
if (TREE_PUBLIC (decl))
{
struct c_binding *b_ext = I_SYMBOL_BINDING (DECL_NAME (decl));
while (b_ext && !B_IN_EXTERNAL_SCOPE (b_ext))
b_ext = b_ext->shadowed;
if (b_ext && TREE_CODE (decl) == TREE_CODE (b_ext->decl))
{
if (b_ext->u.type && comptypes (b_ext->u.type, type))
b_ext->u.type = composite_type (b_ext->u.type, type);
else
b_ext->u.type = type;
}
}
break;
default:
gcc_unreachable ();
}
if (DECL_INITIAL (decl))
TREE_TYPE (DECL_INITIAL (decl)) = type;
relayout_decl (decl);
}
if (VAR_P (decl))
{
if (init && TREE_CODE (init) == CONSTRUCTOR)
add_flexible_array_elts_to_size (decl, init);
if (DECL_SIZE (decl) == NULL_TREE && TREE_TYPE (decl) != error_mark_node
&& COMPLETE_TYPE_P (TREE_TYPE (decl)))
layout_decl (decl, 0);
if (DECL_SIZE (decl) == NULL_TREE
&& TREE_TYPE (decl) != error_mark_node
&& (TREE_STATIC (decl)
? (DECL_INITIAL (decl) != NULL_TREE
|| !DECL_FILE_SCOPE_P (decl))
: !DECL_EXTERNAL (decl)))
{
error ("storage size of %q+D isn%'t known", decl);
TREE_TYPE (decl) = error_mark_node;
}
if ((RECORD_OR_UNION_TYPE_P (TREE_TYPE (decl))
|| TREE_CODE (TREE_TYPE (decl)) == ENUMERAL_TYPE)
&& DECL_SIZE (decl) == NULL_TREE
&& TREE_STATIC (decl))
incomplete_record_decls.safe_push (decl);
if (is_global_var (decl) && DECL_SIZE (decl) != NULL_TREE)
{
if (TREE_CODE (DECL_SIZE (decl)) == INTEGER_CST)
constant_expression_warning (DECL_SIZE (decl));
else
{
error ("storage size of %q+D isn%'t constant", decl);
TREE_TYPE (decl) = error_mark_node;
}
}
if (TREE_USED (type))
{
TREE_USED (decl) = 1;
DECL_READ_P (decl) = 1;
}
}
if (TREE_CODE (decl) == FUNCTION_DECL && asmspec)
{
if (DECL_BUILT_IN_CLASS (decl) == BUILT_IN_NORMAL)
set_builtin_user_assembler_name (decl, asmspec);
set_user_assembler_name (decl, asmspec);
}
maybe_apply_pragma_weak (decl);
if (VAR_OR_FUNCTION_DECL_P (decl))
{
if (TREE_PUBLIC (decl))
c_determine_visibility (decl);
if (c_dialect_objc ())
objc_check_decl (decl);
if (asmspec)
{
if (!DECL_FILE_SCOPE_P (decl)
&& VAR_P (decl)
&& !C_DECL_REGISTER (decl)
&& !TREE_STATIC (decl))
warning (0, "ignoring asm-specifier for non-static local "
"variable %q+D", decl);
else
set_user_assembler_name (decl, asmspec);
}
if (DECL_FILE_SCOPE_P (decl))
{
if (DECL_INITIAL (decl) == NULL_TREE
|| DECL_INITIAL (decl) == error_mark_node)
DECL_DEFER_OUTPUT (decl) = 1;
if (asmspec && VAR_P (decl) && C_DECL_REGISTER (decl))
DECL_HARD_REGISTER (decl) = 1;
rest_of_decl_compilation (decl, true, 0);
}
else
{
if (asmspec && C_DECL_REGISTER (decl))
{
DECL_HARD_REGISTER (decl) = 1;
if (!DECL_REGISTER (decl))
error ("cannot put object with volatile field into register");
}
if (TREE_CODE (decl) != FUNCTION_DECL)
{
if (DECL_SIZE (decl)
&& !TREE_CONSTANT (DECL_SIZE (decl))
&& STATEMENT_LIST_HAS_LABEL (cur_stmt_list))
{
tree bind;
bind = build3 (BIND_EXPR, void_type_node, NULL, NULL, NULL);
TREE_SIDE_EFFECTS (bind) = 1;
add_stmt (bind);
BIND_EXPR_BODY (bind) = push_stmt_list ();
}
add_stmt (build_stmt (DECL_SOURCE_LOCATION (decl),
DECL_EXPR, decl));
}
}
if (!DECL_FILE_SCOPE_P (decl))
{
if (was_incomplete && !is_global_var (decl))
{
TREE_ADDRESSABLE (decl) = TREE_USED (decl);
if (DECL_SIZE (decl) == NULL_TREE)
DECL_INITIAL (decl) = NULL_TREE;
}
}
}
if (TREE_CODE (decl) == TYPE_DECL)
{
if (!DECL_FILE_SCOPE_P (decl)
&& variably_modified_type_p (TREE_TYPE (decl), NULL_TREE))
add_stmt (build_stmt (DECL_SOURCE_LOCATION (decl), DECL_EXPR, decl));
rest_of_decl_compilation (decl, DECL_FILE_SCOPE_P (decl), 0);
}
if (VAR_P (decl) && !TREE_STATIC (decl))
{
tree attr = lookup_attribute ("cleanup", DECL_ATTRIBUTES (decl));
if (attr)
{
tree cleanup_id = TREE_VALUE (TREE_VALUE (attr));
tree cleanup_decl = lookup_name (cleanup_id);
tree cleanup;
vec<tree, va_gc> *v;
cleanup = build_unary_op (input_location, ADDR_EXPR, decl, false);
vec_alloc (v, 1);
v->quick_push (cleanup);
cleanup = c_build_function_call_vec (DECL_SOURCE_LOCATION (decl),
vNULL, cleanup_decl, v, NULL);
vec_free (v);
TREE_USED (decl) = 1;
TREE_USED (cleanup_decl) = 1;
DECL_READ_P (decl) = 1;
push_cleanup (decl, cleanup, false);
}
}
if (warn_cxx_compat
&& VAR_P (decl)
&& !DECL_EXTERNAL (decl)
&& DECL_INITIAL (decl) == NULL_TREE)
{
type = strip_array_types (type);
if (TREE_READONLY (decl))
warning_at (DECL_SOURCE_LOCATION (decl), OPT_Wc___compat,
"uninitialized const %qD is invalid in C++", decl);
else if (RECORD_OR_UNION_TYPE_P (type)
&& C_TYPE_FIELDS_READONLY (type))
diagnose_uninitialized_cst_member (decl, type);
}
if (flag_openmp
&& VAR_P (decl)
&& lookup_attribute ("omp declare target implicit",
DECL_ATTRIBUTES (decl)))
{
DECL_ATTRIBUTES (decl)
= remove_attribute ("omp declare target implicit",
DECL_ATTRIBUTES (decl));
if (!lang_hooks.types.omp_mappable_type (TREE_TYPE (decl)))
error ("%q+D in declare target directive does not have mappable type",
decl);
else if (!lookup_attribute ("omp declare target",
DECL_ATTRIBUTES (decl))
&& !lookup_attribute ("omp declare target link",
DECL_ATTRIBUTES (decl)))
DECL_ATTRIBUTES (decl)
= tree_cons (get_identifier ("omp declare target"),
NULL_TREE, DECL_ATTRIBUTES (decl));
}
invoke_plugin_callbacks (PLUGIN_FINISH_DECL, decl);
}
tree
grokparm (const struct c_parm *parm, tree *expr)
{
tree attrs = parm->attrs;
tree decl = grokdeclarator (parm->declarator, parm->specs, PARM, false,
NULL, &attrs, expr, NULL, DEPRECATED_NORMAL);
decl_attributes (&decl, attrs, 0);
return decl;
}
void
push_parm_decl (const struct c_parm *parm, tree *expr)
{
tree attrs = parm->attrs;
tree decl;
decl = grokdeclarator (parm->declarator, parm->specs, PARM, false, NULL,
&attrs, expr, NULL, DEPRECATED_NORMAL);
if (decl && DECL_P (decl))
DECL_SOURCE_LOCATION (decl) = parm->loc;
decl_attributes (&decl, attrs, 0);
decl = pushdecl (decl);
finish_decl (decl, input_location, NULL_TREE, NULL_TREE, NULL_TREE);
}
void
mark_forward_parm_decls (void)
{
struct c_binding *b;
if (pedantic && !current_scope->warned_forward_parm_decls)
{
pedwarn (input_location, OPT_Wpedantic,
"ISO C forbids forward parameter declarations");
current_scope->warned_forward_parm_decls = true;
}
for (b = current_scope->bindings; b; b = b->prev)
if (TREE_CODE (b->decl) == PARM_DECL)
TREE_ASM_WRITTEN (b->decl) = 1;
}

tree
build_compound_literal (location_t loc, tree type, tree init, bool non_const,
unsigned int alignas_align)
{
tree decl;
tree complit;
tree stmt;
if (type == error_mark_node
|| init == error_mark_node)
return error_mark_node;
decl = build_decl (loc, VAR_DECL, NULL_TREE, type);
DECL_EXTERNAL (decl) = 0;
TREE_PUBLIC (decl) = 0;
TREE_STATIC (decl) = (current_scope == file_scope);
DECL_CONTEXT (decl) = current_function_decl;
TREE_USED (decl) = 1;
DECL_READ_P (decl) = 1;
DECL_ARTIFICIAL (decl) = 1;
DECL_IGNORED_P (decl) = 1;
TREE_TYPE (decl) = type;
c_apply_type_quals_to_decl (TYPE_QUALS (strip_array_types (type)), decl);
if (alignas_align)
{
SET_DECL_ALIGN (decl, alignas_align * BITS_PER_UNIT);
DECL_USER_ALIGN (decl) = 1;
}
store_init_value (loc, decl, init, NULL_TREE);
if (TREE_CODE (type) == ARRAY_TYPE && !COMPLETE_TYPE_P (type))
{
int failure = complete_array_type (&TREE_TYPE (decl),
DECL_INITIAL (decl), true);
gcc_assert (failure == 0 || failure == 3);
type = TREE_TYPE (decl);
TREE_TYPE (DECL_INITIAL (decl)) = type;
}
if (type == error_mark_node || !COMPLETE_TYPE_P (type))
{
c_incomplete_type_error (loc, NULL_TREE, type);
return error_mark_node;
}
stmt = build_stmt (DECL_SOURCE_LOCATION (decl), DECL_EXPR, decl);
complit = build1 (COMPOUND_LITERAL_EXPR, type, stmt);
TREE_SIDE_EFFECTS (complit) = 1;
layout_decl (decl, 0);
if (TREE_STATIC (decl))
{
set_compound_literal_name (decl);
DECL_DEFER_OUTPUT (decl) = 1;
DECL_COMDAT (decl) = 1;
pushdecl (decl);
rest_of_decl_compilation (decl, 1, 0);
}
if (non_const)
{
complit = build2 (C_MAYBE_CONST_EXPR, type, NULL, complit);
C_MAYBE_CONST_EXPR_NON_CONST (complit) = 1;
}
return complit;
}
void
check_compound_literal_type (location_t loc, struct c_type_name *type_name)
{
if (warn_cxx_compat
&& (type_name->specs->typespec_kind == ctsk_tagdef
|| type_name->specs->typespec_kind == ctsk_tagfirstref))
warning_at (loc, OPT_Wc___compat,
"defining a type in a compound literal is invalid in C++");
}

static bool
flexible_array_type_p (tree type)
{
tree x;
switch (TREE_CODE (type))
{
case RECORD_TYPE:
x = TYPE_FIELDS (type);
if (x == NULL_TREE)
return false;
while (DECL_CHAIN (x) != NULL_TREE)
x = DECL_CHAIN (x);
if (TREE_CODE (TREE_TYPE (x)) == ARRAY_TYPE
&& TYPE_SIZE (TREE_TYPE (x)) == NULL_TREE
&& TYPE_DOMAIN (TREE_TYPE (x)) != NULL_TREE
&& TYPE_MAX_VALUE (TYPE_DOMAIN (TREE_TYPE (x))) == NULL_TREE)
return true;
return false;
case UNION_TYPE:
for (x = TYPE_FIELDS (type); x != NULL_TREE; x = DECL_CHAIN (x))
{
if (flexible_array_type_p (TREE_TYPE (x)))
return true;
}
return false;
default:
return false;
}
}

static void
check_bitfield_type_and_width (location_t loc, tree *type, tree *width,
tree orig_name)
{
tree type_mv;
unsigned int max_width;
unsigned HOST_WIDE_INT w;
const char *name = (orig_name
? identifier_to_locale (IDENTIFIER_POINTER (orig_name))
: _("<anonymous>"));
if (!INTEGRAL_TYPE_P (TREE_TYPE (*width)))
{
error_at (loc, "bit-field %qs width not an integer constant", name);
*width = integer_one_node;
}
else
{
if (TREE_CODE (*width) != INTEGER_CST)
{
*width = c_fully_fold (*width, false, NULL);
if (TREE_CODE (*width) == INTEGER_CST)
pedwarn (loc, OPT_Wpedantic,
"bit-field %qs width not an integer constant expression",
name);
}
if (TREE_CODE (*width) != INTEGER_CST)
{
error_at (loc, "bit-field %qs width not an integer constant", name);
*width = integer_one_node;
}
constant_expression_warning (*width);
if (tree_int_cst_sgn (*width) < 0)
{
error_at (loc, "negative width in bit-field %qs", name);
*width = integer_one_node;
}
else if (integer_zerop (*width) && orig_name)
{
error_at (loc, "zero width for bit-field %qs", name);
*width = integer_one_node;
}
}
if (TREE_CODE (*type) != INTEGER_TYPE
&& TREE_CODE (*type) != BOOLEAN_TYPE
&& TREE_CODE (*type) != ENUMERAL_TYPE)
{
error_at (loc, "bit-field %qs has invalid type", name);
*type = unsigned_type_node;
}
if (TYPE_WARN_IF_NOT_ALIGN (*type))
{
error_at (loc, "cannot declare bit-field %qs with %<warn_if_not_aligned%> type",
name);
*type = unsigned_type_node;
}
type_mv = TYPE_MAIN_VARIANT (*type);
if (!in_system_header_at (input_location)
&& type_mv != integer_type_node
&& type_mv != unsigned_type_node
&& type_mv != boolean_type_node)
pedwarn_c90 (loc, OPT_Wpedantic,
"type of bit-field %qs is a GCC extension", name);
max_width = TYPE_PRECISION (*type);
if (compare_tree_int (*width, max_width) > 0)
{
error_at (loc, "width of %qs exceeds its type", name);
w = max_width;
*width = build_int_cst (integer_type_node, w);
}
else
w = tree_to_uhwi (*width);
if (TREE_CODE (*type) == ENUMERAL_TYPE)
{
struct lang_type *lt = TYPE_LANG_SPECIFIC (*type);
if (!lt
|| w < tree_int_cst_min_precision (lt->enum_min, TYPE_SIGN (*type))
|| w < tree_int_cst_min_precision (lt->enum_max, TYPE_SIGN (*type)))
warning_at (loc, 0, "%qs is narrower than values of its type", name);
}
}

static void
warn_variable_length_array (tree name, tree size)
{
if (TREE_CONSTANT (size))
{
if (name)
pedwarn_c90 (input_location, OPT_Wvla,
"ISO C90 forbids array %qE whose size "
"can%'t be evaluated", name);
else
pedwarn_c90 (input_location, OPT_Wvla, "ISO C90 forbids array "
"whose size can%'t be evaluated");
}
else
{
if (name)
pedwarn_c90 (input_location, OPT_Wvla,
"ISO C90 forbids variable length array %qE", name);
else
pedwarn_c90 (input_location, OPT_Wvla, "ISO C90 forbids variable "
"length array");
}
}
static void
warn_defaults_to (location_t location, int opt, const char *gmsgid, ...)
{
diagnostic_info diagnostic;
va_list ap;
rich_location richloc (line_table, location);
va_start (ap, gmsgid);
diagnostic_set_info (&diagnostic, gmsgid, &ap, &richloc,
flag_isoc99 ? DK_PEDWARN : DK_WARNING);
diagnostic.option_index = opt;
diagnostic_report_diagnostic (global_dc, &diagnostic);
va_end (ap);
}
static location_t
smallest_type_quals_location (const location_t *locations,
const c_declspec_word *list)
{
location_t loc = UNKNOWN_LOCATION;
while (*list != cdw_number_of_elements)
{
location_t newloc = locations[*list];
if (loc == UNKNOWN_LOCATION
|| (newloc != UNKNOWN_LOCATION && newloc < loc))
loc = newloc;
list++;
}
return loc;
}
static tree
grokdeclarator (const struct c_declarator *declarator,
struct c_declspecs *declspecs,
enum decl_context decl_context, bool initialized, tree *width,
tree *decl_attrs, tree *expr, bool *expr_const_operands,
enum deprecated_states deprecated_state)
{
tree type = declspecs->type;
bool threadp = declspecs->thread_p;
enum c_storage_class storage_class = declspecs->storage_class;
int constp;
int restrictp;
int volatilep;
int atomicp;
int type_quals = TYPE_UNQUALIFIED;
tree name = NULL_TREE;
bool funcdef_flag = false;
bool funcdef_syntax = false;
bool size_varies = false;
tree decl_attr = declspecs->decl_attr;
int array_ptr_quals = TYPE_UNQUALIFIED;
tree array_ptr_attrs = NULL_TREE;
bool array_parm_static = false;
bool array_parm_vla_unspec_p = false;
tree returned_attrs = NULL_TREE;
bool bitfield = width != NULL;
tree element_type;
tree orig_qual_type = NULL;
size_t orig_qual_indirect = 0;
struct c_arg_info *arg_info = 0;
addr_space_t as1, as2, address_space;
location_t loc = UNKNOWN_LOCATION;
tree expr_dummy;
bool expr_const_operands_dummy;
enum c_declarator_kind first_non_attr_kind;
unsigned int alignas_align = 0;
if (TREE_CODE (type) == ERROR_MARK)
return error_mark_node;
if (expr == NULL)
{
expr = &expr_dummy;
expr_dummy = NULL_TREE;
}
if (expr_const_operands == NULL)
expr_const_operands = &expr_const_operands_dummy;
if (declspecs->expr)
{
if (*expr)
*expr = build2 (COMPOUND_EXPR, TREE_TYPE (declspecs->expr), *expr,
declspecs->expr);
else
*expr = declspecs->expr;
}
*expr_const_operands = declspecs->expr_const_operands;
if (decl_context == FUNCDEF)
funcdef_flag = true, decl_context = NORMAL;
{
const struct c_declarator *decl = declarator;
first_non_attr_kind = cdk_attrs;
while (decl)
switch (decl->kind)
{
case cdk_array:
loc = decl->id_loc;
case cdk_function:
case cdk_pointer:
funcdef_syntax = (decl->kind == cdk_function);
if (first_non_attr_kind == cdk_attrs)
first_non_attr_kind = decl->kind;
decl = decl->declarator;
break;
case cdk_attrs:
decl = decl->declarator;
break;
case cdk_id:
loc = decl->id_loc;
if (decl->u.id)
name = decl->u.id;
if (first_non_attr_kind == cdk_attrs)
first_non_attr_kind = decl->kind;
decl = 0;
break;
default:
gcc_unreachable ();
}
if (name == NULL_TREE)
{
gcc_assert (decl_context == PARM
|| decl_context == TYPENAME
|| (decl_context == FIELD
&& declarator->kind == cdk_id));
gcc_assert (!initialized);
}
}
if (funcdef_flag && !funcdef_syntax)
return NULL_TREE;
if (decl_context == NORMAL && !funcdef_flag && current_scope->parm_flag)
decl_context = PARM;
if (declspecs->deprecated_p && deprecated_state != DEPRECATED_SUPPRESS)
warn_deprecated_use (declspecs->type, declspecs->decl_attr);
if ((decl_context == NORMAL || decl_context == FIELD)
&& current_scope == file_scope
&& variably_modified_type_p (type, NULL_TREE))
{
if (name)
error_at (loc, "variably modified %qE at file scope", name);
else
error_at (loc, "variably modified field at file scope");
type = integer_type_node;
}
size_varies = C_TYPE_VARIABLE_SIZE (type) != 0;
if (declspecs->default_int_p && !in_system_header_at (input_location))
{
if ((warn_implicit_int || warn_return_type || flag_isoc99)
&& funcdef_flag)
warn_about_return_type = 1;
else
{
if (name)
warn_defaults_to (loc, OPT_Wimplicit_int,
"type defaults to %<int%> in declaration "
"of %qE", name);
else
warn_defaults_to (loc, OPT_Wimplicit_int,
"type defaults to %<int%> in type name");
}
}
if (bitfield && !flag_signed_bitfields && !declspecs->explicit_signed_p
&& TREE_CODE (type) == INTEGER_TYPE)
type = unsigned_type_for (type);
element_type = strip_array_types (type);
constp = declspecs->const_p + TYPE_READONLY (element_type);
restrictp = declspecs->restrict_p + TYPE_RESTRICT (element_type);
volatilep = declspecs->volatile_p + TYPE_VOLATILE (element_type);
atomicp = declspecs->atomic_p + TYPE_ATOMIC (element_type);
as1 = declspecs->address_space;
as2 = TYPE_ADDR_SPACE (element_type);
address_space = ADDR_SPACE_GENERIC_P (as1)? as2 : as1;
if (constp > 1)
pedwarn_c90 (loc, OPT_Wpedantic, "duplicate %<const%>");
if (restrictp > 1)
pedwarn_c90 (loc, OPT_Wpedantic, "duplicate %<restrict%>");
if (volatilep > 1)
pedwarn_c90 (loc, OPT_Wpedantic, "duplicate %<volatile%>");
if (atomicp > 1)
pedwarn_c90 (loc, OPT_Wpedantic, "duplicate %<_Atomic%>");
if (!ADDR_SPACE_GENERIC_P (as1) && !ADDR_SPACE_GENERIC_P (as2) && as1 != as2)
error_at (loc, "conflicting named address spaces (%s vs %s)",
c_addr_space_name (as1), c_addr_space_name (as2));
if ((TREE_CODE (type) == ARRAY_TYPE
|| first_non_attr_kind == cdk_array)
&& TYPE_QUALS (element_type))
{
orig_qual_type = type;
type = TYPE_MAIN_VARIANT (type);
}
type_quals = ((constp ? TYPE_QUAL_CONST : 0)
| (restrictp ? TYPE_QUAL_RESTRICT : 0)
| (volatilep ? TYPE_QUAL_VOLATILE : 0)
| (atomicp ? TYPE_QUAL_ATOMIC : 0)
| ENCODE_QUAL_ADDR_SPACE (address_space));
if (type_quals != TYPE_QUALS (element_type))
orig_qual_type = NULL_TREE;
if (declspecs->atomic_p && TREE_CODE (type) == ARRAY_TYPE)
error_at (loc, "%<_Atomic%>-qualified array type");
if (funcdef_flag
&& (threadp
|| storage_class == csc_auto
|| storage_class == csc_register
|| storage_class == csc_typedef))
{
if (storage_class == csc_auto)
pedwarn (loc,
(current_scope == file_scope) ? 0 : OPT_Wpedantic,
"function definition declared %<auto%>");
if (storage_class == csc_register)
error_at (loc, "function definition declared %<register%>");
if (storage_class == csc_typedef)
error_at (loc, "function definition declared %<typedef%>");
if (threadp)
error_at (loc, "function definition declared %qs",
declspecs->thread_gnu_p ? "__thread" : "_Thread_local");
threadp = false;
if (storage_class == csc_auto
|| storage_class == csc_register
|| storage_class == csc_typedef)
storage_class = csc_none;
}
else if (decl_context != NORMAL && (storage_class != csc_none || threadp))
{
if (decl_context == PARM && storage_class == csc_register)
;
else
{
switch (decl_context)
{
case FIELD:
if (name)
error_at (loc, "storage class specified for structure "
"field %qE", name);
else
error_at (loc, "storage class specified for structure field");
break;
case PARM:
if (name)
error_at (loc, "storage class specified for parameter %qE",
name);
else
error_at (loc, "storage class specified for unnamed parameter");
break;
default:
error_at (loc, "storage class specified for typename");
break;
}
storage_class = csc_none;
threadp = false;
}
}
else if (storage_class == csc_extern
&& initialized
&& !funcdef_flag)
{
if (current_scope == file_scope)
{
if (!(warn_cxx_compat && constp))
warning_at (loc, 0, "%qE initialized and declared %<extern%>",
name);
}
else
error_at (loc, "%qE has both %<extern%> and initializer", name);
}
else if (current_scope == file_scope)
{
if (storage_class == csc_auto)
error_at (loc, "file-scope declaration of %qE specifies %<auto%>",
name);
if (pedantic && storage_class == csc_register)
pedwarn (input_location, OPT_Wpedantic,
"file-scope declaration of %qE specifies %<register%>", name);
}
else
{
if (storage_class == csc_extern && funcdef_flag)
error_at (loc, "nested function %qE declared %<extern%>", name);
else if (threadp && storage_class == csc_none)
{
error_at (loc, "function-scope %qE implicitly auto and declared "
"%qs", name,
declspecs->thread_gnu_p ? "__thread" : "_Thread_local");
threadp = false;
}
}
while (declarator && declarator->kind != cdk_id)
{
if (type == error_mark_node)
{
declarator = declarator->declarator;
continue;
}
if (array_ptr_quals != TYPE_UNQUALIFIED
|| array_ptr_attrs != NULL_TREE
|| array_parm_static)
{
error_at (loc, "static or type qualifiers in non-parameter array declarator");
array_ptr_quals = TYPE_UNQUALIFIED;
array_ptr_attrs = NULL_TREE;
array_parm_static = false;
}
switch (declarator->kind)
{
case cdk_attrs:
{
tree attrs = declarator->u.attrs;
const struct c_declarator *inner_decl;
int attr_flags = 0;
declarator = declarator->declarator;
inner_decl = declarator;
while (inner_decl->kind == cdk_attrs)
inner_decl = inner_decl->declarator;
if (inner_decl->kind == cdk_id)
attr_flags |= (int) ATTR_FLAG_DECL_NEXT;
else if (inner_decl->kind == cdk_function)
attr_flags |= (int) ATTR_FLAG_FUNCTION_NEXT;
else if (inner_decl->kind == cdk_array)
attr_flags |= (int) ATTR_FLAG_ARRAY_NEXT;
returned_attrs = decl_attributes (&type,
chainon (returned_attrs, attrs),
attr_flags);
break;
}
case cdk_array:
{
tree itype = NULL_TREE;
tree size = declarator->u.array.dimen;
tree index_type = c_common_signed_type (sizetype);
array_ptr_quals = declarator->u.array.quals;
array_ptr_attrs = declarator->u.array.attrs;
array_parm_static = declarator->u.array.static_p;
array_parm_vla_unspec_p = declarator->u.array.vla_unspec_p;
declarator = declarator->declarator;
if (VOID_TYPE_P (type))
{
if (name)
error_at (loc, "declaration of %qE as array of voids", name);
else
error_at (loc, "declaration of type name as array of voids");
type = error_mark_node;
}
if (TREE_CODE (type) == FUNCTION_TYPE)
{
if (name)
error_at (loc, "declaration of %qE as array of functions",
name);
else
error_at (loc, "declaration of type name as array of "
"functions");
type = error_mark_node;
}
if (pedantic && !in_system_header_at (input_location)
&& flexible_array_type_p (type))
pedwarn (loc, OPT_Wpedantic,
"invalid use of structure with flexible array member");
if (size == error_mark_node)
type = error_mark_node;
if (type == error_mark_node)
continue;
if (size)
{
bool size_maybe_const = true;
bool size_int_const = (TREE_CODE (size) == INTEGER_CST
&& !TREE_OVERFLOW (size));
bool this_size_varies = false;
STRIP_TYPE_NOPS (size);
if (!INTEGRAL_TYPE_P (TREE_TYPE (size)))
{
if (name)
error_at (loc, "size of array %qE has non-integer type",
name);
else
error_at (loc,
"size of unnamed array has non-integer type");
size = integer_one_node;
}
else if (!COMPLETE_TYPE_P (TREE_TYPE (size)))
{
if (name)
error_at (loc, "size of array %qE has incomplete type",
name);
else
error_at (loc, "size of unnamed array has incomplete "
"type");
size = integer_one_node;
}
size = c_fully_fold (size, false, &size_maybe_const);
if (pedantic && size_maybe_const && integer_zerop (size))
{
if (name)
pedwarn (loc, OPT_Wpedantic,
"ISO C forbids zero-size array %qE", name);
else
pedwarn (loc, OPT_Wpedantic,
"ISO C forbids zero-size array");
}
if (TREE_CODE (size) == INTEGER_CST && size_maybe_const)
{
constant_expression_warning (size);
if (tree_int_cst_sgn (size) < 0)
{
if (name)
error_at (loc, "size of array %qE is negative", name);
else
error_at (loc, "size of unnamed array is negative");
size = integer_one_node;
}
if (!size_int_const)
{
if ((decl_context == NORMAL || decl_context == FIELD)
&& current_scope == file_scope)
pedwarn (input_location, 0,
"variably modified %qE at file scope",
name);
else
this_size_varies = size_varies = true;
warn_variable_length_array (name, size);
}
}
else if ((decl_context == NORMAL || decl_context == FIELD)
&& current_scope == file_scope)
{
error_at (loc, "variably modified %qE at file scope", name);
size = integer_one_node;
}
else
{
this_size_varies = size_varies = true;
warn_variable_length_array (name, size);
if (sanitize_flags_p (SANITIZE_VLA)
&& current_function_decl != NULL_TREE
&& decl_context == NORMAL)
{
size = save_expr (size);
size = c_fully_fold (size, false, NULL);
size = fold_build2 (COMPOUND_EXPR, TREE_TYPE (size),
ubsan_instrument_vla (loc, size),
size);
}
}
if (integer_zerop (size) && !this_size_varies)
{
itype = build_range_type (sizetype, size, NULL_TREE);
}
else
{
if (size_varies)
size = save_expr (size);
if (this_size_varies && TREE_CODE (size) == INTEGER_CST)
size = build2 (COMPOUND_EXPR, TREE_TYPE (size),
integer_zero_node, size);
itype = fold_build2_loc (loc, MINUS_EXPR, index_type,
convert (index_type, size),
convert (index_type,
size_one_node));
if (TREE_CODE (size) == INTEGER_CST
&& !int_fits_type_p (size, index_type))
{
if (name)
error_at (loc, "size of array %qE is too large",
name);
else
error_at (loc, "size of unnamed array is too large");
type = error_mark_node;
continue;
}
itype = build_index_type (itype);
}
if (this_size_varies)
{
if (*expr)
*expr = build2 (COMPOUND_EXPR, TREE_TYPE (size),
*expr, size);
else
*expr = size;
*expr_const_operands &= size_maybe_const;
}
}
else if (decl_context == FIELD)
{
bool flexible_array_member = false;
if (array_parm_vla_unspec_p)
size_varies = true;
else
{
const struct c_declarator *t = declarator;
while (t->kind == cdk_attrs)
t = t->declarator;
flexible_array_member = (t->kind == cdk_id);
}
if (flexible_array_member
&& !in_system_header_at (input_location))
pedwarn_c90 (loc, OPT_Wpedantic, "ISO C90 does not "
"support flexible array members");
if (flexible_array_member || array_parm_vla_unspec_p)
itype = build_range_type (sizetype, size_zero_node,
NULL_TREE);
}
else if (decl_context == PARM)
{
if (array_parm_vla_unspec_p)
{
itype = build_range_type (sizetype, size_zero_node, NULL_TREE);
size_varies = true;
}
}
else if (decl_context == TYPENAME)
{
if (array_parm_vla_unspec_p)
{
warning (0, "%<[*]%> not in a declaration");
itype = build_range_type (sizetype, size_zero_node,
NULL_TREE);
size_varies = true;
}
}
if (!COMPLETE_TYPE_P (type))
{
error_at (loc, "array type has incomplete element type %qT",
type);
if (TREE_CODE (type) == ARRAY_TYPE)
{
if (name)
inform (loc, "declaration of %qE as multidimensional "
"array must have bounds for all dimensions "
"except the first", name);
else
inform (loc, "declaration of multidimensional array "
"must have bounds for all dimensions except "
"the first");
}
type = error_mark_node;
}
else
{
addr_space_t as = DECODE_QUAL_ADDR_SPACE (type_quals);
if (!ADDR_SPACE_GENERIC_P (as) && as != TYPE_ADDR_SPACE (type))
type = build_qualified_type (type,
ENCODE_QUAL_ADDR_SPACE (as));
type = build_array_type (type, itype);
}
if (type != error_mark_node)
{
if (size_varies)
{
if (size && TREE_CODE (size) == INTEGER_CST)
type
= build_distinct_type_copy (TYPE_MAIN_VARIANT (type));
C_TYPE_VARIABLE_SIZE (type) = 1;
}
if (size && integer_zerop (size))
{
gcc_assert (itype);
type = build_distinct_type_copy (TYPE_MAIN_VARIANT (type));
TYPE_SIZE (type) = bitsize_zero_node;
TYPE_SIZE_UNIT (type) = size_zero_node;
SET_TYPE_STRUCTURAL_EQUALITY (type);
}
if (array_parm_vla_unspec_p)
{
gcc_assert (itype);
type = build_distinct_type_copy (TYPE_MAIN_VARIANT (type));
TYPE_SIZE (type) = bitsize_zero_node;
TYPE_SIZE_UNIT (type) = size_zero_node;
SET_TYPE_STRUCTURAL_EQUALITY (type);
}
if (!valid_array_size_p (loc, type, name))
type = error_mark_node;
}
if (decl_context != PARM
&& (array_ptr_quals != TYPE_UNQUALIFIED
|| array_ptr_attrs != NULL_TREE
|| array_parm_static))
{
error_at (loc, "static or type qualifiers in non-parameter "
"array declarator");
array_ptr_quals = TYPE_UNQUALIFIED;
array_ptr_attrs = NULL_TREE;
array_parm_static = false;
}
orig_qual_indirect++;
break;
}
case cdk_function:
{
bool really_funcdef = false;
tree arg_types;
orig_qual_type = NULL_TREE;
if (funcdef_flag)
{
const struct c_declarator *t = declarator->declarator;
while (t->kind == cdk_attrs)
t = t->declarator;
really_funcdef = (t->kind == cdk_id);
}
if (type == error_mark_node)
continue;
size_varies = false;
if (TREE_CODE (type) == FUNCTION_TYPE)
{
if (name)
error_at (loc, "%qE declared as function returning a "
"function", name);
else
error_at (loc, "type name declared as function "
"returning a function");
type = integer_type_node;
}
if (TREE_CODE (type) == ARRAY_TYPE)
{
if (name)
error_at (loc, "%qE declared as function returning an array",
name);
else
error_at (loc, "type name declared as function returning "
"an array");
type = integer_type_node;
}
arg_info = declarator->u.arg_info;
arg_types = grokparms (arg_info, really_funcdef);
if (type_quals)
{
const enum c_declspec_word ignored_quals_list[] =
{
cdw_const, cdw_volatile, cdw_restrict, cdw_address_space,
cdw_atomic, cdw_number_of_elements
};
location_t specs_loc
= smallest_type_quals_location (declspecs->locations,
ignored_quals_list);
if (specs_loc == UNKNOWN_LOCATION)
specs_loc = declspecs->locations[cdw_typedef];
if (specs_loc == UNKNOWN_LOCATION)
specs_loc = loc;
int quals_used = type_quals;
if (flag_isoc11)
quals_used &= TYPE_QUAL_ATOMIC;
if (quals_used && VOID_TYPE_P (type) && really_funcdef)
pedwarn (specs_loc, 0,
"function definition has qualified void return type");
else
warning_at (specs_loc, OPT_Wignored_qualifiers,
"type qualifiers ignored on function return type");
if (flag_isoc11
&& (type_quals & TYPE_QUAL_RESTRICT)
&& (!POINTER_TYPE_P (type)
|| !C_TYPE_OBJECT_OR_INCOMPLETE_P (TREE_TYPE (type))))
error_at (loc, "invalid use of %<restrict%>");
if (quals_used)
type = c_build_qualified_type (type, quals_used);
}
type_quals = TYPE_UNQUALIFIED;
type = build_function_type (type, arg_types);
declarator = declarator->declarator;
{
c_arg_tag *tag;
unsigned ix;
FOR_EACH_VEC_SAFE_ELT_REVERSE (arg_info->tags, ix, tag)
TYPE_CONTEXT (tag->type) = type;
}
break;
}
case cdk_pointer:
{
if ((type_quals & TYPE_QUAL_ATOMIC)
&& TREE_CODE (type) == FUNCTION_TYPE)
{
error_at (loc,
"%<_Atomic%>-qualified function type");
type_quals &= ~TYPE_QUAL_ATOMIC;
}
else if (pedantic && TREE_CODE (type) == FUNCTION_TYPE
&& type_quals)
pedwarn (loc, OPT_Wpedantic,
"ISO C forbids qualified function types");
if (type_quals)
type = c_build_qualified_type (type, type_quals, orig_qual_type,
orig_qual_indirect);
orig_qual_type = NULL_TREE;
size_varies = false;
if (!TYPE_NAME (type)
&& variably_modified_type_p (type, NULL_TREE))
{
tree bind = NULL_TREE;
if (decl_context == TYPENAME || decl_context == PARM)
{
bind = build3 (BIND_EXPR, void_type_node, NULL_TREE,
NULL_TREE, NULL_TREE);
TREE_SIDE_EFFECTS (bind) = 1;
BIND_EXPR_BODY (bind) = push_stmt_list ();
push_scope ();
}
tree decl = build_decl (loc, TYPE_DECL, NULL_TREE, type);
DECL_ARTIFICIAL (decl) = 1;
pushdecl (decl);
finish_decl (decl, loc, NULL_TREE, NULL_TREE, NULL_TREE);
TYPE_NAME (type) = decl;
if (bind)
{
pop_scope ();
BIND_EXPR_BODY (bind)
= pop_stmt_list (BIND_EXPR_BODY (bind));
if (*expr)
*expr = build2 (COMPOUND_EXPR, void_type_node, *expr,
bind);
else
*expr = bind;
}
}
type = c_build_pointer_type (type);
type_quals = declarator->u.pointer_quals;
declarator = declarator->declarator;
break;
}
default:
gcc_unreachable ();
}
}
*decl_attrs = chainon (returned_attrs, *decl_attrs);
address_space = DECODE_QUAL_ADDR_SPACE (type_quals);
if (!ADDR_SPACE_GENERIC_P (address_space))
{
if (decl_context == NORMAL)
{
switch (storage_class)
{
case csc_auto:
error ("%qs combined with %<auto%> qualifier for %qE",
c_addr_space_name (address_space), name);
break;
case csc_register:
error ("%qs combined with %<register%> qualifier for %qE",
c_addr_space_name (address_space), name);
break;
case csc_none:
if (current_function_scope)
{
error ("%qs specified for auto variable %qE",
c_addr_space_name (address_space), name);
break;
}
break;
case csc_static:
case csc_extern:
case csc_typedef:
break;
default:
gcc_unreachable ();
}
}
else if (decl_context == PARM && TREE_CODE (type) != ARRAY_TYPE)
{
if (name)
error ("%qs specified for parameter %qE",
c_addr_space_name (address_space), name);
else
error ("%qs specified for unnamed parameter",
c_addr_space_name (address_space));
}
else if (decl_context == FIELD)
{
if (name)
error ("%qs specified for structure field %qE",
c_addr_space_name (address_space), name);
else
error ("%qs specified for structure field",
c_addr_space_name (address_space));
}
}
if (bitfield)
{
check_bitfield_type_and_width (loc, &type, width, name);
if (type_quals & TYPE_QUAL_ATOMIC)
{
if (name)
error_at (loc, "bit-field %qE has atomic type", name);
else
error_at (loc, "bit-field has atomic type");
type_quals &= ~TYPE_QUAL_ATOMIC;
}
}
if (declspecs->alignas_p)
{
if (storage_class == csc_typedef)
error_at (loc, "alignment specified for typedef %qE", name);
else if (storage_class == csc_register)
error_at (loc, "alignment specified for %<register%> object %qE",
name);
else if (decl_context == PARM)
{
if (name)
error_at (loc, "alignment specified for parameter %qE", name);
else
error_at (loc, "alignment specified for unnamed parameter");
}
else if (bitfield)
{
if (name)
error_at (loc, "alignment specified for bit-field %qE", name);
else
error_at (loc, "alignment specified for unnamed bit-field");
}
else if (TREE_CODE (type) == FUNCTION_TYPE)
error_at (loc, "alignment specified for function %qE", name);
else if (declspecs->align_log != -1 && TYPE_P (type))
{
alignas_align = 1U << declspecs->align_log;
if (alignas_align < min_align_of_type (type))
{
if (name)
error_at (loc, "%<_Alignas%> specifiers cannot reduce "
"alignment of %qE", name);
else
error_at (loc, "%<_Alignas%> specifiers cannot reduce "
"alignment of unnamed field");
alignas_align = 0;
}
}
}
if (storage_class == csc_typedef)
{
tree decl;
if ((type_quals & TYPE_QUAL_ATOMIC)
&& TREE_CODE (type) == FUNCTION_TYPE)
{
error_at (loc,
"%<_Atomic%>-qualified function type");
type_quals &= ~TYPE_QUAL_ATOMIC;
}
else if (pedantic && TREE_CODE (type) == FUNCTION_TYPE
&& type_quals)
pedwarn (loc, OPT_Wpedantic,
"ISO C forbids qualified function types");
if (type_quals)
type = c_build_qualified_type (type, type_quals, orig_qual_type,
orig_qual_indirect);
decl = build_decl (declarator->id_loc,
TYPE_DECL, declarator->u.id, type);
if (declspecs->explicit_signed_p)
C_TYPEDEF_EXPLICITLY_SIGNED (decl) = 1;
if (declspecs->inline_p)
pedwarn (loc, 0,"typedef %q+D declared %<inline%>", decl);
if (declspecs->noreturn_p)
pedwarn (loc, 0,"typedef %q+D declared %<_Noreturn%>", decl);
if (warn_cxx_compat && declarator->u.id != NULL_TREE)
{
struct c_binding *b = I_TAG_BINDING (declarator->u.id);
if (b != NULL
&& b->decl != NULL_TREE
&& (B_IN_CURRENT_SCOPE (b)
|| (current_scope == file_scope && B_IN_EXTERNAL_SCOPE (b)))
&& TYPE_MAIN_VARIANT (b->decl) != TYPE_MAIN_VARIANT (type))
{
if (warning_at (declarator->id_loc, OPT_Wc___compat,
("using %qD as both a typedef and a tag is "
"invalid in C++"), decl)
&& b->locus != UNKNOWN_LOCATION)
inform (b->locus, "originally defined here");
}
}
return decl;
}
if (decl_context == TYPENAME)
{
gcc_assert (storage_class == csc_none && !threadp
&& !declspecs->inline_p && !declspecs->noreturn_p);
if ((type_quals & TYPE_QUAL_ATOMIC)
&& TREE_CODE (type) == FUNCTION_TYPE)
{
error_at (loc,
"%<_Atomic%>-qualified function type");
type_quals &= ~TYPE_QUAL_ATOMIC;
}
else if (pedantic && TREE_CODE (type) == FUNCTION_TYPE
&& type_quals)
pedwarn (loc, OPT_Wpedantic,
"ISO C forbids const or volatile function types");
if (type_quals)
type = c_build_qualified_type (type, type_quals, orig_qual_type,
orig_qual_indirect);
return type;
}
if (pedantic && decl_context == FIELD
&& variably_modified_type_p (type, NULL_TREE))
{
pedwarn (loc, OPT_Wpedantic, "a member of a structure or union cannot "
"have a variably modified type");
}
if (VOID_TYPE_P (type) && decl_context != PARM
&& !((decl_context != FIELD && TREE_CODE (type) != FUNCTION_TYPE)
&& (storage_class == csc_extern
|| (current_scope == file_scope
&& !(storage_class == csc_static
|| storage_class == csc_register)))))
{
error_at (loc, "variable or field %qE declared void", name);
type = integer_type_node;
}
{
tree decl;
if (decl_context == PARM)
{
tree promoted_type;
bool array_parameter_p = false;
if (TREE_CODE (type) == ARRAY_TYPE)
{
type = TREE_TYPE (type);
if (orig_qual_type != NULL_TREE)
{
if (orig_qual_indirect == 0)
orig_qual_type = TREE_TYPE (orig_qual_type);
else
orig_qual_indirect--;
}
if (type_quals)
type = c_build_qualified_type (type, type_quals, orig_qual_type,
orig_qual_indirect);
type = c_build_pointer_type (type);
type_quals = array_ptr_quals;
if (type_quals)
type = c_build_qualified_type (type, type_quals);
if (array_ptr_attrs != NULL_TREE)
warning_at (loc, OPT_Wattributes,
"attributes in parameter array declarator ignored");
size_varies = false;
array_parameter_p = true;
}
else if (TREE_CODE (type) == FUNCTION_TYPE)
{
if (type_quals & TYPE_QUAL_ATOMIC)
{
error_at (loc,
"%<_Atomic%>-qualified function type");
type_quals &= ~TYPE_QUAL_ATOMIC;
}
else if (type_quals)
pedwarn (loc, OPT_Wpedantic,
"ISO C forbids qualified function types");
if (type_quals)
type = c_build_qualified_type (type, type_quals);
type = c_build_pointer_type (type);
type_quals = TYPE_UNQUALIFIED;
}
else if (type_quals)
type = c_build_qualified_type (type, type_quals);
decl = build_decl (declarator->id_loc,
PARM_DECL, declarator->u.id, type);
if (size_varies)
C_DECL_VARIABLE_SIZE (decl) = 1;
C_ARRAY_PARAMETER (decl) = array_parameter_p;
if (type == error_mark_node)
promoted_type = type;
else
promoted_type = c_type_promotes_to (type);
DECL_ARG_TYPE (decl) = promoted_type;
if (declspecs->inline_p)
pedwarn (loc, 0, "parameter %q+D declared %<inline%>", decl);
if (declspecs->noreturn_p)
pedwarn (loc, 0, "parameter %q+D declared %<_Noreturn%>", decl);
}
else if (decl_context == FIELD)
{
gcc_assert (storage_class == csc_none && !threadp
&& !declspecs->inline_p && !declspecs->noreturn_p);
if (TREE_CODE (type) == FUNCTION_TYPE)
{
error_at (loc, "field %qE declared as a function", name);
type = build_pointer_type (type);
}
else if (TREE_CODE (type) != ERROR_MARK
&& !COMPLETE_OR_UNBOUND_ARRAY_TYPE_P (type))
{
if (name)
error_at (loc, "field %qE has incomplete type", name);
else
error_at (loc, "unnamed field has incomplete type");
type = error_mark_node;
}
else if (TREE_CODE (type) == ARRAY_TYPE
&& TYPE_DOMAIN (type) == NULL_TREE)
{
if (!in_system_header_at (input_location))
pedwarn_c90 (loc, OPT_Wpedantic, "ISO C90 does not "
"support flexible array members");
type = build_distinct_type_copy (TYPE_MAIN_VARIANT (type));
TYPE_DOMAIN (type) = build_range_type (sizetype, size_zero_node,
NULL_TREE);
if (orig_qual_indirect == 0)
orig_qual_type = NULL_TREE;
}
type = c_build_qualified_type (type, type_quals, orig_qual_type,
orig_qual_indirect);
decl = build_decl (declarator->id_loc,
FIELD_DECL, declarator->u.id, type);
DECL_NONADDRESSABLE_P (decl) = bitfield;
if (bitfield && !declarator->u.id)
{
TREE_NO_WARNING (decl) = 1;
DECL_PADDING_P (decl) = 1;
}
if (size_varies)
C_DECL_VARIABLE_SIZE (decl) = 1;
}
else if (TREE_CODE (type) == FUNCTION_TYPE)
{
if (storage_class == csc_register || threadp)
{
error_at (loc, "invalid storage class for function %qE", name);
}
else if (current_scope != file_scope)
{
if (storage_class == csc_auto)
pedwarn (loc, OPT_Wpedantic,
"invalid storage class for function %qE", name);
else if (storage_class == csc_static)
{
error_at (loc, "invalid storage class for function %qE", name);
if (funcdef_flag)
storage_class = declspecs->storage_class = csc_none;
else
return NULL_TREE;
}
}
decl = build_decl (declarator->id_loc,
FUNCTION_DECL, declarator->u.id, type);
decl = build_decl_attribute_variant (decl, decl_attr);
if (type_quals & TYPE_QUAL_ATOMIC)
{
error_at (loc,
"%<_Atomic%>-qualified function type");
type_quals &= ~TYPE_QUAL_ATOMIC;
}
else if (pedantic && type_quals && !DECL_IN_SYSTEM_HEADER (decl))
pedwarn (loc, OPT_Wpedantic,
"ISO C forbids qualified function types");
if (storage_class == csc_auto && current_scope != file_scope)
DECL_EXTERNAL (decl) = 0;
else if (declspecs->inline_p && storage_class != csc_static)
DECL_EXTERNAL (decl) = ((storage_class == csc_extern)
== flag_gnu89_inline);
else
DECL_EXTERNAL (decl) = !initialized;
TREE_PUBLIC (decl)
= !(storage_class == csc_static || storage_class == csc_auto);
if (funcdef_flag)
current_function_arg_info = arg_info;
if (declspecs->default_int_p)
C_FUNCTION_IMPLICIT_INT (decl) = 1;
if (flag_hosted && MAIN_NAME_P (declarator->u.id))
{
if (declspecs->inline_p)
pedwarn (loc, 0, "cannot inline function %<main%>");
if (declspecs->noreturn_p)
pedwarn (loc, 0, "%<main%> declared %<_Noreturn%>");
}
else
{
if (declspecs->inline_p)
DECL_DECLARED_INLINE_P (decl) = 1;
if (declspecs->noreturn_p)
{
if (flag_isoc99)
pedwarn_c99 (loc, OPT_Wpedantic,
"ISO C99 does not support %<_Noreturn%>");
else
pedwarn_c99 (loc, OPT_Wpedantic,
"ISO C90 does not support %<_Noreturn%>");
TREE_THIS_VOLATILE (decl) = 1;
}
}
}
else
{
int extern_ref = !initialized && storage_class == csc_extern;
type = c_build_qualified_type (type, type_quals, orig_qual_type,
orig_qual_indirect);
if (extern_ref && current_scope != file_scope)
{
tree global_decl  = identifier_global_value (declarator->u.id);
tree visible_decl = lookup_name (declarator->u.id);
if (global_decl
&& global_decl != visible_decl
&& VAR_P (global_decl)
&& !TREE_PUBLIC (global_decl))
error_at (loc, "variable previously declared %<static%> "
"redeclared %<extern%>");
}
decl = build_decl (declarator->id_loc,
VAR_DECL, declarator->u.id, type);
if (size_varies)
C_DECL_VARIABLE_SIZE (decl) = 1;
if (declspecs->inline_p)
pedwarn (loc, 0, "variable %q+D declared %<inline%>", decl);
if (declspecs->noreturn_p)
pedwarn (loc, 0, "variable %q+D declared %<_Noreturn%>", decl);
DECL_EXTERNAL (decl) = (storage_class == csc_extern);
if (current_scope == file_scope)
{
TREE_PUBLIC (decl) = storage_class != csc_static;
TREE_STATIC (decl) = !extern_ref;
}
else
{
TREE_STATIC (decl) = (storage_class == csc_static);
TREE_PUBLIC (decl) = extern_ref;
}
if (threadp)
set_decl_tls_model (decl, decl_default_tls_model (decl));
}
if ((storage_class == csc_extern
|| (storage_class == csc_none
&& TREE_CODE (type) == FUNCTION_TYPE
&& !funcdef_flag))
&& variably_modified_type_p (type, NULL_TREE))
{
if (TREE_CODE (type) == FUNCTION_TYPE)
error_at (loc, "non-nested function with variably modified type");
else
error_at (loc, "object with variably modified type must have "
"no linkage");
}
if (storage_class == csc_register)
{
C_DECL_REGISTER (decl) = 1;
DECL_REGISTER (decl) = 1;
}
c_apply_type_quals_to_decl (type_quals, decl);
if (alignas_align)
{
SET_DECL_ALIGN (decl, alignas_align * BITS_PER_UNIT);
DECL_USER_ALIGN (decl) = 1;
}
if (C_TYPE_FIELDS_VOLATILE (TREE_TYPE (decl))
&& (VAR_P (decl) ||  TREE_CODE (decl) == PARM_DECL
|| TREE_CODE (decl) == RESULT_DECL))
{
int was_reg = C_DECL_REGISTER (decl);
C_DECL_REGISTER (decl) = 0;
DECL_REGISTER (decl) = 0;
c_mark_addressable (decl);
C_DECL_REGISTER (decl) = was_reg;
}
gcc_assert (!HAS_DECL_ASSEMBLER_NAME_P (decl)
|| !DECL_ASSEMBLER_NAME_SET_P (decl));
if (warn_cxx_compat
&& VAR_P (decl)
&& TREE_PUBLIC (decl)
&& TREE_STATIC (decl)
&& (RECORD_OR_UNION_TYPE_P (TREE_TYPE (decl))
|| TREE_CODE (TREE_TYPE (decl)) == ENUMERAL_TYPE)
&& TYPE_NAME (TREE_TYPE (decl)) == NULL_TREE)
warning_at (DECL_SOURCE_LOCATION (decl), OPT_Wc___compat,
("non-local variable %qD with anonymous type is "
"questionable in C++"),
decl);
return decl;
}
}

static tree
grokparms (struct c_arg_info *arg_info, bool funcdef_flag)
{
tree arg_types = arg_info->types;
if (funcdef_flag && arg_info->had_vla_unspec)
{
error ("%<[*]%> not allowed in other than function prototype scope");
}
if (arg_types == NULL_TREE && !funcdef_flag
&& !in_system_header_at (input_location))
warning (OPT_Wstrict_prototypes,
"function declaration isn%'t a prototype");
if (arg_types == error_mark_node)
return NULL_TREE;
else if (arg_types && TREE_CODE (TREE_VALUE (arg_types)) == IDENTIFIER_NODE)
{
if (!funcdef_flag)
{
pedwarn (input_location, 0, "parameter names (without types) in "
"function declaration");
arg_info->parms = NULL_TREE;
}
else
arg_info->parms = arg_info->types;
arg_info->types = NULL_TREE;
return NULL_TREE;
}
else
{
tree parm, type, typelt;
unsigned int parmno;
for (parm = arg_info->parms, typelt = arg_types, parmno = 1;
parm;
parm = DECL_CHAIN (parm), typelt = TREE_CHAIN (typelt), parmno++)
{
type = TREE_VALUE (typelt);
if (type == error_mark_node)
continue;
if (!COMPLETE_TYPE_P (type))
{
if (funcdef_flag)
{
if (DECL_NAME (parm))
error_at (input_location,
"parameter %u (%q+D) has incomplete type",
parmno, parm);
else
error_at (DECL_SOURCE_LOCATION (parm),
"parameter %u has incomplete type",
parmno);
TREE_VALUE (typelt) = error_mark_node;
TREE_TYPE (parm) = error_mark_node;
arg_types = NULL_TREE;
}
else if (VOID_TYPE_P (type))
{
if (DECL_NAME (parm))
warning_at (input_location, 0,
"parameter %u (%q+D) has void type",
parmno, parm);
else
warning_at (DECL_SOURCE_LOCATION (parm), 0,
"parameter %u has void type",
parmno);
}
}
if (DECL_NAME (parm) && TREE_USED (parm))
warn_if_shadowing (parm);
}
return arg_types;
}
}
struct c_arg_info *
build_arg_info (void)
{
struct c_arg_info *ret = XOBNEW (&parser_obstack, struct c_arg_info);
ret->parms = NULL_TREE;
ret->tags = NULL;
ret->types = NULL_TREE;
ret->others = NULL_TREE;
ret->pending_sizes = NULL;
ret->had_vla_unspec = 0;
return ret;
}
struct c_arg_info *
get_parm_info (bool ellipsis, tree expr)
{
struct c_binding *b = current_scope->bindings;
struct c_arg_info *arg_info = build_arg_info ();
tree parms = NULL_TREE;
vec<c_arg_tag, va_gc> *tags = NULL;
tree types = NULL_TREE;
tree others = NULL_TREE;
bool gave_void_only_once_err = false;
arg_info->had_vla_unspec = current_scope->had_vla_unspec;
current_scope->bindings = 0;
gcc_assert (b);
if (b->prev == 0			    
&& TREE_CODE (b->decl) == PARM_DECL   
&& !DECL_NAME (b->decl)               
&& VOID_TYPE_P (TREE_TYPE (b->decl))) 
{
if (TYPE_QUALS (TREE_TYPE (b->decl)) != TYPE_UNQUALIFIED
|| C_DECL_REGISTER (b->decl))
error_at (b->locus, "%<void%> as only parameter may not be qualified");
if (ellipsis)
error_at (b->locus, "%<void%> must be the only parameter");
arg_info->types = void_list_node;
return arg_info;
}
if (!ellipsis)
types = void_list_node;
while (b)
{
tree decl = b->decl;
tree type = TREE_TYPE (decl);
c_arg_tag tag;
const char *keyword;
switch (TREE_CODE (decl))
{
case PARM_DECL:
if (b->id)
{
gcc_assert (I_SYMBOL_BINDING (b->id) == b);
I_SYMBOL_BINDING (b->id) = b->shadowed;
}
if (TREE_ASM_WRITTEN (decl))
error_at (b->locus,
"parameter %q+D has just a forward declaration", decl);
else if (VOID_TYPE_P (type) && !DECL_NAME (decl))
{
if (!gave_void_only_once_err)
{
error_at (b->locus, "%<void%> must be the only parameter");
gave_void_only_once_err = true;
}
}
else
{
DECL_CHAIN (decl) = parms;
parms = decl;
DECL_ARG_TYPE (decl) = type;
types = tree_cons (0, type, types);
}
break;
case ENUMERAL_TYPE: keyword = "enum"; goto tag;
case UNION_TYPE:    keyword = "union"; goto tag;
case RECORD_TYPE:   keyword = "struct"; goto tag;
tag:
if (b->id)
{
gcc_assert (I_TAG_BINDING (b->id) == b);
I_TAG_BINDING (b->id) = b->shadowed;
}
if (TREE_CODE (decl) != UNION_TYPE || b->id != NULL_TREE)
{
if (b->id)
warning_at (b->locus, 0,
"%<%s %E%> declared inside parameter list"
" will not be visible outside of this definition or"
" declaration", keyword, b->id);
else
warning_at (b->locus, 0,
"anonymous %s declared inside parameter list"
" will not be visible outside of this definition or"
" declaration", keyword);
}
tag.id = b->id;
tag.type = decl;
vec_safe_push (tags, tag);
break;
case FUNCTION_DECL:
gcc_assert (b->nested || seen_error ());
goto set_shadowed;
case CONST_DECL:
case TYPE_DECL:
gcc_assert (!b->nested);
DECL_CHAIN (decl) = others;
others = decl;
case ERROR_MARK:
set_shadowed:
if (b->id)
{
gcc_assert (I_SYMBOL_BINDING (b->id) == b);
I_SYMBOL_BINDING (b->id) = b->shadowed;
}
break;
case LABEL_DECL:
case VAR_DECL:
default:
gcc_unreachable ();
}
b = free_binding_and_advance (b);
}
arg_info->parms = parms;
arg_info->tags = tags;
arg_info->types = types;
arg_info->others = others;
arg_info->pending_sizes = expr;
return arg_info;
}

struct c_typespec
parser_xref_tag (location_t loc, enum tree_code code, tree name)
{
struct c_typespec ret;
tree ref;
location_t refloc;
ret.expr = NULL_TREE;
ret.expr_const_operands = true;
ref = lookup_tag (code, name, false, &refloc);
ret.kind = (ref ? ctsk_tagref : ctsk_tagfirstref);
if (ref && TREE_CODE (ref) == code)
{
if (C_TYPE_DEFINED_IN_STRUCT (ref)
&& loc != UNKNOWN_LOCATION
&& warn_cxx_compat)
{
switch (code)
{
case ENUMERAL_TYPE:
warning_at (loc, OPT_Wc___compat,
("enum type defined in struct or union "
"is not visible in C++"));
inform (refloc, "enum type defined here");
break;
case RECORD_TYPE:
warning_at (loc, OPT_Wc___compat,
("struct defined in struct or union "
"is not visible in C++"));
inform (refloc, "struct defined here");
break;
case UNION_TYPE:
warning_at (loc, OPT_Wc___compat,
("union defined in struct or union "
"is not visible in C++"));
inform (refloc, "union defined here");
break;
default:
gcc_unreachable();
}
}
ret.spec = ref;
return ret;
}
ref = make_node (code);
if (code == ENUMERAL_TYPE)
{
SET_TYPE_MODE (ref, TYPE_MODE (unsigned_type_node));
SET_TYPE_ALIGN (ref, TYPE_ALIGN (unsigned_type_node));
TYPE_USER_ALIGN (ref) = 0;
TYPE_UNSIGNED (ref) = 1;
TYPE_PRECISION (ref) = TYPE_PRECISION (unsigned_type_node);
TYPE_MIN_VALUE (ref) = TYPE_MIN_VALUE (unsigned_type_node);
TYPE_MAX_VALUE (ref) = TYPE_MAX_VALUE (unsigned_type_node);
}
pushtag (loc, name, ref);
ret.spec = ref;
return ret;
}
tree
xref_tag (enum tree_code code, tree name)
{
return parser_xref_tag (input_location, code, name).spec;
}

tree
start_struct (location_t loc, enum tree_code code, tree name,
struct c_struct_parse_info **enclosing_struct_parse_info)
{
tree ref = NULL_TREE;
location_t refloc = UNKNOWN_LOCATION;
if (name != NULL_TREE)
ref = lookup_tag (code, name, true, &refloc);
if (ref && TREE_CODE (ref) == code)
{
if (TYPE_STUB_DECL (ref))
refloc = DECL_SOURCE_LOCATION (TYPE_STUB_DECL (ref));
if (TYPE_SIZE (ref))
{
if (code == UNION_TYPE)
error_at (loc, "redefinition of %<union %E%>", name);
else
error_at (loc, "redefinition of %<struct %E%>", name);
if (refloc != UNKNOWN_LOCATION)
inform (refloc, "originally defined here");
ref = NULL_TREE;
}
else if (C_TYPE_BEING_DEFINED (ref))
{
if (code == UNION_TYPE)
error_at (loc, "nested redefinition of %<union %E%>", name);
else
error_at (loc, "nested redefinition of %<struct %E%>", name);
ref = NULL_TREE;
}
}
if (ref == NULL_TREE || TREE_CODE (ref) != code)
{
ref = make_node (code);
pushtag (loc, name, ref);
}
C_TYPE_BEING_DEFINED (ref) = 1;
for (tree v = TYPE_MAIN_VARIANT (ref); v; v = TYPE_NEXT_VARIANT (v))
TYPE_PACKED (v) = flag_pack_struct;
*enclosing_struct_parse_info = struct_parse_info;
struct_parse_info = new c_struct_parse_info ();
if (warn_cxx_compat && (in_sizeof || in_typeof || in_alignof))
warning_at (loc, OPT_Wc___compat,
"defining type in %qs expression is invalid in C++",
(in_sizeof
? "sizeof"
: (in_typeof ? "typeof" : "alignof")));
return ref;
}
tree
grokfield (location_t loc,
struct c_declarator *declarator, struct c_declspecs *declspecs,
tree width, tree *decl_attrs)
{
tree value;
if (declarator->kind == cdk_id && declarator->u.id == NULL_TREE
&& width == NULL_TREE)
{
tree type = declspecs->type;
bool ok = false;
if (RECORD_OR_UNION_TYPE_P (type)
&& (flag_ms_extensions
|| flag_plan9_extensions
|| !declspecs->typedef_p))
{
if (flag_ms_extensions || flag_plan9_extensions)
ok = true;
else if (TYPE_NAME (type) == NULL)
ok = true;
else
ok = false;
}
if (!ok)
{
pedwarn (loc, 0, "declaration does not declare anything");
return NULL_TREE;
}
if (flag_isoc99)
pedwarn_c99 (loc, OPT_Wpedantic,
"ISO C99 doesn%'t support unnamed structs/unions");
else
pedwarn_c99 (loc, OPT_Wpedantic,
"ISO C90 doesn%'t support unnamed structs/unions");
}
value = grokdeclarator (declarator, declspecs, FIELD, false,
width ? &width : NULL, decl_attrs, NULL, NULL,
DEPRECATED_NORMAL);
finish_decl (value, loc, NULL_TREE, NULL_TREE, NULL_TREE);
DECL_INITIAL (value) = width;
if (width)
SET_DECL_C_BIT_FIELD (value);
if (warn_cxx_compat && DECL_NAME (value) != NULL_TREE)
{
struct c_binding *b = I_SYMBOL_BINDING (DECL_NAME (value));
if (b != NULL)
{
if (!b->in_struct)
{
struct_parse_info->fields.safe_push (b);
b->in_struct = 1;
}
}
}
return value;
}

static bool
is_duplicate_field (tree x, tree y)
{
if (DECL_NAME (x) != NULL_TREE && DECL_NAME (x) == DECL_NAME (y))
return true;
if (flag_plan9_extensions
&& (DECL_NAME (x) == NULL_TREE || DECL_NAME (y) == NULL_TREE))
{
tree xt, xn, yt, yn;
xt = TREE_TYPE (x);
if (DECL_NAME (x) != NULL_TREE)
xn = DECL_NAME (x);
else if (RECORD_OR_UNION_TYPE_P (xt)
&& TYPE_NAME (xt) != NULL_TREE
&& TREE_CODE (TYPE_NAME (xt)) == TYPE_DECL)
xn = DECL_NAME (TYPE_NAME (xt));
else
xn = NULL_TREE;
yt = TREE_TYPE (y);
if (DECL_NAME (y) != NULL_TREE)
yn = DECL_NAME (y);
else if (RECORD_OR_UNION_TYPE_P (yt)
&& TYPE_NAME (yt) != NULL_TREE
&& TREE_CODE (TYPE_NAME (yt)) == TYPE_DECL)
yn = DECL_NAME (TYPE_NAME (yt));
else
yn = NULL_TREE;
if (xn != NULL_TREE && xn == yn)
return true;
}
return false;
}
static void
detect_field_duplicates_hash (tree fieldlist,
hash_table<nofree_ptr_hash <tree_node> > *htab)
{
tree x, y;
tree_node **slot;
for (x = fieldlist; x ; x = DECL_CHAIN (x))
if ((y = DECL_NAME (x)) != NULL_TREE)
{
slot = htab->find_slot (y, INSERT);
if (*slot)
{
error ("duplicate member %q+D", x);
DECL_NAME (x) = NULL_TREE;
}
*slot = y;
}
else if (RECORD_OR_UNION_TYPE_P (TREE_TYPE (x)))
{
detect_field_duplicates_hash (TYPE_FIELDS (TREE_TYPE (x)), htab);
if (flag_plan9_extensions
&& TYPE_NAME (TREE_TYPE (x)) != NULL_TREE
&& TREE_CODE (TYPE_NAME (TREE_TYPE (x))) == TYPE_DECL)
{
tree xn = DECL_NAME (TYPE_NAME (TREE_TYPE (x)));
slot = htab->find_slot (xn, INSERT);
if (*slot)
error ("duplicate member %q+D", TYPE_NAME (TREE_TYPE (x)));
*slot = xn;
}
}
}
static void
detect_field_duplicates (tree fieldlist)
{
tree x, y;
int timeout = 10;
if (c_dialect_objc ())
if (objc_detect_field_duplicates (false))
return;
if (!fieldlist || !DECL_CHAIN (fieldlist))
return;
x = fieldlist;
do {
timeout--;
if (DECL_NAME (x) == NULL_TREE
&& RECORD_OR_UNION_TYPE_P (TREE_TYPE (x)))
timeout = 0;
x = DECL_CHAIN (x);
} while (timeout > 0 && x);
if (timeout > 0)
{
for (x = DECL_CHAIN (fieldlist); x; x = DECL_CHAIN (x))
if (DECL_NAME (x)
|| (flag_plan9_extensions
&& DECL_NAME (x) == NULL_TREE
&& RECORD_OR_UNION_TYPE_P (TREE_TYPE (x))
&& TYPE_NAME (TREE_TYPE (x)) != NULL_TREE
&& TREE_CODE (TYPE_NAME (TREE_TYPE (x))) == TYPE_DECL))
{
for (y = fieldlist; y != x; y = TREE_CHAIN (y))
if (is_duplicate_field (y, x))
{
error ("duplicate member %q+D", x);
DECL_NAME (x) = NULL_TREE;
}
}
}
else
{
hash_table<nofree_ptr_hash <tree_node> > htab (37);
detect_field_duplicates_hash (fieldlist, &htab);
}
}
static void
warn_cxx_compat_finish_struct (tree fieldlist, enum tree_code code,
location_t record_loc)
{
unsigned int ix;
tree x;
struct c_binding *b;
if (fieldlist == NULL_TREE)
{
if (code == RECORD_TYPE)
warning_at (record_loc, OPT_Wc___compat,
"empty struct has size 0 in C, size 1 in C++");
else
warning_at (record_loc, OPT_Wc___compat,
"empty union has size 0 in C, size 1 in C++");
}
FOR_EACH_VEC_ELT (struct_parse_info->struct_types, ix, x)
C_TYPE_DEFINED_IN_STRUCT (x) = 1;
if (!struct_parse_info->typedefs_seen.is_empty ()
&& fieldlist != NULL_TREE)
{
hash_set<tree> tset;
FOR_EACH_VEC_ELT (struct_parse_info->typedefs_seen, ix, x)
tset.add (DECL_NAME (x));
for (x = fieldlist; x != NULL_TREE; x = DECL_CHAIN (x))
{
if (DECL_NAME (x) != NULL_TREE
&& tset.contains (DECL_NAME (x)))
{
warning_at (DECL_SOURCE_LOCATION (x), OPT_Wc___compat,
("using %qD as both field and typedef name is "
"invalid in C++"),
x);
}
}
}
FOR_EACH_VEC_ELT (struct_parse_info->fields, ix, b)
b->in_struct = 0;
}
static int
field_decl_cmp (const void *x_p, const void *y_p)
{
const tree *const x = (const tree *) x_p;
const tree *const y = (const tree *) y_p;
if (DECL_NAME (*x) == DECL_NAME (*y))
return (TREE_CODE (*y) == TYPE_DECL) - (TREE_CODE (*x) == TYPE_DECL);
if (DECL_NAME (*x) == NULL_TREE)
return -1;
if (DECL_NAME (*y) == NULL_TREE)
return 1;
if (DECL_NAME (*x) < DECL_NAME (*y))
return -1;
return 1;
}
tree
finish_struct (location_t loc, tree t, tree fieldlist, tree attributes,
struct c_struct_parse_info *enclosing_struct_parse_info)
{
tree x;
bool toplevel = file_scope == current_scope;
TYPE_SIZE (t) = NULL_TREE;
decl_attributes (&t, attributes, (int) ATTR_FLAG_TYPE_IN_PLACE);
if (pedantic)
{
for (x = fieldlist; x; x = DECL_CHAIN (x))
{
if (DECL_NAME (x) != NULL_TREE)
break;
if (flag_isoc11 && RECORD_OR_UNION_TYPE_P (TREE_TYPE (x)))
break;
}
if (x == NULL_TREE)
{
if (TREE_CODE (t) == UNION_TYPE)
{
if (fieldlist)
pedwarn (loc, OPT_Wpedantic, "union has no named members");
else
pedwarn (loc, OPT_Wpedantic, "union has no members");
}
else
{
if (fieldlist)
pedwarn (loc, OPT_Wpedantic, "struct has no named members");
else
pedwarn (loc, OPT_Wpedantic, "struct has no members");
}
}
}
bool saw_named_field = false;
for (x = fieldlist; x; x = DECL_CHAIN (x))
{
if (TREE_TYPE (x) == error_mark_node)
continue;
DECL_CONTEXT (x) = t;
if (TREE_READONLY (x))
C_TYPE_FIELDS_READONLY (t) = 1;
else
{
tree t1 = strip_array_types (TREE_TYPE (x));
if (RECORD_OR_UNION_TYPE_P (t1) && C_TYPE_FIELDS_READONLY (t1))
C_TYPE_FIELDS_READONLY (t) = 1;
}
if (TREE_THIS_VOLATILE (x))
C_TYPE_FIELDS_VOLATILE (t) = 1;
if (C_DECL_VARIABLE_SIZE (x))
C_TYPE_VARIABLE_SIZE (t) = 1;
if (DECL_C_BIT_FIELD (x))
{
unsigned HOST_WIDE_INT width = tree_to_uhwi (DECL_INITIAL (x));
DECL_SIZE (x) = bitsize_int (width);
DECL_BIT_FIELD (x) = 1;
}
if (TYPE_PACKED (t)
&& (DECL_BIT_FIELD (x)
|| TYPE_ALIGN (TREE_TYPE (x)) > BITS_PER_UNIT))
DECL_PACKED (x) = 1;
if (TREE_CODE (TREE_TYPE (x)) == ARRAY_TYPE
&& TYPE_SIZE (TREE_TYPE (x)) == NULL_TREE
&& TYPE_DOMAIN (TREE_TYPE (x)) != NULL_TREE
&& TYPE_MAX_VALUE (TYPE_DOMAIN (TREE_TYPE (x))) == NULL_TREE)
{
if (TREE_CODE (t) == UNION_TYPE)
{
error_at (DECL_SOURCE_LOCATION (x),
"flexible array member in union");
TREE_TYPE (x) = error_mark_node;
}
else if (DECL_CHAIN (x) != NULL_TREE)
{
error_at (DECL_SOURCE_LOCATION (x),
"flexible array member not at end of struct");
TREE_TYPE (x) = error_mark_node;
}
else if (!saw_named_field)
{
error_at (DECL_SOURCE_LOCATION (x),
"flexible array member in a struct with no named "
"members");
TREE_TYPE (x) = error_mark_node;
}
}
if (pedantic && TREE_CODE (t) == RECORD_TYPE
&& flexible_array_type_p (TREE_TYPE (x)))
pedwarn (DECL_SOURCE_LOCATION (x), OPT_Wpedantic,
"invalid use of structure with flexible array member");
if (DECL_NAME (x)
|| RECORD_OR_UNION_TYPE_P (TREE_TYPE (x)))
saw_named_field = true;
}
detect_field_duplicates (fieldlist);
TYPE_FIELDS (t) = fieldlist;
maybe_apply_pragma_scalar_storage_order (t);
layout_type (t);
if (TYPE_SIZE_UNIT (t)
&& TREE_CODE (TYPE_SIZE_UNIT (t)) == INTEGER_CST
&& !TREE_OVERFLOW (TYPE_SIZE_UNIT (t))
&& !valid_constant_size_p (TYPE_SIZE_UNIT (t)))
error ("type %qT is too large", t);
for (tree field = fieldlist; field; field = DECL_CHAIN (field))
{
if (TREE_CODE (field) == FIELD_DECL
&& DECL_INITIAL (field)
&& TREE_TYPE (field) != error_mark_node)
{
unsigned HOST_WIDE_INT width
= tree_to_uhwi (DECL_INITIAL (field));
tree type = TREE_TYPE (field);
if (width != TYPE_PRECISION (type))
{
TREE_TYPE (field)
= c_build_bitfield_integer_type (width, TYPE_UNSIGNED (type));
SET_DECL_MODE (field, TYPE_MODE (TREE_TYPE (field)));
}
DECL_INITIAL (field) = NULL_TREE;
}
else if (TYPE_REVERSE_STORAGE_ORDER (t)
&& TREE_CODE (field) == FIELD_DECL
&& TREE_CODE (TREE_TYPE (field)) == ARRAY_TYPE)
{
tree ftype = TREE_TYPE (field);
tree ctype = strip_array_types (ftype);
if (!RECORD_OR_UNION_TYPE_P (ctype) && TYPE_MODE (ctype) != QImode)
{
tree fmain_type = TYPE_MAIN_VARIANT (ftype);
tree *typep = &fmain_type;
do {
*typep = build_distinct_type_copy (*typep);
TYPE_REVERSE_STORAGE_ORDER (*typep) = 1;
typep = &TREE_TYPE (*typep);
} while (TREE_CODE (*typep) == ARRAY_TYPE);
TREE_TYPE (field)
= c_build_qualified_type (fmain_type, TYPE_QUALS (ftype));
}
}
}
TYPE_FIELDS (t) = fieldlist;
{
int len = 0;
for (x = fieldlist; x; x = DECL_CHAIN (x))
{
if (len > 15 || DECL_NAME (x) == NULL)
break;
len += 1;
}
if (len > 15)
{
tree *field_array;
struct lang_type *space;
struct sorted_fields_type *space2;
len += list_length (x);
space = ggc_cleared_alloc<struct lang_type> ();
space2 = (sorted_fields_type *) ggc_internal_alloc
(sizeof (struct sorted_fields_type) + len * sizeof (tree));
len = 0;
space->s = space2;
field_array = &space2->elts[0];
for (x = fieldlist; x; x = DECL_CHAIN (x))
{
field_array[len++] = x;
if (DECL_NAME (x) == NULL)
break;
}
if (x == NULL)
{
TYPE_LANG_SPECIFIC (t) = space;
TYPE_LANG_SPECIFIC (t)->s->len = len;
field_array = TYPE_LANG_SPECIFIC (t)->s->elts;
qsort (field_array, len, sizeof (tree), field_decl_cmp);
}
}
}
tree incomplete_vars = C_TYPE_INCOMPLETE_VARS (TYPE_MAIN_VARIANT (t));
for (x = TYPE_MAIN_VARIANT (t); x; x = TYPE_NEXT_VARIANT (x))
{
TYPE_FIELDS (x) = TYPE_FIELDS (t);
TYPE_LANG_SPECIFIC (x) = TYPE_LANG_SPECIFIC (t);
C_TYPE_FIELDS_READONLY (x) = C_TYPE_FIELDS_READONLY (t);
C_TYPE_FIELDS_VOLATILE (x) = C_TYPE_FIELDS_VOLATILE (t);
C_TYPE_VARIABLE_SIZE (x) = C_TYPE_VARIABLE_SIZE (t);
C_TYPE_INCOMPLETE_VARS (x) = NULL_TREE;
}
if (TREE_CODE (t) == UNION_TYPE
&& TYPE_TRANSPARENT_AGGR (t)
&& (!TYPE_FIELDS (t) || TYPE_MODE (t) != DECL_MODE (TYPE_FIELDS (t))))
{
TYPE_TRANSPARENT_AGGR (t) = 0;
warning_at (loc, 0, "union cannot be made transparent");
}
if (TYPE_STUB_DECL (t))
DECL_SOURCE_LOCATION (TYPE_STUB_DECL (t)) = loc;
rest_of_type_compilation (t, toplevel);
for (x = incomplete_vars; x; x = TREE_CHAIN (x))
{
tree decl = TREE_VALUE (x);
if (TREE_CODE (TREE_TYPE (decl)) == ARRAY_TYPE)
layout_array_type (TREE_TYPE (decl));
if (TREE_CODE (decl) != TYPE_DECL)
{
layout_decl (decl, 0);
if (c_dialect_objc ())
objc_check_decl (decl);
rest_of_decl_compilation (decl, toplevel, 0);
}
}
if (building_stmt_list_p () && variably_modified_type_p (t, NULL_TREE))
add_stmt (build_stmt (loc,
DECL_EXPR, build_decl (loc, TYPE_DECL, NULL, t)));
if (warn_cxx_compat)
warn_cxx_compat_finish_struct (fieldlist, TREE_CODE (t), loc);
delete struct_parse_info;
struct_parse_info = enclosing_struct_parse_info;
if (warn_cxx_compat
&& struct_parse_info != NULL
&& !in_sizeof && !in_typeof && !in_alignof)
struct_parse_info->struct_types.safe_push (t);
return t;
}
static struct {
gt_pointer_operator new_value;
void *cookie;
} resort_data;
static int
resort_field_decl_cmp (const void *x_p, const void *y_p)
{
const tree *const x = (const tree *) x_p;
const tree *const y = (const tree *) y_p;
if (DECL_NAME (*x) == DECL_NAME (*y))
return (TREE_CODE (*y) == TYPE_DECL) - (TREE_CODE (*x) == TYPE_DECL);
if (DECL_NAME (*x) == NULL_TREE)
return -1;
if (DECL_NAME (*y) == NULL_TREE)
return 1;
{
tree d1 = DECL_NAME (*x);
tree d2 = DECL_NAME (*y);
resort_data.new_value (&d1, resort_data.cookie);
resort_data.new_value (&d2, resort_data.cookie);
if (d1 < d2)
return -1;
}
return 1;
}
void
resort_sorted_fields (void *obj,
void * ARG_UNUSED (orig_obj),
gt_pointer_operator new_value,
void *cookie)
{
struct sorted_fields_type *sf = (struct sorted_fields_type *) obj;
resort_data.new_value = new_value;
resort_data.cookie = cookie;
qsort (&sf->elts[0], sf->len, sizeof (tree),
resort_field_decl_cmp);
}
static void
layout_array_type (tree t)
{
if (TREE_CODE (TREE_TYPE (t)) == ARRAY_TYPE)
layout_array_type (TREE_TYPE (t));
layout_type (t);
}

tree
start_enum (location_t loc, struct c_enum_contents *the_enum, tree name)
{
tree enumtype = NULL_TREE;
location_t enumloc = UNKNOWN_LOCATION;
if (name != NULL_TREE)
enumtype = lookup_tag (ENUMERAL_TYPE, name, true, &enumloc);
if (enumtype == NULL_TREE || TREE_CODE (enumtype) != ENUMERAL_TYPE)
{
enumtype = make_node (ENUMERAL_TYPE);
pushtag (loc, name, enumtype);
}
else if (TYPE_STUB_DECL (enumtype))
{
enumloc = DECL_SOURCE_LOCATION (TYPE_STUB_DECL (enumtype));
DECL_SOURCE_LOCATION (TYPE_STUB_DECL (enumtype)) = loc;
}
if (C_TYPE_BEING_DEFINED (enumtype))
error_at (loc, "nested redefinition of %<enum %E%>", name);
C_TYPE_BEING_DEFINED (enumtype) = 1;
if (TYPE_VALUES (enumtype) != NULL_TREE)
{
error_at (loc, "redeclaration of %<enum %E%>", name);
if (enumloc != UNKNOWN_LOCATION)
inform (enumloc, "originally defined here");
TYPE_VALUES (enumtype) = NULL_TREE;
}
the_enum->enum_next_value = integer_zero_node;
the_enum->enum_overflow = 0;
if (flag_short_enums)
for (tree v = TYPE_MAIN_VARIANT (enumtype); v; v = TYPE_NEXT_VARIANT (v))
TYPE_PACKED (v) = 1;
if (warn_cxx_compat && (in_sizeof || in_typeof || in_alignof))
warning_at (loc, OPT_Wc___compat,
"defining type in %qs expression is invalid in C++",
(in_sizeof
? "sizeof"
: (in_typeof ? "typeof" : "alignof")));
return enumtype;
}
tree
finish_enum (tree enumtype, tree values, tree attributes)
{
tree pair, tem;
tree minnode = NULL_TREE, maxnode = NULL_TREE;
int precision;
signop sign;
bool toplevel = (file_scope == current_scope);
struct lang_type *lt;
decl_attributes (&enumtype, attributes, (int) ATTR_FLAG_TYPE_IN_PLACE);
if (values == error_mark_node)
minnode = maxnode = integer_zero_node;
else
{
minnode = maxnode = TREE_VALUE (values);
for (pair = TREE_CHAIN (values); pair; pair = TREE_CHAIN (pair))
{
tree value = TREE_VALUE (pair);
if (tree_int_cst_lt (maxnode, value))
maxnode = value;
if (tree_int_cst_lt (value, minnode))
minnode = value;
}
}
sign = (tree_int_cst_sgn (minnode) >= 0) ? UNSIGNED : SIGNED;
precision = MAX (tree_int_cst_min_precision (minnode, sign),
tree_int_cst_min_precision (maxnode, sign));
if (TYPE_PRECISION (enumtype) && lookup_attribute ("mode", attributes))
{
if (precision > TYPE_PRECISION (enumtype))
{
TYPE_PRECISION (enumtype) = 0;
error ("specified mode too small for enumeral values");
}
else
precision = TYPE_PRECISION (enumtype);
}
else
TYPE_PRECISION (enumtype) = 0;
if (TYPE_PACKED (enumtype)
|| precision > TYPE_PRECISION (integer_type_node)
|| TYPE_PRECISION (enumtype))
{
tem = c_common_type_for_size (precision, sign == UNSIGNED ? 1 : 0);
if (tem == NULL)
{
warning (0, "enumeration values exceed range of largest integer");
tem = long_long_integer_type_node;
}
}
else
tem = sign == UNSIGNED ? unsigned_type_node : integer_type_node;
TYPE_MIN_VALUE (enumtype) = TYPE_MIN_VALUE (tem);
TYPE_MAX_VALUE (enumtype) = TYPE_MAX_VALUE (tem);
TYPE_UNSIGNED (enumtype) = TYPE_UNSIGNED (tem);
SET_TYPE_ALIGN (enumtype, TYPE_ALIGN (tem));
TYPE_SIZE (enumtype) = NULL_TREE;
TYPE_PRECISION (enumtype) = TYPE_PRECISION (tem);
layout_type (enumtype);
if (values != error_mark_node)
{
for (pair = values; pair; pair = TREE_CHAIN (pair))
{
tree enu = TREE_PURPOSE (pair);
tree ini = DECL_INITIAL (enu);
TREE_TYPE (enu) = enumtype;
if (TREE_TYPE (ini) != integer_type_node)
ini = convert (enumtype, ini);
DECL_INITIAL (enu) = ini;
TREE_PURPOSE (pair) = DECL_NAME (enu);
TREE_VALUE (pair) = ini;
}
TYPE_VALUES (enumtype) = values;
}
lt = ggc_cleared_alloc<struct lang_type> ();
lt->enum_min = minnode;
lt->enum_max = maxnode;
TYPE_LANG_SPECIFIC (enumtype) = lt;
for (tem = TYPE_MAIN_VARIANT (enumtype); tem; tem = TYPE_NEXT_VARIANT (tem))
{
if (tem == enumtype)
continue;
TYPE_VALUES (tem) = TYPE_VALUES (enumtype);
TYPE_MIN_VALUE (tem) = TYPE_MIN_VALUE (enumtype);
TYPE_MAX_VALUE (tem) = TYPE_MAX_VALUE (enumtype);
TYPE_SIZE (tem) = TYPE_SIZE (enumtype);
TYPE_SIZE_UNIT (tem) = TYPE_SIZE_UNIT (enumtype);
SET_TYPE_MODE (tem, TYPE_MODE (enumtype));
TYPE_PRECISION (tem) = TYPE_PRECISION (enumtype);
SET_TYPE_ALIGN (tem, TYPE_ALIGN (enumtype));
TYPE_USER_ALIGN (tem) = TYPE_USER_ALIGN (enumtype);
TYPE_UNSIGNED (tem) = TYPE_UNSIGNED (enumtype);
TYPE_LANG_SPECIFIC (tem) = TYPE_LANG_SPECIFIC (enumtype);
}
rest_of_type_compilation (enumtype, toplevel);
if (warn_cxx_compat
&& struct_parse_info != NULL
&& !in_sizeof && !in_typeof && !in_alignof)
struct_parse_info->struct_types.safe_push (enumtype);
return enumtype;
}
tree
build_enumerator (location_t decl_loc, location_t loc,
struct c_enum_contents *the_enum, tree name, tree value)
{
tree decl, type;
if (value != NULL_TREE)
{
if (value == error_mark_node)
value = NULL_TREE;
else if (!INTEGRAL_TYPE_P (TREE_TYPE (value)))
{
error_at (loc, "enumerator value for %qE is not an integer constant",
name);
value = NULL_TREE;
}
else
{
if (TREE_CODE (value) != INTEGER_CST)
{
value = c_fully_fold (value, false, NULL);
if (TREE_CODE (value) == INTEGER_CST)
pedwarn (loc, OPT_Wpedantic,
"enumerator value for %qE is not an integer "
"constant expression", name);
}
if (TREE_CODE (value) != INTEGER_CST)
{
error ("enumerator value for %qE is not an integer constant",
name);
value = NULL_TREE;
}
else
{
value = default_conversion (value);
constant_expression_warning (value);
}
}
}
if (value == NULL_TREE)
{
value = the_enum->enum_next_value;
if (the_enum->enum_overflow)
error_at (loc, "overflow in enumeration values");
}
else if (!int_fits_type_p (value, integer_type_node))
pedwarn (loc, OPT_Wpedantic,
"ISO C restricts enumerator values to range of %<int%>");
if (int_fits_type_p (value, integer_type_node))
value = convert (integer_type_node, value);
the_enum->enum_next_value
= build_binary_op (EXPR_LOC_OR_LOC (value, input_location),
PLUS_EXPR, value, integer_one_node, false);
the_enum->enum_overflow = tree_int_cst_lt (the_enum->enum_next_value, value);
type = TREE_TYPE (value);
type = c_common_type_for_size (MAX (TYPE_PRECISION (type),
TYPE_PRECISION (integer_type_node)),
(TYPE_PRECISION (type)
>= TYPE_PRECISION (integer_type_node)
&& TYPE_UNSIGNED (type)));
decl = build_decl (decl_loc, CONST_DECL, name, type);
DECL_INITIAL (decl) = convert (type, value);
pushdecl (decl);
return tree_cons (decl, value, NULL_TREE);
}

bool
start_function (struct c_declspecs *declspecs, struct c_declarator *declarator,
tree attributes)
{
tree decl1, old_decl;
tree restype, resdecl;
location_t loc;
current_function_returns_value = 0;  
current_function_returns_null = 0;
current_function_returns_abnormally = 0;
warn_about_return_type = 0;
c_switch_stack = NULL;
c_break_label = c_cont_label = size_zero_node;
decl1 = grokdeclarator (declarator, declspecs, FUNCDEF, true, NULL,
&attributes, NULL, NULL, DEPRECATED_NORMAL);
invoke_plugin_callbacks (PLUGIN_START_PARSE_FUNCTION, decl1);
if (decl1 == NULL_TREE
|| TREE_CODE (decl1) != FUNCTION_DECL)
return false;
loc = DECL_SOURCE_LOCATION (decl1);
c_decl_attributes (&decl1, attributes, 0);
if (DECL_DECLARED_INLINE_P (decl1)
&& DECL_UNINLINABLE (decl1)
&& lookup_attribute ("noinline", DECL_ATTRIBUTES (decl1)))
warning_at (loc, OPT_Wattributes,
"inline function %qD given attribute noinline",
decl1);
if (declspecs->inline_p
&& !flag_gnu89_inline
&& TREE_CODE (decl1) == FUNCTION_DECL
&& (lookup_attribute ("gnu_inline", DECL_ATTRIBUTES (decl1))
|| current_function_decl))
{
if (declspecs->storage_class != csc_static)
DECL_EXTERNAL (decl1) = !DECL_EXTERNAL (decl1);
}
announce_function (decl1);
if (!COMPLETE_OR_VOID_TYPE_P (TREE_TYPE (TREE_TYPE (decl1))))
{
error_at (loc, "return type is an incomplete type");
TREE_TYPE (decl1)
= build_function_type (void_type_node,
TYPE_ARG_TYPES (TREE_TYPE (decl1)));
}
if (warn_about_return_type)
warn_defaults_to (loc, flag_isoc99 ? OPT_Wimplicit_int
: (warn_return_type ? OPT_Wreturn_type
: OPT_Wimplicit_int),
"return type defaults to %<int%>");
DECL_INITIAL (decl1) = error_mark_node;
if (current_function_decl != NULL_TREE)
TREE_PUBLIC (decl1) = 0;
old_decl = lookup_name_in_scope (DECL_NAME (decl1), current_scope);
if (old_decl && TREE_CODE (old_decl) != FUNCTION_DECL)
old_decl = NULL_TREE;
current_function_prototype_locus = UNKNOWN_LOCATION;
current_function_prototype_built_in = false;
current_function_prototype_arg_types = NULL_TREE;
if (!prototype_p (TREE_TYPE (decl1)))
{
if (old_decl != NULL_TREE
&& TREE_CODE (TREE_TYPE (old_decl)) == FUNCTION_TYPE
&& comptypes (TREE_TYPE (TREE_TYPE (decl1)),
TREE_TYPE (TREE_TYPE (old_decl))))
{
if (stdarg_p (TREE_TYPE (old_decl)))
{
warning_at (loc, 0, "%q+D defined as variadic function "
"without prototype", decl1);
locate_old_decl (old_decl);
}
TREE_TYPE (decl1) = composite_type (TREE_TYPE (old_decl),
TREE_TYPE (decl1));
current_function_prototype_locus = DECL_SOURCE_LOCATION (old_decl);
current_function_prototype_built_in
= C_DECL_BUILTIN_PROTOTYPE (old_decl);
current_function_prototype_arg_types
= TYPE_ARG_TYPES (TREE_TYPE (decl1));
}
if (TREE_PUBLIC (decl1))
{
struct c_binding *b;
for (b = I_SYMBOL_BINDING (DECL_NAME (decl1)); b; b = b->shadowed)
if (B_IN_SCOPE (b, external_scope))
break;
if (b)
{
tree ext_decl, ext_type;
ext_decl = b->decl;
ext_type = b->u.type ? b->u.type : TREE_TYPE (ext_decl);
if (TREE_CODE (ext_type) == FUNCTION_TYPE
&& comptypes (TREE_TYPE (TREE_TYPE (decl1)),
TREE_TYPE (ext_type)))
{
current_function_prototype_locus
= DECL_SOURCE_LOCATION (ext_decl);
current_function_prototype_built_in
= C_DECL_BUILTIN_PROTOTYPE (ext_decl);
current_function_prototype_arg_types
= TYPE_ARG_TYPES (ext_type);
}
}
}
}
if (warn_strict_prototypes
&& old_decl != error_mark_node
&& !prototype_p (TREE_TYPE (decl1))
&& C_DECL_ISNT_PROTOTYPE (old_decl))
warning_at (loc, OPT_Wstrict_prototypes,
"function declaration isn%'t a prototype");
else if (warn_missing_prototypes
&& old_decl != error_mark_node
&& TREE_PUBLIC (decl1)
&& !MAIN_NAME_P (DECL_NAME (decl1))
&& C_DECL_ISNT_PROTOTYPE (old_decl)
&& !DECL_DECLARED_INLINE_P (decl1))
warning_at (loc, OPT_Wmissing_prototypes,
"no previous prototype for %qD", decl1);
else if (warn_missing_prototypes
&& old_decl != NULL_TREE
&& old_decl != error_mark_node
&& TREE_USED (old_decl)
&& !prototype_p (TREE_TYPE (old_decl)))
warning_at (loc, OPT_Wmissing_prototypes,
"%qD was used with no prototype before its definition", decl1);
else if (warn_missing_declarations
&& TREE_PUBLIC (decl1)
&& old_decl == NULL_TREE
&& !MAIN_NAME_P (DECL_NAME (decl1))
&& !DECL_DECLARED_INLINE_P (decl1))
warning_at (loc, OPT_Wmissing_declarations,
"no previous declaration for %qD",
decl1);
else if (warn_missing_declarations
&& old_decl != NULL_TREE
&& old_decl != error_mark_node
&& TREE_USED (old_decl)
&& C_DECL_IMPLICIT (old_decl))
warning_at (loc, OPT_Wmissing_declarations,
"%qD was used with no declaration before its definition", decl1);
TREE_STATIC (decl1) = 1;
gcc_assert (!DECL_ASSEMBLER_NAME_SET_P (decl1));
if (current_scope == file_scope)
maybe_apply_pragma_weak (decl1);
if (warn_main && MAIN_NAME_P (DECL_NAME (decl1)))
{
if (TYPE_MAIN_VARIANT (TREE_TYPE (TREE_TYPE (decl1)))
!= integer_type_node)
pedwarn (loc, OPT_Wmain, "return type of %qD is not %<int%>", decl1);
else if (TYPE_ATOMIC (TREE_TYPE (TREE_TYPE (decl1))))
pedwarn (loc, OPT_Wmain, "%<_Atomic%>-qualified return type of %qD",
decl1);
check_main_parameter_types (decl1);
if (!TREE_PUBLIC (decl1))
pedwarn (loc, OPT_Wmain,
"%qD is normally a non-static function", decl1);
}
current_function_decl = pushdecl (decl1);
push_scope ();
declare_parm_level ();
restype = TREE_TYPE (TREE_TYPE (current_function_decl));
resdecl = build_decl (loc, RESULT_DECL, NULL_TREE, restype);
DECL_ARTIFICIAL (resdecl) = 1;
DECL_IGNORED_P (resdecl) = 1;
DECL_RESULT (current_function_decl) = resdecl;
start_fname_decls ();
return true;
}

static void
store_parm_decls_newstyle (tree fndecl, const struct c_arg_info *arg_info)
{
tree decl;
c_arg_tag *tag;
unsigned ix;
if (current_scope->bindings)
{
error_at (DECL_SOURCE_LOCATION (fndecl),
"old-style parameter declarations in prototyped "
"function definition");
pop_scope ();
push_scope ();
}
else if (!in_system_header_at (input_location)
&& !current_function_scope
&& arg_info->types != error_mark_node)
warning_at (DECL_SOURCE_LOCATION (fndecl), OPT_Wtraditional,
"traditional C rejects ISO C style function definitions");
for (decl = arg_info->parms; decl; decl = DECL_CHAIN (decl))
{
DECL_CONTEXT (decl) = current_function_decl;
if (DECL_NAME (decl))
{
bind (DECL_NAME (decl), decl, current_scope,
false, false,
UNKNOWN_LOCATION);
if (!TREE_USED (decl))
warn_if_shadowing (decl);
}
else
error_at (DECL_SOURCE_LOCATION (decl), "parameter name omitted");
}
DECL_ARGUMENTS (fndecl) = arg_info->parms;
for (decl = arg_info->others; decl; decl = DECL_CHAIN (decl))
{
DECL_CONTEXT (decl) = current_function_decl;
if (DECL_NAME (decl))
bind (DECL_NAME (decl), decl, current_scope,
false,
(TREE_CODE (decl) == FUNCTION_DECL),
UNKNOWN_LOCATION);
}
FOR_EACH_VEC_SAFE_ELT_REVERSE (arg_info->tags, ix, tag)
if (tag->id)
bind (tag->id, tag->type, current_scope,
false, false, UNKNOWN_LOCATION);
}
static void
store_parm_decls_oldstyle (tree fndecl, const struct c_arg_info *arg_info)
{
struct c_binding *b;
tree parm, decl, last;
tree parmids = arg_info->parms;
hash_set<tree> seen_args;
if (!in_system_header_at (input_location))
warning_at (DECL_SOURCE_LOCATION (fndecl),
OPT_Wold_style_definition, "old-style function definition");
for (parm = parmids; parm; parm = TREE_CHAIN (parm))
{
if (TREE_VALUE (parm) == NULL_TREE)
{
error_at (DECL_SOURCE_LOCATION (fndecl),
"parameter name missing from parameter list");
TREE_PURPOSE (parm) = NULL_TREE;
continue;
}
b = I_SYMBOL_BINDING (TREE_VALUE (parm));
if (b && B_IN_CURRENT_SCOPE (b))
{
decl = b->decl;
if (decl == error_mark_node)
continue;
if (TREE_CODE (decl) != PARM_DECL)
{
error_at (DECL_SOURCE_LOCATION (decl),
"%qD declared as a non-parameter", decl);
continue;
}
else if (seen_args.contains (decl))
{
error_at (DECL_SOURCE_LOCATION (decl),
"multiple parameters named %qD", decl);
TREE_PURPOSE (parm) = NULL_TREE;
continue;
}
else if (VOID_TYPE_P (TREE_TYPE (decl)))
{
error_at (DECL_SOURCE_LOCATION (decl),
"parameter %qD declared with void type", decl);
TREE_TYPE (decl) = integer_type_node;
DECL_ARG_TYPE (decl) = integer_type_node;
layout_decl (decl, 0);
}
warn_if_shadowing (decl);
}
else
{
decl = build_decl (DECL_SOURCE_LOCATION (fndecl),
PARM_DECL, TREE_VALUE (parm), integer_type_node);
DECL_ARG_TYPE (decl) = TREE_TYPE (decl);
pushdecl (decl);
warn_if_shadowing (decl);
if (flag_isoc99)
pedwarn (DECL_SOURCE_LOCATION (decl),
OPT_Wimplicit_int, "type of %qD defaults to %<int%>",
decl);
else
warning_at (DECL_SOURCE_LOCATION (decl),
OPT_Wmissing_parameter_type,
"type of %qD defaults to %<int%>", decl);
}
TREE_PURPOSE (parm) = decl;
seen_args.add (decl);
}
for (b = current_scope->bindings; b; b = b->prev)
{
parm = b->decl;
if (TREE_CODE (parm) != PARM_DECL)
continue;
if (TREE_TYPE (parm) != error_mark_node
&& !COMPLETE_TYPE_P (TREE_TYPE (parm)))
{
error_at (DECL_SOURCE_LOCATION (parm),
"parameter %qD has incomplete type", parm);
TREE_TYPE (parm) = error_mark_node;
}
if (!seen_args.contains (parm))
{
error_at (DECL_SOURCE_LOCATION (parm),
"declaration for parameter %qD but no such parameter",
parm);
parmids = chainon (parmids, tree_cons (parm, 0, 0));
}
}
DECL_ARGUMENTS (fndecl) = NULL_TREE;
for (parm = parmids; parm; parm = TREE_CHAIN (parm))
if (TREE_PURPOSE (parm))
break;
if (parm && TREE_PURPOSE (parm))
{
last = TREE_PURPOSE (parm);
DECL_ARGUMENTS (fndecl) = last;
for (parm = TREE_CHAIN (parm); parm; parm = TREE_CHAIN (parm))
if (TREE_PURPOSE (parm))
{
DECL_CHAIN (last) = TREE_PURPOSE (parm);
last = TREE_PURPOSE (parm);
}
DECL_CHAIN (last) = NULL_TREE;
}
if (current_function_prototype_arg_types)
{
tree type;
for (parm = DECL_ARGUMENTS (fndecl),
type = current_function_prototype_arg_types;
parm || (type != NULL_TREE
&& TREE_VALUE (type) != error_mark_node
&& TYPE_MAIN_VARIANT (TREE_VALUE (type)) != void_type_node);
parm = DECL_CHAIN (parm), type = TREE_CHAIN (type))
{
if (parm == NULL_TREE
|| type == NULL_TREE
|| (TREE_VALUE (type) != error_mark_node
&& TYPE_MAIN_VARIANT (TREE_VALUE (type)) == void_type_node))
{
if (current_function_prototype_built_in)
warning_at (DECL_SOURCE_LOCATION (fndecl),
0, "number of arguments doesn%'t match "
"built-in prototype");
else
{
error_at (input_location,
"number of arguments doesn%'t match prototype");
error_at (current_function_prototype_locus,
"prototype declaration");
}
break;
}
if (TREE_TYPE (parm) != error_mark_node
&& TREE_VALUE (type) != error_mark_node
&& ((TYPE_ATOMIC (DECL_ARG_TYPE (parm))
!= TYPE_ATOMIC (TREE_VALUE (type)))
|| !comptypes (TYPE_MAIN_VARIANT (DECL_ARG_TYPE (parm)),
TYPE_MAIN_VARIANT (TREE_VALUE (type)))))
{
if ((TYPE_ATOMIC (DECL_ARG_TYPE (parm))
== TYPE_ATOMIC (TREE_VALUE (type)))
&& (TYPE_MAIN_VARIANT (TREE_TYPE (parm))
== TYPE_MAIN_VARIANT (TREE_VALUE (type))))
{
DECL_ARG_TYPE (parm) = TREE_TYPE (parm);
if (targetm.calls.promote_prototypes (TREE_TYPE (current_function_decl))
&& INTEGRAL_TYPE_P (TREE_TYPE (parm))
&& (TYPE_PRECISION (TREE_TYPE (parm))
< TYPE_PRECISION (integer_type_node)))
DECL_ARG_TYPE (parm)
= c_type_promotes_to (TREE_TYPE (parm));
if (current_function_prototype_built_in)
warning_at (DECL_SOURCE_LOCATION (parm),
OPT_Wpedantic, "promoted argument %qD "
"doesn%'t match built-in prototype", parm);
else
{
pedwarn (DECL_SOURCE_LOCATION (parm),
OPT_Wpedantic, "promoted argument %qD "
"doesn%'t match prototype", parm);
pedwarn (current_function_prototype_locus, OPT_Wpedantic,
"prototype declaration");
}
}
else
{
if (current_function_prototype_built_in)
warning_at (DECL_SOURCE_LOCATION (parm),
0, "argument %qD doesn%'t match "
"built-in prototype", parm);
else
{
error_at (DECL_SOURCE_LOCATION (parm),
"argument %qD doesn%'t match prototype", parm);
error_at (current_function_prototype_locus,
"prototype declaration");
}
}
}
}
TYPE_ACTUAL_ARG_TYPES (TREE_TYPE (fndecl)) = NULL_TREE;
}
else
{
tree actual = NULL_TREE, last = NULL_TREE, type;
for (parm = DECL_ARGUMENTS (fndecl); parm; parm = DECL_CHAIN (parm))
{
type = tree_cons (NULL_TREE, DECL_ARG_TYPE (parm), NULL_TREE);
if (last)
TREE_CHAIN (last) = type;
else
actual = type;
last = type;
}
type = tree_cons (NULL_TREE, void_type_node, NULL_TREE);
if (last)
TREE_CHAIN (last) = type;
else
actual = type;
TREE_TYPE (fndecl) = build_variant_type_copy (TREE_TYPE (fndecl));
TYPE_ACTUAL_ARG_TYPES (TREE_TYPE (fndecl)) = actual;
}
}
void
store_parm_decls_from (struct c_arg_info *arg_info)
{
current_function_arg_info = arg_info;
store_parm_decls ();
}
static tree
set_labels_context_r (tree *tp, int *walk_subtrees, void *data)
{
if (TREE_CODE (*tp) == LABEL_EXPR
&& DECL_CONTEXT (LABEL_EXPR_LABEL (*tp)) == NULL_TREE)
{
DECL_CONTEXT (LABEL_EXPR_LABEL (*tp)) = static_cast<tree>(data);
*walk_subtrees = 0;
}
return NULL_TREE;
}
void
store_parm_decls (void)
{
tree fndecl = current_function_decl;
bool proto;
struct c_arg_info *arg_info = current_function_arg_info;
current_function_arg_info = 0;
proto = arg_info->types != 0;
if (proto)
store_parm_decls_newstyle (fndecl, arg_info);
else
store_parm_decls_oldstyle (fndecl, arg_info);
next_is_function_body = true;
gen_aux_info_record (fndecl, 1, 0, proto);
allocate_struct_function (fndecl, false);
if (warn_unused_local_typedefs)
cfun->language = ggc_cleared_alloc<language_function> ();
DECL_SAVED_TREE (fndecl) = push_stmt_list ();
if (arg_info->pending_sizes)
{
walk_tree_without_duplicates (&arg_info->pending_sizes,
set_labels_context_r, fndecl);
add_stmt (arg_info->pending_sizes);
}
}
void
temp_store_parm_decls (tree fndecl, tree parms)
{
push_scope ();
for (tree p = parms; p; p = DECL_CHAIN (p))
{
DECL_CONTEXT (p) = fndecl;
if (DECL_NAME (p))
bind (DECL_NAME (p), p, current_scope,
false, false,
UNKNOWN_LOCATION);
}
}
void
temp_pop_parm_decls (void)
{
struct c_binding *b = current_scope->bindings;
current_scope->bindings = NULL;
for (; b; b = free_binding_and_advance (b))
{
gcc_assert (TREE_CODE (b->decl) == PARM_DECL
|| b->decl == error_mark_node);
gcc_assert (I_SYMBOL_BINDING (b->id) == b);
I_SYMBOL_BINDING (b->id) = b->shadowed;
if (b->shadowed && b->shadowed->u.type)
TREE_TYPE (b->shadowed->decl) = b->shadowed->u.type;
}
pop_scope ();
}

void
finish_function (void)
{
tree fndecl = current_function_decl;
if (c_dialect_objc ())
objc_finish_function ();
if (TREE_CODE (fndecl) == FUNCTION_DECL
&& targetm.calls.promote_prototypes (TREE_TYPE (fndecl)))
{
tree args = DECL_ARGUMENTS (fndecl);
for (; args; args = DECL_CHAIN (args))
{
tree type = TREE_TYPE (args);
if (INTEGRAL_TYPE_P (type)
&& TYPE_PRECISION (type) < TYPE_PRECISION (integer_type_node))
DECL_ARG_TYPE (args) = c_type_promotes_to (type);
}
}
if (DECL_INITIAL (fndecl) && DECL_INITIAL (fndecl) != error_mark_node)
BLOCK_SUPERCONTEXT (DECL_INITIAL (fndecl)) = fndecl;
if (DECL_RESULT (fndecl) && DECL_RESULT (fndecl) != error_mark_node)
DECL_CONTEXT (DECL_RESULT (fndecl)) = fndecl;
if (MAIN_NAME_P (DECL_NAME (fndecl)) && flag_hosted
&& TYPE_MAIN_VARIANT (TREE_TYPE (TREE_TYPE (fndecl)))
== integer_type_node && flag_isoc99)
{
c_finish_return (BUILTINS_LOCATION, integer_zero_node, NULL_TREE);
}
DECL_SAVED_TREE (fndecl) = pop_stmt_list (DECL_SAVED_TREE (fndecl));
finish_fname_decls ();
if (warn_return_type
&& TREE_CODE (TREE_TYPE (TREE_TYPE (fndecl))) != VOID_TYPE
&& !current_function_returns_value && !current_function_returns_null
&& !current_function_returns_abnormally
&& !TREE_THIS_VOLATILE (fndecl)
&& !MAIN_NAME_P (DECL_NAME (fndecl))
&& !C_FUNCTION_IMPLICIT_INT (fndecl)
&& !TREE_PUBLIC (fndecl))
{
warning (OPT_Wreturn_type,
"no return statement in function returning non-void");
TREE_NO_WARNING (fndecl) = 1;
}
if (warn_unused_but_set_parameter)
{
tree decl;
for (decl = DECL_ARGUMENTS (fndecl);
decl;
decl = DECL_CHAIN (decl))
if (TREE_USED (decl)
&& TREE_CODE (decl) == PARM_DECL
&& !DECL_READ_P (decl)
&& DECL_NAME (decl)
&& !DECL_ARTIFICIAL (decl)
&& !TREE_NO_WARNING (decl))
warning_at (DECL_SOURCE_LOCATION (decl),
OPT_Wunused_but_set_parameter,
"parameter %qD set but not used", decl);
}
maybe_warn_unused_local_typedefs ();
if (warn_unused_parameter)
do_warn_unused_parameter (fndecl);
cfun->function_end_locus = input_location;
c_determine_visibility (fndecl);
if (DECL_EXTERNAL (fndecl)
&& DECL_DECLARED_INLINE_P (fndecl)
&& (flag_gnu89_inline
|| lookup_attribute ("gnu_inline", DECL_ATTRIBUTES (fndecl))))
DECL_DISREGARD_INLINE_LIMITS (fndecl) = 1;
if (DECL_INITIAL (fndecl) && DECL_INITIAL (fndecl) != error_mark_node
&& !undef_nested_function)
{
if (!decl_function_context (fndecl))
{
invoke_plugin_callbacks (PLUGIN_PRE_GENERICIZE, fndecl);
c_genericize (fndecl);
if (symtab->global_info_ready)
{
cgraph_node::add_new_function (fndecl, false);
return;
}
cgraph_node::finalize_function (fndecl, false);
}
else
{
(void) cgraph_node::get_create (fndecl);
}
}
if (!decl_function_context (fndecl))
undef_nested_function = false;
if (cfun->language != NULL)
{
ggc_free (cfun->language);
cfun->language = NULL;
}
set_cfun (NULL);
invoke_plugin_callbacks (PLUGIN_FINISH_PARSE_FUNCTION, current_function_decl);
current_function_decl = NULL;
}

tree
check_for_loop_decls (location_t loc, bool turn_off_iso_c99_error)
{
struct c_binding *b;
tree one_decl = NULL_TREE;
int n_decls = 0;
if (!turn_off_iso_c99_error)
{
static bool hint = true;
error_at (loc, "%<for%> loop initial declarations "
"are only allowed in C99 or C11 mode");
if (hint)
{
inform (loc,
"use option -std=c99, -std=gnu99, -std=c11 or -std=gnu11 "
"to compile your code");
hint = false;
}
return NULL_TREE;
}
for (b = current_scope->bindings; b; b = b->prev)
{
tree id = b->id;
tree decl = b->decl;
if (!id)
continue;
switch (TREE_CODE (decl))
{
case VAR_DECL:
{
location_t decl_loc = DECL_SOURCE_LOCATION (decl);
if (TREE_STATIC (decl))
error_at (decl_loc,
"declaration of static variable %qD in %<for%> loop "
"initial declaration", decl);
else if (DECL_EXTERNAL (decl))
error_at (decl_loc,
"declaration of %<extern%> variable %qD in %<for%> loop "
"initial declaration", decl);
}
break;
case RECORD_TYPE:
error_at (loc,
"%<struct %E%> declared in %<for%> loop initial "
"declaration", id);
break;
case UNION_TYPE:
error_at (loc,
"%<union %E%> declared in %<for%> loop initial declaration",
id);
break;
case ENUMERAL_TYPE:
error_at (loc, "%<enum %E%> declared in %<for%> loop "
"initial declaration", id);
break;
default:
error_at (loc, "declaration of non-variable "
"%qD in %<for%> loop initial declaration", decl);
}
n_decls++;
one_decl = decl;
}
return n_decls == 1 ? one_decl : NULL_TREE;
}

void
c_push_function_context (void)
{
struct language_function *p = cfun->language;
if (p == NULL)
cfun->language = p = ggc_cleared_alloc<language_function> ();
p->base.x_stmt_tree = c_stmt_tree;
c_stmt_tree.x_cur_stmt_list = vec_safe_copy (c_stmt_tree.x_cur_stmt_list);
p->x_break_label = c_break_label;
p->x_cont_label = c_cont_label;
p->x_switch_stack = c_switch_stack;
p->arg_info = current_function_arg_info;
p->returns_value = current_function_returns_value;
p->returns_null = current_function_returns_null;
p->returns_abnormally = current_function_returns_abnormally;
p->warn_about_return_type = warn_about_return_type;
push_function_context ();
}
void
c_pop_function_context (void)
{
struct language_function *p;
pop_function_context ();
p = cfun->language;
if (!warn_unused_local_typedefs)
cfun->language = NULL;
if (DECL_STRUCT_FUNCTION (current_function_decl) == 0
&& DECL_SAVED_TREE (current_function_decl) == NULL_TREE)
{
DECL_INITIAL (current_function_decl) = error_mark_node;
DECL_ARGUMENTS (current_function_decl) = NULL_TREE;
}
c_stmt_tree = p->base.x_stmt_tree;
p->base.x_stmt_tree.x_cur_stmt_list = NULL;
c_break_label = p->x_break_label;
c_cont_label = p->x_cont_label;
c_switch_stack = p->x_switch_stack;
current_function_arg_info = p->arg_info;
current_function_returns_value = p->returns_value;
current_function_returns_null = p->returns_null;
current_function_returns_abnormally = p->returns_abnormally;
warn_about_return_type = p->warn_about_return_type;
}
stmt_tree
current_stmt_tree (void)
{
return &c_stmt_tree;
}
tree
identifier_global_value	(tree t)
{
struct c_binding *b;
for (b = I_SYMBOL_BINDING (t); b; b = b->shadowed)
if (B_IN_FILE_SCOPE (b) || B_IN_EXTERNAL_SCOPE (b))
return b->decl;
return NULL_TREE;
}
tree
c_linkage_bindings (tree name)
{
return identifier_global_value (name);
}
void
record_builtin_type (enum rid rid_index, const char *name, tree type)
{
tree id, decl;
if (name == 0)
id = ridpointers[(int) rid_index];
else
id = get_identifier (name);
decl = build_decl (UNKNOWN_LOCATION, TYPE_DECL, id, type);
pushdecl (decl);
if (debug_hooks->type_decl)
debug_hooks->type_decl (decl, false);
}
tree
build_void_list_node (void)
{
tree t = build_tree_list (NULL_TREE, void_type_node);
return t;
}
struct c_parm *
build_c_parm (struct c_declspecs *specs, tree attrs,
struct c_declarator *declarator,
location_t loc)
{
struct c_parm *ret = XOBNEW (&parser_obstack, struct c_parm);
ret->specs = specs;
ret->attrs = attrs;
ret->declarator = declarator;
ret->loc = loc;
return ret;
}
struct c_declarator *
build_attrs_declarator (tree attrs, struct c_declarator *target)
{
struct c_declarator *ret = XOBNEW (&parser_obstack, struct c_declarator);
ret->kind = cdk_attrs;
ret->declarator = target;
ret->u.attrs = attrs;
return ret;
}
struct c_declarator *
build_function_declarator (struct c_arg_info *args,
struct c_declarator *target)
{
struct c_declarator *ret = XOBNEW (&parser_obstack, struct c_declarator);
ret->kind = cdk_function;
ret->declarator = target;
ret->u.arg_info = args;
return ret;
}
struct c_declarator *
build_id_declarator (tree ident)
{
struct c_declarator *ret = XOBNEW (&parser_obstack, struct c_declarator);
ret->kind = cdk_id;
ret->declarator = 0;
ret->u.id = ident;
ret->id_loc = input_location;
return ret;
}
struct c_declarator *
make_pointer_declarator (struct c_declspecs *type_quals_attrs,
struct c_declarator *target)
{
tree attrs;
int quals = 0;
struct c_declarator *itarget = target;
struct c_declarator *ret = XOBNEW (&parser_obstack, struct c_declarator);
if (type_quals_attrs)
{
attrs = type_quals_attrs->attrs;
quals = quals_from_declspecs (type_quals_attrs);
if (attrs != NULL_TREE)
itarget = build_attrs_declarator (attrs, target);
}
ret->kind = cdk_pointer;
ret->declarator = itarget;
ret->u.pointer_quals = quals;
return ret;
}
struct c_declspecs *
build_null_declspecs (void)
{
struct c_declspecs *ret = XOBNEW (&parser_obstack, struct c_declspecs);
memset (ret, 0, sizeof *ret);
ret->align_log = -1;
ret->typespec_word = cts_none;
ret->storage_class = csc_none;
ret->expr_const_operands = true;
ret->typespec_kind = ctsk_none;
ret->address_space = ADDR_SPACE_GENERIC;
return ret;
}
struct c_declspecs *
declspecs_add_addrspace (source_location location,
struct c_declspecs *specs, addr_space_t as)
{
specs->non_sc_seen_p = true;
specs->declspecs_seen_p = true;
if (!ADDR_SPACE_GENERIC_P (specs->address_space)
&& specs->address_space != as)
error ("incompatible address space qualifiers %qs and %qs",
c_addr_space_name (as),
c_addr_space_name (specs->address_space));
else
{
specs->address_space = as;
specs->locations[cdw_address_space] = location;
}
return specs;
}
struct c_declspecs *
declspecs_add_qual (source_location loc,
struct c_declspecs *specs, tree qual)
{
enum rid i;
bool dupe = false;
specs->non_sc_seen_p = true;
specs->declspecs_seen_p = true;
gcc_assert (TREE_CODE (qual) == IDENTIFIER_NODE
&& C_IS_RESERVED_WORD (qual));
i = C_RID_CODE (qual);
location_t prev_loc = UNKNOWN_LOCATION;
switch (i)
{
case RID_CONST:
dupe = specs->const_p;
specs->const_p = true;
prev_loc = specs->locations[cdw_const];
specs->locations[cdw_const] = loc;
break;
case RID_VOLATILE:
dupe = specs->volatile_p;
specs->volatile_p = true;
prev_loc = specs->locations[cdw_volatile];
specs->locations[cdw_volatile] = loc;
break;
case RID_RESTRICT:
dupe = specs->restrict_p;
specs->restrict_p = true;
prev_loc = specs->locations[cdw_restrict];
specs->locations[cdw_restrict] = loc;
break;
case RID_ATOMIC:
dupe = specs->atomic_p;
specs->atomic_p = true;
prev_loc = specs->locations[cdw_atomic];
specs->locations[cdw_atomic] = loc;
break;
default:
gcc_unreachable ();
}
if (dupe)
{
bool warned = pedwarn_c90 (loc, OPT_Wpedantic,
"duplicate %qE declaration specifier", qual);
if (!warned
&& warn_duplicate_decl_specifier
&& prev_loc >= RESERVED_LOCATION_COUNT
&& !from_macro_expansion_at (prev_loc)
&& !from_macro_expansion_at (loc))
warning_at (loc, OPT_Wduplicate_decl_specifier,
"duplicate %qE declaration specifier", qual);
}
return specs;
}
struct c_declspecs *
declspecs_add_type (location_t loc, struct c_declspecs *specs,
struct c_typespec spec)
{
tree type = spec.spec;
specs->non_sc_seen_p = true;
specs->declspecs_seen_p = true;
specs->typespec_kind = spec.kind;
if (TREE_DEPRECATED (type))
specs->deprecated_p = true;
if (TREE_CODE (type) == IDENTIFIER_NODE
&& C_IS_RESERVED_WORD (type)
&& C_RID_CODE (type) != RID_CXX_COMPAT_WARN)
{
enum rid i = C_RID_CODE (type);
if (specs->type)
{
error_at (loc, "two or more data types in declaration specifiers");
return specs;
}
if ((int) i <= (int) RID_LAST_MODIFIER)
{
bool dupe = false;
switch (i)
{
case RID_LONG:
if (specs->long_long_p)
{
error_at (loc, "%<long long long%> is too long for GCC");
break;
}
if (specs->long_p)
{
if (specs->typespec_word == cts_double)
{
error_at (loc,
("both %<long long%> and %<double%> in "
"declaration specifiers"));
break;
}
pedwarn_c90 (loc, OPT_Wlong_long,
"ISO C90 does not support %<long long%>");
specs->long_long_p = 1;
specs->locations[cdw_long_long] = loc;
break;
}
if (specs->short_p)
error_at (loc,
("both %<long%> and %<short%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_auto_type)
error_at (loc,
("both %<long%> and %<__auto_type%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_void)
error_at (loc,
("both %<long%> and %<void%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_int_n)
error_at (loc,
("both %<long%> and %<__int%d%> in "
"declaration specifiers"),
int_n_data[specs->int_n_idx].bitsize);
else if (specs->typespec_word == cts_bool)
error_at (loc,
("both %<long%> and %<_Bool%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_char)
error_at (loc,
("both %<long%> and %<char%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_float)
error_at (loc,
("both %<long%> and %<float%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_floatn_nx)
error_at (loc,
("both %<long%> and %<_Float%d%s%> in "
"declaration specifiers"),
floatn_nx_types[specs->floatn_nx_idx].n,
(floatn_nx_types[specs->floatn_nx_idx].extended
? "x"
: ""));
else if (specs->typespec_word == cts_dfloat32)
error_at (loc,
("both %<long%> and %<_Decimal32%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_dfloat64)
error_at (loc,
("both %<long%> and %<_Decimal64%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_dfloat128)
error_at (loc,
("both %<long%> and %<_Decimal128%> in "
"declaration specifiers"));
else
{
specs->long_p = true;
specs->locations[cdw_long] = loc;
}
break;
case RID_SHORT:
dupe = specs->short_p;
if (specs->long_p)
error_at (loc,
("both %<long%> and %<short%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_auto_type)
error_at (loc,
("both %<short%> and %<__auto_type%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_void)
error_at (loc,
("both %<short%> and %<void%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_int_n)
error_at (loc,
("both %<short%> and %<__int%d%> in "
"declaration specifiers"),
int_n_data[specs->int_n_idx].bitsize);
else if (specs->typespec_word == cts_bool)
error_at (loc,
("both %<short%> and %<_Bool%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_char)
error_at (loc,
("both %<short%> and %<char%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_float)
error_at (loc,
("both %<short%> and %<float%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_double)
error_at (loc,
("both %<short%> and %<double%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_floatn_nx)
error_at (loc,
("both %<short%> and %<_Float%d%s%> in "
"declaration specifiers"),
floatn_nx_types[specs->floatn_nx_idx].n,
(floatn_nx_types[specs->floatn_nx_idx].extended
? "x"
: ""));
else if (specs->typespec_word == cts_dfloat32)
error_at (loc,
("both %<short%> and %<_Decimal32%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_dfloat64)
error_at (loc,
("both %<short%> and %<_Decimal64%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_dfloat128)
error_at (loc,
("both %<short%> and %<_Decimal128%> in "
"declaration specifiers"));
else
{
specs->short_p = true;
specs->locations[cdw_short] = loc;
}
break;
case RID_SIGNED:
dupe = specs->signed_p;
if (specs->unsigned_p)
error_at (loc,
("both %<signed%> and %<unsigned%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_auto_type)
error_at (loc,
("both %<signed%> and %<__auto_type%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_void)
error_at (loc,
("both %<signed%> and %<void%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_bool)
error_at (loc,
("both %<signed%> and %<_Bool%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_float)
error_at (loc,
("both %<signed%> and %<float%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_double)
error_at (loc,
("both %<signed%> and %<double%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_floatn_nx)
error_at (loc,
("both %<signed%> and %<_Float%d%s%> in "
"declaration specifiers"),
floatn_nx_types[specs->floatn_nx_idx].n,
(floatn_nx_types[specs->floatn_nx_idx].extended
? "x"
: ""));
else if (specs->typespec_word == cts_dfloat32)
error_at (loc,
("both %<signed%> and %<_Decimal32%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_dfloat64)
error_at (loc,
("both %<signed%> and %<_Decimal64%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_dfloat128)
error_at (loc,
("both %<signed%> and %<_Decimal128%> in "
"declaration specifiers"));
else
{
specs->signed_p = true;
specs->locations[cdw_signed] = loc;
}
break;
case RID_UNSIGNED:
dupe = specs->unsigned_p;
if (specs->signed_p)
error_at (loc,
("both %<signed%> and %<unsigned%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_auto_type)
error_at (loc,
("both %<unsigned%> and %<__auto_type%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_void)
error_at (loc,
("both %<unsigned%> and %<void%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_bool)
error_at (loc,
("both %<unsigned%> and %<_Bool%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_float)
error_at (loc,
("both %<unsigned%> and %<float%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_double)
error_at (loc,
("both %<unsigned%> and %<double%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_floatn_nx)
error_at (loc,
("both %<unsigned%> and %<_Float%d%s%> in "
"declaration specifiers"),
floatn_nx_types[specs->floatn_nx_idx].n,
(floatn_nx_types[specs->floatn_nx_idx].extended
? "x"
: ""));
else if (specs->typespec_word == cts_dfloat32)
error_at (loc,
("both %<unsigned%> and %<_Decimal32%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_dfloat64)
error_at (loc,
("both %<unsigned%> and %<_Decimal64%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_dfloat128)
error_at (loc,
("both %<unsigned%> and %<_Decimal128%> in "
"declaration specifiers"));
else
{
specs->unsigned_p = true;
specs->locations[cdw_unsigned] = loc;
}
break;
case RID_COMPLEX:
dupe = specs->complex_p;
if (!in_system_header_at (loc))
pedwarn_c90 (loc, OPT_Wpedantic,
"ISO C90 does not support complex types");
if (specs->typespec_word == cts_auto_type)
error_at (loc,
("both %<complex%> and %<__auto_type%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_void)
error_at (loc,
("both %<complex%> and %<void%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_bool)
error_at (loc,
("both %<complex%> and %<_Bool%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_dfloat32)
error_at (loc,
("both %<complex%> and %<_Decimal32%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_dfloat64)
error_at (loc,
("both %<complex%> and %<_Decimal64%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_dfloat128)
error_at (loc,
("both %<complex%> and %<_Decimal128%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_fract)
error_at (loc,
("both %<complex%> and %<_Fract%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_accum)
error_at (loc,
("both %<complex%> and %<_Accum%> in "
"declaration specifiers"));
else if (specs->saturating_p)
error_at (loc,
("both %<complex%> and %<_Sat%> in "
"declaration specifiers"));
else
{
specs->complex_p = true;
specs->locations[cdw_complex] = loc;
}
break;
case RID_SAT:
dupe = specs->saturating_p;
pedwarn (loc, OPT_Wpedantic,
"ISO C does not support saturating types");
if (specs->typespec_word == cts_int_n)
{
error_at (loc,
("both %<_Sat%> and %<__int%d%> in "
"declaration specifiers"),
int_n_data[specs->int_n_idx].bitsize);
}
else if (specs->typespec_word == cts_auto_type)
error_at (loc,
("both %<_Sat%> and %<__auto_type%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_void)
error_at (loc,
("both %<_Sat%> and %<void%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_bool)
error_at (loc,
("both %<_Sat%> and %<_Bool%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_char)
error_at (loc,
("both %<_Sat%> and %<char%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_int)
error_at (loc,
("both %<_Sat%> and %<int%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_float)
error_at (loc,
("both %<_Sat%> and %<float%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_double)
error_at (loc,
("both %<_Sat%> and %<double%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_floatn_nx)
error_at (loc,
("both %<_Sat%> and %<_Float%d%s%> in "
"declaration specifiers"),
floatn_nx_types[specs->floatn_nx_idx].n,
(floatn_nx_types[specs->floatn_nx_idx].extended
? "x"
: ""));
else if (specs->typespec_word == cts_dfloat32)
error_at (loc,
("both %<_Sat%> and %<_Decimal32%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_dfloat64)
error_at (loc,
("both %<_Sat%> and %<_Decimal64%> in "
"declaration specifiers"));
else if (specs->typespec_word == cts_dfloat128)
error_at (loc,
("both %<_Sat%> and %<_Decimal128%> in "
"declaration specifiers"));
else if (specs->complex_p)
error_at (loc,
("both %<_Sat%> and %<complex%> in "
"declaration specifiers"));
else
{
specs->saturating_p = true;
specs->locations[cdw_saturating] = loc;
}
break;
default:
gcc_unreachable ();
}
if (dupe)
error_at (loc, "duplicate %qE", type);
return specs;
}
else
{
if (specs->typespec_word != cts_none)
{
error_at (loc,
"two or more data types in declaration specifiers");
return specs;
}
switch (i)
{
case RID_AUTO_TYPE:
if (specs->long_p)
error_at (loc,
("both %<long%> and %<__auto_type%> in "
"declaration specifiers"));
else if (specs->short_p)
error_at (loc,
("both %<short%> and %<__auto_type%> in "
"declaration specifiers"));
else if (specs->signed_p)
error_at (loc,
("both %<signed%> and %<__auto_type%> in "
"declaration specifiers"));
else if (specs->unsigned_p)
error_at (loc,
("both %<unsigned%> and %<__auto_type%> in "
"declaration specifiers"));
else if (specs->complex_p)
error_at (loc,
("both %<complex%> and %<__auto_type%> in "
"declaration specifiers"));
else if (specs->saturating_p)
error_at (loc,
("both %<_Sat%> and %<__auto_type%> in "
"declaration specifiers"));
else
{
specs->typespec_word = cts_auto_type;
specs->locations[cdw_typespec] = loc;
}
return specs;
case RID_INT_N_0:
case RID_INT_N_1:
case RID_INT_N_2:
case RID_INT_N_3:
specs->int_n_idx = i - RID_INT_N_0;
if (!in_system_header_at (input_location))
pedwarn (loc, OPT_Wpedantic,
"ISO C does not support %<__int%d%> types",
int_n_data[specs->int_n_idx].bitsize);
if (specs->long_p)
error_at (loc,
("both %<__int%d%> and %<long%> in "
"declaration specifiers"),
int_n_data[specs->int_n_idx].bitsize);
else if (specs->saturating_p)
error_at (loc,
("both %<_Sat%> and %<__int%d%> in "
"declaration specifiers"),
int_n_data[specs->int_n_idx].bitsize);
else if (specs->short_p)
error_at (loc,
("both %<__int%d%> and %<short%> in "
"declaration specifiers"),
int_n_data[specs->int_n_idx].bitsize);
else if (! int_n_enabled_p[specs->int_n_idx])
{
specs->typespec_word = cts_int_n;
error_at (loc,
"%<__int%d%> is not supported on this target",
int_n_data[specs->int_n_idx].bitsize);
}
else
{
specs->typespec_word = cts_int_n;
specs->locations[cdw_typespec] = loc;
}
return specs;
case RID_VOID:
if (specs->long_p)
error_at (loc,
("both %<long%> and %<void%> in "
"declaration specifiers"));
else if (specs->short_p)
error_at (loc,
("both %<short%> and %<void%> in "
"declaration specifiers"));
else if (specs->signed_p)
error_at (loc,
("both %<signed%> and %<void%> in "
"declaration specifiers"));
else if (specs->unsigned_p)
error_at (loc,
("both %<unsigned%> and %<void%> in "
"declaration specifiers"));
else if (specs->complex_p)
error_at (loc,
("both %<complex%> and %<void%> in "
"declaration specifiers"));
else if (specs->saturating_p)
error_at (loc,
("both %<_Sat%> and %<void%> in "
"declaration specifiers"));
else
{
specs->typespec_word = cts_void;
specs->locations[cdw_typespec] = loc;
}
return specs;
case RID_BOOL:
if (!in_system_header_at (loc))
pedwarn_c90 (loc, OPT_Wpedantic,
"ISO C90 does not support boolean types");
if (specs->long_p)
error_at (loc,
("both %<long%> and %<_Bool%> in "
"declaration specifiers"));
else if (specs->short_p)
error_at (loc,
("both %<short%> and %<_Bool%> in "
"declaration specifiers"));
else if (specs->signed_p)
error_at (loc,
("both %<signed%> and %<_Bool%> in "
"declaration specifiers"));
else if (specs->unsigned_p)
error_at (loc,
("both %<unsigned%> and %<_Bool%> in "
"declaration specifiers"));
else if (specs->complex_p)
error_at (loc,
("both %<complex%> and %<_Bool%> in "
"declaration specifiers"));
else if (specs->saturating_p)
error_at (loc,
("both %<_Sat%> and %<_Bool%> in "
"declaration specifiers"));
else
{
specs->typespec_word = cts_bool;
specs->locations[cdw_typespec] = loc;
}
return specs;
case RID_CHAR:
if (specs->long_p)
error_at (loc,
("both %<long%> and %<char%> in "
"declaration specifiers"));
else if (specs->short_p)
error_at (loc,
("both %<short%> and %<char%> in "
"declaration specifiers"));
else if (specs->saturating_p)
error_at (loc,
("both %<_Sat%> and %<char%> in "
"declaration specifiers"));
else
{
specs->typespec_word = cts_char;
specs->locations[cdw_typespec] = loc;
}
return specs;
case RID_INT:
if (specs->saturating_p)
error_at (loc,
("both %<_Sat%> and %<int%> in "
"declaration specifiers"));
else
{
specs->typespec_word = cts_int;
specs->locations[cdw_typespec] = loc;
}
return specs;
case RID_FLOAT:
if (specs->long_p)
error_at (loc,
("both %<long%> and %<float%> in "
"declaration specifiers"));
else if (specs->short_p)
error_at (loc,
("both %<short%> and %<float%> in "
"declaration specifiers"));
else if (specs->signed_p)
error_at (loc,
("both %<signed%> and %<float%> in "
"declaration specifiers"));
else if (specs->unsigned_p)
error_at (loc,
("both %<unsigned%> and %<float%> in "
"declaration specifiers"));
else if (specs->saturating_p)
error_at (loc,
("both %<_Sat%> and %<float%> in "
"declaration specifiers"));
else
{
specs->typespec_word = cts_float;
specs->locations[cdw_typespec] = loc;
}
return specs;
case RID_DOUBLE:
if (specs->long_long_p)
error_at (loc,
("both %<long long%> and %<double%> in "
"declaration specifiers"));
else if (specs->short_p)
error_at (loc,
("both %<short%> and %<double%> in "
"declaration specifiers"));
else if (specs->signed_p)
error_at (loc,
("both %<signed%> and %<double%> in "
"declaration specifiers"));
else if (specs->unsigned_p)
error_at (loc,
("both %<unsigned%> and %<double%> in "
"declaration specifiers"));
else if (specs->saturating_p)
error_at (loc,
("both %<_Sat%> and %<double%> in "
"declaration specifiers"));
else
{
specs->typespec_word = cts_double;
specs->locations[cdw_typespec] = loc;
}
return specs;
CASE_RID_FLOATN_NX:
specs->floatn_nx_idx = i - RID_FLOATN_NX_FIRST;
if (!in_system_header_at (input_location))
pedwarn (loc, OPT_Wpedantic,
"ISO C does not support the %<_Float%d%s%> type",
floatn_nx_types[specs->floatn_nx_idx].n,
(floatn_nx_types[specs->floatn_nx_idx].extended
? "x"
: ""));
if (specs->long_p)
error_at (loc,
("both %<long%> and %<_Float%d%s%> in "
"declaration specifiers"),
floatn_nx_types[specs->floatn_nx_idx].n,
(floatn_nx_types[specs->floatn_nx_idx].extended
? "x"
: ""));
else if (specs->short_p)
error_at (loc,
("both %<short%> and %<_Float%d%s%> in "
"declaration specifiers"),
floatn_nx_types[specs->floatn_nx_idx].n,
(floatn_nx_types[specs->floatn_nx_idx].extended
? "x"
: ""));
else if (specs->signed_p)
error_at (loc,
("both %<signed%> and %<_Float%d%s%> in "
"declaration specifiers"),
floatn_nx_types[specs->floatn_nx_idx].n,
(floatn_nx_types[specs->floatn_nx_idx].extended
? "x"
: ""));
else if (specs->unsigned_p)
error_at (loc,
("both %<unsigned%> and %<_Float%d%s%> in "
"declaration specifiers"),
floatn_nx_types[specs->floatn_nx_idx].n,
(floatn_nx_types[specs->floatn_nx_idx].extended
? "x"
: ""));
else if (specs->saturating_p)
error_at (loc,
("both %<_Sat%> and %<_Float%d%s%> in "
"declaration specifiers"),
floatn_nx_types[specs->floatn_nx_idx].n,
(floatn_nx_types[specs->floatn_nx_idx].extended
? "x"
: ""));
else if (FLOATN_NX_TYPE_NODE (specs->floatn_nx_idx) == NULL_TREE)
{
specs->typespec_word = cts_floatn_nx;
error_at (loc,
"%<_Float%d%s%> is not supported on this target",
floatn_nx_types[specs->floatn_nx_idx].n,
(floatn_nx_types[specs->floatn_nx_idx].extended
? "x"
: ""));
}
else
{
specs->typespec_word = cts_floatn_nx;
specs->locations[cdw_typespec] = loc;
}
return specs;
case RID_DFLOAT32:
case RID_DFLOAT64:
case RID_DFLOAT128:
{
const char *str;
if (i == RID_DFLOAT32)
str = "_Decimal32";
else if (i == RID_DFLOAT64)
str = "_Decimal64";
else
str = "_Decimal128";
if (specs->long_long_p)
error_at (loc,
("both %<long long%> and %qs in "
"declaration specifiers"),
str);
if (specs->long_p)
error_at (loc,
("both %<long%> and %qs in "
"declaration specifiers"),
str);
else if (specs->short_p)
error_at (loc,
("both %<short%> and %qs in "
"declaration specifiers"),
str);
else if (specs->signed_p)
error_at (loc,
("both %<signed%> and %qs in "
"declaration specifiers"),
str);
else if (specs->unsigned_p)
error_at (loc,
("both %<unsigned%> and %qs in "
"declaration specifiers"),
str);
else if (specs->complex_p)
error_at (loc,
("both %<complex%> and %qs in "
"declaration specifiers"),
str);
else if (specs->saturating_p)
error_at (loc,
("both %<_Sat%> and %qs in "
"declaration specifiers"),
str);
else if (i == RID_DFLOAT32)
specs->typespec_word = cts_dfloat32;
else if (i == RID_DFLOAT64)
specs->typespec_word = cts_dfloat64;
else
specs->typespec_word = cts_dfloat128;
specs->locations[cdw_typespec] = loc;
}
if (!targetm.decimal_float_supported_p ())
error_at (loc,
("decimal floating point not supported "
"for this target"));
pedwarn (loc, OPT_Wpedantic,
"ISO C does not support decimal floating point");
return specs;
case RID_FRACT:
case RID_ACCUM:
{
const char *str;
if (i == RID_FRACT)
str = "_Fract";
else
str = "_Accum";
if (specs->complex_p)
error_at (loc,
("both %<complex%> and %qs in "
"declaration specifiers"),
str);
else if (i == RID_FRACT)
specs->typespec_word = cts_fract;
else
specs->typespec_word = cts_accum;
specs->locations[cdw_typespec] = loc;
}
if (!targetm.fixed_point_supported_p ())
error_at (loc,
"fixed-point types not supported for this target");
pedwarn (loc, OPT_Wpedantic,
"ISO C does not support fixed-point types");
return specs;
default:
break;
}
}
}
if (specs->type || specs->typespec_word != cts_none
|| specs->long_p || specs->short_p || specs->signed_p
|| specs->unsigned_p || specs->complex_p)
error_at (loc, "two or more data types in declaration specifiers");
else if (TREE_CODE (type) == TYPE_DECL)
{
if (TREE_TYPE (type) == error_mark_node)
; 
else
{
specs->type = TREE_TYPE (type);
specs->decl_attr = DECL_ATTRIBUTES (type);
specs->typedef_p = true;
specs->explicit_signed_p = C_TYPEDEF_EXPLICITLY_SIGNED (type);
specs->locations[cdw_typedef] = loc;
if (warn_cxx_compat
&& I_SYMBOL_BINDING (DECL_NAME (type))->in_struct)
warning_at (loc, OPT_Wc___compat,
"C++ lookup of %qD would return a field, not a type",
type);
if (warn_cxx_compat && struct_parse_info != NULL)
struct_parse_info->typedefs_seen.safe_push (type);
}
}
else if (TREE_CODE (type) == IDENTIFIER_NODE)
{
tree t = lookup_name (type);
if (!t || TREE_CODE (t) != TYPE_DECL)
error_at (loc, "%qE fails to be a typedef or built in type", type);
else if (TREE_TYPE (t) == error_mark_node)
;
else
{
specs->type = TREE_TYPE (t);
specs->locations[cdw_typespec] = loc;
}
}
else
{
if (TREE_CODE (type) != ERROR_MARK && spec.kind == ctsk_typeof)
{
specs->typedef_p = true;
specs->locations[cdw_typedef] = loc;
if (spec.expr)
{
if (specs->expr)
specs->expr = build2 (COMPOUND_EXPR, TREE_TYPE (spec.expr),
specs->expr, spec.expr);
else
specs->expr = spec.expr;
specs->expr_const_operands &= spec.expr_const_operands;
}
}
specs->type = type;
}
return specs;
}
struct c_declspecs *
declspecs_add_scspec (source_location loc,
struct c_declspecs *specs,
tree scspec)
{
enum rid i;
enum c_storage_class n = csc_none;
bool dupe = false;
specs->declspecs_seen_p = true;
gcc_assert (TREE_CODE (scspec) == IDENTIFIER_NODE
&& C_IS_RESERVED_WORD (scspec));
i = C_RID_CODE (scspec);
if (specs->non_sc_seen_p)
warning (OPT_Wold_style_declaration,
"%qE is not at beginning of declaration", scspec);
switch (i)
{
case RID_INLINE:
dupe = false;
specs->inline_p = true;
specs->locations[cdw_inline] = loc;
break;
case RID_NORETURN:
dupe = false;
specs->noreturn_p = true;
specs->locations[cdw_noreturn] = loc;
break;
case RID_THREAD:
dupe = specs->thread_p;
if (specs->storage_class == csc_auto)
error ("%qE used with %<auto%>", scspec);
else if (specs->storage_class == csc_register)
error ("%qE used with %<register%>", scspec);
else if (specs->storage_class == csc_typedef)
error ("%qE used with %<typedef%>", scspec);
else
{
specs->thread_p = true;
specs->thread_gnu_p = (strcmp (IDENTIFIER_POINTER (scspec),
"__thread") == 0);
if (!specs->thread_gnu_p)
{
if (flag_isoc99)
pedwarn_c99 (loc, OPT_Wpedantic,
"ISO C99 does not support %qE", scspec);
else
pedwarn_c99 (loc, OPT_Wpedantic,
"ISO C90 does not support %qE", scspec);
}
specs->locations[cdw_thread] = loc;
}
break;
case RID_AUTO:
n = csc_auto;
break;
case RID_EXTERN:
n = csc_extern;
if (specs->thread_p && specs->thread_gnu_p)
error ("%<__thread%> before %<extern%>");
break;
case RID_REGISTER:
n = csc_register;
break;
case RID_STATIC:
n = csc_static;
if (specs->thread_p && specs->thread_gnu_p)
error ("%<__thread%> before %<static%>");
break;
case RID_TYPEDEF:
n = csc_typedef;
break;
default:
gcc_unreachable ();
}
if (n != csc_none && n == specs->storage_class)
dupe = true;
if (dupe)
{
if (i == RID_THREAD)
error ("duplicate %<_Thread_local%> or %<__thread%>");
else
error ("duplicate %qE", scspec);
}
if (n != csc_none)
{
if (specs->storage_class != csc_none && n != specs->storage_class)
{
error ("multiple storage classes in declaration specifiers");
}
else
{
specs->storage_class = n;
specs->locations[cdw_storage_class] = loc;
if (n != csc_extern && n != csc_static && specs->thread_p)
{
error ("%qs used with %qE",
specs->thread_gnu_p ? "__thread" : "_Thread_local",
scspec);
specs->thread_p = false;
}
}
}
return specs;
}
struct c_declspecs *
declspecs_add_attrs (source_location loc, struct c_declspecs *specs, tree attrs)
{
specs->attrs = chainon (attrs, specs->attrs);
specs->locations[cdw_attributes] = loc;
specs->declspecs_seen_p = true;
return specs;
}
struct c_declspecs *
declspecs_add_alignas (source_location loc,
struct c_declspecs *specs, tree align)
{
int align_log;
specs->alignas_p = true;
specs->locations[cdw_alignas] = loc;
if (align == error_mark_node)
return specs;
align_log = check_user_alignment (align, true);
if (align_log > specs->align_log)
specs->align_log = align_log;
return specs;
}
struct c_declspecs *
finish_declspecs (struct c_declspecs *specs)
{
if (specs->type != NULL_TREE)
{
gcc_assert (!specs->long_p && !specs->long_long_p && !specs->short_p
&& !specs->signed_p && !specs->unsigned_p
&& !specs->complex_p);
if (TREE_CODE (specs->type) == ERROR_MARK)
specs->type = integer_type_node;
return specs;
}
if (specs->typespec_word == cts_none)
{
if (specs->saturating_p)
{
error_at (specs->locations[cdw_saturating],
"%<_Sat%> is used without %<_Fract%> or %<_Accum%>");
if (!targetm.fixed_point_supported_p ())
error_at (specs->locations[cdw_saturating],
"fixed-point types not supported for this target");
specs->typespec_word = cts_fract;
}
else if (specs->long_p || specs->short_p
|| specs->signed_p || specs->unsigned_p)
{
specs->typespec_word = cts_int;
}
else if (specs->complex_p)
{
specs->typespec_word = cts_double;
pedwarn (specs->locations[cdw_complex], OPT_Wpedantic,
"ISO C does not support plain %<complex%> meaning "
"%<double complex%>");
}
else
{
specs->typespec_word = cts_int;
specs->default_int_p = true;
}
}
specs->explicit_signed_p = specs->signed_p;
switch (specs->typespec_word)
{
case cts_auto_type:
gcc_assert (!specs->long_p && !specs->short_p
&& !specs->signed_p && !specs->unsigned_p
&& !specs->complex_p);
break;
case cts_void:
gcc_assert (!specs->long_p && !specs->short_p
&& !specs->signed_p && !specs->unsigned_p
&& !specs->complex_p);
specs->type = void_type_node;
break;
case cts_bool:
gcc_assert (!specs->long_p && !specs->short_p
&& !specs->signed_p && !specs->unsigned_p
&& !specs->complex_p);
specs->type = boolean_type_node;
break;
case cts_char:
gcc_assert (!specs->long_p && !specs->short_p);
gcc_assert (!(specs->signed_p && specs->unsigned_p));
if (specs->signed_p)
specs->type = signed_char_type_node;
else if (specs->unsigned_p)
specs->type = unsigned_char_type_node;
else
specs->type = char_type_node;
if (specs->complex_p)
{
pedwarn (specs->locations[cdw_complex], OPT_Wpedantic,
"ISO C does not support complex integer types");
specs->type = build_complex_type (specs->type);
}
break;
case cts_int_n:
gcc_assert (!specs->long_p && !specs->short_p && !specs->long_long_p);
gcc_assert (!(specs->signed_p && specs->unsigned_p));
if (! int_n_enabled_p[specs->int_n_idx])
specs->type = integer_type_node;
else
specs->type = (specs->unsigned_p
? int_n_trees[specs->int_n_idx].unsigned_type
: int_n_trees[specs->int_n_idx].signed_type);
if (specs->complex_p)
{
pedwarn (specs->locations[cdw_complex], OPT_Wpedantic,
"ISO C does not support complex integer types");
specs->type = build_complex_type (specs->type);
}
break;
case cts_int:
gcc_assert (!(specs->long_p && specs->short_p));
gcc_assert (!(specs->signed_p && specs->unsigned_p));
if (specs->long_long_p)
specs->type = (specs->unsigned_p
? long_long_unsigned_type_node
: long_long_integer_type_node);
else if (specs->long_p)
specs->type = (specs->unsigned_p
? long_unsigned_type_node
: long_integer_type_node);
else if (specs->short_p)
specs->type = (specs->unsigned_p
? short_unsigned_type_node
: short_integer_type_node);
else
specs->type = (specs->unsigned_p
? unsigned_type_node
: integer_type_node);
if (specs->complex_p)
{
pedwarn (specs->locations[cdw_complex], OPT_Wpedantic,
"ISO C does not support complex integer types");
specs->type = build_complex_type (specs->type);
}
break;
case cts_float:
gcc_assert (!specs->long_p && !specs->short_p
&& !specs->signed_p && !specs->unsigned_p);
specs->type = (specs->complex_p
? complex_float_type_node
: float_type_node);
break;
case cts_double:
gcc_assert (!specs->long_long_p && !specs->short_p
&& !specs->signed_p && !specs->unsigned_p);
if (specs->long_p)
{
specs->type = (specs->complex_p
? complex_long_double_type_node
: long_double_type_node);
}
else
{
specs->type = (specs->complex_p
? complex_double_type_node
: double_type_node);
}
break;
case cts_floatn_nx:
gcc_assert (!specs->long_p && !specs->short_p
&& !specs->signed_p && !specs->unsigned_p);
if (FLOATN_NX_TYPE_NODE (specs->floatn_nx_idx) == NULL_TREE)
specs->type = integer_type_node;
else if (specs->complex_p)
specs->type = COMPLEX_FLOATN_NX_TYPE_NODE (specs->floatn_nx_idx);
else
specs->type = FLOATN_NX_TYPE_NODE (specs->floatn_nx_idx);
break;
case cts_dfloat32:
case cts_dfloat64:
case cts_dfloat128:
gcc_assert (!specs->long_p && !specs->long_long_p && !specs->short_p
&& !specs->signed_p && !specs->unsigned_p && !specs->complex_p);
if (specs->typespec_word == cts_dfloat32)
specs->type = dfloat32_type_node;
else if (specs->typespec_word == cts_dfloat64)
specs->type = dfloat64_type_node;
else
specs->type = dfloat128_type_node;
break;
case cts_fract:
gcc_assert (!specs->complex_p);
if (!targetm.fixed_point_supported_p ())
specs->type = integer_type_node;
else if (specs->saturating_p)
{
if (specs->long_long_p)
specs->type = specs->unsigned_p
? sat_unsigned_long_long_fract_type_node
: sat_long_long_fract_type_node;
else if (specs->long_p)
specs->type = specs->unsigned_p
? sat_unsigned_long_fract_type_node
: sat_long_fract_type_node;
else if (specs->short_p)
specs->type = specs->unsigned_p
? sat_unsigned_short_fract_type_node
: sat_short_fract_type_node;
else
specs->type = specs->unsigned_p
? sat_unsigned_fract_type_node
: sat_fract_type_node;
}
else
{
if (specs->long_long_p)
specs->type = specs->unsigned_p
? unsigned_long_long_fract_type_node
: long_long_fract_type_node;
else if (specs->long_p)
specs->type = specs->unsigned_p
? unsigned_long_fract_type_node
: long_fract_type_node;
else if (specs->short_p)
specs->type = specs->unsigned_p
? unsigned_short_fract_type_node
: short_fract_type_node;
else
specs->type = specs->unsigned_p
? unsigned_fract_type_node
: fract_type_node;
}
break;
case cts_accum:
gcc_assert (!specs->complex_p);
if (!targetm.fixed_point_supported_p ())
specs->type = integer_type_node;
else if (specs->saturating_p)
{
if (specs->long_long_p)
specs->type = specs->unsigned_p
? sat_unsigned_long_long_accum_type_node
: sat_long_long_accum_type_node;
else if (specs->long_p)
specs->type = specs->unsigned_p
? sat_unsigned_long_accum_type_node
: sat_long_accum_type_node;
else if (specs->short_p)
specs->type = specs->unsigned_p
? sat_unsigned_short_accum_type_node
: sat_short_accum_type_node;
else
specs->type = specs->unsigned_p
? sat_unsigned_accum_type_node
: sat_accum_type_node;
}
else
{
if (specs->long_long_p)
specs->type = specs->unsigned_p
? unsigned_long_long_accum_type_node
: long_long_accum_type_node;
else if (specs->long_p)
specs->type = specs->unsigned_p
? unsigned_long_accum_type_node
: long_accum_type_node;
else if (specs->short_p)
specs->type = specs->unsigned_p
? unsigned_short_accum_type_node
: short_accum_type_node;
else
specs->type = specs->unsigned_p
? unsigned_accum_type_node
: accum_type_node;
}
break;
default:
gcc_unreachable ();
}
return specs;
}
static void
c_write_global_declarations_1 (tree globals)
{
tree decl;
bool reconsider;
for (decl = globals; decl; decl = DECL_CHAIN (decl))
{
if (TREE_CODE (decl) == FUNCTION_DECL
&& DECL_INITIAL (decl) == NULL_TREE
&& DECL_EXTERNAL (decl)
&& !TREE_PUBLIC (decl))
{
if (C_DECL_USED (decl))
{
pedwarn (input_location, 0, "%q+F used but never defined", decl);
TREE_NO_WARNING (decl) = 1;
}
else if (warn_unused_function
&& ! DECL_ARTIFICIAL (decl)
&& ! TREE_NO_WARNING (decl))
{
warning (OPT_Wunused_function,
"%q+F declared %<static%> but never defined", decl);
TREE_NO_WARNING (decl) = 1;
}
}
wrapup_global_declaration_1 (decl);
}
do
{
reconsider = false;
for (decl = globals; decl; decl = DECL_CHAIN (decl))
reconsider |= wrapup_global_declaration_2 (decl);
}
while (reconsider);
}
static void
collect_source_ref_cb (tree decl)
{
if (!DECL_IS_BUILTIN (decl))
collect_source_ref (LOCATION_FILE (decl_sloc (decl, false)));
}
static GTY(()) tree ext_block;
static void
collect_all_refs (const char *source_file)
{
tree t;
unsigned i;
FOR_EACH_VEC_ELT (*all_translation_units, i, t)
collect_ada_nodes (BLOCK_VARS (DECL_INITIAL (t)), source_file);
collect_ada_nodes (BLOCK_VARS (ext_block), source_file);
}
static void
for_each_global_decl (void (*callback) (tree decl))
{
tree t;
tree decls;
tree decl;
unsigned i;
FOR_EACH_VEC_ELT (*all_translation_units, i, t)
{ 
decls = DECL_INITIAL (t);
for (decl = BLOCK_VARS (decls); decl; decl = TREE_CHAIN (decl))
callback (decl);
}
for (decl = BLOCK_VARS (ext_block); decl; decl = TREE_CHAIN (decl))
callback (decl);
}
void
c_parse_final_cleanups (void)
{
tree t;
unsigned i;
if (pch_file)
return;
timevar_stop (TV_PHASE_PARSING);
timevar_start (TV_PHASE_DEFERRED);
if (c_dialect_objc ())
objc_write_global_declarations ();
ext_block = pop_scope ();
external_scope = 0;
gcc_assert (!current_scope);
if (flag_dump_ada_spec || flag_dump_ada_spec_slim)
{
if (flag_dump_ada_spec_slim)
collect_source_ref (main_input_filename);
else
for_each_global_decl (collect_source_ref_cb);
dump_ada_specs (collect_all_refs, NULL);
}
FOR_EACH_VEC_ELT (*all_translation_units, i, t)
c_write_global_declarations_1 (BLOCK_VARS (DECL_INITIAL (t)));
c_write_global_declarations_1 (BLOCK_VARS (ext_block));
timevar_stop (TV_PHASE_DEFERRED);
timevar_start (TV_PHASE_PARSING);
ext_block = NULL;
}
void
c_register_addr_space (const char *word, addr_space_t as)
{
int rid = RID_FIRST_ADDR_SPACE + as;
tree id;
if (c_dialect_objc () || flag_no_asm)
return;
id = get_identifier (word);
C_SET_RID_CODE (id, rid);
C_IS_RESERVED_WORD (id) = 1;
ridpointers [rid] = id;
}
tree
c_omp_reduction_id (enum tree_code reduction_code, tree reduction_id)
{
const char *p = NULL;
switch (reduction_code)
{
case PLUS_EXPR: p = "+"; break;
case MULT_EXPR: p = "*"; break;
case MINUS_EXPR: p = "-"; break;
case BIT_AND_EXPR: p = "&"; break;
case BIT_XOR_EXPR: p = "^"; break;
case BIT_IOR_EXPR: p = "|"; break;
case TRUTH_ANDIF_EXPR: p = "&&"; break;
case TRUTH_ORIF_EXPR: p = "||"; break;
case MIN_EXPR: p = "min"; break;
case MAX_EXPR: p = "max"; break;
default:
break;
}
if (p == NULL)
{
if (TREE_CODE (reduction_id) != IDENTIFIER_NODE)
return error_mark_node;
p = IDENTIFIER_POINTER (reduction_id);
}
const char prefix[] = "omp declare reduction ";
size_t lenp = sizeof (prefix);
size_t len = strlen (p);
char *name = XALLOCAVEC (char, lenp + len);
memcpy (name, prefix, lenp - 1);
memcpy (name + lenp - 1, p, len + 1);
return get_identifier (name);
}
tree
c_omp_reduction_decl (tree reduction_id)
{
struct c_binding *b = I_SYMBOL_BINDING (reduction_id);
if (b != NULL && B_IN_CURRENT_SCOPE (b))
return b->decl;
tree decl = build_decl (BUILTINS_LOCATION, VAR_DECL,
reduction_id, integer_type_node);
DECL_ARTIFICIAL (decl) = 1;
DECL_EXTERNAL (decl) = 1;
TREE_STATIC (decl) = 1;
TREE_PUBLIC (decl) = 0;
bind (reduction_id, decl, current_scope, true, false, BUILTINS_LOCATION);
return decl;
}
tree
c_omp_reduction_lookup (tree reduction_id, tree type)
{
struct c_binding *b = I_SYMBOL_BINDING (reduction_id);
while (b)
{
tree t;
for (t = DECL_INITIAL (b->decl); t; t = TREE_CHAIN (t))
if (comptypes (TREE_PURPOSE (t), type))
return TREE_VALUE (t);
b = b->shadowed;
}
return error_mark_node;
}
tree
c_check_omp_declare_reduction_r (tree *tp, int *, void *data)
{
tree *vars = (tree *) data;
if (SSA_VAR_P (*tp)
&& !DECL_ARTIFICIAL (*tp)
&& *tp != vars[0]
&& *tp != vars[1])
{
location_t loc = DECL_SOURCE_LOCATION (vars[0]);
if (strcmp (IDENTIFIER_POINTER (DECL_NAME (vars[0])), "omp_out") == 0)
error_at (loc, "%<#pragma omp declare reduction%> combiner refers to "
"variable %qD which is not %<omp_out%> nor %<omp_in%>",
*tp);
else
error_at (loc, "%<#pragma omp declare reduction%> initializer refers "
"to variable %qD which is not %<omp_priv%> nor "
"%<omp_orig%>",
*tp);
return *tp;
}
return NULL_TREE;
}
#include "gt-c-c-decl.h"

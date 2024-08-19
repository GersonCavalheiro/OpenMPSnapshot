#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "target.h"
#include "function.h"		
#include "c-common.h"
#include "memmodel.h"
#include "tm_p.h"		
#include "stringpool.h"
#include "cgraph.h"
#include "diagnostic.h"
#include "attribs.h"
#include "varasm.h"
#include "c-pragma.h"
#include "opts.h"
#include "plugin.h"
#define GCC_BAD(gmsgid) \
do { warning (OPT_Wpragmas, gmsgid); return; } while (0)
#define GCC_BAD2(gmsgid, arg) \
do { warning (OPT_Wpragmas, gmsgid, arg); return; } while (0)
struct GTY(()) align_stack {
int		       alignment;
tree		       id;
struct align_stack * prev;
};
static GTY(()) struct align_stack * alignment_stack;
static void handle_pragma_pack (cpp_reader *);
static int default_alignment;
#define SET_GLOBAL_ALIGNMENT(ALIGN) (maximum_field_alignment = *(alignment_stack == NULL \
? &default_alignment \
: &alignment_stack->alignment) = (ALIGN))
static void push_alignment (int, tree);
static void pop_alignment (tree);
static void
push_alignment (int alignment, tree id)
{
align_stack * entry = ggc_alloc<align_stack> ();
entry->alignment  = alignment;
entry->id	    = id;
entry->prev	    = alignment_stack;
if (alignment_stack == NULL)
default_alignment = maximum_field_alignment;
alignment_stack = entry;
maximum_field_alignment = alignment;
}
static void
pop_alignment (tree id)
{
align_stack * entry;
if (alignment_stack == NULL)
GCC_BAD ("#pragma pack (pop) encountered without matching #pragma pack (push)");
if (id)
{
for (entry = alignment_stack; entry; entry = entry->prev)
if (entry->id == id)
{
alignment_stack = entry;
break;
}
if (entry == NULL)
warning (OPT_Wpragmas, "\
#pragma pack(pop, %E) encountered without matching #pragma pack(push, %E)"
, id, id);
}
entry = alignment_stack->prev;
maximum_field_alignment = entry ? entry->alignment : default_alignment;
alignment_stack = entry;
}
static void
handle_pragma_pack (cpp_reader * ARG_UNUSED (dummy))
{
tree x, id = 0;
int align = -1;
enum cpp_ttype token;
enum { set, push, pop } action;
if (pragma_lex (&x) != CPP_OPEN_PAREN)
GCC_BAD ("missing %<(%> after %<#pragma pack%> - ignored");
token = pragma_lex (&x);
if (token == CPP_CLOSE_PAREN)
{
action = set;
align = initial_max_fld_align;
}
else if (token == CPP_NUMBER)
{
if (TREE_CODE (x) != INTEGER_CST)
GCC_BAD ("invalid constant in %<#pragma pack%> - ignored");
align = TREE_INT_CST_LOW (x);
action = set;
if (pragma_lex (&x) != CPP_CLOSE_PAREN)
GCC_BAD ("malformed %<#pragma pack%> - ignored");
}
else if (token == CPP_NAME)
{
#define GCC_BAD_ACTION do { if (action != pop) \
GCC_BAD ("malformed %<#pragma pack(push[, id][, <n>])%> - ignored"); \
else \
GCC_BAD ("malformed %<#pragma pack(pop[, id])%> - ignored"); \
} while (0)
const char *op = IDENTIFIER_POINTER (x);
if (!strcmp (op, "push"))
action = push;
else if (!strcmp (op, "pop"))
action = pop;
else
GCC_BAD2 ("unknown action %qE for %<#pragma pack%> - ignored", x);
while ((token = pragma_lex (&x)) == CPP_COMMA)
{
token = pragma_lex (&x);
if (token == CPP_NAME && id == 0)
{
id = x;
}
else if (token == CPP_NUMBER && action == push && align == -1)
{
if (TREE_CODE (x) != INTEGER_CST)
GCC_BAD ("invalid constant in %<#pragma pack%> - ignored");
align = TREE_INT_CST_LOW (x);
if (align == -1)
action = set;
}
else
GCC_BAD_ACTION;
}
if (token != CPP_CLOSE_PAREN)
GCC_BAD_ACTION;
#undef GCC_BAD_ACTION
}
else
GCC_BAD ("malformed %<#pragma pack%> - ignored");
if (pragma_lex (&x) != CPP_EOF)
warning (OPT_Wpragmas, "junk at end of %<#pragma pack%>");
if (flag_pack_struct)
GCC_BAD ("#pragma pack has no effect with -fpack-struct - ignored");
if (action != pop)
switch (align)
{
case 0:
case 1:
case 2:
case 4:
case 8:
case 16:
align *= BITS_PER_UNIT;
break;
case -1:
if (action == push)
{
align = maximum_field_alignment;
break;
}
default:
GCC_BAD2 ("alignment must be a small power of two, not %d", align);
}
switch (action)
{
case set:   SET_GLOBAL_ALIGNMENT (align);  break;
case push:  push_alignment (align, id);    break;
case pop:   pop_alignment (id);	       break;
}
}
struct GTY(()) pending_weak
{
tree name;
tree value;
};
static GTY(()) vec<pending_weak, va_gc> *pending_weaks;
static void apply_pragma_weak (tree, tree);
static void handle_pragma_weak (cpp_reader *);
static void
apply_pragma_weak (tree decl, tree value)
{
if (value)
{
value = build_string (IDENTIFIER_LENGTH (value),
IDENTIFIER_POINTER (value));
decl_attributes (&decl, build_tree_list (get_identifier ("alias"),
build_tree_list (NULL, value)),
0);
}
if (SUPPORTS_WEAK && DECL_EXTERNAL (decl) && TREE_USED (decl)
&& !DECL_WEAK (decl) 
&& DECL_ASSEMBLER_NAME_SET_P (decl)
&& TREE_SYMBOL_REFERENCED (DECL_ASSEMBLER_NAME (decl)))
warning (OPT_Wpragmas, "applying #pragma weak %q+D after first use "
"results in unspecified behavior", decl);
declare_weak (decl);
}
void
maybe_apply_pragma_weak (tree decl)
{
tree id;
int i;
pending_weak *pe;
if (vec_safe_is_empty (pending_weaks))
return;
if (!DECL_EXTERNAL (decl) && !TREE_PUBLIC (decl))
return;
if (!VAR_OR_FUNCTION_DECL_P (decl))
return;
if (DECL_ASSEMBLER_NAME_SET_P (decl))
id = DECL_ASSEMBLER_NAME (decl);
else
{
id = DECL_ASSEMBLER_NAME (decl);
SET_DECL_ASSEMBLER_NAME (decl, NULL_TREE);
}
FOR_EACH_VEC_ELT (*pending_weaks, i, pe)
if (id == pe->name)
{
apply_pragma_weak (decl, pe->value);
pending_weaks->unordered_remove (i);
break;
}
}
void
maybe_apply_pending_pragma_weaks (void)
{
tree alias_id, id, decl;
int i;
pending_weak *pe;
symtab_node *target;
if (vec_safe_is_empty (pending_weaks))
return;
FOR_EACH_VEC_ELT (*pending_weaks, i, pe)
{
alias_id = pe->name;
id = pe->value;
if (id == NULL)
continue;
target = symtab_node::get_for_asmname (id);
decl = build_decl (UNKNOWN_LOCATION,
target ? TREE_CODE (target->decl) : FUNCTION_DECL,
alias_id, default_function_type);
DECL_ARTIFICIAL (decl) = 1;
TREE_PUBLIC (decl) = 1;
DECL_WEAK (decl) = 1;
if (VAR_P (decl))
TREE_STATIC (decl) = 1;
if (!target)
{
error ("%q+D aliased to undefined symbol %qE",
decl, id);
continue;
}
assemble_alias (decl, id);
}
}
static void
handle_pragma_weak (cpp_reader * ARG_UNUSED (dummy))
{
tree name, value, x, decl;
enum cpp_ttype t;
value = 0;
if (pragma_lex (&name) != CPP_NAME)
GCC_BAD ("malformed #pragma weak, ignored");
t = pragma_lex (&x);
if (t == CPP_EQ)
{
if (pragma_lex (&value) != CPP_NAME)
GCC_BAD ("malformed #pragma weak, ignored");
t = pragma_lex (&x);
}
if (t != CPP_EOF)
warning (OPT_Wpragmas, "junk at end of %<#pragma weak%>");
decl = identifier_global_value (name);
if (decl && DECL_P (decl))
{
if (!VAR_OR_FUNCTION_DECL_P (decl))
GCC_BAD2 ("%<#pragma weak%> declaration of %q+D not allowed,"
" ignored", decl);
apply_pragma_weak (decl, value);
if (value)
{
DECL_EXTERNAL (decl) = 0;
if (VAR_P (decl))
TREE_STATIC (decl) = 1;
assemble_alias (decl, value);
}
}
else
{
pending_weak pe = {name, value};
vec_safe_push (pending_weaks, pe);
}
}
static enum scalar_storage_order_kind global_sso;
void
maybe_apply_pragma_scalar_storage_order (tree type)
{
if (global_sso == SSO_NATIVE)
return;
gcc_assert (RECORD_OR_UNION_TYPE_P (type));
if (lookup_attribute ("scalar_storage_order", TYPE_ATTRIBUTES (type)))
return;
if (global_sso == SSO_BIG_ENDIAN)
TYPE_REVERSE_STORAGE_ORDER (type) = !BYTES_BIG_ENDIAN;
else if (global_sso == SSO_LITTLE_ENDIAN)
TYPE_REVERSE_STORAGE_ORDER (type) = BYTES_BIG_ENDIAN;
else
gcc_unreachable ();
}
static void
handle_pragma_scalar_storage_order (cpp_reader *ARG_UNUSED(dummy))
{
const char *kind_string;
enum cpp_ttype token;
tree x;
if (BYTES_BIG_ENDIAN != WORDS_BIG_ENDIAN)
{
error ("scalar_storage_order is not supported because endianness "
"is not uniform");
return;
}
if (c_dialect_cxx ())
{
if (warn_unknown_pragmas > in_system_header_at (input_location))
warning (OPT_Wunknown_pragmas,
"%<#pragma scalar_storage_order%> is not supported for C++");
return;
}
token = pragma_lex (&x);
if (token != CPP_NAME)
GCC_BAD ("missing [big-endian|little-endian|default] after %<#pragma scalar_storage_order%>");
kind_string = IDENTIFIER_POINTER (x);
if (strcmp (kind_string, "default") == 0)
global_sso = default_sso;
else if (strcmp (kind_string, "big") == 0)
global_sso = SSO_BIG_ENDIAN;
else if (strcmp (kind_string, "little") == 0)
global_sso = SSO_LITTLE_ENDIAN;
else
GCC_BAD ("expected [big-endian|little-endian|default] after %<#pragma scalar_storage_order%>");
}
struct GTY(()) pending_redefinition {
tree oldname;
tree newname;
};
static GTY(()) vec<pending_redefinition, va_gc> *pending_redefine_extname;
static void handle_pragma_redefine_extname (cpp_reader *);
static void
handle_pragma_redefine_extname (cpp_reader * ARG_UNUSED (dummy))
{
tree oldname, newname, decls, x;
enum cpp_ttype t;
bool found;
if (pragma_lex (&oldname) != CPP_NAME)
GCC_BAD ("malformed #pragma redefine_extname, ignored");
if (pragma_lex (&newname) != CPP_NAME)
GCC_BAD ("malformed #pragma redefine_extname, ignored");
t = pragma_lex (&x);
if (t != CPP_EOF)
warning (OPT_Wpragmas, "junk at end of %<#pragma redefine_extname%>");
found = false;
for (decls = c_linkage_bindings (oldname);
decls; )
{
tree decl;
if (TREE_CODE (decls) == TREE_LIST)
{
decl = TREE_VALUE (decls);
decls = TREE_CHAIN (decls);
}
else
{
decl = decls;
decls = NULL_TREE;
}
if ((TREE_PUBLIC (decl) || DECL_EXTERNAL (decl))
&& VAR_OR_FUNCTION_DECL_P (decl))
{
found = true;
if (DECL_ASSEMBLER_NAME_SET_P (decl))
{
const char *name = IDENTIFIER_POINTER (DECL_ASSEMBLER_NAME (decl));
name = targetm.strip_name_encoding (name);
if (!id_equal (newname, name))
warning (OPT_Wpragmas, "#pragma redefine_extname ignored due to "
"conflict with previous rename");
}
else
symtab->change_decl_assembler_name (decl, newname);
}
}
if (!found)
add_to_renaming_pragma_list (oldname, newname);
}
void
add_to_renaming_pragma_list (tree oldname, tree newname)
{
unsigned ix;
pending_redefinition *p;
FOR_EACH_VEC_SAFE_ELT (pending_redefine_extname, ix, p)
if (oldname == p->oldname)
{
if (p->newname != newname)
warning (OPT_Wpragmas, "#pragma redefine_extname ignored due to "
"conflict with previous #pragma redefine_extname");
return;
}
pending_redefinition e = {oldname, newname};
vec_safe_push (pending_redefine_extname, e);
}
GTY(()) tree pragma_extern_prefix;
tree
maybe_apply_renaming_pragma (tree decl, tree asmname)
{
unsigned ix;
pending_redefinition *p;
if (!VAR_OR_FUNCTION_DECL_P (decl)
|| (!TREE_PUBLIC (decl) && !DECL_EXTERNAL (decl))
|| !has_c_linkage (decl))
return asmname;
if (DECL_ASSEMBLER_NAME_SET_P (decl))
{
const char *oldname = IDENTIFIER_POINTER (DECL_ASSEMBLER_NAME (decl));
oldname = targetm.strip_name_encoding (oldname);
if (asmname && strcmp (TREE_STRING_POINTER (asmname), oldname))
warning (OPT_Wpragmas, "asm declaration ignored due to "
"conflict with previous rename");
FOR_EACH_VEC_SAFE_ELT (pending_redefine_extname, ix, p)
if (DECL_NAME (decl) == p->oldname)
{
if (!id_equal (p->newname, oldname))
warning (OPT_Wpragmas, "#pragma redefine_extname ignored due to "
"conflict with previous rename");
pending_redefine_extname->unordered_remove (ix);
break;
}
return NULL_TREE;
}
FOR_EACH_VEC_SAFE_ELT (pending_redefine_extname, ix, p)
if (DECL_NAME (decl) == p->oldname)
{
tree newname = p->newname;
pending_redefine_extname->unordered_remove (ix);
if (asmname)
{
if (strcmp (TREE_STRING_POINTER (asmname),
IDENTIFIER_POINTER (newname)) != 0)
warning (OPT_Wpragmas, "#pragma redefine_extname ignored due to "
"conflict with __asm__ declaration");
return asmname;
}
return build_string (IDENTIFIER_LENGTH (newname),
IDENTIFIER_POINTER (newname));
}
if (asmname)
return asmname;
if (pragma_extern_prefix)
{
const char *prefix = TREE_STRING_POINTER (pragma_extern_prefix);
size_t plen = TREE_STRING_LENGTH (pragma_extern_prefix) - 1;
const char *id = IDENTIFIER_POINTER (DECL_NAME (decl));
size_t ilen = IDENTIFIER_LENGTH (DECL_NAME (decl));
char *newname = (char *) alloca (plen + ilen + 1);
memcpy (newname,        prefix, plen);
memcpy (newname + plen, id, ilen + 1);
return build_string (plen + ilen, newname);
}
return NULL_TREE;
}
static void handle_pragma_visibility (cpp_reader *);
static vec<int> visstack;
void
push_visibility (const char *str, int kind)
{
visstack.safe_push (((int) default_visibility) | (kind << 8));
if (!strcmp (str, "default"))
default_visibility = VISIBILITY_DEFAULT;
else if (!strcmp (str, "internal"))
default_visibility = VISIBILITY_INTERNAL;
else if (!strcmp (str, "hidden"))
default_visibility = VISIBILITY_HIDDEN;
else if (!strcmp (str, "protected"))
default_visibility = VISIBILITY_PROTECTED;
else
GCC_BAD ("#pragma GCC visibility push() must specify default, internal, hidden or protected");
visibility_options.inpragma = 1;
}
bool
pop_visibility (int kind)
{
if (!visstack.length ())
return false;
if ((visstack.last () >> 8) != kind)
return false;
default_visibility
= (enum symbol_visibility) (visstack.pop () & 0xff);
visibility_options.inpragma
= visstack.length () != 0;
return true;
}
static void
handle_pragma_visibility (cpp_reader *dummy ATTRIBUTE_UNUSED)
{
tree x;
enum cpp_ttype token;
enum { bad, push, pop } action = bad;
token = pragma_lex (&x);
if (token == CPP_NAME)
{
const char *op = IDENTIFIER_POINTER (x);
if (!strcmp (op, "push"))
action = push;
else if (!strcmp (op, "pop"))
action = pop;
}
if (bad == action)
GCC_BAD ("#pragma GCC visibility must be followed by push or pop");
else
{
if (pop == action)
{
if (! pop_visibility (0))
GCC_BAD ("no matching push for %<#pragma GCC visibility pop%>");
}
else
{
if (pragma_lex (&x) != CPP_OPEN_PAREN)
GCC_BAD ("missing %<(%> after %<#pragma GCC visibility push%> - ignored");
token = pragma_lex (&x);
if (token != CPP_NAME)
GCC_BAD ("malformed #pragma GCC visibility push");
else
push_visibility (IDENTIFIER_POINTER (x), 0);
if (pragma_lex (&x) != CPP_CLOSE_PAREN)
GCC_BAD ("missing %<(%> after %<#pragma GCC visibility push%> - ignored");
}
}
if (pragma_lex (&x) != CPP_EOF)
warning (OPT_Wpragmas, "junk at end of %<#pragma GCC visibility%>");
}
static void
handle_pragma_diagnostic(cpp_reader *ARG_UNUSED(dummy))
{
tree x;
location_t loc;
enum cpp_ttype token = pragma_lex (&x, &loc);
if (token != CPP_NAME)
{
warning_at (loc, OPT_Wpragmas,
"missing [error|warning|ignored|push|pop]"
" after %<#pragma GCC diagnostic%>");
return;
}
diagnostic_t kind;
const char *kind_string = IDENTIFIER_POINTER (x);
if (strcmp (kind_string, "error") == 0)
kind = DK_ERROR;
else if (strcmp (kind_string, "warning") == 0)
kind = DK_WARNING;
else if (strcmp (kind_string, "ignored") == 0)
kind = DK_IGNORED;
else if (strcmp (kind_string, "push") == 0)
{
diagnostic_push_diagnostics (global_dc, input_location);
return;
}
else if (strcmp (kind_string, "pop") == 0)
{
diagnostic_pop_diagnostics (global_dc, input_location);
return;
}
else
{
warning_at (loc, OPT_Wpragmas,
"expected [error|warning|ignored|push|pop]"
" after %<#pragma GCC diagnostic%>");
return;
}
token = pragma_lex (&x, &loc);
if (token != CPP_STRING)
{
warning_at (loc, OPT_Wpragmas,
"missing option after %<#pragma GCC diagnostic%> kind");
return;
}
const char *option_string = TREE_STRING_POINTER (x);
unsigned int lang_mask = c_common_option_lang_mask () | CL_COMMON;
unsigned int option_index = find_opt (option_string + 1, lang_mask);
if (option_index == OPT_SPECIAL_unknown)
{
warning_at (loc, OPT_Wpragmas,
"unknown option after %<#pragma GCC diagnostic%> kind");
return;
}
else if (!(cl_options[option_index].flags & CL_WARNING))
{
warning_at (loc, OPT_Wpragmas,
"%qs is not an option that controls warnings", option_string);
return;
}
else if (!(cl_options[option_index].flags & lang_mask))
{
char *ok_langs = write_langs (cl_options[option_index].flags);
char *bad_lang = write_langs (c_common_option_lang_mask ());
warning_at (loc, OPT_Wpragmas,
"option %qs is valid for %s but not for %s",
option_string, ok_langs, bad_lang);
free (ok_langs);
free (bad_lang);
return;
}
struct cl_option_handlers handlers;
set_default_handlers (&handlers, NULL);
const char *arg = NULL;
if (cl_options[option_index].flags & CL_JOINED)
arg = option_string + 1 + cl_options[option_index].opt_len;
control_warning_option (option_index, (int) kind,
arg, kind != DK_IGNORED,
input_location, lang_mask, &handlers,
&global_options, &global_options_set,
global_dc);
}
static void
handle_pragma_target(cpp_reader *ARG_UNUSED(dummy))
{
enum cpp_ttype token;
tree x;
bool close_paren_needed_p = false;
if (cfun)
{
error ("#pragma GCC option is not allowed inside functions");
return;
}
token = pragma_lex (&x);
if (token == CPP_OPEN_PAREN)
{
close_paren_needed_p = true;
token = pragma_lex (&x);
}
if (token != CPP_STRING)
{
GCC_BAD ("%<#pragma GCC option%> is not a string");
return;
}
else
{
tree args = NULL_TREE;
do
{
if (TREE_STRING_LENGTH (x) > 0)
args = tree_cons (NULL_TREE, x, args);
token = pragma_lex (&x);
while (token == CPP_COMMA)
token = pragma_lex (&x);
}
while (token == CPP_STRING);
if (close_paren_needed_p)
{
if (token == CPP_CLOSE_PAREN)
token = pragma_lex (&x);
else
GCC_BAD ("%<#pragma GCC target (string [,string]...)%> does "
"not have a final %<)%>");
}
if (token != CPP_EOF)
{
error ("#pragma GCC target string... is badly formed");
return;
}
args = nreverse (args);
if (targetm.target_option.pragma_parse (args, NULL_TREE))
current_target_pragma = chainon (current_target_pragma, args);
}
}
static void
handle_pragma_optimize (cpp_reader *ARG_UNUSED(dummy))
{
enum cpp_ttype token;
tree x;
bool close_paren_needed_p = false;
tree optimization_previous_node = optimization_current_node;
if (cfun)
{
error ("#pragma GCC optimize is not allowed inside functions");
return;
}
token = pragma_lex (&x);
if (token == CPP_OPEN_PAREN)
{
close_paren_needed_p = true;
token = pragma_lex (&x);
}
if (token != CPP_STRING && token != CPP_NUMBER)
{
GCC_BAD ("%<#pragma GCC optimize%> is not a string or number");
return;
}
else
{
tree args = NULL_TREE;
do
{
if (token != CPP_STRING || TREE_STRING_LENGTH (x) > 0)
args = tree_cons (NULL_TREE, x, args);
token = pragma_lex (&x);
while (token == CPP_COMMA)
token = pragma_lex (&x);
}
while (token == CPP_STRING || token == CPP_NUMBER);
if (close_paren_needed_p)
{
if (token == CPP_CLOSE_PAREN)
token = pragma_lex (&x);
else
GCC_BAD ("%<#pragma GCC optimize (string [,string]...)%> does "
"not have a final %<)%>");
}
if (token != CPP_EOF)
{
error ("#pragma GCC optimize string... is badly formed");
return;
}
args = nreverse (args);
parse_optimize_options (args, false);
current_optimize_pragma = chainon (current_optimize_pragma, args);
optimization_current_node = build_optimization_node (&global_options);
c_cpp_builtins_optimize_pragma (parse_in,
optimization_previous_node,
optimization_current_node);
}
}
struct GTY(()) opt_stack {
struct opt_stack *prev;
tree target_binary;
tree target_strings;
tree optimize_binary;
tree optimize_strings;
};
static GTY(()) struct opt_stack * options_stack;
static void
handle_pragma_push_options (cpp_reader *ARG_UNUSED(dummy))
{
enum cpp_ttype token;
tree x = 0;
token = pragma_lex (&x);
if (token != CPP_EOF)
{
warning (OPT_Wpragmas, "junk at end of %<#pragma push_options%>");
return;
}
opt_stack *p = ggc_alloc<opt_stack> ();
p->prev = options_stack;
options_stack = p;
p->optimize_binary = build_optimization_node (&global_options);
p->target_binary = build_target_option_node (&global_options);
p->optimize_strings = copy_list (current_optimize_pragma);
p->target_strings = copy_list (current_target_pragma);
}
static void
handle_pragma_pop_options (cpp_reader *ARG_UNUSED(dummy))
{
enum cpp_ttype token;
tree x = 0;
opt_stack *p;
token = pragma_lex (&x);
if (token != CPP_EOF)
{
warning (OPT_Wpragmas, "junk at end of %<#pragma pop_options%>");
return;
}
if (! options_stack)
{
warning (OPT_Wpragmas,
"%<#pragma GCC pop_options%> without a corresponding "
"%<#pragma GCC push_options%>");
return;
}
p = options_stack;
options_stack = p->prev;
if (p->target_binary != target_option_current_node)
{
(void) targetm.target_option.pragma_parse (NULL_TREE, p->target_binary);
target_option_current_node = p->target_binary;
}
if (p->optimize_binary != optimization_current_node)
{
tree old_optimize = optimization_current_node;
cl_optimization_restore (&global_options,
TREE_OPTIMIZATION (p->optimize_binary));
c_cpp_builtins_optimize_pragma (parse_in, old_optimize,
p->optimize_binary);
optimization_current_node = p->optimize_binary;
}
current_target_pragma = p->target_strings;
current_optimize_pragma = p->optimize_strings;
}
static void
handle_pragma_reset_options (cpp_reader *ARG_UNUSED(dummy))
{
enum cpp_ttype token;
tree x = 0;
tree new_optimize = optimization_default_node;
tree new_target = target_option_default_node;
token = pragma_lex (&x);
if (token != CPP_EOF)
{
warning (OPT_Wpragmas, "junk at end of %<#pragma reset_options%>");
return;
}
if (new_target != target_option_current_node)
{
(void) targetm.target_option.pragma_parse (NULL_TREE, new_target);
target_option_current_node = new_target;
}
if (new_optimize != optimization_current_node)
{
tree old_optimize = optimization_current_node;
cl_optimization_restore (&global_options,
TREE_OPTIMIZATION (new_optimize));
c_cpp_builtins_optimize_pragma (parse_in, old_optimize, new_optimize);
optimization_current_node = new_optimize;
}
current_target_pragma = NULL_TREE;
current_optimize_pragma = NULL_TREE;
}
static void
handle_pragma_message (cpp_reader *ARG_UNUSED(dummy))
{
enum cpp_ttype token;
tree x, message = 0;
token = pragma_lex (&x);
if (token == CPP_OPEN_PAREN)
{
token = pragma_lex (&x);
if (token == CPP_STRING)
message = x;
else
GCC_BAD ("expected a string after %<#pragma message%>");
if (pragma_lex (&x) != CPP_CLOSE_PAREN)
GCC_BAD ("malformed %<#pragma message%>, ignored");
}
else if (token == CPP_STRING)
message = x;
else
GCC_BAD ("expected a string after %<#pragma message%>");
gcc_assert (message);
if (pragma_lex (&x) != CPP_EOF)
warning (OPT_Wpragmas, "junk at end of %<#pragma message%>");
if (TREE_STRING_LENGTH (message) > 1)
inform (input_location, "#pragma message: %s", TREE_STRING_POINTER (message));
}
static bool valid_location_for_stdc_pragma;
void
mark_valid_location_for_stdc_pragma (bool flag)
{
valid_location_for_stdc_pragma = flag;
}
bool
valid_location_for_stdc_pragma_p (void)
{
return valid_location_for_stdc_pragma;
}
enum pragma_switch_t { PRAGMA_ON, PRAGMA_OFF, PRAGMA_DEFAULT, PRAGMA_BAD };
static enum pragma_switch_t
handle_stdc_pragma (const char *pname)
{
const char *arg;
tree t;
enum pragma_switch_t ret;
if (!valid_location_for_stdc_pragma_p ())
{
warning (OPT_Wpragmas, "invalid location for %<pragma %s%>, ignored",
pname);
return PRAGMA_BAD;
}
if (pragma_lex (&t) != CPP_NAME)
{
warning (OPT_Wpragmas, "malformed %<#pragma %s%>, ignored", pname);
return PRAGMA_BAD;
}
arg = IDENTIFIER_POINTER (t);
if (!strcmp (arg, "ON"))
ret = PRAGMA_ON;
else if (!strcmp (arg, "OFF"))
ret = PRAGMA_OFF;
else if (!strcmp (arg, "DEFAULT"))
ret = PRAGMA_DEFAULT;
else
{
warning (OPT_Wpragmas, "malformed %<#pragma %s%>, ignored", pname);
return PRAGMA_BAD;
}
if (pragma_lex (&t) != CPP_EOF)
{
warning (OPT_Wpragmas, "junk at end of %<#pragma %s%>", pname);
return PRAGMA_BAD;
}
return ret;
}
static void
handle_pragma_float_const_decimal64 (cpp_reader *ARG_UNUSED (dummy))
{
if (c_dialect_cxx ())
{
if (warn_unknown_pragmas > in_system_header_at (input_location))
warning (OPT_Wunknown_pragmas,
"%<#pragma STDC FLOAT_CONST_DECIMAL64%> is not supported"
" for C++");
return;
}
if (!targetm.decimal_float_supported_p ())
{
if (warn_unknown_pragmas > in_system_header_at (input_location))
warning (OPT_Wunknown_pragmas,
"%<#pragma STDC FLOAT_CONST_DECIMAL64%> is not supported"
" on this target");
return;
}
pedwarn (input_location, OPT_Wpedantic,
"ISO C does not support %<#pragma STDC FLOAT_CONST_DECIMAL64%>");
switch (handle_stdc_pragma ("STDC FLOAT_CONST_DECIMAL64"))
{
case PRAGMA_ON:
set_float_const_decimal64 ();
break;
case PRAGMA_OFF:
case PRAGMA_DEFAULT:
clear_float_const_decimal64 ();
break;
case PRAGMA_BAD:
break;
}
}
static vec<internal_pragma_handler> registered_pragmas;
struct pragma_ns_name
{
const char *space;
const char *name;
};
static vec<pragma_ns_name> registered_pp_pragmas;
struct omp_pragma_def { const char *name; unsigned int id; };
static const struct omp_pragma_def oacc_pragmas[] = {
{ "atomic", PRAGMA_OACC_ATOMIC },
{ "cache", PRAGMA_OACC_CACHE },
{ "data", PRAGMA_OACC_DATA },
{ "declare", PRAGMA_OACC_DECLARE },
{ "enter", PRAGMA_OACC_ENTER_DATA },
{ "exit", PRAGMA_OACC_EXIT_DATA },
{ "host_data", PRAGMA_OACC_HOST_DATA },
{ "kernels", PRAGMA_OACC_KERNELS },
{ "loop", PRAGMA_OACC_LOOP },
{ "parallel", PRAGMA_OACC_PARALLEL },
{ "routine", PRAGMA_OACC_ROUTINE },
{ "update", PRAGMA_OACC_UPDATE },
{ "wait", PRAGMA_OACC_WAIT }
};
static const struct omp_pragma_def omp_pragmas[] = {
{ "atomic", PRAGMA_OMP_ATOMIC },
{ "barrier", PRAGMA_OMP_BARRIER },
{ "cancel", PRAGMA_OMP_CANCEL },
{ "cancellation", PRAGMA_OMP_CANCELLATION_POINT },
{ "critical", PRAGMA_OMP_CRITICAL },
{ "end", PRAGMA_OMP_END_DECLARE_TARGET },
{ "flush", PRAGMA_OMP_FLUSH },
{ "master", PRAGMA_OMP_MASTER },
{ "section", PRAGMA_OMP_SECTION },
{ "sections", PRAGMA_OMP_SECTIONS },
{ "single", PRAGMA_OMP_SINGLE },
{ "task", PRAGMA_OMP_TASK },
{ "taskgroup", PRAGMA_OMP_TASKGROUP },
{ "taskwait", PRAGMA_OMP_TASKWAIT },
{ "taskyield", PRAGMA_OMP_TASKYIELD },
{ "threadprivate", PRAGMA_OMP_THREADPRIVATE }
};
static const struct omp_pragma_def omp_pragmas_simd[] = {
{ "declare", PRAGMA_OMP_DECLARE },
{ "distribute", PRAGMA_OMP_DISTRIBUTE },
{ "for", PRAGMA_OMP_FOR },
{ "ordered", PRAGMA_OMP_ORDERED },
{ "parallel", PRAGMA_OMP_PARALLEL },
{ "simd", PRAGMA_OMP_SIMD },
{ "target", PRAGMA_OMP_TARGET },
{ "taskloop", PRAGMA_OMP_TASKLOOP },
{ "teams", PRAGMA_OMP_TEAMS },
};
void
c_pp_lookup_pragma (unsigned int id, const char **space, const char **name)
{
const int n_oacc_pragmas = sizeof (oacc_pragmas) / sizeof (*oacc_pragmas);
const int n_omp_pragmas = sizeof (omp_pragmas) / sizeof (*omp_pragmas);
const int n_omp_pragmas_simd = sizeof (omp_pragmas_simd)
/ sizeof (*omp_pragmas);
int i;
for (i = 0; i < n_oacc_pragmas; ++i)
if (oacc_pragmas[i].id == id)
{
*space = "acc";
*name = oacc_pragmas[i].name;
return;
}
for (i = 0; i < n_omp_pragmas; ++i)
if (omp_pragmas[i].id == id)
{
*space = "omp";
*name = omp_pragmas[i].name;
return;
}
for (i = 0; i < n_omp_pragmas_simd; ++i)
if (omp_pragmas_simd[i].id == id)
{
*space = "omp";
*name = omp_pragmas_simd[i].name;
return;
}
if (id >= PRAGMA_FIRST_EXTERNAL
&& (id < PRAGMA_FIRST_EXTERNAL + registered_pp_pragmas.length ()))
{
*space = registered_pp_pragmas[id - PRAGMA_FIRST_EXTERNAL].space;
*name = registered_pp_pragmas[id - PRAGMA_FIRST_EXTERNAL].name;
return;
}
gcc_unreachable ();
}
static void
c_register_pragma_1 (const char *space, const char *name,
internal_pragma_handler ihandler, bool allow_expansion)
{
unsigned id;
if (flag_preprocess_only)
{
pragma_ns_name ns_name;
if (!allow_expansion)
return;
ns_name.space = space;
ns_name.name = name;
registered_pp_pragmas.safe_push (ns_name);
id = registered_pp_pragmas.length ();
id += PRAGMA_FIRST_EXTERNAL - 1;
}
else
{
registered_pragmas.safe_push (ihandler);
id = registered_pragmas.length ();
id += PRAGMA_FIRST_EXTERNAL - 1;
gcc_assert (id < 256);
}
cpp_register_deferred_pragma (parse_in, space, name, id,
allow_expansion, false);
}
void
c_register_pragma (const char *space, const char *name,
pragma_handler_1arg handler)
{
internal_pragma_handler ihandler;
ihandler.handler.handler_1arg = handler;
ihandler.extra_data = false;
ihandler.data = NULL;
c_register_pragma_1 (space, name, ihandler, false);
}
void
c_register_pragma_with_data (const char *space, const char *name,
pragma_handler_2arg handler, void * data)
{
internal_pragma_handler ihandler;
ihandler.handler.handler_2arg = handler;
ihandler.extra_data = true;
ihandler.data = data;
c_register_pragma_1 (space, name, ihandler, false);
}
void
c_register_pragma_with_expansion (const char *space, const char *name,
pragma_handler_1arg handler)
{
internal_pragma_handler ihandler;
ihandler.handler.handler_1arg = handler;
ihandler.extra_data = false;
ihandler.data = NULL;
c_register_pragma_1 (space, name, ihandler, true);
}
void
c_register_pragma_with_expansion_and_data (const char *space, const char *name,
pragma_handler_2arg handler,
void *data)
{
internal_pragma_handler ihandler;
ihandler.handler.handler_2arg = handler;
ihandler.extra_data = true;
ihandler.data = data;
c_register_pragma_1 (space, name, ihandler, true);
}
void
c_invoke_pragma_handler (unsigned int id)
{
internal_pragma_handler *ihandler;
pragma_handler_1arg handler_1arg;
pragma_handler_2arg handler_2arg;
id -= PRAGMA_FIRST_EXTERNAL;
ihandler = &registered_pragmas[id];
if (ihandler->extra_data)
{
handler_2arg = ihandler->handler.handler_2arg;
handler_2arg (parse_in, ihandler->data);
}
else
{
handler_1arg = ihandler->handler.handler_1arg;
handler_1arg (parse_in);
}
}
void
init_pragma (void)
{
if (flag_openacc)
{
const int n_oacc_pragmas
= sizeof (oacc_pragmas) / sizeof (*oacc_pragmas);
int i;
for (i = 0; i < n_oacc_pragmas; ++i)
cpp_register_deferred_pragma (parse_in, "acc", oacc_pragmas[i].name,
oacc_pragmas[i].id, true, true);
}
if (flag_openmp)
{
const int n_omp_pragmas = sizeof (omp_pragmas) / sizeof (*omp_pragmas);
int i;
for (i = 0; i < n_omp_pragmas; ++i)
cpp_register_deferred_pragma (parse_in, "omp", omp_pragmas[i].name,
omp_pragmas[i].id, true, true);
}
if (flag_openmp || flag_openmp_simd)
{
const int n_omp_pragmas_simd = sizeof (omp_pragmas_simd)
/ sizeof (*omp_pragmas);
int i;
for (i = 0; i < n_omp_pragmas_simd; ++i)
cpp_register_deferred_pragma (parse_in, "omp", omp_pragmas_simd[i].name,
omp_pragmas_simd[i].id, true, true);
}
if (!flag_preprocess_only)
cpp_register_deferred_pragma (parse_in, "GCC", "pch_preprocess",
PRAGMA_GCC_PCH_PREPROCESS, false, false);
if (!flag_preprocess_only)
cpp_register_deferred_pragma (parse_in, "GCC", "ivdep", PRAGMA_IVDEP, false,
false);
if (!flag_preprocess_only)
cpp_register_deferred_pragma (parse_in, "GCC", "unroll", PRAGMA_UNROLL,
false, false);
#ifdef HANDLE_PRAGMA_PACK_WITH_EXPANSION
c_register_pragma_with_expansion (0, "pack", handle_pragma_pack);
#else
c_register_pragma (0, "pack", handle_pragma_pack);
#endif
c_register_pragma (0, "weak", handle_pragma_weak);
c_register_pragma ("GCC", "visibility", handle_pragma_visibility);
c_register_pragma ("GCC", "diagnostic", handle_pragma_diagnostic);
c_register_pragma ("GCC", "target", handle_pragma_target);
c_register_pragma ("GCC", "optimize", handle_pragma_optimize);
c_register_pragma ("GCC", "push_options", handle_pragma_push_options);
c_register_pragma ("GCC", "pop_options", handle_pragma_pop_options);
c_register_pragma ("GCC", "reset_options", handle_pragma_reset_options);
c_register_pragma ("STDC", "FLOAT_CONST_DECIMAL64",
handle_pragma_float_const_decimal64);
c_register_pragma_with_expansion (0, "redefine_extname",
handle_pragma_redefine_extname);
c_register_pragma_with_expansion (0, "message", handle_pragma_message);
#ifdef REGISTER_TARGET_PRAGMAS
REGISTER_TARGET_PRAGMAS ();
#endif
global_sso = default_sso;
c_register_pragma (0, "scalar_storage_order", 
handle_pragma_scalar_storage_order);
invoke_plugin_callbacks (PLUGIN_PRAGMAS, NULL);
}
#include "gt-c-family-c-pragma.h"

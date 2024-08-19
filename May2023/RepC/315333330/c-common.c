#define GCC_C_COMMON_C
#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "target.h"
#include "function.h"
#include "tree.h"
#include "memmodel.h"
#include "c-common.h"
#include "gimple-expr.h"
#include "tm_p.h"
#include "stringpool.h"
#include "cgraph.h"
#include "diagnostic.h"
#include "intl.h"
#include "stor-layout.h"
#include "calls.h"
#include "attribs.h"
#include "varasm.h"
#include "trans-mem.h"
#include "c-objc.h"
#include "common/common-target.h"
#include "langhooks.h"
#include "tree-inline.h"
#include "toplev.h"
#include "tree-iterator.h"
#include "opts.h"
#include "gimplify.h"
#include "substring-locations.h"
#include "spellcheck.h"
#include "selftest.h"
cpp_reader *parse_in;		
machine_mode c_default_pointer_mode = VOIDmode;
tree c_global_trees[CTI_MAX];

char flag_no_line_commands;
char flag_no_output;
char flag_dump_macros;
char flag_dump_includes;
bool flag_pch_preprocess;
const char *pch_file;
int flag_iso;
int flag_cond_mismatch;
int flag_isoc94;
int flag_isoc99;
int flag_isoc11;
int flag_hosted = 1;
int print_struct_values;
const char *constant_string_class_name;
int warn_abi_version = -1;
int flag_use_repository;
enum cxx_dialect cxx_dialect = cxx_unset;
int max_tinst_depth = 900;
tree *ridpointers;
tree (*make_fname_decl) (location_t, tree, int);
int c_inhibit_evaluation_warnings;
bool in_late_binary_op;
bool done_lexing = false;
struct fname_var_t
{
tree *const decl;	
const unsigned rid;	
const int pretty;	
};
const struct fname_var_t fname_vars[] =
{
{&c99_function_name_decl_node, RID_C99_FUNCTION_NAME, 0},
{&function_name_decl_node, RID_FUNCTION_NAME, 0},
{&pretty_function_name_decl_node, RID_PRETTY_FUNCTION_NAME, 1},
{NULL, 0, 0},
};
struct visibility_flags visibility_options;
static tree check_case_value (location_t, tree);
static bool check_case_bounds (location_t, tree, tree, tree *, tree *,
bool *);
static void check_nonnull_arg (void *, tree, unsigned HOST_WIDE_INT);
static bool nonnull_check_p (tree, unsigned HOST_WIDE_INT);
const struct c_common_resword c_common_reswords[] =
{
{ "_Alignas",		RID_ALIGNAS,   D_CONLY },
{ "_Alignof",		RID_ALIGNOF,   D_CONLY },
{ "_Atomic",		RID_ATOMIC,    D_CONLY },
{ "_Bool",		RID_BOOL,      D_CONLY },
{ "_Complex",		RID_COMPLEX,	0 },
{ "_Imaginary",	RID_IMAGINARY, D_CONLY },
{ "_Float16",         RID_FLOAT16,   D_CONLY },
{ "_Float32",         RID_FLOAT32,   D_CONLY },
{ "_Float64",         RID_FLOAT64,   D_CONLY },
{ "_Float128",        RID_FLOAT128,  D_CONLY },
{ "_Float32x",        RID_FLOAT32X,  D_CONLY },
{ "_Float64x",        RID_FLOAT64X,  D_CONLY },
{ "_Float128x",       RID_FLOAT128X, D_CONLY },
{ "_Decimal32",       RID_DFLOAT32,  D_CONLY | D_EXT },
{ "_Decimal64",       RID_DFLOAT64,  D_CONLY | D_EXT },
{ "_Decimal128",      RID_DFLOAT128, D_CONLY | D_EXT },
{ "_Fract",           RID_FRACT,     D_CONLY | D_EXT },
{ "_Accum",           RID_ACCUM,     D_CONLY | D_EXT },
{ "_Sat",             RID_SAT,       D_CONLY | D_EXT },
{ "_Static_assert",   RID_STATIC_ASSERT, D_CONLY },
{ "_Noreturn",        RID_NORETURN,  D_CONLY },
{ "_Generic",         RID_GENERIC,   D_CONLY },
{ "_Thread_local",    RID_THREAD,    D_CONLY },
{ "__FUNCTION__",	RID_FUNCTION_NAME, 0 },
{ "__PRETTY_FUNCTION__", RID_PRETTY_FUNCTION_NAME, 0 },
{ "__alignof",	RID_ALIGNOF,	0 },
{ "__alignof__",	RID_ALIGNOF,	0 },
{ "__asm",		RID_ASM,	0 },
{ "__asm__",		RID_ASM,	0 },
{ "__attribute",	RID_ATTRIBUTE,	0 },
{ "__attribute__",	RID_ATTRIBUTE,	0 },
{ "__auto_type",	RID_AUTO_TYPE,	D_CONLY },
{ "__bases",          RID_BASES, D_CXXONLY },
{ "__builtin_addressof", RID_ADDRESSOF, D_CXXONLY },
{ "__builtin_call_with_static_chain",
RID_BUILTIN_CALL_WITH_STATIC_CHAIN, D_CONLY },
{ "__builtin_choose_expr", RID_CHOOSE_EXPR, D_CONLY },
{ "__builtin_complex", RID_BUILTIN_COMPLEX, D_CONLY },
{ "__builtin_launder", RID_BUILTIN_LAUNDER, D_CXXONLY },
{ "__builtin_shuffle", RID_BUILTIN_SHUFFLE, 0 },
{ "__builtin_tgmath", RID_BUILTIN_TGMATH, D_CONLY },
{ "__builtin_offsetof", RID_OFFSETOF, 0 },
{ "__builtin_types_compatible_p", RID_TYPES_COMPATIBLE_P, D_CONLY },
{ "__builtin_va_arg",	RID_VA_ARG,	0 },
{ "__complex",	RID_COMPLEX,	0 },
{ "__complex__",	RID_COMPLEX,	0 },
{ "__const",		RID_CONST,	0 },
{ "__const__",	RID_CONST,	0 },
{ "__decltype",       RID_DECLTYPE,   D_CXXONLY },
{ "__direct_bases",   RID_DIRECT_BASES, D_CXXONLY },
{ "__extension__",	RID_EXTENSION,	0 },
{ "__func__",		RID_C99_FUNCTION_NAME, 0 },
{ "__has_nothrow_assign", RID_HAS_NOTHROW_ASSIGN, D_CXXONLY },
{ "__has_nothrow_constructor", RID_HAS_NOTHROW_CONSTRUCTOR, D_CXXONLY },
{ "__has_nothrow_copy", RID_HAS_NOTHROW_COPY, D_CXXONLY },
{ "__has_trivial_assign", RID_HAS_TRIVIAL_ASSIGN, D_CXXONLY },
{ "__has_trivial_constructor", RID_HAS_TRIVIAL_CONSTRUCTOR, D_CXXONLY },
{ "__has_trivial_copy", RID_HAS_TRIVIAL_COPY, D_CXXONLY },
{ "__has_trivial_destructor", RID_HAS_TRIVIAL_DESTRUCTOR, D_CXXONLY },
{ "__has_unique_object_representations", RID_HAS_UNIQUE_OBJ_REPRESENTATIONS,
D_CXXONLY },
{ "__has_virtual_destructor", RID_HAS_VIRTUAL_DESTRUCTOR, D_CXXONLY },
{ "__imag",		RID_IMAGPART,	0 },
{ "__imag__",		RID_IMAGPART,	0 },
{ "__inline",		RID_INLINE,	0 },
{ "__inline__",	RID_INLINE,	0 },
{ "__is_abstract",	RID_IS_ABSTRACT, D_CXXONLY },
{ "__is_aggregate",	RID_IS_AGGREGATE, D_CXXONLY },
{ "__is_base_of",	RID_IS_BASE_OF, D_CXXONLY },
{ "__is_class",	RID_IS_CLASS,	D_CXXONLY },
{ "__is_empty",	RID_IS_EMPTY,	D_CXXONLY },
{ "__is_enum",	RID_IS_ENUM,	D_CXXONLY },
{ "__is_final",	RID_IS_FINAL,	D_CXXONLY },
{ "__is_literal_type", RID_IS_LITERAL_TYPE, D_CXXONLY },
{ "__is_pod",		RID_IS_POD,	D_CXXONLY },
{ "__is_polymorphic",	RID_IS_POLYMORPHIC, D_CXXONLY },
{ "__is_same_as",     RID_IS_SAME_AS, D_CXXONLY },
{ "__is_standard_layout", RID_IS_STD_LAYOUT, D_CXXONLY },
{ "__is_trivial",     RID_IS_TRIVIAL, D_CXXONLY },
{ "__is_trivially_assignable", RID_IS_TRIVIALLY_ASSIGNABLE, D_CXXONLY },
{ "__is_trivially_constructible", RID_IS_TRIVIALLY_CONSTRUCTIBLE, D_CXXONLY },
{ "__is_trivially_copyable", RID_IS_TRIVIALLY_COPYABLE, D_CXXONLY },
{ "__is_union",	RID_IS_UNION,	D_CXXONLY },
{ "__label__",	RID_LABEL,	0 },
{ "__null",		RID_NULL,	0 },
{ "__real",		RID_REALPART,	0 },
{ "__real__",		RID_REALPART,	0 },
{ "__restrict",	RID_RESTRICT,	0 },
{ "__restrict__",	RID_RESTRICT,	0 },
{ "__signed",		RID_SIGNED,	0 },
{ "__signed__",	RID_SIGNED,	0 },
{ "__thread",		RID_THREAD,	0 },
{ "__transaction_atomic", RID_TRANSACTION_ATOMIC, 0 },
{ "__transaction_relaxed", RID_TRANSACTION_RELAXED, 0 },
{ "__transaction_cancel", RID_TRANSACTION_CANCEL, 0 },
{ "__typeof",		RID_TYPEOF,	0 },
{ "__typeof__",	RID_TYPEOF,	0 },
{ "__underlying_type", RID_UNDERLYING_TYPE, D_CXXONLY },
{ "__volatile",	RID_VOLATILE,	0 },
{ "__volatile__",	RID_VOLATILE,	0 },
{ "__GIMPLE",		RID_GIMPLE,	D_CONLY },
{ "__PHI",		RID_PHI,	D_CONLY },
{ "__RTL",		RID_RTL,	D_CONLY },
{ "alignas",		RID_ALIGNAS,	D_CXXONLY | D_CXX11 | D_CXXWARN },
{ "alignof",		RID_ALIGNOF,	D_CXXONLY | D_CXX11 | D_CXXWARN },
{ "asm",		RID_ASM,	D_ASM },
{ "auto",		RID_AUTO,	0 },
{ "bool",		RID_BOOL,	D_CXXONLY | D_CXXWARN },
{ "break",		RID_BREAK,	0 },
{ "case",		RID_CASE,	0 },
{ "catch",		RID_CATCH,	D_CXX_OBJC | D_CXXWARN },
{ "char",		RID_CHAR,	0 },
{ "char16_t",		RID_CHAR16,	D_CXXONLY | D_CXX11 | D_CXXWARN },
{ "char32_t",		RID_CHAR32,	D_CXXONLY | D_CXX11 | D_CXXWARN },
{ "class",		RID_CLASS,	D_CXX_OBJC | D_CXXWARN },
{ "const",		RID_CONST,	0 },
{ "constexpr",	RID_CONSTEXPR,	D_CXXONLY | D_CXX11 | D_CXXWARN },
{ "const_cast",	RID_CONSTCAST,	D_CXXONLY | D_CXXWARN },
{ "continue",		RID_CONTINUE,	0 },
{ "decltype",         RID_DECLTYPE,   D_CXXONLY | D_CXX11 | D_CXXWARN },
{ "default",		RID_DEFAULT,	0 },
{ "delete",		RID_DELETE,	D_CXXONLY | D_CXXWARN },
{ "do",		RID_DO,		0 },
{ "double",		RID_DOUBLE,	0 },
{ "dynamic_cast",	RID_DYNCAST,	D_CXXONLY | D_CXXWARN },
{ "else",		RID_ELSE,	0 },
{ "enum",		RID_ENUM,	0 },
{ "explicit",		RID_EXPLICIT,	D_CXXONLY | D_CXXWARN },
{ "export",		RID_EXPORT,	D_CXXONLY | D_CXXWARN },
{ "extern",		RID_EXTERN,	0 },
{ "false",		RID_FALSE,	D_CXXONLY | D_CXXWARN },
{ "float",		RID_FLOAT,	0 },
{ "for",		RID_FOR,	0 },
{ "friend",		RID_FRIEND,	D_CXXONLY | D_CXXWARN },
{ "goto",		RID_GOTO,	0 },
{ "if",		RID_IF,		0 },
{ "inline",		RID_INLINE,	D_EXT89 },
{ "int",		RID_INT,	0 },
{ "long",		RID_LONG,	0 },
{ "mutable",		RID_MUTABLE,	D_CXXONLY | D_CXXWARN },
{ "namespace",	RID_NAMESPACE,	D_CXXONLY | D_CXXWARN },
{ "new",		RID_NEW,	D_CXXONLY | D_CXXWARN },
{ "noexcept",		RID_NOEXCEPT,	D_CXXONLY | D_CXX11 | D_CXXWARN },
{ "nullptr",		RID_NULLPTR,	D_CXXONLY | D_CXX11 | D_CXXWARN },
{ "operator",		RID_OPERATOR,	D_CXXONLY | D_CXXWARN },
{ "private",		RID_PRIVATE,	D_CXX_OBJC | D_CXXWARN },
{ "protected",	RID_PROTECTED,	D_CXX_OBJC | D_CXXWARN },
{ "public",		RID_PUBLIC,	D_CXX_OBJC | D_CXXWARN },
{ "register",		RID_REGISTER,	0 },
{ "reinterpret_cast",	RID_REINTCAST,	D_CXXONLY | D_CXXWARN },
{ "restrict",		RID_RESTRICT,	D_CONLY | D_C99 },
{ "return",		RID_RETURN,	0 },
{ "short",		RID_SHORT,	0 },
{ "signed",		RID_SIGNED,	0 },
{ "sizeof",		RID_SIZEOF,	0 },
{ "static",		RID_STATIC,	0 },
{ "static_assert",    RID_STATIC_ASSERT, D_CXXONLY | D_CXX11 | D_CXXWARN },
{ "static_cast",	RID_STATCAST,	D_CXXONLY | D_CXXWARN },
{ "struct",		RID_STRUCT,	0 },
{ "switch",		RID_SWITCH,	0 },
{ "template",		RID_TEMPLATE,	D_CXXONLY | D_CXXWARN },
{ "this",		RID_THIS,	D_CXXONLY | D_CXXWARN },
{ "thread_local",	RID_THREAD,	D_CXXONLY | D_CXX11 | D_CXXWARN },
{ "throw",		RID_THROW,	D_CXX_OBJC | D_CXXWARN },
{ "true",		RID_TRUE,	D_CXXONLY | D_CXXWARN },
{ "try",		RID_TRY,	D_CXX_OBJC | D_CXXWARN },
{ "typedef",		RID_TYPEDEF,	0 },
{ "typename",		RID_TYPENAME,	D_CXXONLY | D_CXXWARN },
{ "typeid",		RID_TYPEID,	D_CXXONLY | D_CXXWARN },
{ "typeof",		RID_TYPEOF,	D_ASM | D_EXT },
{ "union",		RID_UNION,	0 },
{ "unsigned",		RID_UNSIGNED,	0 },
{ "using",		RID_USING,	D_CXXONLY | D_CXXWARN },
{ "virtual",		RID_VIRTUAL,	D_CXXONLY | D_CXXWARN },
{ "void",		RID_VOID,	0 },
{ "volatile",		RID_VOLATILE,	0 },
{ "wchar_t",		RID_WCHAR,	D_CXXONLY },
{ "while",		RID_WHILE,	0 },
{ "__is_assignable", RID_IS_ASSIGNABLE, D_CXXONLY },
{ "__is_constructible", RID_IS_CONSTRUCTIBLE, D_CXXONLY },
{ "synchronized",	RID_SYNCHRONIZED, D_CXX_OBJC | D_TRANSMEM },
{ "atomic_noexcept",	RID_ATOMIC_NOEXCEPT, D_CXXONLY | D_TRANSMEM },
{ "atomic_cancel",	RID_ATOMIC_CANCEL, D_CXXONLY | D_TRANSMEM },
{ "atomic_commit",	RID_TRANSACTION_ATOMIC, D_CXXONLY | D_TRANSMEM },
{ "concept",		RID_CONCEPT,	D_CXX_CONCEPTS_FLAGS | D_CXXWARN },
{ "requires", 	RID_REQUIRES,	D_CXX_CONCEPTS_FLAGS | D_CXXWARN },
{ "compatibility_alias", RID_AT_ALIAS,	D_OBJC },
{ "defs",		RID_AT_DEFS,		D_OBJC },
{ "encode",		RID_AT_ENCODE,		D_OBJC },
{ "end",		RID_AT_END,		D_OBJC },
{ "implementation",	RID_AT_IMPLEMENTATION,	D_OBJC },
{ "interface",	RID_AT_INTERFACE,	D_OBJC },
{ "protocol",		RID_AT_PROTOCOL,	D_OBJC },
{ "selector",		RID_AT_SELECTOR,	D_OBJC },
{ "finally",		RID_AT_FINALLY,		D_OBJC },
{ "optional",		RID_AT_OPTIONAL,	D_OBJC },
{ "required",		RID_AT_REQUIRED,	D_OBJC },
{ "property",		RID_AT_PROPERTY,	D_OBJC },
{ "package",		RID_AT_PACKAGE,		D_OBJC },
{ "synthesize",	RID_AT_SYNTHESIZE,	D_OBJC },
{ "dynamic",		RID_AT_DYNAMIC,		D_OBJC },
{ "bycopy",		RID_BYCOPY,		D_OBJC },
{ "byref",		RID_BYREF,		D_OBJC },
{ "in",		RID_IN,			D_OBJC },
{ "inout",		RID_INOUT,		D_OBJC },
{ "oneway",		RID_ONEWAY,		D_OBJC },
{ "out",		RID_OUT,		D_OBJC },
{ "assign",	        RID_ASSIGN,		D_OBJC }, 
{ "copy",	        RID_COPY,		D_OBJC }, 
{ "getter",		RID_GETTER,		D_OBJC }, 
{ "nonatomic",	RID_NONATOMIC,		D_OBJC }, 
{ "readonly",		RID_READONLY,		D_OBJC }, 
{ "readwrite",	RID_READWRITE,		D_OBJC }, 
{ "retain",	        RID_RETAIN,		D_OBJC }, 
{ "setter",		RID_SETTER,		D_OBJC }, 
};
const unsigned int num_c_common_reswords =
sizeof c_common_reswords / sizeof (struct c_common_resword);
const char *
c_addr_space_name (addr_space_t as)
{
int rid = RID_FIRST_ADDR_SPACE + as;
gcc_assert (ridpointers [rid]);
return IDENTIFIER_POINTER (ridpointers [rid]);
}
void
start_fname_decls (void)
{
unsigned ix;
tree saved = NULL_TREE;
for (ix = 0; fname_vars[ix].decl; ix++)
{
tree decl = *fname_vars[ix].decl;
if (decl)
{
saved = tree_cons (decl, build_int_cst (integer_type_node, ix),
saved);
*fname_vars[ix].decl = NULL_TREE;
}
}
if (saved || saved_function_name_decls)
saved_function_name_decls = tree_cons (saved, NULL_TREE,
saved_function_name_decls);
}
void
finish_fname_decls (void)
{
unsigned ix;
tree stmts = NULL_TREE;
tree stack = saved_function_name_decls;
for (; stack && TREE_VALUE (stack); stack = TREE_CHAIN (stack))
append_to_statement_list (TREE_VALUE (stack), &stmts);
if (stmts)
{
tree *bodyp = &DECL_SAVED_TREE (current_function_decl);
if (TREE_CODE (*bodyp) == BIND_EXPR)
bodyp = &BIND_EXPR_BODY (*bodyp);
append_to_statement_list_force (*bodyp, &stmts);
*bodyp = stmts;
}
for (ix = 0; fname_vars[ix].decl; ix++)
*fname_vars[ix].decl = NULL_TREE;
if (stack)
{
tree saved;
for (saved = TREE_PURPOSE (stack); saved; saved = TREE_CHAIN (saved))
{
tree decl = TREE_PURPOSE (saved);
unsigned ix = TREE_INT_CST_LOW (TREE_VALUE (saved));
*fname_vars[ix].decl = decl;
}
stack = TREE_CHAIN (stack);
}
saved_function_name_decls = stack;
}
const char *
fname_as_string (int pretty_p)
{
const char *name = "top level";
char *namep;
int vrb = 2, len;
cpp_string cstr = { 0, 0 }, strname;
if (!pretty_p)
{
name = "";
vrb = 0;
}
if (current_function_decl)
name = lang_hooks.decl_printable_name (current_function_decl, vrb);
len = strlen (name) + 3; 
namep = XNEWVEC (char, len);
snprintf (namep, len, "\"%s\"", name);
strname.text = (unsigned char *) namep;
strname.len = len - 1;
if (cpp_interpret_string (parse_in, &strname, 1, &cstr, CPP_STRING))
{
XDELETEVEC (namep);
return (const char *) cstr.text;
}
return namep;
}
tree
fname_decl (location_t loc, unsigned int rid, tree id)
{
unsigned ix;
tree decl = NULL_TREE;
for (ix = 0; fname_vars[ix].decl; ix++)
if (fname_vars[ix].rid == rid)
break;
decl = *fname_vars[ix].decl;
if (!decl)
{
tree stmts;
location_t saved_location = input_location;
input_location = UNKNOWN_LOCATION;
stmts = push_stmt_list ();
decl = (*make_fname_decl) (loc, id, fname_vars[ix].pretty);
stmts = pop_stmt_list (stmts);
if (!IS_EMPTY_STMT (stmts))
saved_function_name_decls
= tree_cons (decl, stmts, saved_function_name_decls);
*fname_vars[ix].decl = decl;
input_location = saved_location;
}
if (!ix && !current_function_decl)
pedwarn (loc, 0, "%qD is not defined outside of function scope", decl);
return decl;
}
tree
fix_string_type (tree value)
{
int length = TREE_STRING_LENGTH (value);
int nchars;
tree e_type, i_type, a_type;
if (TREE_TYPE (value) == char_array_type_node || !TREE_TYPE (value))
{
nchars = length;
e_type = char_type_node;
}
else if (TREE_TYPE (value) == char16_array_type_node)
{
nchars = length / (TYPE_PRECISION (char16_type_node) / BITS_PER_UNIT);
e_type = char16_type_node;
}
else if (TREE_TYPE (value) == char32_array_type_node)
{
nchars = length / (TYPE_PRECISION (char32_type_node) / BITS_PER_UNIT);
e_type = char32_type_node;
}
else
{
nchars = length / (TYPE_PRECISION (wchar_type_node) / BITS_PER_UNIT);
e_type = wchar_type_node;
}
if (warn_overlength_strings)
{
const int nchars_max = flag_isoc99 ? 4095 : 509;
const int relevant_std = flag_isoc99 ? 99 : 90;
if (nchars - 1 > nchars_max)
pedwarn (input_location, OPT_Woverlength_strings,
"string length %qd is greater than the length %qd "
"ISO C%d compilers are required to support",
nchars - 1, nchars_max, relevant_std);
}
i_type = build_index_type (size_int (nchars - 1));
a_type = build_array_type (e_type, i_type);
if (c_dialect_cxx() || warn_write_strings)
a_type = c_build_qualified_type (a_type, TYPE_QUAL_CONST);
TREE_TYPE (value) = a_type;
TREE_CONSTANT (value) = 1;
TREE_READONLY (value) = 1;
TREE_STATIC (value) = 1;
return value;
}
static enum cpp_ttype
get_cpp_ttype_from_string_type (tree string_type)
{
gcc_assert (string_type);
if (TREE_CODE (string_type) == POINTER_TYPE)
string_type = TREE_TYPE (string_type);
if (TREE_CODE (string_type) != ARRAY_TYPE)
return CPP_OTHER;
tree element_type = TREE_TYPE (string_type);
if (TREE_CODE (element_type) != INTEGER_TYPE)
return CPP_OTHER;
int bits_per_character = TYPE_PRECISION (element_type);
switch (bits_per_character)
{
case 8:
return CPP_STRING;  
case 16:
return CPP_STRING16;
case 32:
return CPP_STRING32;
}
return CPP_OTHER;
}
GTY(()) string_concat_db *g_string_concat_db;
const char *
c_get_substring_location (const substring_loc &substr_loc,
location_t *out_loc)
{
enum cpp_ttype tok_type
= get_cpp_ttype_from_string_type (substr_loc.get_string_type ());
if (tok_type == CPP_OTHER)
return "unrecognized string type";
return get_source_location_for_substring (parse_in, g_string_concat_db,
substr_loc.get_fmt_string_loc (),
tok_type,
substr_loc.get_caret_idx (),
substr_loc.get_start_idx (),
substr_loc.get_end_idx (),
out_loc);
}

bool
bool_promoted_to_int_p (tree t)
{
return (CONVERT_EXPR_P (t)
&& TREE_TYPE (t) == integer_type_node
&& TREE_CODE (TREE_TYPE (TREE_OPERAND (t, 0))) == BOOLEAN_TYPE);
}
bool
vector_targets_convertible_p (const_tree t1, const_tree t2)
{
if (VECTOR_TYPE_P (t1) && VECTOR_TYPE_P (t2)
&& (TYPE_VECTOR_OPAQUE (t1) || TYPE_VECTOR_OPAQUE (t2))
&& tree_int_cst_equal (TYPE_SIZE (t1), TYPE_SIZE (t2)))
return true;
return false;
}
bool
vector_types_convertible_p (const_tree t1, const_tree t2, bool emit_lax_note)
{
static bool emitted_lax_note = false;
bool convertible_lax;
if ((TYPE_VECTOR_OPAQUE (t1) || TYPE_VECTOR_OPAQUE (t2))
&& tree_int_cst_equal (TYPE_SIZE (t1), TYPE_SIZE (t2)))
return true;
convertible_lax =
(tree_int_cst_equal (TYPE_SIZE (t1), TYPE_SIZE (t2))
&& (TREE_CODE (TREE_TYPE (t1)) != REAL_TYPE
|| known_eq (TYPE_VECTOR_SUBPARTS (t1),
TYPE_VECTOR_SUBPARTS (t2)))
&& (INTEGRAL_TYPE_P (TREE_TYPE (t1))
== INTEGRAL_TYPE_P (TREE_TYPE (t2))));
if (!convertible_lax || flag_lax_vector_conversions)
return convertible_lax;
if (known_eq (TYPE_VECTOR_SUBPARTS (t1), TYPE_VECTOR_SUBPARTS (t2))
&& lang_hooks.types_compatible_p (TREE_TYPE (t1), TREE_TYPE (t2)))
return true;
if (emit_lax_note && !emitted_lax_note)
{
emitted_lax_note = true;
inform (input_location, "use -flax-vector-conversions to permit "
"conversions between vectors with differing "
"element types or numbers of subparts");
}
return false;
}
tree
c_build_vec_perm_expr (location_t loc, tree v0, tree v1, tree mask,
bool complain)
{
tree ret;
bool wrap = true;
bool maybe_const = false;
bool two_arguments = false;
if (v1 == NULL_TREE)
{
two_arguments = true;
v1 = v0;
}
if (v0 == error_mark_node || v1 == error_mark_node
|| mask == error_mark_node)
return error_mark_node;
if (!VECTOR_INTEGER_TYPE_P (TREE_TYPE (mask)))
{
if (complain)
error_at (loc, "__builtin_shuffle last argument must "
"be an integer vector");
return error_mark_node;
}
if (!VECTOR_TYPE_P (TREE_TYPE (v0))
|| !VECTOR_TYPE_P (TREE_TYPE (v1)))
{
if (complain)
error_at (loc, "__builtin_shuffle arguments must be vectors");
return error_mark_node;
}
if (TYPE_MAIN_VARIANT (TREE_TYPE (v0)) != TYPE_MAIN_VARIANT (TREE_TYPE (v1)))
{
if (complain)
error_at (loc, "__builtin_shuffle argument vectors must be of "
"the same type");
return error_mark_node;
}
if (maybe_ne (TYPE_VECTOR_SUBPARTS (TREE_TYPE (v0)),
TYPE_VECTOR_SUBPARTS (TREE_TYPE (mask)))
&& maybe_ne (TYPE_VECTOR_SUBPARTS (TREE_TYPE (v1)),
TYPE_VECTOR_SUBPARTS (TREE_TYPE (mask))))
{
if (complain)
error_at (loc, "__builtin_shuffle number of elements of the "
"argument vector(s) and the mask vector should "
"be the same");
return error_mark_node;
}
if (GET_MODE_BITSIZE (SCALAR_TYPE_MODE (TREE_TYPE (TREE_TYPE (v0))))
!= GET_MODE_BITSIZE (SCALAR_TYPE_MODE (TREE_TYPE (TREE_TYPE (mask)))))
{
if (complain)
error_at (loc, "__builtin_shuffle argument vector(s) inner type "
"must have the same size as inner type of the mask");
return error_mark_node;
}
if (!c_dialect_cxx ())
{
v0 = c_fully_fold (v0, false, &maybe_const);
wrap &= maybe_const;
if (two_arguments)
v1 = v0 = save_expr (v0);
else
{
v1 = c_fully_fold (v1, false, &maybe_const);
wrap &= maybe_const;
}
mask = c_fully_fold (mask, false, &maybe_const);
wrap &= maybe_const;
}
else if (two_arguments)
v1 = v0 = save_expr (v0);
ret = build3_loc (loc, VEC_PERM_EXPR, TREE_TYPE (v0), v0, v1, mask);
if (!c_dialect_cxx () && !wrap)
ret = c_wrap_maybe_const (ret, true);
return ret;
}
tree
c_common_get_narrower (tree op, int *unsignedp_ptr)
{
op = get_narrower (op, unsignedp_ptr);
if (TREE_CODE (TREE_TYPE (op)) == ENUMERAL_TYPE
&& ENUM_IS_SCOPED (TREE_TYPE (op)))
{
tree type = c_common_type_for_size (TYPE_PRECISION (TREE_TYPE (op)),
TYPE_UNSIGNED (TREE_TYPE (op)));
op = fold_convert (type, op);
}
return op;
}
tree
shorten_binary_op (tree result_type, tree op0, tree op1, bool bitwise)
{
int unsigned0, unsigned1;
tree arg0, arg1;
int uns;
tree type;
op0 = convert (result_type, op0);
op1 = convert (result_type, op1);
arg0 = c_common_get_narrower (op0, &unsigned0);
arg1 = c_common_get_narrower (op1, &unsigned1);
uns = TYPE_UNSIGNED (result_type);
if ((TYPE_PRECISION (TREE_TYPE (op0))
== TYPE_PRECISION (TREE_TYPE (arg0)))
&& TREE_TYPE (op0) != result_type)
unsigned0 = TYPE_UNSIGNED (TREE_TYPE (op0));
if ((TYPE_PRECISION (TREE_TYPE (op1))
== TYPE_PRECISION (TREE_TYPE (arg1)))
&& TREE_TYPE (op1) != result_type)
unsigned1 = TYPE_UNSIGNED (TREE_TYPE (op1));
if (bitwise)
uns = unsigned0;
if ((TYPE_PRECISION (TREE_TYPE (arg0))
< TYPE_PRECISION (result_type))
&& (TYPE_PRECISION (TREE_TYPE (arg1))
== TYPE_PRECISION (TREE_TYPE (arg0)))
&& unsigned0 == unsigned1
&& (unsigned0 || !uns))
return c_common_signed_or_unsigned_type
(unsigned0, common_type (TREE_TYPE (arg0), TREE_TYPE (arg1)));
else if (TREE_CODE (arg0) == INTEGER_CST
&& (unsigned1 || !uns)
&& (TYPE_PRECISION (TREE_TYPE (arg1))
< TYPE_PRECISION (result_type))
&& (type
= c_common_signed_or_unsigned_type (unsigned1,
TREE_TYPE (arg1)))
&& !POINTER_TYPE_P (type)
&& int_fits_type_p (arg0, type))
return type;
else if (TREE_CODE (arg1) == INTEGER_CST
&& (unsigned0 || !uns)
&& (TYPE_PRECISION (TREE_TYPE (arg0))
< TYPE_PRECISION (result_type))
&& (type
= c_common_signed_or_unsigned_type (unsigned0,
TREE_TYPE (arg0)))
&& !POINTER_TYPE_P (type)
&& int_fits_type_p (arg1, type))
return type;
return result_type;
}
static bool
int_safely_convertible_to_real_p (const_tree from_type, const_tree to_type)
{
tree type_low_bound = TYPE_MIN_VALUE (from_type);
tree type_high_bound = TYPE_MAX_VALUE (from_type);
REAL_VALUE_TYPE real_low_bound =
real_value_from_int_cst (0, type_low_bound);
REAL_VALUE_TYPE real_high_bound =
real_value_from_int_cst (0, type_high_bound);
return exact_real_truncate (TYPE_MODE (to_type), &real_low_bound)
&& exact_real_truncate (TYPE_MODE (to_type), &real_high_bound);
}
enum conversion_safety
unsafe_conversion_p (location_t loc, tree type, tree expr, tree result,
bool produce_warns)
{
enum conversion_safety give_warning = SAFE_CONVERSION; 
tree expr_type = TREE_TYPE (expr);
bool cstresult = (result
&& TREE_CODE_CLASS (TREE_CODE (result)) == tcc_constant);
loc = expansion_point_location_if_in_system_header (loc);
if (TREE_CODE (expr) == REAL_CST || TREE_CODE (expr) == INTEGER_CST)
{
if (TREE_CODE (type) == COMPLEX_TYPE)
type = TREE_TYPE (type);
if (TREE_CODE (expr_type) == REAL_TYPE
&& TREE_CODE (type) == INTEGER_TYPE)
{
if (!real_isinteger (TREE_REAL_CST_PTR (expr), TYPE_MODE (expr_type)))
give_warning = UNSAFE_REAL;
}
else if (TREE_CODE (expr_type) == INTEGER_TYPE
&& TREE_CODE (type) == INTEGER_TYPE
&& !int_fits_type_p (expr, type))
{
if (TYPE_UNSIGNED (type) && !TYPE_UNSIGNED (expr_type)
&& tree_int_cst_sgn (expr) < 0)
{
if (produce_warns)
{
if (cstresult)
warning_at (loc, OPT_Wsign_conversion,
"unsigned conversion from %qT to %qT "
"changes value from %qE to %qE",
expr_type, type, expr, result);
else
warning_at (loc, OPT_Wsign_conversion,
"unsigned conversion from %qT to %qT "
"changes the value of %qE",
expr_type, type, expr);
}
}
else if (!TYPE_UNSIGNED (type) && TYPE_UNSIGNED (expr_type))
{
if (cstresult)
warning_at (loc, OPT_Wsign_conversion,
"signed conversion from %qT to %qT changes "
"value from %qE to %qE",
expr_type, type, expr, result);
else
warning_at (loc, OPT_Wsign_conversion,
"signed conversion from %qT to %qT changes "
"the value of %qE",
expr_type, type, expr);
}
else
give_warning = UNSAFE_OTHER;
}
else if (TREE_CODE (type) == REAL_TYPE)
{
if (TREE_CODE (expr_type) == INTEGER_TYPE)
{
REAL_VALUE_TYPE a = real_value_from_int_cst (0, expr);
if (!exact_real_truncate (TYPE_MODE (type), &a))
give_warning = UNSAFE_REAL;
}
else if (TREE_CODE (expr_type) == REAL_TYPE
&& TYPE_PRECISION (type) < TYPE_PRECISION (expr_type))
{
REAL_VALUE_TYPE a = TREE_REAL_CST (expr);
if (!exact_real_truncate (TYPE_MODE (type), &a))
give_warning = UNSAFE_REAL;
}
}
}
else if (TREE_CODE (expr) == COMPLEX_CST)
{
tree imag_part = TREE_IMAGPART (expr);
if ((TREE_CODE (imag_part) == REAL_CST
&& real_zerop (imag_part))
|| (TREE_CODE (imag_part) == INTEGER_CST
&& integer_zerop (imag_part)))
return unsafe_conversion_p (loc, type, TREE_REALPART (expr), result,
produce_warns);
else
{
if (TREE_CODE (type) == COMPLEX_TYPE)
{
enum conversion_safety re_safety =
unsafe_conversion_p (loc, type, TREE_REALPART (expr),
result, false);
enum conversion_safety im_safety =
unsafe_conversion_p (loc, type, imag_part, result, false);
if (re_safety == im_safety)
give_warning = re_safety;
else if (!re_safety && im_safety)
give_warning = im_safety;
else if (re_safety && !im_safety)
give_warning = re_safety;
else
give_warning = UNSAFE_OTHER;
}
else
give_warning = UNSAFE_IMAGINARY;
}
}
else
{
if (TREE_CODE (expr_type) == REAL_TYPE
&& TREE_CODE (type) == INTEGER_TYPE)
give_warning = UNSAFE_REAL;
else if (TREE_CODE (expr_type) == INTEGER_TYPE
&& TREE_CODE (type) == INTEGER_TYPE)
{
expr = get_unwidened (expr, 0);
expr_type = TREE_TYPE (expr);
if (TREE_CODE (expr) == BIT_AND_EXPR
|| TREE_CODE (expr) == BIT_IOR_EXPR
|| TREE_CODE (expr) == BIT_XOR_EXPR)
{
expr_type = shorten_binary_op (expr_type,
TREE_OPERAND (expr, 0),
TREE_OPERAND (expr, 1),
1);
if (TREE_CODE (expr) == BIT_AND_EXPR)
{
tree op0 = TREE_OPERAND (expr, 0);
tree op1 = TREE_OPERAND (expr, 1);
bool unsigned0 = TYPE_UNSIGNED (TREE_TYPE (op0));
bool unsigned1 = TYPE_UNSIGNED (TREE_TYPE (op1));
if ((TREE_CODE (op0) == INTEGER_CST
&& int_fits_type_p (op0, c_common_signed_type (type))
&& int_fits_type_p (op0, c_common_unsigned_type (type)))
|| (TREE_CODE (op1) == INTEGER_CST
&& int_fits_type_p (op1, c_common_signed_type (type))
&& int_fits_type_p (op1,
c_common_unsigned_type (type))))
return SAFE_CONVERSION;
else if ((TREE_CODE (op0) == INTEGER_CST
&& unsigned0
&& int_fits_type_p (op0, type))
|| (TREE_CODE (op1) == INTEGER_CST
&& unsigned1
&& int_fits_type_p (op1, type)))
return SAFE_CONVERSION;
}
}
if (TYPE_PRECISION (type) < TYPE_PRECISION (expr_type))
give_warning = UNSAFE_OTHER;
else if (((TYPE_PRECISION (type) == TYPE_PRECISION (expr_type)
&& TYPE_UNSIGNED (expr_type) != TYPE_UNSIGNED (type))
|| (TYPE_UNSIGNED (type) && !TYPE_UNSIGNED (expr_type)))
&& produce_warns)
warning_at (loc, OPT_Wsign_conversion, "conversion to %qT from %qT "
"may change the sign of the result",
type, expr_type);
}
else if (TREE_CODE (expr_type) == INTEGER_TYPE
&& TREE_CODE (type) == REAL_TYPE)
{
expr = get_unwidened (expr, 0);
expr_type = TREE_TYPE (expr);
if (!int_safely_convertible_to_real_p (expr_type, type))
give_warning = UNSAFE_OTHER;
}
else if (TREE_CODE (expr_type) == REAL_TYPE
&& TREE_CODE (type) == REAL_TYPE
&& TYPE_PRECISION (type) < TYPE_PRECISION (expr_type))
give_warning = UNSAFE_REAL;
else if (TREE_CODE (expr_type) == COMPLEX_TYPE
&& TREE_CODE (type) == COMPLEX_TYPE)
{
tree from_type = TREE_TYPE (expr_type);
tree to_type = TREE_TYPE (type);
if (TREE_CODE (from_type) == REAL_TYPE
&& TREE_CODE (to_type) == INTEGER_TYPE)
give_warning = UNSAFE_REAL;
else if (TREE_CODE (from_type) == REAL_TYPE
&& TREE_CODE (to_type) == REAL_TYPE
&& TYPE_PRECISION (to_type) < TYPE_PRECISION (from_type))
give_warning = UNSAFE_REAL;
else if (TREE_CODE (from_type) == INTEGER_TYPE
&& TREE_CODE (to_type) == INTEGER_TYPE)
{
if (TYPE_PRECISION (to_type) < TYPE_PRECISION (from_type))
give_warning = UNSAFE_OTHER;
else if (((TYPE_PRECISION (to_type) == TYPE_PRECISION (from_type)
&& TYPE_UNSIGNED (to_type) != TYPE_UNSIGNED (from_type))
|| (TYPE_UNSIGNED (to_type) && !TYPE_UNSIGNED (from_type)))
&& produce_warns)
warning_at (loc, OPT_Wsign_conversion,
"conversion to %qT from %qT "
"may change the sign of the result",
type, expr_type);
}
else if (TREE_CODE (from_type) == INTEGER_TYPE
&& TREE_CODE (to_type) == REAL_TYPE
&& !int_safely_convertible_to_real_p (from_type, to_type))
give_warning = UNSAFE_OTHER;
}
else if (TREE_CODE (expr_type) == COMPLEX_TYPE
&& TREE_CODE (type) != COMPLEX_TYPE)
give_warning = UNSAFE_IMAGINARY;
}
return give_warning;
}
tree
convert_and_check (location_t loc, tree type, tree expr)
{
tree result;
tree expr_for_warning;
if (TREE_CODE (expr) == EXCESS_PRECISION_EXPR)
{
tree orig_type = TREE_TYPE (expr);
expr = TREE_OPERAND (expr, 0);
expr_for_warning = convert (orig_type, expr);
if (orig_type == type)
return expr_for_warning;
}
else
expr_for_warning = expr;
if (TREE_TYPE (expr) == type)
return expr;
result = convert (type, expr);
if (c_inhibit_evaluation_warnings == 0
&& !TREE_OVERFLOW_P (expr)
&& result != error_mark_node)
warnings_for_convert_and_check (loc, type, expr_for_warning, result);
return result;
}

struct tlist
{
struct tlist *next;
tree expr, writer;
};
struct tlist_cache
{
struct tlist_cache *next;
struct tlist *cache_before_sp;
struct tlist *cache_after_sp;
tree expr;
};
static struct obstack tlist_obstack;
static char *tlist_firstobj = 0;
static struct tlist *warned_ids;
static struct tlist_cache *save_expr_cache;
static void add_tlist (struct tlist **, struct tlist *, tree, int);
static void merge_tlist (struct tlist **, struct tlist *, int);
static void verify_tree (tree, struct tlist **, struct tlist **, tree);
static bool warning_candidate_p (tree);
static bool candidate_equal_p (const_tree, const_tree);
static void warn_for_collisions (struct tlist *);
static void warn_for_collisions_1 (tree, tree, struct tlist *, int);
static struct tlist *new_tlist (struct tlist *, tree, tree);
static struct tlist *
new_tlist (struct tlist *next, tree t, tree writer)
{
struct tlist *l;
l = XOBNEW (&tlist_obstack, struct tlist);
l->next = next;
l->expr = t;
l->writer = writer;
return l;
}
static void
add_tlist (struct tlist **to, struct tlist *add, tree exclude_writer, int copy)
{
while (add)
{
struct tlist *next = add->next;
if (!copy)
add->next = *to;
if (!exclude_writer || !candidate_equal_p (add->writer, exclude_writer))
*to = copy ? new_tlist (*to, add->expr, add->writer) : add;
add = next;
}
}
static void
merge_tlist (struct tlist **to, struct tlist *add, int copy)
{
struct tlist **end = to;
while (*end)
end = &(*end)->next;
while (add)
{
int found = 0;
struct tlist *tmp2;
struct tlist *next = add->next;
for (tmp2 = *to; tmp2; tmp2 = tmp2->next)
if (candidate_equal_p (tmp2->expr, add->expr))
{
found = 1;
if (!tmp2->writer)
tmp2->writer = add->writer;
}
if (!found)
{
*end = copy ? new_tlist (NULL, add->expr, add->writer) : add;
end = &(*end)->next;
*end = 0;
}
add = next;
}
}
static void
warn_for_collisions_1 (tree written, tree writer, struct tlist *list,
int only_writes)
{
struct tlist *tmp;
for (tmp = warned_ids; tmp; tmp = tmp->next)
if (candidate_equal_p (tmp->expr, written))
return;
while (list)
{
if (candidate_equal_p (list->expr, written)
&& !candidate_equal_p (list->writer, writer)
&& (!only_writes || list->writer))
{
warned_ids = new_tlist (warned_ids, written, NULL_TREE);
warning_at (EXPR_LOC_OR_LOC (writer, input_location),
OPT_Wsequence_point, "operation on %qE may be undefined",
list->expr);
}
list = list->next;
}
}
static void
warn_for_collisions (struct tlist *list)
{
struct tlist *tmp;
for (tmp = list; tmp; tmp = tmp->next)
{
if (tmp->writer)
warn_for_collisions_1 (tmp->expr, tmp->writer, list, 0);
}
}
static bool
warning_candidate_p (tree x)
{
if (DECL_P (x) && DECL_ARTIFICIAL (x))
return false;
if (TREE_CODE (x) == BLOCK)
return false;
if (TREE_TYPE (x) == NULL_TREE || VOID_TYPE_P (TREE_TYPE (x)))
return false;
if (!lvalue_p (x))
return false;
if (TREE_CODE (x) == CALL_EXPR && (call_expr_flags (x) & ECF_CONST) == 0)
return false;
if (TREE_CODE (x) == STRING_CST)
return false;
return true;
}
static bool
candidate_equal_p (const_tree x, const_tree y)
{
return (x == y) || (x && y && operand_equal_p (x, y, 0));
}
static void
verify_tree (tree x, struct tlist **pbefore_sp, struct tlist **pno_sp,
tree writer)
{
struct tlist *tmp_before, *tmp_nosp, *tmp_list2, *tmp_list3;
enum tree_code code;
enum tree_code_class cl;
if (x == NULL)
return;
restart:
code = TREE_CODE (x);
cl = TREE_CODE_CLASS (code);
if (warning_candidate_p (x))
*pno_sp = new_tlist (*pno_sp, x, writer);
switch (code)
{
case CONSTRUCTOR:
case SIZEOF_EXPR:
return;
case COMPOUND_EXPR:
case TRUTH_ANDIF_EXPR:
case TRUTH_ORIF_EXPR:
tmp_before = tmp_nosp = tmp_list2 = tmp_list3 = 0;
verify_tree (TREE_OPERAND (x, 0), &tmp_before, &tmp_nosp, NULL_TREE);
warn_for_collisions (tmp_nosp);
merge_tlist (pbefore_sp, tmp_before, 0);
merge_tlist (pbefore_sp, tmp_nosp, 0);
verify_tree (TREE_OPERAND (x, 1), &tmp_list3, &tmp_list2, NULL_TREE);
warn_for_collisions (tmp_list2);
merge_tlist (pbefore_sp, tmp_list3, 0);
merge_tlist (pno_sp, tmp_list2, 0);
return;
case COND_EXPR:
tmp_before = tmp_list2 = 0;
verify_tree (TREE_OPERAND (x, 0), &tmp_before, &tmp_list2, NULL_TREE);
warn_for_collisions (tmp_list2);
merge_tlist (pbefore_sp, tmp_before, 0);
merge_tlist (pbefore_sp, tmp_list2, 0);
tmp_list3 = tmp_nosp = 0;
verify_tree (TREE_OPERAND (x, 1), &tmp_list3, &tmp_nosp, NULL_TREE);
warn_for_collisions (tmp_nosp);
merge_tlist (pbefore_sp, tmp_list3, 0);
tmp_list3 = tmp_list2 = 0;
verify_tree (TREE_OPERAND (x, 2), &tmp_list3, &tmp_list2, NULL_TREE);
warn_for_collisions (tmp_list2);
merge_tlist (pbefore_sp, tmp_list3, 0);
merge_tlist (&tmp_nosp, tmp_list2, 0);
add_tlist (pno_sp, tmp_nosp, NULL_TREE, 0);
return;
case PREDECREMENT_EXPR:
case PREINCREMENT_EXPR:
case POSTDECREMENT_EXPR:
case POSTINCREMENT_EXPR:
verify_tree (TREE_OPERAND (x, 0), pno_sp, pno_sp, x);
return;
case MODIFY_EXPR:
tmp_before = tmp_nosp = tmp_list3 = 0;
verify_tree (TREE_OPERAND (x, 1), &tmp_before, &tmp_nosp, NULL_TREE);
verify_tree (TREE_OPERAND (x, 0), &tmp_list3, &tmp_list3, x);
add_tlist (&tmp_before, tmp_list3, x, 1);
warn_for_collisions (tmp_before);
add_tlist (pno_sp, tmp_list3, x, 0);
warn_for_collisions_1 (TREE_OPERAND (x, 0), x, tmp_nosp, 1);
merge_tlist (pbefore_sp, tmp_before, 0);
if (warning_candidate_p (TREE_OPERAND (x, 0)))
merge_tlist (&tmp_nosp, new_tlist (NULL, TREE_OPERAND (x, 0), x), 0);
add_tlist (pno_sp, tmp_nosp, NULL_TREE, 1);
return;
case CALL_EXPR:
{
call_expr_arg_iterator iter;
tree arg;
tmp_before = tmp_nosp = 0;
verify_tree (CALL_EXPR_FN (x), &tmp_before, &tmp_nosp, NULL_TREE);
FOR_EACH_CALL_EXPR_ARG (arg, iter, x)
{
tmp_list2 = tmp_list3 = 0;
verify_tree (arg, &tmp_list2, &tmp_list3, NULL_TREE);
merge_tlist (&tmp_list3, tmp_list2, 0);
add_tlist (&tmp_before, tmp_list3, NULL_TREE, 0);
}
add_tlist (&tmp_before, tmp_nosp, NULL_TREE, 0);
warn_for_collisions (tmp_before);
add_tlist (pbefore_sp, tmp_before, NULL_TREE, 0);
return;
}
case TREE_LIST:
while (x)
{
tmp_before = tmp_nosp = 0;
verify_tree (TREE_VALUE (x), &tmp_before, &tmp_nosp, NULL_TREE);
merge_tlist (&tmp_nosp, tmp_before, 0);
add_tlist (pno_sp, tmp_nosp, NULL_TREE, 0);
x = TREE_CHAIN (x);
}
return;
case SAVE_EXPR:
{
struct tlist_cache *t;
for (t = save_expr_cache; t; t = t->next)
if (candidate_equal_p (t->expr, x))
break;
if (!t)
{
t = XOBNEW (&tlist_obstack, struct tlist_cache);
t->next = save_expr_cache;
t->expr = x;
save_expr_cache = t;
tmp_before = tmp_nosp = 0;
verify_tree (TREE_OPERAND (x, 0), &tmp_before, &tmp_nosp, NULL_TREE);
warn_for_collisions (tmp_nosp);
tmp_list3 = 0;
merge_tlist (&tmp_list3, tmp_nosp, 0);
t->cache_before_sp = tmp_before;
t->cache_after_sp = tmp_list3;
}
merge_tlist (pbefore_sp, t->cache_before_sp, 1);
add_tlist (pno_sp, t->cache_after_sp, NULL_TREE, 1);
return;
}
case ADDR_EXPR:
x = TREE_OPERAND (x, 0);
if (DECL_P (x))
return;
writer = 0;
goto restart;
default:
if (cl == tcc_unary)
{
x = TREE_OPERAND (x, 0);
writer = 0;
goto restart;
}
else if (IS_EXPR_CODE_CLASS (cl))
{
int lp;
int max = TREE_OPERAND_LENGTH (x);
for (lp = 0; lp < max; lp++)
{
tmp_before = tmp_nosp = 0;
verify_tree (TREE_OPERAND (x, lp), &tmp_before, &tmp_nosp, 0);
merge_tlist (&tmp_nosp, tmp_before, 0);
add_tlist (pno_sp, tmp_nosp, NULL_TREE, 0);
}
}
return;
}
}
DEBUG_FUNCTION void
verify_sequence_points (tree expr)
{
struct tlist *before_sp = 0, *after_sp = 0;
warned_ids = 0;
save_expr_cache = 0;
if (tlist_firstobj == 0)
{
gcc_obstack_init (&tlist_obstack);
tlist_firstobj = (char *) obstack_alloc (&tlist_obstack, 0);
}
verify_tree (expr, &before_sp, &after_sp, 0);
warn_for_collisions (after_sp);
obstack_free (&tlist_obstack, tlist_firstobj);
}

static tree
check_case_value (location_t loc, tree value)
{
if (value == NULL_TREE)
return value;
if (TREE_CODE (value) == INTEGER_CST)
value = perform_integral_promotions (value);
else if (value != error_mark_node)
{
error_at (loc, "case label does not reduce to an integer constant");
value = error_mark_node;
}
constant_expression_warning (value);
return value;
}

static bool
check_case_bounds (location_t loc, tree type, tree orig_type,
tree *case_low_p, tree *case_high_p,
bool *outside_range_p)
{
tree min_value, max_value;
tree case_low = *case_low_p;
tree case_high = case_high_p ? *case_high_p : case_low;
if (orig_type == error_mark_node)
return true;
min_value = TYPE_MIN_VALUE (orig_type);
max_value = TYPE_MAX_VALUE (orig_type);
case_low = fold (case_low);
case_high = fold (case_high);
if (tree_int_cst_compare (case_low, min_value) < 0
&& tree_int_cst_compare (case_high, min_value) < 0)
{
warning_at (loc, 0, "case label value is less than minimum value "
"for type");
*outside_range_p = true;
return false;
}
if (tree_int_cst_compare (case_low, max_value) > 0
&& tree_int_cst_compare (case_high, max_value) > 0)
{
warning_at (loc, 0, "case label value exceeds maximum value for type");
*outside_range_p = true;
return false;
}
if (tree_int_cst_compare (case_high, min_value) >= 0
&& tree_int_cst_compare (case_low, min_value) < 0)
{
warning_at (loc, 0, "lower value in case label range"
" less than minimum value for type");
*outside_range_p = true;
case_low = min_value;
}
if (tree_int_cst_compare (case_low, max_value) <= 0
&& tree_int_cst_compare (case_high, max_value) > 0)
{
warning_at (loc, 0, "upper value in case label range"
" exceeds maximum value for type");
*outside_range_p = true;
case_high = max_value;
}
if (*case_low_p != case_low)
*case_low_p = convert (type, case_low);
if (case_high_p && *case_high_p != case_high)
*case_high_p = convert (type, case_high);
return true;
}

tree
c_common_type_for_size (unsigned int bits, int unsignedp)
{
int i;
if (bits == TYPE_PRECISION (integer_type_node))
return unsignedp ? unsigned_type_node : integer_type_node;
if (bits == TYPE_PRECISION (signed_char_type_node))
return unsignedp ? unsigned_char_type_node : signed_char_type_node;
if (bits == TYPE_PRECISION (short_integer_type_node))
return unsignedp ? short_unsigned_type_node : short_integer_type_node;
if (bits == TYPE_PRECISION (long_integer_type_node))
return unsignedp ? long_unsigned_type_node : long_integer_type_node;
if (bits == TYPE_PRECISION (long_long_integer_type_node))
return (unsignedp ? long_long_unsigned_type_node
: long_long_integer_type_node);
for (i = 0; i < NUM_INT_N_ENTS; i ++)
if (int_n_enabled_p[i]
&& bits == int_n_data[i].bitsize)
return (unsignedp ? int_n_trees[i].unsigned_type
: int_n_trees[i].signed_type);
if (bits == TYPE_PRECISION (widest_integer_literal_type_node))
return (unsignedp ? widest_unsigned_literal_type_node
: widest_integer_literal_type_node);
if (bits <= TYPE_PRECISION (intQI_type_node))
return unsignedp ? unsigned_intQI_type_node : intQI_type_node;
if (bits <= TYPE_PRECISION (intHI_type_node))
return unsignedp ? unsigned_intHI_type_node : intHI_type_node;
if (bits <= TYPE_PRECISION (intSI_type_node))
return unsignedp ? unsigned_intSI_type_node : intSI_type_node;
if (bits <= TYPE_PRECISION (intDI_type_node))
return unsignedp ? unsigned_intDI_type_node : intDI_type_node;
return NULL_TREE;
}
tree
c_common_fixed_point_type_for_size (unsigned int ibit, unsigned int fbit,
int unsignedp, int satp)
{
enum mode_class mclass;
if (ibit == 0)
mclass = unsignedp ? MODE_UFRACT : MODE_FRACT;
else
mclass = unsignedp ? MODE_UACCUM : MODE_ACCUM;
opt_scalar_mode opt_mode;
scalar_mode mode;
FOR_EACH_MODE_IN_CLASS (opt_mode, mclass)
{
mode = opt_mode.require ();
if (GET_MODE_IBIT (mode) >= ibit && GET_MODE_FBIT (mode) >= fbit)
break;
}
if (!opt_mode.exists (&mode) || !targetm.scalar_mode_supported_p (mode))
{
sorry ("GCC cannot support operators with integer types and "
"fixed-point types that have too many integral and "
"fractional bits together");
return NULL_TREE;
}
return c_common_type_for_mode (mode, satp);
}
tree registered_builtin_types;
tree
c_common_type_for_mode (machine_mode mode, int unsignedp)
{
tree t;
int i;
if (mode == TYPE_MODE (integer_type_node))
return unsignedp ? unsigned_type_node : integer_type_node;
if (mode == TYPE_MODE (signed_char_type_node))
return unsignedp ? unsigned_char_type_node : signed_char_type_node;
if (mode == TYPE_MODE (short_integer_type_node))
return unsignedp ? short_unsigned_type_node : short_integer_type_node;
if (mode == TYPE_MODE (long_integer_type_node))
return unsignedp ? long_unsigned_type_node : long_integer_type_node;
if (mode == TYPE_MODE (long_long_integer_type_node))
return unsignedp ? long_long_unsigned_type_node : long_long_integer_type_node;
for (i = 0; i < NUM_INT_N_ENTS; i ++)
if (int_n_enabled_p[i]
&& mode == int_n_data[i].m)
return (unsignedp ? int_n_trees[i].unsigned_type
: int_n_trees[i].signed_type);
if (mode == QImode)
return unsignedp ? unsigned_intQI_type_node : intQI_type_node;
if (mode == HImode)
return unsignedp ? unsigned_intHI_type_node : intHI_type_node;
if (mode == SImode)
return unsignedp ? unsigned_intSI_type_node : intSI_type_node;
if (mode == DImode)
return unsignedp ? unsigned_intDI_type_node : intDI_type_node;
#if HOST_BITS_PER_WIDE_INT >= 64
if (mode == TYPE_MODE (intTI_type_node))
return unsignedp ? unsigned_intTI_type_node : intTI_type_node;
#endif
if (mode == TYPE_MODE (float_type_node))
return float_type_node;
if (mode == TYPE_MODE (double_type_node))
return double_type_node;
if (mode == TYPE_MODE (long_double_type_node))
return long_double_type_node;
for (i = 0; i < NUM_FLOATN_NX_TYPES; i++)
if (FLOATN_NX_TYPE_NODE (i) != NULL_TREE
&& mode == TYPE_MODE (FLOATN_NX_TYPE_NODE (i)))
return FLOATN_NX_TYPE_NODE (i);
if (mode == TYPE_MODE (void_type_node))
return void_type_node;
if (mode == TYPE_MODE (build_pointer_type (char_type_node))
|| mode == TYPE_MODE (build_pointer_type (integer_type_node)))
{
unsigned int precision
= GET_MODE_PRECISION (as_a <scalar_int_mode> (mode));
return (unsignedp
? make_unsigned_type (precision)
: make_signed_type (precision));
}
if (COMPLEX_MODE_P (mode))
{
machine_mode inner_mode;
tree inner_type;
if (mode == TYPE_MODE (complex_float_type_node))
return complex_float_type_node;
if (mode == TYPE_MODE (complex_double_type_node))
return complex_double_type_node;
if (mode == TYPE_MODE (complex_long_double_type_node))
return complex_long_double_type_node;
for (i = 0; i < NUM_FLOATN_NX_TYPES; i++)
if (COMPLEX_FLOATN_NX_TYPE_NODE (i) != NULL_TREE
&& mode == TYPE_MODE (COMPLEX_FLOATN_NX_TYPE_NODE (i)))
return COMPLEX_FLOATN_NX_TYPE_NODE (i);
if (mode == TYPE_MODE (complex_integer_type_node) && !unsignedp)
return complex_integer_type_node;
inner_mode = GET_MODE_INNER (mode);
inner_type = c_common_type_for_mode (inner_mode, unsignedp);
if (inner_type != NULL_TREE)
return build_complex_type (inner_type);
}
else if (GET_MODE_CLASS (mode) == MODE_VECTOR_BOOL
&& valid_vector_subparts_p (GET_MODE_NUNITS (mode)))
{
unsigned int elem_bits = vector_element_size (GET_MODE_BITSIZE (mode),
GET_MODE_NUNITS (mode));
tree bool_type = build_nonstandard_boolean_type (elem_bits);
return build_vector_type_for_mode (bool_type, mode);
}
else if (VECTOR_MODE_P (mode)
&& valid_vector_subparts_p (GET_MODE_NUNITS (mode)))
{
machine_mode inner_mode = GET_MODE_INNER (mode);
tree inner_type = c_common_type_for_mode (inner_mode, unsignedp);
if (inner_type != NULL_TREE)
return build_vector_type_for_mode (inner_type, mode);
}
if (mode == TYPE_MODE (dfloat32_type_node))
return dfloat32_type_node;
if (mode == TYPE_MODE (dfloat64_type_node))
return dfloat64_type_node;
if (mode == TYPE_MODE (dfloat128_type_node))
return dfloat128_type_node;
if (ALL_SCALAR_FIXED_POINT_MODE_P (mode))
{
if (mode == TYPE_MODE (short_fract_type_node))
return unsignedp ? sat_short_fract_type_node : short_fract_type_node;
if (mode == TYPE_MODE (fract_type_node))
return unsignedp ? sat_fract_type_node : fract_type_node;
if (mode == TYPE_MODE (long_fract_type_node))
return unsignedp ? sat_long_fract_type_node : long_fract_type_node;
if (mode == TYPE_MODE (long_long_fract_type_node))
return unsignedp ? sat_long_long_fract_type_node
: long_long_fract_type_node;
if (mode == TYPE_MODE (unsigned_short_fract_type_node))
return unsignedp ? sat_unsigned_short_fract_type_node
: unsigned_short_fract_type_node;
if (mode == TYPE_MODE (unsigned_fract_type_node))
return unsignedp ? sat_unsigned_fract_type_node
: unsigned_fract_type_node;
if (mode == TYPE_MODE (unsigned_long_fract_type_node))
return unsignedp ? sat_unsigned_long_fract_type_node
: unsigned_long_fract_type_node;
if (mode == TYPE_MODE (unsigned_long_long_fract_type_node))
return unsignedp ? sat_unsigned_long_long_fract_type_node
: unsigned_long_long_fract_type_node;
if (mode == TYPE_MODE (short_accum_type_node))
return unsignedp ? sat_short_accum_type_node : short_accum_type_node;
if (mode == TYPE_MODE (accum_type_node))
return unsignedp ? sat_accum_type_node : accum_type_node;
if (mode == TYPE_MODE (long_accum_type_node))
return unsignedp ? sat_long_accum_type_node : long_accum_type_node;
if (mode == TYPE_MODE (long_long_accum_type_node))
return unsignedp ? sat_long_long_accum_type_node
: long_long_accum_type_node;
if (mode == TYPE_MODE (unsigned_short_accum_type_node))
return unsignedp ? sat_unsigned_short_accum_type_node
: unsigned_short_accum_type_node;
if (mode == TYPE_MODE (unsigned_accum_type_node))
return unsignedp ? sat_unsigned_accum_type_node
: unsigned_accum_type_node;
if (mode == TYPE_MODE (unsigned_long_accum_type_node))
return unsignedp ? sat_unsigned_long_accum_type_node
: unsigned_long_accum_type_node;
if (mode == TYPE_MODE (unsigned_long_long_accum_type_node))
return unsignedp ? sat_unsigned_long_long_accum_type_node
: unsigned_long_long_accum_type_node;
if (mode == QQmode)
return unsignedp ? sat_qq_type_node : qq_type_node;
if (mode == HQmode)
return unsignedp ? sat_hq_type_node : hq_type_node;
if (mode == SQmode)
return unsignedp ? sat_sq_type_node : sq_type_node;
if (mode == DQmode)
return unsignedp ? sat_dq_type_node : dq_type_node;
if (mode == TQmode)
return unsignedp ? sat_tq_type_node : tq_type_node;
if (mode == UQQmode)
return unsignedp ? sat_uqq_type_node : uqq_type_node;
if (mode == UHQmode)
return unsignedp ? sat_uhq_type_node : uhq_type_node;
if (mode == USQmode)
return unsignedp ? sat_usq_type_node : usq_type_node;
if (mode == UDQmode)
return unsignedp ? sat_udq_type_node : udq_type_node;
if (mode == UTQmode)
return unsignedp ? sat_utq_type_node : utq_type_node;
if (mode == HAmode)
return unsignedp ? sat_ha_type_node : ha_type_node;
if (mode == SAmode)
return unsignedp ? sat_sa_type_node : sa_type_node;
if (mode == DAmode)
return unsignedp ? sat_da_type_node : da_type_node;
if (mode == TAmode)
return unsignedp ? sat_ta_type_node : ta_type_node;
if (mode == UHAmode)
return unsignedp ? sat_uha_type_node : uha_type_node;
if (mode == USAmode)
return unsignedp ? sat_usa_type_node : usa_type_node;
if (mode == UDAmode)
return unsignedp ? sat_uda_type_node : uda_type_node;
if (mode == UTAmode)
return unsignedp ? sat_uta_type_node : uta_type_node;
}
for (t = registered_builtin_types; t; t = TREE_CHAIN (t))
if (TYPE_MODE (TREE_VALUE (t)) == mode
&& !!unsignedp == !!TYPE_UNSIGNED (TREE_VALUE (t)))
return TREE_VALUE (t);
return NULL_TREE;
}
tree
c_common_unsigned_type (tree type)
{
return c_common_signed_or_unsigned_type (1, type);
}
tree
c_common_signed_type (tree type)
{
return c_common_signed_or_unsigned_type (0, type);
}
tree
c_common_signed_or_unsigned_type (int unsignedp, tree type)
{
tree type1;
int i;
type1 = TYPE_MAIN_VARIANT (type);
if (type1 == signed_char_type_node || type1 == char_type_node || type1 == unsigned_char_type_node)
return unsignedp ? unsigned_char_type_node : signed_char_type_node;
if (type1 == integer_type_node || type1 == unsigned_type_node)
return unsignedp ? unsigned_type_node : integer_type_node;
if (type1 == short_integer_type_node || type1 == short_unsigned_type_node)
return unsignedp ? short_unsigned_type_node : short_integer_type_node;
if (type1 == long_integer_type_node || type1 == long_unsigned_type_node)
return unsignedp ? long_unsigned_type_node : long_integer_type_node;
if (type1 == long_long_integer_type_node || type1 == long_long_unsigned_type_node)
return unsignedp ? long_long_unsigned_type_node : long_long_integer_type_node;
for (i = 0; i < NUM_INT_N_ENTS; i ++)
if (int_n_enabled_p[i]
&& (type1 == int_n_trees[i].unsigned_type
|| type1 == int_n_trees[i].signed_type))
return (unsignedp ? int_n_trees[i].unsigned_type
: int_n_trees[i].signed_type);
#if HOST_BITS_PER_WIDE_INT >= 64
if (type1 == intTI_type_node || type1 == unsigned_intTI_type_node)
return unsignedp ? unsigned_intTI_type_node : intTI_type_node;
#endif
if (type1 == intDI_type_node || type1 == unsigned_intDI_type_node)
return unsignedp ? unsigned_intDI_type_node : intDI_type_node;
if (type1 == intSI_type_node || type1 == unsigned_intSI_type_node)
return unsignedp ? unsigned_intSI_type_node : intSI_type_node;
if (type1 == intHI_type_node || type1 == unsigned_intHI_type_node)
return unsignedp ? unsigned_intHI_type_node : intHI_type_node;
if (type1 == intQI_type_node || type1 == unsigned_intQI_type_node)
return unsignedp ? unsigned_intQI_type_node : intQI_type_node;
#define C_COMMON_FIXED_TYPES(NAME)	    \
if (type1 == short_ ## NAME ## _type_node \
|| type1 == unsigned_short_ ## NAME ## _type_node) \
return unsignedp ? unsigned_short_ ## NAME ## _type_node \
: short_ ## NAME ## _type_node; \
if (type1 == NAME ## _type_node \
|| type1 == unsigned_ ## NAME ## _type_node) \
return unsignedp ? unsigned_ ## NAME ## _type_node \
: NAME ## _type_node; \
if (type1 == long_ ## NAME ## _type_node \
|| type1 == unsigned_long_ ## NAME ## _type_node) \
return unsignedp ? unsigned_long_ ## NAME ## _type_node \
: long_ ## NAME ## _type_node; \
if (type1 == long_long_ ## NAME ## _type_node \
|| type1 == unsigned_long_long_ ## NAME ## _type_node) \
return unsignedp ? unsigned_long_long_ ## NAME ## _type_node \
: long_long_ ## NAME ## _type_node;
#define C_COMMON_FIXED_MODE_TYPES(NAME) \
if (type1 == NAME ## _type_node \
|| type1 == u ## NAME ## _type_node) \
return unsignedp ? u ## NAME ## _type_node \
: NAME ## _type_node;
#define C_COMMON_FIXED_TYPES_SAT(NAME) \
if (type1 == sat_ ## short_ ## NAME ## _type_node \
|| type1 == sat_ ## unsigned_short_ ## NAME ## _type_node) \
return unsignedp ? sat_ ## unsigned_short_ ## NAME ## _type_node \
: sat_ ## short_ ## NAME ## _type_node; \
if (type1 == sat_ ## NAME ## _type_node \
|| type1 == sat_ ## unsigned_ ## NAME ## _type_node) \
return unsignedp ? sat_ ## unsigned_ ## NAME ## _type_node \
: sat_ ## NAME ## _type_node; \
if (type1 == sat_ ## long_ ## NAME ## _type_node \
|| type1 == sat_ ## unsigned_long_ ## NAME ## _type_node) \
return unsignedp ? sat_ ## unsigned_long_ ## NAME ## _type_node \
: sat_ ## long_ ## NAME ## _type_node; \
if (type1 == sat_ ## long_long_ ## NAME ## _type_node \
|| type1 == sat_ ## unsigned_long_long_ ## NAME ## _type_node) \
return unsignedp ? sat_ ## unsigned_long_long_ ## NAME ## _type_node \
: sat_ ## long_long_ ## NAME ## _type_node;
#define C_COMMON_FIXED_MODE_TYPES_SAT(NAME)	\
if (type1 == sat_ ## NAME ## _type_node \
|| type1 == sat_ ## u ## NAME ## _type_node) \
return unsignedp ? sat_ ## u ## NAME ## _type_node \
: sat_ ## NAME ## _type_node;
C_COMMON_FIXED_TYPES (fract);
C_COMMON_FIXED_TYPES_SAT (fract);
C_COMMON_FIXED_TYPES (accum);
C_COMMON_FIXED_TYPES_SAT (accum);
C_COMMON_FIXED_MODE_TYPES (qq);
C_COMMON_FIXED_MODE_TYPES (hq);
C_COMMON_FIXED_MODE_TYPES (sq);
C_COMMON_FIXED_MODE_TYPES (dq);
C_COMMON_FIXED_MODE_TYPES (tq);
C_COMMON_FIXED_MODE_TYPES_SAT (qq);
C_COMMON_FIXED_MODE_TYPES_SAT (hq);
C_COMMON_FIXED_MODE_TYPES_SAT (sq);
C_COMMON_FIXED_MODE_TYPES_SAT (dq);
C_COMMON_FIXED_MODE_TYPES_SAT (tq);
C_COMMON_FIXED_MODE_TYPES (ha);
C_COMMON_FIXED_MODE_TYPES (sa);
C_COMMON_FIXED_MODE_TYPES (da);
C_COMMON_FIXED_MODE_TYPES (ta);
C_COMMON_FIXED_MODE_TYPES_SAT (ha);
C_COMMON_FIXED_MODE_TYPES_SAT (sa);
C_COMMON_FIXED_MODE_TYPES_SAT (da);
C_COMMON_FIXED_MODE_TYPES_SAT (ta);
if (!INTEGRAL_TYPE_P (type)
|| TYPE_UNSIGNED (type) == unsignedp)
return type;
#define TYPE_OK(node)							    \
(TYPE_MODE (type) == TYPE_MODE (node)					    \
&& TYPE_PRECISION (type) == TYPE_PRECISION (node))
if (TYPE_OK (signed_char_type_node))
return unsignedp ? unsigned_char_type_node : signed_char_type_node;
if (TYPE_OK (integer_type_node))
return unsignedp ? unsigned_type_node : integer_type_node;
if (TYPE_OK (short_integer_type_node))
return unsignedp ? short_unsigned_type_node : short_integer_type_node;
if (TYPE_OK (long_integer_type_node))
return unsignedp ? long_unsigned_type_node : long_integer_type_node;
if (TYPE_OK (long_long_integer_type_node))
return (unsignedp ? long_long_unsigned_type_node
: long_long_integer_type_node);
for (i = 0; i < NUM_INT_N_ENTS; i ++)
if (int_n_enabled_p[i]
&& TYPE_MODE (type) == int_n_data[i].m
&& TYPE_PRECISION (type) == int_n_data[i].bitsize)
return (unsignedp ? int_n_trees[i].unsigned_type
: int_n_trees[i].signed_type);
#if HOST_BITS_PER_WIDE_INT >= 64
if (TYPE_OK (intTI_type_node))
return unsignedp ? unsigned_intTI_type_node : intTI_type_node;
#endif
if (TYPE_OK (intDI_type_node))
return unsignedp ? unsigned_intDI_type_node : intDI_type_node;
if (TYPE_OK (intSI_type_node))
return unsignedp ? unsigned_intSI_type_node : intSI_type_node;
if (TYPE_OK (intHI_type_node))
return unsignedp ? unsigned_intHI_type_node : intHI_type_node;
if (TYPE_OK (intQI_type_node))
return unsignedp ? unsigned_intQI_type_node : intQI_type_node;
#undef TYPE_OK
return build_nonstandard_integer_type (TYPE_PRECISION (type), unsignedp);
}
tree
c_build_bitfield_integer_type (unsigned HOST_WIDE_INT width, int unsignedp)
{
int i;
if (width == TYPE_PRECISION (integer_type_node))
return unsignedp ? unsigned_type_node : integer_type_node;
if (width == TYPE_PRECISION (signed_char_type_node))
return unsignedp ? unsigned_char_type_node : signed_char_type_node;
if (width == TYPE_PRECISION (short_integer_type_node))
return unsignedp ? short_unsigned_type_node : short_integer_type_node;
if (width == TYPE_PRECISION (long_integer_type_node))
return unsignedp ? long_unsigned_type_node : long_integer_type_node;
if (width == TYPE_PRECISION (long_long_integer_type_node))
return (unsignedp ? long_long_unsigned_type_node
: long_long_integer_type_node);
for (i = 0; i < NUM_INT_N_ENTS; i ++)
if (int_n_enabled_p[i]
&& width == int_n_data[i].bitsize)
return (unsignedp ? int_n_trees[i].unsigned_type
: int_n_trees[i].signed_type);
return build_nonstandard_integer_type (width, unsignedp);
}
void
c_register_builtin_type (tree type, const char* name)
{
tree decl;
decl = build_decl (UNKNOWN_LOCATION,
TYPE_DECL, get_identifier (name), type);
DECL_ARTIFICIAL (decl) = 1;
if (!TYPE_NAME (type))
TYPE_NAME (type) = decl;
lang_hooks.decls.pushdecl (decl);
registered_builtin_types = tree_cons (0, type, registered_builtin_types);
}

void
binary_op_error (rich_location *richloc, enum tree_code code,
tree type0, tree type1)
{
const char *opname;
switch (code)
{
case PLUS_EXPR:
opname = "+"; break;
case MINUS_EXPR:
opname = "-"; break;
case MULT_EXPR:
opname = "*"; break;
case MAX_EXPR:
opname = "max"; break;
case MIN_EXPR:
opname = "min"; break;
case EQ_EXPR:
opname = "=="; break;
case NE_EXPR:
opname = "!="; break;
case LE_EXPR:
opname = "<="; break;
case GE_EXPR:
opname = ">="; break;
case LT_EXPR:
opname = "<"; break;
case GT_EXPR:
opname = ">"; break;
case LSHIFT_EXPR:
opname = "<<"; break;
case RSHIFT_EXPR:
opname = ">>"; break;
case TRUNC_MOD_EXPR:
case FLOOR_MOD_EXPR:
opname = "%"; break;
case TRUNC_DIV_EXPR:
case FLOOR_DIV_EXPR:
opname = "/"; break;
case BIT_AND_EXPR:
opname = "&"; break;
case BIT_IOR_EXPR:
opname = "|"; break;
case TRUTH_ANDIF_EXPR:
opname = "&&"; break;
case TRUTH_ORIF_EXPR:
opname = "||"; break;
case BIT_XOR_EXPR:
opname = "^"; break;
default:
gcc_unreachable ();
}
error_at (richloc,
"invalid operands to binary %s (have %qT and %qT)",
opname, type0, type1);
}

static tree
expr_original_type (tree expr)
{
STRIP_SIGN_NOPS (expr);
return TREE_TYPE (expr);
}
tree
shorten_compare (location_t loc, tree *op0_ptr, tree *op1_ptr,
tree *restype_ptr, enum tree_code *rescode_ptr)
{
tree type;
tree op0 = *op0_ptr;
tree op1 = *op1_ptr;
int unsignedp0, unsignedp1;
int real1, real2;
tree primop0, primop1;
enum tree_code code = *rescode_ptr;
primop0 = c_common_get_narrower (op0, &unsignedp0);
primop1 = c_common_get_narrower (op1, &unsignedp1);
if (TYPE_PRECISION (TREE_TYPE (primop0)) < TYPE_PRECISION (TREE_TYPE (op0))
&& TYPE_PRECISION (TREE_TYPE (op0)) < TYPE_PRECISION (*restype_ptr)
&& !unsignedp0
&& TYPE_UNSIGNED (TREE_TYPE (op0)))
primop0 = op0;
if (TYPE_PRECISION (TREE_TYPE (primop1)) < TYPE_PRECISION (TREE_TYPE (op1))
&& TYPE_PRECISION (TREE_TYPE (op1)) < TYPE_PRECISION (*restype_ptr)
&& !unsignedp1
&& TYPE_UNSIGNED (TREE_TYPE (op1)))
primop1 = op1;
if (op0 == primop0 && TREE_TYPE (op0) != *restype_ptr)
unsignedp0 = TYPE_UNSIGNED (TREE_TYPE (op0));
if (op1 == primop1 && TREE_TYPE (op1) != *restype_ptr)
unsignedp1 = TYPE_UNSIGNED (TREE_TYPE (op1));
real1 = TREE_CODE (TREE_TYPE (primop0)) == REAL_TYPE;
real2 = TREE_CODE (TREE_TYPE (primop1)) == REAL_TYPE;
if (TREE_CONSTANT (primop0)
&& !integer_zerop (primop1) && !real_zerop (primop1)
&& !fixed_zerop (primop1))
{
std::swap (primop0, primop1);
std::swap (op0, op1);
*op0_ptr = op0;
*op1_ptr = op1;
std::swap (unsignedp0, unsignedp1);
std::swap (real1, real2);
switch (code)
{
case LT_EXPR:
code = GT_EXPR;
break;
case GT_EXPR:
code = LT_EXPR;
break;
case LE_EXPR:
code = GE_EXPR;
break;
case GE_EXPR:
code = LE_EXPR;
break;
default:
break;
}
*rescode_ptr = code;
}
if (!real1 && !real2
&& TREE_CODE (TREE_TYPE (primop0)) != FIXED_POINT_TYPE
&& TREE_CODE (primop1) == INTEGER_CST
&& TYPE_PRECISION (TREE_TYPE (primop0)) < TYPE_PRECISION (*restype_ptr))
{
int min_gt, max_gt, min_lt, max_lt;
tree maxval, minval;
int unsignedp = TYPE_UNSIGNED (*restype_ptr);
tree val;
type = c_common_signed_or_unsigned_type (unsignedp0,
TREE_TYPE (primop0));
maxval = TYPE_MAX_VALUE (type);
minval = TYPE_MIN_VALUE (type);
if (unsignedp && !unsignedp0)
*restype_ptr = c_common_signed_type (*restype_ptr);
if (TREE_TYPE (primop1) != *restype_ptr)
{
primop1 = force_fit_type (*restype_ptr,
wi::to_wide
(primop1,
TYPE_PRECISION (*restype_ptr)),
0, TREE_OVERFLOW (primop1));
}
if (type != *restype_ptr)
{
minval = convert (*restype_ptr, minval);
maxval = convert (*restype_ptr, maxval);
}
min_gt = tree_int_cst_lt (primop1, minval);
max_gt = tree_int_cst_lt (primop1, maxval);
min_lt = tree_int_cst_lt (minval, primop1);
max_lt = tree_int_cst_lt (maxval, primop1);
val = 0;
if (code == NE_EXPR)
{
if (max_lt || min_gt)
val = truthvalue_true_node;
}
else if (code == EQ_EXPR)
{
if (max_lt || min_gt)
val = truthvalue_false_node;
}
else if (code == LT_EXPR)
{
if (max_lt)
val = truthvalue_true_node;
if (!min_lt)
val = truthvalue_false_node;
}
else if (code == GT_EXPR)
{
if (min_gt)
val = truthvalue_true_node;
if (!max_gt)
val = truthvalue_false_node;
}
else if (code == LE_EXPR)
{
if (!max_gt)
val = truthvalue_true_node;
if (min_gt)
val = truthvalue_false_node;
}
else if (code == GE_EXPR)
{
if (!min_lt)
val = truthvalue_true_node;
if (max_lt)
val = truthvalue_false_node;
}
if (unsignedp && !unsignedp0)
{
if (val != 0)
switch (code)
{
case LT_EXPR:
case GE_EXPR:
primop1 = TYPE_MIN_VALUE (type);
val = 0;
break;
case LE_EXPR:
case GT_EXPR:
primop1 = TYPE_MAX_VALUE (type);
val = 0;
break;
default:
break;
}
type = c_common_unsigned_type (type);
}
if (TREE_CODE (primop0) != INTEGER_CST
&& !(from_macro_expansion_at
(expansion_point_location_if_in_system_header
(EXPR_LOCATION (primop0)))))
{
if (val == truthvalue_false_node)
warning_at (loc, OPT_Wtype_limits,
"comparison is always false due to limited range of data type");
if (val == truthvalue_true_node)
warning_at (loc, OPT_Wtype_limits,
"comparison is always true due to limited range of data type");
}
if (val != 0)
{
if (TREE_SIDE_EFFECTS (primop0))
return build2 (COMPOUND_EXPR, TREE_TYPE (val), primop0, val);
return val;
}
}
else if (real1 && real2
&& DECIMAL_FLOAT_MODE_P (TYPE_MODE (TREE_TYPE (primop0)))
&& DECIMAL_FLOAT_MODE_P (TYPE_MODE (TREE_TYPE (primop1))))
type = common_type (TREE_TYPE (primop0), TREE_TYPE (primop1));
else if (real1 && real2
&& (DECIMAL_FLOAT_MODE_P (TYPE_MODE (TREE_TYPE (primop0)))
|| DECIMAL_FLOAT_MODE_P (TYPE_MODE (TREE_TYPE (primop1)))))
return NULL_TREE;
else if (real1 && real2
&& (TYPE_PRECISION (TREE_TYPE (primop0))
== TYPE_PRECISION (TREE_TYPE (primop1))))
type = TREE_TYPE (primop0);
else if (unsignedp0 == unsignedp1 && real1 == real2
&& TYPE_PRECISION (TREE_TYPE (primop0)) < TYPE_PRECISION (*restype_ptr)
&& TYPE_PRECISION (TREE_TYPE (primop1)) < TYPE_PRECISION (*restype_ptr))
{
type = common_type (TREE_TYPE (primop0), TREE_TYPE (primop1));
type = c_common_signed_or_unsigned_type (unsignedp0
|| TYPE_UNSIGNED (*restype_ptr),
type);
primop0
= convert (c_common_signed_or_unsigned_type (unsignedp0,
TREE_TYPE (primop0)),
primop0);
primop1
= convert (c_common_signed_or_unsigned_type (unsignedp1,
TREE_TYPE (primop1)),
primop1);
}
else
{
type = *restype_ptr;
primop0 = op0;
primop1 = op1;
if (!real1 && !real2 && integer_zerop (primop1)
&& TYPE_UNSIGNED (*restype_ptr))
{
tree value = NULL_TREE;
bool warn = 
warn_type_limits && !in_system_header_at (loc)
&& !(TREE_CODE (primop0) == INTEGER_CST
&& !TREE_OVERFLOW (convert (c_common_signed_type (type),
primop0)))
&& (TREE_CODE (expr_original_type (primop0)) != ENUMERAL_TYPE);
switch (code)
{
case GE_EXPR:
if (warn)
warning_at (loc, OPT_Wtype_limits,
"comparison of unsigned expression >= 0 is always true");
value = truthvalue_true_node;
break;
case LT_EXPR:
if (warn)
warning_at (loc, OPT_Wtype_limits,
"comparison of unsigned expression < 0 is always false");
value = truthvalue_false_node;
break;
default:
break;
}
if (value != NULL_TREE)
{
if (TREE_SIDE_EFFECTS (primop0))
return build2 (COMPOUND_EXPR, TREE_TYPE (value),
primop0, value);
return value;
}
}
}
*op0_ptr = convert (type, primop0);
*op1_ptr = convert (type, primop1);
*restype_ptr = truthvalue_type_node;
return NULL_TREE;
}

tree
pointer_int_sum (location_t loc, enum tree_code resultcode,
tree ptrop, tree intop, bool complain)
{
tree size_exp, ret;
tree result_type = TREE_TYPE (ptrop);
if (TREE_CODE (TREE_TYPE (result_type)) == VOID_TYPE)
{
if (complain && warn_pointer_arith)
pedwarn (loc, OPT_Wpointer_arith,
"pointer of type %<void *%> used in arithmetic");
else if (!complain)
return error_mark_node;
size_exp = integer_one_node;
}
else if (TREE_CODE (TREE_TYPE (result_type)) == FUNCTION_TYPE)
{
if (complain && warn_pointer_arith)
pedwarn (loc, OPT_Wpointer_arith,
"pointer to a function used in arithmetic");
else if (!complain)
return error_mark_node;
size_exp = integer_one_node;
}
else
size_exp = size_in_bytes_loc (loc, TREE_TYPE (result_type));
fold_defer_overflow_warnings ();
if ((TREE_CODE (intop) == PLUS_EXPR || TREE_CODE (intop) == MINUS_EXPR)
&& !TREE_CONSTANT (intop)
&& TREE_CONSTANT (TREE_OPERAND (intop, 1))
&& TREE_CONSTANT (size_exp)
&& TREE_CODE (TREE_TYPE (TREE_OPERAND (intop, 0))) == INTEGER_TYPE
&& (!TYPE_UNSIGNED (TREE_TYPE (intop))
|| (TYPE_PRECISION (TREE_TYPE (intop))
== TYPE_PRECISION (TREE_TYPE (ptrop)))))
{
enum tree_code subcode = resultcode;
tree int_type = TREE_TYPE (intop);
if (TREE_CODE (intop) == MINUS_EXPR)
subcode = (subcode == PLUS_EXPR ? MINUS_EXPR : PLUS_EXPR);
ptrop = build_binary_op (EXPR_LOCATION (TREE_OPERAND (intop, 1)),
subcode, ptrop,
convert (int_type, TREE_OPERAND (intop, 1)),
true);
intop = convert (int_type, TREE_OPERAND (intop, 0));
}
if (TYPE_PRECISION (TREE_TYPE (intop)) != TYPE_PRECISION (sizetype)
|| TYPE_UNSIGNED (TREE_TYPE (intop)) != TYPE_UNSIGNED (sizetype))
intop = convert (c_common_type_for_size (TYPE_PRECISION (sizetype),
TYPE_UNSIGNED (sizetype)), intop);
{
tree t = fold_build2_loc (loc, MULT_EXPR, TREE_TYPE (intop), intop,
convert (TREE_TYPE (intop), size_exp));
intop = convert (sizetype, t);
if (TREE_OVERFLOW_P (intop) && !TREE_OVERFLOW (t))
intop = wide_int_to_tree (TREE_TYPE (intop), wi::to_wide (intop));
}
if (resultcode == MINUS_EXPR)
intop = fold_build1_loc (loc, NEGATE_EXPR, sizetype, intop);
ret = fold_build_pointer_plus_loc (loc, ptrop, intop);
fold_undefer_and_ignore_overflow_warnings ();
return ret;
}

tree
c_wrap_maybe_const (tree expr, bool non_const)
{
bool nowarning = TREE_NO_WARNING (expr);
location_t loc = EXPR_LOCATION (expr);
if (c_dialect_cxx ())
gcc_unreachable ();
STRIP_TYPE_NOPS (expr);
expr = build2 (C_MAYBE_CONST_EXPR, TREE_TYPE (expr), NULL, expr);
C_MAYBE_CONST_EXPR_NON_CONST (expr) = non_const;
if (nowarning)
TREE_NO_WARNING (expr) = 1;
protected_set_expr_location (expr, loc);
return expr;
}
bool
decl_with_nonnull_addr_p (const_tree expr)
{
return (DECL_P (expr)
&& (TREE_CODE (expr) == PARM_DECL
|| TREE_CODE (expr) == LABEL_DECL
|| !DECL_WEAK (expr)));
}
tree
c_common_truthvalue_conversion (location_t location, tree expr)
{
switch (TREE_CODE (expr))
{
case EQ_EXPR:   case NE_EXPR:   case UNEQ_EXPR: case LTGT_EXPR:
case LE_EXPR:   case GE_EXPR:   case LT_EXPR:   case GT_EXPR:
case UNLE_EXPR: case UNGE_EXPR: case UNLT_EXPR: case UNGT_EXPR:
case ORDERED_EXPR: case UNORDERED_EXPR:
if (TREE_TYPE (expr) == truthvalue_type_node)
return expr;
expr = build2 (TREE_CODE (expr), truthvalue_type_node,
TREE_OPERAND (expr, 0), TREE_OPERAND (expr, 1));
goto ret;
case TRUTH_ANDIF_EXPR:
case TRUTH_ORIF_EXPR:
case TRUTH_AND_EXPR:
case TRUTH_OR_EXPR:
case TRUTH_XOR_EXPR:
if (TREE_TYPE (expr) == truthvalue_type_node)
return expr;
expr = build2 (TREE_CODE (expr), truthvalue_type_node,
c_common_truthvalue_conversion (location,
TREE_OPERAND (expr, 0)),
c_common_truthvalue_conversion (location,
TREE_OPERAND (expr, 1)));
goto ret;
case TRUTH_NOT_EXPR:
if (TREE_TYPE (expr) == truthvalue_type_node)
return expr;
expr = build1 (TREE_CODE (expr), truthvalue_type_node,
c_common_truthvalue_conversion (location,
TREE_OPERAND (expr, 0)));
goto ret;
case ERROR_MARK:
return expr;
case INTEGER_CST:
if (TREE_CODE (TREE_TYPE (expr)) == ENUMERAL_TYPE
&& !integer_zerop (expr)
&& !integer_onep (expr))
warning_at (location, OPT_Wint_in_bool_context,
"enum constant in boolean context");
return integer_zerop (expr) ? truthvalue_false_node
: truthvalue_true_node;
case REAL_CST:
return real_compare (NE_EXPR, &TREE_REAL_CST (expr), &dconst0)
? truthvalue_true_node
: truthvalue_false_node;
case FIXED_CST:
return fixed_compare (NE_EXPR, &TREE_FIXED_CST (expr),
&FCONST0 (TYPE_MODE (TREE_TYPE (expr))))
? truthvalue_true_node
: truthvalue_false_node;
case FUNCTION_DECL:
expr = build_unary_op (location, ADDR_EXPR, expr, false);
case ADDR_EXPR:
{
tree inner = TREE_OPERAND (expr, 0);
if (decl_with_nonnull_addr_p (inner))
{
warning_at (location,
OPT_Waddress,
"the address of %qD will always evaluate as %<true%>",
inner);
return truthvalue_true_node;
}
break;
}
case COMPLEX_EXPR:
expr = build_binary_op (EXPR_LOCATION (expr),
(TREE_SIDE_EFFECTS (TREE_OPERAND (expr, 1))
? TRUTH_OR_EXPR : TRUTH_ORIF_EXPR),
c_common_truthvalue_conversion (location,
TREE_OPERAND (expr, 0)),
c_common_truthvalue_conversion (location,
TREE_OPERAND (expr, 1)),
false);
goto ret;
case NEGATE_EXPR:
case ABS_EXPR:
case FLOAT_EXPR:
case EXCESS_PRECISION_EXPR:
return c_common_truthvalue_conversion (location, TREE_OPERAND (expr, 0));
case LROTATE_EXPR:
case RROTATE_EXPR:
if (TREE_SIDE_EFFECTS (TREE_OPERAND (expr, 1)))
{
expr = build2 (COMPOUND_EXPR, truthvalue_type_node,
TREE_OPERAND (expr, 1),
c_common_truthvalue_conversion
(location, TREE_OPERAND (expr, 0)));
goto ret;
}
else
return c_common_truthvalue_conversion (location,
TREE_OPERAND (expr, 0));
case MULT_EXPR:
warning_at (EXPR_LOCATION (expr), OPT_Wint_in_bool_context,
"%<*%> in boolean context, suggest %<&&%> instead");
break;
case LSHIFT_EXPR:
if (TREE_CODE (TREE_TYPE (expr)) == INTEGER_TYPE
&& !TYPE_UNSIGNED (TREE_TYPE (expr)))
warning_at (EXPR_LOCATION (expr), OPT_Wint_in_bool_context,
"%<<<%> in boolean context, did you mean %<<%> ?");
break;
case COND_EXPR:
if (warn_int_in_bool_context
&& !from_macro_definition_at (EXPR_LOCATION (expr)))
{
tree val1 = fold_for_warn (TREE_OPERAND (expr, 1));
tree val2 = fold_for_warn (TREE_OPERAND (expr, 2));
if (TREE_CODE (val1) == INTEGER_CST
&& TREE_CODE (val2) == INTEGER_CST
&& !integer_zerop (val1)
&& !integer_zerop (val2)
&& (!integer_onep (val1)
|| !integer_onep (val2)))
warning_at (EXPR_LOCATION (expr), OPT_Wint_in_bool_context,
"?: using integer constants in boolean context, "
"the expression will always evaluate to %<true%>");
else if ((TREE_CODE (val1) == INTEGER_CST
&& !integer_zerop (val1)
&& !integer_onep (val1))
|| (TREE_CODE (val2) == INTEGER_CST
&& !integer_zerop (val2)
&& !integer_onep (val2)))
warning_at (EXPR_LOCATION (expr), OPT_Wint_in_bool_context,
"?: using integer constants in boolean context");
}
if (c_dialect_cxx ())
break;
else
{
int w = warn_int_in_bool_context;
warn_int_in_bool_context = 0;
expr = build3 (COND_EXPR, truthvalue_type_node,
TREE_OPERAND (expr, 0),
c_common_truthvalue_conversion (location,
TREE_OPERAND (expr, 1)),
c_common_truthvalue_conversion (location,
TREE_OPERAND (expr, 2)));
warn_int_in_bool_context = w;
goto ret;
}
CASE_CONVERT:
{
tree totype = TREE_TYPE (expr);
tree fromtype = TREE_TYPE (TREE_OPERAND (expr, 0));
if (POINTER_TYPE_P (totype)
&& !c_inhibit_evaluation_warnings
&& TREE_CODE (fromtype) == REFERENCE_TYPE)
{
tree inner = expr;
STRIP_NOPS (inner);
if (DECL_P (inner))
warning_at (location,
OPT_Waddress,
"the compiler can assume that the address of "
"%qD will always evaluate to %<true%>",
inner);
}
if (TREE_CODE (totype) == REFERENCE_TYPE
|| TREE_CODE (fromtype) == REFERENCE_TYPE)
break;
if (TREE_CODE (fromtype) == ENUMERAL_TYPE
&& ENUM_IS_SCOPED (fromtype))
break;
if (TYPE_PRECISION (totype) >= TYPE_PRECISION (fromtype))
return c_common_truthvalue_conversion (location,
TREE_OPERAND (expr, 0));
}
break;
case MODIFY_EXPR:
if (!TREE_NO_WARNING (expr)
&& warn_parentheses)
{
warning_at (location, OPT_Wparentheses,
"suggest parentheses around assignment used as "
"truth value");
TREE_NO_WARNING (expr) = 1;
}
break;
default:
break;
}
if (TREE_CODE (TREE_TYPE (expr)) == COMPLEX_TYPE)
{
tree t = save_expr (expr);
expr = (build_binary_op
(EXPR_LOCATION (expr),
(TREE_SIDE_EFFECTS (expr)
? TRUTH_OR_EXPR : TRUTH_ORIF_EXPR),
c_common_truthvalue_conversion
(location,
build_unary_op (location, REALPART_EXPR, t, false)),
c_common_truthvalue_conversion
(location,
build_unary_op (location, IMAGPART_EXPR, t, false)),
false));
goto ret;
}
if (TREE_CODE (TREE_TYPE (expr)) == FIXED_POINT_TYPE)
{
tree fixed_zero_node = build_fixed (TREE_TYPE (expr),
FCONST0 (TYPE_MODE
(TREE_TYPE (expr))));
return build_binary_op (location, NE_EXPR, expr, fixed_zero_node, true);
}
else
return build_binary_op (location, NE_EXPR, expr, integer_zero_node, true);
ret:
protected_set_expr_location (expr, location);
return expr;
}

static void def_builtin_1  (enum built_in_function fncode,
const char *name,
enum built_in_class fnclass,
tree fntype, tree libtype,
bool both_p, bool fallback_p, bool nonansi_p,
tree fnattrs, bool implicit_p);
void
c_apply_type_quals_to_decl (int type_quals, tree decl)
{
tree type = TREE_TYPE (decl);
if (type == error_mark_node)
return;
if ((type_quals & TYPE_QUAL_CONST)
|| (type && TREE_CODE (type) == REFERENCE_TYPE))
TREE_READONLY (decl) = 1;
if (type_quals & TYPE_QUAL_VOLATILE)
{
TREE_SIDE_EFFECTS (decl) = 1;
TREE_THIS_VOLATILE (decl) = 1;
}
if (type_quals & TYPE_QUAL_RESTRICT)
{
while (type && TREE_CODE (type) == ARRAY_TYPE)
type = TREE_TYPE (type);
if (!type
|| !POINTER_TYPE_P (type)
|| !C_TYPE_OBJECT_OR_INCOMPLETE_P (TREE_TYPE (type)))
error ("invalid use of %<restrict%>");
}
}
alias_set_type
c_common_get_alias_set (tree t)
{
if (TYPE_P (t) && TYPE_STRUCTURAL_EQUALITY_P (t))
{
if (TREE_CODE (t) == ARRAY_TYPE)
return get_alias_set (TREE_TYPE (t));
return -1;
}
if (!TYPE_P (t))
return -1;
if (t == char_type_node
|| t == signed_char_type_node
|| t == unsigned_char_type_node)
return 0;
if (TREE_CODE (t) == INTEGER_TYPE && TYPE_UNSIGNED (t))
{
tree t1 = c_common_signed_type (t);
if (t1 != t)
return get_alias_set (t1);
}
return -1;
}

tree
c_sizeof_or_alignof_type (location_t loc,
tree type, bool is_sizeof, bool min_alignof,
int complain)
{
const char *op_name;
tree value = NULL;
enum tree_code type_code = TREE_CODE (type);
op_name = is_sizeof ? "sizeof" : "__alignof__";
if (type_code == FUNCTION_TYPE)
{
if (is_sizeof)
{
if (complain && warn_pointer_arith)
pedwarn (loc, OPT_Wpointer_arith,
"invalid application of %<sizeof%> to a function type");
else if (!complain)
return error_mark_node;
value = size_one_node;
}
else
{
if (complain)
{
if (c_dialect_cxx ())
pedwarn (loc, OPT_Wpedantic, "ISO C++ does not permit "
"%<alignof%> applied to a function type");
else
pedwarn (loc, OPT_Wpedantic, "ISO C does not permit "
"%<_Alignof%> applied to a function type");
}
value = size_int (FUNCTION_BOUNDARY / BITS_PER_UNIT);
}
}
else if (type_code == VOID_TYPE || type_code == ERROR_MARK)
{
if (type_code == VOID_TYPE
&& complain && warn_pointer_arith)
pedwarn (loc, OPT_Wpointer_arith,
"invalid application of %qs to a void type", op_name);
else if (!complain)
return error_mark_node;
value = size_one_node;
}
else if (!COMPLETE_TYPE_P (type)
&& (!c_dialect_cxx () || is_sizeof || type_code != ARRAY_TYPE))
{
if (complain)
error_at (loc, "invalid application of %qs to incomplete type %qT",
op_name, type);
return error_mark_node;
}
else if (c_dialect_cxx () && type_code == ARRAY_TYPE
&& !COMPLETE_TYPE_P (TREE_TYPE (type)))
{
if (complain)
error_at (loc, "invalid application of %qs to array type %qT of "
"incomplete element type", op_name, type);
return error_mark_node;
}
else
{
if (is_sizeof)
value = size_binop_loc (loc, CEIL_DIV_EXPR, TYPE_SIZE_UNIT (type),
size_int (TYPE_PRECISION (char_type_node)
/ BITS_PER_UNIT));
else if (min_alignof)
value = size_int (min_align_of_type (type));
else
value = size_int (TYPE_ALIGN_UNIT (type));
}
value = fold_convert_loc (loc, size_type_node, value);
return value;
}
tree
c_alignof_expr (location_t loc, tree expr)
{
tree t;
if (VAR_OR_FUNCTION_DECL_P (expr))
t = size_int (DECL_ALIGN_UNIT (expr));
else if (TREE_CODE (expr) == COMPONENT_REF
&& DECL_C_BIT_FIELD (TREE_OPERAND (expr, 1)))
{
error_at (loc, "%<__alignof%> applied to a bit-field");
t = size_one_node;
}
else if (TREE_CODE (expr) == COMPONENT_REF
&& TREE_CODE (TREE_OPERAND (expr, 1)) == FIELD_DECL)
t = size_int (DECL_ALIGN_UNIT (TREE_OPERAND (expr, 1)));
else if (INDIRECT_REF_P (expr))
{
tree t = TREE_OPERAND (expr, 0);
tree best = t;
int bestalign = TYPE_ALIGN (TREE_TYPE (TREE_TYPE (t)));
while (CONVERT_EXPR_P (t)
&& TREE_CODE (TREE_TYPE (TREE_OPERAND (t, 0))) == POINTER_TYPE)
{
int thisalign;
t = TREE_OPERAND (t, 0);
thisalign = TYPE_ALIGN (TREE_TYPE (TREE_TYPE (t)));
if (thisalign > bestalign)
best = t, bestalign = thisalign;
}
return c_alignof (loc, TREE_TYPE (TREE_TYPE (best)));
}
else
return c_alignof (loc, TREE_TYPE (expr));
return fold_convert_loc (loc, size_type_node, t);
}

enum built_in_attribute
{
#define DEF_ATTR_NULL_TREE(ENUM) ENUM,
#define DEF_ATTR_INT(ENUM, VALUE) ENUM,
#define DEF_ATTR_STRING(ENUM, VALUE) ENUM,
#define DEF_ATTR_IDENT(ENUM, STRING) ENUM,
#define DEF_ATTR_TREE_LIST(ENUM, PURPOSE, VALUE, CHAIN) ENUM,
#include "builtin-attrs.def"
#undef DEF_ATTR_NULL_TREE
#undef DEF_ATTR_INT
#undef DEF_ATTR_STRING
#undef DEF_ATTR_IDENT
#undef DEF_ATTR_TREE_LIST
ATTR_LAST
};
static GTY(()) tree built_in_attributes[(int) ATTR_LAST];
static void c_init_attributes (void);
enum c_builtin_type
{
#define DEF_PRIMITIVE_TYPE(NAME, VALUE) NAME,
#define DEF_FUNCTION_TYPE_0(NAME, RETURN) NAME,
#define DEF_FUNCTION_TYPE_1(NAME, RETURN, ARG1) NAME,
#define DEF_FUNCTION_TYPE_2(NAME, RETURN, ARG1, ARG2) NAME,
#define DEF_FUNCTION_TYPE_3(NAME, RETURN, ARG1, ARG2, ARG3) NAME,
#define DEF_FUNCTION_TYPE_4(NAME, RETURN, ARG1, ARG2, ARG3, ARG4) NAME,
#define DEF_FUNCTION_TYPE_5(NAME, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5) NAME,
#define DEF_FUNCTION_TYPE_6(NAME, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6) NAME,
#define DEF_FUNCTION_TYPE_7(NAME, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6, ARG7) NAME,
#define DEF_FUNCTION_TYPE_8(NAME, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6, ARG7, ARG8) NAME,
#define DEF_FUNCTION_TYPE_9(NAME, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6, ARG7, ARG8, ARG9) NAME,
#define DEF_FUNCTION_TYPE_10(NAME, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6, ARG7, ARG8, ARG9, ARG10) NAME,
#define DEF_FUNCTION_TYPE_11(NAME, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6, ARG7, ARG8, ARG9, ARG10, ARG11) NAME,
#define DEF_FUNCTION_TYPE_VAR_0(NAME, RETURN) NAME,
#define DEF_FUNCTION_TYPE_VAR_1(NAME, RETURN, ARG1) NAME,
#define DEF_FUNCTION_TYPE_VAR_2(NAME, RETURN, ARG1, ARG2) NAME,
#define DEF_FUNCTION_TYPE_VAR_3(NAME, RETURN, ARG1, ARG2, ARG3) NAME,
#define DEF_FUNCTION_TYPE_VAR_4(NAME, RETURN, ARG1, ARG2, ARG3, ARG4) NAME,
#define DEF_FUNCTION_TYPE_VAR_5(NAME, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5) \
NAME,
#define DEF_FUNCTION_TYPE_VAR_6(NAME, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6) NAME,
#define DEF_FUNCTION_TYPE_VAR_7(NAME, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6, ARG7) NAME,
#define DEF_POINTER_TYPE(NAME, TYPE) NAME,
#include "builtin-types.def"
#undef DEF_PRIMITIVE_TYPE
#undef DEF_FUNCTION_TYPE_0
#undef DEF_FUNCTION_TYPE_1
#undef DEF_FUNCTION_TYPE_2
#undef DEF_FUNCTION_TYPE_3
#undef DEF_FUNCTION_TYPE_4
#undef DEF_FUNCTION_TYPE_5
#undef DEF_FUNCTION_TYPE_6
#undef DEF_FUNCTION_TYPE_7
#undef DEF_FUNCTION_TYPE_8
#undef DEF_FUNCTION_TYPE_9
#undef DEF_FUNCTION_TYPE_10
#undef DEF_FUNCTION_TYPE_11
#undef DEF_FUNCTION_TYPE_VAR_0
#undef DEF_FUNCTION_TYPE_VAR_1
#undef DEF_FUNCTION_TYPE_VAR_2
#undef DEF_FUNCTION_TYPE_VAR_3
#undef DEF_FUNCTION_TYPE_VAR_4
#undef DEF_FUNCTION_TYPE_VAR_5
#undef DEF_FUNCTION_TYPE_VAR_6
#undef DEF_FUNCTION_TYPE_VAR_7
#undef DEF_POINTER_TYPE
BT_LAST
};
typedef enum c_builtin_type builtin_type;
static tree builtin_types[(int) BT_LAST + 1];
static void
def_fn_type (builtin_type def, builtin_type ret, bool var, int n, ...)
{
tree t;
tree *args = XALLOCAVEC (tree, n);
va_list list;
int i;
va_start (list, n);
for (i = 0; i < n; ++i)
{
builtin_type a = (builtin_type) va_arg (list, int);
t = builtin_types[a];
if (t == error_mark_node)
goto egress;
args[i] = t;
}
t = builtin_types[ret];
if (t == error_mark_node)
goto egress;
if (var)
t = build_varargs_function_type_array (t, n, args);
else
t = build_function_type_array (t, n, args);
egress:
builtin_types[def] = t;
va_end (list);
}
static void
c_define_builtins (tree va_list_ref_type_node, tree va_list_arg_type_node)
{
#define DEF_PRIMITIVE_TYPE(ENUM, VALUE) \
builtin_types[ENUM] = VALUE;
#define DEF_FUNCTION_TYPE_0(ENUM, RETURN) \
def_fn_type (ENUM, RETURN, 0, 0);
#define DEF_FUNCTION_TYPE_1(ENUM, RETURN, ARG1) \
def_fn_type (ENUM, RETURN, 0, 1, ARG1);
#define DEF_FUNCTION_TYPE_2(ENUM, RETURN, ARG1, ARG2) \
def_fn_type (ENUM, RETURN, 0, 2, ARG1, ARG2);
#define DEF_FUNCTION_TYPE_3(ENUM, RETURN, ARG1, ARG2, ARG3) \
def_fn_type (ENUM, RETURN, 0, 3, ARG1, ARG2, ARG3);
#define DEF_FUNCTION_TYPE_4(ENUM, RETURN, ARG1, ARG2, ARG3, ARG4) \
def_fn_type (ENUM, RETURN, 0, 4, ARG1, ARG2, ARG3, ARG4);
#define DEF_FUNCTION_TYPE_5(ENUM, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5)	\
def_fn_type (ENUM, RETURN, 0, 5, ARG1, ARG2, ARG3, ARG4, ARG5);
#define DEF_FUNCTION_TYPE_6(ENUM, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6)					\
def_fn_type (ENUM, RETURN, 0, 6, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6);
#define DEF_FUNCTION_TYPE_7(ENUM, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6, ARG7)					\
def_fn_type (ENUM, RETURN, 0, 7, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7);
#define DEF_FUNCTION_TYPE_8(ENUM, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6, ARG7, ARG8)				\
def_fn_type (ENUM, RETURN, 0, 8, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6,	\
ARG7, ARG8);
#define DEF_FUNCTION_TYPE_9(ENUM, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6, ARG7, ARG8, ARG9)			\
def_fn_type (ENUM, RETURN, 0, 9, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6,	\
ARG7, ARG8, ARG9);
#define DEF_FUNCTION_TYPE_10(ENUM, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6, ARG7, ARG8, ARG9, ARG10)		 \
def_fn_type (ENUM, RETURN, 0, 10, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6,	 \
ARG7, ARG8, ARG9, ARG10);
#define DEF_FUNCTION_TYPE_11(ENUM, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6, ARG7, ARG8, ARG9, ARG10, ARG11)	 \
def_fn_type (ENUM, RETURN, 0, 11, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6,	 \
ARG7, ARG8, ARG9, ARG10, ARG11);
#define DEF_FUNCTION_TYPE_VAR_0(ENUM, RETURN) \
def_fn_type (ENUM, RETURN, 1, 0);
#define DEF_FUNCTION_TYPE_VAR_1(ENUM, RETURN, ARG1) \
def_fn_type (ENUM, RETURN, 1, 1, ARG1);
#define DEF_FUNCTION_TYPE_VAR_2(ENUM, RETURN, ARG1, ARG2) \
def_fn_type (ENUM, RETURN, 1, 2, ARG1, ARG2);
#define DEF_FUNCTION_TYPE_VAR_3(ENUM, RETURN, ARG1, ARG2, ARG3) \
def_fn_type (ENUM, RETURN, 1, 3, ARG1, ARG2, ARG3);
#define DEF_FUNCTION_TYPE_VAR_4(ENUM, RETURN, ARG1, ARG2, ARG3, ARG4) \
def_fn_type (ENUM, RETURN, 1, 4, ARG1, ARG2, ARG3, ARG4);
#define DEF_FUNCTION_TYPE_VAR_5(ENUM, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5) \
def_fn_type (ENUM, RETURN, 1, 5, ARG1, ARG2, ARG3, ARG4, ARG5);
#define DEF_FUNCTION_TYPE_VAR_6(ENUM, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6) \
def_fn_type (ENUM, RETURN, 1, 6, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6);
#define DEF_FUNCTION_TYPE_VAR_7(ENUM, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6, ARG7)				\
def_fn_type (ENUM, RETURN, 1, 7, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7);
#define DEF_POINTER_TYPE(ENUM, TYPE) \
builtin_types[(int) ENUM] = build_pointer_type (builtin_types[(int) TYPE]);
#include "builtin-types.def"
#undef DEF_PRIMITIVE_TYPE
#undef DEF_FUNCTION_TYPE_0
#undef DEF_FUNCTION_TYPE_1
#undef DEF_FUNCTION_TYPE_2
#undef DEF_FUNCTION_TYPE_3
#undef DEF_FUNCTION_TYPE_4
#undef DEF_FUNCTION_TYPE_5
#undef DEF_FUNCTION_TYPE_6
#undef DEF_FUNCTION_TYPE_7
#undef DEF_FUNCTION_TYPE_8
#undef DEF_FUNCTION_TYPE_9
#undef DEF_FUNCTION_TYPE_10
#undef DEF_FUNCTION_TYPE_11
#undef DEF_FUNCTION_TYPE_VAR_0
#undef DEF_FUNCTION_TYPE_VAR_1
#undef DEF_FUNCTION_TYPE_VAR_2
#undef DEF_FUNCTION_TYPE_VAR_3
#undef DEF_FUNCTION_TYPE_VAR_4
#undef DEF_FUNCTION_TYPE_VAR_5
#undef DEF_FUNCTION_TYPE_VAR_6
#undef DEF_FUNCTION_TYPE_VAR_7
#undef DEF_POINTER_TYPE
builtin_types[(int) BT_LAST] = NULL_TREE;
c_init_attributes ();
#define DEF_BUILTIN(ENUM, NAME, CLASS, TYPE, LIBTYPE, BOTH_P, FALLBACK_P, \
NONANSI_P, ATTRS, IMPLICIT, COND)			\
if (NAME && COND)							\
def_builtin_1 (ENUM, NAME, CLASS,                                   \
builtin_types[(int) TYPE],                           \
builtin_types[(int) LIBTYPE],                        \
BOTH_P, FALLBACK_P, NONANSI_P,                       \
built_in_attributes[(int) ATTRS], IMPLICIT);
#include "builtins.def"
targetm.init_builtins ();
build_common_builtin_nodes ();
}
static inline tree
c_get_ident (const char *id)
{
return get_identifier (id);
}
void
c_common_nodes_and_builtins (void)
{
int char16_type_size;
int char32_type_size;
int wchar_type_size;
tree array_domain_type;
tree va_list_ref_type_node;
tree va_list_arg_type_node;
int i;
build_common_tree_nodes (flag_signed_char);
record_builtin_type (RID_INT, NULL, integer_type_node);
record_builtin_type (RID_CHAR, "char", char_type_node);
if (c_dialect_cxx ())
record_builtin_type (RID_SIGNED, NULL, integer_type_node);
record_builtin_type (RID_LONG, "long int", long_integer_type_node);
record_builtin_type (RID_UNSIGNED, "unsigned int", unsigned_type_node);
record_builtin_type (RID_MAX, "long unsigned int",
long_unsigned_type_node);
for (i = 0; i < NUM_INT_N_ENTS; i ++)
{
char name[25];
sprintf (name, "__int%d", int_n_data[i].bitsize);
record_builtin_type ((enum rid)(RID_FIRST_INT_N + i), name,
int_n_trees[i].signed_type);
sprintf (name, "__int%d unsigned", int_n_data[i].bitsize);
record_builtin_type (RID_MAX, name, int_n_trees[i].unsigned_type);
}
if (c_dialect_cxx ())
record_builtin_type (RID_MAX, "unsigned long", long_unsigned_type_node);
record_builtin_type (RID_MAX, "long long int",
long_long_integer_type_node);
record_builtin_type (RID_MAX, "long long unsigned int",
long_long_unsigned_type_node);
if (c_dialect_cxx ())
record_builtin_type (RID_MAX, "long long unsigned",
long_long_unsigned_type_node);
record_builtin_type (RID_SHORT, "short int", short_integer_type_node);
record_builtin_type (RID_MAX, "short unsigned int",
short_unsigned_type_node);
if (c_dialect_cxx ())
record_builtin_type (RID_MAX, "unsigned short",
short_unsigned_type_node);
record_builtin_type (RID_MAX, "signed char", signed_char_type_node);
record_builtin_type (RID_MAX, "unsigned char", unsigned_char_type_node);
lang_hooks.decls.pushdecl (build_decl (UNKNOWN_LOCATION,
TYPE_DECL, NULL_TREE,
intQI_type_node));
lang_hooks.decls.pushdecl (build_decl (UNKNOWN_LOCATION,
TYPE_DECL, NULL_TREE,
intHI_type_node));
lang_hooks.decls.pushdecl (build_decl (UNKNOWN_LOCATION,
TYPE_DECL, NULL_TREE,
intSI_type_node));
lang_hooks.decls.pushdecl (build_decl (UNKNOWN_LOCATION,
TYPE_DECL, NULL_TREE,
intDI_type_node));
#if HOST_BITS_PER_WIDE_INT >= 64
if (targetm.scalar_mode_supported_p (TImode))
lang_hooks.decls.pushdecl (build_decl (UNKNOWN_LOCATION,
TYPE_DECL,
get_identifier ("__int128_t"),
intTI_type_node));
#endif
lang_hooks.decls.pushdecl (build_decl (UNKNOWN_LOCATION,
TYPE_DECL, NULL_TREE,
unsigned_intQI_type_node));
lang_hooks.decls.pushdecl (build_decl (UNKNOWN_LOCATION,
TYPE_DECL, NULL_TREE,
unsigned_intHI_type_node));
lang_hooks.decls.pushdecl (build_decl (UNKNOWN_LOCATION,
TYPE_DECL, NULL_TREE,
unsigned_intSI_type_node));
lang_hooks.decls.pushdecl (build_decl (UNKNOWN_LOCATION,
TYPE_DECL, NULL_TREE,
unsigned_intDI_type_node));
#if HOST_BITS_PER_WIDE_INT >= 64
if (targetm.scalar_mode_supported_p (TImode))
lang_hooks.decls.pushdecl (build_decl (UNKNOWN_LOCATION,
TYPE_DECL,
get_identifier ("__uint128_t"),
unsigned_intTI_type_node));
#endif
if (targetm.scalar_mode_supported_p (TImode))
{
widest_integer_literal_type_node = intTI_type_node;
widest_unsigned_literal_type_node = unsigned_intTI_type_node;
}
else
{
widest_integer_literal_type_node = intDI_type_node;
widest_unsigned_literal_type_node = unsigned_intDI_type_node;
}
signed_size_type_node = c_common_signed_type (size_type_node);
pid_type_node =
TREE_TYPE (identifier_global_value (get_identifier (PID_TYPE)));
record_builtin_type (RID_FLOAT, NULL, float_type_node);
record_builtin_type (RID_DOUBLE, NULL, double_type_node);
record_builtin_type (RID_MAX, "long double", long_double_type_node);
if (!c_dialect_cxx ())
for (i = 0; i < NUM_FLOATN_NX_TYPES; i++)
if (FLOATN_NX_TYPE_NODE (i) != NULL_TREE)
record_builtin_type ((enum rid) (RID_FLOATN_NX_FIRST + i), NULL,
FLOATN_NX_TYPE_NODE (i));
if (targetm.scalar_mode_supported_p (SDmode)
&& targetm.scalar_mode_supported_p (DDmode)
&& targetm.scalar_mode_supported_p (TDmode))
{
record_builtin_type (RID_DFLOAT32, NULL, dfloat32_type_node);
record_builtin_type (RID_DFLOAT64, NULL, dfloat64_type_node);
record_builtin_type (RID_DFLOAT128, NULL, dfloat128_type_node);
}
if (targetm.fixed_point_supported_p ())
{
record_builtin_type (RID_MAX, "short _Fract", short_fract_type_node);
record_builtin_type (RID_FRACT, NULL, fract_type_node);
record_builtin_type (RID_MAX, "long _Fract", long_fract_type_node);
record_builtin_type (RID_MAX, "long long _Fract",
long_long_fract_type_node);
record_builtin_type (RID_MAX, "unsigned short _Fract",
unsigned_short_fract_type_node);
record_builtin_type (RID_MAX, "unsigned _Fract",
unsigned_fract_type_node);
record_builtin_type (RID_MAX, "unsigned long _Fract",
unsigned_long_fract_type_node);
record_builtin_type (RID_MAX, "unsigned long long _Fract",
unsigned_long_long_fract_type_node);
record_builtin_type (RID_MAX, "_Sat short _Fract",
sat_short_fract_type_node);
record_builtin_type (RID_MAX, "_Sat _Fract", sat_fract_type_node);
record_builtin_type (RID_MAX, "_Sat long _Fract",
sat_long_fract_type_node);
record_builtin_type (RID_MAX, "_Sat long long _Fract",
sat_long_long_fract_type_node);
record_builtin_type (RID_MAX, "_Sat unsigned short _Fract",
sat_unsigned_short_fract_type_node);
record_builtin_type (RID_MAX, "_Sat unsigned _Fract",
sat_unsigned_fract_type_node);
record_builtin_type (RID_MAX, "_Sat unsigned long _Fract",
sat_unsigned_long_fract_type_node);
record_builtin_type (RID_MAX, "_Sat unsigned long long _Fract",
sat_unsigned_long_long_fract_type_node);
record_builtin_type (RID_MAX, "short _Accum", short_accum_type_node);
record_builtin_type (RID_ACCUM, NULL, accum_type_node);
record_builtin_type (RID_MAX, "long _Accum", long_accum_type_node);
record_builtin_type (RID_MAX, "long long _Accum",
long_long_accum_type_node);
record_builtin_type (RID_MAX, "unsigned short _Accum",
unsigned_short_accum_type_node);
record_builtin_type (RID_MAX, "unsigned _Accum",
unsigned_accum_type_node);
record_builtin_type (RID_MAX, "unsigned long _Accum",
unsigned_long_accum_type_node);
record_builtin_type (RID_MAX, "unsigned long long _Accum",
unsigned_long_long_accum_type_node);
record_builtin_type (RID_MAX, "_Sat short _Accum",
sat_short_accum_type_node);
record_builtin_type (RID_MAX, "_Sat _Accum", sat_accum_type_node);
record_builtin_type (RID_MAX, "_Sat long _Accum",
sat_long_accum_type_node);
record_builtin_type (RID_MAX, "_Sat long long _Accum",
sat_long_long_accum_type_node);
record_builtin_type (RID_MAX, "_Sat unsigned short _Accum",
sat_unsigned_short_accum_type_node);
record_builtin_type (RID_MAX, "_Sat unsigned _Accum",
sat_unsigned_accum_type_node);
record_builtin_type (RID_MAX, "_Sat unsigned long _Accum",
sat_unsigned_long_accum_type_node);
record_builtin_type (RID_MAX, "_Sat unsigned long long _Accum",
sat_unsigned_long_long_accum_type_node);
}
lang_hooks.decls.pushdecl (build_decl (UNKNOWN_LOCATION,
TYPE_DECL,
get_identifier ("complex int"),
complex_integer_type_node));
lang_hooks.decls.pushdecl (build_decl (UNKNOWN_LOCATION,
TYPE_DECL,
get_identifier ("complex float"),
complex_float_type_node));
lang_hooks.decls.pushdecl (build_decl (UNKNOWN_LOCATION,
TYPE_DECL,
get_identifier ("complex double"),
complex_double_type_node));
lang_hooks.decls.pushdecl
(build_decl (UNKNOWN_LOCATION,
TYPE_DECL, get_identifier ("complex long double"),
complex_long_double_type_node));
if (!c_dialect_cxx ())
for (i = 0; i < NUM_FLOATN_NX_TYPES; i++)
if (COMPLEX_FLOATN_NX_TYPE_NODE (i) != NULL_TREE)
{
char buf[30];
sprintf (buf, "complex _Float%d%s", floatn_nx_types[i].n,
floatn_nx_types[i].extended ? "x" : "");
lang_hooks.decls.pushdecl
(build_decl (UNKNOWN_LOCATION,
TYPE_DECL,
get_identifier (buf),
COMPLEX_FLOATN_NX_TYPE_NODE (i)));
}
if (c_dialect_cxx ())
{
for (unsigned i = 0;
i < sizeof (builtin_structptr_types)
/ sizeof (builtin_structptr_type);
++i)
builtin_structptr_types[i].node =
build_variant_type_copy (builtin_structptr_types[i].base);
}
record_builtin_type (RID_VOID, NULL, void_type_node);
{
tree void_name = TYPE_NAME (void_type_node);
TYPE_NAME (void_type_node) = NULL_TREE;
TYPE_NAME (build_qualified_type (void_type_node, TYPE_QUAL_CONST))
= void_name;
TYPE_NAME (void_type_node) = void_name;
}
void_list_node = build_void_list_node ();
array_domain_type = build_index_type (size_int (200));
char_array_type_node
= build_array_type (char_type_node, array_domain_type);
string_type_node = build_pointer_type (char_type_node);
const_string_type_node
= build_pointer_type (build_qualified_type
(char_type_node, TYPE_QUAL_CONST));
wchar_type_node = get_identifier (MODIFIED_WCHAR_TYPE);
wchar_type_node = TREE_TYPE (identifier_global_value (wchar_type_node));
wchar_type_size = TYPE_PRECISION (wchar_type_node);
underlying_wchar_type_node = wchar_type_node;
if (c_dialect_cxx ())
{
if (TYPE_UNSIGNED (wchar_type_node))
wchar_type_node = make_unsigned_type (wchar_type_size);
else
wchar_type_node = make_signed_type (wchar_type_size);
record_builtin_type (RID_WCHAR, "wchar_t", wchar_type_node);
}
wchar_array_type_node
= build_array_type (wchar_type_node, array_domain_type);
char16_type_node = get_identifier (CHAR16_TYPE);
char16_type_node = TREE_TYPE (identifier_global_value (char16_type_node));
char16_type_size = TYPE_PRECISION (char16_type_node);
if (c_dialect_cxx ())
{
char16_type_node = make_unsigned_type (char16_type_size);
if (cxx_dialect >= cxx11)
record_builtin_type (RID_CHAR16, "char16_t", char16_type_node);
}
char16_array_type_node
= build_array_type (char16_type_node, array_domain_type);
char32_type_node = get_identifier (CHAR32_TYPE);
char32_type_node = TREE_TYPE (identifier_global_value (char32_type_node));
char32_type_size = TYPE_PRECISION (char32_type_node);
if (c_dialect_cxx ())
{
char32_type_node = make_unsigned_type (char32_type_size);
if (cxx_dialect >= cxx11)
record_builtin_type (RID_CHAR32, "char32_t", char32_type_node);
}
char32_array_type_node
= build_array_type (char32_type_node, array_domain_type);
wint_type_node =
TREE_TYPE (identifier_global_value (get_identifier (WINT_TYPE)));
intmax_type_node =
TREE_TYPE (identifier_global_value (get_identifier (INTMAX_TYPE)));
uintmax_type_node =
TREE_TYPE (identifier_global_value (get_identifier (UINTMAX_TYPE)));
if (SIG_ATOMIC_TYPE)
sig_atomic_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (SIG_ATOMIC_TYPE)));
if (INT8_TYPE)
int8_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (INT8_TYPE)));
if (INT16_TYPE)
int16_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (INT16_TYPE)));
if (INT32_TYPE)
int32_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (INT32_TYPE)));
if (INT64_TYPE)
int64_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (INT64_TYPE)));
if (UINT8_TYPE)
uint8_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (UINT8_TYPE)));
if (UINT16_TYPE)
c_uint16_type_node = uint16_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (UINT16_TYPE)));
if (UINT32_TYPE)
c_uint32_type_node = uint32_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (UINT32_TYPE)));
if (UINT64_TYPE)
c_uint64_type_node = uint64_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (UINT64_TYPE)));
if (INT_LEAST8_TYPE)
int_least8_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (INT_LEAST8_TYPE)));
if (INT_LEAST16_TYPE)
int_least16_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (INT_LEAST16_TYPE)));
if (INT_LEAST32_TYPE)
int_least32_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (INT_LEAST32_TYPE)));
if (INT_LEAST64_TYPE)
int_least64_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (INT_LEAST64_TYPE)));
if (UINT_LEAST8_TYPE)
uint_least8_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (UINT_LEAST8_TYPE)));
if (UINT_LEAST16_TYPE)
uint_least16_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (UINT_LEAST16_TYPE)));
if (UINT_LEAST32_TYPE)
uint_least32_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (UINT_LEAST32_TYPE)));
if (UINT_LEAST64_TYPE)
uint_least64_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (UINT_LEAST64_TYPE)));
if (INT_FAST8_TYPE)
int_fast8_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (INT_FAST8_TYPE)));
if (INT_FAST16_TYPE)
int_fast16_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (INT_FAST16_TYPE)));
if (INT_FAST32_TYPE)
int_fast32_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (INT_FAST32_TYPE)));
if (INT_FAST64_TYPE)
int_fast64_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (INT_FAST64_TYPE)));
if (UINT_FAST8_TYPE)
uint_fast8_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (UINT_FAST8_TYPE)));
if (UINT_FAST16_TYPE)
uint_fast16_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (UINT_FAST16_TYPE)));
if (UINT_FAST32_TYPE)
uint_fast32_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (UINT_FAST32_TYPE)));
if (UINT_FAST64_TYPE)
uint_fast64_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (UINT_FAST64_TYPE)));
if (INTPTR_TYPE)
intptr_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (INTPTR_TYPE)));
if (UINTPTR_TYPE)
uintptr_type_node =
TREE_TYPE (identifier_global_value (c_get_ident (UINTPTR_TYPE)));
default_function_type
= build_varargs_function_type_list (integer_type_node, NULL_TREE);
unsigned_ptrdiff_type_node = c_common_unsigned_type (ptrdiff_type_node);
lang_hooks.decls.pushdecl
(build_decl (UNKNOWN_LOCATION,
TYPE_DECL, get_identifier ("__builtin_va_list"),
va_list_type_node));
if (targetm.enum_va_list_p)
{
int l;
const char *pname;
tree ptype;
for (l = 0; targetm.enum_va_list_p (l, &pname, &ptype); ++l)
{
lang_hooks.decls.pushdecl
(build_decl (UNKNOWN_LOCATION,
TYPE_DECL, get_identifier (pname),
ptype));
}
}
if (TREE_CODE (va_list_type_node) == ARRAY_TYPE)
{
va_list_arg_type_node = va_list_ref_type_node =
build_pointer_type (TREE_TYPE (va_list_type_node));
}
else
{
va_list_arg_type_node = va_list_type_node;
va_list_ref_type_node = build_reference_type (va_list_type_node);
}
if (!flag_preprocess_only)
c_define_builtins (va_list_ref_type_node, va_list_arg_type_node);
main_identifier_node = get_identifier ("main");
null_node = make_int_cst (1, 1);
TREE_TYPE (null_node) = c_common_type_for_size (POINTER_SIZE, 0);
memset (builtin_types, 0, sizeof (builtin_types));
}
static GTY(()) int compound_literal_number;
void
set_compound_literal_name (tree decl)
{
char *name;
ASM_FORMAT_PRIVATE_NAME (name, "__compound_literal",
compound_literal_number);
compound_literal_number++;
DECL_NAME (decl) = get_identifier (name);
}
static tree
build_va_arg_1 (location_t loc, tree type, tree op)
{
tree expr = build1 (VA_ARG_EXPR, type, op);
SET_EXPR_LOCATION (expr, loc);
return expr;
}
tree
build_va_arg (location_t loc, tree expr, tree type)
{
tree va_type = TREE_TYPE (expr);
tree canon_va_type = (va_type == error_mark_node
? error_mark_node
: targetm.canonical_va_list_type (va_type));
if (va_type == error_mark_node
|| canon_va_type == NULL_TREE)
{
if (canon_va_type == NULL_TREE)
error_at (loc, "first argument to %<va_arg%> not of type %<va_list%>");
return build_va_arg_1 (loc, type, error_mark_node);
}
if (TREE_CODE (canon_va_type) != ARRAY_TYPE)
{
mark_addressable (expr);
expr = build1 (ADDR_EXPR, build_pointer_type (TREE_TYPE (expr)), expr);
return build_va_arg_1 (loc, type, expr);
}
if (TREE_CODE (va_type) == ARRAY_TYPE)
{
mark_addressable (expr);
expr = build1 (ADDR_EXPR, build_pointer_type (TREE_TYPE (canon_va_type)),
expr);
tree canon_expr_type
= targetm.canonical_va_list_type (TREE_TYPE (expr));
gcc_assert (canon_expr_type != NULL_TREE);
}
else
{
gcc_assert (POINTER_TYPE_P (va_type));
gcc_assert (TYPE_MAIN_VARIANT (TREE_TYPE (va_type))
== TYPE_MAIN_VARIANT (TREE_TYPE (canon_va_type)));
;
}
return build_va_arg_1 (loc, type, expr);
}
struct disabled_builtin
{
const char *name;
struct disabled_builtin *next;
};
static disabled_builtin *disabled_builtins = NULL;
static bool builtin_function_disabled_p (const char *);
void
disable_builtin_function (const char *name)
{
if (strncmp (name, "__builtin_", strlen ("__builtin_")) == 0)
error ("cannot disable built-in function %qs", name);
else
{
disabled_builtin *new_disabled_builtin = XNEW (disabled_builtin);
new_disabled_builtin->name = name;
new_disabled_builtin->next = disabled_builtins;
disabled_builtins = new_disabled_builtin;
}
}
static bool
builtin_function_disabled_p (const char *name)
{
disabled_builtin *p;
for (p = disabled_builtins; p != NULL; p = p->next)
{
if (strcmp (name, p->name) == 0)
return true;
}
return false;
}
static void
def_builtin_1 (enum built_in_function fncode,
const char *name,
enum built_in_class fnclass,
tree fntype, tree libtype,
bool both_p, bool fallback_p, bool nonansi_p,
tree fnattrs, bool implicit_p)
{
tree decl;
const char *libname;
if (fntype == error_mark_node)
return;
gcc_assert ((!both_p && !fallback_p)
|| !strncmp (name, "__builtin_",
strlen ("__builtin_")));
libname = name + strlen ("__builtin_");
decl = add_builtin_function (name, fntype, fncode, fnclass,
(fallback_p ? libname : NULL),
fnattrs);
set_builtin_decl (fncode, decl, implicit_p);
if (both_p
&& !flag_no_builtin && !builtin_function_disabled_p (libname)
&& !(nonansi_p && flag_no_nonansi_builtin))
add_builtin_function (libname, libtype, fncode, fnclass,
NULL, fnattrs);
}

bool
c_promoting_integer_type_p (const_tree t)
{
switch (TREE_CODE (t))
{
case INTEGER_TYPE:
return (TYPE_MAIN_VARIANT (t) == char_type_node
|| TYPE_MAIN_VARIANT (t) == signed_char_type_node
|| TYPE_MAIN_VARIANT (t) == unsigned_char_type_node
|| TYPE_MAIN_VARIANT (t) == short_integer_type_node
|| TYPE_MAIN_VARIANT (t) == short_unsigned_type_node
|| TYPE_PRECISION (t) < TYPE_PRECISION (integer_type_node));
case ENUMERAL_TYPE:
return TYPE_PRECISION (t) < TYPE_PRECISION (integer_type_node);
case BOOLEAN_TYPE:
return true;
default:
return false;
}
}
bool
self_promoting_args_p (const_tree parms)
{
const_tree t;
for (t = parms; t; t = TREE_CHAIN (t))
{
tree type = TREE_VALUE (t);
if (type == error_mark_node)
continue;
if (TREE_CHAIN (t) == NULL_TREE && type != void_type_node)
return false;
if (type == NULL_TREE)
return false;
if (TYPE_MAIN_VARIANT (type) == float_type_node)
return false;
if (c_promoting_integer_type_p (type))
return false;
}
return true;
}
tree
strip_pointer_operator (tree t)
{
while (POINTER_TYPE_P (t))
t = TREE_TYPE (t);
return t;
}
tree
strip_pointer_or_array_types (tree t)
{
while (TREE_CODE (t) == ARRAY_TYPE || POINTER_TYPE_P (t))
t = TREE_TYPE (t);
return t;
}
int
case_compare (splay_tree_key k1, splay_tree_key k2)
{
if (!k1)
return k2 ? -1 : 0;
else if (!k2)
return k1 ? 1 : 0;
return tree_int_cst_compare ((tree) k1, (tree) k2);
}
tree
c_add_case_label (location_t loc, splay_tree cases, tree cond, tree orig_type,
tree low_value, tree high_value, bool *outside_range_p)
{
tree type;
tree label;
tree case_label;
splay_tree_node node;
label = create_artificial_label (loc);
if (!cond || cond == error_mark_node)
goto error_out;
if ((low_value && TREE_TYPE (low_value)
&& POINTER_TYPE_P (TREE_TYPE (low_value)))
|| (high_value && TREE_TYPE (high_value)
&& POINTER_TYPE_P (TREE_TYPE (high_value))))
{
error_at (loc, "pointers are not permitted as case values");
goto error_out;
}
if (high_value)
pedwarn (loc, OPT_Wpedantic,
"range expressions in switch statements are non-standard");
type = TREE_TYPE (cond);
if (low_value)
{
low_value = check_case_value (loc, low_value);
low_value = convert_and_check (loc, type, low_value);
if (low_value == error_mark_node)
goto error_out;
}
if (high_value)
{
high_value = check_case_value (loc, high_value);
high_value = convert_and_check (loc, type, high_value);
if (high_value == error_mark_node)
goto error_out;
}
if (low_value && high_value)
{
if (tree_int_cst_equal (low_value, high_value))
high_value = NULL_TREE;
else if (!tree_int_cst_lt (low_value, high_value))
warning_at (loc, 0, "empty range specified");
}
if (low_value
&& !check_case_bounds (loc, type, orig_type,
&low_value, high_value ? &high_value : NULL,
outside_range_p))
return NULL_TREE;
node = splay_tree_lookup (cases, (splay_tree_key) low_value);
if (!node && (low_value || high_value))
{
splay_tree_node low_bound;
splay_tree_node high_bound;
low_bound = splay_tree_predecessor (cases,
(splay_tree_key) low_value);
high_bound = splay_tree_successor (cases,
(splay_tree_key) low_value);
if (low_bound
&& CASE_HIGH ((tree) low_bound->value)
&& tree_int_cst_compare (CASE_HIGH ((tree) low_bound->value),
low_value) >= 0)
node = low_bound;
else if (high_bound
&& high_value
&& (tree_int_cst_compare ((tree) high_bound->key,
high_value)
<= 0))
node = high_bound;
}
if (node)
{
tree duplicate = CASE_LABEL ((tree) node->value);
if (high_value)
{
error_at (loc, "duplicate (or overlapping) case value");
inform (DECL_SOURCE_LOCATION (duplicate),
"this is the first entry overlapping that value");
}
else if (low_value)
{
error_at (loc, "duplicate case value") ;
inform (DECL_SOURCE_LOCATION (duplicate), "previously used here");
}
else
{
error_at (loc, "multiple default labels in one switch");
inform (DECL_SOURCE_LOCATION (duplicate),
"this is the first default label");
}
goto error_out;
}
case_label = add_stmt (build_case_label (low_value, high_value, label));
splay_tree_insert (cases,
(splay_tree_key) low_value,
(splay_tree_value) case_label);
return case_label;
error_out:
if (!cases->root)
{
tree t = create_artificial_label (loc);
add_stmt (build_stmt (loc, LABEL_EXPR, t));
}
return error_mark_node;
}
static int
c_switch_covers_all_cases_p_1 (splay_tree_node node, void *data)
{
tree label = (tree) node->value;
tree *args = (tree *) data;
gcc_assert (CASE_LOW (label));
if (args[0] == NULL_TREE)
{
if (wi::to_widest (args[1]) < wi::to_widest (CASE_LOW (label)))
return 1;
}
else if (wi::add (wi::to_widest (args[0]), 1)
!= wi::to_widest (CASE_LOW (label)))
return 1;
if (CASE_HIGH (label))
args[0] = CASE_HIGH (label);
else
args[0] = CASE_LOW (label);
return 0;
}
bool
c_switch_covers_all_cases_p (splay_tree cases, tree type)
{
splay_tree_node default_node
= splay_tree_lookup (cases, (splay_tree_key) NULL);
if (default_node)
return true;
if (!INTEGRAL_TYPE_P (type))
return false;
tree args[2] = { NULL_TREE, TYPE_MIN_VALUE (type) };
if (splay_tree_foreach (cases, c_switch_covers_all_cases_p_1, args))
return false;
if (args[0] == NULL_TREE
|| wi::to_widest (args[0]) < wi::to_widest (TYPE_MAX_VALUE (type)))
return false;
return true;
}
tree
finish_label_address_expr (tree label, location_t loc)
{
tree result;
pedwarn (input_location, OPT_Wpedantic, "taking the address of a label is non-standard");
if (label == error_mark_node)
return error_mark_node;
label = lookup_label (label);
if (label == NULL_TREE)
result = null_pointer_node;
else
{
TREE_USED (label) = 1;
result = build1 (ADDR_EXPR, ptr_type_node, label);
protected_set_expr_location (result, loc);
}
return result;
}

tree
boolean_increment (enum tree_code code, tree arg)
{
tree val;
tree true_res = build_int_cst (TREE_TYPE (arg), 1);
arg = stabilize_reference (arg);
switch (code)
{
case PREINCREMENT_EXPR:
val = build2 (MODIFY_EXPR, TREE_TYPE (arg), arg, true_res);
break;
case POSTINCREMENT_EXPR:
val = build2 (MODIFY_EXPR, TREE_TYPE (arg), arg, true_res);
arg = save_expr (arg);
val = build2 (COMPOUND_EXPR, TREE_TYPE (arg), val, arg);
val = build2 (COMPOUND_EXPR, TREE_TYPE (arg), arg, val);
break;
case PREDECREMENT_EXPR:
val = build2 (MODIFY_EXPR, TREE_TYPE (arg), arg,
invert_truthvalue_loc (input_location, arg));
break;
case POSTDECREMENT_EXPR:
val = build2 (MODIFY_EXPR, TREE_TYPE (arg), arg,
invert_truthvalue_loc (input_location, arg));
arg = save_expr (arg);
val = build2 (COMPOUND_EXPR, TREE_TYPE (arg), val, arg);
val = build2 (COMPOUND_EXPR, TREE_TYPE (arg), arg, val);
break;
default:
gcc_unreachable ();
}
TREE_SIDE_EFFECTS (val) = 1;
return val;
}

void
c_stddef_cpp_builtins(void)
{
builtin_define_with_value ("__SIZE_TYPE__", SIZE_TYPE, 0);
builtin_define_with_value ("__PTRDIFF_TYPE__", PTRDIFF_TYPE, 0);
builtin_define_with_value ("__WCHAR_TYPE__", MODIFIED_WCHAR_TYPE, 0);
builtin_define_with_value ("__WINT_TYPE__", WINT_TYPE, 0);
builtin_define_with_value ("__INTMAX_TYPE__", INTMAX_TYPE, 0);
builtin_define_with_value ("__UINTMAX_TYPE__", UINTMAX_TYPE, 0);
builtin_define_with_value ("__CHAR16_TYPE__", CHAR16_TYPE, 0);
builtin_define_with_value ("__CHAR32_TYPE__", CHAR32_TYPE, 0);
if (SIG_ATOMIC_TYPE)
builtin_define_with_value ("__SIG_ATOMIC_TYPE__", SIG_ATOMIC_TYPE, 0);
if (INT8_TYPE)
builtin_define_with_value ("__INT8_TYPE__", INT8_TYPE, 0);
if (INT16_TYPE)
builtin_define_with_value ("__INT16_TYPE__", INT16_TYPE, 0);
if (INT32_TYPE)
builtin_define_with_value ("__INT32_TYPE__", INT32_TYPE, 0);
if (INT64_TYPE)
builtin_define_with_value ("__INT64_TYPE__", INT64_TYPE, 0);
if (UINT8_TYPE)
builtin_define_with_value ("__UINT8_TYPE__", UINT8_TYPE, 0);
if (UINT16_TYPE)
builtin_define_with_value ("__UINT16_TYPE__", UINT16_TYPE, 0);
if (UINT32_TYPE)
builtin_define_with_value ("__UINT32_TYPE__", UINT32_TYPE, 0);
if (UINT64_TYPE)
builtin_define_with_value ("__UINT64_TYPE__", UINT64_TYPE, 0);
if (INT_LEAST8_TYPE)
builtin_define_with_value ("__INT_LEAST8_TYPE__", INT_LEAST8_TYPE, 0);
if (INT_LEAST16_TYPE)
builtin_define_with_value ("__INT_LEAST16_TYPE__", INT_LEAST16_TYPE, 0);
if (INT_LEAST32_TYPE)
builtin_define_with_value ("__INT_LEAST32_TYPE__", INT_LEAST32_TYPE, 0);
if (INT_LEAST64_TYPE)
builtin_define_with_value ("__INT_LEAST64_TYPE__", INT_LEAST64_TYPE, 0);
if (UINT_LEAST8_TYPE)
builtin_define_with_value ("__UINT_LEAST8_TYPE__", UINT_LEAST8_TYPE, 0);
if (UINT_LEAST16_TYPE)
builtin_define_with_value ("__UINT_LEAST16_TYPE__", UINT_LEAST16_TYPE, 0);
if (UINT_LEAST32_TYPE)
builtin_define_with_value ("__UINT_LEAST32_TYPE__", UINT_LEAST32_TYPE, 0);
if (UINT_LEAST64_TYPE)
builtin_define_with_value ("__UINT_LEAST64_TYPE__", UINT_LEAST64_TYPE, 0);
if (INT_FAST8_TYPE)
builtin_define_with_value ("__INT_FAST8_TYPE__", INT_FAST8_TYPE, 0);
if (INT_FAST16_TYPE)
builtin_define_with_value ("__INT_FAST16_TYPE__", INT_FAST16_TYPE, 0);
if (INT_FAST32_TYPE)
builtin_define_with_value ("__INT_FAST32_TYPE__", INT_FAST32_TYPE, 0);
if (INT_FAST64_TYPE)
builtin_define_with_value ("__INT_FAST64_TYPE__", INT_FAST64_TYPE, 0);
if (UINT_FAST8_TYPE)
builtin_define_with_value ("__UINT_FAST8_TYPE__", UINT_FAST8_TYPE, 0);
if (UINT_FAST16_TYPE)
builtin_define_with_value ("__UINT_FAST16_TYPE__", UINT_FAST16_TYPE, 0);
if (UINT_FAST32_TYPE)
builtin_define_with_value ("__UINT_FAST32_TYPE__", UINT_FAST32_TYPE, 0);
if (UINT_FAST64_TYPE)
builtin_define_with_value ("__UINT_FAST64_TYPE__", UINT_FAST64_TYPE, 0);
if (INTPTR_TYPE)
builtin_define_with_value ("__INTPTR_TYPE__", INTPTR_TYPE, 0);
if (UINTPTR_TYPE)
builtin_define_with_value ("__UINTPTR_TYPE__", UINTPTR_TYPE, 0);
}
static void
c_init_attributes (void)
{
#define DEF_ATTR_NULL_TREE(ENUM)				\
built_in_attributes[(int) ENUM] = NULL_TREE;
#define DEF_ATTR_INT(ENUM, VALUE)				\
built_in_attributes[(int) ENUM] = build_int_cst (integer_type_node, VALUE);
#define DEF_ATTR_STRING(ENUM, VALUE)				\
built_in_attributes[(int) ENUM] = build_string (strlen (VALUE), VALUE);
#define DEF_ATTR_IDENT(ENUM, STRING)				\
built_in_attributes[(int) ENUM] = get_identifier (STRING);
#define DEF_ATTR_TREE_LIST(ENUM, PURPOSE, VALUE, CHAIN)	\
built_in_attributes[(int) ENUM]			\
= tree_cons (built_in_attributes[(int) PURPOSE],	\
built_in_attributes[(int) VALUE],	\
built_in_attributes[(int) CHAIN]);
#include "builtin-attrs.def"
#undef DEF_ATTR_NULL_TREE
#undef DEF_ATTR_INT
#undef DEF_ATTR_IDENT
#undef DEF_ATTR_TREE_LIST
}
int
check_user_alignment (const_tree align, bool allow_zero)
{
int i;
if (error_operand_p (align))
return -1;
if (TREE_CODE (align) != INTEGER_CST
|| !INTEGRAL_TYPE_P (TREE_TYPE (align)))
{
error ("requested alignment is not an integer constant");
return -1;
}
else if (allow_zero && integer_zerop (align))
return -1;
else if (tree_int_cst_sgn (align) == -1
|| (i = tree_log2 (align)) == -1)
{
error ("requested alignment is not a positive power of 2");
return -1;
}
else if (i >= HOST_BITS_PER_INT - LOG2_BITS_PER_UNIT)
{
error ("requested alignment is too large");
return -1;
}
return i;
}
bool
c_determine_visibility (tree decl)
{
gcc_assert (VAR_OR_FUNCTION_DECL_P (decl));
if (lookup_attribute ("visibility", DECL_ATTRIBUTES (decl))
|| (TARGET_DLLIMPORT_DECL_ATTRIBUTES
&& (lookup_attribute ("dllimport", DECL_ATTRIBUTES (decl))
|| lookup_attribute ("dllexport", DECL_ATTRIBUTES (decl)))))
return true;
if (!DECL_VISIBILITY_SPECIFIED (decl))
{
if (visibility_options.inpragma
|| DECL_VISIBILITY (decl) != default_visibility)
{
DECL_VISIBILITY (decl) = default_visibility;
DECL_VISIBILITY_SPECIFIED (decl) = visibility_options.inpragma;
if (((VAR_P (decl) && TREE_STATIC (decl))
|| TREE_CODE (decl) == FUNCTION_DECL)
&& DECL_RTL_SET_P (decl))
make_decl_rtl (decl);
}
}
return false;
}
struct nonnull_arg_ctx
{
location_t loc;
bool warned_p;
};
static bool
check_function_nonnull (location_t loc, tree attrs, int nargs, tree *argarray)
{
tree a;
int i;
attrs = lookup_attribute ("nonnull", attrs);
if (attrs == NULL_TREE)
return false;
a = attrs;
if (TREE_VALUE (a) != NULL_TREE)
do
a = lookup_attribute ("nonnull", TREE_CHAIN (a));
while (a != NULL_TREE && TREE_VALUE (a) != NULL_TREE);
struct nonnull_arg_ctx ctx = { loc, false };
if (a != NULL_TREE)
for (i = 0; i < nargs; i++)
check_function_arguments_recurse (check_nonnull_arg, &ctx, argarray[i],
i + 1);
else
{
for (i = 0; i < nargs; i++)
{
for (a = attrs; ; a = TREE_CHAIN (a))
{
a = lookup_attribute ("nonnull", a);
if (a == NULL_TREE || nonnull_check_p (TREE_VALUE (a), i + 1))
break;
}
if (a != NULL_TREE)
check_function_arguments_recurse (check_nonnull_arg, &ctx,
argarray[i], i + 1);
}
}
return ctx.warned_p;
}
static void
check_function_sentinel (const_tree fntype, int nargs, tree *argarray)
{
tree attr = lookup_attribute ("sentinel", TYPE_ATTRIBUTES (fntype));
if (attr)
{
int len = 0;
int pos = 0;
tree sentinel;
function_args_iterator iter;
tree t;
FOREACH_FUNCTION_ARGS (fntype, t, iter)
{
if (len == nargs)
break;
len++;
}
if (TREE_VALUE (attr))
{
tree p = TREE_VALUE (TREE_VALUE (attr));
pos = TREE_INT_CST_LOW (p);
}
if ((nargs - 1 - pos) < len)
{
warning (OPT_Wformat_,
"not enough variable arguments to fit a sentinel");
return;
}
sentinel = argarray[nargs - 1 - pos];
if ((!POINTER_TYPE_P (TREE_TYPE (sentinel))
|| !integer_zerop (sentinel))
&& (warn_strict_null_sentinel || null_node != sentinel))
warning (OPT_Wformat_, "missing sentinel in function call");
}
}
static bool
check_function_restrict (const_tree fndecl, const_tree fntype,
int nargs, tree *argarray)
{
int i;
tree parms = TYPE_ARG_TYPES (fntype);
if (fndecl
&& TREE_CODE (fndecl) == FUNCTION_DECL)
{
if (DECL_BUILT_IN (fndecl)
&& DECL_BUILT_IN_CLASS (fndecl) == BUILT_IN_NORMAL
&& nargs == 3
&& TREE_CODE (argarray[2]) == INTEGER_CST
&& integer_zerop (argarray[2]))
return false;
if (DECL_ARGUMENTS (fndecl))
parms = DECL_ARGUMENTS (fndecl);
}
for (i = 0; i < nargs; i++)
TREE_VISITED (argarray[i]) = 0;
bool warned = false;
for (i = 0; i < nargs && parms && parms != void_list_node; i++)
{
tree type;
if (TREE_CODE (parms) == PARM_DECL)
{
type = TREE_TYPE (parms);
parms = DECL_CHAIN (parms);
}
else
{
type = TREE_VALUE (parms);
parms = TREE_CHAIN (parms);
}
if (POINTER_TYPE_P (type)
&& TYPE_RESTRICT (type)
&& !TYPE_READONLY (TREE_TYPE (type)))
warned |= warn_for_restrict (i, argarray, nargs);
}
for (i = 0; i < nargs; i++)
TREE_VISITED (argarray[i]) = 0;
return warned;
}
static bool
nonnull_check_p (tree args, unsigned HOST_WIDE_INT param_num)
{
unsigned HOST_WIDE_INT arg_num = 0;
for (; args; args = TREE_CHAIN (args))
{
bool found = get_nonnull_operand (TREE_VALUE (args), &arg_num);
gcc_assert (found);
if (arg_num == param_num)
return true;
}
return false;
}
static void
check_nonnull_arg (void *ctx, tree param, unsigned HOST_WIDE_INT param_num)
{
struct nonnull_arg_ctx *pctx = (struct nonnull_arg_ctx *) ctx;
if (TREE_CODE (TREE_TYPE (param)) != POINTER_TYPE)
return;
if (integer_zerop (fold_for_warn (param)))
{
warning_at (pctx->loc, OPT_Wnonnull, "null argument where non-null "
"required (argument %lu)", (unsigned long) param_num);
pctx->warned_p = true;
}
}
bool
get_nonnull_operand (tree arg_num_expr, unsigned HOST_WIDE_INT *valp)
{
if (tree_fits_uhwi_p (arg_num_expr))
{
*valp = tree_to_uhwi (arg_num_expr);
return true;
}
else
return false;
}
typedef const char *const_char_p;		
static GTY(()) vec<const_char_p, va_gc> *optimize_args;
bool
parse_optimize_options (tree args, bool attr_p)
{
bool ret = true;
unsigned opt_argc;
unsigned i;
const char **opt_argv;
struct cl_decoded_option *decoded_options;
unsigned int decoded_options_count;
tree ap;
vec_safe_truncate (optimize_args, 0);
vec_safe_push (optimize_args, (const char *) NULL);
for (ap = args; ap != NULL_TREE; ap = TREE_CHAIN (ap))
{
tree value = TREE_VALUE (ap);
if (TREE_CODE (value) == INTEGER_CST)
{
char buffer[20];
sprintf (buffer, "-O%ld", (long) TREE_INT_CST_LOW (value));
vec_safe_push (optimize_args, ggc_strdup (buffer));
}
else if (TREE_CODE (value) == STRING_CST)
{
size_t len = TREE_STRING_LENGTH (value);
char *p = ASTRDUP (TREE_STRING_POINTER (value));
char *end = p + len;
char *comma;
char *next_p = p;
while (next_p != NULL)
{
size_t len2;
char *q, *r;
p = next_p;
comma = strchr (p, ',');
if (comma)
{
len2 = comma - p;
*comma = '\0';
next_p = comma+1;
}
else
{
len2 = end - p;
next_p = NULL;
}
r = q = (char *) ggc_alloc_atomic (len2 + 3);
if (*p == '-' && p[1] != 'O' && p[1] != 'f')
{
ret = false;
if (attr_p)
warning (OPT_Wattributes,
"bad option %qs to attribute %<optimize%>", p);
else
warning (OPT_Wpragmas,
"bad option %qs to pragma %<optimize%>", p);
continue;
}
if (*p != '-')
{
*r++ = '-';
if ((*p >= '0' && *p <= '9')
|| (p[0] == 's' && p[1] == '\0'))
*r++ = 'O';
else if (*p != 'O')
*r++ = 'f';
}
memcpy (r, p, len2);
r[len2] = '\0';
vec_safe_push (optimize_args, (const char *) q);
}
}
}
opt_argc = optimize_args->length ();
opt_argv = (const char **) alloca (sizeof (char *) * (opt_argc + 1));
for (i = 1; i < opt_argc; i++)
opt_argv[i] = (*optimize_args)[i];
decode_cmdline_options_to_array_default_mask (opt_argc, opt_argv,
&decoded_options,
&decoded_options_count);
unsigned j = 1;
for (i = 1; i < decoded_options_count; ++i)
{
if (! (cl_options[decoded_options[i].opt_index].flags & CL_OPTIMIZATION))
{
ret = false;
if (attr_p)
warning (OPT_Wattributes,
"bad option %qs to attribute %<optimize%>",
decoded_options[i].orig_option_with_args_text);
else
warning (OPT_Wpragmas,
"bad option %qs to pragma %<optimize%>",
decoded_options[i].orig_option_with_args_text);
continue;
}
if (i != j)
decoded_options[j] = decoded_options[i];
j++;
}
decoded_options_count = j;
decode_options (&global_options, &global_options_set,
decoded_options, decoded_options_count,
input_location, global_dc, NULL);
targetm.override_options_after_change();
optimize_args->truncate (0);
return ret;
}
bool
attribute_fallthrough_p (tree attr)
{
if (attr == error_mark_node)
return false;
tree t = lookup_attribute ("fallthrough", attr);
if (t == NULL_TREE)
return false;
if (lookup_attribute ("fallthrough", TREE_CHAIN (t)))
warning (OPT_Wattributes, "%<fallthrough%> attribute specified multiple "
"times");
else if (TREE_VALUE (t) != NULL_TREE)
warning (OPT_Wattributes, "%<fallthrough%> attribute specified with "
"a parameter");
for (t = attr; t != NULL_TREE; t = TREE_CHAIN (t))
{
tree name = get_attribute_name (t);
if (!is_attribute_p ("fallthrough", name))
warning (OPT_Wattributes, "%qE attribute ignored", name);
}
return true;
}

bool
check_function_arguments (location_t loc, const_tree fndecl, const_tree fntype,
int nargs, tree *argarray, vec<location_t> *arglocs)
{
bool warned_p = false;
if (warn_nonnull)
warned_p = check_function_nonnull (loc, TYPE_ATTRIBUTES (fntype),
nargs, argarray);
if (warn_format || warn_suggest_attribute_format)
check_function_format (TYPE_ATTRIBUTES (fntype), nargs, argarray, arglocs);
if (warn_format)
check_function_sentinel (fntype, nargs, argarray);
if (warn_restrict)
warned_p |= check_function_restrict (fndecl, fntype, nargs, argarray);
return warned_p;
}
void
check_function_arguments_recurse (void (*callback)
(void *, tree, unsigned HOST_WIDE_INT),
void *ctx, tree param,
unsigned HOST_WIDE_INT param_num)
{
if (CONVERT_EXPR_P (param)
&& (TYPE_PRECISION (TREE_TYPE (param))
== TYPE_PRECISION (TREE_TYPE (TREE_OPERAND (param, 0)))))
{
check_function_arguments_recurse (callback, ctx,
TREE_OPERAND (param, 0), param_num);
return;
}
if (TREE_CODE (param) == CALL_EXPR)
{
tree type = TREE_TYPE (TREE_TYPE (CALL_EXPR_FN (param)));
tree attrs;
bool found_format_arg = false;
for (attrs = TYPE_ATTRIBUTES (type);
attrs;
attrs = TREE_CHAIN (attrs))
if (is_attribute_p ("format_arg", TREE_PURPOSE (attrs)))
{
tree inner_arg;
tree format_num_expr;
int format_num;
int i;
call_expr_arg_iterator iter;
format_num_expr = TREE_VALUE (TREE_VALUE (attrs));
format_num = tree_to_uhwi (format_num_expr);
for (inner_arg = first_call_expr_arg (param, &iter), i = 1;
inner_arg != NULL_TREE;
inner_arg = next_call_expr_arg (&iter), i++)
if (i == format_num)
{
check_function_arguments_recurse (callback, ctx,
inner_arg, param_num);
found_format_arg = true;
break;
}
}
if (found_format_arg)
return;
}
if (TREE_CODE (param) == COND_EXPR)
{
param = fold_for_warn (param);
if (TREE_CODE (param) == COND_EXPR)
{
check_function_arguments_recurse (callback, ctx,
TREE_OPERAND (param, 1),
param_num);
check_function_arguments_recurse (callback, ctx,
TREE_OPERAND (param, 2),
param_num);
return;
}
}
(*callback) (ctx, param, param_num);
}
static bool
builtin_function_validate_nargs (location_t loc, tree fndecl, int nargs,
int required)
{
if (nargs < required)
{
error_at (loc, "too few arguments to function %qE", fndecl);
return false;
}
else if (nargs > required)
{
error_at (loc, "too many arguments to function %qE", fndecl);
return false;
}
return true;
}
#define ARG_LOCATION(N)					\
(arg_loc.is_empty ()					\
? EXPR_LOC_OR_LOC (args[(N)], input_location)	\
: expansion_point_location (arg_loc[(N)]))
bool
check_builtin_function_arguments (location_t loc, vec<location_t> arg_loc,
tree fndecl, int nargs, tree *args)
{
if (!DECL_BUILT_IN (fndecl)
|| DECL_BUILT_IN_CLASS (fndecl) != BUILT_IN_NORMAL)
return true;
switch (DECL_FUNCTION_CODE (fndecl))
{
case BUILT_IN_ALLOCA_WITH_ALIGN_AND_MAX:
if (!tree_fits_uhwi_p (args[2]))
{
error_at (ARG_LOCATION (2),
"third argument to function %qE must be a constant integer",
fndecl);
return false;
}
case BUILT_IN_ALLOCA_WITH_ALIGN:
{
unsigned HOST_WIDE_INT align
= tree_fits_uhwi_p (args[1]) ? tree_to_uhwi (args[1]) : 0;
if ((align & (align - 1)))
align = 0;
unsigned maxalign = (UINT_MAX >> 1) + 1;
if (align < BITS_PER_UNIT || maxalign < align)
{
error_at (ARG_LOCATION (1),
"second argument to function %qE must be a constant "
"integer power of 2 between %qi and %qu bits",
fndecl, BITS_PER_UNIT, maxalign);
return false;
}
return true;
}
case BUILT_IN_CONSTANT_P:
return builtin_function_validate_nargs (loc, fndecl, nargs, 1);
case BUILT_IN_ISFINITE:
case BUILT_IN_ISINF:
case BUILT_IN_ISINF_SIGN:
case BUILT_IN_ISNAN:
case BUILT_IN_ISNORMAL:
case BUILT_IN_SIGNBIT:
if (builtin_function_validate_nargs (loc, fndecl, nargs, 1))
{
if (TREE_CODE (TREE_TYPE (args[0])) != REAL_TYPE)
{
error_at (ARG_LOCATION (0), "non-floating-point argument in "
"call to function %qE", fndecl);
return false;
}
return true;
}
return false;
case BUILT_IN_ISGREATER:
case BUILT_IN_ISGREATEREQUAL:
case BUILT_IN_ISLESS:
case BUILT_IN_ISLESSEQUAL:
case BUILT_IN_ISLESSGREATER:
case BUILT_IN_ISUNORDERED:
if (builtin_function_validate_nargs (loc, fndecl, nargs, 2))
{
enum tree_code code0, code1;
code0 = TREE_CODE (TREE_TYPE (args[0]));
code1 = TREE_CODE (TREE_TYPE (args[1]));
if (!((code0 == REAL_TYPE && code1 == REAL_TYPE)
|| (code0 == REAL_TYPE && code1 == INTEGER_TYPE)
|| (code0 == INTEGER_TYPE && code1 == REAL_TYPE)))
{
error_at (loc, "non-floating-point arguments in call to "
"function %qE", fndecl);
return false;
}
return true;
}
return false;
case BUILT_IN_FPCLASSIFY:
if (builtin_function_validate_nargs (loc, fndecl, nargs, 6))
{
for (unsigned int i = 0; i < 5; i++)
if (TREE_CODE (args[i]) != INTEGER_CST)
{
error_at (ARG_LOCATION (i), "non-const integer argument %u in "
"call to function %qE", i + 1, fndecl);
return false;
}
if (TREE_CODE (TREE_TYPE (args[5])) != REAL_TYPE)
{
error_at (ARG_LOCATION (5), "non-floating-point argument in "
"call to function %qE", fndecl);
return false;
}
return true;
}
return false;
case BUILT_IN_ASSUME_ALIGNED:
if (builtin_function_validate_nargs (loc, fndecl, nargs, 2 + (nargs > 2)))
{
if (nargs >= 3 && TREE_CODE (TREE_TYPE (args[2])) != INTEGER_TYPE)
{
error_at (ARG_LOCATION (2), "non-integer argument 3 in call to "
"function %qE", fndecl);
return false;
}
return true;
}
return false;
case BUILT_IN_ADD_OVERFLOW:
case BUILT_IN_SUB_OVERFLOW:
case BUILT_IN_MUL_OVERFLOW:
if (builtin_function_validate_nargs (loc, fndecl, nargs, 3))
{
unsigned i;
for (i = 0; i < 2; i++)
if (!INTEGRAL_TYPE_P (TREE_TYPE (args[i])))
{
error_at (ARG_LOCATION (i), "argument %u in call to function "
"%qE does not have integral type", i + 1, fndecl);
return false;
}
if (TREE_CODE (TREE_TYPE (args[2])) != POINTER_TYPE
|| !INTEGRAL_TYPE_P (TREE_TYPE (TREE_TYPE (args[2]))))
{
error_at (ARG_LOCATION (2), "argument 3 in call to function %qE "
"does not have pointer to integral type", fndecl);
return false;
}
else if (TREE_CODE (TREE_TYPE (TREE_TYPE (args[2]))) == ENUMERAL_TYPE)
{
error_at (ARG_LOCATION (2), "argument 3 in call to function %qE "
"has pointer to enumerated type", fndecl);
return false;
}
else if (TREE_CODE (TREE_TYPE (TREE_TYPE (args[2]))) == BOOLEAN_TYPE)
{
error_at (ARG_LOCATION (2), "argument 3 in call to function %qE "
"has pointer to boolean type", fndecl);
return false;
}
return true;
}
return false;
case BUILT_IN_ADD_OVERFLOW_P:
case BUILT_IN_SUB_OVERFLOW_P:
case BUILT_IN_MUL_OVERFLOW_P:
if (builtin_function_validate_nargs (loc, fndecl, nargs, 3))
{
unsigned i;
for (i = 0; i < 3; i++)
if (!INTEGRAL_TYPE_P (TREE_TYPE (args[i])))
{
error_at (ARG_LOCATION (i), "argument %u in call to function "
"%qE does not have integral type", i + 1, fndecl);
return false;
}
if (TREE_CODE (TREE_TYPE (args[2])) == ENUMERAL_TYPE)
{
error_at (ARG_LOCATION (2), "argument 3 in call to function "
"%qE has enumerated type", fndecl);
return false;
}
else if (TREE_CODE (TREE_TYPE (args[2])) == BOOLEAN_TYPE)
{
error_at (ARG_LOCATION (2), "argument 3 in call to function "
"%qE has boolean type", fndecl);
return false;
}
return true;
}
return false;
default:
return true;
}
}
static char *
catenate_strings (const char *lhs, const char *rhs_start, int rhs_size)
{
const size_t lhs_size = strlen (lhs);
char *result = XNEWVEC (char, lhs_size + rhs_size);
memcpy (result, lhs, lhs_size);
memcpy (result + lhs_size, rhs_start, rhs_size);
return result;
}
void
c_parse_error (const char *gmsgid, enum cpp_ttype token_type,
tree value, unsigned char token_flags,
rich_location *richloc)
{
#define catenate_messages(M1, M2) catenate_strings ((M1), (M2), sizeof (M2))
char *message = NULL;
if (token_type == CPP_EOF)
message = catenate_messages (gmsgid, " at end of input");
else if (token_type == CPP_CHAR
|| token_type == CPP_WCHAR
|| token_type == CPP_CHAR16
|| token_type == CPP_CHAR32
|| token_type == CPP_UTF8CHAR)
{
unsigned int val = TREE_INT_CST_LOW (value);
const char *prefix;
switch (token_type)
{
default:
prefix = "";
break;
case CPP_WCHAR:
prefix = "L";
break;
case CPP_CHAR16:
prefix = "u";
break;
case CPP_CHAR32:
prefix = "U";
break;
case CPP_UTF8CHAR:
prefix = "u8";
break;
}
if (val <= UCHAR_MAX && ISGRAPH (val))
message = catenate_messages (gmsgid, " before %s'%c'");
else
message = catenate_messages (gmsgid, " before %s'\\x%x'");
error_at (richloc, message, prefix, val);
free (message);
message = NULL;
}
else if (token_type == CPP_CHAR_USERDEF
|| token_type == CPP_WCHAR_USERDEF
|| token_type == CPP_CHAR16_USERDEF
|| token_type == CPP_CHAR32_USERDEF
|| token_type == CPP_UTF8CHAR_USERDEF)
message = catenate_messages (gmsgid,
" before user-defined character literal");
else if (token_type == CPP_STRING_USERDEF
|| token_type == CPP_WSTRING_USERDEF
|| token_type == CPP_STRING16_USERDEF
|| token_type == CPP_STRING32_USERDEF
|| token_type == CPP_UTF8STRING_USERDEF)
message = catenate_messages (gmsgid, " before user-defined string literal");
else if (token_type == CPP_STRING
|| token_type == CPP_WSTRING
|| token_type == CPP_STRING16
|| token_type == CPP_STRING32
|| token_type == CPP_UTF8STRING)
message = catenate_messages (gmsgid, " before string constant");
else if (token_type == CPP_NUMBER)
message = catenate_messages (gmsgid, " before numeric constant");
else if (token_type == CPP_NAME)
{
message = catenate_messages (gmsgid, " before %qE");
error_at (richloc, message, value);
free (message);
message = NULL;
}
else if (token_type == CPP_PRAGMA)
message = catenate_messages (gmsgid, " before %<#pragma%>");
else if (token_type == CPP_PRAGMA_EOL)
message = catenate_messages (gmsgid, " before end of line");
else if (token_type == CPP_DECLTYPE)
message = catenate_messages (gmsgid, " before %<decltype%>");
else if (token_type < N_TTYPES)
{
message = catenate_messages (gmsgid, " before %qs token");
error_at (richloc, message, cpp_type2name (token_type, token_flags));
free (message);
message = NULL;
}
else
error_at (richloc, gmsgid);
if (message)
{
error_at (richloc, message);
free (message);
}
#undef catenate_messages
}
static int
c_option_controlling_cpp_error (int reason)
{
const struct cpp_reason_option_codes_t *entry;
for (entry = cpp_reason_option_codes; entry->reason != CPP_W_NONE; entry++)
{
if (entry->reason == reason)
return entry->option_code;
}
return 0;
}
bool
c_cpp_error (cpp_reader *pfile ATTRIBUTE_UNUSED, int level, int reason,
rich_location *richloc,
const char *msg, va_list *ap)
{
diagnostic_info diagnostic;
diagnostic_t dlevel;
bool save_warn_system_headers = global_dc->dc_warn_system_headers;
bool ret;
switch (level)
{
case CPP_DL_WARNING_SYSHDR:
if (flag_no_output)
return false;
global_dc->dc_warn_system_headers = 1;
case CPP_DL_WARNING:
if (flag_no_output)
return false;
dlevel = DK_WARNING;
break;
case CPP_DL_PEDWARN:
if (flag_no_output && !flag_pedantic_errors)
return false;
dlevel = DK_PEDWARN;
break;
case CPP_DL_ERROR:
dlevel = DK_ERROR;
break;
case CPP_DL_ICE:
dlevel = DK_ICE;
break;
case CPP_DL_NOTE:
dlevel = DK_NOTE;
break;
case CPP_DL_FATAL:
dlevel = DK_FATAL;
break;
default:
gcc_unreachable ();
}
if (done_lexing)
richloc->set_range (line_table, 0, input_location, true);
diagnostic_set_info_translated (&diagnostic, msg, ap,
richloc, dlevel);
diagnostic_override_option_index (&diagnostic,
c_option_controlling_cpp_error (reason));
ret = diagnostic_report_diagnostic (global_dc, &diagnostic);
if (level == CPP_DL_WARNING_SYSHDR)
global_dc->dc_warn_system_headers = save_warn_system_headers;
return ret;
}
HOST_WIDE_INT
c_common_to_target_charset (HOST_WIDE_INT c)
{
cppchar_t uc = ((cppchar_t)c) & ((((cppchar_t)1) << CHAR_BIT)-1);
uc = cpp_host_to_exec_charset (parse_in, uc);
if (flag_signed_char)
return ((HOST_WIDE_INT)uc) << (HOST_BITS_PER_WIDE_INT - CHAR_TYPE_SIZE)
>> (HOST_BITS_PER_WIDE_INT - CHAR_TYPE_SIZE);
else
return uc;
}
tree
fold_offsetof (tree expr, tree type, enum tree_code ctx)
{
tree base, off, t;
tree_code code = TREE_CODE (expr);
switch (code)
{
case ERROR_MARK:
return expr;
case VAR_DECL:
error ("cannot apply %<offsetof%> to static data member %qD", expr);
return error_mark_node;
case CALL_EXPR:
case TARGET_EXPR:
error ("cannot apply %<offsetof%> when %<operator[]%> is overloaded");
return error_mark_node;
case NOP_EXPR:
case INDIRECT_REF:
if (!TREE_CONSTANT (TREE_OPERAND (expr, 0)))
{
error ("cannot apply %<offsetof%> to a non constant address");
return error_mark_node;
}
return convert (type, TREE_OPERAND (expr, 0));
case COMPONENT_REF:
base = fold_offsetof (TREE_OPERAND (expr, 0), type, code);
if (base == error_mark_node)
return base;
t = TREE_OPERAND (expr, 1);
if (DECL_C_BIT_FIELD (t))
{
error ("attempt to take address of bit-field structure "
"member %qD", t);
return error_mark_node;
}
off = size_binop_loc (input_location, PLUS_EXPR, DECL_FIELD_OFFSET (t),
size_int (tree_to_uhwi (DECL_FIELD_BIT_OFFSET (t))
/ BITS_PER_UNIT));
break;
case ARRAY_REF:
base = fold_offsetof (TREE_OPERAND (expr, 0), type, code);
if (base == error_mark_node)
return base;
t = TREE_OPERAND (expr, 1);
if (TREE_CODE (t) == INTEGER_CST && tree_int_cst_sgn (t) >= 0)
{
tree upbound = array_ref_up_bound (expr);
if (upbound != NULL_TREE
&& TREE_CODE (upbound) == INTEGER_CST
&& !tree_int_cst_equal (upbound,
TYPE_MAX_VALUE (TREE_TYPE (upbound))))
{
if (ctx != ARRAY_REF && ctx != COMPONENT_REF)
upbound = size_binop (PLUS_EXPR, upbound,
build_int_cst (TREE_TYPE (upbound), 1));
if (tree_int_cst_lt (upbound, t))
{
tree v;
for (v = TREE_OPERAND (expr, 0);
TREE_CODE (v) == COMPONENT_REF;
v = TREE_OPERAND (v, 0))
if (TREE_CODE (TREE_TYPE (TREE_OPERAND (v, 0)))
== RECORD_TYPE)
{
tree fld_chain = DECL_CHAIN (TREE_OPERAND (v, 1));
for (; fld_chain; fld_chain = DECL_CHAIN (fld_chain))
if (TREE_CODE (fld_chain) == FIELD_DECL)
break;
if (fld_chain)
break;
}
if (TREE_CODE (v) == ARRAY_REF
|| TREE_CODE (v) == COMPONENT_REF)
warning (OPT_Warray_bounds,
"index %E denotes an offset "
"greater than size of %qT",
t, TREE_TYPE (TREE_OPERAND (expr, 0)));
}
}
}
t = convert (sizetype, t);
off = size_binop (MULT_EXPR, TYPE_SIZE_UNIT (TREE_TYPE (expr)), t);
break;
case COMPOUND_EXPR:
t = TREE_OPERAND (expr, 1);
gcc_checking_assert (VAR_P (get_base_address (t)));
return fold_offsetof (t, type);
default:
gcc_unreachable ();
}
if (!POINTER_TYPE_P (type))
return size_binop (PLUS_EXPR, base, convert (type, off));
return fold_build_pointer_plus (base, off);
}

int
complete_array_type (tree *ptype, tree initial_value, bool do_default)
{
tree maxindex, type, main_type, elt, unqual_elt;
int failure = 0, quals;
bool overflow_p = false;
maxindex = size_zero_node;
if (initial_value)
{
if (TREE_CODE (initial_value) == STRING_CST)
{
int eltsize
= int_size_in_bytes (TREE_TYPE (TREE_TYPE (initial_value)));
maxindex = size_int (TREE_STRING_LENGTH (initial_value)/eltsize - 1);
}
else if (TREE_CODE (initial_value) == CONSTRUCTOR)
{
vec<constructor_elt, va_gc> *v = CONSTRUCTOR_ELTS (initial_value);
if (vec_safe_is_empty (v))
{
if (pedantic)
failure = 3;
maxindex = ssize_int (-1);
}
else
{
tree curindex;
unsigned HOST_WIDE_INT cnt;
constructor_elt *ce;
bool fold_p = false;
if ((*v)[0].index)
maxindex = (*v)[0].index, fold_p = true;
curindex = maxindex;
for (cnt = 1; vec_safe_iterate (v, cnt, &ce); cnt++)
{
bool curfold_p = false;
if (ce->index)
curindex = ce->index, curfold_p = true;
else
{
if (fold_p)
{
tree orig = curindex;
curindex = fold_convert (sizetype, curindex);
overflow_p |= tree_int_cst_lt (curindex, orig);
}
curindex = size_binop (PLUS_EXPR, curindex,
size_one_node);
}
if (tree_int_cst_lt (maxindex, curindex))
maxindex = curindex, fold_p = curfold_p;
}
if (fold_p)
{
tree orig = maxindex;
maxindex = fold_convert (sizetype, maxindex);
overflow_p |= tree_int_cst_lt (maxindex, orig);
}
}
}
else
{
if (initial_value != error_mark_node)
failure = 1;
}
}
else
{
failure = 2;
if (!do_default)
return failure;
}
type = *ptype;
elt = TREE_TYPE (type);
quals = TYPE_QUALS (strip_array_types (elt));
if (quals == 0)
unqual_elt = elt;
else
unqual_elt = c_build_qualified_type (elt, KEEP_QUAL_ADDR_SPACE (quals));
main_type = build_distinct_type_copy (TYPE_MAIN_VARIANT (type));
TREE_TYPE (main_type) = unqual_elt;
TYPE_DOMAIN (main_type)
= build_range_type (TREE_TYPE (maxindex),
build_int_cst (TREE_TYPE (maxindex), 0), maxindex);
TYPE_TYPELESS_STORAGE (main_type) = TYPE_TYPELESS_STORAGE (type);
layout_type (main_type);
hashval_t hashcode = type_hash_canon_hash (main_type);
main_type = type_hash_canon (hashcode, main_type);
if (TYPE_STRUCTURAL_EQUALITY_P (TREE_TYPE (main_type))
|| TYPE_STRUCTURAL_EQUALITY_P (TYPE_DOMAIN (main_type)))
SET_TYPE_STRUCTURAL_EQUALITY (main_type);
else if (TYPE_CANONICAL (TREE_TYPE (main_type)) != TREE_TYPE (main_type)
|| (TYPE_CANONICAL (TYPE_DOMAIN (main_type))
!= TYPE_DOMAIN (main_type)))
TYPE_CANONICAL (main_type)
= build_array_type (TYPE_CANONICAL (TREE_TYPE (main_type)),
TYPE_CANONICAL (TYPE_DOMAIN (main_type)),
TYPE_TYPELESS_STORAGE (main_type));
else
TYPE_CANONICAL (main_type) = main_type;
if (quals == 0)
type = main_type;
else
type = c_build_qualified_type (main_type, quals);
if (COMPLETE_TYPE_P (type)
&& TREE_CODE (TYPE_SIZE_UNIT (type)) == INTEGER_CST
&& (overflow_p || TREE_OVERFLOW (TYPE_SIZE_UNIT (type))))
{
error ("size of array is too large");
type = error_mark_node;
}
*ptype = type;
return failure;
}
void 
c_common_mark_addressable_vec (tree t)
{   
if (TREE_CODE (t) == C_MAYBE_CONST_EXPR)
t = C_MAYBE_CONST_EXPR_EXPR (t);
while (handled_component_p (t))
t = TREE_OPERAND (t, 0);
if (!VAR_P (t)
&& TREE_CODE (t) != PARM_DECL
&& TREE_CODE (t) != COMPOUND_LITERAL_EXPR)
return;
if (!VAR_P (t) || !DECL_HARD_REGISTER (t))
TREE_ADDRESSABLE (t) = 1;
}

tree
builtin_type_for_size (int size, bool unsignedp)
{
tree type = c_common_type_for_size (size, unsignedp);
return type ? type : error_mark_node;
}
static int
sync_resolve_size (tree function, vec<tree, va_gc> *params, bool fetch)
{
tree argtype;
tree type;
int size;
if (vec_safe_is_empty (params))
{
error ("too few arguments to function %qE", function);
return 0;
}
argtype = type = TREE_TYPE ((*params)[0]);
if (TREE_CODE (type) == ARRAY_TYPE && c_dialect_cxx ())
{
(*params)[0] = default_conversion ((*params)[0]);
type = TREE_TYPE ((*params)[0]);
}
if (TREE_CODE (type) != POINTER_TYPE)
goto incompatible;
type = TREE_TYPE (type);
if (!INTEGRAL_TYPE_P (type) && !POINTER_TYPE_P (type))
goto incompatible;
if (!COMPLETE_TYPE_P (type))
goto incompatible;
if (fetch && TREE_CODE (type) == BOOLEAN_TYPE)
goto incompatible;
size = tree_to_uhwi (TYPE_SIZE_UNIT (type));
if (size == 1 || size == 2 || size == 4 || size == 8 || size == 16)
return size;
incompatible:
if (argtype != error_mark_node)
error ("operand type %qT is incompatible with argument %d of %qE",
argtype, 1, function);
return 0;
}
static bool
sync_resolve_params (location_t loc, tree orig_function, tree function,
vec<tree, va_gc> *params, bool orig_format)
{
function_args_iterator iter;
tree ptype;
unsigned int parmnum;
function_args_iter_init (&iter, TREE_TYPE (function));
function_args_iter_next (&iter);
ptype = TREE_TYPE (TREE_TYPE ((*params)[0]));
ptype = TYPE_MAIN_VARIANT (ptype);
parmnum = 0;
while (1)
{
tree val, arg_type;
arg_type = function_args_iter_cond (&iter);
if (arg_type == void_type_node)
break;
++parmnum;
if (params->length () <= parmnum)
{
error_at (loc, "too few arguments to function %qE", orig_function);
return false;
}
if (TREE_CODE (arg_type) == INTEGER_TYPE && TYPE_UNSIGNED (arg_type))
{
val = (*params)[parmnum];
val = convert (ptype, val);
val = convert (arg_type, val);
(*params)[parmnum] = val;
}
function_args_iter_next (&iter);
}
if (!orig_format && params->length () != parmnum + 1)
{
error_at (loc, "too many arguments to function %qE", orig_function);
return false;
}
params->truncate (parmnum + 1);
return true;
}
static tree
sync_resolve_return (tree first_param, tree result, bool orig_format)
{
tree ptype = TREE_TYPE (TREE_TYPE (first_param));
tree rtype = TREE_TYPE (result);
ptype = TYPE_MAIN_VARIANT (ptype);
if (orig_format || tree_int_cst_equal (TYPE_SIZE (ptype), TYPE_SIZE (rtype)))
return convert (ptype, result);
else
return result;
}
static int
get_atomic_generic_size (location_t loc, tree function,
vec<tree, va_gc> *params)
{
unsigned int n_param;
unsigned int n_model;
unsigned int x;
int size_0;
tree type_0;
switch (DECL_FUNCTION_CODE (function))
{
case BUILT_IN_ATOMIC_EXCHANGE:
n_param = 4;
n_model = 1;
break;
case BUILT_IN_ATOMIC_LOAD:
case BUILT_IN_ATOMIC_STORE:
n_param = 3;
n_model = 1;
break;
case BUILT_IN_ATOMIC_COMPARE_EXCHANGE:
n_param = 6;
n_model = 2;
break;
default:
gcc_unreachable ();
}
if (vec_safe_length (params) != n_param)
{
error_at (loc, "incorrect number of arguments to function %qE", function);
return 0;
}
type_0 = TREE_TYPE ((*params)[0]);
if (TREE_CODE (type_0) == ARRAY_TYPE && c_dialect_cxx ())
{
(*params)[0] = default_conversion ((*params)[0]);
type_0 = TREE_TYPE ((*params)[0]);
}
if (TREE_CODE (type_0) != POINTER_TYPE || VOID_TYPE_P (TREE_TYPE (type_0)))
{
error_at (loc, "argument 1 of %qE must be a non-void pointer type",
function);
return 0;
}
if (TREE_CODE ((TYPE_SIZE_UNIT (TREE_TYPE (type_0)))) != INTEGER_CST)
{
error_at (loc, 
"argument 1 of %qE must be a pointer to a constant size type",
function);
return 0;
}
size_0 = tree_to_uhwi (TYPE_SIZE_UNIT (TREE_TYPE (type_0)));
if (size_0 == 0)
{
error_at (loc, 
"argument 1 of %qE must be a pointer to a nonzero size object",
function);
return 0;
}
for (x = 0; x < n_param - n_model; x++)
{
int size;
tree type = TREE_TYPE ((*params)[x]);
if (n_param == 6 && x == 3)
continue;
if (TREE_CODE (type) == ARRAY_TYPE && c_dialect_cxx ())
{
(*params)[x] = default_conversion ((*params)[x]);
type = TREE_TYPE ((*params)[x]);
}
if (!POINTER_TYPE_P (type))
{
error_at (loc, "argument %d of %qE must be a pointer type", x + 1,
function);
return 0;
}
else if (TYPE_SIZE_UNIT (TREE_TYPE (type))
&& TREE_CODE ((TYPE_SIZE_UNIT (TREE_TYPE (type))))
!= INTEGER_CST)
{
error_at (loc, "argument %d of %qE must be a pointer to a constant "
"size type", x + 1, function);
return 0;
}
else if (FUNCTION_POINTER_TYPE_P (type))
{
error_at (loc, "argument %d of %qE must not be a pointer to a "
"function", x + 1, function);
return 0;
}
tree type_size = TYPE_SIZE_UNIT (TREE_TYPE (type));
size = type_size ? tree_to_uhwi (type_size) : 0;
if (size != size_0)
{
error_at (loc, "size mismatch in argument %d of %qE", x + 1,
function);
return 0;
}
}
for (x = n_param - n_model ; x < n_param; x++)
{
tree p = (*params)[x];
if (!INTEGRAL_TYPE_P (TREE_TYPE (p)))
{
error_at (loc, "non-integer memory model argument %d of %qE", x + 1,
function);
return 0;
}
p = fold_for_warn (p);
if (TREE_CODE (p) == INTEGER_CST)
{
if (memmodel_base (TREE_INT_CST_LOW (p)) >= MEMMODEL_LAST)
warning_at (loc, OPT_Winvalid_memory_model,
"invalid memory model argument %d of %qE", x + 1,
function);
}
}
return size_0;
}
static tree
add_atomic_size_parameter (unsigned n, location_t loc, tree function, 
vec<tree, va_gc> *params)
{
tree size_node;
if (!params->space (1))
{
unsigned int z, len;
vec<tree, va_gc> *v;
tree f;
len = params->length ();
vec_alloc (v, len + 1);
v->quick_push (build_int_cst (size_type_node, n));
for (z = 0; z < len; z++)
v->quick_push ((*params)[z]);
f = build_function_call_vec (loc, vNULL, function, v, NULL);
vec_free (v);
return f;
}
size_node = build_int_cst (size_type_node, n);
params->quick_insert (0, size_node);
return NULL_TREE;
}
static bool
atomic_size_supported_p (int n)
{
switch (n)
{
case 1:
case 2:
case 4:
case 8:
return true;
case 16:
return targetm.scalar_mode_supported_p (TImode);
default:
return false;
}
}
static bool
resolve_overloaded_atomic_exchange (location_t loc, tree function, 
vec<tree, va_gc> *params, tree *new_return)
{	
tree p0, p1, p2, p3;
tree I_type, I_type_ptr;
int n = get_atomic_generic_size (loc, function, params);
if (n == 0)
{
*new_return = error_mark_node;
return true;
}
if (!atomic_size_supported_p (n))
{
*new_return = add_atomic_size_parameter (n, loc, function, params);
return true;
}
p0 = (*params)[0];
p1 = (*params)[1];
p2 = (*params)[2];
p3 = (*params)[3];
I_type = builtin_type_for_size (BITS_PER_UNIT * n, 1);
I_type_ptr = build_pointer_type (I_type);
p0 = build1 (VIEW_CONVERT_EXPR, I_type_ptr, p0);
(*params)[0] = p0; 
p1 = build_indirect_ref (loc, p1, RO_UNARY_STAR);
p1 = build1 (VIEW_CONVERT_EXPR, I_type, p1);
(*params)[1] = p1;
(*params)[2] = p3;
params->truncate (3);
*new_return = build_indirect_ref (loc, p2, RO_UNARY_STAR);
return false;
}
static bool
resolve_overloaded_atomic_compare_exchange (location_t loc, tree function, 
vec<tree, va_gc> *params, 
tree *new_return)
{	
tree p0, p1, p2;
tree I_type, I_type_ptr;
int n = get_atomic_generic_size (loc, function, params);
if (n == 0)
{
*new_return = error_mark_node;
return true;
}
if (!atomic_size_supported_p (n))
{
if (n > 0)
{
(*params)[3] = (*params)[4];
(*params)[4] = (*params)[5];
params->truncate (5);
}
*new_return = add_atomic_size_parameter (n, loc, function, params);
return true;
}
p0 = (*params)[0];
p1 = (*params)[1];
p2 = (*params)[2];
I_type = builtin_type_for_size (BITS_PER_UNIT * n, 1);
I_type_ptr = build_pointer_type (I_type);
p0 = build1 (VIEW_CONVERT_EXPR, I_type_ptr, p0);
(*params)[0] = p0;
p1 = build1 (VIEW_CONVERT_EXPR, I_type_ptr, p1);
(*params)[1] = p1;
p2 = build_indirect_ref (loc, p2, RO_UNARY_STAR);
p2 = build1 (VIEW_CONVERT_EXPR, I_type, p2);
(*params)[2] = p2;
*new_return = NULL;
return false;
}
static bool
resolve_overloaded_atomic_load (location_t loc, tree function, 
vec<tree, va_gc> *params, tree *new_return)
{	
tree p0, p1, p2;
tree I_type, I_type_ptr;
int n = get_atomic_generic_size (loc, function, params);
if (n == 0)
{
*new_return = error_mark_node;
return true;
}
if (!atomic_size_supported_p (n))
{
*new_return = add_atomic_size_parameter (n, loc, function, params);
return true;
}
p0 = (*params)[0];
p1 = (*params)[1];
p2 = (*params)[2];
I_type = builtin_type_for_size (BITS_PER_UNIT * n, 1);
I_type_ptr = build_pointer_type (I_type);
p0 = build1 (VIEW_CONVERT_EXPR, I_type_ptr, p0);
(*params)[0] = p0;
(*params)[1] = p2;
params->truncate (2);
*new_return = build_indirect_ref (loc, p1, RO_UNARY_STAR);
return false;
}
static bool
resolve_overloaded_atomic_store (location_t loc, tree function, 
vec<tree, va_gc> *params, tree *new_return)
{	
tree p0, p1;
tree I_type, I_type_ptr;
int n = get_atomic_generic_size (loc, function, params);
if (n == 0)
{
*new_return = error_mark_node;
return true;
}
if (!atomic_size_supported_p (n))
{
*new_return = add_atomic_size_parameter (n, loc, function, params);
return true;
}
p0 = (*params)[0];
p1 = (*params)[1];
I_type = builtin_type_for_size (BITS_PER_UNIT * n, 1);
I_type_ptr = build_pointer_type (I_type);
p0 = build1 (VIEW_CONVERT_EXPR, I_type_ptr, p0);
(*params)[0] = p0;
p1 = build_indirect_ref (loc, p1, RO_UNARY_STAR);
p1 = build1 (VIEW_CONVERT_EXPR, I_type, p1);
(*params)[1] = p1;
*new_return = NULL_TREE;
return false;
}
tree
resolve_overloaded_builtin (location_t loc, tree function,
vec<tree, va_gc> *params)
{
enum built_in_function orig_code = DECL_FUNCTION_CODE (function);
bool fetch_op = true;
bool orig_format = true;
tree new_return = NULL_TREE;
switch (DECL_BUILT_IN_CLASS (function))
{
case BUILT_IN_NORMAL:
break;
case BUILT_IN_MD:
if (targetm.resolve_overloaded_builtin)
return targetm.resolve_overloaded_builtin (loc, function, params);
else
return NULL_TREE;
default:
return NULL_TREE;
}
switch (orig_code)
{
case BUILT_IN_ATOMIC_EXCHANGE:
case BUILT_IN_ATOMIC_COMPARE_EXCHANGE:
case BUILT_IN_ATOMIC_LOAD:
case BUILT_IN_ATOMIC_STORE:
{
switch (orig_code)
{
case BUILT_IN_ATOMIC_EXCHANGE:
{
if (resolve_overloaded_atomic_exchange (loc, function, params,
&new_return))
return new_return;
orig_code = BUILT_IN_ATOMIC_EXCHANGE_N;
break;
}
case BUILT_IN_ATOMIC_COMPARE_EXCHANGE:
{
if (resolve_overloaded_atomic_compare_exchange (loc, function,
params,
&new_return))
return new_return;
orig_code = BUILT_IN_ATOMIC_COMPARE_EXCHANGE_N;
break;
}
case BUILT_IN_ATOMIC_LOAD:
{
if (resolve_overloaded_atomic_load (loc, function, params,
&new_return))
return new_return;
orig_code = BUILT_IN_ATOMIC_LOAD_N;
break;
}
case BUILT_IN_ATOMIC_STORE:
{
if (resolve_overloaded_atomic_store (loc, function, params,
&new_return))
return new_return;
orig_code = BUILT_IN_ATOMIC_STORE_N;
break;
}
default:
gcc_unreachable ();
}
}
case BUILT_IN_ATOMIC_EXCHANGE_N:
case BUILT_IN_ATOMIC_COMPARE_EXCHANGE_N:
case BUILT_IN_ATOMIC_LOAD_N:
case BUILT_IN_ATOMIC_STORE_N:
fetch_op = false;
case BUILT_IN_ATOMIC_ADD_FETCH_N:
case BUILT_IN_ATOMIC_SUB_FETCH_N:
case BUILT_IN_ATOMIC_AND_FETCH_N:
case BUILT_IN_ATOMIC_NAND_FETCH_N:
case BUILT_IN_ATOMIC_XOR_FETCH_N:
case BUILT_IN_ATOMIC_OR_FETCH_N:
case BUILT_IN_ATOMIC_FETCH_ADD_N:
case BUILT_IN_ATOMIC_FETCH_SUB_N:
case BUILT_IN_ATOMIC_FETCH_AND_N:
case BUILT_IN_ATOMIC_FETCH_NAND_N:
case BUILT_IN_ATOMIC_FETCH_XOR_N:
case BUILT_IN_ATOMIC_FETCH_OR_N:
orig_format = false;
case BUILT_IN_SYNC_FETCH_AND_ADD_N:
case BUILT_IN_SYNC_FETCH_AND_SUB_N:
case BUILT_IN_SYNC_FETCH_AND_OR_N:
case BUILT_IN_SYNC_FETCH_AND_AND_N:
case BUILT_IN_SYNC_FETCH_AND_XOR_N:
case BUILT_IN_SYNC_FETCH_AND_NAND_N:
case BUILT_IN_SYNC_ADD_AND_FETCH_N:
case BUILT_IN_SYNC_SUB_AND_FETCH_N:
case BUILT_IN_SYNC_OR_AND_FETCH_N:
case BUILT_IN_SYNC_AND_AND_FETCH_N:
case BUILT_IN_SYNC_XOR_AND_FETCH_N:
case BUILT_IN_SYNC_NAND_AND_FETCH_N:
case BUILT_IN_SYNC_BOOL_COMPARE_AND_SWAP_N:
case BUILT_IN_SYNC_VAL_COMPARE_AND_SWAP_N:
case BUILT_IN_SYNC_LOCK_TEST_AND_SET_N:
case BUILT_IN_SYNC_LOCK_RELEASE_N:
{
if (fetch_op)
fetch_op =
(orig_code != BUILT_IN_SYNC_BOOL_COMPARE_AND_SWAP_N
&& orig_code != BUILT_IN_SYNC_VAL_COMPARE_AND_SWAP_N
&& orig_code != BUILT_IN_SYNC_LOCK_TEST_AND_SET_N
&& orig_code != BUILT_IN_SYNC_LOCK_RELEASE_N);
int n = sync_resolve_size (function, params, fetch_op);
tree new_function, first_param, result;
enum built_in_function fncode;
if (n == 0)
return error_mark_node;
fncode = (enum built_in_function)((int)orig_code + exact_log2 (n) + 1);
new_function = builtin_decl_explicit (fncode);
if (!sync_resolve_params (loc, function, new_function, params,
orig_format))
return error_mark_node;
first_param = (*params)[0];
result = build_function_call_vec (loc, vNULL, new_function, params,
NULL);
if (result == error_mark_node)
return result;
if (orig_code != BUILT_IN_SYNC_BOOL_COMPARE_AND_SWAP_N
&& orig_code != BUILT_IN_SYNC_LOCK_RELEASE_N
&& orig_code != BUILT_IN_ATOMIC_STORE_N
&& orig_code != BUILT_IN_ATOMIC_COMPARE_EXCHANGE_N)
result = sync_resolve_return (first_param, result, orig_format);
if (fetch_op)
TREE_USED (result) = true;
if (new_return)
{
result = build1 (VIEW_CONVERT_EXPR, TREE_TYPE (new_return), result);
result = build2 (MODIFY_EXPR, TREE_TYPE (new_return), new_return,
result);
TREE_SIDE_EFFECTS (result) = 1;
protected_set_expr_location (result, loc);
result = convert (void_type_node, result);
}
return result;
}
default:
return NULL_TREE;
}
}
bool
vector_types_compatible_elements_p (tree t1, tree t2)
{
bool opaque = TYPE_VECTOR_OPAQUE (t1) || TYPE_VECTOR_OPAQUE (t2);
t1 = TREE_TYPE (t1);
t2 = TREE_TYPE (t2);
enum tree_code c1 = TREE_CODE (t1), c2 = TREE_CODE (t2);
gcc_assert ((c1 == INTEGER_TYPE || c1 == REAL_TYPE || c1 == FIXED_POINT_TYPE)
&& (c2 == INTEGER_TYPE || c2 == REAL_TYPE
|| c2 == FIXED_POINT_TYPE));
t1 = c_common_signed_type (t1);
t2 = c_common_signed_type (t2);
if (t1 == t2)
return true;
if (opaque && c1 == c2
&& (c1 == INTEGER_TYPE || c1 == REAL_TYPE)
&& TYPE_PRECISION (t1) == TYPE_PRECISION (t2))
return true;
return false;
}
bool
check_missing_format_attribute (tree ltype, tree rtype)
{
tree const ttr = TREE_TYPE (rtype), ttl = TREE_TYPE (ltype);
tree ra;
for (ra = TYPE_ATTRIBUTES (ttr); ra; ra = TREE_CHAIN (ra))
if (is_attribute_p ("format", TREE_PURPOSE (ra)))
break;
if (ra)
{
tree la;
for (la = TYPE_ATTRIBUTES (ttl); la; la = TREE_CHAIN (la))
if (is_attribute_p ("format", TREE_PURPOSE (la)))
break;
return !la;
}
else
return false;
}
void
set_underlying_type (tree x)
{
if (x == error_mark_node)
return;
if (DECL_IS_BUILTIN (x) && TREE_CODE (TREE_TYPE (x)) != ARRAY_TYPE)
{
if (TYPE_NAME (TREE_TYPE (x)) == 0)
TYPE_NAME (TREE_TYPE (x)) = x;
}
else if (TREE_TYPE (x) != error_mark_node
&& DECL_ORIGINAL_TYPE (x) == NULL_TREE)
{
tree tt = TREE_TYPE (x);
DECL_ORIGINAL_TYPE (x) = tt;
tt = build_variant_type_copy (tt);
TYPE_STUB_DECL (tt) = TYPE_STUB_DECL (DECL_ORIGINAL_TYPE (x));
TYPE_NAME (tt) = x;
if (lookup_attribute ("unused", DECL_ATTRIBUTES (x)))
TREE_USED (tt) = 1;
TREE_TYPE (x) = tt;
}
}
void
record_types_used_by_current_var_decl (tree decl)
{
gcc_assert (decl && DECL_P (decl) && TREE_STATIC (decl));
while (types_used_by_cur_var_decl && !types_used_by_cur_var_decl->is_empty ())
{
tree type = types_used_by_cur_var_decl->pop ();
types_used_by_var_decl_insert (type, decl);
}
}
typedef vec<tree, va_gc> *tree_gc_vec;
static GTY((deletable)) vec<tree_gc_vec, va_gc> *tree_vector_cache;
vec<tree, va_gc> *
make_tree_vector (void)
{
if (tree_vector_cache && !tree_vector_cache->is_empty ())
return tree_vector_cache->pop ();
else
{
vec<tree, va_gc> *v;
vec_alloc (v, 4);
return v;
}
}
void
release_tree_vector (vec<tree, va_gc> *vec)
{
if (vec != NULL)
{
vec->truncate (0);
vec_safe_push (tree_vector_cache, vec);
}
}
vec<tree, va_gc> *
make_tree_vector_single (tree t)
{
vec<tree, va_gc> *ret = make_tree_vector ();
ret->quick_push (t);
return ret;
}
vec<tree, va_gc> *
make_tree_vector_from_list (tree list)
{
vec<tree, va_gc> *ret = make_tree_vector ();
for (; list; list = TREE_CHAIN (list))
vec_safe_push (ret, TREE_VALUE (list));
return ret;
}
vec<tree, va_gc> *
make_tree_vector_from_ctor (tree ctor)
{
vec<tree,va_gc> *ret = make_tree_vector ();
vec_safe_reserve (ret, CONSTRUCTOR_NELTS (ctor));
for (unsigned i = 0; i < CONSTRUCTOR_NELTS (ctor); ++i)
ret->quick_push (CONSTRUCTOR_ELT (ctor, i)->value);
return ret;
}
vec<tree, va_gc> *
make_tree_vector_copy (const vec<tree, va_gc> *orig)
{
vec<tree, va_gc> *ret;
unsigned int ix;
tree t;
ret = make_tree_vector ();
vec_safe_reserve (ret, vec_safe_length (orig));
FOR_EACH_VEC_SAFE_ELT (orig, ix, t)
ret->quick_push (t);
return ret;
}
bool
keyword_begins_type_specifier (enum rid keyword)
{
switch (keyword)
{
case RID_AUTO_TYPE:
case RID_INT:
case RID_CHAR:
case RID_FLOAT:
case RID_DOUBLE:
case RID_VOID:
case RID_UNSIGNED:
case RID_LONG:
case RID_SHORT:
case RID_SIGNED:
CASE_RID_FLOATN_NX:
case RID_DFLOAT32:
case RID_DFLOAT64:
case RID_DFLOAT128:
case RID_FRACT:
case RID_ACCUM:
case RID_BOOL:
case RID_WCHAR:
case RID_CHAR16:
case RID_CHAR32:
case RID_SAT:
case RID_COMPLEX:
case RID_TYPEOF:
case RID_STRUCT:
case RID_CLASS:
case RID_UNION:
case RID_ENUM:
return true;
default:
if (keyword >= RID_FIRST_INT_N
&& keyword < RID_FIRST_INT_N + NUM_INT_N_ENTS
&& int_n_enabled_p[keyword-RID_FIRST_INT_N])
return true;
return false;
}
}
bool
keyword_is_type_qualifier (enum rid keyword)
{
switch (keyword)
{
case RID_CONST:
case RID_VOLATILE:
case RID_RESTRICT:
case RID_ATOMIC:
return true;
default:
return false;
}
}
bool
keyword_is_storage_class_specifier (enum rid keyword)
{
switch (keyword)
{
case RID_STATIC:
case RID_EXTERN:
case RID_REGISTER:
case RID_AUTO:
case RID_MUTABLE:
case RID_THREAD:
return true;
default:
return false;
}
}
static bool
keyword_is_function_specifier (enum rid keyword)
{
switch (keyword)
{
case RID_INLINE:
case RID_NORETURN:
case RID_VIRTUAL:
case RID_EXPLICIT:
return true;
default:
return false;
}
}
bool
keyword_is_decl_specifier (enum rid keyword)
{
if (keyword_is_storage_class_specifier (keyword)
|| keyword_is_type_qualifier (keyword)
|| keyword_is_function_specifier (keyword))
return true;
switch (keyword)
{
case RID_TYPEDEF:
case RID_FRIEND:
case RID_CONSTEXPR:
return true;
default:
return false;
}
}
void
c_common_init_ts (void)
{
MARK_TS_TYPED (C_MAYBE_CONST_EXPR);
MARK_TS_TYPED (EXCESS_PRECISION_EXPR);
}
tree
build_userdef_literal (tree suffix_id, tree value,
enum overflow_type overflow, tree num_string)
{
tree literal = make_node (USERDEF_LITERAL);
USERDEF_LITERAL_SUFFIX_ID (literal) = suffix_id;
USERDEF_LITERAL_VALUE (literal) = value;
USERDEF_LITERAL_OVERFLOW (literal) = overflow;
USERDEF_LITERAL_NUM_STRING (literal) = num_string;
return literal;
}
bool
convert_vector_to_array_for_subscript (location_t loc,
tree *vecp, tree index)
{
bool ret = false;
if (VECTOR_TYPE_P (TREE_TYPE (*vecp)))
{
tree type = TREE_TYPE (*vecp);
ret = !lvalue_p (*vecp);
if (TREE_CODE (index) == INTEGER_CST)
if (!tree_fits_uhwi_p (index)
|| maybe_ge (tree_to_uhwi (index), TYPE_VECTOR_SUBPARTS (type)))
warning_at (loc, OPT_Warray_bounds, "index value is out of bound");
c_common_mark_addressable_vec (*vecp);
*vecp = build1 (VIEW_CONVERT_EXPR,
build_array_type_nelts (TREE_TYPE (type),
TYPE_VECTOR_SUBPARTS (type)),
*vecp);
}
return ret;
}
enum stv_conv
scalar_to_vector (location_t loc, enum tree_code code, tree op0, tree op1,
bool complain)
{
tree type0 = TREE_TYPE (op0);
tree type1 = TREE_TYPE (op1);
bool integer_only_op = false;
enum stv_conv ret = stv_firstarg;
gcc_assert (VECTOR_TYPE_P (type0) || VECTOR_TYPE_P (type1));
switch (code)
{
case RSHIFT_EXPR:
case LSHIFT_EXPR:
if (TREE_CODE (type0) == INTEGER_TYPE
&& TREE_CODE (TREE_TYPE (type1)) == INTEGER_TYPE)
{
if (unsafe_conversion_p (loc, TREE_TYPE (type1), op0,
NULL_TREE, false))
{
if (complain)
error_at (loc, "conversion of scalar %qT to vector %qT "
"involves truncation", type0, type1);
return stv_error;
}
else
return stv_firstarg;
}
break;
case BIT_IOR_EXPR:
case BIT_XOR_EXPR:
case BIT_AND_EXPR:
integer_only_op = true;
case VEC_COND_EXPR:
case PLUS_EXPR:
case MINUS_EXPR:
case MULT_EXPR:
case TRUNC_DIV_EXPR:
case CEIL_DIV_EXPR:
case FLOOR_DIV_EXPR:
case ROUND_DIV_EXPR:
case EXACT_DIV_EXPR:
case TRUNC_MOD_EXPR:
case FLOOR_MOD_EXPR:
case RDIV_EXPR:
case EQ_EXPR:
case NE_EXPR:
case LE_EXPR:
case GE_EXPR:
case LT_EXPR:
case GT_EXPR:
if (VECTOR_TYPE_P (type0))
{
ret = stv_secondarg;
std::swap (type0, type1);
std::swap (op0, op1);
}
if (TREE_CODE (type0) == INTEGER_TYPE
&& TREE_CODE (TREE_TYPE (type1)) == INTEGER_TYPE)
{
if (unsafe_conversion_p (loc, TREE_TYPE (type1), op0,
NULL_TREE, false))
{
if (complain)
error_at (loc, "conversion of scalar %qT to vector %qT "
"involves truncation", type0, type1);
return stv_error;
}
return ret;
}
else if (!integer_only_op
&& (TREE_CODE (type0) == REAL_TYPE
|| TREE_CODE (type0) == INTEGER_TYPE)
&& SCALAR_FLOAT_TYPE_P (TREE_TYPE (type1)))
{
if (unsafe_conversion_p (loc, TREE_TYPE (type1), op0,
NULL_TREE, false))
{
if (complain)
error_at (loc, "conversion of scalar %qT to vector %qT "
"involves truncation", type0, type1);
return stv_error;
}
return ret;
}
default:
break;
}
return stv_nothing;
}
unsigned
max_align_t_align ()
{
unsigned int max_align = MAX (TYPE_ALIGN (long_long_integer_type_node),
TYPE_ALIGN (long_double_type_node));
if (float128_type_node != NULL_TREE)
max_align = MAX (max_align, TYPE_ALIGN (float128_type_node));
return max_align;
}
bool
cxx_fundamental_alignment_p (unsigned align)
{
return (align <= max_align_t_align ());
}
bool
pointer_to_zero_sized_aggr_p (tree t)
{
if (!POINTER_TYPE_P (t))
return false;
t = TREE_TYPE (t);
return (TYPE_SIZE (t) && integer_zerop (TYPE_SIZE (t)));
}
bool
reject_gcc_builtin (const_tree expr, location_t loc )
{
if (TREE_CODE (expr) == ADDR_EXPR)
expr = TREE_OPERAND (expr, 0);
STRIP_ANY_LOCATION_WRAPPER (expr);
if (TREE_TYPE (expr)
&& TREE_CODE (TREE_TYPE (expr)) == FUNCTION_TYPE
&& TREE_CODE (expr) == FUNCTION_DECL
&& DECL_BUILT_IN (expr)
&& DECL_IS_BUILTIN (expr)
&& !c_decl_implicit (expr)
&& !DECL_ASSEMBLER_NAME_SET_P (expr))
{
if (loc == UNKNOWN_LOCATION)
loc = EXPR_LOC_OR_LOC (expr, input_location);
error_at (loc, "built-in function %qE must be directly called", expr);
return true;
}
return false;
}
bool
valid_array_size_p (location_t loc, tree type, tree name)
{
if (type != error_mark_node
&& COMPLETE_TYPE_P (type)
&& TREE_CODE (TYPE_SIZE_UNIT (type)) == INTEGER_CST
&& !valid_constant_size_p (TYPE_SIZE_UNIT (type)))
{
if (name)
error_at (loc, "size of array %qE is too large", name);
else
error_at (loc, "size of unnamed array is too large");
return false;
}
return true;
}
time_t
cb_get_source_date_epoch (cpp_reader *pfile ATTRIBUTE_UNUSED)
{
char *source_date_epoch;
int64_t epoch;
char *endptr;
source_date_epoch = getenv ("SOURCE_DATE_EPOCH");
if (!source_date_epoch)
return (time_t) -1;
errno = 0;
#if defined(INT64_T_IS_LONG)
epoch = strtol (source_date_epoch, &endptr, 10);
#else
epoch = strtoll (source_date_epoch, &endptr, 10);
#endif
if (errno != 0 || endptr == source_date_epoch || *endptr != '\0'
|| epoch < 0 || epoch > MAX_SOURCE_DATE_EPOCH)
{
error_at (input_location, "environment variable SOURCE_DATE_EPOCH must "
"expand to a non-negative integer less than or equal to %wd",
MAX_SOURCE_DATE_EPOCH);
return (time_t) -1;
}
return (time_t) epoch;
}
const char *
cb_get_suggestion (cpp_reader *, const char *goal,
const char *const *candidates)
{
best_match<const char *, const char *> bm (goal);
while (*candidates)
bm.consider (*candidates++);
return bm.get_best_meaningful_candidate ();
}
enum flt_eval_method
excess_precision_mode_join (enum flt_eval_method x,
enum flt_eval_method y)
{
if (x == FLT_EVAL_METHOD_UNPREDICTABLE
|| y == FLT_EVAL_METHOD_UNPREDICTABLE)
return FLT_EVAL_METHOD_UNPREDICTABLE;
if (x == FLT_EVAL_METHOD_PROMOTE_TO_FLOAT16)
return y;
if (y == FLT_EVAL_METHOD_PROMOTE_TO_FLOAT16)
return x;
return MAX (x, y);
}
static enum flt_eval_method
c_ts18661_flt_eval_method (void)
{
enum flt_eval_method implicit
= targetm.c.excess_precision (EXCESS_PRECISION_TYPE_IMPLICIT);
enum excess_precision_type flag_type
= (flag_excess_precision_cmdline == EXCESS_PRECISION_STANDARD
? EXCESS_PRECISION_TYPE_STANDARD
: EXCESS_PRECISION_TYPE_FAST);
enum flt_eval_method requested
= targetm.c.excess_precision (flag_type);
return excess_precision_mode_join (implicit, requested);
}
static enum flt_eval_method
c_c11_flt_eval_method (void)
{
return excess_precision_mode_join (c_ts18661_flt_eval_method (),
FLT_EVAL_METHOD_PROMOTE_TO_FLOAT);
}
int
c_flt_eval_method (bool maybe_c11_only_p)
{
if (maybe_c11_only_p
&& flag_permitted_flt_eval_methods
== PERMITTED_FLT_EVAL_METHODS_C11)
return c_c11_flt_eval_method ();
else
return c_ts18661_flt_eval_method ();
}
enum missing_token_insertion_kind
{
MTIK_IMPOSSIBLE,
MTIK_INSERT_BEFORE_NEXT,
MTIK_INSERT_AFTER_PREV
};
static enum missing_token_insertion_kind
get_missing_token_insertion_kind (enum cpp_ttype type)
{
switch (type)
{
case CPP_OPEN_SQUARE:
case CPP_OPEN_PAREN:
return MTIK_INSERT_BEFORE_NEXT;
case CPP_CLOSE_PAREN:
case CPP_CLOSE_SQUARE:
case CPP_SEMICOLON:
case CPP_COMMA:
case CPP_COLON:
return MTIK_INSERT_AFTER_PREV;
default:
return MTIK_IMPOSSIBLE;
}
}
void
maybe_suggest_missing_token_insertion (rich_location *richloc,
enum cpp_ttype token_type,
location_t prev_token_loc)
{
gcc_assert (richloc);
enum missing_token_insertion_kind mtik
= get_missing_token_insertion_kind (token_type);
switch (mtik)
{
default:
gcc_unreachable ();
break;
case MTIK_IMPOSSIBLE:
return;
case MTIK_INSERT_BEFORE_NEXT:
richloc->add_fixit_insert_before (cpp_type2name (token_type, 0));
break;
case MTIK_INSERT_AFTER_PREV:
richloc->add_fixit_insert_after (prev_token_loc,
cpp_type2name (token_type, 0));
break;
}
if (!richloc->seen_impossible_fixit_p ())
{
fixit_hint *hint = richloc->get_last_fixit_hint ();
location_t hint_loc = hint->get_start_loc ();
location_t old_loc = richloc->get_loc ();
richloc->set_range (line_table, 0, hint_loc, true);
richloc->add_range (old_loc, false);
}
}
#if CHECKING_P
namespace selftest {
static void
test_fold_for_warn ()
{
ASSERT_EQ (error_mark_node, fold_for_warn (error_mark_node));
}
static void
c_common_c_tests ()
{
test_fold_for_warn ();
}
void
c_family_tests (void)
{
c_common_c_tests ();
c_format_c_tests ();
c_pretty_print_c_tests ();
c_spellcheck_cc_tests ();
}
} 
#endif 
static location_t
try_to_locate_new_include_insertion_point (const char *file, location_t loc)
{
const line_map_ordinary *last_include_ord_map = NULL;
const line_map_ordinary *last_ord_map_after_include = NULL;
const line_map_ordinary *first_ord_map_in_file = NULL;
const line_map_ordinary *ord_map_for_loc = NULL;
loc = linemap_resolve_location (line_table, loc, LRK_MACRO_EXPANSION_POINT,
&ord_map_for_loc);
gcc_assert (ord_map_for_loc);
for (unsigned int i = 0; i < LINEMAPS_ORDINARY_USED (line_table); i++)
{
const line_map_ordinary *ord_map
= LINEMAPS_ORDINARY_MAP_AT (line_table, i);
const line_map_ordinary *from = INCLUDED_FROM (line_table, ord_map);
if (from)
if (from->to_file == file)
{
last_include_ord_map = from;
last_ord_map_after_include = NULL;
}
if (ord_map->to_file == file)
{
if (!first_ord_map_in_file)
first_ord_map_in_file = ord_map;
if (last_include_ord_map && !last_ord_map_after_include)
last_ord_map_after_include = ord_map;
}
if (ord_map == ord_map_for_loc)
break;
}
const line_map_ordinary *ord_map_for_insertion;
if (last_ord_map_after_include)
ord_map_for_insertion = last_ord_map_after_include;
else
ord_map_for_insertion = first_ord_map_in_file;
if (!ord_map_for_insertion)
return UNKNOWN_LOCATION;
location_t col_0 = ord_map_for_insertion->start_location;
return linemap_position_for_loc_and_offset (line_table, col_0, 1);
}
typedef hash_set <const char *, nofree_string_hash> per_file_includes_t;
typedef hash_map <const char *, per_file_includes_t *> added_includes_t;
static added_includes_t *added_includes;
void
maybe_add_include_fixit (rich_location *richloc, const char *header)
{
location_t loc = richloc->get_loc ();
const char *file = LOCATION_FILE (loc);
if (!file)
return;
if (!added_includes)
added_includes = new added_includes_t ();
per_file_includes_t *&set = added_includes->get_or_insert (file);
if (set)
if (set->contains (header))
return;
if (!set)
set = new per_file_includes_t ();
set->add (header);
location_t include_insert_loc
= try_to_locate_new_include_insertion_point (file, loc);
if (include_insert_loc == UNKNOWN_LOCATION)
return;
char *text = xasprintf ("#include %s\n", header);
richloc->add_fixit_insert_before (include_insert_loc, text);
free (text);
}
#include "gt-c-family-c-common.h"

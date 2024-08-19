#ifndef GCC_CP_TREE_H
#define GCC_CP_TREE_H
#include "tm.h"
#include "hard-reg-set.h"
#include "function.h"
#undef GCC_DIAG_STYLE
#define GCC_DIAG_STYLE __gcc_cxxdiag__
#if defined(GCC_DIAGNOSTIC_CORE_H) || defined (GCC_C_COMMON_H)
#error \
In order for the format checking to accept the C++ front end diagnostic \
framework extensions, you must include this file before diagnostic-core.h and \
c-common.h, not after.
#endif
#include "c-family/c-common.h"
#include "diagnostic.h"
class cp_expr
{
public:
cp_expr () :
m_value (NULL), m_loc (UNKNOWN_LOCATION) {}
cp_expr (tree value) :
m_value (value), m_loc (EXPR_LOCATION (m_value)) {}
cp_expr (tree value, location_t loc):
m_value (value), m_loc (loc) {}
cp_expr (const cp_expr &other) :
m_value (other.m_value), m_loc (other.m_loc) {}
operator tree () const { return m_value; }
tree & operator* () { return m_value; }
tree operator* () const { return m_value; }
tree & operator-> () { return m_value; }
tree operator-> () const { return m_value; }
tree get_value () const { return m_value; }
location_t get_location () const { return m_loc; }
location_t get_start () const
{
source_range src_range = get_range_from_loc (line_table, m_loc);
return src_range.m_start;
}
location_t get_finish () const
{
source_range src_range = get_range_from_loc (line_table, m_loc);
return src_range.m_finish;
}
void set_location (location_t loc)
{
protected_set_expr_location (m_value, loc);
m_loc = loc;
}
void set_range (location_t start, location_t finish)
{
set_location (make_location (m_loc, start, finish));
}
cp_expr& maybe_add_location_wrapper ()
{
m_value = maybe_wrap_with_location (m_value, m_loc);
return *this;
}
private:
tree m_value;
location_t m_loc;
};
inline bool
operator == (const cp_expr &lhs, tree rhs)
{
return lhs.get_value () == rhs;
}

enum cp_tree_index
{
CPTI_WCHAR_DECL,
CPTI_VTABLE_ENTRY_TYPE,
CPTI_DELTA_TYPE,
CPTI_VTABLE_INDEX_TYPE,
CPTI_CLEANUP_TYPE,
CPTI_VTT_PARM_TYPE,
CPTI_CLASS_TYPE,
CPTI_UNKNOWN_TYPE,
CPTI_INIT_LIST_TYPE,
CPTI_VTBL_TYPE,
CPTI_VTBL_PTR_TYPE,
CPTI_STD,
CPTI_ABI,
CPTI_GLOBAL,
CPTI_GLOBAL_TYPE,
CPTI_CONST_TYPE_INFO_TYPE,
CPTI_TYPE_INFO_PTR_TYPE,
CPTI_ABORT_FNDECL,
CPTI_AGGR_TAG,
CPTI_CONV_OP_MARKER,
CPTI_CTOR_IDENTIFIER,
CPTI_COMPLETE_CTOR_IDENTIFIER,
CPTI_BASE_CTOR_IDENTIFIER,
CPTI_DTOR_IDENTIFIER,
CPTI_COMPLETE_DTOR_IDENTIFIER,
CPTI_BASE_DTOR_IDENTIFIER,
CPTI_DELETING_DTOR_IDENTIFIER,
CPTI_CONV_OP_IDENTIFIER,
CPTI_DELTA_IDENTIFIER,
CPTI_IN_CHARGE_IDENTIFIER,
CPTI_VTT_PARM_IDENTIFIER,
CPTI_THIS_IDENTIFIER,
CPTI_PFN_IDENTIFIER,
CPTI_VPTR_IDENTIFIER,
CPTI_GLOBAL_IDENTIFIER,
CPTI_STD_IDENTIFIER,
CPTI_ANON_IDENTIFIER,
CPTI_AUTO_IDENTIFIER,
CPTI_DECLTYPE_AUTO_IDENTIFIER,
CPTI_INIT_LIST_IDENTIFIER,
CPTI_LANG_NAME_C,
CPTI_LANG_NAME_CPLUSPLUS,
CPTI_EMPTY_EXCEPT_SPEC,
CPTI_NOEXCEPT_TRUE_SPEC,
CPTI_NOEXCEPT_FALSE_SPEC,
CPTI_NOEXCEPT_DEFERRED_SPEC,
CPTI_TERMINATE_FN,
CPTI_CALL_UNEXPECTED_FN,
CPTI_GET_EXCEPTION_PTR_FN,
CPTI_BEGIN_CATCH_FN,
CPTI_END_CATCH_FN,
CPTI_ALLOCATE_EXCEPTION_FN,
CPTI_FREE_EXCEPTION_FN,
CPTI_THROW_FN,
CPTI_RETHROW_FN,
CPTI_ATEXIT_FN_PTR_TYPE,
CPTI_ATEXIT,
CPTI_DSO_HANDLE,
CPTI_DCAST,
CPTI_NULLPTR,
CPTI_NULLPTR_TYPE,
CPTI_ALIGN_TYPE,
CPTI_ANY_TARG,
CPTI_MAX
};
extern GTY(()) tree cp_global_trees[CPTI_MAX];
#define wchar_decl_node			cp_global_trees[CPTI_WCHAR_DECL]
#define vtable_entry_type		cp_global_trees[CPTI_VTABLE_ENTRY_TYPE]
#define delta_type_node			cp_global_trees[CPTI_DELTA_TYPE]
#define vtable_index_type		cp_global_trees[CPTI_VTABLE_INDEX_TYPE]
#define class_type_node			cp_global_trees[CPTI_CLASS_TYPE]
#define unknown_type_node		cp_global_trees[CPTI_UNKNOWN_TYPE]
#define init_list_type_node		cp_global_trees[CPTI_INIT_LIST_TYPE]
#define vtbl_type_node			cp_global_trees[CPTI_VTBL_TYPE]
#define vtbl_ptr_type_node		cp_global_trees[CPTI_VTBL_PTR_TYPE]
#define std_node			cp_global_trees[CPTI_STD]
#define abi_node			cp_global_trees[CPTI_ABI]
#define global_namespace		cp_global_trees[CPTI_GLOBAL]
#define global_type_node		cp_global_trees[CPTI_GLOBAL_TYPE]
#define const_type_info_type_node	cp_global_trees[CPTI_CONST_TYPE_INFO_TYPE]
#define type_info_ptr_type		cp_global_trees[CPTI_TYPE_INFO_PTR_TYPE]
#define conv_op_marker			cp_global_trees[CPTI_CONV_OP_MARKER]
#define abort_fndecl			cp_global_trees[CPTI_ABORT_FNDECL]
#define current_aggr			cp_global_trees[CPTI_AGGR_TAG]
#define nullptr_node			cp_global_trees[CPTI_NULLPTR]
#define nullptr_type_node		cp_global_trees[CPTI_NULLPTR_TYPE]
#define align_type_node			cp_global_trees[CPTI_ALIGN_TYPE]
#define ctor_identifier			cp_global_trees[CPTI_CTOR_IDENTIFIER]
#define complete_ctor_identifier	cp_global_trees[CPTI_COMPLETE_CTOR_IDENTIFIER]
#define base_ctor_identifier		cp_global_trees[CPTI_BASE_CTOR_IDENTIFIER]
#define dtor_identifier			cp_global_trees[CPTI_DTOR_IDENTIFIER]
#define complete_dtor_identifier	cp_global_trees[CPTI_COMPLETE_DTOR_IDENTIFIER]
#define base_dtor_identifier		cp_global_trees[CPTI_BASE_DTOR_IDENTIFIER]
#define deleting_dtor_identifier	cp_global_trees[CPTI_DELETING_DTOR_IDENTIFIER]
#define ovl_op_identifier(ISASS, CODE)  (OVL_OP_INFO(ISASS, CODE)->identifier)
#define assign_op_identifier (ovl_op_info[true][OVL_OP_NOP_EXPR].identifier)
#define call_op_identifier (ovl_op_info[false][OVL_OP_CALL_EXPR].identifier)
#define conv_op_identifier		cp_global_trees[CPTI_CONV_OP_IDENTIFIER]
#define delta_identifier		cp_global_trees[CPTI_DELTA_IDENTIFIER]
#define in_charge_identifier		cp_global_trees[CPTI_IN_CHARGE_IDENTIFIER]
#define vtt_parm_identifier		cp_global_trees[CPTI_VTT_PARM_IDENTIFIER]
#define this_identifier			cp_global_trees[CPTI_THIS_IDENTIFIER]
#define pfn_identifier			cp_global_trees[CPTI_PFN_IDENTIFIER]
#define vptr_identifier			cp_global_trees[CPTI_VPTR_IDENTIFIER]
#define global_identifier		cp_global_trees[CPTI_GLOBAL_IDENTIFIER]
#define std_identifier			cp_global_trees[CPTI_STD_IDENTIFIER]
#define anon_identifier			cp_global_trees[CPTI_ANON_IDENTIFIER]
#define auto_identifier			cp_global_trees[CPTI_AUTO_IDENTIFIER]
#define decltype_auto_identifier	cp_global_trees[CPTI_DECLTYPE_AUTO_IDENTIFIER]
#define init_list_identifier		cp_global_trees[CPTI_INIT_LIST_IDENTIFIER]
#define lang_name_c			cp_global_trees[CPTI_LANG_NAME_C]
#define lang_name_cplusplus		cp_global_trees[CPTI_LANG_NAME_CPLUSPLUS]
#define empty_except_spec		cp_global_trees[CPTI_EMPTY_EXCEPT_SPEC]
#define noexcept_true_spec		cp_global_trees[CPTI_NOEXCEPT_TRUE_SPEC]
#define noexcept_false_spec		cp_global_trees[CPTI_NOEXCEPT_FALSE_SPEC]
#define noexcept_deferred_spec		cp_global_trees[CPTI_NOEXCEPT_DEFERRED_SPEC]
#define terminate_fn			cp_global_trees[CPTI_TERMINATE_FN]
#define call_unexpected_fn		cp_global_trees[CPTI_CALL_UNEXPECTED_FN]
#define get_exception_ptr_fn		cp_global_trees[CPTI_GET_EXCEPTION_PTR_FN]
#define begin_catch_fn			cp_global_trees[CPTI_BEGIN_CATCH_FN]
#define end_catch_fn			cp_global_trees[CPTI_END_CATCH_FN]
#define allocate_exception_fn		cp_global_trees[CPTI_ALLOCATE_EXCEPTION_FN]
#define free_exception_fn		cp_global_trees[CPTI_FREE_EXCEPTION_FN]
#define throw_fn			cp_global_trees[CPTI_THROW_FN]
#define rethrow_fn			cp_global_trees[CPTI_RETHROW_FN]
#define atexit_fn_ptr_type_node         cp_global_trees[CPTI_ATEXIT_FN_PTR_TYPE]
#define atexit_node			cp_global_trees[CPTI_ATEXIT]
#define dso_handle_node			cp_global_trees[CPTI_DSO_HANDLE]
#define dynamic_cast_node		cp_global_trees[CPTI_DCAST]
#define cleanup_type			cp_global_trees[CPTI_CLEANUP_TYPE]
#define vtt_parm_type			cp_global_trees[CPTI_VTT_PARM_TYPE]
#define any_targ_node			cp_global_trees[CPTI_ANY_TARG]
#define access_default_node		null_node

#include "name-lookup.h"
#define VAR_OR_FUNCTION_DECL_CHECK(NODE) \
TREE_CHECK2(NODE,VAR_DECL,FUNCTION_DECL)
#define TYPE_FUNCTION_OR_TEMPLATE_DECL_CHECK(NODE) \
TREE_CHECK3(NODE,TYPE_DECL,TEMPLATE_DECL,FUNCTION_DECL)
#define TYPE_FUNCTION_OR_TEMPLATE_DECL_P(NODE) \
(TREE_CODE (NODE) == TYPE_DECL || TREE_CODE (NODE) == TEMPLATE_DECL \
|| TREE_CODE (NODE) == FUNCTION_DECL)
#define VAR_FUNCTION_OR_PARM_DECL_CHECK(NODE) \
TREE_CHECK3(NODE,VAR_DECL,FUNCTION_DECL,PARM_DECL)
#define VAR_TEMPL_TYPE_OR_FUNCTION_DECL_CHECK(NODE) \
TREE_CHECK4(NODE,VAR_DECL,FUNCTION_DECL,TYPE_DECL,TEMPLATE_DECL)
#define VAR_TEMPL_TYPE_FIELD_OR_FUNCTION_DECL_CHECK(NODE) \
TREE_CHECK5(NODE,VAR_DECL,FIELD_DECL,FUNCTION_DECL,TYPE_DECL,TEMPLATE_DECL)
#define BOUND_TEMPLATE_TEMPLATE_PARM_TYPE_CHECK(NODE) \
TREE_CHECK(NODE,BOUND_TEMPLATE_TEMPLATE_PARM)
#if defined ENABLE_TREE_CHECKING && (GCC_VERSION >= 2007)
#define THUNK_FUNCTION_CHECK(NODE) __extension__			\
({  __typeof (NODE) const __t = (NODE);					\
if (TREE_CODE (__t) != FUNCTION_DECL || !__t->decl_common.lang_specific \
|| !__t->decl_common.lang_specific->u.fn.thunk_p)		\
tree_check_failed (__t, __FILE__, __LINE__, __FUNCTION__, 0);	\
__t; })
#else
#define THUNK_FUNCTION_CHECK(NODE) (NODE)
#endif

struct GTY(()) lang_identifier {
struct c_common_identifier c_common;
cxx_binding *bindings;
};
inline lang_identifier*
identifier_p (tree t)
{
if (TREE_CODE (t) == IDENTIFIER_NODE)
return (lang_identifier*) t;
return NULL;
}
#define LANG_IDENTIFIER_CAST(NODE) \
((struct lang_identifier*)IDENTIFIER_NODE_CHECK (NODE))
struct GTY(()) template_parm_index {
struct tree_common common;
int index;
int level;
int orig_level;
tree decl;
};
struct GTY(()) ptrmem_cst {
struct tree_common common;
tree member;
};
typedef struct ptrmem_cst * ptrmem_cst_t;
#define CLEANUP_P(NODE)		TREE_LANG_FLAG_0 (TRY_BLOCK_CHECK (NODE))
#define BIND_EXPR_TRY_BLOCK(NODE) \
TREE_LANG_FLAG_0 (BIND_EXPR_CHECK (NODE))
#define BIND_EXPR_BODY_BLOCK(NODE) \
TREE_LANG_FLAG_3 (BIND_EXPR_CHECK (NODE))
#define FUNCTION_NEEDS_BODY_BLOCK(NODE) \
(DECL_CONSTRUCTOR_P (NODE) || DECL_DESTRUCTOR_P (NODE) \
|| LAMBDA_FUNCTION_P (NODE))
#define STATEMENT_LIST_NO_SCOPE(NODE) \
TREE_LANG_FLAG_0 (STATEMENT_LIST_CHECK (NODE))
#define STATEMENT_LIST_TRY_BLOCK(NODE) \
TREE_LANG_FLAG_2 (STATEMENT_LIST_CHECK (NODE))
#define BLOCK_OUTER_CURLY_BRACE_P(NODE)	TREE_LANG_FLAG_0 (BLOCK_CHECK (NODE))
#define STMT_IS_FULL_EXPR_P(NODE) TREE_LANG_FLAG_1 ((NODE))
#define EXPR_STMT_STMT_EXPR_RESULT(NODE) \
TREE_LANG_FLAG_0 (EXPR_STMT_CHECK (NODE))
#define STMT_EXPR_NO_SCOPE(NODE) \
TREE_LANG_FLAG_0 (STMT_EXPR_CHECK (NODE))
#define COND_EXPR_IS_VEC_DELETE(NODE) \
TREE_LANG_FLAG_0 (COND_EXPR_CHECK (NODE))
#define REINTERPRET_CAST_P(NODE)		\
TREE_LANG_FLAG_0 (NOP_EXPR_CHECK (NODE))
#define same_type_p(TYPE1, TYPE2) \
comptypes ((TYPE1), (TYPE2), COMPARE_STRICT)
#define DECL_MAIN_P(NODE)				\
(DECL_EXTERN_C_FUNCTION_P (NODE)			\
&& DECL_NAME (NODE) != NULL_TREE			\
&& MAIN_NAME_P (DECL_NAME (NODE))			\
&& flag_hosted)
#define LOOKUP_SEEN_P(NODE) TREE_VISITED(NODE)
#define LOOKUP_FOUND_P(NODE) \
TREE_LANG_FLAG_4 (TREE_CHECK3(NODE,RECORD_TYPE,UNION_TYPE,NAMESPACE_DECL))
#define OVL_FUNCTION(NODE) \
(((struct tree_overload*)OVERLOAD_CHECK (NODE))->function)
#define OVL_CHAIN(NODE) \
(((struct tree_overload*)OVERLOAD_CHECK (NODE))->common.chain)
#define OVL_USING_P(NODE)	TREE_LANG_FLAG_1 (OVERLOAD_CHECK (NODE))
#define OVL_HIDDEN_P(NODE)	TREE_LANG_FLAG_2 (OVERLOAD_CHECK (NODE))
#define OVL_NESTED_P(NODE)	TREE_LANG_FLAG_3 (OVERLOAD_CHECK (NODE))
#define OVL_LOOKUP_P(NODE)	TREE_LANG_FLAG_4 (OVERLOAD_CHECK (NODE))
#define OVL_USED_P(NODE)	TREE_USED (OVERLOAD_CHECK (NODE))
#define OVL_FIRST(NODE)	ovl_first (NODE)
#define OVL_NAME(NODE) DECL_NAME (OVL_FIRST (NODE))
#define OVL_P(NODE) \
(TREE_CODE (NODE) == FUNCTION_DECL || TREE_CODE (NODE) == OVERLOAD)
#define OVL_SINGLE_P(NODE) \
(TREE_CODE (NODE) != OVERLOAD || !OVL_CHAIN (NODE))
struct GTY(()) tree_overload {
struct tree_common common;
tree function;
};
class ovl_iterator 
{
tree ovl;
const bool allow_inner; 
public:
explicit ovl_iterator (tree o, bool allow = false)
: ovl (o), allow_inner (allow)
{
}
private:
ovl_iterator &operator= (const ovl_iterator &);
ovl_iterator (const ovl_iterator &);
public:
operator bool () const
{
return ovl;
}
ovl_iterator &operator++ ()
{
ovl = TREE_CODE (ovl) != OVERLOAD ? NULL_TREE : OVL_CHAIN (ovl);
return *this;
}
tree operator* () const
{
tree fn = TREE_CODE (ovl) != OVERLOAD ? ovl : OVL_FUNCTION (ovl);
gcc_checking_assert (allow_inner || TREE_CODE (fn) != OVERLOAD);
return fn;
}
public:
bool using_p () const
{
return TREE_CODE (ovl) == OVERLOAD && OVL_USING_P (ovl);
}
bool hidden_p () const
{
return TREE_CODE (ovl) == OVERLOAD && OVL_HIDDEN_P (ovl);
}
public:
tree remove_node (tree head)
{
return remove_node (head, ovl);
}
tree reveal_node (tree head)
{
return reveal_node (head, ovl);
}
protected:
tree maybe_push ()
{
tree r = NULL_TREE;
if (ovl && TREE_CODE (ovl) == OVERLOAD && OVL_NESTED_P (ovl))
{
r = OVL_CHAIN (ovl);
ovl = OVL_FUNCTION (ovl);
}
return r;
}
void pop (tree outer)
{
gcc_checking_assert (!ovl);
ovl = outer;
}
private:
static tree remove_node (tree head, tree node);
static tree reveal_node (tree ovl, tree node);
};
class lkp_iterator : public ovl_iterator
{
typedef ovl_iterator parent;
tree outer;
public:
explicit lkp_iterator (tree o)
: parent (o, true), outer (maybe_push ())
{
}
public:
lkp_iterator &operator++ ()
{
bool repush = !outer;
if (!parent::operator++ () && !repush)
{
pop (outer);
repush = true;
}
if (repush)
outer = maybe_push ();
return *this;
}
};
struct named_decl_hash : ggc_remove <tree>
{
typedef tree value_type; 
typedef tree compare_type; 
inline static hashval_t hash (const value_type decl);
inline static bool equal (const value_type existing, compare_type candidate);
static inline void mark_empty (value_type &p) {p = NULL_TREE;}
static inline bool is_empty (value_type p) {return !p;}
static bool is_deleted (value_type) { return false; }
static void mark_deleted (value_type) { gcc_unreachable (); }
};
struct GTY(()) tree_template_decl {
struct tree_decl_common common;
tree arguments;
tree result;
};
#define BASELINK_P(NODE) \
(TREE_CODE (NODE) == BASELINK)
#define BASELINK_BINFO(NODE) \
(((struct tree_baselink*) BASELINK_CHECK (NODE))->binfo)
#define BASELINK_FUNCTIONS(NODE) \
(((struct tree_baselink*) BASELINK_CHECK (NODE))->functions)
#define MAYBE_BASELINK_FUNCTIONS(T) \
(BASELINK_P (T) ? BASELINK_FUNCTIONS (T) : T)
#define BASELINK_ACCESS_BINFO(NODE) \
(((struct tree_baselink*) BASELINK_CHECK (NODE))->access_binfo)
#define BASELINK_OPTYPE(NODE) \
(TREE_CHAIN (BASELINK_CHECK (NODE)))
#define BASELINK_QUALIFIED_P(NODE) \
TREE_LANG_FLAG_0 (BASELINK_CHECK (NODE))
struct GTY(()) tree_baselink {
struct tree_common common;
tree binfo;
tree functions;
tree access_binfo;
};
enum cp_id_kind
{
CP_ID_KIND_NONE,
CP_ID_KIND_UNQUALIFIED,
CP_ID_KIND_UNQUALIFIED_DEPENDENT,
CP_ID_KIND_TEMPLATE_ID,
CP_ID_KIND_QUALIFIED
};
enum cpp0x_warn_str
{
CPP0X_INITIALIZER_LISTS,
CPP0X_EXPLICIT_CONVERSION,
CPP0X_VARIADIC_TEMPLATES,
CPP0X_LAMBDA_EXPR,
CPP0X_AUTO,
CPP0X_SCOPED_ENUMS,
CPP0X_DEFAULTED_DELETED,
CPP0X_INLINE_NAMESPACES,
CPP0X_OVERRIDE_CONTROLS,
CPP0X_NSDMI,
CPP0X_USER_DEFINED_LITERALS,
CPP0X_DELEGATING_CTORS,
CPP0X_INHERITING_CTORS,
CPP0X_ATTRIBUTES,
CPP0X_REF_QUALIFIER
};
enum composite_pointer_operation
{
CPO_COMPARISON,
CPO_CONVERSION,
CPO_CONDITIONAL_EXPR
};
enum expr_list_kind {
ELK_INIT,		
ELK_MEM_INIT,		
ELK_FUNC_CAST		
};
enum impl_conv_rhs {
ICR_DEFAULT_ARGUMENT, 
ICR_CONVERTING,       
ICR_INIT,             
ICR_ARGPASS,          
ICR_RETURN,           
ICR_ASSIGN            
};
enum impl_conv_void {
ICV_CAST,            
ICV_SECOND_OF_COND,  
ICV_THIRD_OF_COND,   
ICV_RIGHT_OF_COMMA,  
ICV_LEFT_OF_COMMA,   
ICV_STATEMENT,       
ICV_THIRD_IN_FOR     
};
enum GTY(()) abstract_class_use {
ACU_UNKNOWN,			
ACU_CAST,			
ACU_NEW,			
ACU_THROW,			
ACU_CATCH,			
ACU_ARRAY,			
ACU_RETURN,			
ACU_PARM			
};
#define IDENTIFIER_BINDING(NODE) \
(LANG_IDENTIFIER_CAST (NODE)->bindings)
#define IDENTIFIER_TYPE_VALUE(NODE) identifier_type_value (NODE)
#define REAL_IDENTIFIER_TYPE_VALUE(NODE) TREE_TYPE (NODE)
#define SET_IDENTIFIER_TYPE_VALUE(NODE,TYPE) (TREE_TYPE (NODE) = (TYPE))
#define IDENTIFIER_HAS_TYPE_VALUE(NODE) (IDENTIFIER_TYPE_VALUE (NODE) ? 1 : 0)
enum cp_identifier_kind {
cik_normal = 0,	
cik_keyword = 1,	
cik_ctor = 2,		
cik_dtor = 3,		
cik_simple_op = 4,	
cik_assign_op = 5,	
cik_conv_op = 6,	
cik_reserved_for_udlit = 7,	
cik_max
};
#define IDENTIFIER_KIND_BIT_0(NODE) \
TREE_LANG_FLAG_0 (IDENTIFIER_NODE_CHECK (NODE))
#define IDENTIFIER_KIND_BIT_1(NODE) \
TREE_LANG_FLAG_1 (IDENTIFIER_NODE_CHECK (NODE))
#define IDENTIFIER_KIND_BIT_2(NODE) \
TREE_LANG_FLAG_2 (IDENTIFIER_NODE_CHECK (NODE))
#define IDENTIFIER_MARKED(NODE) \
TREE_LANG_FLAG_4 (IDENTIFIER_NODE_CHECK (NODE))
#define IDENTIFIER_VIRTUAL_P(NODE) \
TREE_LANG_FLAG_5 (IDENTIFIER_NODE_CHECK (NODE))
#define IDENTIFIER_REPO_CHOSEN(NAME) \
(TREE_LANG_FLAG_6 (IDENTIFIER_NODE_CHECK (NAME)))
#define IDENTIFIER_KEYWORD_P(NODE)		\
((!IDENTIFIER_KIND_BIT_2 (NODE))		\
& (!IDENTIFIER_KIND_BIT_1 (NODE))		\
& IDENTIFIER_KIND_BIT_0 (NODE))
#define IDENTIFIER_CDTOR_P(NODE)		\
((!IDENTIFIER_KIND_BIT_2 (NODE))		\
& IDENTIFIER_KIND_BIT_1 (NODE))
#define IDENTIFIER_CTOR_P(NODE)			\
(IDENTIFIER_CDTOR_P(NODE)			\
& (!IDENTIFIER_KIND_BIT_0 (NODE)))
#define IDENTIFIER_DTOR_P(NODE)			\
(IDENTIFIER_CDTOR_P(NODE)			\
& IDENTIFIER_KIND_BIT_0 (NODE))
#define IDENTIFIER_ANY_OP_P(NODE)		\
(IDENTIFIER_KIND_BIT_2 (NODE))
#define IDENTIFIER_OVL_OP_P(NODE)		\
(IDENTIFIER_ANY_OP_P (NODE)			\
& (!IDENTIFIER_KIND_BIT_1 (NODE)))
#define IDENTIFIER_ASSIGN_OP_P(NODE)		\
(IDENTIFIER_OVL_OP_P (NODE)			\
& IDENTIFIER_KIND_BIT_0 (NODE))
#define IDENTIFIER_CONV_OP_P(NODE)		\
(IDENTIFIER_ANY_OP_P (NODE)			\
& IDENTIFIER_KIND_BIT_1 (NODE)		\
& (!IDENTIFIER_KIND_BIT_0 (NODE)))
#define IDENTIFIER_NEWDEL_OP_P(NODE)		\
(IDENTIFIER_OVL_OP_P (NODE)			\
&& IDENTIFIER_OVL_OP_FLAGS (NODE) & OVL_OP_FLAG_ALLOC)
#define IDENTIFIER_NEW_OP_P(NODE)					\
(IDENTIFIER_OVL_OP_P (NODE)						\
&& (IDENTIFIER_OVL_OP_FLAGS (NODE)					\
& (OVL_OP_FLAG_ALLOC | OVL_OP_FLAG_DELETE)) == OVL_OP_FLAG_ALLOC)
#define IDENTIFIER_CP_INDEX(NODE)		\
(IDENTIFIER_NODE_CHECK(NODE)->base.u.bits.address_space)
#define C_TYPE_FIELDS_READONLY(TYPE) \
(LANG_TYPE_CLASS_CHECK (TYPE)->fields_readonly)
#define DEFARG_TOKENS(NODE) \
(((struct tree_default_arg *)DEFAULT_ARG_CHECK (NODE))->tokens)
#define DEFARG_INSTANTIATIONS(NODE) \
(((struct tree_default_arg *)DEFAULT_ARG_CHECK (NODE))->instantiations)
struct GTY (()) tree_default_arg {
struct tree_common common;
struct cp_token_cache *tokens;
vec<tree, va_gc> *instantiations;
};
#define DEFERRED_NOEXCEPT_PATTERN(NODE) \
(((struct tree_deferred_noexcept *)DEFERRED_NOEXCEPT_CHECK (NODE))->pattern)
#define DEFERRED_NOEXCEPT_ARGS(NODE) \
(((struct tree_deferred_noexcept *)DEFERRED_NOEXCEPT_CHECK (NODE))->args)
#define DEFERRED_NOEXCEPT_SPEC_P(NODE)				\
((NODE) && (TREE_PURPOSE (NODE))				\
&& (TREE_CODE (TREE_PURPOSE (NODE)) == DEFERRED_NOEXCEPT))
#define UNEVALUATED_NOEXCEPT_SPEC_P(NODE)				\
(DEFERRED_NOEXCEPT_SPEC_P (NODE)					\
&& DEFERRED_NOEXCEPT_PATTERN (TREE_PURPOSE (NODE)) == NULL_TREE)
struct GTY (()) tree_deferred_noexcept {
struct tree_base base;
tree pattern;
tree args;
};
#define STATIC_ASSERT_CONDITION(NODE) \
(((struct tree_static_assert *)STATIC_ASSERT_CHECK (NODE))->condition)
#define STATIC_ASSERT_MESSAGE(NODE) \
(((struct tree_static_assert *)STATIC_ASSERT_CHECK (NODE))->message)
#define STATIC_ASSERT_SOURCE_LOCATION(NODE) \
(((struct tree_static_assert *)STATIC_ASSERT_CHECK (NODE))->location)
struct GTY (()) tree_static_assert {
struct tree_common common;
tree condition;
tree message;
location_t location;
};
struct GTY (()) tree_argument_pack_select {
struct tree_common common;
tree argument_pack;
int index;
};
enum cp_trait_kind
{
CPTK_BASES,
CPTK_DIRECT_BASES,
CPTK_HAS_NOTHROW_ASSIGN,
CPTK_HAS_NOTHROW_CONSTRUCTOR,
CPTK_HAS_NOTHROW_COPY,
CPTK_HAS_TRIVIAL_ASSIGN,
CPTK_HAS_TRIVIAL_CONSTRUCTOR,
CPTK_HAS_TRIVIAL_COPY,
CPTK_HAS_TRIVIAL_DESTRUCTOR,
CPTK_HAS_UNIQUE_OBJ_REPRESENTATIONS,
CPTK_HAS_VIRTUAL_DESTRUCTOR,
CPTK_IS_ABSTRACT,
CPTK_IS_AGGREGATE,
CPTK_IS_BASE_OF,
CPTK_IS_CLASS,
CPTK_IS_EMPTY,
CPTK_IS_ENUM,
CPTK_IS_FINAL,
CPTK_IS_LITERAL_TYPE,
CPTK_IS_POD,
CPTK_IS_POLYMORPHIC,
CPTK_IS_SAME_AS,
CPTK_IS_STD_LAYOUT,
CPTK_IS_TRIVIAL,
CPTK_IS_TRIVIALLY_ASSIGNABLE,
CPTK_IS_TRIVIALLY_CONSTRUCTIBLE,
CPTK_IS_TRIVIALLY_COPYABLE,
CPTK_IS_UNION,
CPTK_UNDERLYING_TYPE,
CPTK_IS_ASSIGNABLE,
CPTK_IS_CONSTRUCTIBLE
};
#define TRAIT_EXPR_TYPE1(NODE) \
(((struct tree_trait_expr *)TRAIT_EXPR_CHECK (NODE))->type1)
#define TRAIT_EXPR_TYPE2(NODE) \
(((struct tree_trait_expr *)TRAIT_EXPR_CHECK (NODE))->type2)
#define TRAIT_EXPR_KIND(NODE) \
(((struct tree_trait_expr *)TRAIT_EXPR_CHECK (NODE))->kind)
struct GTY (()) tree_trait_expr {
struct tree_common common;
tree type1;
tree type2;  
enum cp_trait_kind kind;
};
#define LAMBDA_TYPE_P(NODE) \
(CLASS_TYPE_P (NODE) && CLASSTYPE_LAMBDA_EXPR (NODE))
#define LAMBDA_FUNCTION_P(FNDECL)				\
(DECL_DECLARES_FUNCTION_P (FNDECL)				\
&& DECL_OVERLOADED_OPERATOR_P (FNDECL)			\
&& DECL_OVERLOADED_OPERATOR_IS (FNDECL, CALL_EXPR)		\
&& LAMBDA_TYPE_P (CP_DECL_CONTEXT (FNDECL)))
enum cp_lambda_default_capture_mode_type {
CPLD_NONE,
CPLD_COPY,
CPLD_REFERENCE
};
#define LAMBDA_EXPR_DEFAULT_CAPTURE_MODE(NODE) \
(((struct tree_lambda_expr *)LAMBDA_EXPR_CHECK (NODE))->default_capture_mode)
#define LAMBDA_EXPR_CAPTURE_LIST(NODE) \
(((struct tree_lambda_expr *)LAMBDA_EXPR_CHECK (NODE))->capture_list)
#define LAMBDA_EXPR_THIS_CAPTURE(NODE) \
(((struct tree_lambda_expr *)LAMBDA_EXPR_CHECK (NODE))->this_capture)
#define LAMBDA_EXPR_CAPTURES_THIS_P(NODE) \
LAMBDA_EXPR_THIS_CAPTURE(NODE)
#define LAMBDA_EXPR_MUTABLE_P(NODE) \
TREE_LANG_FLAG_1 (LAMBDA_EXPR_CHECK (NODE))
#define LAMBDA_EXPR_CAPTURE_OPTIMIZED(NODE) \
TREE_LANG_FLAG_2 (LAMBDA_EXPR_CHECK (NODE))
#define LAMBDA_CAPTURE_EXPLICIT_P(NODE) \
TREE_LANG_FLAG_0 (TREE_LIST_CHECK (NODE))
#define LAMBDA_EXPR_LOCATION(NODE) \
(((struct tree_lambda_expr *)LAMBDA_EXPR_CHECK (NODE))->locus)
#define LAMBDA_EXPR_EXTRA_SCOPE(NODE) \
(((struct tree_lambda_expr *)LAMBDA_EXPR_CHECK (NODE))->extra_scope)
#define LAMBDA_EXPR_DISCRIMINATOR(NODE) \
(((struct tree_lambda_expr *)LAMBDA_EXPR_CHECK (NODE))->discriminator)
#define LAMBDA_EXPR_PENDING_PROXIES(NODE) \
(((struct tree_lambda_expr *)LAMBDA_EXPR_CHECK (NODE))->pending_proxies)
#define LAMBDA_EXPR_CLOSURE(NODE) \
(TREE_TYPE (LAMBDA_EXPR_CHECK (NODE)))
struct GTY (()) tree_lambda_expr
{
struct tree_typed typed;
tree capture_list;
tree this_capture;
tree extra_scope;
vec<tree, va_gc> *pending_proxies;
location_t locus;
enum cp_lambda_default_capture_mode_type default_capture_mode;
int discriminator;
};
struct GTY(()) qualified_typedef_usage_s {
tree typedef_decl;
tree context;
location_t locus;
};
typedef struct qualified_typedef_usage_s qualified_typedef_usage_t;
#define TINFO_HAS_ACCESS_ERRORS(NODE) \
(TREE_LANG_FLAG_0 (TEMPLATE_INFO_CHECK (NODE)))
#define FNDECL_HAS_ACCESS_ERRORS(NODE) \
(TINFO_HAS_ACCESS_ERRORS (DECL_TEMPLATE_INFO (NODE)))
#define TINFO_USED_TEMPLATE_ID(NODE) \
(TREE_LANG_FLAG_1 (TEMPLATE_INFO_CHECK (NODE)))
struct GTY(()) tree_template_info {
struct tree_common common;
vec<qualified_typedef_usage_t, va_gc> *typedefs_needing_access_checking;
};
struct GTY(()) tree_constraint_info {
struct tree_base base;
tree template_reqs;
tree declarator_reqs;
tree associated_constr;
};
template<typename T>
inline T*
check_nonnull (T* p)
{
gcc_assert (p);
return p;
}
inline tree_constraint_info *
check_constraint_info (tree t)
{
if (t && TREE_CODE (t) == CONSTRAINT_INFO)
return (tree_constraint_info *)t;
return NULL;
}
#define CI_TEMPLATE_REQS(NODE) \
check_constraint_info (check_nonnull(NODE))->template_reqs
#define CI_DECLARATOR_REQS(NODE) \
check_constraint_info (check_nonnull(NODE))->declarator_reqs
#define CI_ASSOCIATED_CONSTRAINTS(NODE) \
check_constraint_info (check_nonnull(NODE))->associated_constr
#define TEMPLATE_PARMS_CONSTRAINTS(NODE) \
TREE_TYPE (TREE_LIST_CHECK (NODE))
#define TEMPLATE_PARM_CONSTRAINTS(NODE) \
TREE_TYPE (TREE_LIST_CHECK (NODE))
#define COMPOUND_REQ_NOEXCEPT_P(NODE) \
TREE_LANG_FLAG_0 (TREE_CHECK (NODE, COMPOUND_REQ))
#define PLACEHOLDER_TYPE_CONSTRAINTS(NODE) \
DECL_SIZE_UNIT (TYPE_NAME (NODE))
#define PRED_CONSTR_EXPR(NODE) \
TREE_OPERAND (TREE_CHECK (NODE, PRED_CONSTR), 0)
#define CHECK_CONSTR_CONCEPT(NODE) \
TREE_OPERAND (TREE_CHECK (NODE, CHECK_CONSTR), 0)
#define CHECK_CONSTR_ARGS(NODE) \
TREE_OPERAND (TREE_CHECK (NODE, CHECK_CONSTR), 1)
#define EXPR_CONSTR_EXPR(NODE) \
TREE_OPERAND (TREE_CHECK (NODE, EXPR_CONSTR), 0)
#define TYPE_CONSTR_TYPE(NODE) \
TREE_OPERAND (TREE_CHECK (NODE, TYPE_CONSTR), 0)
#define ICONV_CONSTR_EXPR(NODE) \
TREE_OPERAND (TREE_CHECK (NODE, ICONV_CONSTR), 0)
#define ICONV_CONSTR_TYPE(NODE) \
TREE_OPERAND (TREE_CHECK (NODE, ICONV_CONSTR), 1)
#define DEDUCT_CONSTR_EXPR(NODE) \
TREE_OPERAND (TREE_CHECK (NODE, DEDUCT_CONSTR), 0)
#define DEDUCT_CONSTR_PATTERN(NODE) \
TREE_OPERAND (TREE_CHECK (NODE, DEDUCT_CONSTR), 1)
#define DEDUCT_CONSTR_PLACEHOLDER(NODE) \
TREE_OPERAND (TREE_CHECK (NODE, DEDUCT_CONSTR), 2)
#define EXCEPT_CONSTR_EXPR(NODE) \
TREE_OPERAND (TREE_CHECK (NODE, EXCEPT_CONSTR), 0)
#define PARM_CONSTR_PARMS(NODE) \
TREE_OPERAND (TREE_CHECK (NODE, PARM_CONSTR), 0)
#define PARM_CONSTR_OPERAND(NODE) \
TREE_OPERAND (TREE_CHECK (NODE, PARM_CONSTR), 1)
#define CONSTRAINT_VAR_P(NODE) \
DECL_LANG_FLAG_2 (TREE_CHECK (NODE, PARM_DECL))
#define CONSTRAINED_PARM_CONCEPT(NODE) \
DECL_SIZE_UNIT (TYPE_DECL_CHECK (NODE))
#define CONSTRAINED_PARM_EXTRA_ARGS(NODE) \
DECL_SIZE (TYPE_DECL_CHECK (NODE))
#define CONSTRAINED_PARM_PROTOTYPE(NODE) \
DECL_INITIAL (TYPE_DECL_CHECK (NODE))
enum cp_tree_node_structure_enum {
TS_CP_GENERIC,
TS_CP_IDENTIFIER,
TS_CP_TPI,
TS_CP_PTRMEM,
TS_CP_OVERLOAD,
TS_CP_BASELINK,
TS_CP_TEMPLATE_DECL,
TS_CP_DEFAULT_ARG,
TS_CP_DEFERRED_NOEXCEPT,
TS_CP_STATIC_ASSERT,
TS_CP_ARGUMENT_PACK_SELECT,
TS_CP_TRAIT_EXPR,
TS_CP_LAMBDA_EXPR,
TS_CP_TEMPLATE_INFO,
TS_CP_CONSTRAINT_INFO,
TS_CP_USERDEF_LITERAL
};
union GTY((desc ("cp_tree_node_structure (&%h)"),
chain_next ("(union lang_tree_node *) c_tree_chain_next (&%h.generic)"))) lang_tree_node {
union tree_node GTY ((tag ("TS_CP_GENERIC"),
desc ("tree_node_structure (&%h)"))) generic;
struct template_parm_index GTY ((tag ("TS_CP_TPI"))) tpi;
struct ptrmem_cst GTY ((tag ("TS_CP_PTRMEM"))) ptrmem;
struct tree_overload GTY ((tag ("TS_CP_OVERLOAD"))) overload;
struct tree_baselink GTY ((tag ("TS_CP_BASELINK"))) baselink;
struct tree_template_decl GTY ((tag ("TS_CP_TEMPLATE_DECL"))) template_decl;
struct tree_default_arg GTY ((tag ("TS_CP_DEFAULT_ARG"))) default_arg;
struct tree_deferred_noexcept GTY ((tag ("TS_CP_DEFERRED_NOEXCEPT"))) deferred_noexcept;
struct lang_identifier GTY ((tag ("TS_CP_IDENTIFIER"))) identifier;
struct tree_static_assert GTY ((tag ("TS_CP_STATIC_ASSERT"))) 
static_assertion;
struct tree_argument_pack_select GTY ((tag ("TS_CP_ARGUMENT_PACK_SELECT")))
argument_pack_select;
struct tree_trait_expr GTY ((tag ("TS_CP_TRAIT_EXPR")))
trait_expression;
struct tree_lambda_expr GTY ((tag ("TS_CP_LAMBDA_EXPR")))
lambda_expression;
struct tree_template_info GTY ((tag ("TS_CP_TEMPLATE_INFO")))
template_info;
struct tree_constraint_info GTY ((tag ("TS_CP_CONSTRAINT_INFO")))
constraint_info;
struct tree_userdef_literal GTY ((tag ("TS_CP_USERDEF_LITERAL")))
userdef_literal;
};

struct GTY(()) saved_scope {
vec<cxx_saved_binding, va_gc> *old_bindings;
tree old_namespace;
vec<tree, va_gc> *decl_ns_list;
tree class_name;
tree class_type;
tree access_specifier;
tree function_decl;
vec<tree, va_gc> *lang_base;
tree lang_name;
tree template_parms;
cp_binding_level *x_previous_class_level;
tree x_saved_tree;
tree x_current_class_ptr;
tree x_current_class_ref;
int x_processing_template_decl;
int x_processing_specialization;
BOOL_BITFIELD x_processing_explicit_instantiation : 1;
BOOL_BITFIELD need_pop_function_context : 1;
BOOL_BITFIELD discarded_stmt : 1;
int unevaluated_operand;
int inhibit_evaluation_warnings;
int noexcept_operand;
int omp_declare_target_attribute;
struct stmt_tree_s x_stmt_tree;
cp_binding_level *class_bindings;
cp_binding_level *bindings;
hash_map<tree, tree> *GTY((skip)) x_local_specializations;
struct saved_scope *prev;
};
extern GTY(()) struct saved_scope *scope_chain;
#define current_namespace scope_chain->old_namespace
#define decl_namespace_list scope_chain->decl_ns_list
#define current_class_name scope_chain->class_name
#define current_class_type scope_chain->class_type
#define current_access_specifier scope_chain->access_specifier
#define current_lang_base scope_chain->lang_base
#define current_lang_name scope_chain->lang_name
#define current_template_parms scope_chain->template_parms
#define processing_template_decl scope_chain->x_processing_template_decl
#define processing_specialization scope_chain->x_processing_specialization
#define processing_explicit_instantiation scope_chain->x_processing_explicit_instantiation
#define in_discarded_stmt scope_chain->discarded_stmt
struct processing_template_decl_sentinel
{
int saved;
processing_template_decl_sentinel (bool reset = true)
: saved (processing_template_decl)
{
if (reset)
processing_template_decl = 0;
}
~processing_template_decl_sentinel()
{
processing_template_decl = saved;
}
};
struct warning_sentinel
{
int &flag;
int val;
warning_sentinel(int& flag, bool suppress=true)
: flag(flag), val(flag) { if (suppress) flag = 0; }
~warning_sentinel() { flag = val; }
};
template <typename T>
class temp_override
{
T& overridden_variable;
T saved_value;
public:
temp_override(T& var) : overridden_variable (var), saved_value (var) {}
temp_override(T& var, T overrider)
: overridden_variable (var), saved_value (var)
{
overridden_variable = overrider;
}
~temp_override() { overridden_variable = saved_value; }
};
#define previous_class_level scope_chain->x_previous_class_level
#define local_specializations scope_chain->x_local_specializations
#define cp_noexcept_operand scope_chain->noexcept_operand
struct GTY((for_user)) cxx_int_tree_map {
unsigned int uid;
tree to;
};
struct cxx_int_tree_map_hasher : ggc_ptr_hash<cxx_int_tree_map>
{
static hashval_t hash (cxx_int_tree_map *);
static bool equal (cxx_int_tree_map *, cxx_int_tree_map *);
};
struct named_label_entry; 
struct named_label_hash : ggc_remove <named_label_entry *>
{
typedef named_label_entry *value_type;
typedef tree compare_type; 
inline static hashval_t hash (value_type);
inline static bool equal (const value_type, compare_type);
inline static void mark_empty (value_type &p) {p = NULL;}
inline static bool is_empty (value_type p) {return !p;}
inline static bool is_deleted (value_type) { return false; }
inline static void mark_deleted (value_type) { gcc_unreachable (); }
};
struct GTY(()) language_function {
struct c_language_function base;
tree x_cdtor_label;
tree x_current_class_ptr;
tree x_current_class_ref;
tree x_eh_spec_block;
tree x_in_charge_parm;
tree x_vtt_parm;
tree x_return_value;
tree x_auto_return_pattern;
BOOL_BITFIELD returns_value : 1;
BOOL_BITFIELD returns_null : 1;
BOOL_BITFIELD returns_abnormally : 1;
BOOL_BITFIELD infinite_loop: 1;
BOOL_BITFIELD x_in_function_try_handler : 1;
BOOL_BITFIELD x_in_base_initializer : 1;
BOOL_BITFIELD can_throw : 1;
BOOL_BITFIELD invalid_constexpr : 1;
hash_table<named_label_hash> *x_named_labels;
cp_binding_level *bindings;
vec<tree, va_gc> *x_local_names;
vec<tree, va_gc> *infinite_loops;
hash_table<cxx_int_tree_map_hasher> *extern_decl_map;
};
#define cp_function_chain (cfun->language)
#define cdtor_label cp_function_chain->x_cdtor_label
#define current_class_ptr			\
(*(cfun && cp_function_chain			\
? &cp_function_chain->x_current_class_ptr	\
: &scope_chain->x_current_class_ptr))
#define current_class_ref			\
(*(cfun && cp_function_chain			\
? &cp_function_chain->x_current_class_ref	\
: &scope_chain->x_current_class_ref))
#define current_eh_spec_block cp_function_chain->x_eh_spec_block
#define current_in_charge_parm cp_function_chain->x_in_charge_parm
#define current_vtt_parm cp_function_chain->x_vtt_parm
#define current_function_returns_value cp_function_chain->returns_value
#define current_function_returns_null cp_function_chain->returns_null
#define current_function_returns_abnormally \
cp_function_chain->returns_abnormally
#define current_function_infinite_loop cp_function_chain->infinite_loop
#define in_base_initializer cp_function_chain->x_in_base_initializer
#define in_function_try_handler cp_function_chain->x_in_function_try_handler
#define current_function_return_value \
(cp_function_chain->x_return_value)
#define current_function_auto_return_pattern \
(cp_function_chain->x_auto_return_pattern)
extern tree cp_literal_operator_id (const char *);
extern bool statement_code_p[MAX_TREE_CODES];
#define STATEMENT_CODE_P(CODE) statement_code_p[(int) (CODE)]
enum languages { lang_c, lang_cplusplus };
#define TYPE_LINKAGE_IDENTIFIER(NODE) \
(TYPE_IDENTIFIER (TYPE_MAIN_VARIANT (NODE)))
#define TYPE_NAME_STRING(NODE) (IDENTIFIER_POINTER (TYPE_IDENTIFIER (NODE)))
#define TYPE_NAME_LENGTH(NODE) (IDENTIFIER_LENGTH (TYPE_IDENTIFIER (NODE)))
#define TYPE_UNNAMED_P(NODE) \
(OVERLOAD_TYPE_P (NODE) && anon_aggrname_p (TYPE_LINKAGE_IDENTIFIER (NODE)))
#define TYPE_MAIN_DECL(NODE) (TYPE_STUB_DECL (TYPE_MAIN_VARIANT (NODE)))
#define WILDCARD_TYPE_P(T)				\
(TREE_CODE (T) == TEMPLATE_TYPE_PARM			\
|| TREE_CODE (T) == TYPENAME_TYPE			\
|| TREE_CODE (T) == TYPEOF_TYPE			\
|| TREE_CODE (T) == BOUND_TEMPLATE_TEMPLATE_PARM	\
|| TREE_CODE (T) == DECLTYPE_TYPE)
#define MAYBE_CLASS_TYPE_P(T) (WILDCARD_TYPE_P (T) || CLASS_TYPE_P (T))
#define SET_CLASS_TYPE_P(T, VAL) \
(TYPE_LANG_FLAG_5 (RECORD_OR_UNION_CHECK (T)) = (VAL))
#define CLASS_TYPE_P(T) \
(RECORD_OR_UNION_CODE_P (TREE_CODE (T)) && TYPE_LANG_FLAG_5 (T))
#define NON_UNION_CLASS_TYPE_P(T) \
(TREE_CODE (T) == RECORD_TYPE && TYPE_LANG_FLAG_5 (T))
#define RECORD_OR_UNION_CODE_P(T)	\
((T) == RECORD_TYPE || (T) == UNION_TYPE)
#define OVERLOAD_TYPE_P(T) \
(CLASS_TYPE_P (T) || TREE_CODE (T) == ENUMERAL_TYPE)
#define TYPE_DEPENDENT_P(NODE) TYPE_LANG_FLAG_0 (NODE)
#define TYPE_DEPENDENT_P_VALID(NODE) TYPE_LANG_FLAG_6(NODE)
#define CP_TYPE_CONST_P(NODE)				\
((cp_type_quals (NODE) & TYPE_QUAL_CONST) != 0)
#define CP_TYPE_VOLATILE_P(NODE)			\
((cp_type_quals (NODE) & TYPE_QUAL_VOLATILE) != 0)
#define CP_TYPE_RESTRICT_P(NODE)			\
((cp_type_quals (NODE) & TYPE_QUAL_RESTRICT) != 0)
#define CP_TYPE_CONST_NON_VOLATILE_P(NODE)				\
((cp_type_quals (NODE) & (TYPE_QUAL_CONST | TYPE_QUAL_VOLATILE))	\
== TYPE_QUAL_CONST)
#define FUNCTION_ARG_CHAIN(NODE) \
TREE_CHAIN (TYPE_ARG_TYPES (TREE_TYPE (NODE)))
#define FUNCTION_FIRST_USER_PARMTYPE(NODE) \
skip_artificial_parms_for ((NODE), TYPE_ARG_TYPES (TREE_TYPE (NODE)))
#define FUNCTION_FIRST_USER_PARM(NODE) \
skip_artificial_parms_for ((NODE), DECL_ARGUMENTS (NODE))
#define DERIVED_FROM_P(PARENT, TYPE) \
(lookup_base ((TYPE), (PARENT), ba_any, NULL, tf_none) != NULL_TREE)
#define CLASSTYPE_VISIBILITY(TYPE)		\
DECL_VISIBILITY (TYPE_MAIN_DECL (TYPE))
#define CLASSTYPE_VISIBILITY_SPECIFIED(TYPE)	\
DECL_VISIBILITY_SPECIFIED (TYPE_MAIN_DECL (TYPE))
struct GTY (()) tree_pair_s {
tree purpose;
tree value;
};
typedef tree_pair_s *tree_pair_p;
struct GTY(()) lang_type {
unsigned char align;
unsigned has_type_conversion : 1;
unsigned has_copy_ctor : 1;
unsigned has_default_ctor : 1;
unsigned const_needs_init : 1;
unsigned ref_needs_init : 1;
unsigned has_const_copy_assign : 1;
unsigned use_template : 2;
unsigned has_mutable : 1;
unsigned com_interface : 1;
unsigned non_pod_class : 1;
unsigned nearly_empty_p : 1;
unsigned user_align : 1;
unsigned has_copy_assign : 1;
unsigned has_new : 1;
unsigned has_array_new : 1;
unsigned gets_delete : 2;
unsigned interface_only : 1;
unsigned interface_unknown : 1;
unsigned contains_empty_class_p : 1;
unsigned anon_aggr : 1;
unsigned non_zero_init : 1;
unsigned empty_p : 1;
unsigned vec_new_uses_cookie : 1;
unsigned declared_class : 1;
unsigned diamond_shaped : 1;
unsigned repeated_base : 1;
unsigned being_defined : 1;
unsigned debug_requested : 1;
unsigned fields_readonly : 1;
unsigned ptrmemfunc_flag : 1;
unsigned was_anonymous : 1;
unsigned lazy_default_ctor : 1;
unsigned lazy_copy_ctor : 1;
unsigned lazy_copy_assign : 1;
unsigned lazy_destructor : 1;
unsigned has_const_copy_ctor : 1;
unsigned has_complex_copy_ctor : 1;
unsigned has_complex_copy_assign : 1;
unsigned non_aggregate : 1;
unsigned has_complex_dflt : 1;
unsigned has_list_ctor : 1;
unsigned non_std_layout : 1;
unsigned is_literal : 1;
unsigned lazy_move_ctor : 1;
unsigned lazy_move_assign : 1;
unsigned has_complex_move_ctor : 1;
unsigned has_complex_move_assign : 1;
unsigned has_constexpr_ctor : 1;
unsigned unique_obj_representations : 1;
unsigned unique_obj_representations_set : 1;
unsigned dummy : 4;
tree primary_base;
vec<tree_pair_s, va_gc> *vcall_indices;
tree vtables;
tree typeinfo_var;
vec<tree, va_gc> *vbases;
binding_table nested_udts;
tree as_base;
vec<tree, va_gc> *pure_virtuals;
tree friend_classes;
vec<tree, va_gc> * GTY((reorder ("resort_type_member_vec"))) members;
tree key_method;
tree decl_list;
tree befriending_classes;
tree objc_info;
tree lambda_expr;
};
#define LANG_TYPE_CLASS_CHECK(NODE) (TYPE_LANG_SPECIFIC (NODE))
#define TYPE_GETS_DELETE(NODE) (LANG_TYPE_CLASS_CHECK (NODE)->gets_delete)
#define TYPE_GETS_REG_DELETE(NODE) (TYPE_GETS_DELETE (NODE) & 1)
#define TYPE_VEC_NEW_USES_COOKIE(NODE)			\
(CLASS_TYPE_P (NODE)					\
&& LANG_TYPE_CLASS_CHECK (NODE)->vec_new_uses_cookie)
#define TYPE_HAS_CONVERSION(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->has_type_conversion)
#define CLASSTYPE_LAZY_DEFAULT_CTOR(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->lazy_default_ctor)
#define CLASSTYPE_LAZY_COPY_CTOR(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->lazy_copy_ctor)
#define CLASSTYPE_LAZY_MOVE_CTOR(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->lazy_move_ctor)
#define CLASSTYPE_LAZY_COPY_ASSIGN(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->lazy_copy_assign)
#define CLASSTYPE_LAZY_MOVE_ASSIGN(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->lazy_move_assign)
#define CLASSTYPE_LAZY_DESTRUCTOR(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->lazy_destructor)
#define CLASSTYPE_FINAL(NODE) \
TYPE_FINAL_P (NODE)
#define TYPE_HAS_COPY_ASSIGN(NODE) (LANG_TYPE_CLASS_CHECK (NODE)->has_copy_assign)
#define TYPE_HAS_CONST_COPY_ASSIGN(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->has_const_copy_assign)
#define TYPE_HAS_COPY_CTOR(NODE) (LANG_TYPE_CLASS_CHECK (NODE)->has_copy_ctor)
#define TYPE_HAS_CONST_COPY_CTOR(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->has_const_copy_ctor)
#define TYPE_HAS_LIST_CTOR(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->has_list_ctor)
#define TYPE_HAS_CONSTEXPR_CTOR(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->has_constexpr_ctor)
#define TYPE_HAS_NEW_OPERATOR(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->has_new)
#define TYPE_HAS_ARRAY_NEW_OPERATOR(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->has_array_new)
#define TYPE_BEING_DEFINED(NODE) (LANG_TYPE_CLASS_CHECK (NODE)->being_defined)
#define COMPLETE_OR_OPEN_TYPE_P(NODE) \
(COMPLETE_TYPE_P (NODE) || (CLASS_TYPE_P (NODE) && TYPE_BEING_DEFINED (NODE)))
#define TYPE_MARKED_P(NODE) TREE_LANG_FLAG_6 (TYPE_CHECK (NODE))
#define CLASSTYPE_DIAMOND_SHAPED_P(NODE) \
(LANG_TYPE_CLASS_CHECK(NODE)->diamond_shaped)
#define CLASSTYPE_REPEATED_BASE_P(NODE) \
(LANG_TYPE_CLASS_CHECK(NODE)->repeated_base)
#define CLASSTYPE_KEY_METHOD(NODE) (LANG_TYPE_CLASS_CHECK (NODE)->key_method)
#define CLASSTYPE_MEMBER_VEC(NODE) (LANG_TYPE_CLASS_CHECK (NODE)->members)
#define CLASSTYPE_DECL_LIST(NODE) (LANG_TYPE_CLASS_CHECK (NODE)->decl_list)
#define CLASSTYPE_CONSTRUCTORS(NODE) \
(get_class_binding_direct (NODE, ctor_identifier))
#define CLASSTYPE_DESTRUCTOR(NODE) \
(get_class_binding_direct (NODE, dtor_identifier))
#define CLASSTYPE_NESTED_UTDS(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->nested_udts)
#define CLASSTYPE_HAS_PRIMARY_BASE_P(NODE) \
(CLASSTYPE_PRIMARY_BINFO (NODE) != NULL_TREE)
#define CLASSTYPE_PRIMARY_BINFO(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->primary_base)
#define CLASSTYPE_VBASECLASSES(NODE) (LANG_TYPE_CLASS_CHECK (NODE)->vbases)
#define CLASSTYPE_AS_BASE(NODE) (LANG_TYPE_CLASS_CHECK (NODE)->as_base)
#define IS_FAKE_BASE_TYPE(NODE)					\
(TREE_CODE (NODE) == RECORD_TYPE				\
&& TYPE_CONTEXT (NODE) && CLASS_TYPE_P (TYPE_CONTEXT (NODE))	\
&& CLASSTYPE_AS_BASE (TYPE_CONTEXT (NODE)) == (NODE))
#define CLASSTYPE_SIZE(NODE) TYPE_SIZE (CLASSTYPE_AS_BASE (NODE))
#define CLASSTYPE_SIZE_UNIT(NODE) TYPE_SIZE_UNIT (CLASSTYPE_AS_BASE (NODE))
#define CLASSTYPE_ALIGN(NODE) TYPE_ALIGN (CLASSTYPE_AS_BASE (NODE))
#define CLASSTYPE_USER_ALIGN(NODE) TYPE_USER_ALIGN (CLASSTYPE_AS_BASE (NODE))
#define CLASSTYPE_ALIGN_UNIT(NODE) \
(CLASSTYPE_ALIGN (NODE) / BITS_PER_UNIT)
#define CLASSTYPE_PURE_VIRTUALS(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->pure_virtuals)
#define ABSTRACT_CLASS_TYPE_P(NODE) \
(CLASS_TYPE_P (NODE) && CLASSTYPE_PURE_VIRTUALS(NODE))
#define TYPE_HAS_DEFAULT_CONSTRUCTOR(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->has_default_ctor)
#define CLASSTYPE_HAS_MUTABLE(NODE) (LANG_TYPE_CLASS_CHECK (NODE)->has_mutable)
#define TYPE_HAS_MUTABLE_P(NODE) (cp_has_mutable_p (NODE))
#define CLASSTYPE_NON_LAYOUT_POD_P(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->non_pod_class)
#define CLASSTYPE_NON_STD_LAYOUT(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->non_std_layout)
#define CLASSTYPE_UNIQUE_OBJ_REPRESENTATIONS(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->unique_obj_representations)
#define CLASSTYPE_UNIQUE_OBJ_REPRESENTATIONS_SET(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->unique_obj_representations_set)
#define CLASSTYPE_NON_ZERO_INIT_P(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->non_zero_init)
#define CLASSTYPE_EMPTY_P(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->empty_p)
#define CLASSTYPE_NEARLY_EMPTY_P(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->nearly_empty_p)
#define CLASSTYPE_CONTAINS_EMPTY_CLASS_P(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->contains_empty_class_p)
#define CLASSTYPE_FRIEND_CLASSES(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->friend_classes)
#define CLASSTYPE_BEFRIENDING_CLASSES(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->befriending_classes)
#define CLASSTYPE_LAMBDA_EXPR(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->lambda_expr)
#define LAMBDA_TYPE_EXTRA_SCOPE(NODE) \
(LAMBDA_EXPR_EXTRA_SCOPE (CLASSTYPE_LAMBDA_EXPR (NODE)))
#define CLASSTYPE_DECLARED_CLASS(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->declared_class)
#define CLASSTYPE_READONLY_FIELDS_NEED_INIT(NODE)	\
(TYPE_LANG_SPECIFIC (NODE)				\
? LANG_TYPE_CLASS_CHECK (NODE)->const_needs_init : 0)
#define SET_CLASSTYPE_READONLY_FIELDS_NEED_INIT(NODE, VALUE) \
(LANG_TYPE_CLASS_CHECK (NODE)->const_needs_init = (VALUE))
#define CLASSTYPE_REF_FIELDS_NEED_INIT(NODE)		\
(TYPE_LANG_SPECIFIC (NODE)				\
? LANG_TYPE_CLASS_CHECK (NODE)->ref_needs_init : 0)
#define SET_CLASSTYPE_REF_FIELDS_NEED_INIT(NODE, VALUE) \
(LANG_TYPE_CLASS_CHECK (NODE)->ref_needs_init = (VALUE))
#define CLASSTYPE_INTERFACE_ONLY(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->interface_only)
#define CLASSTYPE_INTERFACE_KNOWN(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->interface_unknown == 0)
#define CLASSTYPE_INTERFACE_UNKNOWN(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->interface_unknown)
#define SET_CLASSTYPE_INTERFACE_UNKNOWN_X(NODE,X) \
(LANG_TYPE_CLASS_CHECK (NODE)->interface_unknown = !!(X))
#define SET_CLASSTYPE_INTERFACE_UNKNOWN(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->interface_unknown = 1)
#define SET_CLASSTYPE_INTERFACE_KNOWN(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->interface_unknown = 0)
#define CLASSTYPE_DEBUG_REQUESTED(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->debug_requested)

#define BINFO_VTABLE_PATH_MARKED(NODE) BINFO_FLAG_1 (NODE)
#define BINFO_NEW_VTABLE_MARKED(B) (BINFO_FLAG_2 (B))
#define SAME_BINFO_TYPE_P(A, B) ((A) == (B))
#define SET_BINFO_NEW_VTABLE_MARKED(B)					 \
(BINFO_NEW_VTABLE_MARKED (B) = 1,					 \
gcc_assert (!BINFO_PRIMARY_P (B) || BINFO_VIRTUAL_P (B)),		 \
gcc_assert (TYPE_VFIELD (BINFO_TYPE (B))))
#define BINFO_DEPENDENT_BASE_P(NODE) BINFO_FLAG_3 (NODE)
#define BINFO_LOST_PRIMARY_P(NODE) BINFO_FLAG_4 (NODE)
#define BINFO_PRIMARY_P(NODE) BINFO_FLAG_5(NODE)

#define CLASSTYPE_VCALL_INDICES(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->vcall_indices)
#define CLASSTYPE_VTABLES(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->vtables)
#define CLASSTYPE_TYPEINFO_VAR(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->typeinfo_var)
#define BV_DELTA(NODE) (TREE_PURPOSE (NODE))
#define BV_VCALL_INDEX(NODE) (TREE_TYPE (NODE))
#define BV_FN(NODE) (TREE_VALUE (NODE))
#define BV_LOST_PRIMARY(NODE) (TREE_LANG_FLAG_0 (NODE))

#define TYPE_RAISES_EXCEPTIONS(NODE) \
TYPE_LANG_SLOT_1 (FUNC_OR_METHOD_CHECK (NODE))
#define TYPE_NOTHROW_P(NODE) nothrow_spec_p (TYPE_RAISES_EXCEPTIONS (NODE))
#define TYPE_NOEXCEPT_P(NODE) type_noexcept_p (NODE)
#define NAMESPACE_LEVEL(NODE) \
(LANG_DECL_NS_CHECK (NODE)->level)

enum lang_decl_selector
{
lds_min,
lds_fn,
lds_ns,
lds_parm,
lds_decomp
};
struct GTY(()) lang_decl_base {
ENUM_BITFIELD(lang_decl_selector) selector : 16;
ENUM_BITFIELD(languages) language : 1;
unsigned use_template : 2;
unsigned not_really_extern : 1;	   
unsigned initialized_in_class : 1;	   
unsigned repo_available_p : 1;	   
unsigned threadprivate_or_deleted_p : 1; 
unsigned anticipated_p : 1;		   
unsigned friend_or_tls : 1;		   
unsigned unknown_bound_p : 1;		   
unsigned odr_used : 1;		   
unsigned u2sel : 1;
unsigned concept_p : 1;                  
unsigned var_declared_inline_p : 1;	   
unsigned dependent_init_p : 1;	   
};
#define LANG_DECL_HAS_MIN(NODE)			\
(VAR_OR_FUNCTION_DECL_P (NODE)		\
|| TREE_CODE (NODE) == FIELD_DECL		\
|| TREE_CODE (NODE) == CONST_DECL		\
|| TREE_CODE (NODE) == TYPE_DECL		\
|| TREE_CODE (NODE) == TEMPLATE_DECL		\
|| TREE_CODE (NODE) == USING_DECL)
struct GTY(()) lang_decl_min {
struct lang_decl_base base;
tree template_info;
union lang_decl_u2 {
tree GTY ((tag ("0"))) access;
int GTY ((tag ("1"))) discriminator;
} GTY ((desc ("%0.u.base.u2sel"))) u2;
};
struct GTY(()) lang_decl_fn {
struct lang_decl_min min;
unsigned ovl_op_code : 6;
unsigned global_ctor_p : 1;
unsigned global_dtor_p : 1;
unsigned static_function : 1;
unsigned pure_virtual : 1;
unsigned defaulted_p : 1;
unsigned has_in_charge_parm_p : 1;
unsigned has_vtt_parm_p : 1;
unsigned pending_inline_p : 1;
unsigned nonconverting : 1;
unsigned thunk_p : 1;
unsigned this_thunk_p : 1;
unsigned hidden_friend_p : 1;
unsigned omp_declare_reduction_p : 1;
unsigned spare : 13;
tree befriending_classes;
tree context;
union lang_decl_u5
{
tree GTY ((tag ("0"))) cloned_function;
HOST_WIDE_INT GTY ((tag ("1"))) fixed_offset;
} GTY ((desc ("%1.thunk_p"))) u5;
union lang_decl_u3
{
struct cp_token_cache * GTY ((tag ("1"))) pending_inline_info;
struct language_function * GTY ((tag ("0")))
saved_language_function;
} GTY ((desc ("%1.pending_inline_p"))) u;
};
struct GTY(()) lang_decl_ns {
struct lang_decl_base base;
cp_binding_level *level;
vec<tree, va_gc> *usings;
vec<tree, va_gc> *inlinees;
hash_table<named_decl_hash> *bindings;
};
struct GTY(()) lang_decl_parm {
struct lang_decl_base base;
int level;
int index;
};
struct GTY(()) lang_decl_decomp {
struct lang_decl_min min;
tree base;
};
struct GTY(()) lang_decl {
union GTY((desc ("%h.base.selector"))) lang_decl_u {
struct lang_decl_base GTY ((default)) base;
struct lang_decl_min GTY((tag ("lds_min"))) min;
struct lang_decl_fn GTY ((tag ("lds_fn"))) fn;
struct lang_decl_ns GTY((tag ("lds_ns"))) ns;
struct lang_decl_parm GTY((tag ("lds_parm"))) parm;
struct lang_decl_decomp GTY((tag ("lds_decomp"))) decomp;
} u;
};
#define STRIP_TEMPLATE(NODE) \
(TREE_CODE (NODE) == TEMPLATE_DECL ? DECL_TEMPLATE_RESULT (NODE) : NODE)
#if defined ENABLE_TREE_CHECKING && (GCC_VERSION >= 2007)
#define LANG_DECL_MIN_CHECK(NODE) __extension__			\
({ struct lang_decl *lt = DECL_LANG_SPECIFIC (NODE);		\
if (!LANG_DECL_HAS_MIN (NODE))				\
lang_check_failed (__FILE__, __LINE__, __FUNCTION__);	\
&lt->u.min; })
#define LANG_DECL_FN_CHECK(NODE) __extension__				\
({ struct lang_decl *lt = DECL_LANG_SPECIFIC (STRIP_TEMPLATE (NODE));	\
if (!DECL_DECLARES_FUNCTION_P (NODE)					\
|| lt->u.base.selector != lds_fn)				\
lang_check_failed (__FILE__, __LINE__, __FUNCTION__);		\
&lt->u.fn; })
#define LANG_DECL_NS_CHECK(NODE) __extension__				\
({ struct lang_decl *lt = DECL_LANG_SPECIFIC (NODE);			\
if (TREE_CODE (NODE) != NAMESPACE_DECL				\
|| lt->u.base.selector != lds_ns)				\
lang_check_failed (__FILE__, __LINE__, __FUNCTION__);		\
&lt->u.ns; })
#define LANG_DECL_PARM_CHECK(NODE) __extension__		\
({ struct lang_decl *lt = DECL_LANG_SPECIFIC (NODE);		\
if (TREE_CODE (NODE) != PARM_DECL				\
|| lt->u.base.selector != lds_parm)			\
lang_check_failed (__FILE__, __LINE__, __FUNCTION__);	\
&lt->u.parm; })
#define LANG_DECL_DECOMP_CHECK(NODE) __extension__		\
({ struct lang_decl *lt = DECL_LANG_SPECIFIC (NODE);		\
if (!VAR_P (NODE)						\
|| lt->u.base.selector != lds_decomp)			\
lang_check_failed (__FILE__, __LINE__, __FUNCTION__);	\
&lt->u.decomp; })
#define LANG_DECL_U2_CHECK(NODE, TF) __extension__		\
({  struct lang_decl *lt = DECL_LANG_SPECIFIC (NODE);		\
if (!LANG_DECL_HAS_MIN (NODE) || lt->u.base.u2sel != TF)	\
lang_check_failed (__FILE__, __LINE__, __FUNCTION__);	\
&lt->u.min.u2; })
#else
#define LANG_DECL_MIN_CHECK(NODE) \
(&DECL_LANG_SPECIFIC (NODE)->u.min)
#define LANG_DECL_FN_CHECK(NODE) \
(&DECL_LANG_SPECIFIC (STRIP_TEMPLATE (NODE))->u.fn)
#define LANG_DECL_NS_CHECK(NODE) \
(&DECL_LANG_SPECIFIC (NODE)->u.ns)
#define LANG_DECL_PARM_CHECK(NODE) \
(&DECL_LANG_SPECIFIC (NODE)->u.parm)
#define LANG_DECL_DECOMP_CHECK(NODE) \
(&DECL_LANG_SPECIFIC (NODE)->u.decomp)
#define LANG_DECL_U2_CHECK(NODE, TF) \
(&DECL_LANG_SPECIFIC (NODE)->u.min.u2)
#endif 
#define DECL_LANGUAGE(NODE)				\
(DECL_LANG_SPECIFIC (NODE)				\
? DECL_LANG_SPECIFIC (NODE)->u.base.language		\
: (TREE_CODE (NODE) == FUNCTION_DECL			\
? lang_c : lang_cplusplus))
#define SET_DECL_LANGUAGE(NODE, LANGUAGE) \
(DECL_LANG_SPECIFIC (NODE)->u.base.language = (LANGUAGE))
#define DECL_CONSTRUCTOR_P(NODE) \
IDENTIFIER_CTOR_P (DECL_NAME (NODE))
#define DECL_COMPLETE_CONSTRUCTOR_P(NODE)		\
(DECL_NAME (NODE) == complete_ctor_identifier)
#define DECL_BASE_CONSTRUCTOR_P(NODE)		\
(DECL_NAME (NODE) == base_ctor_identifier)
#define DECL_MAYBE_IN_CHARGE_CONSTRUCTOR_P(NODE)		\
(DECL_NAME (NODE) == ctor_identifier)
#define DECL_COPY_CONSTRUCTOR_P(NODE) \
(DECL_CONSTRUCTOR_P (NODE) && copy_fn_p (NODE) > 0)
#define DECL_MOVE_CONSTRUCTOR_P(NODE) \
(DECL_CONSTRUCTOR_P (NODE) && move_fn_p (NODE))
#define DECL_DESTRUCTOR_P(NODE)				\
IDENTIFIER_DTOR_P (DECL_NAME (NODE))
#define DECL_MAYBE_IN_CHARGE_DESTRUCTOR_P(NODE)			\
(DECL_NAME (NODE) == dtor_identifier)
#define DECL_COMPLETE_DESTRUCTOR_P(NODE)		\
(DECL_NAME (NODE) == complete_dtor_identifier)
#define DECL_BASE_DESTRUCTOR_P(NODE)		\
(DECL_NAME (NODE) == base_dtor_identifier)
#define DECL_DELETING_DESTRUCTOR_P(NODE)		\
(DECL_NAME (NODE) == deleting_dtor_identifier)
#define DECL_CLONED_FUNCTION_P(NODE) (!!decl_cloned_function_p (NODE, true))
#define DECL_CLONED_FUNCTION(NODE) (*decl_cloned_function_p (NODE, false))
#define FOR_EACH_CLONE(CLONE, FN)			\
if (!(TREE_CODE (FN) == FUNCTION_DECL			\
&& (DECL_MAYBE_IN_CHARGE_CONSTRUCTOR_P (FN)	\
|| DECL_MAYBE_IN_CHARGE_DESTRUCTOR_P (FN))))\
;							\
else							\
for (CLONE = DECL_CHAIN (FN);			\
CLONE && DECL_CLONED_FUNCTION_P (CLONE);	\
CLONE = DECL_CHAIN (CLONE))
#define DECL_DISCRIMINATOR_P(NODE)	\
(VAR_P (NODE) && DECL_FUNCTION_SCOPE_P (NODE))
#define DECL_DISCRIMINATOR(NODE) (LANG_DECL_U2_CHECK (NODE, 1)->discriminator)
#define DECL_DISCRIMINATOR_SET_P(NODE) \
(DECL_LANG_SPECIFIC (NODE) && DECL_LANG_SPECIFIC (NODE)->u.base.u2sel == 1)
#define DECL_PARM_INDEX(NODE) \
(LANG_DECL_PARM_CHECK (NODE)->index)
#define DECL_PARM_LEVEL(NODE) \
(LANG_DECL_PARM_CHECK (NODE)->level)
#define DECL_HAS_VTT_PARM_P(NODE) \
(LANG_DECL_FN_CHECK (NODE)->has_vtt_parm_p)
#define DECL_NEEDS_VTT_PARM_P(NODE)			\
(CLASSTYPE_VBASECLASSES (DECL_CONTEXT (NODE))		\
&& (DECL_BASE_CONSTRUCTOR_P (NODE)			\
|| DECL_BASE_DESTRUCTOR_P (NODE)))
#define DECL_CONV_FN_P(NODE) IDENTIFIER_CONV_OP_P (DECL_NAME (NODE))
#define DECL_CONV_FN_TYPE(FN) \
TREE_TYPE ((gcc_checking_assert (DECL_CONV_FN_P (FN)), DECL_NAME (FN)))
#define VAR_HAD_UNKNOWN_BOUND(NODE)			\
(DECL_LANG_SPECIFIC (VAR_DECL_CHECK (NODE))		\
? DECL_LANG_SPECIFIC (NODE)->u.base.unknown_bound_p	\
: false)
#define SET_VAR_HAD_UNKNOWN_BOUND(NODE) \
(DECL_LANG_SPECIFIC (VAR_DECL_CHECK (NODE))->u.base.unknown_bound_p = true)
#define DECL_OVERLOADED_OPERATOR_P(NODE)		\
IDENTIFIER_ANY_OP_P (DECL_NAME (NODE))
#define DECL_ASSIGNMENT_OPERATOR_P(NODE)		 \
IDENTIFIER_ASSIGN_OP_P (DECL_NAME (NODE))
#define DECL_OVERLOADED_OPERATOR_CODE_RAW(NODE)		\
(LANG_DECL_FN_CHECK (NODE)->ovl_op_code)
#define DECL_OVERLOADED_OPERATOR_IS(DECL, CODE)			\
(DECL_OVERLOADED_OPERATOR_CODE_RAW (DECL) == OVL_OP_##CODE)
#define DECL_HAS_IN_CHARGE_PARM_P(NODE) \
(LANG_DECL_FN_CHECK (NODE)->has_in_charge_parm_p)
#define DECL_IS_BUILTIN_CONSTANT_P(NODE)		\
(TREE_CODE (NODE) == FUNCTION_DECL			\
&& DECL_BUILT_IN_CLASS (NODE) == BUILT_IN_NORMAL	\
&& DECL_FUNCTION_CODE (NODE) == BUILT_IN_CONSTANT_P)
#define DECL_IN_AGGR_P(NODE) (DECL_LANG_FLAG_3 (NODE))
#define DECL_INITIALIZED_P(NODE) \
(TREE_LANG_FLAG_1 (VAR_DECL_CHECK (NODE)))
#define DECL_NONTRIVIALLY_INITIALIZED_P(NODE)	\
(TREE_LANG_FLAG_3 (VAR_DECL_CHECK (NODE)))
#define DECL_INITIALIZED_BY_CONSTANT_EXPRESSION_P(NODE) \
(TREE_LANG_FLAG_2 (VAR_DECL_CHECK (NODE)))
#define DECL_INITIALIZED_IN_CLASS_P(DECL) \
(DECL_LANG_SPECIFIC (VAR_OR_FUNCTION_DECL_CHECK (DECL)) \
->u.base.initialized_in_class)
#define DECL_ODR_USED(DECL) \
(DECL_LANG_SPECIFIC (VAR_OR_FUNCTION_DECL_CHECK (DECL)) \
->u.base.odr_used)
#define DECL_FRIEND_P(NODE) \
(DECL_LANG_SPECIFIC (TYPE_FUNCTION_OR_TEMPLATE_DECL_CHECK (NODE)) \
->u.base.friend_or_tls)
#define DECL_GNU_TLS_P(NODE)				\
(DECL_LANG_SPECIFIC (VAR_DECL_CHECK (NODE))		\
&& DECL_LANG_SPECIFIC (NODE)->u.base.friend_or_tls)
#define SET_DECL_GNU_TLS_P(NODE)				\
(retrofit_lang_decl (VAR_DECL_CHECK (NODE)),			\
DECL_LANG_SPECIFIC (NODE)->u.base.friend_or_tls = true)
#define DECL_BEFRIENDING_CLASSES(NODE) \
(LANG_DECL_FN_CHECK (NODE)->befriending_classes)
#define DECL_STATIC_FUNCTION_P(NODE) \
(LANG_DECL_FN_CHECK (NODE)->static_function)
#define DECL_NONSTATIC_MEMBER_FUNCTION_P(NODE) \
(TREE_CODE (TREE_TYPE (NODE)) == METHOD_TYPE)
#define DECL_FUNCTION_MEMBER_P(NODE) \
(DECL_NONSTATIC_MEMBER_FUNCTION_P (NODE) || DECL_STATIC_FUNCTION_P (NODE))
#define DECL_CONST_MEMFUNC_P(NODE)					 \
(DECL_NONSTATIC_MEMBER_FUNCTION_P (NODE)				 \
&& CP_TYPE_CONST_P (TREE_TYPE (TREE_VALUE				 \
(TYPE_ARG_TYPES (TREE_TYPE (NODE))))))
#define DECL_VOLATILE_MEMFUNC_P(NODE)					 \
(DECL_NONSTATIC_MEMBER_FUNCTION_P (NODE)				 \
&& CP_TYPE_VOLATILE_P (TREE_TYPE (TREE_VALUE				 \
(TYPE_ARG_TYPES (TREE_TYPE (NODE))))))
#define DECL_NONSTATIC_MEMBER_P(NODE)		\
(DECL_NONSTATIC_MEMBER_FUNCTION_P (NODE)	\
|| TREE_CODE (NODE) == FIELD_DECL)
#define DECL_MUTABLE_P(NODE) (DECL_LANG_FLAG_0 (NODE))
#define DECL_NONCONVERTING_P(NODE) \
(LANG_DECL_FN_CHECK (NODE)->nonconverting)
#define DECL_PURE_VIRTUAL_P(NODE) \
(LANG_DECL_FN_CHECK (NODE)->pure_virtual)
#define DECL_INVALID_OVERRIDER_P(NODE) \
(DECL_LANG_FLAG_4 (NODE))
#define DECL_OVERRIDE_P(NODE) (TREE_LANG_FLAG_0 (NODE))
#define DECL_THUNKS(NODE) \
(DECL_VIRTUAL_P (NODE) ? LANG_DECL_FN_CHECK (NODE)->context : NULL_TREE)
#define SET_DECL_THUNKS(NODE,THUNKS) \
(LANG_DECL_FN_CHECK (NODE)->context = (THUNKS))
#define DECL_INHERITED_CTOR(NODE) \
(DECL_DECLARES_FUNCTION_P (NODE) && DECL_CONSTRUCTOR_P (NODE) \
? LANG_DECL_FN_CHECK (NODE)->context : NULL_TREE)
#define DECL_INHERITED_CTOR_BASE(NODE)			\
(DECL_INHERITED_CTOR (NODE)				\
? DECL_CONTEXT (flag_new_inheriting_ctors		\
? strip_inheriting_ctors (NODE)	\
: DECL_INHERITED_CTOR (NODE))	\
: NULL_TREE)
#define SET_DECL_INHERITED_CTOR(NODE,INH) \
(LANG_DECL_FN_CHECK (NODE)->context = (INH))
#define DECL_THUNK_P(NODE)			\
(TREE_CODE (NODE) == FUNCTION_DECL		\
&& DECL_LANG_SPECIFIC (NODE)			\
&& LANG_DECL_FN_CHECK (NODE)->thunk_p)
#define SET_DECL_THUNK_P(NODE, THIS_ADJUSTING)			\
(LANG_DECL_FN_CHECK (NODE)->thunk_p = 1,			\
LANG_DECL_FN_CHECK (NODE)->this_thunk_p = (THIS_ADJUSTING))
#define DECL_THIS_THUNK_P(NODE)			\
(DECL_THUNK_P (NODE) && LANG_DECL_FN_CHECK (NODE)->this_thunk_p)
#define DECL_RESULT_THUNK_P(NODE)			\
(DECL_THUNK_P (NODE) && !LANG_DECL_FN_CHECK (NODE)->this_thunk_p)
#define DECL_NON_THUNK_FUNCTION_P(NODE)				\
(TREE_CODE (NODE) == FUNCTION_DECL && !DECL_THUNK_P (NODE))
#define DECL_EXTERN_C_P(NODE) \
(DECL_LANGUAGE (NODE) == lang_c)
#define DECL_EXTERN_C_FUNCTION_P(NODE) \
(DECL_NON_THUNK_FUNCTION_P (NODE) && DECL_EXTERN_C_P (NODE))
#define DECL_REPO_AVAILABLE_P(NODE) \
(DECL_LANG_SPECIFIC (NODE)->u.base.repo_available_p)
#define DECL_DECLARED_CONSTEXPR_P(DECL) \
DECL_LANG_FLAG_8 (VAR_OR_FUNCTION_DECL_CHECK (STRIP_TEMPLATE (DECL)))
#define DECL_DECLARED_CONCEPT_P(NODE) \
(DECL_LANG_SPECIFIC (NODE)->u.base.concept_p)
#define DECL_PRETTY_FUNCTION_P(NODE) \
(DECL_NAME (NODE) \
&& id_equal (DECL_NAME (NODE), "__PRETTY_FUNCTION__"))
#define CP_DECL_THREAD_LOCAL_P(NODE) \
(TREE_LANG_FLAG_0 (VAR_DECL_CHECK (NODE)))
#define DECL_CLASS_CONTEXT(NODE) \
(DECL_CLASS_SCOPE_P (NODE) ? DECL_CONTEXT (NODE) : NULL_TREE)
#define DECL_FRIEND_CONTEXT(NODE)				\
((DECL_DECLARES_FUNCTION_P (NODE)				\
&& DECL_FRIEND_P (NODE) && !DECL_FUNCTION_MEMBER_P (NODE))	\
? LANG_DECL_FN_CHECK (NODE)->context				\
: NULL_TREE)
#define SET_DECL_FRIEND_CONTEXT(NODE, CONTEXT) \
(LANG_DECL_FN_CHECK (NODE)->context = (CONTEXT))
#define CP_DECL_CONTEXT(NODE) \
(!DECL_FILE_SCOPE_P (NODE) ? DECL_CONTEXT (NODE) : global_namespace)
#define CP_TYPE_CONTEXT(NODE) \
(!TYPE_FILE_SCOPE_P (NODE) ? TYPE_CONTEXT (NODE) : global_namespace)
#define FROB_CONTEXT(NODE) \
((NODE) == global_namespace ? DECL_CONTEXT (NODE) : (NODE))
#define DECL_NAMESPACE_SCOPE_P(NODE)				\
(!DECL_TEMPLATE_PARM_P (NODE)					\
&& TREE_CODE (CP_DECL_CONTEXT (NODE)) == NAMESPACE_DECL)
#define TYPE_NAMESPACE_SCOPE_P(NODE) \
(TREE_CODE (CP_TYPE_CONTEXT (NODE)) == NAMESPACE_DECL)
#define NAMESPACE_SCOPE_P(NODE) \
((DECL_P (NODE) && DECL_NAMESPACE_SCOPE_P (NODE)) \
|| (TYPE_P (NODE) && TYPE_NAMESPACE_SCOPE_P (NODE)))
#define DECL_CLASS_SCOPE_P(NODE) \
(DECL_CONTEXT (NODE) && TYPE_P (DECL_CONTEXT (NODE)))
#define TYPE_CLASS_SCOPE_P(NODE) \
(TYPE_CONTEXT (NODE) && TYPE_P (TYPE_CONTEXT (NODE)))
#define DECL_FUNCTION_SCOPE_P(NODE) \
(DECL_CONTEXT (NODE) \
&& TREE_CODE (DECL_CONTEXT (NODE)) == FUNCTION_DECL)
#define TYPE_FUNCTION_SCOPE_P(NODE) \
(TYPE_CONTEXT (NODE) && TREE_CODE (TYPE_CONTEXT (NODE)) == FUNCTION_DECL)
#define DECL_TINFO_P(NODE) TREE_LANG_FLAG_4 (VAR_DECL_CHECK (NODE))
#define DECL_VTABLE_OR_VTT_P(NODE) TREE_LANG_FLAG_5 (VAR_DECL_CHECK (NODE))
#define FUNCTION_REF_QUALIFIED(NODE) \
TREE_LANG_FLAG_4 (FUNC_OR_METHOD_CHECK (NODE))
#define FUNCTION_RVALUE_QUALIFIED(NODE) \
TREE_LANG_FLAG_5 (FUNC_OR_METHOD_CHECK (NODE))
#define DECL_CONSTRUCTION_VTABLE_P(NODE) \
TREE_LANG_FLAG_6 (VAR_DECL_CHECK (NODE))
#define LOCAL_CLASS_P(NODE)				\
(decl_function_context (TYPE_MAIN_DECL (NODE)) != NULL_TREE)
#define SCOPE_DEPTH(NODE) \
(NAMESPACE_DECL_CHECK (NODE)->base.u.bits.address_space)
#define DECL_NAMESPACE_INLINE_P(NODE) \
TREE_LANG_FLAG_0 (NAMESPACE_DECL_CHECK (NODE))
#define DECL_NAMESPACE_USING(NODE) \
(LANG_DECL_NS_CHECK (NODE)->usings)
#define DECL_NAMESPACE_INLINEES(NODE) \
(LANG_DECL_NS_CHECK (NODE)->inlinees)
#define DECL_NAMESPACE_BINDINGS(NODE) \
(LANG_DECL_NS_CHECK (NODE)->bindings)
#define DECL_NAMESPACE_ALIAS(NODE) \
DECL_ABSTRACT_ORIGIN (NAMESPACE_DECL_CHECK (NODE))
#define ORIGINAL_NAMESPACE(NODE)  \
(DECL_NAMESPACE_ALIAS (NODE) ? DECL_NAMESPACE_ALIAS (NODE) : (NODE))
#define DECL_NAMESPACE_STD_P(NODE)			\
(TREE_CODE (NODE) == NAMESPACE_DECL			\
&& CP_DECL_CONTEXT (NODE) == global_namespace	\
&& DECL_NAME (NODE) == std_identifier)
#define ATTR_IS_DEPENDENT(NODE) TREE_LANG_FLAG_0 (TREE_LIST_CHECK (NODE))
#define ABI_TAG_IMPLICIT(NODE) TREE_LANG_FLAG_0 (TREE_LIST_CHECK (NODE))
extern tree decl_shadowed_for_var_lookup (tree);
extern void decl_shadowed_for_var_insert (tree, tree);
#define DECL_DEPENDENT_P(NODE) DECL_LANG_FLAG_0 (USING_DECL_CHECK (NODE))
#define USING_DECL_SCOPE(NODE) TREE_TYPE (USING_DECL_CHECK (NODE))
#define USING_DECL_DECLS(NODE) DECL_INITIAL (USING_DECL_CHECK (NODE))
#define USING_DECL_TYPENAME_P(NODE) DECL_LANG_FLAG_1 (USING_DECL_CHECK (NODE))
#define DECL_HAS_SHADOWED_FOR_VAR_P(NODE) \
(VAR_DECL_CHECK (NODE)->decl_with_vis.shadowed_for_var_p)
#define DECL_SHADOWED_FOR_VAR(NODE) \
(DECL_HAS_SHADOWED_FOR_VAR_P(NODE) ? decl_shadowed_for_var_lookup (NODE) : NULL)
#define SET_DECL_SHADOWED_FOR_VAR(NODE, VAL) \
(decl_shadowed_for_var_insert (NODE, VAL))
#define DECL_PENDING_INLINE_P(NODE) \
(LANG_DECL_FN_CHECK (NODE)->pending_inline_p)
#define DECL_PENDING_INLINE_INFO(NODE) \
(LANG_DECL_FN_CHECK (NODE)->u.pending_inline_info)
#define TYPE_DECL_ALIAS_P(NODE) \
DECL_LANG_FLAG_6 (TYPE_DECL_CHECK (NODE))
#define TEMPLATE_DECL_COMPLEX_ALIAS_P(NODE) \
DECL_LANG_FLAG_2 (TEMPLATE_DECL_CHECK (NODE))
#define TYPE_ALIAS_P(NODE)			\
(TYPE_P (NODE)				\
&& TYPE_NAME (NODE)				\
&& TREE_CODE (TYPE_NAME (NODE)) == TYPE_DECL	\
&& TYPE_DECL_ALIAS_P (TYPE_NAME (NODE)))
#define DECL_TEMPLATE_INFO(NODE) \
(DECL_LANG_SPECIFIC (VAR_TEMPL_TYPE_FIELD_OR_FUNCTION_DECL_CHECK (NODE)) \
->u.min.template_info)
#define DECL_CAPTURED_VARIABLE(NODE) \
(LANG_DECL_U2_CHECK (NODE, 0)->access)
#define DECL_ANON_UNION_VAR_P(NODE) \
(DECL_LANG_FLAG_4 (VAR_DECL_CHECK (NODE)))
#define CLASSTYPE_TEMPLATE_INFO(NODE) \
(TYPE_LANG_SLOT_1 (RECORD_OR_UNION_CHECK (NODE)))
#define TEMPLATE_TEMPLATE_PARM_TEMPLATE_INFO(NODE) \
(TYPE_LANG_SLOT_1 (BOUND_TEMPLATE_TEMPLATE_PARM_TYPE_CHECK (NODE)))
#define TYPE_TEMPLATE_INFO(NODE)					\
(TREE_CODE (NODE) == ENUMERAL_TYPE					\
|| TREE_CODE (NODE) == BOUND_TEMPLATE_TEMPLATE_PARM			\
|| RECORD_OR_UNION_TYPE_P (NODE)					\
? TYPE_LANG_SLOT_1 (NODE) : NULL_TREE)
#define TYPE_ALIAS_TEMPLATE_INFO(NODE)					\
(DECL_LANG_SPECIFIC (TYPE_NAME (NODE))				\
? DECL_TEMPLATE_INFO (TYPE_NAME (NODE))				\
: NULL_TREE)
#define TYPE_TEMPLATE_INFO_MAYBE_ALIAS(NODE)				\
(TYPE_ALIAS_P (NODE)							\
? TYPE_ALIAS_TEMPLATE_INFO (NODE)					\
: TYPE_TEMPLATE_INFO (NODE))
#define SET_TYPE_TEMPLATE_INFO(NODE, VAL)				\
(TREE_CODE (NODE) == ENUMERAL_TYPE					\
|| (CLASS_TYPE_P (NODE) && !TYPE_ALIAS_P (NODE))			\
? (TYPE_LANG_SLOT_1 (NODE) = (VAL))				\
: (DECL_TEMPLATE_INFO (TYPE_NAME (NODE)) = (VAL)))
#define TI_TEMPLATE(NODE) TREE_TYPE (TEMPLATE_INFO_CHECK (NODE))
#define TI_ARGS(NODE) TREE_CHAIN (TEMPLATE_INFO_CHECK (NODE))
#define TI_PENDING_TEMPLATE_FLAG(NODE) TREE_LANG_FLAG_1 (NODE)
#define NON_DEFAULT_TEMPLATE_ARGS_COUNT(NODE) TREE_CHAIN (TREE_VEC_CHECK (NODE))
#define SET_NON_DEFAULT_TEMPLATE_ARGS_COUNT(NODE, INT_VALUE) \
NON_DEFAULT_TEMPLATE_ARGS_COUNT(NODE) = build_int_cst (NULL_TREE, INT_VALUE)
#if CHECKING_P
#define GET_NON_DEFAULT_TEMPLATE_ARGS_COUNT(NODE) \
int_cst_value (NON_DEFAULT_TEMPLATE_ARGS_COUNT (NODE))
#else
#define GET_NON_DEFAULT_TEMPLATE_ARGS_COUNT(NODE) \
NON_DEFAULT_TEMPLATE_ARGS_COUNT (NODE) \
? int_cst_value (NON_DEFAULT_TEMPLATE_ARGS_COUNT (NODE)) \
: TREE_VEC_LENGTH (INNERMOST_TEMPLATE_ARGS (NODE))
#endif
#define TI_TYPEDEFS_NEEDING_ACCESS_CHECKING(NODE) \
((struct tree_template_info*)TEMPLATE_INFO_CHECK \
(NODE))->typedefs_needing_access_checking
#define TMPL_ARGS_HAVE_MULTIPLE_LEVELS(NODE)		     \
(NODE && TREE_VEC_LENGTH (NODE) && TREE_VEC_ELT (NODE, 0)  \
&& TREE_CODE (TREE_VEC_ELT (NODE, 0)) == TREE_VEC)
#define TMPL_ARGS_DEPTH(NODE)					\
(TMPL_ARGS_HAVE_MULTIPLE_LEVELS (NODE) ? TREE_VEC_LENGTH (NODE) : 1)
#define TMPL_ARGS_LEVEL(ARGS, LEVEL)		\
(TMPL_ARGS_HAVE_MULTIPLE_LEVELS (ARGS)	\
? TREE_VEC_ELT (ARGS, (LEVEL) - 1) : (ARGS))
#define SET_TMPL_ARGS_LEVEL(ARGS, LEVEL, VAL)	\
(TREE_VEC_ELT (ARGS, (LEVEL) - 1) = (VAL))
#define TMPL_ARG(ARGS, LEVEL, IDX)				\
(TREE_VEC_ELT (TMPL_ARGS_LEVEL (ARGS, LEVEL), IDX))
#define NUM_TMPL_ARGS(NODE)				\
(TREE_VEC_LENGTH (NODE))
#define INNERMOST_TEMPLATE_ARGS(NODE) \
(get_innermost_template_args ((NODE), 1))
#define TMPL_PARMS_DEPTH(NODE) \
((HOST_WIDE_INT) TREE_INT_CST_LOW (TREE_PURPOSE (NODE)))
#define DECL_TI_TEMPLATE(NODE)      TI_TEMPLATE (DECL_TEMPLATE_INFO (NODE))
#define DECL_TI_ARGS(NODE)	    TI_ARGS (DECL_TEMPLATE_INFO (NODE))
#define CLASSTYPE_TI_TEMPLATE(NODE) TI_TEMPLATE (CLASSTYPE_TEMPLATE_INFO (NODE))
#define CLASSTYPE_TI_ARGS(NODE)     TI_ARGS (CLASSTYPE_TEMPLATE_INFO (NODE))
#define CLASSTYPE_PRIMARY_TEMPLATE_TYPE(TYPE)				\
((CLASSTYPE_USE_TEMPLATE ((TYPE))					\
&& !CLASSTYPE_TEMPLATE_SPECIALIZATION ((TYPE)))			\
? TREE_TYPE (DECL_TEMPLATE_RESULT (DECL_PRIMARY_TEMPLATE		\
(CLASSTYPE_TI_TEMPLATE ((TYPE))))) \
: (TYPE))
#define TYPE_TI_TEMPLATE(NODE)			\
(TI_TEMPLATE (TYPE_TEMPLATE_INFO (NODE)))
#define TYPE_TI_ARGS(NODE)			\
(TI_ARGS (TYPE_TEMPLATE_INFO (NODE)))
#define INNERMOST_TEMPLATE_PARMS(NODE)  TREE_VALUE (NODE)
#define DECL_MEMBER_TEMPLATE_P(NODE) \
(DECL_LANG_FLAG_1 (TEMPLATE_DECL_CHECK (NODE)))
#define TEMPLATE_PARMS_FOR_INLINE(NODE) TREE_LANG_FLAG_1 (NODE)
#define DECL_PACK_P(NODE) \
(DECL_P (NODE) && PACK_EXPANSION_P (TREE_TYPE (NODE)))
#define PACK_EXPANSION_P(NODE)                 \
(TREE_CODE (NODE) == TYPE_PACK_EXPANSION     \
|| TREE_CODE (NODE) == EXPR_PACK_EXPANSION)
#define PACK_EXPANSION_PATTERN(NODE)                            \
(TREE_CODE (NODE) == TYPE_PACK_EXPANSION ? TREE_TYPE (NODE)    \
: TREE_OPERAND (NODE, 0))
#define SET_PACK_EXPANSION_PATTERN(NODE,VALUE)  \
if (TREE_CODE (NODE) == TYPE_PACK_EXPANSION)  \
TREE_TYPE (NODE) = VALUE;                   \
else                                          \
TREE_OPERAND (NODE, 0) = VALUE
#define PACK_EXPANSION_PARAMETER_PACKS(NODE)		\
*(TREE_CODE (NODE) == EXPR_PACK_EXPANSION		\
? &TREE_OPERAND (NODE, 1)				\
: &TYPE_MIN_VALUE_RAW (TYPE_PACK_EXPANSION_CHECK (NODE)))
#define PACK_EXPANSION_EXTRA_ARGS(NODE)		\
*(TREE_CODE (NODE) == TYPE_PACK_EXPANSION	\
? &TYPE_MAX_VALUE_RAW (NODE)			\
: &TREE_OPERAND ((NODE), 2))
#define PACK_EXPANSION_LOCAL_P(NODE) TREE_LANG_FLAG_0 (NODE)
#define PACK_EXPANSION_SIZEOF_P(NODE) TREE_LANG_FLAG_1 (NODE)
#define WILDCARD_PACK_P(NODE) TREE_LANG_FLAG_0 (NODE)
#define ARGUMENT_PACK_P(NODE)                          \
(TREE_CODE (NODE) == TYPE_ARGUMENT_PACK              \
|| TREE_CODE (NODE) == NONTYPE_ARGUMENT_PACK)
#define ARGUMENT_PACK_ARGS(NODE)                               \
(TREE_CODE (NODE) == TYPE_ARGUMENT_PACK? TREE_TYPE (NODE)    \
: TREE_OPERAND (NODE, 0))
#define SET_ARGUMENT_PACK_ARGS(NODE,VALUE)     \
if (TREE_CODE (NODE) == TYPE_ARGUMENT_PACK)  \
TREE_TYPE (NODE) = VALUE;                           \
else                                                  \
TREE_OPERAND (NODE, 0) = VALUE
#define ARGUMENT_PACK_INCOMPLETE_P(NODE)        \
TREE_ADDRESSABLE (ARGUMENT_PACK_ARGS (NODE))
#define ARGUMENT_PACK_EXPLICIT_ARGS(NODE)       \
TREE_TYPE (ARGUMENT_PACK_ARGS (NODE))
#define ARGUMENT_PACK_SELECT_FROM_PACK(NODE)				\
(((struct tree_argument_pack_select *)ARGUMENT_PACK_SELECT_CHECK (NODE))->argument_pack)
#define ARGUMENT_PACK_SELECT_INDEX(NODE)				\
(((struct tree_argument_pack_select *)ARGUMENT_PACK_SELECT_CHECK (NODE))->index)
#define FOLD_EXPR_CHECK(NODE)						\
TREE_CHECK4 (NODE, UNARY_LEFT_FOLD_EXPR, UNARY_RIGHT_FOLD_EXPR,	\
BINARY_LEFT_FOLD_EXPR, BINARY_RIGHT_FOLD_EXPR)
#define BINARY_FOLD_EXPR_CHECK(NODE) \
TREE_CHECK2 (NODE, BINARY_LEFT_FOLD_EXPR, BINARY_RIGHT_FOLD_EXPR)
#define FOLD_EXPR_P(NODE)				\
(TREE_CODE (NODE) == UNARY_LEFT_FOLD_EXPR		\
|| TREE_CODE (NODE) == UNARY_RIGHT_FOLD_EXPR		\
|| TREE_CODE (NODE) == BINARY_LEFT_FOLD_EXPR		\
|| TREE_CODE (NODE) == BINARY_RIGHT_FOLD_EXPR)
#define FOLD_EXPR_MODIFY_P(NODE) \
TREE_LANG_FLAG_0 (FOLD_EXPR_CHECK (NODE))
#define FOLD_EXPR_OP(NODE) \
TREE_OPERAND (FOLD_EXPR_CHECK (NODE), 0)
#define FOLD_EXPR_PACK(NODE) \
TREE_OPERAND (FOLD_EXPR_CHECK (NODE), 1)
#define FOLD_EXPR_INIT(NODE) \
TREE_OPERAND (BINARY_FOLD_EXPR_CHECK (NODE), 2)
#define DECL_SAVED_FUNCTION_DATA(NODE)			\
(LANG_DECL_FN_CHECK (FUNCTION_DECL_CHECK (NODE))	\
->u.saved_language_function)
#define REFERENCE_REF_P(NODE)				\
(INDIRECT_REF_P (NODE)				\
&& TREE_TYPE (TREE_OPERAND (NODE, 0))		\
&& (TREE_CODE (TREE_TYPE (TREE_OPERAND ((NODE), 0)))	\
== REFERENCE_TYPE))
#define REFERENCE_VLA_OK(NODE) \
(TYPE_LANG_FLAG_5 (REFERENCE_TYPE_CHECK (NODE)))
#define NEW_EXPR_USE_GLOBAL(NODE) \
TREE_LANG_FLAG_0 (NEW_EXPR_CHECK (NODE))
#define DELETE_EXPR_USE_GLOBAL(NODE) \
TREE_LANG_FLAG_0 (DELETE_EXPR_CHECK (NODE))
#define DELETE_EXPR_USE_VEC(NODE) \
TREE_LANG_FLAG_1 (DELETE_EXPR_CHECK (NODE))
#define CALL_OR_AGGR_INIT_CHECK(NODE) \
TREE_CHECK2 ((NODE), CALL_EXPR, AGGR_INIT_EXPR)
#define COMPOUND_EXPR_OVERLOADED(NODE) \
TREE_LANG_FLAG_0 (COMPOUND_EXPR_CHECK (NODE))
#define KOENIG_LOOKUP_P(NODE) TREE_LANG_FLAG_0 (CALL_EXPR_CHECK (NODE))
#define CALL_EXPR_ORDERED_ARGS(NODE) \
TREE_LANG_FLAG_3 (CALL_OR_AGGR_INIT_CHECK (NODE))
#define CALL_EXPR_REVERSE_ARGS(NODE) \
TREE_LANG_FLAG_5 (CALL_OR_AGGR_INIT_CHECK (NODE))
#define CALL_EXPR_OPERATOR_SYNTAX(NODE) \
TREE_LANG_FLAG_6 (CALL_OR_AGGR_INIT_CHECK (NODE))
#define PAREN_STRING_LITERAL_P(NODE) \
TREE_LANG_FLAG_0 (STRING_CST_CHECK (NODE))
#define REF_PARENTHESIZED_P(NODE) \
TREE_LANG_FLAG_2 (TREE_CHECK3 ((NODE), COMPONENT_REF, INDIRECT_REF, SCOPE_REF))
#define AGGR_INIT_VIA_CTOR_P(NODE) \
TREE_LANG_FLAG_0 (AGGR_INIT_EXPR_CHECK (NODE))
#define AGGR_INIT_ZERO_FIRST(NODE) \
TREE_LANG_FLAG_2 (AGGR_INIT_EXPR_CHECK (NODE))
#define AGGR_INIT_FROM_THUNK_P(NODE) \
(AGGR_INIT_EXPR_CHECK (NODE)->base.protected_flag)
#define AGGR_INIT_EXPR_FN(NODE) TREE_OPERAND (AGGR_INIT_EXPR_CHECK (NODE), 1)
#define AGGR_INIT_EXPR_SLOT(NODE) \
TREE_OPERAND (AGGR_INIT_EXPR_CHECK (NODE), 2)
#define AGGR_INIT_EXPR_ARG(NODE, I) \
TREE_OPERAND (AGGR_INIT_EXPR_CHECK (NODE), (I) + 3)
#define aggr_init_expr_nargs(NODE) (VL_EXP_OPERAND_LENGTH(NODE) - 3)
#define AGGR_INIT_EXPR_ARGP(NODE) \
(&(TREE_OPERAND (AGGR_INIT_EXPR_CHECK (NODE), 0)) + 3)
struct aggr_init_expr_arg_iterator {
tree t;	
int n;	
int i;	
};
inline void
init_aggr_init_expr_arg_iterator (tree exp,
aggr_init_expr_arg_iterator *iter)
{
iter->t = exp;
iter->n = aggr_init_expr_nargs (exp);
iter->i = 0;
}
inline tree
next_aggr_init_expr_arg (aggr_init_expr_arg_iterator *iter)
{
tree result;
if (iter->i >= iter->n)
return NULL_TREE;
result = AGGR_INIT_EXPR_ARG (iter->t, iter->i);
iter->i++;
return result;
}
inline tree
first_aggr_init_expr_arg (tree exp, aggr_init_expr_arg_iterator *iter)
{
init_aggr_init_expr_arg_iterator (exp, iter);
return next_aggr_init_expr_arg (iter);
}
inline bool
more_aggr_init_expr_args_p (const aggr_init_expr_arg_iterator *iter)
{
return (iter->i < iter->n);
}
#define FOR_EACH_AGGR_INIT_EXPR_ARG(arg, iter, call)			\
for ((arg) = first_aggr_init_expr_arg ((call), &(iter)); (arg);	\
(arg) = next_aggr_init_expr_arg (&(iter)))
#define VEC_INIT_EXPR_SLOT(NODE) TREE_OPERAND (VEC_INIT_EXPR_CHECK (NODE), 0)
#define VEC_INIT_EXPR_INIT(NODE) TREE_OPERAND (VEC_INIT_EXPR_CHECK (NODE), 1)
#define VEC_INIT_EXPR_IS_CONSTEXPR(NODE) \
TREE_LANG_FLAG_0 (VEC_INIT_EXPR_CHECK (NODE))
#define VEC_INIT_EXPR_VALUE_INIT(NODE) \
TREE_LANG_FLAG_1 (VEC_INIT_EXPR_CHECK (NODE))
#define MUST_NOT_THROW_COND(NODE) \
TREE_OPERAND (MUST_NOT_THROW_EXPR_CHECK (NODE), 1)
#define CLASSTYPE_IS_TEMPLATE(NODE)  \
(CLASSTYPE_TEMPLATE_INFO (NODE)    \
&& !CLASSTYPE_USE_TEMPLATE (NODE) \
&& PRIMARY_TEMPLATE_P (CLASSTYPE_TI_TEMPLATE (NODE)))
#define TYPENAME_TYPE_FULLNAME(NODE) \
(TYPE_VALUES_RAW (TYPENAME_TYPE_CHECK (NODE)))
#define TYPENAME_IS_ENUM_P(NODE) \
(TREE_LANG_FLAG_0 (TYPENAME_TYPE_CHECK (NODE)))
#define TYPENAME_IS_CLASS_P(NODE) \
(TREE_LANG_FLAG_1 (TYPENAME_TYPE_CHECK (NODE)))
#define TYPENAME_IS_RESOLVING_P(NODE) \
(TREE_LANG_FLAG_2 (TYPENAME_TYPE_CHECK (NODE)))
#define TYPE_POLYMORPHIC_P(NODE) (TREE_LANG_FLAG_2 (NODE))
#define TYPE_CONTAINS_VPTR_P(NODE)		\
(TYPE_POLYMORPHIC_P (NODE) || CLASSTYPE_VBASECLASSES (NODE))
#define DECL_DEAD_FOR_LOCAL(NODE) DECL_LANG_FLAG_7 (VAR_DECL_CHECK (NODE))
#define DECL_ERROR_REPORTED(NODE) DECL_LANG_FLAG_0 (VAR_DECL_CHECK (NODE))
#define DECL_LOCAL_FUNCTION_P(NODE) \
DECL_LANG_FLAG_0 (FUNCTION_DECL_CHECK (NODE))
#define LABEL_DECL_BREAK(NODE) \
DECL_LANG_FLAG_0 (LABEL_DECL_CHECK (NODE))
#define LABEL_DECL_CONTINUE(NODE) \
DECL_LANG_FLAG_1 (LABEL_DECL_CHECK (NODE))
#define LABEL_DECL_CDTOR(NODE) \
DECL_LANG_FLAG_2 (LABEL_DECL_CHECK (NODE))
#define FNDECL_USED_AUTO(NODE) \
TREE_LANG_FLAG_2 (FUNCTION_DECL_CHECK (NODE))
#define DECL_ANTICIPATED(NODE) \
(DECL_LANG_SPECIFIC (TYPE_FUNCTION_OR_TEMPLATE_DECL_CHECK (NODE)) \
->u.base.anticipated_p)
#define DECL_HIDDEN_P(NODE) \
(DECL_LANG_SPECIFIC (NODE) && TYPE_FUNCTION_OR_TEMPLATE_DECL_P (NODE) \
&& DECL_ANTICIPATED (NODE))
#define TYPE_HIDDEN_P(NODE) \
(DECL_LANG_SPECIFIC (TYPE_NAME (NODE)) \
&& DECL_ANTICIPATED (TYPE_NAME (NODE)))
#define DECL_OMP_PRIVATIZED_MEMBER(NODE) \
(DECL_LANG_SPECIFIC (VAR_DECL_CHECK (NODE))->u.base.anticipated_p)
#define DECL_HIDDEN_FRIEND_P(NODE) \
(LANG_DECL_FN_CHECK (DECL_COMMON_CHECK (NODE))->hidden_friend_p)
#define DECL_OMP_DECLARE_REDUCTION_P(NODE) \
(LANG_DECL_FN_CHECK (DECL_COMMON_CHECK (NODE))->omp_declare_reduction_p)
#define CP_DECL_THREADPRIVATE_P(DECL) \
(DECL_LANG_SPECIFIC (VAR_DECL_CHECK (DECL))->u.base.threadprivate_or_deleted_p)
#define DECL_VAR_DECLARED_INLINE_P(NODE) \
(DECL_LANG_SPECIFIC (VAR_DECL_CHECK (NODE))			\
? DECL_LANG_SPECIFIC (NODE)->u.base.var_declared_inline_p	\
: false)
#define SET_DECL_VAR_DECLARED_INLINE_P(NODE) \
(DECL_LANG_SPECIFIC (VAR_DECL_CHECK (NODE))->u.base.var_declared_inline_p \
= true)
#define DECL_DEPENDENT_INIT_P(NODE)				\
(DECL_LANG_SPECIFIC (VAR_DECL_CHECK (NODE))			\
&& DECL_LANG_SPECIFIC (NODE)->u.base.dependent_init_p)
#define SET_DECL_DEPENDENT_INIT_P(NODE, X) \
(DECL_LANG_SPECIFIC (VAR_DECL_CHECK (NODE))->u.base.dependent_init_p = (X))
#define DECL_DECOMPOSITION_P(NODE) \
(VAR_P (NODE) && DECL_LANG_SPECIFIC (NODE)			\
? DECL_LANG_SPECIFIC (NODE)->u.base.selector == lds_decomp		\
: false)
#define DECL_DECOMP_BASE(NODE) \
(LANG_DECL_DECOMP_CHECK (NODE)->base)
#define DECL_INLINE_VAR_P(NODE) \
(DECL_VAR_DECLARED_INLINE_P (NODE)				\
|| (cxx_dialect >= cxx17					\
&& DECL_DECLARED_CONSTEXPR_P (NODE)			\
&& DECL_CLASS_SCOPE_P (NODE)))
#define DECL_DELETED_FN(DECL) \
(LANG_DECL_FN_CHECK (DECL)->min.base.threadprivate_or_deleted_p)
#define DECL_DEFAULTED_FN(DECL) \
(LANG_DECL_FN_CHECK (DECL)->defaulted_p)
#define DECL_DEFAULTED_IN_CLASS_P(DECL)					\
(DECL_DEFAULTED_FN (DECL) && DECL_INITIALIZED_IN_CLASS_P (DECL))
#define DECL_DEFAULTED_OUTSIDE_CLASS_P(DECL)				\
(DECL_DEFAULTED_FN (DECL)						\
&& !(DECL_ARTIFICIAL (DECL) || DECL_INITIALIZED_IN_CLASS_P (DECL)))
#define C_TYPEDEF_EXPLICITLY_SIGNED(EXP) DECL_LANG_FLAG_1 (EXP)
#define DECL_EXTERNAL_LINKAGE_P(DECL) \
(decl_linkage (DECL) == lk_external)
#define INTEGRAL_CODE_P(CODE)	\
((CODE) == ENUMERAL_TYPE	\
|| (CODE) == BOOLEAN_TYPE	\
|| (CODE) == INTEGER_TYPE)
#define CP_INTEGRAL_TYPE_P(TYPE)		\
(TREE_CODE (TYPE) == BOOLEAN_TYPE		\
|| TREE_CODE (TYPE) == INTEGER_TYPE)
#define INTEGRAL_OR_ENUMERATION_TYPE_P(TYPE) \
(TREE_CODE (TYPE) == ENUMERAL_TYPE || CP_INTEGRAL_TYPE_P (TYPE))
#define INTEGRAL_OR_UNSCOPED_ENUMERATION_TYPE_P(TYPE) \
(UNSCOPED_ENUM_P (TYPE) || CP_INTEGRAL_TYPE_P (TYPE))
#define CLASSTYPE_LITERAL_P(TYPE)              \
(LANG_TYPE_CLASS_CHECK (TYPE)->is_literal)
#define ARITHMETIC_TYPE_P(TYPE) \
(CP_INTEGRAL_TYPE_P (TYPE) \
|| TREE_CODE (TYPE) == REAL_TYPE \
|| TREE_CODE (TYPE) == COMPLEX_TYPE)
#define NULLPTR_TYPE_P(TYPE) (TREE_CODE (TYPE) == NULLPTR_TYPE)
#define SCALAR_TYPE_P(TYPE)			\
(TYPE_PTRDATAMEM_P (TYPE)			\
|| TREE_CODE (TYPE) == ENUMERAL_TYPE		\
|| ARITHMETIC_TYPE_P (TYPE)			\
|| TYPE_PTR_P (TYPE)				\
|| TYPE_PTRMEMFUNC_P (TYPE)                  \
|| NULLPTR_TYPE_P (TYPE))
#define SCOPED_ENUM_P(TYPE)                                             \
(TREE_CODE (TYPE) == ENUMERAL_TYPE && ENUM_IS_SCOPED (TYPE))
#define UNSCOPED_ENUM_P(TYPE)                                           \
(TREE_CODE (TYPE) == ENUMERAL_TYPE && !ENUM_IS_SCOPED (TYPE))
#define SET_SCOPED_ENUM_P(TYPE, VAL)                    \
(ENUM_IS_SCOPED (TYPE) = (VAL))
#define SET_OPAQUE_ENUM_P(TYPE, VAL)                    \
(ENUM_IS_OPAQUE (TYPE) = (VAL))
#define OPAQUE_ENUM_P(TYPE)				\
(TREE_CODE (TYPE) == ENUMERAL_TYPE && ENUM_IS_OPAQUE (TYPE))
#define ENUM_FIXED_UNDERLYING_TYPE_P(NODE) (TYPE_LANG_FLAG_5 (NODE))
#define ENUM_UNDERLYING_TYPE(TYPE) \
TREE_TYPE (ENUMERAL_TYPE_CHECK (TYPE))
#define CP_AGGREGATE_TYPE_P(TYPE)				\
(TREE_CODE (TYPE) == VECTOR_TYPE				\
||TREE_CODE (TYPE) == ARRAY_TYPE				\
|| (CLASS_TYPE_P (TYPE) && !CLASSTYPE_NON_AGGREGATE (TYPE)))
#define TYPE_HAS_USER_CONSTRUCTOR(NODE) (TYPE_LANG_FLAG_1 (NODE))
#define TYPE_HAS_LATE_RETURN_TYPE(NODE) \
(TYPE_LANG_FLAG_2 (FUNC_OR_METHOD_CHECK (NODE)))
#define TREE_HAS_CONSTRUCTOR(NODE) (TREE_LANG_FLAG_4 (NODE))
#define BRACE_ENCLOSED_INITIALIZER_P(NODE) \
(TREE_CODE (NODE) == CONSTRUCTOR && TREE_TYPE (NODE) == init_list_type_node)
#define COMPOUND_LITERAL_P(NODE) \
(TREE_CODE (NODE) == CONSTRUCTOR && TREE_HAS_CONSTRUCTOR (NODE))
#define EMPTY_CONSTRUCTOR_P(NODE) (TREE_CODE (NODE) == CONSTRUCTOR \
&& vec_safe_is_empty(CONSTRUCTOR_ELTS(NODE))\
&& !TREE_HAS_CONSTRUCTOR (NODE))
#define CONSTRUCTOR_IS_DIRECT_INIT(NODE) (TREE_LANG_FLAG_0 (CONSTRUCTOR_CHECK (NODE)))
#define CONSTRUCTOR_NO_IMPLICIT_ZERO(NODE) \
(TREE_LANG_FLAG_1 (CONSTRUCTOR_CHECK (NODE)))
#define CONSTRUCTOR_MUTABLE_POISON(NODE) \
(TREE_LANG_FLAG_2 (CONSTRUCTOR_CHECK (NODE)))
#define CONSTRUCTOR_C99_COMPOUND_LITERAL(NODE) \
(TREE_LANG_FLAG_3 (CONSTRUCTOR_CHECK (NODE)))
#define CONSTRUCTOR_PLACEHOLDER_BOUNDARY(NODE) \
(TREE_LANG_FLAG_5 (CONSTRUCTOR_CHECK (NODE)))
#define DIRECT_LIST_INIT_P(NODE) \
(BRACE_ENCLOSED_INITIALIZER_P (NODE) && CONSTRUCTOR_IS_DIRECT_INIT (NODE))
#define IMPLICIT_CONV_EXPR_DIRECT_INIT(NODE) \
(TREE_LANG_FLAG_0 (IMPLICIT_CONV_EXPR_CHECK (NODE)))
#define IMPLICIT_CONV_EXPR_NONTYPE_ARG(NODE) \
(TREE_LANG_FLAG_1 (IMPLICIT_CONV_EXPR_CHECK (NODE)))
#define CLASSTYPE_NON_AGGREGATE(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->non_aggregate)
#define TYPE_NON_AGGREGATE_CLASS(NODE) \
(CLASS_TYPE_P (NODE) && CLASSTYPE_NON_AGGREGATE (NODE))
#define TYPE_HAS_COMPLEX_COPY_ASSIGN(NODE) (LANG_TYPE_CLASS_CHECK (NODE)->has_complex_copy_assign)
#define TYPE_HAS_COMPLEX_COPY_CTOR(NODE) (LANG_TYPE_CLASS_CHECK (NODE)->has_complex_copy_ctor)
#define TYPE_HAS_COMPLEX_MOVE_ASSIGN(NODE) (LANG_TYPE_CLASS_CHECK (NODE)->has_complex_move_assign)
#define TYPE_HAS_COMPLEX_MOVE_CTOR(NODE) (LANG_TYPE_CLASS_CHECK (NODE)->has_complex_move_ctor)
#define TYPE_HAS_COMPLEX_DFLT(NODE) (LANG_TYPE_CLASS_CHECK (NODE)->has_complex_dflt)
#define TYPE_HAS_TRIVIAL_DESTRUCTOR(NODE) \
(!TYPE_HAS_NONTRIVIAL_DESTRUCTOR (NODE))
#define TYPE_HAS_NONTRIVIAL_DESTRUCTOR(NODE) \
(TYPE_LANG_FLAG_4 (NODE))
#define TYPE_HAS_TRIVIAL_DFLT(NODE) \
(TYPE_HAS_DEFAULT_CONSTRUCTOR (NODE) && ! TYPE_HAS_COMPLEX_DFLT (NODE))
#define TYPE_HAS_TRIVIAL_COPY_CTOR(NODE) \
(TYPE_HAS_COPY_CTOR (NODE) && ! TYPE_HAS_COMPLEX_COPY_CTOR (NODE))
#define TYPE_HAS_TRIVIAL_COPY_ASSIGN(NODE) \
(TYPE_HAS_COPY_ASSIGN (NODE) && ! TYPE_HAS_COMPLEX_COPY_ASSIGN (NODE))
#define TYPE_PTRDATAMEM_P(NODE)			\
(TREE_CODE (NODE) == OFFSET_TYPE)
#define TYPE_PTR_P(NODE)			\
(TREE_CODE (NODE) == POINTER_TYPE)
#define TYPE_OBJ_P(NODE)			\
(TREE_CODE (NODE) != REFERENCE_TYPE		\
&& !VOID_TYPE_P (NODE)  		        \
&& TREE_CODE (NODE) != FUNCTION_TYPE		\
&& TREE_CODE (NODE) != METHOD_TYPE)
#define TYPE_PTROB_P(NODE)					\
(TYPE_PTR_P (NODE) && TYPE_OBJ_P (TREE_TYPE (NODE)))
#define TYPE_REF_OBJ_P(NODE)					\
(TREE_CODE (NODE) == REFERENCE_TYPE && TYPE_OBJ_P (TREE_TYPE (NODE)))
#define TYPE_PTROBV_P(NODE)					\
(TYPE_PTR_P (NODE)						\
&& !(TREE_CODE (TREE_TYPE (NODE)) == FUNCTION_TYPE		\
|| TREE_CODE (TREE_TYPE (NODE)) == METHOD_TYPE))
#define TYPE_PTRFN_P(NODE)				\
(TYPE_PTR_P (NODE)			                \
&& TREE_CODE (TREE_TYPE (NODE)) == FUNCTION_TYPE)
#define TYPE_REFFN_P(NODE)				\
(TREE_CODE (NODE) == REFERENCE_TYPE			\
&& TREE_CODE (TREE_TYPE (NODE)) == FUNCTION_TYPE)
#define TYPE_PTRMEMFUNC_P(NODE)		\
(TREE_CODE (NODE) == RECORD_TYPE	\
&& TYPE_PTRMEMFUNC_FLAG (NODE))
#define TYPE_PTRMEMFUNC_FLAG(NODE) \
(TYPE_LANG_FLAG_2 (RECORD_TYPE_CHECK (NODE)))
#define TYPE_PTRMEM_P(NODE) \
(TYPE_PTRDATAMEM_P (NODE) || TYPE_PTRMEMFUNC_P (NODE))
#define TYPE_PTR_OR_PTRMEM_P(NODE) \
(TYPE_PTR_P (NODE) || TYPE_PTRMEM_P (NODE))
#define PTRMEM_OK_P(NODE) \
TREE_LANG_FLAG_0 (TREE_CHECK3 ((NODE), ADDR_EXPR, OFFSET_REF, SCOPE_REF))
#define TYPE_PTRMEMFUNC_FN_TYPE(NODE) \
(cp_build_qualified_type (TREE_TYPE (TYPE_FIELDS (NODE)),\
cp_type_quals (NODE)))
#define TYPE_PTRMEMFUNC_FN_TYPE_RAW(NODE) \
(TREE_TYPE (TYPE_FIELDS (NODE)))
#define TYPE_PTRMEMFUNC_OBJECT_TYPE(NODE) \
TYPE_METHOD_BASETYPE (TREE_TYPE (TYPE_PTRMEMFUNC_FN_TYPE (NODE)))
#define TYPE_PTRMEMFUNC_TYPE(NODE) \
TYPE_LANG_SLOT_1 (NODE)
#define TYPE_PTRMEM_CLASS_TYPE(NODE)			\
(TYPE_PTRDATAMEM_P (NODE)					\
? TYPE_OFFSET_BASETYPE (NODE)		\
: TYPE_PTRMEMFUNC_OBJECT_TYPE (NODE))
#define TYPE_PTRMEM_POINTED_TO_TYPE(NODE)		\
(TYPE_PTRDATAMEM_P (NODE)				\
? TREE_TYPE (NODE)					\
: TREE_TYPE (TYPE_PTRMEMFUNC_FN_TYPE (NODE)))
#define PTRMEM_CST_CLASS(NODE) \
TYPE_PTRMEM_CLASS_TYPE (TREE_TYPE (PTRMEM_CST_CHECK (NODE)))
#define PTRMEM_CST_MEMBER(NODE) \
(((ptrmem_cst_t)PTRMEM_CST_CHECK (NODE))->member)
#define TYPEOF_TYPE_EXPR(NODE) (TYPE_VALUES_RAW (TYPEOF_TYPE_CHECK (NODE)))
#define UNDERLYING_TYPE_TYPE(NODE) \
(TYPE_VALUES_RAW (UNDERLYING_TYPE_CHECK (NODE)))
#define BASES_TYPE(NODE) \
(TYPE_VALUES_RAW (BASES_CHECK (NODE)))
#define BASES_DIRECT(NODE) \
TREE_LANG_FLAG_0 (BASES_CHECK (NODE))
#define DECLTYPE_TYPE_EXPR(NODE) (TYPE_VALUES_RAW (DECLTYPE_TYPE_CHECK (NODE)))
#define DECLTYPE_TYPE_ID_EXPR_OR_MEMBER_ACCESS_P(NODE) \
(DECLTYPE_TYPE_CHECK (NODE))->type_common.string_flag
#define DECLTYPE_FOR_LAMBDA_CAPTURE(NODE) \
TREE_LANG_FLAG_0 (DECLTYPE_TYPE_CHECK (NODE))
#define DECLTYPE_FOR_INIT_CAPTURE(NODE) \
TREE_LANG_FLAG_1 (DECLTYPE_TYPE_CHECK (NODE))
#define DECLTYPE_FOR_LAMBDA_PROXY(NODE) \
TREE_LANG_FLAG_2 (DECLTYPE_TYPE_CHECK (NODE))
#define DECLTYPE_FOR_REF_CAPTURE(NODE) \
TREE_LANG_FLAG_3 (DECLTYPE_TYPE_CHECK (NODE))
#define DECL_THIS_EXTERN(NODE) \
DECL_LANG_FLAG_2 (VAR_FUNCTION_OR_PARM_DECL_CHECK (NODE))
#define DECL_THIS_STATIC(NODE) \
DECL_LANG_FLAG_6 (VAR_FUNCTION_OR_PARM_DECL_CHECK (NODE))
#define DECL_VLA_CAPTURE_P(NODE) \
DECL_LANG_FLAG_1 (FIELD_DECL_CHECK (NODE))
#define DECL_ARRAY_PARAMETER_P(NODE) \
DECL_LANG_FLAG_1 (PARM_DECL_CHECK (NODE))
#define DECL_INSTANTIATING_NSDMI_P(NODE) \
DECL_LANG_FLAG_2 (FIELD_DECL_CHECK (NODE))
#define DECL_FIELD_IS_BASE(NODE) \
DECL_LANG_FLAG_6 (FIELD_DECL_CHECK (NODE))
#define DECL_NORMAL_CAPTURE_P(NODE) \
DECL_LANG_FLAG_7 (FIELD_DECL_CHECK (NODE))
#define ANON_AGGR_TYPE_P(NODE)				\
(CLASS_TYPE_P (NODE) && LANG_TYPE_CLASS_CHECK (NODE)->anon_aggr)
#define SET_ANON_AGGR_TYPE_P(NODE)			\
(LANG_TYPE_CLASS_CHECK (NODE)->anon_aggr = 1)
#define ANON_UNION_TYPE_P(NODE) \
(TREE_CODE (NODE) == UNION_TYPE && ANON_AGGR_TYPE_P (NODE))
#define TYPE_WAS_UNNAMED(NODE) (LANG_TYPE_CLASS_CHECK (NODE)->was_anonymous)
#define DECL_FRIENDLIST(NODE)		(DECL_INITIAL (NODE))
#define FRIEND_NAME(LIST) (TREE_PURPOSE (LIST))
#define FRIEND_DECLS(LIST) (TREE_VALUE (LIST))
#define DECL_ACCESS(NODE) (LANG_DECL_U2_CHECK (NODE, 0)->access)
#define DECL_GLOBAL_CTOR_P(NODE) \
(LANG_DECL_FN_CHECK (NODE)->global_ctor_p)
#define DECL_GLOBAL_DTOR_P(NODE) \
(LANG_DECL_FN_CHECK (NODE)->global_dtor_p)
#define DECL_TEMPLATE_PARMS(NODE)       \
((struct tree_template_decl *)CONST_CAST_TREE (TEMPLATE_DECL_CHECK (NODE)))->arguments
#define DECL_INNERMOST_TEMPLATE_PARMS(NODE) \
INNERMOST_TEMPLATE_PARMS (DECL_TEMPLATE_PARMS (NODE))
#define DECL_NTPARMS(NODE) \
TREE_VEC_LENGTH (DECL_INNERMOST_TEMPLATE_PARMS (NODE))
#define DECL_TEMPLATE_RESULT(NODE)      \
((struct tree_template_decl *)CONST_CAST_TREE(TEMPLATE_DECL_CHECK (NODE)))->result
#define DECL_TEMPLATE_INSTANTIATIONS(NODE) \
DECL_SIZE_UNIT (TEMPLATE_DECL_CHECK (NODE))
#define DECL_TEMPLATE_SPECIALIZATIONS(NODE)     \
DECL_SIZE (TEMPLATE_DECL_CHECK (NODE))
#define DECL_TEMPLATE_PARM_P(NODE)		\
(DECL_LANG_FLAG_0 (NODE)			\
&& (TREE_CODE (NODE) == CONST_DECL		\
|| TREE_CODE (NODE) == PARM_DECL		\
|| TREE_CODE (NODE) == TYPE_DECL		\
|| TREE_CODE (NODE) == TEMPLATE_DECL))
#define TEMPLATE_PARM_P(NODE)					\
(TREE_CODE (NODE) == TEMPLATE_TYPE_PARM			\
|| TREE_CODE (NODE) == TEMPLATE_TEMPLATE_PARM		\
|| TREE_CODE (NODE) == TEMPLATE_PARM_INDEX)
#define SET_DECL_TEMPLATE_PARM_P(NODE) \
(DECL_LANG_FLAG_0 (NODE) = 1)
#define DECL_TEMPLATE_TEMPLATE_PARM_P(NODE) \
(TREE_CODE (NODE) == TEMPLATE_DECL && DECL_TEMPLATE_PARM_P (NODE))
#define DECL_FUNCTION_TEMPLATE_P(NODE)                          \
(TREE_CODE (NODE) == TEMPLATE_DECL                            \
&& DECL_TEMPLATE_RESULT (NODE) != NULL_TREE			\
&& TREE_CODE (DECL_TEMPLATE_RESULT (NODE)) == FUNCTION_DECL)
#define DECL_TYPE_TEMPLATE_P(NODE)				\
(TREE_CODE (NODE) == TEMPLATE_DECL				\
&& DECL_TEMPLATE_RESULT (NODE) != NULL_TREE			\
&& TREE_CODE (DECL_TEMPLATE_RESULT (NODE)) == TYPE_DECL)
#define DECL_CLASS_TEMPLATE_P(NODE)				\
(DECL_TYPE_TEMPLATE_P (NODE)					\
&& DECL_IMPLICIT_TYPEDEF_P (DECL_TEMPLATE_RESULT (NODE)))
#define DECL_ALIAS_TEMPLATE_P(NODE)			\
(DECL_TYPE_TEMPLATE_P (NODE)				\
&& !DECL_ARTIFICIAL (DECL_TEMPLATE_RESULT (NODE)))
#define DECL_DECLARES_TYPE_P(NODE) \
(TREE_CODE (NODE) == TYPE_DECL || DECL_TYPE_TEMPLATE_P (NODE))
#define DECL_DECLARES_FUNCTION_P(NODE) \
(TREE_CODE (NODE) == FUNCTION_DECL || DECL_FUNCTION_TEMPLATE_P (NODE))
#define DECL_IMPLICIT_TYPEDEF_P(NODE) \
(TREE_CODE (NODE) == TYPE_DECL && DECL_LANG_FLAG_2 (NODE))
#define SET_DECL_IMPLICIT_TYPEDEF_P(NODE) \
(DECL_LANG_FLAG_2 (NODE) = 1)
#define DECL_SELF_REFERENCE_P(NODE) \
(TREE_CODE (NODE) == TYPE_DECL && DECL_LANG_FLAG_4 (NODE))
#define SET_DECL_SELF_REFERENCE_P(NODE) \
(DECL_LANG_FLAG_4 (NODE) = 1)
#define DECL_PRIMARY_TEMPLATE(NODE) \
(TREE_TYPE (DECL_INNERMOST_TEMPLATE_PARMS (NODE)))
#define PRIMARY_TEMPLATE_P(NODE) (DECL_PRIMARY_TEMPLATE (NODE) == (NODE))
#define DECL_USE_TEMPLATE(NODE) (DECL_LANG_SPECIFIC (NODE)->u.base.use_template)
#define CLASSTYPE_USE_TEMPLATE(NODE) \
(LANG_TYPE_CLASS_CHECK (NODE)->use_template)
#define CLASSTYPE_SPECIALIZATION_OF_PRIMARY_TEMPLATE_P(NODE)	\
(CLASS_TYPE_P (NODE)						\
&& CLASSTYPE_USE_TEMPLATE (NODE)				\
&& PRIMARY_TEMPLATE_P (CLASSTYPE_TI_TEMPLATE (NODE)))
#define DECL_TEMPLATE_INSTANTIATION(NODE) (DECL_USE_TEMPLATE (NODE) & 1)
#define CLASSTYPE_TEMPLATE_INSTANTIATION(NODE) \
(CLASSTYPE_USE_TEMPLATE (NODE) & 1)
#define DECL_TEMPLATE_SPECIALIZATION(NODE) (DECL_USE_TEMPLATE (NODE) == 2)
#define SET_DECL_TEMPLATE_SPECIALIZATION(NODE) (DECL_USE_TEMPLATE (NODE) = 2)
#define CLASSTYPE_TEMPLATE_SPECIALIZATION(NODE) \
(CLASSTYPE_USE_TEMPLATE (NODE) == 2)
#define SET_CLASSTYPE_TEMPLATE_SPECIALIZATION(NODE) \
(CLASSTYPE_USE_TEMPLATE (NODE) = 2)
#define DECL_IMPLICIT_INSTANTIATION(NODE) (DECL_USE_TEMPLATE (NODE) == 1)
#define SET_DECL_IMPLICIT_INSTANTIATION(NODE) (DECL_USE_TEMPLATE (NODE) = 1)
#define CLASSTYPE_IMPLICIT_INSTANTIATION(NODE) \
(CLASSTYPE_USE_TEMPLATE (NODE) == 1)
#define SET_CLASSTYPE_IMPLICIT_INSTANTIATION(NODE) \
(CLASSTYPE_USE_TEMPLATE (NODE) = 1)
#define DECL_EXPLICIT_INSTANTIATION(NODE) (DECL_USE_TEMPLATE (NODE) == 3)
#define SET_DECL_EXPLICIT_INSTANTIATION(NODE) (DECL_USE_TEMPLATE (NODE) = 3)
#define CLASSTYPE_EXPLICIT_INSTANTIATION(NODE) \
(CLASSTYPE_USE_TEMPLATE (NODE) == 3)
#define SET_CLASSTYPE_EXPLICIT_INSTANTIATION(NODE) \
(CLASSTYPE_USE_TEMPLATE (NODE) = 3)
#define DECL_FRIEND_PSEUDO_TEMPLATE_INSTANTIATION(DECL) \
(DECL_LANG_SPECIFIC (DECL) && DECL_TEMPLATE_INFO (DECL) \
&& !DECL_USE_TEMPLATE (DECL))
#define DECL_TEMPLOID_INSTANTIATION(DECL)		\
(DECL_TEMPLATE_INSTANTIATION (DECL)			\
|| DECL_FRIEND_PSEUDO_TEMPLATE_INSTANTIATION (DECL))
#define DECL_GENERATED_P(DECL) \
(DECL_TEMPLOID_INSTANTIATION (DECL) || DECL_DEFAULTED_FN (DECL))
#define PROCESSING_REAL_TEMPLATE_DECL_P() \
(!processing_template_parmlist \
&& processing_template_decl > template_class_depth (current_scope ()))
#define DECL_TEMPLATE_INSTANTIATED(NODE) \
DECL_LANG_FLAG_1 (VAR_OR_FUNCTION_DECL_CHECK (NODE))
#define DECL_INTERFACE_KNOWN(NODE) DECL_LANG_FLAG_5 (NODE)
#define DECL_NOT_REALLY_EXTERN(NODE) \
(DECL_LANG_SPECIFIC (NODE)->u.base.not_really_extern)
#define DECL_REALLY_EXTERN(NODE) \
(DECL_EXTERNAL (NODE)				\
&& (!DECL_LANG_SPECIFIC (NODE) || !DECL_NOT_REALLY_EXTERN (NODE)))
#define THUNK_FIXED_OFFSET(DECL) \
(DECL_LANG_SPECIFIC (THUNK_FUNCTION_CHECK (DECL))->u.fn.u5.fixed_offset)
#define THUNK_VIRTUAL_OFFSET(DECL) \
(LANG_DECL_U2_CHECK (FUNCTION_DECL_CHECK (DECL), 0)->access)
#define THUNK_ALIAS(DECL) \
(DECL_LANG_SPECIFIC (FUNCTION_DECL_CHECK (DECL))->u.min.template_info)
#define THUNK_TARGET(NODE)				\
(LANG_DECL_FN_CHECK (NODE)->befriending_classes)
#define QUALIFIED_NAME_IS_TEMPLATE(NODE) \
(TREE_LANG_FLAG_1 (SCOPE_REF_CHECK (NODE)))
#define OMP_ATOMIC_DEPENDENT_P(NODE) \
(TREE_CODE (TREE_OPERAND (OMP_ATOMIC_CHECK (NODE), 0)) == INTEGER_CST)
#define OMP_FOR_GIMPLIFYING_P(NODE) \
(TREE_LANG_FLAG_0 (OMP_LOOP_CHECK (NODE)))
#define CP_OMP_CLAUSE_INFO(NODE) \
TREE_TYPE (OMP_CLAUSE_RANGE_CHECK (NODE, OMP_CLAUSE_PRIVATE, \
OMP_CLAUSE_LINEAR))
#define TRANSACTION_EXPR_IS_STMT(NODE) \
TREE_LANG_FLAG_0 (TRANSACTION_EXPR_CHECK (NODE))
#define TRY_STMTS(NODE)		TREE_OPERAND (TRY_BLOCK_CHECK (NODE), 0)
#define TRY_HANDLERS(NODE)	TREE_OPERAND (TRY_BLOCK_CHECK (NODE), 1)
#define EH_SPEC_STMTS(NODE)	TREE_OPERAND (EH_SPEC_BLOCK_CHECK (NODE), 0)
#define EH_SPEC_RAISES(NODE)	TREE_OPERAND (EH_SPEC_BLOCK_CHECK (NODE), 1)
#define USING_STMT_NAMESPACE(NODE) TREE_OPERAND (USING_STMT_CHECK (NODE), 0)
#define FN_TRY_BLOCK_P(NODE)	TREE_LANG_FLAG_3 (TRY_BLOCK_CHECK (NODE))
#define HANDLER_PARMS(NODE)	TREE_OPERAND (HANDLER_CHECK (NODE), 0)
#define HANDLER_BODY(NODE)	TREE_OPERAND (HANDLER_CHECK (NODE), 1)
#define HANDLER_TYPE(NODE)	TREE_TYPE (HANDLER_CHECK (NODE))
#define CLEANUP_BODY(NODE)	TREE_OPERAND (CLEANUP_STMT_CHECK (NODE), 0)
#define CLEANUP_EXPR(NODE)	TREE_OPERAND (CLEANUP_STMT_CHECK (NODE), 1)
#define CLEANUP_DECL(NODE)	TREE_OPERAND (CLEANUP_STMT_CHECK (NODE), 2)
#define IF_COND(NODE)		TREE_OPERAND (IF_STMT_CHECK (NODE), 0)
#define THEN_CLAUSE(NODE)	TREE_OPERAND (IF_STMT_CHECK (NODE), 1)
#define ELSE_CLAUSE(NODE)	TREE_OPERAND (IF_STMT_CHECK (NODE), 2)
#define IF_SCOPE(NODE)		TREE_OPERAND (IF_STMT_CHECK (NODE), 3)
#define IF_STMT_CONSTEXPR_P(NODE) TREE_LANG_FLAG_0 (IF_STMT_CHECK (NODE))
#define IF_STMT_EXTRA_ARGS(NODE) IF_SCOPE (NODE)
#define WHILE_COND(NODE)	TREE_OPERAND (WHILE_STMT_CHECK (NODE), 0)
#define WHILE_BODY(NODE)	TREE_OPERAND (WHILE_STMT_CHECK (NODE), 1)
#define DO_COND(NODE)		TREE_OPERAND (DO_STMT_CHECK (NODE), 0)
#define DO_BODY(NODE)		TREE_OPERAND (DO_STMT_CHECK (NODE), 1)
#define FOR_INIT_STMT(NODE)	TREE_OPERAND (FOR_STMT_CHECK (NODE), 0)
#define FOR_COND(NODE)		TREE_OPERAND (FOR_STMT_CHECK (NODE), 1)
#define FOR_EXPR(NODE)		TREE_OPERAND (FOR_STMT_CHECK (NODE), 2)
#define FOR_BODY(NODE)		TREE_OPERAND (FOR_STMT_CHECK (NODE), 3)
#define FOR_SCOPE(NODE)		TREE_OPERAND (FOR_STMT_CHECK (NODE), 4)
#define RANGE_FOR_DECL(NODE)	TREE_OPERAND (RANGE_FOR_STMT_CHECK (NODE), 0)
#define RANGE_FOR_EXPR(NODE)	TREE_OPERAND (RANGE_FOR_STMT_CHECK (NODE), 1)
#define RANGE_FOR_BODY(NODE)	TREE_OPERAND (RANGE_FOR_STMT_CHECK (NODE), 2)
#define RANGE_FOR_SCOPE(NODE)	TREE_OPERAND (RANGE_FOR_STMT_CHECK (NODE), 3)
#define RANGE_FOR_UNROLL(NODE)	TREE_OPERAND (RANGE_FOR_STMT_CHECK (NODE), 4)
#define RANGE_FOR_IVDEP(NODE)	TREE_LANG_FLAG_6 (RANGE_FOR_STMT_CHECK (NODE))
#define SWITCH_STMT_COND(NODE)	TREE_OPERAND (SWITCH_STMT_CHECK (NODE), 0)
#define SWITCH_STMT_BODY(NODE)	TREE_OPERAND (SWITCH_STMT_CHECK (NODE), 1)
#define SWITCH_STMT_TYPE(NODE)	TREE_OPERAND (SWITCH_STMT_CHECK (NODE), 2)
#define SWITCH_STMT_SCOPE(NODE)	TREE_OPERAND (SWITCH_STMT_CHECK (NODE), 3)
#define SWITCH_STMT_ALL_CASES_P(NODE) \
TREE_LANG_FLAG_0 (SWITCH_STMT_CHECK (NODE))
#define SWITCH_STMT_NO_BREAK_P(NODE) \
TREE_LANG_FLAG_2 (SWITCH_STMT_CHECK (NODE))
#define STMT_EXPR_STMT(NODE)	TREE_OPERAND (STMT_EXPR_CHECK (NODE), 0)
#define EXPR_STMT_EXPR(NODE)	TREE_OPERAND (EXPR_STMT_CHECK (NODE), 0)
#define TARGET_EXPR_IMPLICIT_P(NODE) \
TREE_LANG_FLAG_0 (TARGET_EXPR_CHECK (NODE))
#define TARGET_EXPR_LIST_INIT_P(NODE) \
TREE_LANG_FLAG_1 (TARGET_EXPR_CHECK (NODE))
#define TARGET_EXPR_DIRECT_INIT_P(NODE) \
TREE_LANG_FLAG_2 (TARGET_EXPR_CHECK (NODE))
#define SIMPLE_TARGET_EXPR_P(NODE)				\
(TREE_CODE (NODE) == TARGET_EXPR				\
&& !VOID_TYPE_P (TREE_TYPE (TARGET_EXPR_INITIAL (NODE))))
#define DIRECT_INIT_EXPR_P(TYPE,EXPR)					\
(TREE_CODE (EXPR) == TARGET_EXPR && TREE_LANG_FLAG_2 (EXPR)		\
&& same_type_ignoring_top_level_qualifiers_p (TYPE, TREE_TYPE (EXPR)))
#define CONVERT_EXPR_VBASE_PATH(NODE) \
TREE_LANG_FLAG_0 (CONVERT_EXPR_CHECK (NODE))
#define SIZEOF_EXPR_TYPE_P(NODE) \
TREE_LANG_FLAG_0 (SIZEOF_EXPR_CHECK (NODE))
#define ALIGNOF_EXPR_STD_P(NODE) \
TREE_LANG_FLAG_0 (ALIGNOF_EXPR_CHECK (NODE))
enum tag_types {
none_type = 0, 
record_type,   
class_type,    
union_type,    
enum_type,     
typename_type, 
scope_type	 
};
enum cp_lvalue_kind_flags {
clk_none = 0,     
clk_ordinary = 1, 
clk_rvalueref = 2,
clk_class = 4,    
clk_bitfield = 8, 
clk_packed = 16   
};
typedef int cp_lvalue_kind;
enum tmpl_spec_kind {
tsk_none,		   
tsk_invalid_member_spec, 
tsk_invalid_expl_inst,   
tsk_excessive_parms,	   
tsk_insufficient_parms,  
tsk_template,		   
tsk_expl_spec,	   
tsk_expl_inst		   
};
enum access_kind {
ak_none = 0,		   
ak_public = 1,	   
ak_protected = 2,	   
ak_private = 3	   
};
enum special_function_kind {
sfk_none = 0,		   
sfk_constructor,	   
sfk_copy_constructor,    
sfk_move_constructor,    
sfk_copy_assignment,     
sfk_move_assignment,     
sfk_destructor,	   
sfk_complete_destructor, 
sfk_base_destructor,     
sfk_deleting_destructor, 
sfk_conversion,	   
sfk_deduction_guide,	   
sfk_inheriting_constructor 
};
enum linkage_kind {
lk_none,			
lk_internal,			
lk_external			
};
enum duration_kind {
dk_static,
dk_thread,
dk_auto,
dk_dynamic
};
enum tsubst_flags {
tf_none = 0,			 
tf_error = 1 << 0,		 
tf_warning = 1 << 1,	 	 
tf_ignore_bad_quals = 1 << 2,	 
tf_keep_type_decl = 1 << 3,	 
tf_ptrmem_ok = 1 << 4,	 
tf_user = 1 << 5,		 
tf_conv = 1 << 6,		 
tf_decltype = 1 << 7,          
tf_partial = 1 << 8,		 
tf_fndecl_type = 1 << 9,   
tf_no_cleanup = 1 << 10,   
tf_warning_or_error = tf_warning | tf_error
};
typedef int tsubst_flags_t;
enum base_access_flags {
ba_any = 0,  
ba_unique = 1 << 0,  
ba_check_bit = 1 << 1,   
ba_check = ba_unique | ba_check_bit,
ba_ignore_scope = 1 << 2 
};
typedef int base_access;
enum deferring_kind {
dk_no_deferred = 0, 
dk_deferred = 1,    
dk_no_check = 2     
};
enum base_kind {
bk_inaccessible = -3,   
bk_ambig = -2,	  
bk_not_base = -1,	  
bk_same_type = 0,	  
bk_proper_base = 1,	  
bk_via_virtual = 2	  
};
#define vfunc_ptr_type_node  vtable_entry_type
extern GTY(()) tree integer_two_node;
extern int function_depth;
extern int comparing_specializations;
extern int cp_unevaluated_operand;
struct cp_unevaluated
{
cp_unevaluated ();
~cp_unevaluated ();
};
enum unification_kind_t {
DEDUCE_CALL,
DEDUCE_CONV,
DEDUCE_EXACT
};
enum lss_policy { lss_blank, lss_copy };
struct local_specialization_stack
{
local_specialization_stack (lss_policy = lss_blank);
~local_specialization_stack ();
hash_map<tree, tree> *saved;
};
extern int current_class_depth;
extern GTY(()) vec<tree, va_gc> *local_classes;
extern GTY(()) vec<tree, va_gc> *static_decls;
extern GTY(()) vec<tree, va_gc> *keyed_classes;

#ifndef NO_DOT_IN_LABEL
#define JOINER '.'
#define AUTO_TEMP_NAME "_.tmp_"
#define VFIELD_BASE ".vf"
#define VFIELD_NAME "_vptr."
#define VFIELD_NAME_FORMAT "_vptr.%s"
#else 
#ifndef NO_DOLLAR_IN_LABEL
#define JOINER '$'
#define AUTO_TEMP_NAME "_$tmp_"
#define VFIELD_BASE "$vf"
#define VFIELD_NAME "_vptr$"
#define VFIELD_NAME_FORMAT "_vptr$%s"
#else 
#define AUTO_TEMP_NAME "__tmp_"
#define TEMP_NAME_P(ID_NODE) \
(!strncmp (IDENTIFIER_POINTER (ID_NODE), AUTO_TEMP_NAME, \
sizeof (AUTO_TEMP_NAME) - 1))
#define VTABLE_NAME "__vt_"
#define VTABLE_NAME_P(ID_NODE) \
(!strncmp (IDENTIFIER_POINTER (ID_NODE), VTABLE_NAME, \
sizeof (VTABLE_NAME) - 1))
#define VFIELD_BASE "__vfb"
#define VFIELD_NAME "__vptr_"
#define VFIELD_NAME_P(ID_NODE) \
(!strncmp (IDENTIFIER_POINTER (ID_NODE), VFIELD_NAME, \
sizeof (VFIELD_NAME) - 1))
#define VFIELD_NAME_FORMAT "__vptr_%s"
#endif	
#endif	
#define LAMBDANAME_PREFIX "__lambda"
#define LAMBDANAME_FORMAT LAMBDANAME_PREFIX "%d"
#define UDLIT_OP_ANSI_PREFIX "operator\"\""
#define UDLIT_OP_ANSI_FORMAT UDLIT_OP_ANSI_PREFIX "%s"
#define UDLIT_OP_MANGLED_PREFIX "li"
#define UDLIT_OP_MANGLED_FORMAT UDLIT_OP_MANGLED_PREFIX "%s"
#define UDLIT_OPER_P(ID_NODE) \
(!strncmp (IDENTIFIER_POINTER (ID_NODE), \
UDLIT_OP_ANSI_PREFIX, \
sizeof (UDLIT_OP_ANSI_PREFIX) - 1))
#define UDLIT_OP_SUFFIX(ID_NODE) \
(IDENTIFIER_POINTER (ID_NODE) + sizeof (UDLIT_OP_ANSI_PREFIX) - 1)
#if !defined(NO_DOLLAR_IN_LABEL) || !defined(NO_DOT_IN_LABEL)
#define VTABLE_NAME_P(ID_NODE) (IDENTIFIER_POINTER (ID_NODE)[1] == 'v' \
&& IDENTIFIER_POINTER (ID_NODE)[2] == 't' \
&& IDENTIFIER_POINTER (ID_NODE)[3] == JOINER)
#define TEMP_NAME_P(ID_NODE) \
(!strncmp (IDENTIFIER_POINTER (ID_NODE), AUTO_TEMP_NAME, sizeof (AUTO_TEMP_NAME)-1))
#define VFIELD_NAME_P(ID_NODE) \
(!strncmp (IDENTIFIER_POINTER (ID_NODE), VFIELD_NAME, sizeof(VFIELD_NAME)-1))
#endif 

extern int at_eof;
extern bool defer_mangling_aliases;
extern bool flag_noexcept_type;
extern GTY(()) tree static_aggregates;
extern GTY(()) tree tls_aggregates;
enum overload_flags { NO_SPECIAL = 0, DTOR_FLAG, TYPENAME_FLAG };
#define LOOKUP_PROTECT (1 << 0)
#define LOOKUP_NORMAL (LOOKUP_PROTECT)
#define LOOKUP_NONVIRTUAL (1 << 1)
#define LOOKUP_ONLYCONVERTING (1 << 2)
#define LOOKUP_IMPLICIT (LOOKUP_NORMAL | LOOKUP_ONLYCONVERTING)
#define DIRECT_BIND (1 << 3)
#define LOOKUP_NO_CONVERSION (1 << 4)
#define LOOKUP_DESTRUCTOR (1 << 5)
#define LOOKUP_NO_TEMP_BIND (1 << 6)
#define LOOKUP_PREFER_TYPES (1 << 7)
#define LOOKUP_PREFER_NAMESPACES (1 << 8)
#define LOOKUP_PREFER_BOTH (LOOKUP_PREFER_TYPES | LOOKUP_PREFER_NAMESPACES)
#define LOOKUP_HIDDEN (LOOKUP_PREFER_NAMESPACES << 1)
#define LOOKUP_PREFER_RVALUE (LOOKUP_HIDDEN << 1)
#define LOOKUP_NO_NARROWING (LOOKUP_PREFER_RVALUE << 1)
#define LOOKUP_LIST_INIT_CTOR (LOOKUP_NO_NARROWING << 1)
#define LOOKUP_COPY_PARM (LOOKUP_LIST_INIT_CTOR << 1)
#define LOOKUP_LIST_ONLY (LOOKUP_COPY_PARM << 1)
#define LOOKUP_SPECULATIVE (LOOKUP_LIST_ONLY << 1)
#define LOOKUP_DEFAULTED (LOOKUP_SPECULATIVE << 1)
#define LOOKUP_ALREADY_DIGESTED (LOOKUP_DEFAULTED << 1)
#define LOOKUP_EXPLICIT_TMPL_ARGS (LOOKUP_ALREADY_DIGESTED << 1)
#define LOOKUP_NO_RVAL_BIND (LOOKUP_EXPLICIT_TMPL_ARGS << 1)
#define LOOKUP_NO_NON_INTEGRAL (LOOKUP_NO_RVAL_BIND << 1)
#define LOOKUP_DELEGATING_CONS (LOOKUP_NO_NON_INTEGRAL << 1)
#define LOOKUP_NAMESPACES_ONLY(F)  \
(((F) & LOOKUP_PREFER_NAMESPACES) && !((F) & LOOKUP_PREFER_TYPES))
#define LOOKUP_TYPES_ONLY(F)  \
(!((F) & LOOKUP_PREFER_NAMESPACES) && ((F) & LOOKUP_PREFER_TYPES))
#define LOOKUP_QUALIFIERS_ONLY(F)     ((F) & LOOKUP_PREFER_BOTH)
#define CONV_IMPLICIT    1
#define CONV_STATIC      2
#define CONV_CONST       4
#define CONV_REINTERPRET 8
#define CONV_PRIVATE	 16
#define CONV_FORCE_TEMP  64
#define CONV_FOLD	 128
#define CONV_OLD_CONVERT (CONV_IMPLICIT | CONV_STATIC | CONV_CONST \
| CONV_REINTERPRET)
#define CONV_C_CAST      (CONV_IMPLICIT | CONV_STATIC | CONV_CONST \
| CONV_REINTERPRET | CONV_PRIVATE | CONV_FORCE_TEMP)
#define CONV_BACKEND_CONVERT (CONV_OLD_CONVERT | CONV_FOLD)
#define WANT_INT	1 
#define WANT_FLOAT	2 
#define WANT_ENUM	4 
#define WANT_POINTER	8 
#define WANT_NULL      16 
#define WANT_VECTOR_OR_COMPLEX 32 
#define WANT_ARITH	(WANT_INT | WANT_FLOAT | WANT_VECTOR_OR_COMPLEX)
#define COMPARE_STRICT	      0 
#define COMPARE_BASE	      1 
#define COMPARE_DERIVED	      2 
#define COMPARE_REDECLARATION 4 
#define COMPARE_STRUCTURAL    8 
#define SF_DEFAULT	     0  
#define SF_PRE_PARSED	     1  
#define SF_INCLASS_INLINE    2  
#define SD_UNINITIALIZED     0
#define SD_INITIALIZED       1
#define SD_DEFAULTED         2
#define SD_DELETED           3
#define same_or_base_type_p(TYPE1, TYPE2) \
comptypes ((TYPE1), (TYPE2), COMPARE_BASE)
#define TEMPLATE_PARM_INDEX_CAST(NODE) \
((template_parm_index*)TEMPLATE_PARM_INDEX_CHECK (NODE))
#define TEMPLATE_PARM_IDX(NODE) (TEMPLATE_PARM_INDEX_CAST (NODE)->index)
#define TEMPLATE_PARM_LEVEL(NODE) (TEMPLATE_PARM_INDEX_CAST (NODE)->level)
#define TEMPLATE_PARM_DESCENDANTS(NODE) (TREE_CHAIN (NODE))
#define TEMPLATE_PARM_ORIG_LEVEL(NODE) (TEMPLATE_PARM_INDEX_CAST (NODE)->orig_level)
#define TEMPLATE_PARM_DECL(NODE) (TEMPLATE_PARM_INDEX_CAST (NODE)->decl)
#define TEMPLATE_PARM_PARAMETER_PACK(NODE) \
(TREE_LANG_FLAG_0 (TEMPLATE_PARM_INDEX_CHECK (NODE)))
#define TEMPLATE_TYPE_PARM_INDEX(NODE)					\
(TYPE_VALUES_RAW (TREE_CHECK3 ((NODE), TEMPLATE_TYPE_PARM,		\
TEMPLATE_TEMPLATE_PARM,		\
BOUND_TEMPLATE_TEMPLATE_PARM)))
#define TEMPLATE_TYPE_IDX(NODE) \
(TEMPLATE_PARM_IDX (TEMPLATE_TYPE_PARM_INDEX (NODE)))
#define TEMPLATE_TYPE_LEVEL(NODE) \
(TEMPLATE_PARM_LEVEL (TEMPLATE_TYPE_PARM_INDEX (NODE)))
#define TEMPLATE_TYPE_ORIG_LEVEL(NODE) \
(TEMPLATE_PARM_ORIG_LEVEL (TEMPLATE_TYPE_PARM_INDEX (NODE)))
#define TEMPLATE_TYPE_DECL(NODE) \
(TEMPLATE_PARM_DECL (TEMPLATE_TYPE_PARM_INDEX (NODE)))
#define TEMPLATE_TYPE_PARAMETER_PACK(NODE) \
(TEMPLATE_PARM_PARAMETER_PACK (TEMPLATE_TYPE_PARM_INDEX (NODE)))
#define CLASS_PLACEHOLDER_TEMPLATE(NODE) \
(DECL_INITIAL (TYPE_NAME (TEMPLATE_TYPE_PARM_CHECK (NODE))))
enum auto_deduction_context
{
adc_unspecified,   
adc_variable_type, 
adc_return_type,   
adc_unify,         
adc_requirement,   
adc_decomp_type    
};
#define TEMPLATE_TYPE_PARM_FOR_CLASS(NODE) \
(TREE_LANG_FLAG_0 (TEMPLATE_TYPE_PARM_CHECK (NODE)))
#define AUTO_IS_DECLTYPE(NODE) \
(TYPE_LANG_FLAG_5 (TEMPLATE_TYPE_PARM_CHECK (NODE)))
#define TFF_PLAIN_IDENTIFIER			(0)
#define TFF_SCOPE				(1)
#define TFF_CHASE_TYPEDEF			(1 << 1)
#define TFF_DECL_SPECIFIERS			(1 << 2)
#define TFF_CLASS_KEY_OR_ENUM			(1 << 3)
#define TFF_RETURN_TYPE				(1 << 4)
#define TFF_FUNCTION_DEFAULT_ARGUMENTS		(1 << 5)
#define TFF_EXCEPTION_SPECIFICATION		(1 << 6)
#define TFF_TEMPLATE_HEADER			(1 << 7)
#define TFF_TEMPLATE_NAME			(1 << 8)
#define TFF_EXPR_IN_PARENS			(1 << 9)
#define TFF_NO_FUNCTION_ARGUMENTS		(1 << 10)
#define TFF_UNQUALIFIED_NAME			(1 << 11)
#define TFF_NO_OMIT_DEFAULT_TEMPLATE_ARGUMENTS	(1 << 12)
#define TFF_NO_TEMPLATE_BINDINGS		(1 << 13)
#define TFF_POINTER		                (1 << 14)
#define TEMPLATE_TEMPLATE_PARM_TEMPLATE_DECL(NODE)	\
((TREE_CODE (NODE) == BOUND_TEMPLATE_TEMPLATE_PARM)	\
? TYPE_TI_TEMPLATE (NODE)				\
: TYPE_NAME (NODE))
extern void init_reswords (void);
enum ovl_op_flags
{
OVL_OP_FLAG_NONE = 0,	
OVL_OP_FLAG_UNARY = 1,	
OVL_OP_FLAG_BINARY = 2,	
OVL_OP_FLAG_AMBIARY = 3,	
OVL_OP_FLAG_ALLOC = 4,  	
OVL_OP_FLAG_DELETE = 1,	
OVL_OP_FLAG_VEC = 2		
};
enum ovl_op_code
{
OVL_OP_ERROR_MARK,
OVL_OP_NOP_EXPR,
#define DEF_OPERATOR(NAME, CODE, MANGLING, FLAGS) OVL_OP_##CODE,
#define DEF_ASSN_OPERATOR(NAME, CODE, MANGLING) 
#include "operators.def"
OVL_OP_MAX
};
struct GTY(()) ovl_op_info_t {
tree identifier;
const char *name;
const char *mangled_name;
enum tree_code tree_code : 16;
enum ovl_op_code ovl_op_code : 8;
unsigned flags : 8;
};
extern GTY(()) ovl_op_info_t ovl_op_info[2][OVL_OP_MAX];
extern GTY(()) unsigned char ovl_op_mapping[MAX_TREE_CODES];
extern GTY(()) unsigned char ovl_op_alternate[OVL_OP_MAX];
#define OVL_OP_INFO(IS_ASS_P, TREE_CODE)			\
(&ovl_op_info[(IS_ASS_P) != 0][ovl_op_mapping[(TREE_CODE)]])
#define IDENTIFIER_OVL_OP_INFO(NODE) \
(&ovl_op_info[IDENTIFIER_KIND_BIT_0 (NODE)][IDENTIFIER_CP_INDEX (NODE)])
#define IDENTIFIER_OVL_OP_FLAGS(NODE) \
(IDENTIFIER_OVL_OP_INFO (NODE)->flags)
typedef int cp_cv_quals;
enum virt_specifier
{
VIRT_SPEC_UNSPECIFIED = 0x0,
VIRT_SPEC_FINAL       = 0x1,
VIRT_SPEC_OVERRIDE    = 0x2
};
typedef int cp_virt_specifiers;
enum cp_ref_qualifier {
REF_QUAL_NONE = 0,
REF_QUAL_LVALUE = 1,
REF_QUAL_RVALUE = 2
};
enum cp_storage_class {
sc_none = 0,
sc_auto,
sc_register,
sc_static,
sc_extern,
sc_mutable
};
enum cp_decl_spec {
ds_first,
ds_signed = ds_first,
ds_unsigned,
ds_short,
ds_long,
ds_const,
ds_volatile,
ds_restrict,
ds_inline,
ds_virtual,
ds_explicit,
ds_friend,
ds_typedef,
ds_alias,
ds_constexpr,
ds_complex,
ds_thread,
ds_type_spec,
ds_redefined_builtin_type_spec,
ds_attribute,
ds_std_attribute,
ds_storage_class,
ds_long_long,
ds_concept,
ds_last 
};
struct cp_decl_specifier_seq {
source_location locations[ds_last];
tree type;
tree attributes;
tree std_attributes;
tree redefined_builtin_type;
cp_storage_class storage_class;
int int_n_idx;
BOOL_BITFIELD type_definition_p : 1;
BOOL_BITFIELD multiple_types_p : 1;
BOOL_BITFIELD conflicting_specifiers_p : 1;
BOOL_BITFIELD any_specifiers_p : 1;
BOOL_BITFIELD any_type_specifiers_p : 1;
BOOL_BITFIELD explicit_int_p : 1;
BOOL_BITFIELD explicit_intN_p : 1;
BOOL_BITFIELD explicit_char_p : 1;
BOOL_BITFIELD gnu_thread_keyword_p : 1;
BOOL_BITFIELD decltype_p : 1;
};
enum cp_declarator_kind {
cdk_id,
cdk_function,
cdk_array,
cdk_pointer,
cdk_reference,
cdk_ptrmem,
cdk_decomp,
cdk_error
};
typedef struct cp_declarator cp_declarator;
typedef struct cp_parameter_declarator cp_parameter_declarator;
struct cp_parameter_declarator {
cp_parameter_declarator *next;
cp_decl_specifier_seq decl_specifiers;
cp_declarator *declarator;
tree default_argument;
bool template_parameter_pack_p;
location_t loc;
};
struct cp_declarator {
ENUM_BITFIELD (cp_declarator_kind) kind : 4;
BOOL_BITFIELD parameter_pack_p : 1;
location_t parenthesized;
location_t id_loc; 
tree attributes;
tree std_attributes;
cp_declarator *declarator;
union {
struct {
tree qualifying_scope;
tree unqualified_name;
special_function_kind sfk;
} id;
struct {
tree parameters;
cp_cv_quals qualifiers;
cp_virt_specifiers virt_specifiers;
cp_ref_qualifier ref_qualifier;
tree tx_qualifier;
tree exception_specification;
tree late_return_type;
tree requires_clause;
} function;
struct {
tree bounds;
} array;
struct {
cp_cv_quals qualifiers;
tree class_type;
} pointer;
struct {
cp_cv_quals qualifiers;
bool rvalue_ref;
} reference;
} u;
};
struct GTY((chain_next ("%h.next"))) tinst_level {
struct tinst_level *next;
tree tldcl, targs;
private:
bool split_list_p () const { return targs; }
bool tree_list_p () const
{
return !split_list_p () && TREE_CODE (tldcl) == TREE_LIST;
}
bool not_list_p () const
{
return !split_list_p () && !tree_list_p ();
}
tree to_list ();
public:
static void free (tinst_level *obj);
bool list_p () const { return !not_list_p (); }
tree get_node () {
if (!split_list_p ()) return tldcl;
else return to_list ();
}
tree maybe_get_node () const {
if (!split_list_p ()) return tldcl;
else return NULL_TREE;
}
location_t locus;
unsigned short errors;
unsigned short refcount;
static const unsigned short refcount_infinity = (unsigned short) ~0;
};
bool decl_spec_seq_has_spec_p (const cp_decl_specifier_seq *, cp_decl_spec);
inline tree
type_of_this_parm (const_tree fntype)
{
function_args_iterator iter;
gcc_assert (TREE_CODE (fntype) == METHOD_TYPE);
function_args_iter_init (&iter, fntype);
return function_args_iter_cond (&iter);
}
inline tree
class_of_this_parm (const_tree fntype)
{
return TREE_TYPE (type_of_this_parm (fntype));
}
inline bool
variable_template_p (tree t)
{
if (TREE_CODE (t) != TEMPLATE_DECL)
return false;
if (!PRIMARY_TEMPLATE_P (t))
return false;
if (tree r = DECL_TEMPLATE_RESULT (t))
return VAR_P (r);
return false;
}
inline bool
variable_concept_p (tree t)
{
if (TREE_CODE (t) != TEMPLATE_DECL)
return false;
if (tree r = DECL_TEMPLATE_RESULT (t))
return VAR_P (r) && DECL_DECLARED_CONCEPT_P (r);
return false;
}
inline bool
concept_template_p (tree t)
{
if (TREE_CODE (t) != TEMPLATE_DECL)
return false;
if (tree r = DECL_TEMPLATE_RESULT (t))
return VAR_OR_FUNCTION_DECL_P (r) && DECL_DECLARED_CONCEPT_P (r);
return false;
}
extern cp_parameter_declarator *no_parameters;
extern int class_dump_id;
extern int raw_dump_id;
extern bool check_dtor_name			(tree, tree);
int magic_varargs_p				(tree);
extern tree build_conditional_expr		(location_t, tree, tree, tree, 
tsubst_flags_t);
extern tree build_addr_func			(tree, tsubst_flags_t);
extern void set_flags_from_callee		(tree);
extern tree build_call_a			(tree, int, tree*);
extern tree build_call_n			(tree, int, ...);
extern bool null_ptr_cst_p			(tree);
extern bool null_member_pointer_value_p		(tree);
extern bool sufficient_parms_p			(const_tree);
extern tree type_decays_to			(tree);
extern tree extract_call_expr			(tree);
extern tree build_user_type_conversion		(tree, tree, int,
tsubst_flags_t);
extern tree build_new_function_call		(tree, vec<tree, va_gc> **,
tsubst_flags_t);
extern tree build_operator_new_call		(tree, vec<tree, va_gc> **,
tree *, tree *, tree, tree,
tree *, tsubst_flags_t);
extern tree build_new_method_call		(tree, tree,
vec<tree, va_gc> **, tree,
int, tree *, tsubst_flags_t);
extern tree build_special_member_call		(tree, tree,
vec<tree, va_gc> **,
tree, int, tsubst_flags_t);
extern tree build_new_op			(location_t, enum tree_code,
int, tree, tree, tree, tree *,
tsubst_flags_t);
extern tree build_op_call			(tree, vec<tree, va_gc> **,
tsubst_flags_t);
extern bool aligned_allocation_fn_p		(tree);
extern bool usual_deallocation_fn_p		(tree);
extern tree build_op_delete_call		(enum tree_code, tree, tree,
bool, tree, tree,
tsubst_flags_t);
extern bool can_convert				(tree, tree, tsubst_flags_t);
extern bool can_convert_standard		(tree, tree, tsubst_flags_t);
extern bool can_convert_arg			(tree, tree, tree, int,
tsubst_flags_t);
extern bool can_convert_arg_bad			(tree, tree, tree, int,
tsubst_flags_t);
extern location_t get_fndecl_argument_location  (tree, int);
class access_failure_info
{
public:
access_failure_info () : m_was_inaccessible (false), m_basetype_path (NULL_TREE),
m_field_decl (NULL_TREE) {}
void record_access_failure (tree basetype_path, tree field_decl);
void maybe_suggest_accessor (bool const_p) const;
private:
bool m_was_inaccessible;
tree m_basetype_path;
tree m_field_decl;
};
extern bool enforce_access			(tree, tree, tree,
tsubst_flags_t,
access_failure_info *afi = NULL);
extern void push_defarg_context			(tree);
extern void pop_defarg_context			(void);
extern tree convert_default_arg			(tree, tree, tree, int,
tsubst_flags_t);
extern tree convert_arg_to_ellipsis		(tree, tsubst_flags_t);
extern tree build_x_va_arg			(source_location, tree, tree);
extern tree cxx_type_promotes_to		(tree);
extern tree type_passed_as			(tree);
extern tree convert_for_arg_passing		(tree, tree, tsubst_flags_t);
extern bool is_properly_derived_from		(tree, tree);
extern tree initialize_reference		(tree, tree, int,
tsubst_flags_t);
extern tree extend_ref_init_temps		(tree, tree, vec<tree, va_gc>**);
extern tree make_temporary_var_for_ref_to_temp	(tree, tree);
extern bool type_has_extended_temps		(tree);
extern tree strip_top_quals			(tree);
extern bool reference_related_p			(tree, tree);
extern int remaining_arguments			(tree);
extern tree perform_implicit_conversion		(tree, tree, tsubst_flags_t);
extern tree perform_implicit_conversion_flags	(tree, tree, tsubst_flags_t, int);
extern tree build_converted_constant_expr	(tree, tree, tsubst_flags_t);
extern tree perform_direct_initialization_if_possible (tree, tree, bool,
tsubst_flags_t);
extern tree in_charge_arg_for_name		(tree);
extern tree build_cxx_call			(tree, int, tree *,
tsubst_flags_t);
extern bool is_std_init_list			(tree);
extern bool is_list_ctor			(tree);
extern void validate_conversion_obstack		(void);
extern void mark_versions_used			(tree);
extern tree get_function_version_dispatcher	(tree);
extern tree build_vfield_ref			(tree, tree);
extern tree build_if_in_charge			(tree true_stmt, tree false_stmt = void_node);
extern tree build_base_path			(enum tree_code, tree,
tree, int, tsubst_flags_t);
extern tree convert_to_base			(tree, tree, bool, bool,
tsubst_flags_t);
extern tree convert_to_base_statically		(tree, tree);
extern tree build_vtbl_ref			(tree, tree);
extern tree build_vfn_ref			(tree, tree);
extern tree get_vtable_decl			(tree, int);
extern bool add_method				(tree, tree, bool);
extern tree declared_access			(tree);
extern tree currently_open_class		(tree);
extern tree currently_open_derived_class	(tree);
extern tree outermost_open_class		(void);
extern tree current_nonlambda_class_type	(void);
extern tree finish_struct			(tree, tree);
extern void finish_struct_1			(tree);
extern int resolves_to_fixed_type_p		(tree, int *);
extern void init_class_processing		(void);
extern int is_empty_class			(tree);
extern bool is_really_empty_class		(tree);
extern void pushclass				(tree);
extern void popclass				(void);
extern void push_nested_class			(tree);
extern void pop_nested_class			(void);
extern int current_lang_depth			(void);
extern void push_lang_context			(tree);
extern void pop_lang_context			(void);
extern tree instantiate_type			(tree, tree, tsubst_flags_t);
extern void build_self_reference		(void);
extern int same_signature_p			(const_tree, const_tree);
extern void maybe_add_class_template_decl_list	(tree, tree, int);
extern void unreverse_member_declarations	(tree);
extern void invalidate_class_lookup_cache	(void);
extern void maybe_note_name_used_in_class	(tree, tree);
extern void note_name_declared_in_class		(tree, tree);
extern tree get_vtbl_decl_for_binfo		(tree);
extern bool vptr_via_virtual_p			(tree);
extern void debug_class				(tree);
extern void debug_thunks			(tree);
extern void set_linkage_according_to_type	(tree, tree);
extern void determine_key_method		(tree);
extern void check_for_override			(tree, tree);
extern void push_class_stack			(void);
extern void pop_class_stack			(void);
extern bool default_ctor_p			(tree);
extern bool type_has_user_nondefault_constructor (tree);
extern tree in_class_defaulted_default_constructor (tree);
extern bool user_provided_p			(tree);
extern bool type_has_user_provided_constructor  (tree);
extern bool type_has_non_user_provided_default_constructor (tree);
extern bool vbase_has_user_provided_move_assign (tree);
extern tree default_init_uninitialized_part (tree);
extern bool trivial_default_constructor_is_constexpr (tree);
extern bool type_has_constexpr_default_constructor (tree);
extern bool type_has_virtual_destructor		(tree);
extern bool classtype_has_move_assign_or_move_ctor_p (tree, bool user_declared);
extern bool classtype_has_non_deleted_move_ctor (tree);
extern bool type_build_ctor_call		(tree);
extern bool type_build_dtor_call		(tree);
extern void explain_non_literal_class		(tree);
extern void inherit_targ_abi_tags		(tree);
extern void defaulted_late_check		(tree);
extern bool defaultable_fn_check		(tree);
extern void check_abi_tags			(tree);
extern tree missing_abi_tags			(tree);
extern void fixup_type_variants			(tree);
extern void fixup_attribute_variants		(tree);
extern tree* decl_cloned_function_p		(const_tree, bool);
extern void clone_function_decl			(tree, bool);
extern void adjust_clone_args			(tree);
extern void deduce_noexcept_on_destructor       (tree);
extern bool uniquely_derived_from_p             (tree, tree);
extern bool publicly_uniquely_derived_p         (tree, tree);
extern tree common_enclosing_class		(tree, tree);
extern tree convert_to_reference		(tree, tree, int, int, tree,
tsubst_flags_t);
extern tree convert_from_reference		(tree);
extern tree force_rvalue			(tree, tsubst_flags_t);
extern tree ocp_convert				(tree, tree, int, int,
tsubst_flags_t);
extern tree cp_convert				(tree, tree, tsubst_flags_t);
extern tree cp_convert_and_check                (tree, tree, tsubst_flags_t);
extern tree cp_fold_convert			(tree, tree);
extern tree cp_get_callee			(tree);
extern tree cp_get_callee_fndecl		(tree);
extern tree cp_get_callee_fndecl_nofold		(tree);
extern tree cp_get_fndecl_from_callee		(tree, bool fold = true);
extern tree convert_to_void			(tree, impl_conv_void,
tsubst_flags_t);
extern tree convert_force			(tree, tree, int,
tsubst_flags_t);
extern tree build_expr_type_conversion		(int, tree, bool);
extern tree type_promotes_to			(tree);
extern bool can_convert_qual			(tree, tree);
extern tree perform_qualification_conversions	(tree, tree);
extern bool tx_safe_fn_type_p			(tree);
extern tree tx_unsafe_fn_variant		(tree);
extern bool fnptr_conv_p			(tree, tree);
extern tree strip_fnptr_conv			(tree);
extern void maybe_push_cleanup_level		(tree);
extern tree make_anon_name			(void);
extern tree check_for_out_of_scope_variable	(tree);
extern void dump				(cp_binding_level &ref);
extern void dump				(cp_binding_level *ptr);
extern void print_other_binding_stack		(cp_binding_level *);
extern tree maybe_push_decl			(tree);
extern tree current_decl_namespace		(void);
extern tree poplevel				(int, int, int);
extern void cxx_init_decl_processing		(void);
enum cp_tree_node_structure_enum cp_tree_node_structure
(union lang_tree_node *);
extern void finish_scope			(void);
extern void push_switch				(tree);
extern void pop_switch				(void);
extern void note_break_stmt			(void);
extern bool note_iteration_stmt_body_start	(void);
extern void note_iteration_stmt_body_end	(bool);
extern tree make_lambda_name			(void);
extern int decls_match				(tree, tree, bool = true);
extern bool maybe_version_functions		(tree, tree, bool);
extern tree duplicate_decls			(tree, tree, bool);
extern tree declare_local_label			(tree);
extern tree define_label			(location_t, tree);
extern void check_goto				(tree);
extern bool check_omp_return			(void);
extern tree make_typename_type			(tree, tree, enum tag_types, tsubst_flags_t);
extern tree build_typename_type			(tree, tree, tree, tag_types);
extern tree make_unbound_class_template		(tree, tree, tree, tsubst_flags_t);
extern tree build_library_fn_ptr		(const char *, tree, int);
extern tree build_cp_library_fn_ptr		(const char *, tree, int);
extern tree push_library_fn			(tree, tree, tree, int);
extern tree push_void_library_fn		(tree, tree, int);
extern tree push_throw_library_fn		(tree, tree);
extern void warn_misplaced_attr_for_class_type  (source_location location,
tree class_type);
extern tree check_tag_decl			(cp_decl_specifier_seq *, bool);
extern tree shadow_tag				(cp_decl_specifier_seq *);
extern tree groktypename			(cp_decl_specifier_seq *, const cp_declarator *, bool);
extern tree start_decl				(const cp_declarator *, cp_decl_specifier_seq *, int, tree, tree, tree *);
extern void start_decl_1			(tree, bool);
extern bool check_array_initializer		(tree, tree, tree);
extern void cp_finish_decl			(tree, tree, bool, tree, int);
extern tree lookup_decomp_type			(tree);
extern void cp_maybe_mangle_decomp		(tree, tree, unsigned int);
extern void cp_finish_decomp			(tree, tree, unsigned int);
extern int cp_complete_array_type		(tree *, tree, bool);
extern int cp_complete_array_type_or_error	(tree *, tree, bool, tsubst_flags_t);
extern tree build_ptrmemfunc_type		(tree);
extern tree build_ptrmem_type			(tree, tree);
extern tree build_this_parm			(tree, tree, cp_cv_quals);
extern tree grokparms				(tree, tree *);
extern int copy_fn_p				(const_tree);
extern bool move_fn_p                           (const_tree);
extern bool move_signature_fn_p                 (const_tree);
extern tree get_scope_of_declarator		(const cp_declarator *);
extern void grok_special_member_properties	(tree);
extern bool grok_ctor_properties		(const_tree, const_tree);
extern bool grok_op_properties			(tree, bool);
extern tree xref_tag				(enum tag_types, tree, tag_scope, bool);
extern tree xref_tag_from_type			(tree, tree, tag_scope);
extern void xref_basetypes			(tree, tree);
extern tree start_enum				(tree, tree, tree, tree, bool, bool *);
extern void finish_enum_value_list		(tree);
extern void finish_enum				(tree);
extern void build_enumerator			(tree, tree, tree, tree, location_t);
extern tree lookup_enumerator			(tree, tree);
extern bool start_preparsed_function		(tree, tree, int);
extern bool start_function			(cp_decl_specifier_seq *,
const cp_declarator *, tree);
extern tree begin_function_body			(void);
extern void finish_function_body		(tree);
extern tree outer_curly_brace_block		(tree);
extern tree finish_function			(bool);
extern tree grokmethod				(cp_decl_specifier_seq *, const cp_declarator *, tree);
extern void maybe_register_incomplete_var	(tree);
extern void maybe_commonize_var			(tree);
extern void complete_vars			(tree);
extern tree static_fn_type			(tree);
extern void revert_static_member_fn		(tree);
extern void fixup_anonymous_aggr		(tree);
extern tree compute_array_index_type		(tree, tree, tsubst_flags_t);
extern tree check_default_argument		(tree, tree, tsubst_flags_t);
extern int wrapup_namespace_globals		();
extern tree create_implicit_typedef		(tree, tree);
extern int local_variable_p			(const_tree);
extern tree register_dtor_fn			(tree);
extern tmpl_spec_kind current_tmpl_spec_kind	(int);
extern tree cp_fname_init			(const char *, tree *);
extern tree cxx_builtin_function		(tree decl);
extern tree cxx_builtin_function_ext_scope	(tree decl);
extern tree check_elaborated_type_specifier	(enum tag_types, tree, bool);
extern void warn_extern_redeclared_static	(tree, tree);
extern tree cxx_comdat_group			(tree);
extern bool cp_missing_noreturn_ok_p		(tree);
extern bool is_direct_enum_init			(tree, tree);
extern void initialize_artificial_var		(tree, vec<constructor_elt, va_gc> *);
extern tree check_var_type			(tree, tree);
extern tree reshape_init                        (tree, tree, tsubst_flags_t);
extern tree next_initializable_field (tree);
extern tree fndecl_declared_return_type		(tree);
extern bool undeduced_auto_decl			(tree);
extern bool require_deduced_type		(tree, tsubst_flags_t = tf_warning_or_error);
extern tree finish_case_label			(location_t, tree, tree);
extern tree cxx_maybe_build_cleanup		(tree, tsubst_flags_t);
extern bool check_array_designated_initializer  (constructor_elt *,
unsigned HOST_WIDE_INT);
extern bool check_for_uninitialized_const_var   (tree, bool, tsubst_flags_t);
extern void record_mangling			(tree, bool);
extern void overwrite_mangling			(tree, tree);
extern void note_mangling_alias			(tree, tree);
extern void generate_mangling_aliases		(void);
extern tree build_memfn_type			(tree, tree, cp_cv_quals, cp_ref_qualifier);
extern tree build_pointer_ptrmemfn_type	(tree);
extern tree change_return_type			(tree, tree);
extern void maybe_retrofit_in_chrg		(tree);
extern void maybe_make_one_only			(tree);
extern bool vague_linkage_p			(tree);
extern void grokclassfn				(tree, tree,
enum overload_flags);
extern tree grok_array_decl			(location_t, tree, tree, bool);
extern tree delete_sanity			(tree, tree, bool, int, tsubst_flags_t);
extern tree check_classfn			(tree, tree, tree);
extern void check_member_template		(tree);
extern tree grokfield (const cp_declarator *, cp_decl_specifier_seq *,
tree, bool, tree, tree);
extern tree grokbitfield (const cp_declarator *, cp_decl_specifier_seq *,
tree, tree, tree);
extern bool any_dependent_type_attributes_p	(tree);
extern tree cp_reconstruct_complex_type		(tree, tree);
extern bool attributes_naming_typedef_ok	(tree);
extern void cplus_decl_attributes		(tree *, tree, int);
extern void finish_anon_union			(tree);
extern void cxx_post_compilation_parsing_cleanups (void);
extern tree coerce_new_type			(tree);
extern tree coerce_delete_type			(tree);
extern void comdat_linkage			(tree);
extern void determine_visibility		(tree);
extern void constrain_class_visibility		(tree);
extern void reset_type_linkage			(tree);
extern void tentative_decl_linkage		(tree);
extern void import_export_decl			(tree);
extern tree build_cleanup			(tree);
extern tree build_offset_ref_call_from_tree	(tree, vec<tree, va_gc> **,
tsubst_flags_t);
extern bool decl_defined_p			(tree);
extern bool decl_constant_var_p			(tree);
extern bool decl_maybe_constant_var_p		(tree);
extern void no_linkage_error			(tree);
extern void check_default_args			(tree);
extern bool mark_used				(tree);
extern bool mark_used			        (tree, tsubst_flags_t);
extern void finish_static_data_member_decl	(tree, tree, bool, tree, int);
extern tree cp_build_parm_decl			(tree, tree, tree);
extern tree get_guard				(tree);
extern tree get_guard_cond			(tree, bool);
extern tree set_guard				(tree);
extern tree get_tls_wrapper_fn			(tree);
extern void mark_needed				(tree);
extern bool decl_needed_p			(tree);
extern void note_vague_linkage_fn		(tree);
extern void note_variable_template_instantiation (tree);
extern tree build_artificial_parm		(tree, tree, tree);
extern bool possibly_inlined_p			(tree);
extern int parm_index                           (tree);
extern tree vtv_start_verification_constructor_init_function (void);
extern tree vtv_finish_verification_constructor_init_function (tree);
extern bool cp_omp_mappable_type		(tree);
extern const char *type_as_string		(tree, int);
extern const char *type_as_string_translate	(tree, int);
extern const char *decl_as_string		(tree, int);
extern const char *decl_as_string_translate	(tree, int);
extern const char *decl_as_dwarf_string		(tree, int);
extern const char *expr_as_string		(tree, int);
extern const char *lang_decl_name		(tree, int, bool);
extern const char *lang_decl_dwarf_name		(tree, int, bool);
extern const char *language_to_string		(enum languages);
extern const char *class_key_or_enum_as_string	(tree);
extern void maybe_warn_variadic_templates       (void);
extern void maybe_warn_cpp0x			(cpp0x_warn_str str);
extern bool pedwarn_cxx98                       (location_t, int, const char *, ...) ATTRIBUTE_GCC_DIAG(3,4);
extern location_t location_of                   (tree);
extern void qualified_name_lookup_error		(tree, tree, tree,
location_t);
extern void init_exception_processing		(void);
extern tree expand_start_catch_block		(tree);
extern void expand_end_catch_block		(void);
extern tree build_exc_ptr			(void);
extern tree build_throw				(tree);
extern int nothrow_libfn_p			(const_tree);
extern void check_handlers			(tree);
extern tree finish_noexcept_expr		(tree, tsubst_flags_t);
extern bool expr_noexcept_p			(tree, tsubst_flags_t);
extern void perform_deferred_noexcept_checks	(void);
extern bool nothrow_spec_p			(const_tree);
extern bool type_noexcept_p			(const_tree);
extern bool type_throw_all_p			(const_tree);
extern tree build_noexcept_spec			(tree, int);
extern void choose_personality_routine		(enum languages);
extern tree build_must_not_throw_expr		(tree,tree);
extern tree eh_type_info			(tree);
extern tree begin_eh_spec_block			(void);
extern void finish_eh_spec_block		(tree, tree);
extern tree build_eh_type_type			(tree);
extern tree cp_protect_cleanup_actions		(void);
extern tree create_try_catch_expr               (tree, tree);
extern tree cplus_expand_constant		(tree);
extern tree mark_use (tree expr, bool rvalue_p, bool read_p,
location_t = UNKNOWN_LOCATION,
bool reject_builtin = true);
extern tree mark_rvalue_use			(tree,
location_t = UNKNOWN_LOCATION,
bool reject_builtin = true);
extern tree mark_lvalue_use			(tree);
extern tree mark_lvalue_use_nonread		(tree);
extern tree mark_type_use			(tree);
extern tree mark_discarded_use			(tree);
extern void mark_exp_read			(tree);
extern int is_friend				(tree, tree);
extern void make_friend_class			(tree, tree, bool);
extern void add_friend				(tree, tree, bool);
extern tree do_friend				(tree, tree, tree, tree,
enum overload_flags, bool);
extern void set_global_friend			(tree);
extern bool is_global_friend			(tree);
extern tree expand_member_init			(tree);
extern void emit_mem_initializers		(tree);
extern tree build_aggr_init			(tree, tree, int,
tsubst_flags_t);
extern int is_class_type			(tree, int);
extern tree get_type_value			(tree);
extern tree build_zero_init			(tree, tree, bool);
extern tree build_value_init			(tree, tsubst_flags_t);
extern tree build_value_init_noctor		(tree, tsubst_flags_t);
extern tree get_nsdmi				(tree, bool, tsubst_flags_t);
extern tree build_offset_ref			(tree, tree, bool,
tsubst_flags_t);
extern tree throw_bad_array_new_length		(void);
extern bool type_has_new_extended_alignment	(tree);
extern unsigned malloc_alignment		(void);
extern tree build_new				(vec<tree, va_gc> **, tree, tree,
vec<tree, va_gc> **, int,
tsubst_flags_t);
extern tree get_temp_regvar			(tree, tree);
extern tree build_vec_init			(tree, tree, tree, bool, int,
tsubst_flags_t);
extern tree build_delete			(tree, tree,
special_function_kind,
int, int, tsubst_flags_t);
extern void push_base_cleanups			(void);
extern tree build_vec_delete			(tree, tree,
special_function_kind, int,
tsubst_flags_t);
extern tree create_temporary_var		(tree);
extern void initialize_vtbl_ptrs		(tree);
extern tree scalar_constant_value		(tree);
extern tree decl_really_constant_value		(tree);
extern int diagnose_uninitialized_cst_or_ref_member (tree, bool, bool);
extern tree build_vtbl_address                  (tree);
extern bool maybe_reject_flexarray_init		(tree, tree);
extern void cxx_dup_lang_specific_decl		(tree);
extern void yyungetc				(int, int);
extern tree unqualified_name_lookup_error	(tree,
location_t = UNKNOWN_LOCATION);
extern tree unqualified_fn_lookup_error		(cp_expr);
extern tree make_conv_op_name			(tree);
extern tree build_lang_decl			(enum tree_code, tree, tree);
extern tree build_lang_decl_loc			(location_t, enum tree_code, tree, tree);
extern void retrofit_lang_decl			(tree);
extern void fit_decomposition_lang_decl		(tree, tree);
extern tree copy_decl				(tree CXX_MEM_STAT_INFO);
extern tree copy_type				(tree CXX_MEM_STAT_INFO);
extern tree cxx_make_type			(enum tree_code);
extern tree make_class_type			(enum tree_code);
extern const char *get_identifier_kind_name	(tree);
extern void set_identifier_kind			(tree, cp_identifier_kind);
extern bool cxx_init				(void);
extern void cxx_finish				(void);
extern bool in_main_input_context		(void);
extern void init_method				(void);
extern tree make_thunk				(tree, bool, tree, tree);
extern void finish_thunk			(tree);
extern void use_thunk				(tree, bool);
extern bool trivial_fn_p			(tree);
extern tree forward_parm			(tree);
extern bool is_trivially_xible			(enum tree_code, tree, tree);
extern bool is_xible				(enum tree_code, tree, tree);
extern tree get_defaulted_eh_spec		(tree, tsubst_flags_t = tf_warning_or_error);
extern void after_nsdmi_defaulted_late_checks   (tree);
extern bool maybe_explain_implicit_delete	(tree);
extern void explain_implicit_non_constexpr	(tree);
extern void deduce_inheriting_ctor		(tree);
extern void synthesize_method			(tree);
extern tree lazily_declare_fn			(special_function_kind,
tree);
extern tree skip_artificial_parms_for		(const_tree, tree);
extern int num_artificial_parms_for		(const_tree);
extern tree make_alias_for			(tree, tree);
extern tree get_copy_ctor			(tree, tsubst_flags_t);
extern tree get_copy_assign			(tree);
extern tree get_default_ctor			(tree);
extern tree get_dtor				(tree, tsubst_flags_t);
extern tree strip_inheriting_ctors		(tree);
extern tree inherited_ctor_binfo		(tree);
extern bool ctor_omit_inherited_parms		(tree);
extern tree locate_ctor				(tree);
extern tree implicitly_declare_fn               (special_function_kind, tree,
bool, tree, tree);
extern bool maybe_clone_body			(tree);
extern tree cp_convert_range_for (tree, tree, tree, tree, unsigned int, bool,
unsigned short);
extern bool parsing_nsdmi (void);
extern bool parsing_default_capturing_generic_lambda_in_template (void);
extern void inject_this_parameter (tree, cp_cv_quals);
extern location_t defarg_location (tree);
extern void maybe_show_extern_c_location (void);
extern bool check_template_shadow		(tree);
extern bool check_auto_in_tmpl_args             (tree, tree);
extern tree get_innermost_template_args		(tree, int);
extern void maybe_begin_member_template_processing (tree);
extern void maybe_end_member_template_processing (void);
extern tree finish_member_template_decl		(tree);
extern void begin_template_parm_list		(void);
extern bool begin_specialization		(void);
extern void reset_specialization		(void);
extern void end_specialization			(void);
extern void begin_explicit_instantiation	(void);
extern void end_explicit_instantiation		(void);
extern void check_unqualified_spec_or_inst	(tree, location_t);
extern tree check_explicit_specialization	(tree, tree, int, int,
tree = NULL_TREE);
extern int num_template_headers_for_class	(tree);
extern void check_template_variable		(tree);
extern tree make_auto				(void);
extern tree make_decltype_auto			(void);
extern tree make_template_placeholder		(tree);
extern bool template_placeholder_p		(tree);
extern tree do_auto_deduction                   (tree, tree, tree,
tsubst_flags_t
= tf_warning_or_error,
auto_deduction_context
= adc_unspecified,
tree = NULL_TREE,
int = LOOKUP_NORMAL);
extern tree type_uses_auto			(tree);
extern tree type_uses_auto_or_concept		(tree);
extern void append_type_to_template_for_access_check (tree, tree, tree,
location_t);
extern tree convert_generic_types_to_packs	(tree, int, int);
extern tree splice_late_return_type		(tree, tree);
extern bool is_auto				(const_tree);
extern tree process_template_parm		(tree, location_t, tree, 
bool, bool);
extern tree end_template_parm_list		(tree);
extern void end_template_parm_list		(void);
extern void end_template_decl			(void);
extern tree maybe_update_decl_type		(tree, tree);
extern bool check_default_tmpl_args             (tree, tree, bool, bool, int);
extern tree push_template_decl			(tree);
extern tree push_template_decl_real		(tree, bool);
extern tree add_inherited_template_parms	(tree, tree);
extern bool redeclare_class_template		(tree, tree, tree);
extern tree lookup_template_class		(tree, tree, tree, tree,
int, tsubst_flags_t);
extern tree lookup_template_function		(tree, tree);
extern tree lookup_template_variable		(tree, tree);
extern int uses_template_parms			(tree);
extern bool uses_template_parms_level		(tree, int);
extern bool in_template_function		(void);
extern bool need_generic_capture		(void);
extern tree instantiate_class_template		(tree);
extern tree instantiate_template		(tree, tree, tsubst_flags_t);
extern tree fn_type_unification			(tree, tree, tree,
const tree *, unsigned int,
tree, unification_kind_t, int,
bool, bool);
extern void mark_decl_instantiated		(tree, int);
extern int more_specialized_fn			(tree, tree, int);
extern void do_decl_instantiation		(tree, tree);
extern void do_type_instantiation		(tree, tree, tsubst_flags_t);
extern bool always_instantiate_p		(tree);
extern bool maybe_instantiate_noexcept		(tree, tsubst_flags_t = tf_warning_or_error);
extern tree instantiate_decl			(tree, bool, bool);
extern int comp_template_parms			(const_tree, const_tree);
extern bool builtin_pack_fn_p			(tree);
extern bool uses_parameter_packs                (tree);
extern bool template_parameter_pack_p           (const_tree);
extern bool function_parameter_pack_p		(const_tree);
extern bool function_parameter_expanded_from_pack_p (tree, tree);
extern tree make_pack_expansion                 (tree, tsubst_flags_t = tf_warning_or_error);
extern bool check_for_bare_parameter_packs      (tree, location_t = UNKNOWN_LOCATION);
extern tree build_template_info			(tree, tree);
extern tree get_template_info			(const_tree);
extern vec<qualified_typedef_usage_t, va_gc> *get_types_needing_access_check (tree);
extern int template_class_depth			(tree);
extern int is_specialization_of			(tree, tree);
extern bool is_specialization_of_friend		(tree, tree);
extern tree get_pattern_parm			(tree, tree);
extern int comp_template_args			(tree, tree, tree * = NULL,
tree * = NULL, bool = false);
extern int template_args_equal                  (tree, tree, bool = false);
extern tree maybe_process_partial_specialization (tree);
extern tree most_specialized_instantiation	(tree);
extern void print_candidates			(tree);
extern void instantiate_pending_templates	(int);
extern tree tsubst_default_argument		(tree, int, tree, tree,
tsubst_flags_t);
extern tree tsubst (tree, tree, tsubst_flags_t, tree);
extern tree tsubst_copy_and_build		(tree, tree, tsubst_flags_t,
tree, bool, bool);
extern tree tsubst_expr                         (tree, tree, tsubst_flags_t,
tree, bool);
extern tree tsubst_pack_expansion               (tree, tree, tsubst_flags_t, tree);
extern tree most_general_template		(tree);
extern tree get_mostly_instantiated_function_type (tree);
extern bool problematic_instantiation_changed	(void);
extern void record_last_problematic_instantiation (void);
extern struct tinst_level *current_instantiation(void);
extern bool instantiating_current_function_p    (void);
extern tree maybe_get_template_decl_from_type_decl (tree);
extern int processing_template_parmlist;
extern bool dependent_type_p			(tree);
extern bool dependent_scope_p			(tree);
extern bool any_dependent_template_arguments_p  (const_tree);
extern bool any_erroneous_template_args_p       (const_tree);
extern bool dependent_template_p		(tree);
extern bool dependent_template_id_p		(tree, tree);
extern bool type_dependent_expression_p		(tree);
extern bool type_dependent_object_expression_p	(tree);
extern bool any_type_dependent_arguments_p      (const vec<tree, va_gc> *);
extern bool any_type_dependent_elements_p       (const_tree);
extern bool type_dependent_expression_p_push	(tree);
extern bool value_dependent_expression_p	(tree);
extern bool instantiation_dependent_expression_p (tree);
extern bool instantiation_dependent_uneval_expression_p (tree);
extern bool any_value_dependent_elements_p      (const_tree);
extern bool dependent_omp_for_p			(tree, tree, tree, tree);
extern tree resolve_typename_type		(tree, bool);
extern tree template_for_substitution		(tree);
extern tree build_non_dependent_expr		(tree);
extern void make_args_non_dependent		(vec<tree, va_gc> *);
extern bool reregister_specialization		(tree, tree, tree);
extern tree instantiate_non_dependent_expr	(tree);
extern tree instantiate_non_dependent_expr_sfinae (tree, tsubst_flags_t);
extern tree instantiate_non_dependent_expr_internal (tree, tsubst_flags_t);
extern tree instantiate_non_dependent_or_null   (tree);
extern bool variable_template_specialization_p  (tree);
extern bool alias_type_or_template_p            (tree);
extern bool alias_template_specialization_p     (const_tree);
extern bool dependent_alias_template_spec_p     (const_tree);
extern bool explicit_class_specialization_p     (tree);
extern bool push_tinst_level                    (tree);
extern bool push_tinst_level_loc                (tree, location_t);
extern void pop_tinst_level                     (void);
extern struct tinst_level *outermost_tinst_level(void);
extern void init_template_processing		(void);
extern void print_template_statistics		(void);
bool template_template_parameter_p		(const_tree);
bool template_type_parameter_p                  (const_tree);
extern bool primary_template_specialization_p   (const_tree);
extern tree get_primary_template_innermost_parameters	(const_tree);
extern tree get_template_parms_at_level (tree, int);
extern tree get_template_innermost_arguments	(const_tree);
extern tree get_template_argument_pack_elems	(const_tree);
extern tree get_function_template_decl		(const_tree);
extern tree resolve_nondeduced_context		(tree, tsubst_flags_t);
extern hashval_t iterative_hash_template_arg (tree arg, hashval_t val);
extern tree coerce_template_parms               (tree, tree, tree);
extern tree coerce_template_parms               (tree, tree, tree, tsubst_flags_t);
extern void register_local_specialization       (tree, tree);
extern tree retrieve_local_specialization       (tree);
extern tree extract_fnparm_pack                 (tree, tree *);
extern tree template_parm_to_arg                (tree);
extern tree dguide_name				(tree);
extern bool dguide_name_p			(tree);
extern bool deduction_guide_p			(const_tree);
extern bool copy_guide_p			(const_tree);
extern bool template_guide_p			(const_tree);
extern void init_repo				(void);
extern int repo_emit_p				(tree);
extern bool repo_export_class_p			(const_tree);
extern void finish_repo				(void);
extern GTY(()) vec<tree, va_gc> *unemitted_tinfo_decls;
extern void init_rtti_processing		(void);
extern tree build_typeid			(tree, tsubst_flags_t);
extern tree get_tinfo_decl			(tree);
extern tree get_typeid				(tree, tsubst_flags_t);
extern tree build_headof			(tree);
extern tree build_dynamic_cast			(tree, tree, tsubst_flags_t);
extern void emit_support_tinfos			(void);
extern bool emit_tinfo_decl			(tree);
extern bool accessible_base_p			(tree, tree, bool);
extern tree lookup_base                         (tree, tree, base_access,
base_kind *, tsubst_flags_t);
extern tree dcast_base_hint			(tree, tree);
extern int accessible_p				(tree, tree, bool);
extern int accessible_in_template_p		(tree, tree);
extern tree lookup_field			(tree, tree, int, bool);
extern tree lookup_fnfields			(tree, tree, int);
extern tree lookup_member			(tree, tree, int, bool,
tsubst_flags_t,
access_failure_info *afi = NULL);
extern tree lookup_member_fuzzy		(tree, tree, bool);
extern tree locate_field_accessor		(tree, tree, bool);
extern int look_for_overrides			(tree, tree);
extern void get_pure_virtuals			(tree);
extern void maybe_suppress_debug_info		(tree);
extern void note_debug_info_needed		(tree);
extern tree current_scope			(void);
extern int at_function_scope_p			(void);
extern bool at_class_scope_p			(void);
extern bool at_namespace_scope_p		(void);
extern tree context_for_name_lookup		(tree);
extern tree lookup_conversions			(tree);
extern tree binfo_from_vbase			(tree);
extern tree binfo_for_vbase			(tree, tree);
extern tree look_for_overrides_here		(tree, tree);
#define dfs_skip_bases ((tree)1)
extern tree dfs_walk_all (tree, tree (*) (tree, void *),
tree (*) (tree, void *), void *);
extern tree dfs_walk_once (tree, tree (*) (tree, void *),
tree (*) (tree, void *), void *);
extern tree binfo_via_virtual			(tree, tree);
extern bool binfo_direct_p			(tree);
extern tree build_baselink			(tree, tree, tree, tree);
extern tree adjust_result_of_qualified_name_lookup
(tree, tree, tree);
extern tree copied_binfo			(tree, tree);
extern tree original_binfo			(tree, tree);
extern int shared_member_p			(tree);
extern bool any_dependent_bases_p (tree = current_nonlambda_class_type ());
struct GTY(()) deferred_access_check {
tree binfo;
tree decl;
tree diag_decl;
location_t loc;
};
extern void push_deferring_access_checks	(deferring_kind);
extern void resume_deferring_access_checks	(void);
extern void stop_deferring_access_checks	(void);
extern void pop_deferring_access_checks		(void);
extern vec<deferred_access_check, va_gc> *get_deferred_access_checks (void);
extern void reopen_deferring_access_checks (vec<deferred_access_check, va_gc> *);
extern void pop_to_parent_deferring_access_checks (void);
extern bool perform_access_checks (vec<deferred_access_check, va_gc> *,
tsubst_flags_t);
extern bool perform_deferred_access_checks	(tsubst_flags_t);
extern bool perform_or_defer_access_check	(tree, tree, tree,
tsubst_flags_t,
access_failure_info *afi = NULL);
struct deferring_access_check_sentinel
{
deferring_access_check_sentinel (enum deferring_kind kind = dk_deferred)
{
push_deferring_access_checks (kind);
}
~deferring_access_check_sentinel ()
{
pop_deferring_access_checks ();
}
};
extern int stmts_are_full_exprs_p		(void);
extern void init_cp_semantics			(void);
extern tree do_poplevel				(tree);
extern void break_maybe_infinite_loop		(void);
extern void add_decl_expr			(tree);
extern tree maybe_cleanup_point_expr_void	(tree);
extern tree finish_expr_stmt			(tree);
extern tree begin_if_stmt			(void);
extern tree finish_if_stmt_cond			(tree, tree);
extern tree finish_then_clause			(tree);
extern void begin_else_clause			(tree);
extern void finish_else_clause			(tree);
extern void finish_if_stmt			(tree);
extern tree begin_while_stmt			(void);
extern void finish_while_stmt_cond	(tree, tree, bool, unsigned short);
extern void finish_while_stmt			(tree);
extern tree begin_do_stmt			(void);
extern void finish_do_body			(tree);
extern void finish_do_stmt		(tree, tree, bool, unsigned short);
extern tree finish_return_stmt			(tree);
extern tree begin_for_scope			(tree *);
extern tree begin_for_stmt			(tree, tree);
extern void finish_init_stmt			(tree);
extern void finish_for_cond		(tree, tree, bool, unsigned short);
extern void finish_for_expr			(tree, tree);
extern void finish_for_stmt			(tree);
extern tree begin_range_for_stmt		(tree, tree);
extern void finish_range_for_decl		(tree, tree, tree);
extern void finish_range_for_stmt		(tree);
extern tree finish_break_stmt			(void);
extern tree finish_continue_stmt		(void);
extern tree begin_switch_stmt			(void);
extern void finish_switch_cond			(tree, tree);
extern void finish_switch_stmt			(tree);
extern tree finish_goto_stmt			(tree);
extern tree begin_try_block			(void);
extern void finish_try_block			(tree);
extern void finish_handler_sequence		(tree);
extern tree begin_function_try_block		(tree *);
extern void finish_function_try_block		(tree);
extern void finish_function_handler_sequence    (tree, tree);
extern void finish_cleanup_try_block		(tree);
extern tree begin_handler			(void);
extern void finish_handler_parms		(tree, tree);
extern void finish_handler			(tree);
extern void finish_cleanup			(tree, tree);
extern bool is_this_parameter                   (tree);
enum {
BCS_NORMAL = 0,
BCS_NO_SCOPE = 1,
BCS_TRY_BLOCK = 2,
BCS_FN_BODY = 4,
BCS_TRANSACTION = 8
};
extern tree begin_compound_stmt			(unsigned int);
extern void finish_compound_stmt		(tree);
extern tree finish_asm_stmt			(int, tree, tree, tree, tree,
tree, bool);
extern tree finish_label_stmt			(tree);
extern void finish_label_decl			(tree);
extern cp_expr finish_parenthesized_expr	(cp_expr);
extern tree force_paren_expr			(tree);
extern tree maybe_undo_parenthesized_ref	(tree);
extern tree finish_non_static_data_member       (tree, tree, tree);
extern tree begin_stmt_expr			(void);
extern tree finish_stmt_expr_expr		(tree, tree);
extern tree finish_stmt_expr			(tree, bool);
extern tree stmt_expr_value_expr		(tree);
bool empty_expr_stmt_p				(tree);
extern cp_expr perform_koenig_lookup		(cp_expr, vec<tree, va_gc> *,
tsubst_flags_t);
extern tree finish_call_expr			(tree, vec<tree, va_gc> **, bool,
bool, tsubst_flags_t);
extern tree lookup_and_finish_template_variable (tree, tree, tsubst_flags_t = tf_warning_or_error);
extern tree finish_template_variable		(tree, tsubst_flags_t = tf_warning_or_error);
extern cp_expr finish_increment_expr		(cp_expr, enum tree_code);
extern tree finish_this_expr			(void);
extern tree finish_pseudo_destructor_expr       (tree, tree, tree, location_t);
extern cp_expr finish_unary_op_expr		(location_t, enum tree_code, cp_expr,
tsubst_flags_t);
enum fcl_t { fcl_functional, fcl_c99 };
extern tree finish_compound_literal		(tree, tree, tsubst_flags_t, fcl_t = fcl_functional);
extern tree finish_fname			(tree);
extern void finish_translation_unit		(void);
extern tree finish_template_type_parm		(tree, tree);
extern tree finish_template_template_parm       (tree, tree);
extern tree begin_class_definition		(tree);
extern void finish_template_decl		(tree);
extern tree finish_template_type		(tree, tree, int);
extern tree finish_base_specifier		(tree, tree, bool);
extern void finish_member_declaration		(tree);
extern bool outer_automatic_var_p		(tree);
extern tree process_outer_var_ref		(tree, tsubst_flags_t, bool force_use = false);
extern cp_expr finish_id_expression		(tree, tree, tree,
cp_id_kind *,
bool, bool, bool *,
bool, bool, bool, bool,
const char **,
location_t);
extern tree finish_typeof			(tree);
extern tree finish_underlying_type	        (tree);
extern tree calculate_bases                     (tree, tsubst_flags_t);
extern tree finish_bases                        (tree, bool);
extern tree calculate_direct_bases              (tree, tsubst_flags_t);
extern tree finish_offsetof			(tree, tree, location_t);
extern void finish_decl_cleanup			(tree, tree);
extern void finish_eh_cleanup			(tree);
extern void emit_associated_thunks		(tree);
extern void finish_mem_initializers		(tree);
extern tree check_template_template_default_arg (tree);
extern bool expand_or_defer_fn_1		(tree);
extern void expand_or_defer_fn			(tree);
extern void add_typedef_to_current_template_for_access_check (tree, tree,
location_t);
extern void check_accessibility_of_qualified_id (tree, tree, tree);
extern tree finish_qualified_id_expr		(tree, tree, bool, bool,
bool, bool, tsubst_flags_t);
extern void simplify_aggr_init_expr		(tree *);
extern void finalize_nrv			(tree *, tree, tree);
extern tree omp_reduction_id			(enum tree_code, tree, tree);
extern tree cp_remove_omp_priv_cleanup_stmt	(tree *, int *, void *);
extern void cp_check_omp_declare_reduction	(tree);
extern void finish_omp_declare_simd_methods	(tree);
extern tree finish_omp_clauses			(tree, enum c_omp_region_type);
extern tree push_omp_privatization_clauses	(bool);
extern void pop_omp_privatization_clauses	(tree);
extern void save_omp_privatization_clauses	(vec<tree> &);
extern void restore_omp_privatization_clauses	(vec<tree> &);
extern void finish_omp_threadprivate		(tree);
extern tree begin_omp_structured_block		(void);
extern tree finish_omp_structured_block		(tree);
extern tree finish_oacc_data			(tree, tree);
extern tree finish_oacc_host_data		(tree, tree);
extern tree finish_omp_construct		(enum tree_code, tree, tree);
extern tree begin_omp_parallel			(void);
extern tree finish_omp_parallel			(tree, tree);
extern tree begin_omp_task			(void);
extern tree finish_omp_task			(tree, tree);
extern tree finish_omp_for			(location_t, enum tree_code,
tree, tree, tree, tree, tree,
tree, tree, vec<tree> *, tree);
extern void finish_omp_atomic			(enum tree_code, enum tree_code,
tree, tree, tree, tree, tree,
bool);
extern void finish_omp_barrier			(void);
extern void finish_omp_flush			(void);
extern void finish_omp_taskwait			(void);
extern void finish_omp_taskyield		(void);
extern void finish_omp_cancel			(tree);
extern void finish_omp_cancellation_point	(tree);
extern tree omp_privatize_field			(tree, bool);
extern tree begin_transaction_stmt		(location_t, tree *, int);
extern void finish_transaction_stmt		(tree, tree, int, tree);
extern tree build_transaction_expr		(location_t, tree, int, tree);
extern bool cxx_omp_create_clause_info		(tree, tree, bool, bool,
bool, bool);
extern tree baselink_for_fns                    (tree);
extern void finish_static_assert                (tree, tree, location_t,
bool);
extern tree finish_decltype_type                (tree, bool, tsubst_flags_t);
extern tree finish_trait_expr			(enum cp_trait_kind, tree, tree);
extern tree build_lambda_expr                   (void);
extern tree build_lambda_object			(tree);
extern tree begin_lambda_type                   (tree);
extern tree lambda_capture_field_type		(tree, bool, bool);
extern tree lambda_return_type			(tree);
extern tree lambda_proxy_type			(tree);
extern tree lambda_function			(tree);
extern void apply_deduced_return_type           (tree, tree);
extern tree add_capture                         (tree, tree, tree, bool, bool);
extern tree add_default_capture                 (tree, tree, tree);
extern void insert_capture_proxy		(tree);
extern void insert_pending_capture_proxies	(void);
extern bool is_capture_proxy			(tree);
extern bool is_normal_capture_proxy             (tree);
extern bool is_constant_capture_proxy           (tree);
extern void register_capture_members		(tree);
extern tree lambda_expr_this_capture            (tree, bool);
extern void maybe_generic_this_capture		(tree, tree);
extern tree maybe_resolve_dummy			(tree, bool);
extern tree current_nonlambda_function		(void);
extern tree nonlambda_method_basetype		(void);
extern tree current_nonlambda_scope		(void);
extern tree current_lambda_expr			(void);
extern bool generic_lambda_fn_p			(tree);
extern tree do_dependent_capture		(tree, bool = false);
extern bool lambda_fn_in_template_p		(tree);
extern void maybe_add_lambda_conv_op            (tree);
extern bool is_lambda_ignored_entity            (tree);
extern bool lambda_static_thunk_p		(tree);
extern tree finish_builtin_launder		(location_t, tree,
tsubst_flags_t);
extern void start_lambda_scope			(tree);
extern void record_lambda_scope			(tree);
extern void record_null_lambda_scope		(tree);
extern void finish_lambda_scope			(void);
extern tree start_lambda_function		(tree fn, tree lambda_expr);
extern void finish_lambda_function		(tree body);
extern int cp_tree_operand_length		(const_tree);
extern int cp_tree_code_length			(enum tree_code);
extern void cp_free_lang_data 			(tree t);
extern tree force_target_expr			(tree, tree, tsubst_flags_t);
extern tree build_target_expr_with_type		(tree, tree, tsubst_flags_t);
extern void lang_check_failed			(const char *, int,
const char *) ATTRIBUTE_NORETURN
ATTRIBUTE_COLD;
extern tree stabilize_expr			(tree, tree *);
extern void stabilize_call			(tree, tree *);
extern bool stabilize_init			(tree, tree *);
extern tree add_stmt_to_compound		(tree, tree);
extern void init_tree				(void);
extern bool pod_type_p				(const_tree);
extern bool layout_pod_type_p			(const_tree);
extern bool std_layout_type_p			(const_tree);
extern bool trivial_type_p			(const_tree);
extern bool trivially_copyable_p		(const_tree);
extern bool type_has_unique_obj_representations (const_tree);
extern bool scalarish_type_p			(const_tree);
extern bool type_has_nontrivial_default_init	(const_tree);
extern bool type_has_nontrivial_copy_init	(const_tree);
extern void maybe_warn_parm_abi			(tree, location_t);
extern bool class_tmpl_impl_spec_p		(const_tree);
extern int zero_init_p				(const_tree);
extern bool check_abi_tag_redeclaration		(const_tree, const_tree,
const_tree);
extern bool check_abi_tag_args			(tree, tree);
extern tree strip_typedefs			(tree, bool * = NULL);
extern tree strip_typedefs_expr			(tree, bool * = NULL);
extern tree copy_binfo				(tree, tree, tree,
tree *, int);
extern int member_p				(const_tree);
extern cp_lvalue_kind real_lvalue_p		(const_tree);
extern cp_lvalue_kind lvalue_kind		(const_tree);
extern bool glvalue_p				(const_tree);
extern bool obvalue_p				(const_tree);
extern bool xvalue_p	                        (const_tree);
extern bool bitfield_p				(const_tree);
extern tree cp_stabilize_reference		(tree);
extern bool builtin_valid_in_constant_expr_p    (const_tree);
extern tree build_min				(enum tree_code, tree, ...);
extern tree build_min_nt_loc			(location_t, enum tree_code,
...);
extern tree build_min_non_dep			(enum tree_code, tree, ...);
extern tree build_min_non_dep_op_overload	(enum tree_code, tree, tree, ...);
extern tree build_min_nt_call_vec (tree, vec<tree, va_gc> *);
extern tree build_min_non_dep_call_vec		(tree, tree, vec<tree, va_gc> *);
extern vec<tree, va_gc>* vec_copy_and_insert    (vec<tree, va_gc>*, tree, unsigned);
extern tree build_cplus_new			(tree, tree, tsubst_flags_t);
extern tree build_aggr_init_expr		(tree, tree);
extern tree get_target_expr			(tree);
extern tree get_target_expr_sfinae		(tree, tsubst_flags_t);
extern tree build_cplus_array_type		(tree, tree);
extern tree build_array_of_n_type		(tree, int);
extern bool array_of_runtime_bound_p		(tree);
extern bool vla_type_p				(tree);
extern tree build_array_copy			(tree);
extern tree build_vec_init_expr			(tree, tree, tsubst_flags_t);
extern void diagnose_non_constexpr_vec_init	(tree);
extern tree hash_tree_cons			(tree, tree, tree);
extern tree hash_tree_chain			(tree, tree);
extern tree build_qualified_name		(tree, tree, tree, bool);
extern tree build_ref_qualified_type		(tree, cp_ref_qualifier);
inline tree ovl_first				(tree) ATTRIBUTE_PURE;
extern tree ovl_make				(tree fn,
tree next = NULL_TREE);
extern tree ovl_insert				(tree fn, tree maybe_ovl,
bool using_p = false);
extern tree ovl_skip_hidden			(tree) ATTRIBUTE_PURE;
extern void lookup_mark				(tree lookup, bool val);
extern tree lookup_add				(tree fns, tree lookup);
extern tree lookup_maybe_add			(tree fns, tree lookup,
bool deduping);
extern void lookup_keep				(tree lookup, bool keep);
extern void lookup_list_keep			(tree list, bool keep);
extern int is_overloaded_fn			(tree) ATTRIBUTE_PURE;
extern bool really_overloaded_fn		(tree) ATTRIBUTE_PURE;
extern tree dependent_name			(tree);
extern tree get_fns				(tree) ATTRIBUTE_PURE;
extern tree get_first_fn			(tree) ATTRIBUTE_PURE;
extern tree ovl_scope				(tree);
extern const char *cxx_printable_name		(tree, int);
extern const char *cxx_printable_name_translate	(tree, int);
extern tree canonical_eh_spec			(tree);
extern tree build_exception_variant		(tree, tree);
extern tree bind_template_template_parm		(tree, tree);
extern tree array_type_nelts_total		(tree);
extern tree array_type_nelts_top		(tree);
extern tree break_out_target_exprs		(tree, bool = false);
extern tree build_ctor_subob_ref		(tree, tree, tree);
extern tree replace_placeholders		(tree, tree, bool * = NULL);
extern bool find_placeholders			(tree);
extern tree get_type_decl			(tree);
extern tree decl_namespace_context		(tree);
extern bool decl_anon_ns_mem_p			(const_tree);
extern tree lvalue_type				(tree);
extern tree error_type				(tree);
extern int varargs_function_p			(const_tree);
extern bool cp_tree_equal			(tree, tree);
extern tree no_linkage_check			(tree, bool);
extern void debug_binfo				(tree);
extern tree build_dummy_object			(tree);
extern tree maybe_dummy_object			(tree, tree *);
extern int is_dummy_object			(const_tree);
extern const struct attribute_spec cxx_attribute_table[];
extern tree make_ptrmem_cst			(tree, tree);
extern tree cp_build_type_attribute_variant     (tree, tree);
extern tree cp_build_reference_type		(tree, bool);
extern tree move				(tree);
extern tree cp_build_qualified_type_real	(tree, int, tsubst_flags_t);
#define cp_build_qualified_type(TYPE, QUALS) \
cp_build_qualified_type_real ((TYPE), (QUALS), tf_warning_or_error)
extern bool cv_qualified_p			(const_tree);
extern tree cv_unqualified			(tree);
extern special_function_kind special_function_p (const_tree);
extern int count_trees				(tree);
extern int char_type_p				(tree);
extern void verify_stmt_tree			(tree);
extern linkage_kind decl_linkage		(tree);
extern duration_kind decl_storage_duration	(tree);
extern tree cp_walk_subtrees (tree*, int*, walk_tree_fn,
void*, hash_set<tree> *);
#define cp_walk_tree(tp,func,data,pset) \
walk_tree_1 (tp, func, data, pset, cp_walk_subtrees)
#define cp_walk_tree_without_duplicates(tp,func,data) \
walk_tree_without_duplicates_1 (tp, func, data, cp_walk_subtrees)
extern tree rvalue				(tree);
extern tree convert_bitfield_to_declared_type   (tree);
extern tree cp_save_expr			(tree);
extern bool cast_valid_in_integral_constant_expression_p (tree);
extern bool cxx_type_hash_eq			(const_tree, const_tree);
extern tree cxx_copy_lang_qualifiers		(const_tree, const_tree);
extern void cxx_print_statistics		(void);
extern bool maybe_warn_zero_as_null_pointer_constant (tree, location_t);
extern void cp_warn_deprecated_use		(tree);
extern void cxx_print_xnode			(FILE *, tree, int);
extern void cxx_print_decl			(FILE *, tree, int);
extern void cxx_print_type			(FILE *, tree, int);
extern void cxx_print_identifier		(FILE *, tree, int);
extern void cxx_print_error_function		(diagnostic_context *,
const char *,
struct diagnostic_info *);
extern bool cxx_mark_addressable		(tree, bool = false);
extern int string_conv_p			(const_tree, const_tree, int);
extern tree cp_truthvalue_conversion		(tree);
extern tree condition_conversion		(tree);
extern tree require_complete_type		(tree);
extern tree require_complete_type_sfinae	(tree, tsubst_flags_t);
extern tree complete_type			(tree);
extern tree complete_type_or_else		(tree, tree);
extern tree complete_type_or_maybe_complain	(tree, tree, tsubst_flags_t);
inline bool type_unknown_p			(const_tree);
enum { ce_derived, ce_type, ce_normal, ce_exact };
extern bool comp_except_specs			(const_tree, const_tree, int);
extern bool comptypes				(tree, tree, int);
extern bool same_type_ignoring_top_level_qualifiers_p (tree, tree);
extern bool compparms				(const_tree, const_tree);
extern int comp_cv_qualification		(const_tree, const_tree);
extern int comp_cv_qualification		(int, int);
extern int comp_cv_qual_signature		(tree, tree);
extern tree cxx_sizeof_or_alignof_expr		(tree, enum tree_code, bool);
extern tree cxx_sizeof_or_alignof_type		(tree, enum tree_code, bool, bool);
extern tree cxx_alignas_expr                    (tree);
extern tree cxx_sizeof_nowarn                   (tree);
extern tree is_bitfield_expr_with_lowered_type  (const_tree);
extern tree unlowered_expr_type                 (const_tree);
extern tree decay_conversion			(tree,
tsubst_flags_t,
bool = true);
extern tree build_class_member_access_expr      (cp_expr, tree, tree, bool,
tsubst_flags_t);
extern tree finish_class_member_access_expr     (cp_expr, tree, bool,
tsubst_flags_t);
extern tree build_x_indirect_ref		(location_t, tree,
ref_operator, tsubst_flags_t);
extern tree cp_build_indirect_ref		(tree, ref_operator,
tsubst_flags_t);
extern tree cp_build_fold_indirect_ref		(tree);
extern tree build_array_ref			(location_t, tree, tree);
extern tree cp_build_array_ref			(location_t, tree, tree,
tsubst_flags_t);
extern tree get_member_function_from_ptrfunc	(tree *, tree, tsubst_flags_t);
extern tree cp_build_function_call_nary         (tree, tsubst_flags_t, ...)
ATTRIBUTE_SENTINEL;
extern tree cp_build_function_call_vec		(tree, vec<tree, va_gc> **,
tsubst_flags_t);
extern tree build_x_binary_op			(location_t,
enum tree_code, tree,
enum tree_code, tree,
enum tree_code, tree *,
tsubst_flags_t);
extern tree build_x_array_ref			(location_t, tree, tree,
tsubst_flags_t);
extern tree build_x_unary_op			(location_t,
enum tree_code, cp_expr,
tsubst_flags_t);
extern tree cp_build_addressof			(location_t, tree,
tsubst_flags_t);
extern tree cp_build_addr_expr			(tree, tsubst_flags_t);
extern tree cp_build_unary_op                   (enum tree_code, tree, bool,
tsubst_flags_t);
extern tree genericize_compound_lvalue		(tree);
extern tree unary_complex_lvalue		(enum tree_code, tree);
extern tree build_x_conditional_expr		(location_t, tree, tree, tree, 
tsubst_flags_t);
extern tree build_x_compound_expr_from_list	(tree, expr_list_kind,
tsubst_flags_t);
extern tree build_x_compound_expr_from_vec	(vec<tree, va_gc> *,
const char *, tsubst_flags_t);
extern tree build_x_compound_expr		(location_t, tree, tree,
tsubst_flags_t);
extern tree build_compound_expr                 (location_t, tree, tree);
extern tree cp_build_compound_expr		(tree, tree, tsubst_flags_t);
extern tree build_static_cast			(tree, tree, tsubst_flags_t);
extern tree build_reinterpret_cast		(tree, tree, tsubst_flags_t);
extern tree build_const_cast			(tree, tree, tsubst_flags_t);
extern tree build_c_cast			(location_t, tree, tree);
extern cp_expr build_c_cast			(location_t loc, tree type,
cp_expr expr);
extern tree cp_build_c_cast			(tree, tree, tsubst_flags_t);
extern cp_expr build_x_modify_expr		(location_t, tree,
enum tree_code, tree,
tsubst_flags_t);
extern tree cp_build_modify_expr		(location_t, tree,
enum tree_code, tree,
tsubst_flags_t);
extern tree convert_for_initialization		(tree, tree, tree, int,
impl_conv_rhs, tree, int,
tsubst_flags_t);
extern int comp_ptr_ttypes			(tree, tree);
extern bool comp_ptr_ttypes_const		(tree, tree);
extern bool error_type_p			(const_tree);
extern bool ptr_reasonably_similar		(const_tree, const_tree);
extern tree build_ptrmemfunc			(tree, tree, int, bool,
tsubst_flags_t);
extern int cp_type_quals			(const_tree);
extern int type_memfn_quals			(const_tree);
extern cp_ref_qualifier type_memfn_rqual	(const_tree);
extern tree apply_memfn_quals			(tree, cp_cv_quals, cp_ref_qualifier);
extern bool cp_has_mutable_p			(const_tree);
extern bool at_least_as_qualified_p		(const_tree, const_tree);
extern void cp_apply_type_quals_to_decl		(int, tree);
extern tree build_ptrmemfunc1			(tree, tree, tree);
extern void expand_ptrmemfunc_cst		(tree, tree *, tree *);
extern tree type_after_usual_arithmetic_conversions (tree, tree);
extern tree common_pointer_type                 (tree, tree);
extern tree composite_pointer_type		(tree, tree, tree, tree,
composite_pointer_operation, 
tsubst_flags_t);
extern tree merge_types				(tree, tree);
extern tree strip_array_domain			(tree);
extern tree check_return_expr			(tree, bool *);
extern tree cp_build_binary_op                  (location_t,
enum tree_code, tree, tree,
tsubst_flags_t);
extern tree build_x_vec_perm_expr               (location_t,
tree, tree, tree,
tsubst_flags_t);
#define cxx_sizeof(T)  cxx_sizeof_or_alignof_type (T, SIZEOF_EXPR, false, true)
extern tree build_simple_component_ref		(tree, tree);
extern tree build_ptrmemfunc_access_expr	(tree, tree);
extern tree build_address			(tree);
extern tree build_nop				(tree, tree);
extern tree non_reference			(tree);
extern tree lookup_anon_field			(tree, tree);
extern bool invalid_nonstatic_memfn_p		(location_t, tree,
tsubst_flags_t);
extern tree convert_member_func_to_ptr		(tree, tree, tsubst_flags_t);
extern tree convert_ptrmem			(tree, tree, bool, bool,
tsubst_flags_t);
extern int lvalue_or_else			(tree, enum lvalue_use,
tsubst_flags_t);
extern void check_template_keyword		(tree);
extern bool check_raw_literal_operator		(const_tree decl);
extern bool check_literal_operator_args		(const_tree, bool *, bool *);
extern void maybe_warn_about_useless_cast       (tree, tree, tsubst_flags_t);
extern tree cp_perform_integral_promotions      (tree, tsubst_flags_t);
extern tree finish_left_unary_fold_expr      (tree, int);
extern tree finish_right_unary_fold_expr     (tree, int);
extern tree finish_binary_fold_expr          (tree, tree, int);
extern void require_complete_eh_spec_types	(tree, tree);
extern void cxx_incomplete_type_diagnostic	(location_t, const_tree,
const_tree, diagnostic_t);
inline void
cxx_incomplete_type_diagnostic (const_tree value, const_tree type,
diagnostic_t diag_kind)
{
cxx_incomplete_type_diagnostic (EXPR_LOC_OR_LOC (value, input_location),
value, type, diag_kind);
}
extern void cxx_incomplete_type_error		(location_t, const_tree,
const_tree);
inline void
cxx_incomplete_type_error (const_tree value, const_tree type)
{
cxx_incomplete_type_diagnostic (value, type, DK_ERROR);
}
extern void cxx_incomplete_type_inform 	        (const_tree);
extern tree error_not_base_type			(tree, tree);
extern tree binfo_or_else			(tree, tree);
extern void cxx_readonly_error			(tree, enum lvalue_use);
extern void complete_type_check_abstract	(tree);
extern int abstract_virtuals_error		(tree, tree);
extern int abstract_virtuals_error		(abstract_class_use, tree);
extern int abstract_virtuals_error_sfinae	(tree, tree, tsubst_flags_t);
extern int abstract_virtuals_error_sfinae	(abstract_class_use, tree, tsubst_flags_t);
extern tree store_init_value			(tree, tree, vec<tree, va_gc>**, int);
extern tree split_nonconstant_init		(tree, tree);
extern bool check_narrowing			(tree, tree, tsubst_flags_t);
extern tree digest_init				(tree, tree, tsubst_flags_t);
extern tree digest_init_flags			(tree, tree, int, tsubst_flags_t);
extern tree digest_nsdmi_init		        (tree, tree, tsubst_flags_t);
extern tree build_scoped_ref			(tree, tree, tree *);
extern tree build_x_arrow			(location_t, tree,
tsubst_flags_t);
extern tree build_m_component_ref		(tree, tree, tsubst_flags_t);
extern tree build_functional_cast		(tree, tree, tsubst_flags_t);
extern tree add_exception_specifier		(tree, tree, int);
extern tree merge_exception_specifiers		(tree, tree);
extern void init_mangle				(void);
extern void mangle_decl				(tree);
extern const char *mangle_type_string		(tree);
extern tree mangle_typeinfo_for_type		(tree);
extern tree mangle_typeinfo_string_for_type	(tree);
extern tree mangle_vtbl_for_type		(tree);
extern tree mangle_vtt_for_type			(tree);
extern tree mangle_ctor_vtbl_for_type		(tree, tree);
extern tree mangle_thunk			(tree, int, tree, tree, tree);
extern tree mangle_guard_variable		(tree);
extern tree mangle_tls_init_fn			(tree);
extern tree mangle_tls_wrapper_fn		(tree);
extern bool decl_tls_wrapper_p			(tree);
extern tree mangle_ref_init_variable		(tree);
extern char * get_mangled_vtable_map_var_name   (tree);
extern bool mangle_return_type_p		(tree);
extern tree mangle_decomp			(tree, vec<tree> &);
extern bool cp_dump_tree			(void *, tree);
extern alias_set_type cxx_get_alias_set		(tree);
extern bool cxx_warn_unused_global_decl		(const_tree);
extern size_t cp_tree_size			(enum tree_code);
extern bool cp_var_mod_type_p			(tree, tree);
extern void cxx_initialize_diagnostics		(diagnostic_context *);
extern int cxx_types_compatible_p		(tree, tree);
extern void init_shadowed_var_for_decl		(void);
extern bool cxx_block_may_fallthru		(const_tree);
extern int cp_gimplify_expr			(tree *, gimple_seq *,
gimple_seq *);
extern void cp_genericize			(tree);
extern bool cxx_omp_const_qual_no_mutable	(tree);
extern enum omp_clause_default_kind cxx_omp_predetermined_sharing_1 (tree);
extern enum omp_clause_default_kind cxx_omp_predetermined_sharing (tree);
extern tree cxx_omp_clause_default_ctor		(tree, tree, tree);
extern tree cxx_omp_clause_copy_ctor		(tree, tree, tree);
extern tree cxx_omp_clause_assign_op		(tree, tree, tree);
extern tree cxx_omp_clause_dtor			(tree, tree);
extern void cxx_omp_finish_clause		(tree, gimple_seq *);
extern bool cxx_omp_privatize_by_reference	(const_tree);
extern bool cxx_omp_disregard_value_expr	(tree, bool);
extern void cp_fold_function			(tree);
extern tree cp_fully_fold			(tree);
extern void clear_fold_cache			(void);
extern void suggest_alternatives_for            (location_t, tree, bool);
extern bool suggest_alternative_in_explicit_scope (location_t, tree, tree);
extern tree strip_using_decl                    (tree);
enum cp_oracle_request
{
CP_ORACLE_IDENTIFIER
};
typedef void cp_binding_oracle_function (enum cp_oracle_request, tree identifier);
extern cp_binding_oracle_function *cp_binding_oracle;
extern void init_constraint_processing          ();
extern bool constraint_p                        (tree);
extern tree conjoin_constraints                 (tree, tree);
extern tree conjoin_constraints                 (tree);
extern tree get_constraints                     (tree);
extern void set_constraints                     (tree, tree);
extern void remove_constraints                  (tree);
extern tree current_template_constraints	(void);
extern tree associate_classtype_constraints     (tree);
extern tree build_constraints                   (tree, tree);
extern tree get_shorthand_constraints           (tree);
extern tree build_concept_check                 (tree, tree, tree = NULL_TREE);
extern tree build_constrained_parameter         (tree, tree, tree = NULL_TREE);
extern tree make_constrained_auto               (tree, tree);
extern void placeholder_extract_concept_and_args (tree, tree&, tree&);
extern bool equivalent_placeholder_constraints  (tree, tree);
extern hashval_t hash_placeholder_constraint	(tree);
extern bool deduce_constrained_parameter        (tree, tree&, tree&);
extern tree resolve_constraint_check            (tree);
extern tree check_function_concept              (tree);
extern tree finish_template_introduction        (tree, tree);
extern bool valid_requirements_p                (tree);
extern tree finish_concept_name                 (tree);
extern tree finish_shorthand_constraint         (tree, tree);
extern tree finish_requires_expr                (tree, tree);
extern tree finish_simple_requirement           (tree);
extern tree finish_type_requirement             (tree);
extern tree finish_compound_requirement         (tree, tree, bool);
extern tree finish_nested_requirement           (tree);
extern void check_constrained_friend            (tree, tree);
extern tree tsubst_requires_expr                (tree, tree, tsubst_flags_t, tree);
extern tree tsubst_constraint                   (tree, tree, tsubst_flags_t, tree);
extern tree tsubst_constraint_info              (tree, tree, tsubst_flags_t, tree);
extern bool function_concept_check_p            (tree);
extern tree normalize_expression                (tree);
extern tree expand_concept                      (tree, tree);
extern bool expanding_concept                   ();
extern tree evaluate_constraints                (tree, tree);
extern tree evaluate_function_concept           (tree, tree);
extern tree evaluate_variable_concept           (tree, tree);
extern tree evaluate_constraint_expression      (tree, tree);
extern bool constraints_satisfied_p             (tree);
extern bool constraints_satisfied_p             (tree, tree);
extern tree lookup_constraint_satisfaction      (tree, tree);
extern tree memoize_constraint_satisfaction     (tree, tree, tree);
extern tree lookup_concept_satisfaction         (tree, tree);
extern tree memoize_concept_satisfaction        (tree, tree, tree);
extern tree get_concept_expansion               (tree, tree);
extern tree save_concept_expansion              (tree, tree, tree);
extern bool* lookup_subsumption_result          (tree, tree);
extern bool save_subsumption_result             (tree, tree, bool);
extern bool equivalent_constraints              (tree, tree);
extern bool equivalently_constrained            (tree, tree);
extern bool subsumes_constraints                (tree, tree);
extern bool strictly_subsumes			(tree, tree);
extern int more_constrained                     (tree, tree);
extern void diagnose_constraints                (location_t, tree, tree);
extern tree decompose_conclusions               (tree);
extern bool subsumes                            (tree, tree);
extern void cp_finish_injected_record_type (tree);
extern void vtv_compute_class_hierarchy_transitive_closure (void);
extern void vtv_generate_init_routine           (void);
extern void vtv_save_class_info                 (tree);
extern void vtv_recover_class_info              (void);
extern void vtv_build_vtable_verify_fndecl      (void);
extern void fini_constexpr			(void);
extern bool literal_type_p                      (tree);
extern tree register_constexpr_fundef           (tree, tree);
extern bool is_valid_constexpr_fn		(tree, bool);
extern bool check_constexpr_ctor_body           (tree, tree, bool);
extern tree constexpr_fn_retval		(tree);
extern tree ensure_literal_type_for_constexpr_object (tree);
extern bool potential_constant_expression       (tree);
extern bool is_constant_expression (tree);
extern bool is_nondependent_constant_expression (tree);
extern bool is_nondependent_static_init_expression (tree);
extern bool is_static_init_expression    (tree);
extern bool potential_rvalue_constant_expression (tree);
extern bool require_potential_constant_expression (tree);
extern bool require_constant_expression (tree);
extern bool require_rvalue_constant_expression (tree);
extern bool require_potential_rvalue_constant_expression (tree);
extern tree cxx_constant_value			(tree, tree = NULL_TREE);
extern tree cxx_constant_init			(tree, tree = NULL_TREE);
extern tree maybe_constant_value		(tree, tree = NULL_TREE);
extern tree maybe_constant_init			(tree, tree = NULL_TREE);
extern tree fold_non_dependent_expr		(tree, tsubst_flags_t = tf_none);
extern tree fold_simple				(tree);
extern bool is_sub_constant_expr                (tree);
extern bool reduced_constant_expression_p       (tree);
extern bool is_instantiation_of_constexpr       (tree);
extern bool var_in_constexpr_fn                 (tree);
extern bool var_in_maybe_constexpr_fn           (tree);
extern void explain_invalid_constexpr_fn        (tree);
extern vec<tree> cx_error_context               (void);
extern tree fold_sizeof_expr			(tree);
extern void clear_cv_and_fold_caches		(void);
extern void cp_ubsan_maybe_instrument_member_call (tree);
extern void cp_ubsan_instrument_member_accesses (tree *);
extern tree cp_ubsan_maybe_instrument_downcast	(location_t, tree, tree, tree);
extern tree cp_ubsan_maybe_instrument_cast_to_vbase (location_t, tree, tree);
extern void cp_ubsan_maybe_initialize_vtbl_ptrs (tree);
inline tree
ovl_first (tree node)
{
while (TREE_CODE (node) == OVERLOAD)
node = OVL_FUNCTION (node);
return node;
}
inline bool
type_unknown_p (const_tree expr)
{
return TREE_TYPE (expr) == unknown_type_node;
}
inline hashval_t
named_decl_hash::hash (const value_type decl)
{
tree name = OVL_NAME (decl);
return name ? IDENTIFIER_HASH_VALUE (name) : 0;
}
inline bool
named_decl_hash::equal (const value_type existing, compare_type candidate)
{
tree name = OVL_NAME (existing);
return candidate == name;
}
inline bool
null_node_p (const_tree expr)
{
STRIP_ANY_LOCATION_WRAPPER (expr);
return expr == null_node;
}
#if CHECKING_P
namespace selftest {
extern void run_cp_tests (void);
extern void cp_pt_c_tests ();
extern void cp_tree_c_tests (void);
} 
#endif 
#endif 

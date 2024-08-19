#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "backend.h"
#include "target.h"
#include "rtl.h"
#include "tree.h"
#include "gimple.h"
#include "df.h"
#include "memmodel.h"
#include "tm_p.h"
#include "gimple-ssa.h"
#include "emit-rtl.h"
#include "alias.h"
#include "fold-const.h"
#include "varasm.h"
#include "cselib.h"
#include "langhooks.h"
#include "cfganal.h"
#include "rtl-iter.h"
#include "cgraph.h"
struct alias_set_hash : int_hash <int, INT_MIN, INT_MIN + 1> {};
struct GTY(()) alias_set_entry {
alias_set_type alias_set;
bool has_zero_child;
bool is_pointer;
bool has_pointer;
hash_map<alias_set_hash, int> *children;
};
static int rtx_equal_for_memref_p (const_rtx, const_rtx);
static void record_set (rtx, const_rtx, void *);
static int base_alias_check (rtx, rtx, rtx, rtx, machine_mode,
machine_mode);
static rtx find_base_value (rtx);
static int mems_in_disjoint_alias_sets_p (const_rtx, const_rtx);
static alias_set_entry *get_alias_set_entry (alias_set_type);
static tree decl_for_component_ref (tree);
static int write_dependence_p (const_rtx,
const_rtx, machine_mode, rtx,
bool, bool, bool);
static int compare_base_symbol_refs (const_rtx, const_rtx);
static void memory_modified_1 (rtx, const_rtx, void *);
static struct {
unsigned long long num_alias_zero;
unsigned long long num_same_alias_set;
unsigned long long num_same_objects;
unsigned long long num_volatile;
unsigned long long num_dag;
unsigned long long num_universal;
unsigned long long num_disambiguated;
} alias_stats;
#define SIZE_FOR_MODE(X) (GET_MODE_SIZE (GET_MODE (X)))
#define MAX_ALIAS_LOOP_PASSES 10
static GTY(()) vec<rtx, va_gc> *reg_base_value;
static rtx *new_reg_base_value;
static GTY(()) rtx arg_base_value;
static int unique_id;
static GTY((deletable)) vec<rtx, va_gc> *old_reg_base_value;
#define UNIQUE_BASE_VALUE_SP	-1
#define UNIQUE_BASE_VALUE_ARGP	-2
#define UNIQUE_BASE_VALUE_FP	-3
#define UNIQUE_BASE_VALUE_HFP	-4
#define static_reg_base_value \
(this_target_rtl->x_static_reg_base_value)
#define REG_BASE_VALUE(X)					\
(REGNO (X) < vec_safe_length (reg_base_value)			\
? (*reg_base_value)[REGNO (X)] : 0)
static GTY(()) vec<rtx, va_gc> *reg_known_value;
static sbitmap reg_known_equiv_p;
static bool copying_arguments;
static GTY (()) vec<alias_set_entry *, va_gc> *alias_sets;

static bool
ao_ref_from_mem (ao_ref *ref, const_rtx mem)
{
tree expr = MEM_EXPR (mem);
tree base;
if (!expr)
return false;
ao_ref_init (ref, expr);
base = ao_ref_base (ref);
if (base == NULL_TREE)
return false;
if (!(DECL_P (base)
|| (TREE_CODE (base) == MEM_REF
&& TREE_CODE (TREE_OPERAND (base, 0)) == SSA_NAME)
|| (TREE_CODE (base) == TARGET_MEM_REF
&& TREE_CODE (TMR_BASE (base)) == SSA_NAME)))
return false;
if (VAR_P (base)
&& ! is_global_var (base)
&& cfun->gimple_df->decls_to_pointers != NULL)
{
tree *namep = cfun->gimple_df->decls_to_pointers->get (base);
if (namep)
ref->base = build_simple_mem_ref (*namep);
}
ref->ref_alias_set = MEM_ALIAS_SET (mem);
if (!MEM_OFFSET_KNOWN_P (mem)
|| !MEM_SIZE_KNOWN_P (mem))
return true;
if (maybe_lt (MEM_OFFSET (mem), 0)
|| (ref->max_size_known_p ()
&& maybe_gt ((MEM_OFFSET (mem) + MEM_SIZE (mem)) * BITS_PER_UNIT,
ref->max_size)))
ref->ref = NULL_TREE;
ref->offset += MEM_OFFSET (mem) * BITS_PER_UNIT;
ref->size = MEM_SIZE (mem) * BITS_PER_UNIT;
if (ref->max_size_known_p ())
ref->max_size = upper_bound (ref->max_size, ref->size);
if (MEM_EXPR (mem) != get_spill_slot_decl (false)
&& (maybe_lt (ref->offset, 0)
|| (DECL_P (ref->base)
&& (DECL_SIZE (ref->base) == NULL_TREE
|| !poly_int_tree_p (DECL_SIZE (ref->base))
|| maybe_lt (wi::to_poly_offset (DECL_SIZE (ref->base)),
ref->offset + ref->size)))))
return false;
return true;
}
static bool
rtx_refs_may_alias_p (const_rtx x, const_rtx mem, bool tbaa_p)
{
ao_ref ref1, ref2;
if (!ao_ref_from_mem (&ref1, x)
|| !ao_ref_from_mem (&ref2, mem))
return true;
return refs_may_alias_p_1 (&ref1, &ref2,
tbaa_p
&& MEM_ALIAS_SET (x) != 0
&& MEM_ALIAS_SET (mem) != 0);
}
static inline alias_set_entry *
get_alias_set_entry (alias_set_type alias_set)
{
return (*alias_sets)[alias_set];
}
static inline int
mems_in_disjoint_alias_sets_p (const_rtx mem1, const_rtx mem2)
{
return (flag_strict_aliasing
&& ! alias_sets_conflict_p (MEM_ALIAS_SET (mem1),
MEM_ALIAS_SET (mem2)));
}
bool
alias_set_subset_of (alias_set_type set1, alias_set_type set2)
{
alias_set_entry *ase2;
if (!flag_strict_aliasing)
return true;
if (set2 == 0)
return true;
ase2 = get_alias_set_entry (set2);
if (ase2 != 0
&& (ase2->has_zero_child
|| (ase2->children && ase2->children->get (set1))))
return true;
if (ase2 && ase2->has_pointer)
{
alias_set_entry *ase1 = get_alias_set_entry (set1);
if (ase1 && ase1->is_pointer)
{
alias_set_type voidptr_set = TYPE_ALIAS_SET (ptr_type_node);
if (set1 == voidptr_set || set2 == voidptr_set)
return true;
if (ase2->children && set1 != voidptr_set
&& ase2->children->get (voidptr_set))
return true;
}
}
return false;
}
int
alias_sets_conflict_p (alias_set_type set1, alias_set_type set2)
{
alias_set_entry *ase1;
alias_set_entry *ase2;
if (alias_sets_must_conflict_p (set1, set2))
return 1;
ase1 = get_alias_set_entry (set1);
if (ase1 != 0
&& ase1->children && ase1->children->get (set2))
{
++alias_stats.num_dag;
return 1;
}
ase2 = get_alias_set_entry (set2);
if (ase2 != 0
&& ase2->children && ase2->children->get (set1))
{
++alias_stats.num_dag;
return 1;
}
if (ase1 && ase2 && ase1->has_pointer && ase2->has_pointer)
{
alias_set_type voidptr_set = TYPE_ALIAS_SET (ptr_type_node);
if (set1 == voidptr_set || set2 == voidptr_set)
{
++alias_stats.num_universal;
return true;
}
if (ase1->is_pointer && set2 != voidptr_set
&& ase2->children && ase2->children->get (voidptr_set))
{
++alias_stats.num_universal;
return true;
}
if (ase2->is_pointer && set1 != voidptr_set
&& ase1->children && ase1->children->get (voidptr_set))
{
++alias_stats.num_universal;
return true;
}
}
++alias_stats.num_disambiguated;
return 0;
}
int
alias_sets_must_conflict_p (alias_set_type set1, alias_set_type set2)
{
if (!flag_strict_aliasing)
return 1;
if (set1 == 0 || set2 == 0)
{
++alias_stats.num_alias_zero;
return 1;
}
if (set1 == set2)
{
++alias_stats.num_same_alias_set;
return 1;
}
return 0;
}
int
objects_must_conflict_p (tree t1, tree t2)
{
alias_set_type set1, set2;
if (t1 == 0 && t2 == 0)
return 0;
if (t1 == t2)
{
++alias_stats.num_same_objects;
return 1;
}
if (t1 != 0 && TYPE_VOLATILE (t1) && t2 != 0 && TYPE_VOLATILE (t2))
{
++alias_stats.num_volatile;
return 1;
}
set1 = t1 ? get_alias_set (t1) : 0;
set2 = t2 ? get_alias_set (t2) : 0;
return alias_sets_must_conflict_p (set1, set2);
}

tree
component_uses_parent_alias_set_from (const_tree t)
{
const_tree found = NULL_TREE;
if (AGGREGATE_TYPE_P (TREE_TYPE (t))
&& TYPE_TYPELESS_STORAGE (TREE_TYPE (t)))
return const_cast <tree> (t);
while (handled_component_p (t))
{
switch (TREE_CODE (t))
{
case COMPONENT_REF:
if (DECL_NONADDRESSABLE_P (TREE_OPERAND (t, 1)))
found = t;
else if (TREE_CODE (TREE_TYPE (TREE_OPERAND (t, 0))) == UNION_TYPE)
found = t;
break;
case ARRAY_REF:
case ARRAY_RANGE_REF:
if (TYPE_NONALIASED_COMPONENT (TREE_TYPE (TREE_OPERAND (t, 0))))
found = t;
break;
case REALPART_EXPR:
case IMAGPART_EXPR:
break;
case BIT_FIELD_REF:
case VIEW_CONVERT_EXPR:
found = t;
break;
default:
gcc_unreachable ();
}
if (get_alias_set (TREE_TYPE (TREE_OPERAND (t, 0))) == 0)
found = t;
t = TREE_OPERAND (t, 0);
}
if (found)
return TREE_OPERAND (found, 0);
return NULL_TREE;
}
static bool
ref_all_alias_ptr_type_p (const_tree t)
{
return (TREE_CODE (TREE_TYPE (t)) == VOID_TYPE
|| TYPE_REF_CAN_ALIAS_ALL (t));
}
static alias_set_type
get_deref_alias_set_1 (tree t)
{
if (! TYPE_P (t))
t = TREE_TYPE (t);
if (ref_all_alias_ptr_type_p (t))
return 0;
return -1;
}
alias_set_type
get_deref_alias_set (tree t)
{
if (!flag_strict_aliasing)
return 0;
alias_set_type set = get_deref_alias_set_1 (t);
if (set == -1)
{
if (! TYPE_P (t))
t = TREE_TYPE (t);
set = get_alias_set (TREE_TYPE (t));
}
return set;
}
static tree
reference_alias_ptr_type_1 (tree *t)
{
tree inner;
inner = *t;
while (handled_component_p (inner))
{
if (TREE_CODE (inner) == VIEW_CONVERT_EXPR)
*t = TREE_OPERAND (inner, 0);
inner = TREE_OPERAND (inner, 0);
}
if (INDIRECT_REF_P (inner)
&& ref_all_alias_ptr_type_p (TREE_TYPE (TREE_OPERAND (inner, 0))))
return TREE_TYPE (TREE_OPERAND (inner, 0));
else if (TREE_CODE (inner) == TARGET_MEM_REF)
return TREE_TYPE (TMR_OFFSET (inner));
else if (TREE_CODE (inner) == MEM_REF
&& ref_all_alias_ptr_type_p (TREE_TYPE (TREE_OPERAND (inner, 1))))
return TREE_TYPE (TREE_OPERAND (inner, 1));
if (TREE_CODE (inner) == MEM_REF
&& (TYPE_MAIN_VARIANT (TREE_TYPE (inner))
!= TYPE_MAIN_VARIANT
(TREE_TYPE (TREE_TYPE (TREE_OPERAND (inner, 1))))))
return TREE_TYPE (TREE_OPERAND (inner, 1));
tree tem = component_uses_parent_alias_set_from (*t);
if (tem)
*t = tem;
return NULL_TREE;
}
tree
reference_alias_ptr_type (tree t)
{
if (lang_hooks.get_alias_set (t) == 0)
return ptr_type_node;
tree ptype = reference_alias_ptr_type_1 (&t);
if (ptype != NULL_TREE)
return ptype;
if (TREE_CODE (t) == MEM_REF
|| TREE_CODE (t) == TARGET_MEM_REF)
return TREE_TYPE (TREE_OPERAND (t, 1));
else
return build_pointer_type (TYPE_MAIN_VARIANT (TREE_TYPE (t)));
}
bool
alias_ptr_types_compatible_p (tree t1, tree t2)
{
if (TYPE_MAIN_VARIANT (t1) == TYPE_MAIN_VARIANT (t2))
return true;
if (ref_all_alias_ptr_type_p (t1)
|| ref_all_alias_ptr_type_p (t2))
return false;
return (TYPE_MAIN_VARIANT (TREE_TYPE (t1))
== TYPE_MAIN_VARIANT (TREE_TYPE (t2)));
}
alias_set_entry *
init_alias_set_entry (alias_set_type set)
{
alias_set_entry *ase = ggc_alloc<alias_set_entry> ();
ase->alias_set = set;
ase->children = NULL;
ase->has_zero_child = false;
ase->is_pointer = false;
ase->has_pointer = false;
gcc_checking_assert (!get_alias_set_entry (set));
(*alias_sets)[set] = ase;
return ase;
}
alias_set_type
get_alias_set (tree t)
{
alias_set_type set;
if (t == error_mark_node
|| (! TYPE_P (t)
&& (TREE_TYPE (t) == 0 || TREE_TYPE (t) == error_mark_node)))
return 0;
if (! TYPE_P (t))
{
STRIP_NOPS (t);
set = lang_hooks.get_alias_set (t);
if (set != -1)
return set;
tree ptype = reference_alias_ptr_type_1 (&t);
if (ptype != NULL)
return get_deref_alias_set (ptype);
if (VAR_P (t)
&& DECL_RTL_SET_P (t) && MEM_P (DECL_RTL (t)))
return MEM_ALIAS_SET (DECL_RTL (t));
t = TREE_TYPE (t);
}
t = TYPE_MAIN_VARIANT (t);
if (AGGREGATE_TYPE_P (t)
&& TYPE_TYPELESS_STORAGE (t))
return 0;
if (TYPE_STRUCTURAL_EQUALITY_P (t))
{
set = lang_hooks.get_alias_set (t);
if (set != -1)
return set;
if (canonical_type_used_p (t))
{
gcc_checking_assert (!in_lto_p || !type_with_alias_set_p (t));
return 0;
}
}
else
{
t = TYPE_CANONICAL (t);
gcc_checking_assert (!TYPE_STRUCTURAL_EQUALITY_P (t));
}
gcc_checking_assert (t == TYPE_MAIN_VARIANT (t));
if (TYPE_ALIAS_SET_KNOWN_P (t))
return TYPE_ALIAS_SET (t);
if (!COMPLETE_TYPE_P (t))
{
if (TREE_CODE (t) == ARRAY_TYPE)
return get_alias_set (TREE_TYPE (t));
return 0;
}
set = lang_hooks.get_alias_set (t);
if (set != -1)
return set;
else if (TREE_CODE (t) == FUNCTION_TYPE || TREE_CODE (t) == METHOD_TYPE)
set = 0;
else if (TREE_CODE (t) == VECTOR_TYPE)
set = get_alias_set (TREE_TYPE (t));
else if (TREE_CODE (t) == ARRAY_TYPE
&& (!TYPE_NONALIASED_COMPONENT (t)
|| TYPE_STRUCTURAL_EQUALITY_P (t)))
set = get_alias_set (TREE_TYPE (t));
else if (POINTER_TYPE_P (t) && t != ptr_type_node)
{
tree p;
auto_vec <bool, 8> reference;
for (p = t; POINTER_TYPE_P (p)
|| (TREE_CODE (p) == ARRAY_TYPE
&& (!TYPE_NONALIASED_COMPONENT (p)
|| !COMPLETE_TYPE_P (p)
|| TYPE_STRUCTURAL_EQUALITY_P (p)))
|| TREE_CODE (p) == VECTOR_TYPE;
p = TREE_TYPE (p))
{
if (reference.length () == 8)
{
p = ptr_type_node;
break;
}
if (TREE_CODE (p) == REFERENCE_TYPE)
reference.safe_push (true && !in_lto_p);
if (TREE_CODE (p) == POINTER_TYPE)
reference.safe_push (false);
}
p = TYPE_MAIN_VARIANT (p);
if (TREE_CODE (p) == VOID_TYPE || TYPE_STRUCTURAL_EQUALITY_P (p))
set = get_alias_set (ptr_type_node);
else
{
p = TYPE_CANONICAL (p);
while (!reference.is_empty ())
{
if (reference.pop ())
p = build_reference_type (p);
else
p = build_pointer_type (p);
gcc_checking_assert (p == TYPE_MAIN_VARIANT (p));
gcc_checking_assert (!TYPE_CANONICAL (p)
|| p == TYPE_CANONICAL (p));
}
gcc_checking_assert (p == TYPE_MAIN_VARIANT (p));
if (TYPE_ALIAS_SET_KNOWN_P (p))
set = TYPE_ALIAS_SET (p);
else
{
set = new_alias_set ();
TYPE_ALIAS_SET (p) = set;
}
}
}
else if (t == ptr_type_node)
set = new_alias_set ();
else
{
gcc_checking_assert (TYPE_CANONICAL (t) == t);
set = new_alias_set ();
}
TYPE_ALIAS_SET (t) = set;
if (AGGREGATE_TYPE_P (t) || TREE_CODE (t) == COMPLEX_TYPE)
record_component_aliases (t);
if (POINTER_TYPE_P (t) && set)
{
alias_set_entry *ase = get_alias_set_entry (set);
if (!ase)
ase = init_alias_set_entry (set);
ase->is_pointer = true;
ase->has_pointer = true;
}
return set;
}
alias_set_type
new_alias_set (void)
{
if (alias_sets == 0)
vec_safe_push (alias_sets, (alias_set_entry *) NULL);
vec_safe_push (alias_sets, (alias_set_entry *) NULL);
return alias_sets->length () - 1;
}
void
record_alias_subset (alias_set_type superset, alias_set_type subset)
{
alias_set_entry *superset_entry;
alias_set_entry *subset_entry;
if (superset == subset)
return;
gcc_assert (superset);
superset_entry = get_alias_set_entry (superset);
if (superset_entry == 0)
{
superset_entry = init_alias_set_entry (superset);
}
if (subset == 0)
superset_entry->has_zero_child = 1;
else
{
subset_entry = get_alias_set_entry (subset);
if (!superset_entry->children)
superset_entry->children
= hash_map<alias_set_hash, int>::create_ggc (64);
if (subset_entry)
{
if (subset_entry->has_zero_child)
superset_entry->has_zero_child = true;
if (subset_entry->has_pointer)
superset_entry->has_pointer = true;
if (subset_entry->children)
{
hash_map<alias_set_hash, int>::iterator iter
= subset_entry->children->begin ();
for (; iter != subset_entry->children->end (); ++iter)
superset_entry->children->put ((*iter).first, (*iter).second);
}
}
superset_entry->children->put (subset, 0);
}
}
void
record_component_aliases (tree type)
{
alias_set_type superset = get_alias_set (type);
tree field;
if (superset == 0)
return;
switch (TREE_CODE (type))
{
case RECORD_TYPE:
case UNION_TYPE:
case QUAL_UNION_TYPE:
for (field = TYPE_FIELDS (type); field != 0; field = DECL_CHAIN (field))
if (TREE_CODE (field) == FIELD_DECL && !DECL_NONADDRESSABLE_P (field))
{
tree t = TREE_TYPE (field);
if (in_lto_p)
{
while (!canonical_type_used_p (t) && !POINTER_TYPE_P (t))
{
gcc_checking_assert (TYPE_STRUCTURAL_EQUALITY_P (t));
t = TREE_TYPE (t);
}
if (POINTER_TYPE_P (t))
t = ptr_type_node;
else if (flag_checking)
gcc_checking_assert (get_alias_set (t)
== get_alias_set (TREE_TYPE (field)));
}
record_alias_subset (superset, get_alias_set (t));
}
break;
case COMPLEX_TYPE:
record_alias_subset (superset, get_alias_set (TREE_TYPE (type)));
break;
default:
break;
}
}
static GTY(()) alias_set_type varargs_set = -1;
alias_set_type
get_varargs_alias_set (void)
{
#if 1
return 0;
#else
if (varargs_set == -1)
varargs_set = new_alias_set ();
return varargs_set;
#endif
}
static GTY(()) alias_set_type frame_set = -1;
alias_set_type
get_frame_alias_set (void)
{
if (frame_set == -1)
frame_set = new_alias_set ();
return frame_set;
}
static rtx
unique_base_value (HOST_WIDE_INT id)
{
return gen_rtx_ADDRESS (Pmode, id);
}
static bool
unique_base_value_p (rtx x)
{
return GET_CODE (x) == ADDRESS && GET_MODE (x) == Pmode;
}
static bool
known_base_value_p (rtx x)
{
switch (GET_CODE (x))
{
case LABEL_REF:
case SYMBOL_REF:
return true;
case ADDRESS:
return GET_MODE (x) != VOIDmode;
default:
return false;
}
}
static rtx
find_base_value (rtx src)
{
unsigned int regno;
scalar_int_mode int_mode;
#if defined (FIND_BASE_TERM)
src = FIND_BASE_TERM (src);
#endif
switch (GET_CODE (src))
{
case SYMBOL_REF:
case LABEL_REF:
return src;
case REG:
regno = REGNO (src);
if (regno < FIRST_PSEUDO_REGISTER && copying_arguments)
return new_reg_base_value[regno];
if ((regno >= FIRST_PSEUDO_REGISTER || fixed_regs[regno])
&& regno < vec_safe_length (reg_base_value))
{
if (new_reg_base_value && new_reg_base_value[regno]
&& DF_REG_DEF_COUNT (regno) == 1)
return new_reg_base_value[regno];
if ((*reg_base_value)[regno])
return (*reg_base_value)[regno];
}
return 0;
case MEM:
if (copying_arguments
&& (XEXP (src, 0) == arg_pointer_rtx
|| (GET_CODE (XEXP (src, 0)) == PLUS
&& XEXP (XEXP (src, 0), 0) == arg_pointer_rtx)))
return arg_base_value;
return 0;
case CONST:
src = XEXP (src, 0);
if (GET_CODE (src) != PLUS && GET_CODE (src) != MINUS)
break;
case PLUS:
case MINUS:
{
rtx temp, src_0 = XEXP (src, 0), src_1 = XEXP (src, 1);
if (REG_P (src_0) && REG_POINTER (src_0))
return find_base_value (src_0);
if (REG_P (src_1) && REG_POINTER (src_1))
return find_base_value (src_1);
if (REG_P (src_0))
{
temp = find_base_value (src_0);
if (temp != 0)
src_0 = temp;
}
if (REG_P (src_1))
{
temp = find_base_value (src_1);
if (temp!= 0)
src_1 = temp;
}
if (src_0 != 0 && known_base_value_p (src_0))
return src_0;
if (src_1 != 0 && known_base_value_p (src_1))
return src_1;
if (CONST_INT_P (src_1) || CONSTANT_P (src_0))
return find_base_value (src_0);
else if (CONST_INT_P (src_0) || CONSTANT_P (src_1))
return find_base_value (src_1);
return 0;
}
case LO_SUM:
return find_base_value (XEXP (src, 1));
case AND:
if (CONST_INT_P (XEXP (src, 1)) && INTVAL (XEXP (src, 1)) != 0)
return find_base_value (XEXP (src, 0));
return 0;
case TRUNCATE:
if (!target_default_pointer_address_modes_p ())
break;
if (!is_a <scalar_int_mode> (GET_MODE (src), &int_mode)
|| GET_MODE_PRECISION (int_mode) < GET_MODE_PRECISION (Pmode))
break;
case HIGH:
case PRE_INC:
case PRE_DEC:
case POST_INC:
case POST_DEC:
case PRE_MODIFY:
case POST_MODIFY:
return find_base_value (XEXP (src, 0));
case ZERO_EXTEND:
case SIGN_EXTEND:	
if (!target_default_pointer_address_modes_p ())
break;
{
rtx temp = find_base_value (XEXP (src, 0));
if (temp != 0 && CONSTANT_P (temp))
temp = convert_memory_address (Pmode, temp);
return temp;
}
default:
break;
}
return 0;
}
static sbitmap reg_seen;
static void
record_set (rtx dest, const_rtx set, void *data ATTRIBUTE_UNUSED)
{
unsigned regno;
rtx src;
int n;
if (!REG_P (dest))
return;
regno = REGNO (dest);
gcc_checking_assert (regno < reg_base_value->length ());
n = REG_NREGS (dest);
if (n != 1)
{
while (--n >= 0)
{
bitmap_set_bit (reg_seen, regno + n);
new_reg_base_value[regno + n] = 0;
}
return;
}
if (set)
{
if (GET_CODE (set) == CLOBBER)
{
new_reg_base_value[regno] = 0;
return;
}
src = SET_SRC (set);
}
else
{
if (bitmap_bit_p (reg_seen, regno))
{
new_reg_base_value[regno] = 0;
return;
}
bitmap_set_bit (reg_seen, regno);
new_reg_base_value[regno] = unique_base_value (unique_id++);
return;
}
if (new_reg_base_value[regno] != 0
&& find_base_value (src) != new_reg_base_value[regno])
switch (GET_CODE (src))
{
case LO_SUM:
case MINUS:
if (XEXP (src, 0) != dest && XEXP (src, 1) != dest)
new_reg_base_value[regno] = 0;
break;
case PLUS:
{
rtx other = NULL_RTX;
if (XEXP (src, 0) == dest)
other = XEXP (src, 1);
else if (XEXP (src, 1) == dest)
other = XEXP (src, 0);
if (! other || find_base_value (other))
new_reg_base_value[regno] = 0;
break;
}
case AND:
if (XEXP (src, 0) != dest || !CONST_INT_P (XEXP (src, 1)))
new_reg_base_value[regno] = 0;
break;
default:
new_reg_base_value[regno] = 0;
break;
}
else if ((regno >= FIRST_PSEUDO_REGISTER || ! fixed_regs[regno])
&& ! bitmap_bit_p (reg_seen, regno) && new_reg_base_value[regno] == 0)
new_reg_base_value[regno] = find_base_value (src);
bitmap_set_bit (reg_seen, regno);
}
rtx
get_reg_base_value (unsigned int regno)
{
return (*reg_base_value)[regno];
}
rtx
get_reg_known_value (unsigned int regno)
{
if (regno >= FIRST_PSEUDO_REGISTER)
{
regno -= FIRST_PSEUDO_REGISTER;
if (regno < vec_safe_length (reg_known_value))
return (*reg_known_value)[regno];
}
return NULL;
}
static void
set_reg_known_value (unsigned int regno, rtx val)
{
if (regno >= FIRST_PSEUDO_REGISTER)
{
regno -= FIRST_PSEUDO_REGISTER;
if (regno < vec_safe_length (reg_known_value))
(*reg_known_value)[regno] = val;
}
}
bool
get_reg_known_equiv_p (unsigned int regno)
{
if (regno >= FIRST_PSEUDO_REGISTER)
{
regno -= FIRST_PSEUDO_REGISTER;
if (regno < vec_safe_length (reg_known_value))
return bitmap_bit_p (reg_known_equiv_p, regno);
}
return false;
}
static void
set_reg_known_equiv_p (unsigned int regno, bool val)
{
if (regno >= FIRST_PSEUDO_REGISTER)
{
regno -= FIRST_PSEUDO_REGISTER;
if (regno < vec_safe_length (reg_known_value))
{
if (val)
bitmap_set_bit (reg_known_equiv_p, regno);
else
bitmap_clear_bit (reg_known_equiv_p, regno);
}
}
}
rtx
canon_rtx (rtx x)
{
if (REG_P (x) && REGNO (x) >= FIRST_PSEUDO_REGISTER)
{
rtx t = get_reg_known_value (REGNO (x));
if (t == x)
return x;
if (t)
return canon_rtx (t);
}
if (GET_CODE (x) == PLUS)
{
rtx x0 = canon_rtx (XEXP (x, 0));
rtx x1 = canon_rtx (XEXP (x, 1));
if (x0 != XEXP (x, 0) || x1 != XEXP (x, 1))
return simplify_gen_binary (PLUS, GET_MODE (x), x0, x1);
}
else if (MEM_P (x))
x = replace_equiv_address_nv (x, canon_rtx (XEXP (x, 0)));
return x;
}
static int
rtx_equal_for_memref_p (const_rtx x, const_rtx y)
{
int i;
int j;
enum rtx_code code;
const char *fmt;
if (x == 0 && y == 0)
return 1;
if (x == 0 || y == 0)
return 0;
if (x == y)
return 1;
code = GET_CODE (x);
if (code != GET_CODE (y))
return 0;
if (GET_MODE (x) != GET_MODE (y))
return 0;
switch (code)
{
case REG:
return REGNO (x) == REGNO (y);
case LABEL_REF:
return label_ref_label (x) == label_ref_label (y);
case SYMBOL_REF:
return compare_base_symbol_refs (x, y) == 1;
case ENTRY_VALUE:
return rtx_equal_p (ENTRY_VALUE_EXP (x), ENTRY_VALUE_EXP (y));
case VALUE:
CASE_CONST_UNIQUE:
return 0;
default:
break;
}
if (code == PLUS)
return ((rtx_equal_for_memref_p (XEXP (x, 0), XEXP (y, 0))
&& rtx_equal_for_memref_p (XEXP (x, 1), XEXP (y, 1)))
|| (rtx_equal_for_memref_p (XEXP (x, 0), XEXP (y, 1))
&& rtx_equal_for_memref_p (XEXP (x, 1), XEXP (y, 0))));
if (COMMUTATIVE_P (x))
{
rtx xop0 = canon_rtx (XEXP (x, 0));
rtx yop0 = canon_rtx (XEXP (y, 0));
rtx yop1 = canon_rtx (XEXP (y, 1));
return ((rtx_equal_for_memref_p (xop0, yop0)
&& rtx_equal_for_memref_p (canon_rtx (XEXP (x, 1)), yop1))
|| (rtx_equal_for_memref_p (xop0, yop1)
&& rtx_equal_for_memref_p (canon_rtx (XEXP (x, 1)), yop0)));
}
else if (NON_COMMUTATIVE_P (x))
{
return (rtx_equal_for_memref_p (canon_rtx (XEXP (x, 0)),
canon_rtx (XEXP (y, 0)))
&& rtx_equal_for_memref_p (canon_rtx (XEXP (x, 1)),
canon_rtx (XEXP (y, 1))));
}
else if (UNARY_P (x))
return rtx_equal_for_memref_p (canon_rtx (XEXP (x, 0)),
canon_rtx (XEXP (y, 0)));
fmt = GET_RTX_FORMAT (code);
for (i = GET_RTX_LENGTH (code) - 1; i >= 0; i--)
{
switch (fmt[i])
{
case 'i':
if (XINT (x, i) != XINT (y, i))
return 0;
break;
case 'p':
if (maybe_ne (SUBREG_BYTE (x), SUBREG_BYTE (y)))
return 0;
break;
case 'E':
if (XVECLEN (x, i) != XVECLEN (y, i))
return 0;
for (j = 0; j < XVECLEN (x, i); j++)
if (rtx_equal_for_memref_p (canon_rtx (XVECEXP (x, i, j)),
canon_rtx (XVECEXP (y, i, j))) == 0)
return 0;
break;
case 'e':
if (rtx_equal_for_memref_p (canon_rtx (XEXP (x, i)),
canon_rtx (XEXP (y, i))) == 0)
return 0;
break;
case 's':
if (strcmp (XSTR (x, i), XSTR (y, i)))
return 0;
break;
case '0':
break;
default:
gcc_unreachable ();
}
}
return 1;
}
static rtx
find_base_term (rtx x, vec<std::pair<cselib_val *,
struct elt_loc_list *> > &visited_vals)
{
cselib_val *val;
struct elt_loc_list *l, *f;
rtx ret;
scalar_int_mode int_mode;
#if defined (FIND_BASE_TERM)
x = FIND_BASE_TERM (x);
#endif
switch (GET_CODE (x))
{
case REG:
return REG_BASE_VALUE (x);
case TRUNCATE:
if (!target_default_pointer_address_modes_p ())
return 0;
if (!is_a <scalar_int_mode> (GET_MODE (x), &int_mode)
|| GET_MODE_PRECISION (int_mode) < GET_MODE_PRECISION (Pmode))
return 0;
case HIGH:
case PRE_INC:
case PRE_DEC:
case POST_INC:
case POST_DEC:
case PRE_MODIFY:
case POST_MODIFY:
return find_base_term (XEXP (x, 0), visited_vals);
case ZERO_EXTEND:
case SIGN_EXTEND:	
if (!target_default_pointer_address_modes_p ())
return 0;
{
rtx temp = find_base_term (XEXP (x, 0), visited_vals);
if (temp != 0 && CONSTANT_P (temp))
temp = convert_memory_address (Pmode, temp);
return temp;
}
case VALUE:
val = CSELIB_VAL_PTR (x);
ret = NULL_RTX;
if (!val)
return ret;
if (cselib_sp_based_value_p (val))
return static_reg_base_value[STACK_POINTER_REGNUM];
f = val->locs;
if (f)
visited_vals.safe_push (std::make_pair (val, f));
val->locs = NULL;
for (l = f; l; l = l->next)
if (GET_CODE (l->loc) == VALUE
&& CSELIB_VAL_PTR (l->loc)->locs
&& !CSELIB_VAL_PTR (l->loc)->locs->next
&& CSELIB_VAL_PTR (l->loc)->locs->loc == x)
continue;
else if ((ret = find_base_term (l->loc, visited_vals)) != 0)
break;
return ret;
case LO_SUM:
return find_base_term (XEXP (x, 1), visited_vals);
case CONST:
x = XEXP (x, 0);
if (GET_CODE (x) != PLUS && GET_CODE (x) != MINUS)
return 0;
case PLUS:
case MINUS:
{
rtx tmp1 = XEXP (x, 0);
rtx tmp2 = XEXP (x, 1);
if (tmp1 == pic_offset_table_rtx && CONSTANT_P (tmp2))
return find_base_term (tmp2, visited_vals);
if (REG_P (tmp1) && REG_POINTER (tmp1))
;
else if (REG_P (tmp2) && REG_POINTER (tmp2))
std::swap (tmp1, tmp2);
else if (CONSTANT_P (tmp2) && !CONST_INT_P (tmp2))
std::swap (tmp1, tmp2);
rtx base = find_base_term (tmp1, visited_vals);
if (base != NULL_RTX
&& ((REG_P (tmp1) && REG_POINTER (tmp1))
|| known_base_value_p (base)))
return base;
base = find_base_term (tmp2, visited_vals);
if (base != NULL_RTX
&& ((REG_P (tmp2) && REG_POINTER (tmp2))
|| known_base_value_p (base)))
return base;
return 0;
}
case AND:
if (CONST_INT_P (XEXP (x, 1)) && INTVAL (XEXP (x, 1)) != 0)
return find_base_term (XEXP (x, 0), visited_vals);
return 0;
case SYMBOL_REF:
case LABEL_REF:
return x;
default:
return 0;
}
}
static rtx
find_base_term (rtx x)
{
auto_vec<std::pair<cselib_val *, struct elt_loc_list *>, 32> visited_vals;
rtx res = find_base_term (x, visited_vals);
for (unsigned i = 0; i < visited_vals.length (); ++i)
visited_vals[i].first->locs = visited_vals[i].second;
return res;
}
bool
may_be_sp_based_p (rtx x)
{
rtx base = find_base_term (x);
return !base || base == static_reg_base_value[STACK_POINTER_REGNUM];
}
int
compare_base_decls (tree base1, tree base2)
{
int ret;
gcc_checking_assert (DECL_P (base1) && DECL_P (base2));
if (base1 == base2)
return 1;
if (DECL_REGISTER (base1)
&& DECL_REGISTER (base2)
&& HAS_DECL_ASSEMBLER_NAME_P (base1)
&& HAS_DECL_ASSEMBLER_NAME_P (base2)
&& DECL_ASSEMBLER_NAME_SET_P (base1)
&& DECL_ASSEMBLER_NAME_SET_P (base2))
{
if (DECL_ASSEMBLER_NAME_RAW (base1) == DECL_ASSEMBLER_NAME_RAW (base2))
return 1;
return -1;
}
if (!decl_in_symtab_p (base1)
|| !decl_in_symtab_p (base2))
return 0;
symtab_node *node1 = symtab_node::get (base1);
if (!node1)
return 0;
symtab_node *node2 = symtab_node::get (base2);
if (!node2)
return 0;
ret = node1->equal_address_to (node2, true);
return ret;
}
static int
compare_base_symbol_refs (const_rtx x_base, const_rtx y_base)
{
tree x_decl = SYMBOL_REF_DECL (x_base);
tree y_decl = SYMBOL_REF_DECL (y_base);
bool binds_def = true;
if (XSTR (x_base, 0) == XSTR (y_base, 0))
return 1;
if (x_decl && y_decl)
return compare_base_decls (x_decl, y_decl);
if (x_decl || y_decl)
{
if (!x_decl)
{
std::swap (x_decl, y_decl);
std::swap (x_base, y_base);
}
if (!SYMBOL_REF_HAS_BLOCK_INFO_P (y_base))
return -1;
if (!VAR_P (x_decl)
|| (!TREE_STATIC (x_decl) && !TREE_PUBLIC (x_decl)))
return 0;
symtab_node *x_node = symtab_node::get_create (x_decl)
->ultimate_alias_target ();
if (!x_node->definition)
return 0;
x_base = XEXP (DECL_RTL (x_node->decl), 0);
if (!SYMBOL_REF_HAS_BLOCK_INFO_P (x_base))
return 0;
binds_def = decl_binds_to_current_def_p (x_decl);
}
if (SYMBOL_REF_HAS_BLOCK_INFO_P (x_base)
&& SYMBOL_REF_HAS_BLOCK_INFO_P (y_base))
{
if (SYMBOL_REF_BLOCK (x_base) != SYMBOL_REF_BLOCK (y_base))
return 0;
if (SYMBOL_REF_BLOCK_OFFSET (x_base) == SYMBOL_REF_BLOCK_OFFSET (y_base))
return binds_def ? 1 : -1;
if (SYMBOL_REF_ANCHOR_P (x_base) != SYMBOL_REF_ANCHOR_P (y_base))
return -1;
return 0;
}
return -1;
}
static int
base_alias_check (rtx x, rtx x_base, rtx y, rtx y_base,
machine_mode x_mode, machine_mode y_mode)
{
if (x_base == 0)
{
rtx x_c;
if (! flag_expensive_optimizations || (x_c = canon_rtx (x)) == x)
return 1;
x_base = find_base_term (x_c);
if (x_base == 0)
return 1;
}
if (y_base == 0)
{
rtx y_c;
if (! flag_expensive_optimizations || (y_c = canon_rtx (y)) == y)
return 1;
y_base = find_base_term (y_c);
if (y_base == 0)
return 1;
}
if (rtx_equal_p (x_base, y_base))
return 1;
if (GET_CODE (x) == AND && GET_CODE (y) == AND)
return 1;
if (GET_CODE (x) == AND
&& (!CONST_INT_P (XEXP (x, 1))
|| (int) GET_MODE_UNIT_SIZE (y_mode) < -INTVAL (XEXP (x, 1))))
return 1;
if (GET_CODE (y) == AND
&& (!CONST_INT_P (XEXP (y, 1))
|| (int) GET_MODE_UNIT_SIZE (x_mode) < -INTVAL (XEXP (y, 1))))
return 1;
if (GET_CODE (x_base) == SYMBOL_REF && GET_CODE (y_base) == SYMBOL_REF)
return compare_base_symbol_refs (x_base, y_base) != 0;
if (GET_CODE (x_base) != ADDRESS && GET_CODE (y_base) != ADDRESS)
return 0;
if (unique_base_value_p (x_base) || unique_base_value_p (y_base))
return 0;
return 1;
}
static bool
refs_newer_value_p (const_rtx expr, rtx v)
{
int minuid = CSELIB_VAL_PTR (v)->uid;
subrtx_iterator::array_type array;
FOR_EACH_SUBRTX (iter, array, expr, NONCONST)
if (GET_CODE (*iter) == VALUE && CSELIB_VAL_PTR (*iter)->uid >= minuid)
return true;
return false;
}
rtx
get_addr (rtx x)
{
cselib_val *v;
struct elt_loc_list *l;
if (GET_CODE (x) != VALUE)
{
if ((GET_CODE (x) == PLUS || GET_CODE (x) == MINUS)
&& GET_CODE (XEXP (x, 0)) == VALUE
&& CONST_SCALAR_INT_P (XEXP (x, 1)))
{
rtx op0 = get_addr (XEXP (x, 0));
if (op0 != XEXP (x, 0))
{
if (GET_CODE (x) == PLUS
&& GET_CODE (XEXP (x, 1)) == CONST_INT)
return plus_constant (GET_MODE (x), op0, INTVAL (XEXP (x, 1)));
return simplify_gen_binary (GET_CODE (x), GET_MODE (x),
op0, XEXP (x, 1));
}
}
return x;
}
v = CSELIB_VAL_PTR (x);
if (v)
{
bool have_equivs = cselib_have_permanent_equivalences ();
if (have_equivs)
v = canonical_cselib_val (v);
for (l = v->locs; l; l = l->next)
if (CONSTANT_P (l->loc))
return l->loc;
for (l = v->locs; l; l = l->next)
if (!REG_P (l->loc) && !MEM_P (l->loc)
&& (!have_equivs
|| (GET_CODE (l->loc) != VALUE
&& !refs_newer_value_p (l->loc, x))))
return l->loc;
if (have_equivs)
{
for (l = v->locs; l; l = l->next)
if (REG_P (l->loc)
|| (GET_CODE (l->loc) != VALUE
&& !refs_newer_value_p (l->loc, x)))
return l->loc;
return v->val_rtx;
}
if (v->locs)
return v->locs->loc;
}
return x;
}
static rtx
addr_side_effect_eval (rtx addr, poly_int64 size, int n_refs)
{
poly_int64 offset = 0;
switch (GET_CODE (addr))
{
case PRE_INC:
offset = (n_refs + 1) * size;
break;
case PRE_DEC:
offset = -(n_refs + 1) * size;
break;
case POST_INC:
offset = n_refs * size;
break;
case POST_DEC:
offset = -n_refs * size;
break;
default:
return addr;
}
addr = plus_constant (GET_MODE (addr), XEXP (addr, 0), offset);
addr = canon_rtx (addr);
return addr;
}
static inline bool
offset_overlap_p (poly_int64 c, poly_int64 xsize, poly_int64 ysize)
{
if (known_eq (xsize, 0) || known_eq (ysize, 0))
return true;
if (maybe_ge (c, 0))
return maybe_gt (maybe_lt (xsize, 0) ? -xsize : xsize, c);
else
return maybe_gt (maybe_lt (ysize, 0) ? -ysize : ysize, -c);
}
static int
memrefs_conflict_p (poly_int64 xsize, rtx x, poly_int64 ysize, rtx y,
poly_int64 c)
{
if (GET_CODE (x) == VALUE)
{
if (REG_P (y))
{
struct elt_loc_list *l = NULL;
if (CSELIB_VAL_PTR (x))
for (l = canonical_cselib_val (CSELIB_VAL_PTR (x))->locs;
l; l = l->next)
if (REG_P (l->loc) && rtx_equal_for_memref_p (l->loc, y))
break;
if (l)
x = y;
else
x = get_addr (x);
}
else if (x != y)
x = get_addr (x);
}
if (GET_CODE (y) == VALUE)
{
if (REG_P (x))
{
struct elt_loc_list *l = NULL;
if (CSELIB_VAL_PTR (y))
for (l = canonical_cselib_val (CSELIB_VAL_PTR (y))->locs;
l; l = l->next)
if (REG_P (l->loc) && rtx_equal_for_memref_p (l->loc, x))
break;
if (l)
y = x;
else
y = get_addr (y);
}
else if (y != x)
y = get_addr (y);
}
if (GET_CODE (x) == HIGH)
x = XEXP (x, 0);
else if (GET_CODE (x) == LO_SUM)
x = XEXP (x, 1);
else
x = addr_side_effect_eval (x, maybe_lt (xsize, 0) ? -xsize : xsize, 0);
if (GET_CODE (y) == HIGH)
y = XEXP (y, 0);
else if (GET_CODE (y) == LO_SUM)
y = XEXP (y, 1);
else
y = addr_side_effect_eval (y, maybe_lt (ysize, 0) ? -ysize : ysize, 0);
if (GET_CODE (x) == SYMBOL_REF && GET_CODE (y) == SYMBOL_REF)
{
int cmp = compare_base_symbol_refs (x,y);
if (cmp == 1)
return offset_overlap_p (c, xsize, ysize);
if (maybe_lt (xsize, 0) || maybe_lt (ysize, 0))
return -1;
if (!cmp || !offset_overlap_p (c, xsize, ysize))
return 0;
return -1;
}
else if (rtx_equal_for_memref_p (x, y))
{
return offset_overlap_p (c, xsize, ysize);
}
if (GET_CODE (x) == PLUS)
{
rtx x0 = XEXP (x, 0);
rtx x1 = XEXP (x, 1);
if (x0 == y)
return memrefs_conflict_p (xsize, x1, ysize, const0_rtx, c);
else if (x1 == y)
return memrefs_conflict_p (xsize, x0, ysize, const0_rtx, c);
poly_int64 cx1, cy1;
if (GET_CODE (y) == PLUS)
{
rtx y0 = XEXP (y, 0);
rtx y1 = XEXP (y, 1);
if (x0 == y1)
return memrefs_conflict_p (xsize, x1, ysize, y0, c);
if (x1 == y0)
return memrefs_conflict_p (xsize, x0, ysize, y1, c);
if (rtx_equal_for_memref_p (x1, y1))
return memrefs_conflict_p (xsize, x0, ysize, y0, c);
if (rtx_equal_for_memref_p (x0, y0))
return memrefs_conflict_p (xsize, x1, ysize, y1, c);
if (poly_int_rtx_p (x1, &cx1))
{
if (poly_int_rtx_p (y1, &cy1))
return memrefs_conflict_p (xsize, x0, ysize, y0,
c - cx1 + cy1);
else
return memrefs_conflict_p (xsize, x0, ysize, y, c - cx1);
}
else if (poly_int_rtx_p (y1, &cy1))
return memrefs_conflict_p (xsize, x, ysize, y0, c + cy1);
return -1;
}
else if (poly_int_rtx_p (x1, &cx1))
return memrefs_conflict_p (xsize, x0, ysize, y, c - cx1);
}
else if (GET_CODE (y) == PLUS)
{
rtx y0 = XEXP (y, 0);
rtx y1 = XEXP (y, 1);
if (x == y0)
return memrefs_conflict_p (xsize, const0_rtx, ysize, y1, c);
if (x == y1)
return memrefs_conflict_p (xsize, const0_rtx, ysize, y0, c);
poly_int64 cy1;
if (poly_int_rtx_p (y1, &cy1))
return memrefs_conflict_p (xsize, x, ysize, y0, c + cy1);
else
return -1;
}
if (GET_CODE (x) == GET_CODE (y))
switch (GET_CODE (x))
{
case MULT:
{
rtx x0, y0;
rtx x1 = canon_rtx (XEXP (x, 1));
rtx y1 = canon_rtx (XEXP (y, 1));
if (! rtx_equal_for_memref_p (x1, y1))
return -1;
x0 = canon_rtx (XEXP (x, 0));
y0 = canon_rtx (XEXP (y, 0));
if (rtx_equal_for_memref_p (x0, y0))
return offset_overlap_p (c, xsize, ysize);
if (!CONST_INT_P (x1)
|| !can_div_trunc_p (xsize, INTVAL (x1), &xsize)
|| !can_div_trunc_p (ysize, INTVAL (x1), &ysize)
|| !can_div_trunc_p (c, INTVAL (x1), &c))
return -1;
return memrefs_conflict_p (xsize, x0, ysize, y0, c);
}
default:
break;
}
if (GET_CODE (x) == AND && CONST_INT_P (XEXP (x, 1)))
{
HOST_WIDE_INT sc = INTVAL (XEXP (x, 1));
unsigned HOST_WIDE_INT uc = sc;
if (sc < 0 && pow2_or_zerop (-uc))
{
if (maybe_gt (xsize, 0))
xsize = -xsize;
if (maybe_ne (xsize, 0))
xsize += sc + 1;
c -= sc + 1;
return memrefs_conflict_p (xsize, canon_rtx (XEXP (x, 0)),
ysize, y, c);
}
}
if (GET_CODE (y) == AND && CONST_INT_P (XEXP (y, 1)))
{
HOST_WIDE_INT sc = INTVAL (XEXP (y, 1));
unsigned HOST_WIDE_INT uc = sc;
if (sc < 0 && pow2_or_zerop (-uc))
{
if (maybe_gt (ysize, 0))
ysize = -ysize;
if (maybe_ne (ysize, 0))
ysize += sc + 1;
c += sc + 1;
return memrefs_conflict_p (xsize, x,
ysize, canon_rtx (XEXP (y, 0)), c);
}
}
if (CONSTANT_P (x))
{
poly_int64 cx, cy;
if (poly_int_rtx_p (x, &cx) && poly_int_rtx_p (y, &cy))
{
c += cy - cx;
return offset_overlap_p (c, xsize, ysize);
}
if (GET_CODE (x) == CONST)
{
if (GET_CODE (y) == CONST)
return memrefs_conflict_p (xsize, canon_rtx (XEXP (x, 0)),
ysize, canon_rtx (XEXP (y, 0)), c);
else
return memrefs_conflict_p (xsize, canon_rtx (XEXP (x, 0)),
ysize, y, c);
}
if (GET_CODE (y) == CONST)
return memrefs_conflict_p (xsize, x, ysize,
canon_rtx (XEXP (y, 0)), c);
if (CONSTANT_P (y))
return (maybe_lt (xsize, 0)
|| maybe_lt (ysize, 0)
|| offset_overlap_p (c, xsize, ysize));
return -1;
}
return -1;
}
int
read_dependence (const_rtx mem, const_rtx x)
{
if (MEM_VOLATILE_P (x) && MEM_VOLATILE_P (mem))
return true;
if (MEM_ALIAS_SET (x) == ALIAS_SET_MEMORY_BARRIER
|| MEM_ALIAS_SET (mem) == ALIAS_SET_MEMORY_BARRIER)
return true;
return false;
}
static tree
decl_for_component_ref (tree x)
{
do
{
x = TREE_OPERAND (x, 0);
}
while (x && TREE_CODE (x) == COMPONENT_REF);
return x && DECL_P (x) ? x : NULL_TREE;
}
static void
adjust_offset_for_component_ref (tree x, bool *known_p,
poly_int64 *offset)
{
if (!*known_p)
return;
do
{
tree xoffset = component_ref_field_offset (x);
tree field = TREE_OPERAND (x, 1);
if (TREE_CODE (xoffset) != INTEGER_CST)
{
*known_p = false;
return;
}
offset_int woffset
= (wi::to_offset (xoffset)
+ (wi::to_offset (DECL_FIELD_BIT_OFFSET (field))
>> LOG2_BITS_PER_UNIT));
if (!wi::fits_uhwi_p (woffset))
{
*known_p = false;
return;
}
*offset += woffset.to_uhwi ();
x = TREE_OPERAND (x, 0);
}
while (x && TREE_CODE (x) == COMPONENT_REF);
}
int
nonoverlapping_memrefs_p (const_rtx x, const_rtx y, bool loop_invariant)
{
tree exprx = MEM_EXPR (x), expry = MEM_EXPR (y);
rtx rtlx, rtly;
rtx basex, basey;
bool moffsetx_known_p, moffsety_known_p;
poly_int64 moffsetx = 0, moffsety = 0;
poly_int64 offsetx = 0, offsety = 0, sizex, sizey;
if (exprx == 0 || expry == 0)
return 0;
if ((exprx == get_spill_slot_decl (false)
&& ! MEM_OFFSET_KNOWN_P (x))
|| (expry == get_spill_slot_decl (false)
&& ! MEM_OFFSET_KNOWN_P (y)))
return 0;
moffsetx_known_p = MEM_OFFSET_KNOWN_P (x);
if (moffsetx_known_p)
moffsetx = MEM_OFFSET (x);
if (TREE_CODE (exprx) == COMPONENT_REF)
{
tree t = decl_for_component_ref (exprx);
if (! t)
return 0;
adjust_offset_for_component_ref (exprx, &moffsetx_known_p, &moffsetx);
exprx = t;
}
moffsety_known_p = MEM_OFFSET_KNOWN_P (y);
if (moffsety_known_p)
moffsety = MEM_OFFSET (y);
if (TREE_CODE (expry) == COMPONENT_REF)
{
tree t = decl_for_component_ref (expry);
if (! t)
return 0;
adjust_offset_for_component_ref (expry, &moffsety_known_p, &moffsety);
expry = t;
}
if (! DECL_P (exprx) || ! DECL_P (expry))
return 0;
if (is_gimple_reg (exprx) || is_gimple_reg (expry))
return exprx != expry
|| (moffsetx_known_p && moffsety_known_p
&& MEM_SIZE_KNOWN_P (x) && MEM_SIZE_KNOWN_P (y)
&& !offset_overlap_p (moffsety - moffsetx,
MEM_SIZE (x), MEM_SIZE (y)));
if (TREE_CODE (exprx) == CONST_DECL
|| TREE_CODE (expry) == CONST_DECL)
return 1;
if ((TREE_CODE (exprx) == FUNCTION_DECL
|| TREE_CODE (exprx) == LABEL_DECL)
!= (TREE_CODE (expry) == FUNCTION_DECL
|| TREE_CODE (expry) == LABEL_DECL))
return 1;
if ((!DECL_RTL_SET_P (exprx) && TREE_CODE (exprx) != FUNCTION_DECL)
|| (!DECL_RTL_SET_P (expry) && TREE_CODE (expry) != FUNCTION_DECL))
return 0;
rtlx = DECL_RTL (exprx);
rtly = DECL_RTL (expry);
if ((!MEM_P (rtlx) || !MEM_P (rtly))
&& ! rtx_equal_p (rtlx, rtly))
return 1;
if (MEM_P (rtlx) && MEM_P (rtly)
&& MEM_ADDR_SPACE (rtlx) != MEM_ADDR_SPACE (rtly))
return 0;
basex = MEM_P (rtlx) ? XEXP (rtlx, 0) : rtlx;
basex = strip_offset_and_add (basex, &offsetx);
basey = MEM_P (rtly) ? XEXP (rtly, 0) : rtly;
basey = strip_offset_and_add (basey, &offsety);
if (compare_base_decls (exprx, expry) == 0)
return ((CONSTANT_P (basex) && CONSTANT_P (basey))
|| (CONSTANT_P (basex) && REG_P (basey)
&& REGNO_PTR_FRAME_P (REGNO (basey)))
|| (CONSTANT_P (basey) && REG_P (basex)
&& REGNO_PTR_FRAME_P (REGNO (basex))));
if (loop_invariant)
return 0;
sizex = (!MEM_P (rtlx) ? poly_int64 (GET_MODE_SIZE (GET_MODE (rtlx)))
: MEM_SIZE_KNOWN_P (rtlx) ? MEM_SIZE (rtlx)
: -1);
sizey = (!MEM_P (rtly) ? poly_int64 (GET_MODE_SIZE (GET_MODE (rtly)))
: MEM_SIZE_KNOWN_P (rtly) ? MEM_SIZE (rtly)
: -1);
if (moffsetx_known_p)
offsetx += moffsetx, sizex -= moffsetx;
if (moffsety_known_p)
offsety += moffsety, sizey -= moffsety;
if (MEM_SIZE_KNOWN_P (x) && moffsetx_known_p)
sizex = MEM_SIZE (x);
if (MEM_SIZE_KNOWN_P (y) && moffsety_known_p)
sizey = MEM_SIZE (y);
return !ranges_maybe_overlap_p (offsetx, sizex, offsety, sizey);
}
static int
true_dependence_1 (const_rtx mem, machine_mode mem_mode, rtx mem_addr,
const_rtx x, rtx x_addr, bool mem_canonicalized)
{
rtx true_mem_addr;
rtx base;
int ret;
gcc_checking_assert (mem_canonicalized ? (mem_addr != NULL_RTX)
: (mem_addr == NULL_RTX && x_addr == NULL_RTX));
if (MEM_VOLATILE_P (x) && MEM_VOLATILE_P (mem))
return 1;
if (GET_MODE (x) == BLKmode && GET_CODE (XEXP (x, 0)) == SCRATCH)
return 1;
if (GET_MODE (mem) == BLKmode && GET_CODE (XEXP (mem, 0)) == SCRATCH)
return 1;
if (MEM_ALIAS_SET (x) == ALIAS_SET_MEMORY_BARRIER
|| MEM_ALIAS_SET (mem) == ALIAS_SET_MEMORY_BARRIER)
return 1;
if (! x_addr)
x_addr = XEXP (x, 0);
x_addr = get_addr (x_addr);
if (! mem_addr)
{
mem_addr = XEXP (mem, 0);
if (mem_mode == VOIDmode)
mem_mode = GET_MODE (mem);
}
true_mem_addr = get_addr (mem_addr);
if (MEM_READONLY_P (x)
&& GET_CODE (x_addr) != AND
&& GET_CODE (true_mem_addr) != AND)
return 0;
if (MEM_ADDR_SPACE (mem) != MEM_ADDR_SPACE (x))
return 1;
base = find_base_term (x_addr);
if (base && (GET_CODE (base) == LABEL_REF
|| (GET_CODE (base) == SYMBOL_REF
&& CONSTANT_POOL_ADDRESS_P (base))))
return 0;
rtx mem_base = find_base_term (true_mem_addr);
if (! base_alias_check (x_addr, base, true_mem_addr, mem_base,
GET_MODE (x), mem_mode))
return 0;
x_addr = canon_rtx (x_addr);
if (!mem_canonicalized)
mem_addr = canon_rtx (true_mem_addr);
if ((ret = memrefs_conflict_p (GET_MODE_SIZE (mem_mode), mem_addr,
SIZE_FOR_MODE (x), x_addr, 0)) != -1)
return ret;
if (mems_in_disjoint_alias_sets_p (x, mem))
return 0;
if (nonoverlapping_memrefs_p (mem, x, false))
return 0;
return rtx_refs_may_alias_p (x, mem, true);
}
int
true_dependence (const_rtx mem, machine_mode mem_mode, const_rtx x)
{
return true_dependence_1 (mem, mem_mode, NULL_RTX,
x, NULL_RTX, false);
}
int
canon_true_dependence (const_rtx mem, machine_mode mem_mode, rtx mem_addr,
const_rtx x, rtx x_addr)
{
return true_dependence_1 (mem, mem_mode, mem_addr,
x, x_addr, true);
}
static int
write_dependence_p (const_rtx mem,
const_rtx x, machine_mode x_mode, rtx x_addr,
bool mem_canonicalized, bool x_canonicalized, bool writep)
{
rtx mem_addr;
rtx true_mem_addr, true_x_addr;
rtx base;
int ret;
gcc_checking_assert (x_canonicalized
? (x_addr != NULL_RTX
&& (x_mode != VOIDmode || GET_MODE (x) == VOIDmode))
: (x_addr == NULL_RTX && x_mode == VOIDmode));
if (MEM_VOLATILE_P (x) && MEM_VOLATILE_P (mem))
return 1;
if (GET_MODE (x) == BLKmode && GET_CODE (XEXP (x, 0)) == SCRATCH)
return 1;
if (GET_MODE (mem) == BLKmode && GET_CODE (XEXP (mem, 0)) == SCRATCH)
return 1;
if (MEM_ALIAS_SET (x) == ALIAS_SET_MEMORY_BARRIER
|| MEM_ALIAS_SET (mem) == ALIAS_SET_MEMORY_BARRIER)
return 1;
if (!x_addr)
x_addr = XEXP (x, 0);
true_x_addr = get_addr (x_addr);
mem_addr = XEXP (mem, 0);
true_mem_addr = get_addr (mem_addr);
if (!writep
&& MEM_READONLY_P (mem)
&& GET_CODE (true_x_addr) != AND
&& GET_CODE (true_mem_addr) != AND)
return 0;
if (MEM_ADDR_SPACE (mem) != MEM_ADDR_SPACE (x))
return 1;
base = find_base_term (true_mem_addr);
if (! writep
&& base
&& (GET_CODE (base) == LABEL_REF
|| (GET_CODE (base) == SYMBOL_REF
&& CONSTANT_POOL_ADDRESS_P (base))))
return 0;
rtx x_base = find_base_term (true_x_addr);
if (! base_alias_check (true_x_addr, x_base, true_mem_addr, base,
GET_MODE (x), GET_MODE (mem)))
return 0;
if (!x_canonicalized)
{
x_addr = canon_rtx (true_x_addr);
x_mode = GET_MODE (x);
}
if (!mem_canonicalized)
mem_addr = canon_rtx (true_mem_addr);
if ((ret = memrefs_conflict_p (SIZE_FOR_MODE (mem), mem_addr,
GET_MODE_SIZE (x_mode), x_addr, 0)) != -1)
return ret;
if (nonoverlapping_memrefs_p (x, mem, false))
return 0;
return rtx_refs_may_alias_p (x, mem, false);
}
int
anti_dependence (const_rtx mem, const_rtx x)
{
return write_dependence_p (mem, x, VOIDmode, NULL_RTX,
false,
false, false);
}
int
canon_anti_dependence (const_rtx mem, bool mem_canonicalized,
const_rtx x, machine_mode x_mode, rtx x_addr)
{
return write_dependence_p (mem, x, x_mode, x_addr,
mem_canonicalized, true,
false);
}
int
output_dependence (const_rtx mem, const_rtx x)
{
return write_dependence_p (mem, x, VOIDmode, NULL_RTX,
false,
false, true);
}
int
canon_output_dependence (const_rtx mem, bool mem_canonicalized,
const_rtx x, machine_mode x_mode, rtx x_addr)
{
return write_dependence_p (mem, x, x_mode, x_addr,
mem_canonicalized, true,
true);
}

int
may_alias_p (const_rtx mem, const_rtx x)
{
rtx x_addr, mem_addr;
if (MEM_VOLATILE_P (x) && MEM_VOLATILE_P (mem))
return 1;
if (GET_MODE (x) == BLKmode && GET_CODE (XEXP (x, 0)) == SCRATCH)
return 1;
if (GET_MODE (mem) == BLKmode && GET_CODE (XEXP (mem, 0)) == SCRATCH)
return 1;
if (MEM_ALIAS_SET (x) == ALIAS_SET_MEMORY_BARRIER
|| MEM_ALIAS_SET (mem) == ALIAS_SET_MEMORY_BARRIER)
return 1;
x_addr = XEXP (x, 0);
x_addr = get_addr (x_addr);
mem_addr = XEXP (mem, 0);
mem_addr = get_addr (mem_addr);
if (MEM_READONLY_P (x)
&& GET_CODE (x_addr) != AND
&& GET_CODE (mem_addr) != AND)
return 0;
if (MEM_ADDR_SPACE (mem) != MEM_ADDR_SPACE (x))
return 1;
rtx x_base = find_base_term (x_addr);
rtx mem_base = find_base_term (mem_addr);
if (! base_alias_check (x_addr, x_base, mem_addr, mem_base,
GET_MODE (x), GET_MODE (mem_addr)))
return 0;
if (nonoverlapping_memrefs_p (mem, x, true))
return 0;
return rtx_refs_may_alias_p (x, mem, false);
}
void
init_alias_target (void)
{
int i;
if (!arg_base_value)
arg_base_value = gen_rtx_ADDRESS (VOIDmode, 0);
memset (static_reg_base_value, 0, sizeof static_reg_base_value);
for (i = 0; i < FIRST_PSEUDO_REGISTER; i++)
if (FUNCTION_ARG_REGNO_P (OUTGOING_REGNO (i))
&& targetm.hard_regno_mode_ok (i, Pmode))
static_reg_base_value[i] = arg_base_value;
static_reg_base_value[STACK_POINTER_REGNUM]
= unique_base_value (UNIQUE_BASE_VALUE_SP);
static_reg_base_value[ARG_POINTER_REGNUM]
= unique_base_value (UNIQUE_BASE_VALUE_ARGP);
static_reg_base_value[FRAME_POINTER_REGNUM]
= unique_base_value (UNIQUE_BASE_VALUE_FP);
if (!HARD_FRAME_POINTER_IS_FRAME_POINTER)
static_reg_base_value[HARD_FRAME_POINTER_REGNUM]
= unique_base_value (UNIQUE_BASE_VALUE_HFP);
}
static bool memory_modified;
static void
memory_modified_1 (rtx x, const_rtx pat ATTRIBUTE_UNUSED, void *data)
{
if (MEM_P (x))
{
if (anti_dependence (x, (const_rtx)data) || output_dependence (x, (const_rtx)data))
memory_modified = true;
}
}
bool
memory_modified_in_insn_p (const_rtx mem, const_rtx insn)
{
if (!INSN_P (insn))
return false;
if (CALL_P (insn))
return true;
memory_modified = false;
note_stores (PATTERN (insn), memory_modified_1, CONST_CAST_RTX(mem));
return memory_modified;
}
static inline bool
set_dest_equal_p (const_rtx set, const_rtx item)
{
rtx dest = SET_DEST (set);
return rtx_equal_p (dest, item);
}
void
init_alias_analysis (void)
{
unsigned int maxreg = max_reg_num ();
int changed, pass;
int i;
unsigned int ui;
rtx_insn *insn;
rtx val;
int rpo_cnt;
int *rpo;
timevar_push (TV_ALIAS_ANALYSIS);
vec_safe_grow_cleared (reg_known_value, maxreg - FIRST_PSEUDO_REGISTER);
reg_known_equiv_p = sbitmap_alloc (maxreg - FIRST_PSEUDO_REGISTER);
bitmap_clear (reg_known_equiv_p);
if (old_reg_base_value)
reg_base_value = old_reg_base_value;
if (reg_base_value)
reg_base_value->truncate (0);
vec_safe_grow_cleared (reg_base_value, maxreg);
new_reg_base_value = XNEWVEC (rtx, maxreg);
reg_seen = sbitmap_alloc (maxreg);
rpo = XNEWVEC (int, n_basic_blocks_for_fn (cfun));
rpo_cnt = pre_and_rev_post_order_compute (NULL, rpo, false);
bool could_be_prologue_epilogue = ((targetm.have_prologue ()
|| targetm.have_epilogue ())
&& reload_completed);
pass = 0;
do
{
changed = 0;
unique_id = 1;
copying_arguments = true;
memset (new_reg_base_value, 0, maxreg * sizeof (rtx));
bitmap_clear (reg_seen);
for (i = 0; i < FIRST_PSEUDO_REGISTER; i++)
if (static_reg_base_value[i]
&& !(i == HARD_FRAME_POINTER_REGNUM
&& reload_completed
&& !frame_pointer_needed
&& targetm.can_eliminate (FRAME_POINTER_REGNUM,
STACK_POINTER_REGNUM)))
{
new_reg_base_value[i] = static_reg_base_value[i];
bitmap_set_bit (reg_seen, i);
}
for (i = 0; i < rpo_cnt; i++)
{
basic_block bb = BASIC_BLOCK_FOR_FN (cfun, rpo[i]);
FOR_BB_INSNS (bb, insn)
{
if (NONDEBUG_INSN_P (insn))
{
rtx note, set;
if (could_be_prologue_epilogue
&& prologue_epilogue_contains (insn))
continue;
if (GET_CODE (PATTERN (insn)) == SET
&& REG_NOTES (insn) != 0
&& find_reg_note (insn, REG_NOALIAS, NULL_RTX))
record_set (SET_DEST (PATTERN (insn)), NULL_RTX, NULL);
else
note_stores (PATTERN (insn), record_set, NULL);
set = single_set (insn);
if (set != 0
&& REG_P (SET_DEST (set))
&& REGNO (SET_DEST (set)) >= FIRST_PSEUDO_REGISTER)
{
unsigned int regno = REGNO (SET_DEST (set));
rtx src = SET_SRC (set);
rtx t;
note = find_reg_equal_equiv_note (insn);
if (note && REG_NOTE_KIND (note) == REG_EQUAL
&& DF_REG_DEF_COUNT (regno) != 1)
note = NULL_RTX;
if (note != NULL_RTX
&& GET_CODE (XEXP (note, 0)) != EXPR_LIST
&& ! rtx_varies_p (XEXP (note, 0), 1)
&& ! reg_overlap_mentioned_p (SET_DEST (set),
XEXP (note, 0)))
{
set_reg_known_value (regno, XEXP (note, 0));
set_reg_known_equiv_p (regno,
REG_NOTE_KIND (note) == REG_EQUIV);
}
else if (DF_REG_DEF_COUNT (regno) == 1
&& GET_CODE (src) == PLUS
&& REG_P (XEXP (src, 0))
&& (t = get_reg_known_value (REGNO (XEXP (src, 0))))
&& CONST_INT_P (XEXP (src, 1)))
{
t = plus_constant (GET_MODE (src), t,
INTVAL (XEXP (src, 1)));
set_reg_known_value (regno, t);
set_reg_known_equiv_p (regno, false);
}
else if (DF_REG_DEF_COUNT (regno) == 1
&& ! rtx_varies_p (src, 1))
{
set_reg_known_value (regno, src);
set_reg_known_equiv_p (regno, false);
}
}
}
else if (NOTE_P (insn)
&& NOTE_KIND (insn) == NOTE_INSN_FUNCTION_BEG)
copying_arguments = false;
}
}
gcc_assert (maxreg == (unsigned int) max_reg_num ());
for (ui = 0; ui < maxreg; ui++)
{
if (new_reg_base_value[ui]
&& new_reg_base_value[ui] != (*reg_base_value)[ui]
&& ! rtx_equal_p (new_reg_base_value[ui], (*reg_base_value)[ui]))
{
(*reg_base_value)[ui] = new_reg_base_value[ui];
changed = 1;
}
}
}
while (changed && ++pass < MAX_ALIAS_LOOP_PASSES);
XDELETEVEC (rpo);
FOR_EACH_VEC_ELT (*reg_known_value, i, val)
{
int regno = i + FIRST_PSEUDO_REGISTER;
if (! val)
set_reg_known_value (regno, regno_reg_rtx[regno]);
}
free (new_reg_base_value);
new_reg_base_value = 0;
sbitmap_free (reg_seen);
reg_seen = 0;
timevar_pop (TV_ALIAS_ANALYSIS);
}
void
vt_equate_reg_base_value (const_rtx reg1, const_rtx reg2)
{
(*reg_base_value)[REGNO (reg1)] = REG_BASE_VALUE (reg2);
}
void
end_alias_analysis (void)
{
old_reg_base_value = reg_base_value;
vec_free (reg_known_value);
sbitmap_free (reg_known_equiv_p);
}
void
dump_alias_stats_in_alias_c (FILE *s)
{
fprintf (s, "  TBAA oracle: %llu disambiguations %llu queries\n"
"               %llu are in alias set 0\n"
"               %llu queries asked about the same object\n"
"               %llu queries asked about the same alias set\n"
"               %llu access volatile\n"
"               %llu are dependent in the DAG\n"
"               %llu are aritificially in conflict with void *\n",
alias_stats.num_disambiguated,
alias_stats.num_alias_zero + alias_stats.num_same_alias_set
+ alias_stats.num_same_objects + alias_stats.num_volatile
+ alias_stats.num_dag + alias_stats.num_disambiguated
+ alias_stats.num_universal,
alias_stats.num_alias_zero, alias_stats.num_same_alias_set,
alias_stats.num_same_objects, alias_stats.num_volatile,
alias_stats.num_dag, alias_stats.num_universal);
}
#include "gt-alias.h"

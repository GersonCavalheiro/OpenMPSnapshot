#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "backend.h"
#include "target.h"
#include "rtl.h"
#include "tree.h"
#include "predict.h"
#include "df.h"
#include "memmodel.h"
#include "tm_p.h"
#include "optabs.h"
#include "regs.h"
#include "ira.h"
#include "recog.h"
#include "rtl-error.h"
#include "expr.h"
#include "addresses.h"
#include "cfgrtl.h"
#include "cfgbuild.h"
#include "reload.h"
#include "except.h"
#include "dumpfile.h"
#include "rtl-iter.h"

struct target_reload default_target_reload;
#if SWITCHABLE_TARGET
struct target_reload *this_target_reload = &default_target_reload;
#endif
#define spill_indirect_levels			\
(this_target_reload->x_spill_indirect_levels)
static rtx *reg_last_reload_reg;
static regset_head reg_has_output_reload;
static HARD_REG_SET reg_is_output_reload;
static machine_mode *reg_max_ref_mode;
static short *reg_old_renumber;
static int reg_reloaded_contents[FIRST_PSEUDO_REGISTER];
static rtx_insn *reg_reloaded_insn[FIRST_PSEUDO_REGISTER];
static HARD_REG_SET reg_reloaded_valid;
static HARD_REG_SET reg_reloaded_dead;
static HARD_REG_SET reg_reloaded_call_part_clobbered;
static int n_spills;
static rtx spill_reg_rtx[FIRST_PSEUDO_REGISTER];
static rtx_insn *spill_reg_store[FIRST_PSEUDO_REGISTER];
static rtx spill_reg_stored_to[FIRST_PSEUDO_REGISTER];
static short spill_reg_order[FIRST_PSEUDO_REGISTER];
static HARD_REG_SET bad_spill_regs;
static HARD_REG_SET bad_spill_regs_global;
static short spill_regs[FIRST_PSEUDO_REGISTER];
static HARD_REG_SET *pseudo_previous_regs;
static HARD_REG_SET *pseudo_forbidden_regs;
static HARD_REG_SET used_spill_regs;
static int last_spill_reg;
static rtx spill_stack_slot[FIRST_PSEUDO_REGISTER];
static poly_uint64_pod spill_stack_slot_width[FIRST_PSEUDO_REGISTER];
static regset_head spilled_pseudos;
static regset_head changed_allocation_pseudos;
static regset_head pseudos_counted;
int reload_first_uid;
int caller_save_needed;
int reload_in_progress = 0;
static struct obstack reload_obstack;
static char *reload_startobj;
static char *reload_firstobj;
static char *reload_insn_firstobj;
struct insn_chain *reload_insn_chain;
static bool need_dce;
static struct insn_chain *insns_need_reload;

struct elim_table
{
int from;			
int to;			
poly_int64_pod initial_offset; 
int can_eliminate;		
int can_eliminate_previous;	
poly_int64_pod offset;	
poly_int64_pod previous_offset; 
int ref_outside_mem;		
rtx from_rtx;			
rtx to_rtx;			
};
static struct elim_table *reg_eliminate = 0;
static const struct elim_table_1
{
const int from;
const int to;
} reg_eliminate_1[] =
ELIMINABLE_REGS;
#define NUM_ELIMINABLE_REGS ARRAY_SIZE (reg_eliminate_1)
int num_not_at_initial_offset;
static int num_eliminable;
static int num_eliminable_invariants;
static int first_label_num;
static char *offsets_known_at;
static poly_int64_pod (*offsets_at)[NUM_ELIMINABLE_REGS];
vec<reg_equivs_t, va_gc> *reg_equivs;
typedef rtx *rtx_p;
static vec<rtx_p> substitute_stack;
static int num_labels;

static void replace_pseudos_in (rtx *, machine_mode, rtx);
static void maybe_fix_stack_asms (void);
static void copy_reloads (struct insn_chain *);
static void calculate_needs_all_insns (int);
static int find_reg (struct insn_chain *, int);
static void find_reload_regs (struct insn_chain *);
static void select_reload_regs (void);
static void delete_caller_save_insns (void);
static void spill_failure (rtx_insn *, enum reg_class);
static void count_spilled_pseudo (int, int, int);
static void delete_dead_insn (rtx_insn *);
static void alter_reg (int, int, bool);
static void set_label_offsets (rtx, rtx_insn *, int);
static void check_eliminable_occurrences (rtx);
static void elimination_effects (rtx, machine_mode);
static rtx eliminate_regs_1 (rtx, machine_mode, rtx, bool, bool);
static int eliminate_regs_in_insn (rtx_insn *, int);
static void update_eliminable_offsets (void);
static void mark_not_eliminable (rtx, const_rtx, void *);
static void set_initial_elim_offsets (void);
static bool verify_initial_elim_offsets (void);
static void set_initial_label_offsets (void);
static void set_offsets_for_label (rtx_insn *);
static void init_eliminable_invariants (rtx_insn *, bool);
static void init_elim_table (void);
static void free_reg_equiv (void);
static void update_eliminables (HARD_REG_SET *);
static bool update_eliminables_and_spill (void);
static void elimination_costs_in_insn (rtx_insn *);
static void spill_hard_reg (unsigned int, int);
static int finish_spills (int);
static void scan_paradoxical_subregs (rtx);
static void count_pseudo (int);
static void order_regs_for_reload (struct insn_chain *);
static void reload_as_needed (int);
static void forget_old_reloads_1 (rtx, const_rtx, void *);
static void forget_marked_reloads (regset);
static int reload_reg_class_lower (const void *, const void *);
static void mark_reload_reg_in_use (unsigned int, int, enum reload_type,
machine_mode);
static void clear_reload_reg_in_use (unsigned int, int, enum reload_type,
machine_mode);
static int reload_reg_free_p (unsigned int, int, enum reload_type);
static int reload_reg_free_for_value_p (int, int, int, enum reload_type,
rtx, rtx, int, int);
static int free_for_value_p (int, machine_mode, int, enum reload_type,
rtx, rtx, int, int);
static int allocate_reload_reg (struct insn_chain *, int, int);
static int conflicts_with_override (rtx);
static void failed_reload (rtx_insn *, int);
static int set_reload_reg (int, int);
static void choose_reload_regs_init (struct insn_chain *, rtx *);
static void choose_reload_regs (struct insn_chain *);
static void emit_input_reload_insns (struct insn_chain *, struct reload *,
rtx, int);
static void emit_output_reload_insns (struct insn_chain *, struct reload *,
int);
static void do_input_reload (struct insn_chain *, struct reload *, int);
static void do_output_reload (struct insn_chain *, struct reload *, int);
static void emit_reload_insns (struct insn_chain *);
static void delete_output_reload (rtx_insn *, int, int, rtx);
static void delete_address_reloads (rtx_insn *, rtx_insn *);
static void delete_address_reloads_1 (rtx_insn *, rtx, rtx_insn *);
static void inc_for_reload (rtx, rtx, rtx, poly_int64);
static void add_auto_inc_notes (rtx_insn *, rtx);
static void substitute (rtx *, const_rtx, rtx);
static bool gen_reload_chain_without_interm_reg_p (int, int);
static int reloads_conflict (int, int);
static rtx_insn *gen_reload (rtx, rtx, int, enum reload_type);
static rtx_insn *emit_insn_if_valid_for_reload (rtx);

void
init_reload (void)
{
int i;
rtx tem
= gen_rtx_MEM (Pmode,
gen_rtx_PLUS (Pmode,
gen_rtx_REG (Pmode,
LAST_VIRTUAL_REGISTER + 1),
gen_int_mode (4, Pmode)));
spill_indirect_levels = 0;
while (memory_address_p (QImode, tem))
{
spill_indirect_levels++;
tem = gen_rtx_MEM (Pmode, tem);
}
tem = gen_rtx_MEM (Pmode, gen_rtx_SYMBOL_REF (Pmode, "foo"));
indirect_symref_ok = memory_address_p (QImode, tem);
for (i = 0; i < FIRST_PSEUDO_REGISTER; i++)
{
tem = gen_rtx_PLUS (Pmode,
gen_rtx_REG (Pmode, HARD_FRAME_POINTER_REGNUM),
gen_rtx_REG (Pmode, i));
tem = plus_constant (Pmode, tem, 4);
for (int mode = 0; mode < MAX_MACHINE_MODE; mode++)
if (!double_reg_address_ok[mode]
&& memory_address_p ((enum machine_mode)mode, tem))
double_reg_address_ok[mode] = 1;
}
if (reload_startobj == NULL)
{
gcc_obstack_init (&reload_obstack);
reload_startobj = XOBNEWVAR (&reload_obstack, char, 0);
}
INIT_REG_SET (&spilled_pseudos);
INIT_REG_SET (&changed_allocation_pseudos);
INIT_REG_SET (&pseudos_counted);
}
static struct insn_chain *unused_insn_chains = 0;
struct insn_chain *
new_insn_chain (void)
{
struct insn_chain *c;
if (unused_insn_chains == 0)
{
c = XOBNEW (&reload_obstack, struct insn_chain);
INIT_REG_SET (&c->live_throughout);
INIT_REG_SET (&c->dead_or_set);
}
else
{
c = unused_insn_chains;
unused_insn_chains = c->next;
}
c->is_caller_save_insn = 0;
c->need_operand_change = 0;
c->need_reload = 0;
c->need_elim = 0;
return c;
}
void
compute_use_by_pseudos (HARD_REG_SET *to, regset from)
{
unsigned int regno;
reg_set_iterator rsi;
EXECUTE_IF_SET_IN_REG_SET (from, FIRST_PSEUDO_REGISTER, regno, rsi)
{
int r = reg_renumber[regno];
if (r < 0)
{
gcc_assert (ira_conflicts_p || reload_completed);
}
else
add_to_hard_reg_set (to, PSEUDO_REGNO_MODE (regno), r);
}
}
static void
replace_pseudos_in (rtx *loc, machine_mode mem_mode, rtx usage)
{
rtx x = *loc;
enum rtx_code code;
const char *fmt;
int i, j;
if (! x)
return;
code = GET_CODE (x);
if (code == REG)
{
unsigned int regno = REGNO (x);
if (regno < FIRST_PSEUDO_REGISTER)
return;
x = eliminate_regs_1 (x, mem_mode, usage, true, false);
if (x != *loc)
{
*loc = x;
replace_pseudos_in (loc, mem_mode, usage);
return;
}
if (reg_equiv_constant (regno))
*loc = reg_equiv_constant (regno);
else if (reg_equiv_invariant (regno))
*loc = reg_equiv_invariant (regno);
else if (reg_equiv_mem (regno))
*loc = reg_equiv_mem (regno);
else if (reg_equiv_address (regno))
*loc = gen_rtx_MEM (GET_MODE (x), reg_equiv_address (regno));
else
{
gcc_assert (!REG_P (regno_reg_rtx[regno])
|| REGNO (regno_reg_rtx[regno]) != regno);
*loc = regno_reg_rtx[regno];
}
return;
}
else if (code == MEM)
{
replace_pseudos_in (& XEXP (x, 0), GET_MODE (x), usage);
return;
}
fmt = GET_RTX_FORMAT (code);
for (i = 0; i < GET_RTX_LENGTH (code); i++, fmt++)
if (*fmt == 'e')
replace_pseudos_in (&XEXP (x, i), mem_mode, usage);
else if (*fmt == 'E')
for (j = 0; j < XVECLEN (x, i); j++)
replace_pseudos_in (& XVECEXP (x, i, j), mem_mode, usage);
}
static bool
has_nonexceptional_receiver (void)
{
edge e;
edge_iterator ei;
basic_block *tos, *worklist, bb;
if (!optimize)
return true;
tos = worklist = XNEWVEC (basic_block, n_basic_blocks_for_fn (cfun) + 1);
FOR_EACH_BB_FN (bb, cfun)
bb->flags &= ~BB_REACHABLE;
EXIT_BLOCK_PTR_FOR_FN (cfun)->flags |= BB_REACHABLE;
*tos++ = EXIT_BLOCK_PTR_FOR_FN (cfun);
while (tos != worklist)
{
bb = *--tos;
FOR_EACH_EDGE (e, ei, bb->preds)
if (!(e->flags & EDGE_ABNORMAL))
{
basic_block src = e->src;
if (!(src->flags & BB_REACHABLE))
{
src->flags |= BB_REACHABLE;
*tos++ = src;
}
}
}
free (worklist);
FOR_EACH_BB_FN (bb, cfun)
if (bb->flags & BB_REACHABLE && bb_has_abnormal_pred (bb))
return true;
return false;
}
void
grow_reg_equivs (void)
{
int old_size = vec_safe_length (reg_equivs);
int max_regno = max_reg_num ();
int i;
reg_equivs_t ze;
memset (&ze, 0, sizeof (reg_equivs_t));
vec_safe_reserve (reg_equivs, max_regno);
for (i = old_size; i < max_regno; i++)
reg_equivs->quick_insert (i, ze);
}

static basic_block elim_bb;
static int something_needs_elimination;
static int something_needs_operands_changed;
static bool something_was_spilled;
static int failure;
static int *temp_pseudo_reg_arr;
static void
remove_init_insns ()
{
for (int i = FIRST_PSEUDO_REGISTER; i < max_regno; i++)
{
if (reg_renumber[i] < 0 && reg_equiv_init (i) != 0)
{
rtx list;
for (list = reg_equiv_init (i); list; list = XEXP (list, 1))
{
rtx_insn *equiv_insn = as_a <rtx_insn *> (XEXP (list, 0));
if (NOTE_P (equiv_insn)
|| can_throw_internal (equiv_insn))
;
else if (reg_set_p (regno_reg_rtx[i], PATTERN (equiv_insn)))
delete_dead_insn (equiv_insn);
else
SET_INSN_DELETED (equiv_insn);
}
}
}
}
static bool
will_delete_init_insn_p (rtx_insn *insn)
{
rtx set = single_set (insn);
if (!set || !REG_P (SET_DEST (set)))
return false;
unsigned regno = REGNO (SET_DEST (set));
if (can_throw_internal (insn))
return false;
if (regno < FIRST_PSEUDO_REGISTER || reg_renumber[regno] >= 0)
return false;
for (rtx list = reg_equiv_init (regno); list; list = XEXP (list, 1))
{
rtx equiv_insn = XEXP (list, 0);
if (equiv_insn == insn)
return true;
}
return false;
}
bool
reload (rtx_insn *first, int global)
{
int i, n;
rtx_insn *insn;
struct elim_table *ep;
basic_block bb;
bool inserted;
init_recog ();
failure = 0;
reload_firstobj = XOBNEWVAR (&reload_obstack, char, 0);
emit_note (NOTE_INSN_DELETED);
reload_first_uid = get_max_uid ();
clear_secondary_mem ();
memset (spill_stack_slot, 0, sizeof spill_stack_slot);
memset (spill_stack_slot_width, 0, sizeof spill_stack_slot_width);
init_save_areas ();
for (i = FIRST_PSEUDO_REGISTER; i < max_regno; i++)
mark_home_live (i);
if (cfun->has_nonlocal_label
&& has_nonexceptional_receiver ())
crtl->saves_all_registers = 1;
if (crtl->saves_all_registers)
for (i = 0; i < FIRST_PSEUDO_REGISTER; i++)
if (! call_used_regs[i] && ! fixed_regs[i] && ! LOCAL_REGNO (i))
df_set_regs_ever_live (i, true);
grow_reg_equivs ();
reg_old_renumber = XCNEWVEC (short, max_regno);
memcpy (reg_old_renumber, reg_renumber, max_regno * sizeof (short));
pseudo_forbidden_regs = XNEWVEC (HARD_REG_SET, max_regno);
pseudo_previous_regs = XCNEWVEC (HARD_REG_SET, max_regno);
CLEAR_HARD_REG_SET (bad_spill_regs_global);
init_eliminable_invariants (first, true);
init_elim_table ();
temp_pseudo_reg_arr = XNEWVEC (int, max_regno - LAST_VIRTUAL_REGISTER - 1);
for (n = 0, i = LAST_VIRTUAL_REGISTER + 1; i < max_regno; i++)
temp_pseudo_reg_arr[n++] = i;
if (ira_conflicts_p)
ira_sort_regnos_for_alter_reg (temp_pseudo_reg_arr, n, reg_max_ref_mode);
for (i = 0; i < n; i++)
alter_reg (temp_pseudo_reg_arr[i], -1, false);
for (insn = first; insn && num_eliminable; insn = NEXT_INSN (insn))
if (INSN_P (insn))
note_stores (PATTERN (insn), mark_not_eliminable, NULL);
maybe_fix_stack_asms ();
insns_need_reload = 0;
something_needs_elimination = 0;
last_spill_reg = -1;
CLEAR_HARD_REG_SET (used_spill_regs);
for (ep = reg_eliminate; ep < &reg_eliminate[NUM_ELIMINABLE_REGS]; )
{
int from = ep->from;
int can_eliminate = 0;
do
{
can_eliminate |= ep->can_eliminate;
ep++;
}
while (ep < &reg_eliminate[NUM_ELIMINABLE_REGS] && ep->from == from);
if (! can_eliminate)
spill_hard_reg (from, 1);
}
if (!HARD_FRAME_POINTER_IS_FRAME_POINTER && frame_pointer_needed)
spill_hard_reg (HARD_FRAME_POINTER_REGNUM, 1);
finish_spills (global);
reload_in_progress = 1;
for (;;)
{
int something_changed;
poly_int64 starting_frame_size;
starting_frame_size = get_frame_size ();
something_was_spilled = false;
set_initial_elim_offsets ();
set_initial_label_offsets ();
for (i = FIRST_PSEUDO_REGISTER; i < max_regno; i++)
if (reg_renumber[i] < 0 && reg_equiv_memory_loc (i))
{
rtx x = eliminate_regs (reg_equiv_memory_loc (i), VOIDmode,
NULL_RTX);
if (strict_memory_address_addr_space_p
(GET_MODE (regno_reg_rtx[i]), XEXP (x, 0),
MEM_ADDR_SPACE (x)))
reg_equiv_mem (i) = x, reg_equiv_address (i) = 0;
else if (CONSTANT_P (XEXP (x, 0))
|| (REG_P (XEXP (x, 0))
&& REGNO (XEXP (x, 0)) < FIRST_PSEUDO_REGISTER)
|| (GET_CODE (XEXP (x, 0)) == PLUS
&& REG_P (XEXP (XEXP (x, 0), 0))
&& (REGNO (XEXP (XEXP (x, 0), 0))
< FIRST_PSEUDO_REGISTER)
&& CONSTANT_P (XEXP (XEXP (x, 0), 1))))
reg_equiv_address (i) = XEXP (x, 0), reg_equiv_mem (i) = 0;
else
{
reg_equiv_memory_loc (i) = 0;
reg_equiv_init (i) = 0;
alter_reg (i, -1, true);
}
}
if (caller_save_needed)
setup_save_areas ();
if (maybe_ne (starting_frame_size, 0) && crtl->stack_alignment_needed)
{
assign_stack_local (BLKmode, 0, crtl->stack_alignment_needed);
}
if (something_was_spilled
|| maybe_ne (starting_frame_size, get_frame_size ()))
{
if (update_eliminables_and_spill ())
finish_spills (0);
continue;
}
if (caller_save_needed)
{
save_call_clobbered_regs ();
reload_firstobj = XOBNEWVAR (&reload_obstack, char, 0);
}
calculate_needs_all_insns (global);
if (! ira_conflicts_p)
CLEAR_REG_SET (&spilled_pseudos);
something_changed = 0;
if (something_was_spilled
|| maybe_ne (starting_frame_size, get_frame_size ()))
something_changed = 1;
else if (!verify_initial_elim_offsets ())
something_changed = 1;
if (update_eliminables_and_spill ())
{
finish_spills (0);
something_changed = 1;
}
else
{
select_reload_regs ();
if (failure)
goto failed;
if (insns_need_reload)
something_changed |= finish_spills (global);
}
if (! something_changed)
break;
if (caller_save_needed)
delete_caller_save_insns ();
obstack_free (&reload_obstack, reload_firstobj);
}
if (global)
for (ep = reg_eliminate; ep < &reg_eliminate[NUM_ELIMINABLE_REGS]; ep++)
if (ep->can_eliminate)
mark_elimination (ep->from, ep->to);
remove_init_insns ();
if (insns_need_reload != 0 || something_needs_elimination
|| something_needs_operands_changed)
{
poly_int64 old_frame_size = get_frame_size ();
reload_as_needed (global);
gcc_assert (known_eq (old_frame_size, get_frame_size ()));
gcc_assert (verify_initial_elim_offsets ());
}
if (! frame_pointer_needed)
FOR_EACH_BB_FN (bb, cfun)
bitmap_clear_bit (df_get_live_in (bb), HARD_FRAME_POINTER_REGNUM);
failed:
CLEAR_REG_SET (&changed_allocation_pseudos);
CLEAR_REG_SET (&spilled_pseudos);
reload_in_progress = 0;
for (i = FIRST_PSEUDO_REGISTER; i < max_regno; i++)
{
rtx addr = 0;
if (reg_equiv_mem (i))
addr = XEXP (reg_equiv_mem (i), 0);
if (reg_equiv_address (i))
addr = reg_equiv_address (i);
if (addr)
{
if (reg_renumber[i] < 0)
{
rtx reg = regno_reg_rtx[i];
REG_USERVAR_P (reg) = 0;
PUT_CODE (reg, MEM);
XEXP (reg, 0) = addr;
if (reg_equiv_memory_loc (i))
MEM_COPY_ATTRIBUTES (reg, reg_equiv_memory_loc (i));
else
MEM_ATTRS (reg) = 0;
MEM_NOTRAP_P (reg) = 1;
}
else if (reg_equiv_mem (i))
XEXP (reg_equiv_mem (i), 0) = addr;
}
if (MAY_HAVE_DEBUG_BIND_INSNS && reg_renumber[i] < 0)
{
rtx reg = regno_reg_rtx[i];
rtx equiv = 0;
df_ref use, next;
if (reg_equiv_constant (i))
equiv = reg_equiv_constant (i);
else if (reg_equiv_invariant (i))
equiv = reg_equiv_invariant (i);
else if (reg && MEM_P (reg))
equiv = targetm.delegitimize_address (reg);
else if (reg && REG_P (reg) && (int)REGNO (reg) != i)
equiv = reg;
if (equiv == reg)
continue;
for (use = DF_REG_USE_CHAIN (i); use; use = next)
{
insn = DF_REF_INSN (use);
next = DF_REF_NEXT_REG (use);
while (next && DF_REF_INSN (next) == insn)
next = DF_REF_NEXT_REG (next);
if (DEBUG_BIND_INSN_P (insn))
{
if (!equiv)
{
INSN_VAR_LOCATION_LOC (insn) = gen_rtx_UNKNOWN_VAR_LOC ();
df_insn_rescan_debug_internal (insn);
}
else
INSN_VAR_LOCATION_LOC (insn)
= simplify_replace_rtx (INSN_VAR_LOCATION_LOC (insn),
reg, equiv);
}
}
}
}
reload_completed = 1;
for (insn = first; insn; insn = NEXT_INSN (insn))
if (INSN_P (insn))
{
rtx *pnote;
if (CALL_P (insn))
replace_pseudos_in (& CALL_INSN_FUNCTION_USAGE (insn),
VOIDmode, CALL_INSN_FUNCTION_USAGE (insn));
if ((GET_CODE (PATTERN (insn)) == USE
&& (GET_MODE (insn) == QImode
|| find_reg_note (insn, REG_EQUAL, NULL_RTX)))
|| (GET_CODE (PATTERN (insn)) == CLOBBER
&& (!MEM_P (XEXP (PATTERN (insn), 0))
|| GET_MODE (XEXP (PATTERN (insn), 0)) != BLKmode
|| (GET_CODE (XEXP (XEXP (PATTERN (insn), 0), 0)) != SCRATCH
&& XEXP (XEXP (PATTERN (insn), 0), 0)
!= stack_pointer_rtx))
&& (!REG_P (XEXP (PATTERN (insn), 0))
|| ! REG_FUNCTION_VALUE_P (XEXP (PATTERN (insn), 0)))))
{
delete_insn (insn);
continue;
}
if (GET_CODE (PATTERN (insn)) == CLOBBER)
replace_pseudos_in (& XEXP (PATTERN (insn), 0),
VOIDmode, PATTERN (insn));
if (NONJUMP_INSN_P (insn)
&& GET_CODE (PATTERN (insn)) == SET
&& REG_P (SET_SRC (PATTERN (insn)))
&& REG_P (SET_DEST (PATTERN (insn)))
&& (REGNO (SET_SRC (PATTERN (insn)))
== REGNO (SET_DEST (PATTERN (insn)))))
{
delete_insn (insn);
continue;
}
pnote = &REG_NOTES (insn);
while (*pnote != 0)
{
if (REG_NOTE_KIND (*pnote) == REG_DEAD
|| REG_NOTE_KIND (*pnote) == REG_UNUSED
|| REG_NOTE_KIND (*pnote) == REG_INC)
*pnote = XEXP (*pnote, 1);
else
pnote = &XEXP (*pnote, 1);
}
if (AUTO_INC_DEC)
add_auto_inc_notes (insn, PATTERN (insn));
cleanup_subreg_operands (insn);
if (asm_noperands (PATTERN (insn)) >= 0)
{
extract_insn (insn);
if (!constrain_operands (1, get_enabled_alternatives (insn)))
{
error_for_asm (insn,
"%<asm%> operand has impossible constraints");
delete_insn (insn);
continue;
}
}
}
free (temp_pseudo_reg_arr);
free_reg_equiv ();
free (reg_max_ref_mode);
free (reg_old_renumber);
free (pseudo_previous_regs);
free (pseudo_forbidden_regs);
CLEAR_HARD_REG_SET (used_spill_regs);
for (i = 0; i < n_spills; i++)
SET_HARD_REG_BIT (used_spill_regs, spill_regs[i]);
obstack_free (&reload_obstack, reload_startobj);
unused_insn_chains = 0;
inserted = fixup_abnormal_edges ();
if (cfun->can_throw_non_call_exceptions)
{
auto_sbitmap blocks (last_basic_block_for_fn (cfun));
bitmap_ones (blocks);
find_many_sub_basic_blocks (blocks);
}
if (inserted)
commit_edge_insertions ();
unshare_all_rtl_again (first);
#ifdef STACK_BOUNDARY
if (!frame_pointer_needed)
REGNO_POINTER_ALIGN (HARD_FRAME_POINTER_REGNUM) = BITS_PER_UNIT;
#endif
substitute_stack.release ();
gcc_assert (bitmap_empty_p (&spilled_pseudos));
reload_completed = !failure;
return need_dce;
}
static void
maybe_fix_stack_asms (void)
{
#ifdef STACK_REGS
const char *constraints[MAX_RECOG_OPERANDS];
machine_mode operand_mode[MAX_RECOG_OPERANDS];
struct insn_chain *chain;
for (chain = reload_insn_chain; chain != 0; chain = chain->next)
{
int i, noperands;
HARD_REG_SET clobbered, allowed;
rtx pat;
if (! INSN_P (chain->insn)
|| (noperands = asm_noperands (PATTERN (chain->insn))) < 0)
continue;
pat = PATTERN (chain->insn);
if (GET_CODE (pat) != PARALLEL)
continue;
CLEAR_HARD_REG_SET (clobbered);
CLEAR_HARD_REG_SET (allowed);
for (i = 0; i < XVECLEN (pat, 0); i++)
{
rtx t = XVECEXP (pat, 0, i);
if (GET_CODE (t) == CLOBBER && STACK_REG_P (XEXP (t, 0)))
SET_HARD_REG_BIT (clobbered, REGNO (XEXP (t, 0)));
}
decode_asm_operands (pat, recog_data.operand, recog_data.operand_loc,
constraints, operand_mode, NULL);
for (i = 0; i < noperands; i++)
{
const char *p = constraints[i];
int cls = (int) NO_REGS;
for (;;)
{
char c = *p;
if (c == '\0' || c == ',' || c == '#')
{
IOR_HARD_REG_SET (allowed, reg_class_contents[cls]);
cls = NO_REGS;
p++;
if (c == '#')
do {
c = *p++;
} while (c != '\0' && c != ',');
if (c == '\0')
break;
continue;
}
switch (c)
{
case 'g':
cls = (int) reg_class_subunion[cls][(int) GENERAL_REGS];
break;
default:
enum constraint_num cn = lookup_constraint (p);
if (insn_extra_address_constraint (cn))
cls = (int) reg_class_subunion[cls]
[(int) base_reg_class (VOIDmode, ADDR_SPACE_GENERIC,
ADDRESS, SCRATCH)];
else
cls = (int) reg_class_subunion[cls]
[reg_class_for_constraint (cn)];
break;
}
p += CONSTRAINT_LEN (c, p);
}
}
AND_HARD_REG_SET (allowed, clobbered);
for (i = 0; i < FIRST_PSEUDO_REGISTER; i++)
if (TEST_HARD_REG_BIT (allowed, i))
{
CLEAR_REGNO_REG_SET (&chain->live_throughout, i);
CLEAR_REGNO_REG_SET (&chain->dead_or_set, i);
}
}
#endif
}

static void
copy_reloads (struct insn_chain *chain)
{
chain->n_reloads = n_reloads;
chain->rld = XOBNEWVEC (&reload_obstack, struct reload, n_reloads);
memcpy (chain->rld, rld, n_reloads * sizeof (struct reload));
reload_insn_firstobj = XOBNEWVAR (&reload_obstack, char, 0);
}
static void
calculate_needs_all_insns (int global)
{
struct insn_chain **pprev_reload = &insns_need_reload;
struct insn_chain *chain, *next = 0;
something_needs_elimination = 0;
reload_insn_firstobj = XOBNEWVAR (&reload_obstack, char, 0);
for (chain = reload_insn_chain; chain != 0; chain = next)
{
rtx_insn *insn = chain->insn;
next = chain->next;
chain->n_reloads = 0;
chain->need_elim = 0;
chain->need_reload = 0;
chain->need_operand_change = 0;
if (LABEL_P (insn) || JUMP_P (insn) || JUMP_TABLE_DATA_P (insn)
|| (INSN_P (insn) && REG_NOTES (insn) != 0))
set_label_offsets (insn, insn, 0);
if (INSN_P (insn))
{
rtx old_body = PATTERN (insn);
int old_code = INSN_CODE (insn);
rtx old_notes = REG_NOTES (insn);
int did_elimination = 0;
int operands_changed = 0;
if (will_delete_init_insn_p (insn))
continue;
if (num_eliminable || num_eliminable_invariants)
did_elimination = eliminate_regs_in_insn (insn, 0);
operands_changed = find_reloads (insn, 0, spill_indirect_levels,
global, spill_reg_order);
if (flag_expensive_optimizations && n_reloads > 1)
{
rtx set = single_set (insn);
if (set
&&
((SET_SRC (set) == SET_DEST (set)
&& REG_P (SET_SRC (set))
&& REGNO (SET_SRC (set)) >= FIRST_PSEUDO_REGISTER)
|| (REG_P (SET_SRC (set)) && REG_P (SET_DEST (set))
&& reg_renumber[REGNO (SET_SRC (set))] < 0
&& reg_renumber[REGNO (SET_DEST (set))] < 0
&& reg_equiv_memory_loc (REGNO (SET_SRC (set))) != NULL
&& reg_equiv_memory_loc (REGNO (SET_DEST (set))) != NULL
&& rtx_equal_p (reg_equiv_memory_loc (REGNO (SET_SRC (set))),
reg_equiv_memory_loc (REGNO (SET_DEST (set)))))))
{
if (ira_conflicts_p)
ira_mark_memory_move_deletion (REGNO (SET_DEST (set)),
REGNO (SET_SRC (set)));
delete_insn (insn);
if (chain->prev)
chain->prev->next = next;
else
reload_insn_chain = next;
if (next)
next->prev = chain->prev;
chain->next = unused_insn_chains;
unused_insn_chains = chain;
continue;
}
}
if (num_eliminable)
update_eliminable_offsets ();
chain->need_elim = did_elimination;
chain->need_reload = n_reloads > 0;
chain->need_operand_change = operands_changed;
if (did_elimination)
{
obstack_free (&reload_obstack, reload_insn_firstobj);
PATTERN (insn) = old_body;
INSN_CODE (insn) = old_code;
REG_NOTES (insn) = old_notes;
something_needs_elimination = 1;
}
something_needs_operands_changed |= operands_changed;
if (n_reloads != 0)
{
copy_reloads (chain);
*pprev_reload = chain;
pprev_reload = &chain->next_need_reload;
}
}
}
*pprev_reload = 0;
}

void
calculate_elim_costs_all_insns (void)
{
int *reg_equiv_init_cost;
basic_block bb;
int i;
reg_equiv_init_cost = XCNEWVEC (int, max_regno);
init_elim_table ();
init_eliminable_invariants (get_insns (), false);
set_initial_elim_offsets ();
set_initial_label_offsets ();
FOR_EACH_BB_FN (bb, cfun)
{
rtx_insn *insn;
elim_bb = bb;
FOR_BB_INSNS (bb, insn)
{
if (LABEL_P (insn) || JUMP_P (insn) || JUMP_TABLE_DATA_P (insn)
|| (INSN_P (insn) && REG_NOTES (insn) != 0))
set_label_offsets (insn, insn, 0);
if (INSN_P (insn))
{
rtx set = single_set (insn);
if (set && REG_P (SET_DEST (set))
&& reg_renumber[REGNO (SET_DEST (set))] < 0
&& (reg_equiv_constant (REGNO (SET_DEST (set)))
|| reg_equiv_invariant (REGNO (SET_DEST (set)))))
{
unsigned regno = REGNO (SET_DEST (set));
rtx_insn_list *init = reg_equiv_init (regno);
if (init)
{
rtx t = eliminate_regs_1 (SET_SRC (set), VOIDmode, insn,
false, true);
machine_mode mode = GET_MODE (SET_DEST (set));
int cost = set_src_cost (t, mode,
optimize_bb_for_speed_p (bb));
int freq = REG_FREQ_FROM_BB (bb);
reg_equiv_init_cost[regno] = cost * freq;
continue;
}
}
if (num_eliminable || num_eliminable_invariants)
elimination_costs_in_insn (insn);
if (num_eliminable)
update_eliminable_offsets ();
}
}
}
for (i = FIRST_PSEUDO_REGISTER; i < max_regno; i++)
{
if (reg_equiv_invariant (i))
{
if (reg_equiv_init (i))
{
int cost = reg_equiv_init_cost[i];
if (dump_file)
fprintf (dump_file,
"Reg %d has equivalence, initial gains %d\n", i, cost);
if (cost != 0)
ira_adjust_equiv_reg_cost (i, cost);
}
else
{
if (dump_file)
fprintf (dump_file,
"Reg %d had equivalence, but can't be eliminated\n",
i);
ira_adjust_equiv_reg_cost (i, 0);
}
}
}
free (reg_equiv_init_cost);
free (offsets_known_at);
free (offsets_at);
offsets_at = NULL;
offsets_known_at = NULL;
}

static int
reload_reg_class_lower (const void *r1p, const void *r2p)
{
int r1 = *(const short *) r1p, r2 = *(const short *) r2p;
int t;
t = rld[r1].optional - rld[r2].optional;
if (t != 0)
return t;
t = ((reg_class_size[(int) rld[r2].rclass] == 1)
- (reg_class_size[(int) rld[r1].rclass] == 1));
if (t != 0)
return t;
t = rld[r2].nregs - rld[r1].nregs;
if (t != 0)
return t;
t = (int) rld[r1].rclass - (int) rld[r2].rclass;
if (t != 0)
return t;
return r1 - r2;
}

static int spill_cost[FIRST_PSEUDO_REGISTER];
static int spill_add_cost[FIRST_PSEUDO_REGISTER];
static int hard_regno_to_pseudo_regno[FIRST_PSEUDO_REGISTER];
static void
count_pseudo (int reg)
{
int freq = REG_FREQ (reg);
int r = reg_renumber[reg];
int nregs;
if (ira_conflicts_p && r < 0)
return;
if (REGNO_REG_SET_P (&pseudos_counted, reg)
|| REGNO_REG_SET_P (&spilled_pseudos, reg))
return;
SET_REGNO_REG_SET (&pseudos_counted, reg);
gcc_assert (r >= 0);
spill_add_cost[r] += freq;
nregs = hard_regno_nregs (r, PSEUDO_REGNO_MODE (reg));
while (nregs-- > 0)
{
hard_regno_to_pseudo_regno[r + nregs] = reg;
spill_cost[r + nregs] += freq;
}
}
static void
order_regs_for_reload (struct insn_chain *chain)
{
unsigned i;
HARD_REG_SET used_by_pseudos;
HARD_REG_SET used_by_pseudos2;
reg_set_iterator rsi;
COPY_HARD_REG_SET (bad_spill_regs, fixed_reg_set);
memset (spill_cost, 0, sizeof spill_cost);
memset (spill_add_cost, 0, sizeof spill_add_cost);
for (i = 0; i < FIRST_PSEUDO_REGISTER; i++)
hard_regno_to_pseudo_regno[i] = -1;
REG_SET_TO_HARD_REG_SET (used_by_pseudos, &chain->live_throughout);
REG_SET_TO_HARD_REG_SET (used_by_pseudos2, &chain->dead_or_set);
IOR_HARD_REG_SET (bad_spill_regs, used_by_pseudos);
IOR_HARD_REG_SET (bad_spill_regs, used_by_pseudos2);
CLEAR_REG_SET (&pseudos_counted);
EXECUTE_IF_SET_IN_REG_SET
(&chain->live_throughout, FIRST_PSEUDO_REGISTER, i, rsi)
{
count_pseudo (i);
}
EXECUTE_IF_SET_IN_REG_SET
(&chain->dead_or_set, FIRST_PSEUDO_REGISTER, i, rsi)
{
count_pseudo (i);
}
CLEAR_REG_SET (&pseudos_counted);
}

static short reload_order[MAX_RELOADS];
static HARD_REG_SET used_spill_regs_local;
static void
count_spilled_pseudo (int spilled, int spilled_nregs, int reg)
{
int freq = REG_FREQ (reg);
int r = reg_renumber[reg];
int nregs;
if (ira_conflicts_p && r < 0)
return;
gcc_assert (r >= 0);
nregs = hard_regno_nregs (r, PSEUDO_REGNO_MODE (reg));
if (REGNO_REG_SET_P (&spilled_pseudos, reg)
|| spilled + spilled_nregs <= r || r + nregs <= spilled)
return;
SET_REGNO_REG_SET (&spilled_pseudos, reg);
spill_add_cost[r] -= freq;
while (nregs-- > 0)
{
hard_regno_to_pseudo_regno[r + nregs] = -1;
spill_cost[r + nregs] -= freq;
}
}
static int
find_reg (struct insn_chain *chain, int order)
{
int rnum = reload_order[order];
struct reload *rl = rld + rnum;
int best_cost = INT_MAX;
int best_reg = -1;
unsigned int i, j, n;
int k;
HARD_REG_SET not_usable;
HARD_REG_SET used_by_other_reload;
reg_set_iterator rsi;
static int regno_pseudo_regs[FIRST_PSEUDO_REGISTER];
static int best_regno_pseudo_regs[FIRST_PSEUDO_REGISTER];
COPY_HARD_REG_SET (not_usable, bad_spill_regs);
IOR_HARD_REG_SET (not_usable, bad_spill_regs_global);
IOR_COMPL_HARD_REG_SET (not_usable, reg_class_contents[rl->rclass]);
CLEAR_HARD_REG_SET (used_by_other_reload);
for (k = 0; k < order; k++)
{
int other = reload_order[k];
if (rld[other].regno >= 0 && reloads_conflict (other, rnum))
for (j = 0; j < rld[other].nregs; j++)
SET_HARD_REG_BIT (used_by_other_reload, rld[other].regno + j);
}
for (i = 0; i < FIRST_PSEUDO_REGISTER; i++)
{
#ifdef REG_ALLOC_ORDER
unsigned int regno = reg_alloc_order[i];
#else
unsigned int regno = i;
#endif
if (! TEST_HARD_REG_BIT (not_usable, regno)
&& ! TEST_HARD_REG_BIT (used_by_other_reload, regno)
&& targetm.hard_regno_mode_ok (regno, rl->mode))
{
int this_cost = spill_cost[regno];
int ok = 1;
unsigned int this_nregs = hard_regno_nregs (regno, rl->mode);
for (j = 1; j < this_nregs; j++)
{
this_cost += spill_add_cost[regno + j];
if ((TEST_HARD_REG_BIT (not_usable, regno + j))
|| TEST_HARD_REG_BIT (used_by_other_reload, regno + j))
ok = 0;
}
if (! ok)
continue;
if (ira_conflicts_p)
{
for (n = j = 0; j < this_nregs; j++)
{
int r = hard_regno_to_pseudo_regno[regno + j];
if (r < 0)
continue;
if (n == 0 || regno_pseudo_regs[n - 1] != r)
regno_pseudo_regs[n++] = r;
}
regno_pseudo_regs[n++] = -1;
if (best_reg < 0
|| ira_better_spill_reload_regno_p (regno_pseudo_regs,
best_regno_pseudo_regs,
rl->in, rl->out,
chain->insn))
{
best_reg = regno;
for (j = 0;; j++)
{
best_regno_pseudo_regs[j] = regno_pseudo_regs[j];
if (regno_pseudo_regs[j] < 0)
break;
}
}
continue;
}
if (rl->in && REG_P (rl->in) && REGNO (rl->in) == regno)
this_cost--;
if (rl->out && REG_P (rl->out) && REGNO (rl->out) == regno)
this_cost--;
if (this_cost < best_cost
|| (this_cost == best_cost
#ifdef REG_ALLOC_ORDER
&& (inv_reg_alloc_order[regno]
< inv_reg_alloc_order[best_reg])
#else
&& call_used_regs[regno]
&& ! call_used_regs[best_reg]
#endif
))
{
best_reg = regno;
best_cost = this_cost;
}
}
}
if (best_reg == -1)
return 0;
if (dump_file)
fprintf (dump_file, "Using reg %d for reload %d\n", best_reg, rnum);
rl->nregs = hard_regno_nregs (best_reg, rl->mode);
rl->regno = best_reg;
EXECUTE_IF_SET_IN_REG_SET
(&chain->live_throughout, FIRST_PSEUDO_REGISTER, j, rsi)
{
count_spilled_pseudo (best_reg, rl->nregs, j);
}
EXECUTE_IF_SET_IN_REG_SET
(&chain->dead_or_set, FIRST_PSEUDO_REGISTER, j, rsi)
{
count_spilled_pseudo (best_reg, rl->nregs, j);
}
for (i = 0; i < rl->nregs; i++)
{
gcc_assert (spill_cost[best_reg + i] == 0);
gcc_assert (spill_add_cost[best_reg + i] == 0);
gcc_assert (hard_regno_to_pseudo_regno[best_reg + i] == -1);
SET_HARD_REG_BIT (used_spill_regs_local, best_reg + i);
}
return 1;
}
static void
find_reload_regs (struct insn_chain *chain)
{
int i;
for (i = 0; i < chain->n_reloads; i++)
{
if (chain->rld[i].reg_rtx)
{
chain->rld[i].regno = REGNO (chain->rld[i].reg_rtx);
chain->rld[i].nregs = REG_NREGS (chain->rld[i].reg_rtx);
}
else
chain->rld[i].regno = -1;
reload_order[i] = i;
}
n_reloads = chain->n_reloads;
memcpy (rld, chain->rld, n_reloads * sizeof (struct reload));
CLEAR_HARD_REG_SET (used_spill_regs_local);
if (dump_file)
fprintf (dump_file, "Spilling for insn %d.\n", INSN_UID (chain->insn));
qsort (reload_order, n_reloads, sizeof (short), reload_reg_class_lower);
order_regs_for_reload (chain);
for (i = 0; i < n_reloads; i++)
{
int r = reload_order[i];
if ((rld[r].out != 0 || rld[r].in != 0 || rld[r].secondary_p)
&& ! rld[r].optional
&& rld[r].regno == -1)
if (! find_reg (chain, i))
{
if (dump_file)
fprintf (dump_file, "reload failure for reload %d\n", r);
spill_failure (chain->insn, rld[r].rclass);
failure = 1;
return;
}
}
COPY_HARD_REG_SET (chain->used_spill_regs, used_spill_regs_local);
IOR_HARD_REG_SET (used_spill_regs, used_spill_regs_local);
memcpy (chain->rld, rld, n_reloads * sizeof (struct reload));
}
static void
select_reload_regs (void)
{
struct insn_chain *chain;
for (chain = insns_need_reload; chain != 0;
chain = chain->next_need_reload)
find_reload_regs (chain);
}

static void
delete_caller_save_insns (void)
{
struct insn_chain *c = reload_insn_chain;
while (c != 0)
{
while (c != 0 && c->is_caller_save_insn)
{
struct insn_chain *next = c->next;
rtx_insn *insn = c->insn;
if (c == reload_insn_chain)
reload_insn_chain = next;
delete_insn (insn);
if (next)
next->prev = c->prev;
if (c->prev)
c->prev->next = next;
c->next = unused_insn_chains;
unused_insn_chains = c;
c = next;
}
if (c != 0)
c = c->next;
}
}

static void
spill_failure (rtx_insn *insn, enum reg_class rclass)
{
if (asm_noperands (PATTERN (insn)) >= 0)
error_for_asm (insn, "can%'t find a register in class %qs while "
"reloading %<asm%>",
reg_class_names[rclass]);
else
{
error ("unable to find a register to spill in class %qs",
reg_class_names[rclass]);
if (dump_file)
{
fprintf (dump_file, "\nReloads for insn # %d\n", INSN_UID (insn));
debug_reload_to_stream (dump_file);
}
fatal_insn ("this is the insn:", insn);
}
}

static void
delete_dead_insn (rtx_insn *insn)
{
rtx_insn *prev = prev_active_insn (insn);
rtx prev_dest;
if (prev && BLOCK_FOR_INSN (prev) == BLOCK_FOR_INSN (insn)
&& GET_CODE (PATTERN (prev)) == SET
&& (prev_dest = SET_DEST (PATTERN (prev)), REG_P (prev_dest))
&& reg_mentioned_p (prev_dest, PATTERN (insn))
&& find_regno_note (insn, REG_DEAD, REGNO (prev_dest))
&& ! side_effects_p (SET_SRC (PATTERN (prev))))
need_dce = 1;
SET_INSN_DELETED (insn);
}
static void
alter_reg (int i, int from_reg, bool dont_share_p)
{
if (regno_reg_rtx[i] == 0)
return;
if (!REG_P (regno_reg_rtx[i]))
return;
SET_REGNO (regno_reg_rtx[i],
reg_renumber[i] >= 0 ? reg_renumber[i] : i);
if (reg_renumber[i] < 0
&& REG_N_REFS (i) > 0
&& reg_equiv_constant (i) == 0
&& (reg_equiv_invariant (i) == 0
|| reg_equiv_init (i) == 0)
&& reg_equiv_memory_loc (i) == 0)
{
rtx x = NULL_RTX;
machine_mode mode = GET_MODE (regno_reg_rtx[i]);
poly_uint64 inherent_size = GET_MODE_SIZE (mode);
unsigned int inherent_align = GET_MODE_ALIGNMENT (mode);
machine_mode wider_mode = wider_subreg_mode (mode, reg_max_ref_mode[i]);
poly_uint64 total_size = GET_MODE_SIZE (wider_mode);
unsigned int min_align
= constant_lower_bound (GET_MODE_BITSIZE (reg_max_ref_mode[i]));
poly_int64 adjust = 0;
something_was_spilled = true;
if (ira_conflicts_p)
{
SET_REGNO_REG_SET (&spilled_pseudos, i);
if (!dont_share_p)
x = ira_reuse_stack_slot (i, inherent_size, total_size);
}
if (x)
;
else if (from_reg == -1 || (!dont_share_p && ira_conflicts_p))
{
rtx stack_slot;
gcc_checking_assert (ordered_p (total_size, inherent_size));
x = assign_stack_local (mode, total_size,
min_align > inherent_align
|| maybe_gt (total_size, inherent_size)
? -1 : 0);
stack_slot = x;
if (BYTES_BIG_ENDIAN)
{
adjust = inherent_size - total_size;
if (maybe_ne (adjust, 0))
{
poly_uint64 total_bits = total_size * BITS_PER_UNIT;
machine_mode mem_mode
= int_mode_for_size (total_bits, 1).else_blk ();
stack_slot = adjust_address_nv (x, mem_mode, adjust);
}
}
if (! dont_share_p && ira_conflicts_p)
ira_mark_new_stack_slot (stack_slot, i, total_size);
}
else if (spill_stack_slot[from_reg] != 0
&& known_ge (spill_stack_slot_width[from_reg], total_size)
&& known_ge (GET_MODE_SIZE
(GET_MODE (spill_stack_slot[from_reg])),
inherent_size)
&& MEM_ALIGN (spill_stack_slot[from_reg]) >= min_align)
x = spill_stack_slot[from_reg];
else
{
rtx stack_slot;
if (spill_stack_slot[from_reg])
{
if (partial_subreg_p (mode,
GET_MODE (spill_stack_slot[from_reg])))
mode = GET_MODE (spill_stack_slot[from_reg]);
total_size = ordered_max (total_size,
spill_stack_slot_width[from_reg]);
if (MEM_ALIGN (spill_stack_slot[from_reg]) > min_align)
min_align = MEM_ALIGN (spill_stack_slot[from_reg]);
}
gcc_checking_assert (ordered_p (total_size, inherent_size));
x = assign_stack_local (mode, total_size,
min_align > inherent_align
|| maybe_gt (total_size, inherent_size)
? -1 : 0);
stack_slot = x;
if (BYTES_BIG_ENDIAN)
{
adjust = GET_MODE_SIZE (mode) - total_size;
if (maybe_ne (adjust, 0))
{
poly_uint64 total_bits = total_size * BITS_PER_UNIT;
machine_mode mem_mode
= int_mode_for_size (total_bits, 1).else_blk ();
stack_slot = adjust_address_nv (x, mem_mode, adjust);
}
}
spill_stack_slot[from_reg] = stack_slot;
spill_stack_slot_width[from_reg] = total_size;
}
adjust += subreg_size_lowpart_offset (inherent_size, total_size);
x = adjust_address_nv (x, GET_MODE (regno_reg_rtx[i]), adjust);
set_mem_attrs_for_spill (x);
reg_equiv_memory_loc (i) = x;
}
}
static void
mark_home_live_1 (int regno, machine_mode mode)
{
int i, lim;
i = reg_renumber[regno];
if (i < 0)
return;
lim = end_hard_regno (mode, i);
while (i < lim)
df_set_regs_ever_live (i++, true);
}
void
mark_home_live (int regno)
{
if (reg_renumber[regno] >= 0)
mark_home_live_1 (regno, PSEUDO_REGNO_MODE (regno));
}

static void
set_label_offsets (rtx x, rtx_insn *insn, int initial_p)
{
enum rtx_code code = GET_CODE (x);
rtx tem;
unsigned int i;
struct elim_table *p;
switch (code)
{
case LABEL_REF:
if (LABEL_REF_NONLOCAL_P (x))
return;
x = label_ref_label (x);
case CODE_LABEL:
if (! offsets_known_at[CODE_LABEL_NUMBER (x) - first_label_num])
{
for (i = 0; i < NUM_ELIMINABLE_REGS; i++)
offsets_at[CODE_LABEL_NUMBER (x) - first_label_num][i]
= (initial_p ? reg_eliminate[i].initial_offset
: reg_eliminate[i].offset);
offsets_known_at[CODE_LABEL_NUMBER (x) - first_label_num] = 1;
}
else if (x == insn
&& (tem = prev_nonnote_insn (insn)) != 0
&& BARRIER_P (tem))
set_offsets_for_label (insn);
else
for (i = 0; i < NUM_ELIMINABLE_REGS; i++)
if (maybe_ne (offsets_at[CODE_LABEL_NUMBER (x) - first_label_num][i],
(initial_p ? reg_eliminate[i].initial_offset
: reg_eliminate[i].offset)))
reg_eliminate[i].can_eliminate = 0;
return;
case JUMP_TABLE_DATA:
set_label_offsets (PATTERN (insn), insn, initial_p);
return;
case JUMP_INSN:
set_label_offsets (PATTERN (insn), insn, initial_p);
case INSN:
case CALL_INSN:
for (tem = REG_NOTES (x); tem; tem = XEXP (tem, 1))
if (REG_NOTE_KIND (tem) == REG_LABEL_OPERAND)
set_label_offsets (XEXP (tem, 0), insn, 1);
return;
case PARALLEL:
case ADDR_VEC:
case ADDR_DIFF_VEC:
for (i = 0; i < (unsigned) XVECLEN (x, code == ADDR_DIFF_VEC); i++)
set_label_offsets (XVECEXP (x, code == ADDR_DIFF_VEC, i),
insn, initial_p);
return;
case SET:
if (SET_DEST (x) != pc_rtx)
return;
switch (GET_CODE (SET_SRC (x)))
{
case PC:
case RETURN:
return;
case LABEL_REF:
set_label_offsets (SET_SRC (x), insn, initial_p);
return;
case IF_THEN_ELSE:
tem = XEXP (SET_SRC (x), 1);
if (GET_CODE (tem) == LABEL_REF)
set_label_offsets (label_ref_label (tem), insn, initial_p);
else if (GET_CODE (tem) != PC && GET_CODE (tem) != RETURN)
break;
tem = XEXP (SET_SRC (x), 2);
if (GET_CODE (tem) == LABEL_REF)
set_label_offsets (label_ref_label (tem), insn, initial_p);
else if (GET_CODE (tem) != PC && GET_CODE (tem) != RETURN)
break;
return;
default:
break;
}
for (p = reg_eliminate; p < &reg_eliminate[NUM_ELIMINABLE_REGS]; p++)
if (maybe_ne (p->offset, p->initial_offset))
p->can_eliminate = 0;
break;
default:
break;
}
}

static void
note_reg_elim_costly (const_rtx x, rtx insn)
{
subrtx_iterator::array_type array;
FOR_EACH_SUBRTX (iter, array, x, NONCONST)
{
const_rtx x = *iter;
if (MEM_P (x))
iter.skip_subrtxes ();
else if (REG_P (x)
&& REGNO (x) >= FIRST_PSEUDO_REGISTER
&& reg_equiv_init (REGNO (x))
&& reg_equiv_invariant (REGNO (x)))
{
rtx t = reg_equiv_invariant (REGNO (x));
rtx new_rtx = eliminate_regs_1 (t, Pmode, insn, true, true);
int cost = set_src_cost (new_rtx, Pmode,
optimize_bb_for_speed_p (elim_bb));
int freq = REG_FREQ_FROM_BB (elim_bb);
if (cost != 0)
ira_adjust_equiv_reg_cost (REGNO (x), -cost * freq);
}
}
}
static rtx
eliminate_regs_1 (rtx x, machine_mode mem_mode, rtx insn,
bool may_use_invariant, bool for_costs)
{
enum rtx_code code = GET_CODE (x);
struct elim_table *ep;
int regno;
rtx new_rtx;
int i, j;
const char *fmt;
int copied = 0;
if (! current_function_decl)
return x;
switch (code)
{
CASE_CONST_ANY:
case CONST:
case SYMBOL_REF:
case CODE_LABEL:
case PC:
case CC0:
case ASM_INPUT:
case ADDR_VEC:
case ADDR_DIFF_VEC:
case RETURN:
return x;
case REG:
regno = REGNO (x);
if (regno < FIRST_PSEUDO_REGISTER)
{
for (ep = reg_eliminate; ep < &reg_eliminate[NUM_ELIMINABLE_REGS];
ep++)
if (ep->from_rtx == x && ep->can_eliminate)
return plus_constant (Pmode, ep->to_rtx, ep->previous_offset);
}
else if (reg_renumber && reg_renumber[regno] < 0
&& reg_equivs
&& reg_equiv_invariant (regno))
{
if (may_use_invariant || (insn && DEBUG_INSN_P (insn)))
return eliminate_regs_1 (copy_rtx (reg_equiv_invariant (regno)),
mem_mode, insn, true, for_costs);
reg_equiv_init (regno) = NULL;
if (!for_costs)
alter_reg (regno, -1, true);
}
return x;
case PLUS:
if (REG_P (XEXP (x, 0))
&& REGNO (XEXP (x, 0)) < FIRST_PSEUDO_REGISTER
&& CONSTANT_P (XEXP (x, 1)))
{
for (ep = reg_eliminate; ep < &reg_eliminate[NUM_ELIMINABLE_REGS];
ep++)
if (ep->from_rtx == XEXP (x, 0) && ep->can_eliminate)
{
if (mem_mode != 0
&& CONST_INT_P (XEXP (x, 1))
&& known_eq (INTVAL (XEXP (x, 1)), -ep->previous_offset))
return ep->to_rtx;
else
return gen_rtx_PLUS (Pmode, ep->to_rtx,
plus_constant (Pmode, XEXP (x, 1),
ep->previous_offset));
}
return x;
}
{
rtx new0 = eliminate_regs_1 (XEXP (x, 0), mem_mode, insn, true,
for_costs);
rtx new1 = eliminate_regs_1 (XEXP (x, 1), mem_mode, insn, true,
for_costs);
if (reg_renumber && (new0 != XEXP (x, 0) || new1 != XEXP (x, 1)))
{
if (GET_CODE (new0) == PLUS && REG_P (new1)
&& REGNO (new1) >= FIRST_PSEUDO_REGISTER
&& reg_renumber[REGNO (new1)] < 0
&& reg_equivs
&& reg_equiv_constant (REGNO (new1)) != 0)
new1 = reg_equiv_constant (REGNO (new1));
else if (GET_CODE (new1) == PLUS && REG_P (new0)
&& REGNO (new0) >= FIRST_PSEUDO_REGISTER
&& reg_renumber[REGNO (new0)] < 0
&& reg_equiv_constant (REGNO (new0)) != 0)
new0 = reg_equiv_constant (REGNO (new0));
new_rtx = form_sum (GET_MODE (x), new0, new1);
if (! mem_mode && GET_CODE (new_rtx) != PLUS)
return gen_rtx_PLUS (GET_MODE (x), new_rtx, const0_rtx);
else
return new_rtx;
}
}
return x;
case MULT:
if (REG_P (XEXP (x, 0))
&& REGNO (XEXP (x, 0)) < FIRST_PSEUDO_REGISTER
&& CONST_INT_P (XEXP (x, 1)))
for (ep = reg_eliminate; ep < &reg_eliminate[NUM_ELIMINABLE_REGS];
ep++)
if (ep->from_rtx == XEXP (x, 0) && ep->can_eliminate)
{
if (! mem_mode
&& ! (insn != 0 && (GET_CODE (insn) == EXPR_LIST
|| GET_CODE (insn) == INSN_LIST
|| DEBUG_INSN_P (insn))))
ep->ref_outside_mem = 1;
return
plus_constant (Pmode,
gen_rtx_MULT (Pmode, ep->to_rtx, XEXP (x, 1)),
ep->previous_offset * INTVAL (XEXP (x, 1)));
}
case CALL:
case COMPARE:
case MINUS:
case DIV:      case UDIV:
case MOD:      case UMOD:
case AND:      case IOR:      case XOR:
case ROTATERT: case ROTATE:
case ASHIFTRT: case LSHIFTRT: case ASHIFT:
case NE:       case EQ:
case GE:       case GT:       case GEU:    case GTU:
case LE:       case LT:       case LEU:    case LTU:
{
rtx new0 = eliminate_regs_1 (XEXP (x, 0), mem_mode, insn, false,
for_costs);
rtx new1 = XEXP (x, 1)
? eliminate_regs_1 (XEXP (x, 1), mem_mode, insn, false,
for_costs) : 0;
if (new0 != XEXP (x, 0) || new1 != XEXP (x, 1))
return gen_rtx_fmt_ee (code, GET_MODE (x), new0, new1);
}
return x;
case EXPR_LIST:
if (XEXP (x, 0))
{
new_rtx = eliminate_regs_1 (XEXP (x, 0), mem_mode, insn, true,
for_costs);
if (new_rtx != XEXP (x, 0))
{
if (REG_NOTE_KIND (x) == REG_DEAD)
return (XEXP (x, 1)
? eliminate_regs_1 (XEXP (x, 1), mem_mode, insn, true,
for_costs)
: NULL_RTX);
x = alloc_reg_note (REG_NOTE_KIND (x), new_rtx, XEXP (x, 1));
}
}
case INSN_LIST:
case INT_LIST:
if (XEXP (x, 1))
{
new_rtx = eliminate_regs_1 (XEXP (x, 1), mem_mode, insn, true,
for_costs);
if (new_rtx != XEXP (x, 1))
return
gen_rtx_fmt_ee (GET_CODE (x), GET_MODE (x), XEXP (x, 0), new_rtx);
}
return x;
case PRE_INC:
case POST_INC:
case PRE_DEC:
case POST_DEC:
return x;
case PRE_MODIFY:
case POST_MODIFY:
if (GET_CODE (XEXP (x, 1)) == PLUS
&& XEXP (XEXP (x, 1), 0) == XEXP (x, 0))
{
rtx new_rtx = eliminate_regs_1 (XEXP (XEXP (x, 1), 1), mem_mode,
insn, true, for_costs);
if (new_rtx != XEXP (XEXP (x, 1), 1))
return gen_rtx_fmt_ee (code, GET_MODE (x), XEXP (x, 0),
gen_rtx_PLUS (GET_MODE (x),
XEXP (x, 0), new_rtx));
}
return x;
case STRICT_LOW_PART:
case NEG:          case NOT:
case SIGN_EXTEND:  case ZERO_EXTEND:
case TRUNCATE:     case FLOAT_EXTEND: case FLOAT_TRUNCATE:
case FLOAT:        case FIX:
case UNSIGNED_FIX: case UNSIGNED_FLOAT:
case ABS:
case SQRT:
case FFS:
case CLZ:
case CTZ:
case POPCOUNT:
case PARITY:
case BSWAP:
new_rtx = eliminate_regs_1 (XEXP (x, 0), mem_mode, insn, false,
for_costs);
if (new_rtx != XEXP (x, 0))
return gen_rtx_fmt_e (code, GET_MODE (x), new_rtx);
return x;
case SUBREG:
if (REG_P (SUBREG_REG (x))
&& !paradoxical_subreg_p (x)
&& reg_equivs
&& reg_equiv_memory_loc (REGNO (SUBREG_REG (x))) != 0)
{
new_rtx = SUBREG_REG (x);
}
else
new_rtx = eliminate_regs_1 (SUBREG_REG (x), mem_mode, insn, false, for_costs);
if (new_rtx != SUBREG_REG (x))
{
poly_int64 x_size = GET_MODE_SIZE (GET_MODE (x));
poly_int64 new_size = GET_MODE_SIZE (GET_MODE (new_rtx));
if (MEM_P (new_rtx)
&& ((partial_subreg_p (GET_MODE (x), GET_MODE (new_rtx))
&& !(WORD_REGISTER_OPERATIONS
&& known_equal_after_align_down (x_size - 1,
new_size - 1,
UNITS_PER_WORD)))
|| known_eq (x_size, new_size))
)
return adjust_address_nv (new_rtx, GET_MODE (x), SUBREG_BYTE (x));
else if (insn && GET_CODE (insn) == DEBUG_INSN)
return gen_rtx_raw_SUBREG (GET_MODE (x), new_rtx, SUBREG_BYTE (x));
else
return gen_rtx_SUBREG (GET_MODE (x), new_rtx, SUBREG_BYTE (x));
}
return x;
case MEM:
new_rtx = eliminate_regs_1 (XEXP (x, 0), GET_MODE (x), insn, true,
for_costs);
if (for_costs
&& memory_address_p (GET_MODE (x), XEXP (x, 0))
&& !memory_address_p (GET_MODE (x), new_rtx))
note_reg_elim_costly (XEXP (x, 0), insn);
return replace_equiv_address_nv (x, new_rtx);
case USE:
new_rtx = eliminate_regs_1 (XEXP (x, 0), VOIDmode, insn, false,
for_costs);
if (new_rtx != XEXP (x, 0))
return gen_rtx_USE (GET_MODE (x), new_rtx);
return x;
case CLOBBER:
case ASM_OPERANDS:
gcc_assert (insn && DEBUG_INSN_P (insn));
break;
case SET:
gcc_unreachable ();
default:
break;
}
fmt = GET_RTX_FORMAT (code);
for (i = 0; i < GET_RTX_LENGTH (code); i++, fmt++)
{
if (*fmt == 'e')
{
new_rtx = eliminate_regs_1 (XEXP (x, i), mem_mode, insn, false,
for_costs);
if (new_rtx != XEXP (x, i) && ! copied)
{
x = shallow_copy_rtx (x);
copied = 1;
}
XEXP (x, i) = new_rtx;
}
else if (*fmt == 'E')
{
int copied_vec = 0;
for (j = 0; j < XVECLEN (x, i); j++)
{
new_rtx = eliminate_regs_1 (XVECEXP (x, i, j), mem_mode, insn, false,
for_costs);
if (new_rtx != XVECEXP (x, i, j) && ! copied_vec)
{
rtvec new_v = gen_rtvec_v (XVECLEN (x, i),
XVEC (x, i)->elem);
if (! copied)
{
x = shallow_copy_rtx (x);
copied = 1;
}
XVEC (x, i) = new_v;
copied_vec = 1;
}
XVECEXP (x, i, j) = new_rtx;
}
}
}
return x;
}
rtx
eliminate_regs (rtx x, machine_mode mem_mode, rtx insn)
{
if (reg_eliminate == NULL)
{
gcc_assert (targetm.no_register_allocation);
return x;
}
return eliminate_regs_1 (x, mem_mode, insn, false, false);
}
static void
elimination_effects (rtx x, machine_mode mem_mode)
{
enum rtx_code code = GET_CODE (x);
struct elim_table *ep;
int regno;
int i, j;
const char *fmt;
switch (code)
{
CASE_CONST_ANY:
case CONST:
case SYMBOL_REF:
case CODE_LABEL:
case PC:
case CC0:
case ASM_INPUT:
case ADDR_VEC:
case ADDR_DIFF_VEC:
case RETURN:
return;
case REG:
regno = REGNO (x);
if (regno < FIRST_PSEUDO_REGISTER)
{
for (ep = reg_eliminate; ep < &reg_eliminate[NUM_ELIMINABLE_REGS];
ep++)
if (ep->from_rtx == x && ep->can_eliminate)
{
if (! mem_mode)
ep->ref_outside_mem = 1;
return;
}
}
else if (reg_renumber[regno] < 0
&& reg_equivs
&& reg_equiv_constant (regno)
&& ! function_invariant_p (reg_equiv_constant (regno)))
elimination_effects (reg_equiv_constant (regno), mem_mode);
return;
case PRE_INC:
case POST_INC:
case PRE_DEC:
case POST_DEC:
case POST_MODIFY:
case PRE_MODIFY:
for (ep = reg_eliminate; ep < &reg_eliminate[NUM_ELIMINABLE_REGS]; ep++)
if (ep->from_rtx == XEXP (x, 0))
ep->can_eliminate = 0;
for (ep = reg_eliminate; ep < &reg_eliminate[NUM_ELIMINABLE_REGS]; ep++)
if (ep->to_rtx == XEXP (x, 0))
{
poly_int64 size = GET_MODE_SIZE (mem_mode);
#ifdef PUSH_ROUNDING
if (ep->to_rtx == stack_pointer_rtx)
size = PUSH_ROUNDING (size);
#endif
if (code == PRE_DEC || code == POST_DEC)
ep->offset += size;
else if (code == PRE_INC || code == POST_INC)
ep->offset -= size;
else if (code == PRE_MODIFY || code == POST_MODIFY)
{
if (GET_CODE (XEXP (x, 1)) == PLUS
&& XEXP (x, 0) == XEXP (XEXP (x, 1), 0)
&& CONST_INT_P (XEXP (XEXP (x, 1), 1)))
ep->offset -= INTVAL (XEXP (XEXP (x, 1), 1));
else
ep->can_eliminate = 0;
}
}
if (code == POST_MODIFY || code == PRE_MODIFY)
break;
gcc_fallthrough ();
case STRICT_LOW_PART:
case NEG:          case NOT:
case SIGN_EXTEND:  case ZERO_EXTEND:
case TRUNCATE:     case FLOAT_EXTEND: case FLOAT_TRUNCATE:
case FLOAT:        case FIX:
case UNSIGNED_FIX: case UNSIGNED_FLOAT:
case ABS:
case SQRT:
case FFS:
case CLZ:
case CTZ:
case POPCOUNT:
case PARITY:
case BSWAP:
elimination_effects (XEXP (x, 0), mem_mode);
return;
case SUBREG:
if (REG_P (SUBREG_REG (x))
&& !paradoxical_subreg_p (x)
&& reg_equivs
&& reg_equiv_memory_loc (REGNO (SUBREG_REG (x))) != 0)
return;
elimination_effects (SUBREG_REG (x), mem_mode);
return;
case USE:
for (ep = reg_eliminate; ep < &reg_eliminate[NUM_ELIMINABLE_REGS]; ep++)
if (ep->from_rtx == XEXP (x, 0))
ep->can_eliminate = 0;
elimination_effects (XEXP (x, 0), mem_mode);
return;
case CLOBBER:
for (ep = reg_eliminate; ep < &reg_eliminate[NUM_ELIMINABLE_REGS]; ep++)
if (ep->to_rtx == XEXP (x, 0))
ep->can_eliminate = 0;
elimination_effects (XEXP (x, 0), mem_mode);
return;
case SET:
if (REG_P (SET_DEST (x)))
{
for (ep = reg_eliminate; ep < &reg_eliminate[NUM_ELIMINABLE_REGS];
ep++)
if (ep->to_rtx == SET_DEST (x)
&& SET_DEST (x) != hard_frame_pointer_rtx)
{
rtx src = SET_SRC (x);
if (GET_CODE (src) == PLUS
&& XEXP (src, 0) == SET_DEST (x)
&& CONST_INT_P (XEXP (src, 1)))
ep->offset -= INTVAL (XEXP (src, 1));
else
ep->can_eliminate = 0;
}
}
elimination_effects (SET_DEST (x), VOIDmode);
elimination_effects (SET_SRC (x), VOIDmode);
return;
case MEM:
elimination_effects (XEXP (x, 0), GET_MODE (x));
return;
default:
break;
}
fmt = GET_RTX_FORMAT (code);
for (i = 0; i < GET_RTX_LENGTH (code); i++, fmt++)
{
if (*fmt == 'e')
elimination_effects (XEXP (x, i), mem_mode);
else if (*fmt == 'E')
for (j = 0; j < XVECLEN (x, i); j++)
elimination_effects (XVECEXP (x, i, j), mem_mode);
}
}
static void
check_eliminable_occurrences (rtx x)
{
const char *fmt;
int i;
enum rtx_code code;
if (x == 0)
return;
code = GET_CODE (x);
if (code == REG && REGNO (x) < FIRST_PSEUDO_REGISTER)
{
struct elim_table *ep;
for (ep = reg_eliminate; ep < &reg_eliminate[NUM_ELIMINABLE_REGS]; ep++)
if (ep->from_rtx == x)
ep->can_eliminate = 0;
return;
}
fmt = GET_RTX_FORMAT (code);
for (i = 0; i < GET_RTX_LENGTH (code); i++, fmt++)
{
if (*fmt == 'e')
check_eliminable_occurrences (XEXP (x, i));
else if (*fmt == 'E')
{
int j;
for (j = 0; j < XVECLEN (x, i); j++)
check_eliminable_occurrences (XVECEXP (x, i, j));
}
}
}

static int
eliminate_regs_in_insn (rtx_insn *insn, int replace)
{
int icode = recog_memoized (insn);
rtx old_body = PATTERN (insn);
int insn_is_asm = asm_noperands (old_body) >= 0;
rtx old_set = single_set (insn);
rtx new_body;
int val = 0;
int i;
rtx substed_operand[MAX_RECOG_OPERANDS];
rtx orig_operand[MAX_RECOG_OPERANDS];
struct elim_table *ep;
rtx plus_src, plus_cst_src;
if (! insn_is_asm && icode < 0)
{
gcc_assert (DEBUG_INSN_P (insn)
|| GET_CODE (PATTERN (insn)) == USE
|| GET_CODE (PATTERN (insn)) == CLOBBER
|| GET_CODE (PATTERN (insn)) == ASM_INPUT);
if (DEBUG_BIND_INSN_P (insn))
INSN_VAR_LOCATION_LOC (insn)
= eliminate_regs (INSN_VAR_LOCATION_LOC (insn), VOIDmode, insn);
return 0;
}
if (old_set != 0 && REG_P (SET_DEST (old_set))
&& REGNO (SET_DEST (old_set)) < FIRST_PSEUDO_REGISTER)
{
for (ep = reg_eliminate; ep < &reg_eliminate[NUM_ELIMINABLE_REGS]; ep++)
if (ep->from_rtx == SET_DEST (old_set) && ep->can_eliminate)
{
if (!HARD_FRAME_POINTER_IS_FRAME_POINTER
&& ep->from == FRAME_POINTER_REGNUM
&& ep->to == HARD_FRAME_POINTER_REGNUM)
{
rtx base = SET_SRC (old_set);
rtx_insn *base_insn = insn;
HOST_WIDE_INT offset = 0;
while (base != ep->to_rtx)
{
rtx_insn *prev_insn;
rtx prev_set;
if (GET_CODE (base) == PLUS
&& CONST_INT_P (XEXP (base, 1)))
{
offset += INTVAL (XEXP (base, 1));
base = XEXP (base, 0);
}
else if ((prev_insn = prev_nonnote_insn (base_insn)) != 0
&& (prev_set = single_set (prev_insn)) != 0
&& rtx_equal_p (SET_DEST (prev_set), base))
{
base = SET_SRC (prev_set);
base_insn = prev_insn;
}
else
break;
}
if (base == ep->to_rtx)
{
rtx src = plus_constant (Pmode, ep->to_rtx,
offset - ep->offset);
new_body = old_body;
if (! replace)
{
new_body = copy_insn (old_body);
if (REG_NOTES (insn))
REG_NOTES (insn) = copy_insn_1 (REG_NOTES (insn));
}
PATTERN (insn) = new_body;
old_set = single_set (insn);
validate_change (insn, &SET_SRC (old_set), src, 1);
validate_change (insn, &SET_DEST (old_set),
ep->to_rtx, 1);
if (! apply_change_group ())
{
SET_SRC (old_set) = src;
SET_DEST (old_set) = ep->to_rtx;
}
val = 1;
goto done;
}
}
if (replace)
{
delete_dead_insn (insn);
return 1;
}
val = 1;
goto done;
}
}
plus_src = plus_cst_src = 0;
if (old_set && REG_P (SET_DEST (old_set)))
{
if (GET_CODE (SET_SRC (old_set)) == PLUS)
plus_src = SET_SRC (old_set);
if (plus_src
&& CONST_INT_P (XEXP (plus_src, 1)))
plus_cst_src = plus_src;
else if (REG_P (SET_SRC (old_set))
|| plus_src)
{
rtx links;
for (links = REG_NOTES (insn); links; links = XEXP (links, 1))
{
if ((REG_NOTE_KIND (links) == REG_EQUAL
|| REG_NOTE_KIND (links) == REG_EQUIV)
&& GET_CODE (XEXP (links, 0)) == PLUS
&& CONST_INT_P (XEXP (XEXP (links, 0), 1)))
{
plus_cst_src = XEXP (links, 0);
break;
}
}
}
if (plus_cst_src)
{
rtx reg = XEXP (plus_cst_src, 0);
if (GET_CODE (reg) == SUBREG && subreg_lowpart_p (reg))
reg = SUBREG_REG (reg);
if (!REG_P (reg) || REGNO (reg) >= FIRST_PSEUDO_REGISTER)
plus_cst_src = 0;
}
}
if (plus_cst_src)
{
rtx reg = XEXP (plus_cst_src, 0);
poly_int64 offset = INTVAL (XEXP (plus_cst_src, 1));
if (GET_CODE (reg) == SUBREG)
reg = SUBREG_REG (reg);
for (ep = reg_eliminate; ep < &reg_eliminate[NUM_ELIMINABLE_REGS]; ep++)
if (ep->from_rtx == reg && ep->can_eliminate)
{
rtx to_rtx = ep->to_rtx;
offset += ep->offset;
offset = trunc_int_for_mode (offset, GET_MODE (plus_cst_src));
if (GET_CODE (XEXP (plus_cst_src, 0)) == SUBREG)
to_rtx = gen_lowpart (GET_MODE (XEXP (plus_cst_src, 0)),
to_rtx);
if (known_eq (offset, 0) || plus_src)
{
rtx new_src = plus_constant (GET_MODE (to_rtx),
to_rtx, offset);
new_body = old_body;
if (! replace)
{
new_body = copy_insn (old_body);
if (REG_NOTES (insn))
REG_NOTES (insn) = copy_insn_1 (REG_NOTES (insn));
}
PATTERN (insn) = new_body;
old_set = single_set (insn);
if (!validate_change (insn, &SET_SRC (old_set), new_src, 0))
{
rtx new_pat = gen_rtx_SET (SET_DEST (old_set), new_src);
if (!validate_change (insn, &PATTERN (insn), new_pat, 0))
SET_SRC (old_set) = new_src;
}
}
else
break;
val = 1;
goto done;
}
}
elimination_effects (old_body, VOIDmode);
extract_insn (insn);
for (i = 0; i < recog_data.n_operands; i++)
{
orig_operand[i] = recog_data.operand[i];
substed_operand[i] = recog_data.operand[i];
if (insn_is_asm || insn_data[icode].operand[i].eliminable)
{
bool is_set_src, in_plus;
if (recog_data.operand_type[i] != OP_IN
&& REG_P (orig_operand[i]))
{
for (ep = reg_eliminate; ep < &reg_eliminate[NUM_ELIMINABLE_REGS];
ep++)
if (ep->from_rtx == orig_operand[i])
ep->can_eliminate = 0;
}
is_set_src = false;
if (old_set
&& recog_data.operand_loc[i] == &SET_SRC (old_set))
is_set_src = true;
in_plus = false;
if (plus_src
&& (recog_data.operand_loc[i] == &XEXP (plus_src, 0)
|| recog_data.operand_loc[i] == &XEXP (plus_src, 1)))
in_plus = true;
substed_operand[i]
= eliminate_regs_1 (recog_data.operand[i], VOIDmode,
replace ? insn : NULL_RTX,
is_set_src || in_plus, false);
if (substed_operand[i] != orig_operand[i])
val = 1;
*recog_data.operand_loc[i] = 0;
if (recog_data.operand_type[i] != OP_IN
&& REG_P (orig_operand[i])
&& MEM_P (substed_operand[i])
&& replace)
emit_insn_after (gen_clobber (orig_operand[i]), insn);
}
}
for (i = 0; i < recog_data.n_dups; i++)
*recog_data.dup_loc[i]
= *recog_data.operand_loc[(int) recog_data.dup_num[i]];
check_eliminable_occurrences (old_body);
for (i = 0; i < recog_data.n_operands; i++)
*recog_data.operand_loc[i] = substed_operand[i];
for (i = 0; i < recog_data.n_dups; i++)
*recog_data.dup_loc[i] = substed_operand[(int) recog_data.dup_num[i]];
if (val)
{
new_body = old_body;
if (! replace)
{
new_body = copy_insn (old_body);
if (REG_NOTES (insn))
REG_NOTES (insn) = copy_insn_1 (REG_NOTES (insn));
}
PATTERN (insn) = new_body;
if (! insn_is_asm
&& old_set != 0
&& ((REG_P (SET_SRC (old_set))
&& (GET_CODE (new_body) != SET
|| !REG_P (SET_SRC (new_body))))
|| (old_set != 0
&& ((MEM_P (SET_SRC (old_set))
&& SET_SRC (old_set) != recog_data.operand[1])
|| (MEM_P (SET_DEST (old_set))
&& SET_DEST (old_set) != recog_data.operand[0])))
|| GET_CODE (SET_SRC (old_set)) == PLUS))
{
int new_icode = recog (PATTERN (insn), insn, 0);
if (new_icode >= 0)
INSN_CODE (insn) = new_icode;
}
}
if (! replace)
{
for (i = 0; i < recog_data.n_operands; i++)
if (recog_data.operand_loc[i] != &PATTERN (insn))
*recog_data.operand_loc[i] = orig_operand[i];
for (i = 0; i < recog_data.n_dups; i++)
*recog_data.dup_loc[i] = orig_operand[(int) recog_data.dup_num[i]];
}
for (ep = reg_eliminate; ep < &reg_eliminate[NUM_ELIMINABLE_REGS]; ep++)
{
if (maybe_ne (ep->previous_offset, ep->offset) && ep->ref_outside_mem)
ep->can_eliminate = 0;
ep->ref_outside_mem = 0;
if (maybe_ne (ep->previous_offset, ep->offset))
val = 1;
}
done:
if (val && REG_NOTES (insn) != 0)
REG_NOTES (insn)
= eliminate_regs_1 (REG_NOTES (insn), VOIDmode, REG_NOTES (insn), true,
false);
return val;
}
#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wmaybe-uninitialized"
static void
elimination_costs_in_insn (rtx_insn *insn)
{
int icode = recog_memoized (insn);
rtx old_body = PATTERN (insn);
int insn_is_asm = asm_noperands (old_body) >= 0;
rtx old_set = single_set (insn);
int i;
rtx orig_operand[MAX_RECOG_OPERANDS];
rtx orig_dup[MAX_RECOG_OPERANDS];
struct elim_table *ep;
rtx plus_src, plus_cst_src;
bool sets_reg_p;
if (! insn_is_asm && icode < 0)
{
gcc_assert (DEBUG_INSN_P (insn)
|| GET_CODE (PATTERN (insn)) == USE
|| GET_CODE (PATTERN (insn)) == CLOBBER
|| GET_CODE (PATTERN (insn)) == ASM_INPUT);
return;
}
if (old_set != 0 && REG_P (SET_DEST (old_set))
&& REGNO (SET_DEST (old_set)) < FIRST_PSEUDO_REGISTER)
{
for (ep = reg_eliminate; ep < &reg_eliminate[NUM_ELIMINABLE_REGS]; ep++)
if (ep->from_rtx == SET_DEST (old_set) && ep->can_eliminate)
return;
}
plus_src = plus_cst_src = 0;
sets_reg_p = false;
if (old_set && REG_P (SET_DEST (old_set)))
{
sets_reg_p = true;
if (GET_CODE (SET_SRC (old_set)) == PLUS)
plus_src = SET_SRC (old_set);
if (plus_src
&& CONST_INT_P (XEXP (plus_src, 1)))
plus_cst_src = plus_src;
else if (REG_P (SET_SRC (old_set))
|| plus_src)
{
rtx links;
for (links = REG_NOTES (insn); links; links = XEXP (links, 1))
{
if ((REG_NOTE_KIND (links) == REG_EQUAL
|| REG_NOTE_KIND (links) == REG_EQUIV)
&& GET_CODE (XEXP (links, 0)) == PLUS
&& CONST_INT_P (XEXP (XEXP (links, 0), 1)))
{
plus_cst_src = XEXP (links, 0);
break;
}
}
}
}
elimination_effects (old_body, VOIDmode);
extract_insn (insn);
int n_dups = recog_data.n_dups;
for (i = 0; i < n_dups; i++)
orig_dup[i] = *recog_data.dup_loc[i];
int n_operands = recog_data.n_operands;
for (i = 0; i < n_operands; i++)
{
orig_operand[i] = recog_data.operand[i];
if (insn_is_asm || insn_data[icode].operand[i].eliminable)
{
bool is_set_src, in_plus;
if (recog_data.operand_type[i] != OP_IN
&& REG_P (orig_operand[i]))
{
for (ep = reg_eliminate; ep < &reg_eliminate[NUM_ELIMINABLE_REGS];
ep++)
if (ep->from_rtx == orig_operand[i])
ep->can_eliminate = 0;
}
is_set_src = false;
if (old_set && recog_data.operand_loc[i] == &SET_SRC (old_set))
is_set_src = true;
if (is_set_src && !sets_reg_p)
note_reg_elim_costly (SET_SRC (old_set), insn);
in_plus = false;
if (plus_src && sets_reg_p
&& (recog_data.operand_loc[i] == &XEXP (plus_src, 0)
|| recog_data.operand_loc[i] == &XEXP (plus_src, 1)))
in_plus = true;
eliminate_regs_1 (recog_data.operand[i], VOIDmode,
NULL_RTX,
is_set_src || in_plus, true);
*recog_data.operand_loc[i] = 0;
}
}
for (i = 0; i < n_dups; i++)
*recog_data.dup_loc[i]
= *recog_data.operand_loc[(int) recog_data.dup_num[i]];
check_eliminable_occurrences (old_body);
for (i = 0; i < n_operands; i++)
*recog_data.operand_loc[i] = orig_operand[i];
for (i = 0; i < n_dups; i++)
*recog_data.dup_loc[i] = orig_dup[i];
for (ep = reg_eliminate; ep < &reg_eliminate[NUM_ELIMINABLE_REGS]; ep++)
{
if (maybe_ne (ep->previous_offset, ep->offset) && ep->ref_outside_mem)
ep->can_eliminate = 0;
ep->ref_outside_mem = 0;
}
return;
}
#pragma GCC diagnostic pop
static void
update_eliminable_offsets (void)
{
struct elim_table *ep;
num_not_at_initial_offset = 0;
for (ep = reg_eliminate; ep < &reg_eliminate[NUM_ELIMINABLE_REGS]; ep++)
{
ep->previous_offset = ep->offset;
if (ep->can_eliminate && maybe_ne (ep->offset, ep->initial_offset))
num_not_at_initial_offset++;
}
}
static void
mark_not_eliminable (rtx dest, const_rtx x, void *data ATTRIBUTE_UNUSED)
{
unsigned int i;
if (GET_CODE (dest) == SUBREG)
dest = SUBREG_REG (dest);
if (dest == hard_frame_pointer_rtx)
return;
for (i = 0; i < NUM_ELIMINABLE_REGS; i++)
if (reg_eliminate[i].can_eliminate && dest == reg_eliminate[i].to_rtx
&& (GET_CODE (x) != SET
|| GET_CODE (SET_SRC (x)) != PLUS
|| XEXP (SET_SRC (x), 0) != dest
|| !CONST_INT_P (XEXP (SET_SRC (x), 1))))
{
reg_eliminate[i].can_eliminate_previous
= reg_eliminate[i].can_eliminate = 0;
num_eliminable--;
}
}
static bool
verify_initial_elim_offsets (void)
{
poly_int64 t;
struct elim_table *ep;
if (!num_eliminable)
return true;
targetm.compute_frame_layout ();
for (ep = reg_eliminate; ep < &reg_eliminate[NUM_ELIMINABLE_REGS]; ep++)
{
INITIAL_ELIMINATION_OFFSET (ep->from, ep->to, t);
if (maybe_ne (t, ep->initial_offset))
return false;
}
return true;
}
static void
set_initial_elim_offsets (void)
{
struct elim_table *ep = reg_eliminate;
targetm.compute_frame_layout ();
for (; ep < &reg_eliminate[NUM_ELIMINABLE_REGS]; ep++)
{
INITIAL_ELIMINATION_OFFSET (ep->from, ep->to, ep->initial_offset);
ep->previous_offset = ep->offset = ep->initial_offset;
}
num_not_at_initial_offset = 0;
}
static void
set_initial_eh_label_offset (rtx label)
{
set_label_offsets (label, NULL, 1);
}
static void
set_initial_label_offsets (void)
{
memset (offsets_known_at, 0, num_labels);
unsigned int i;
rtx_insn *insn;
FOR_EACH_VEC_SAFE_ELT (forced_labels, i, insn)
set_label_offsets (insn, NULL, 1);
for (rtx_insn_list *x = nonlocal_goto_handler_labels; x; x = x->next ())
if (x->insn ())
set_label_offsets (x->insn (), NULL, 1);
for_each_eh_label (set_initial_eh_label_offset);
}
static void
set_offsets_for_label (rtx_insn *insn)
{
unsigned int i;
int label_nr = CODE_LABEL_NUMBER (insn);
struct elim_table *ep;
num_not_at_initial_offset = 0;
for (i = 0, ep = reg_eliminate; i < NUM_ELIMINABLE_REGS; ep++, i++)
{
ep->offset = ep->previous_offset
= offsets_at[label_nr - first_label_num][i];
if (ep->can_eliminate && maybe_ne (ep->offset, ep->initial_offset))
num_not_at_initial_offset++;
}
}
static void
update_eliminables (HARD_REG_SET *pset)
{
int previous_frame_pointer_needed = frame_pointer_needed;
struct elim_table *ep;
for (ep = reg_eliminate; ep < &reg_eliminate[NUM_ELIMINABLE_REGS]; ep++)
if ((ep->from == HARD_FRAME_POINTER_REGNUM
&& targetm.frame_pointer_required ())
|| ! targetm.can_eliminate (ep->from, ep->to)
)
ep->can_eliminate = 0;
for (ep = reg_eliminate; ep < &reg_eliminate[NUM_ELIMINABLE_REGS]; ep++)
{
struct elim_table *op;
int new_to = -1;
if (! ep->can_eliminate && ep->can_eliminate_previous)
{
for (op = reg_eliminate;
op < &reg_eliminate[NUM_ELIMINABLE_REGS]; op++)
if (op->from == ep->from && op->can_eliminate)
{
new_to = op->to;
break;
}
for (op = reg_eliminate;
op < &reg_eliminate[NUM_ELIMINABLE_REGS]; op++)
if (op->from == new_to && op->to == ep->to)
op->can_eliminate = 0;
}
}
frame_pointer_needed = 1;
for (ep = reg_eliminate; ep < &reg_eliminate[NUM_ELIMINABLE_REGS]; ep++)
{
if (ep->can_eliminate
&& ep->from == FRAME_POINTER_REGNUM
&& ep->to != HARD_FRAME_POINTER_REGNUM
&& (! SUPPORTS_STACK_ALIGNMENT
|| ! crtl->stack_realign_needed))
frame_pointer_needed = 0;
if (! ep->can_eliminate && ep->can_eliminate_previous)
{
ep->can_eliminate_previous = 0;
SET_HARD_REG_BIT (*pset, ep->from);
num_eliminable--;
}
}
if (frame_pointer_needed && ! previous_frame_pointer_needed)
SET_HARD_REG_BIT (*pset, HARD_FRAME_POINTER_REGNUM);
}
static bool
update_eliminables_and_spill (void)
{
int i;
bool did_spill = false;
HARD_REG_SET to_spill;
CLEAR_HARD_REG_SET (to_spill);
update_eliminables (&to_spill);
AND_COMPL_HARD_REG_SET (used_spill_regs, to_spill);
for (i = 0; i < FIRST_PSEUDO_REGISTER; i++)
if (TEST_HARD_REG_BIT (to_spill, i))
{
spill_hard_reg (i, 1);
did_spill = true;
}
return did_spill;
}
bool
elimination_target_reg_p (rtx x)
{
struct elim_table *ep;
for (ep = reg_eliminate; ep < &reg_eliminate[NUM_ELIMINABLE_REGS]; ep++)
if (ep->to_rtx == x && ep->can_eliminate)
return true;
return false;
}
static void
init_elim_table (void)
{
struct elim_table *ep;
const struct elim_table_1 *ep1;
if (!reg_eliminate)
reg_eliminate = XCNEWVEC (struct elim_table, NUM_ELIMINABLE_REGS);
num_eliminable = 0;
for (ep = reg_eliminate, ep1 = reg_eliminate_1;
ep < &reg_eliminate[NUM_ELIMINABLE_REGS]; ep++, ep1++)
{
ep->from = ep1->from;
ep->to = ep1->to;
ep->can_eliminate = ep->can_eliminate_previous
= (targetm.can_eliminate (ep->from, ep->to)
&& ! (ep->to == STACK_POINTER_REGNUM
&& frame_pointer_needed
&& (! SUPPORTS_STACK_ALIGNMENT
|| ! stack_realign_fp)));
}
for (ep = reg_eliminate; ep < &reg_eliminate[NUM_ELIMINABLE_REGS]; ep++)
{
num_eliminable += ep->can_eliminate;
ep->from_rtx = gen_rtx_REG (Pmode, ep->from);
ep->to_rtx = gen_rtx_REG (Pmode, ep->to);
}
}
static void
init_eliminable_invariants (rtx_insn *first, bool do_subregs)
{
int i;
rtx_insn *insn;
grow_reg_equivs ();
if (do_subregs)
reg_max_ref_mode = XCNEWVEC (machine_mode, max_regno);
else
reg_max_ref_mode = NULL;
num_eliminable_invariants = 0;
first_label_num = get_first_label_num ();
num_labels = max_label_num () - first_label_num;
offsets_known_at = XNEWVEC (char, num_labels);
offsets_at = (poly_int64_pod (*)[NUM_ELIMINABLE_REGS])
xmalloc (num_labels * NUM_ELIMINABLE_REGS * sizeof (poly_int64));
for (insn = first; insn; insn = NEXT_INSN (insn))
{
rtx set = single_set (insn);
if (INSN_P (insn) && GET_CODE (PATTERN (insn)) == USE
&& GET_MODE (insn) != VOIDmode)
PUT_MODE (insn, VOIDmode);
if (do_subregs && NONDEBUG_INSN_P (insn))
scan_paradoxical_subregs (PATTERN (insn));
if (set != 0 && REG_P (SET_DEST (set)))
{
rtx note = find_reg_note (insn, REG_EQUIV, NULL_RTX);
rtx x;
if (! note)
continue;
i = REGNO (SET_DEST (set));
x = XEXP (note, 0);
if (i <= LAST_VIRTUAL_REGISTER)
continue;
if (!CONSTANT_P (x)
|| !flag_pic || LEGITIMATE_PIC_OPERAND_P (x))
{
if (memory_operand (x, VOIDmode))
{
reg_equiv_memory_loc (i) = copy_rtx (x);
}
else if (function_invariant_p (x))
{
machine_mode mode;
mode = GET_MODE (SET_DEST (set));
if (GET_CODE (x) == PLUS)
{
reg_equiv_invariant (i) = copy_rtx (x);
num_eliminable_invariants++;
}
else if (x == frame_pointer_rtx || x == arg_pointer_rtx)
{
reg_equiv_invariant (i) = x;
num_eliminable_invariants++;
}
else if (targetm.legitimate_constant_p (mode, x))
reg_equiv_constant (i) = x;
else
{
reg_equiv_memory_loc (i) = force_const_mem (mode, x);
if (! reg_equiv_memory_loc (i))
reg_equiv_init (i) = NULL;
}
}
else
{
reg_equiv_init (i) = NULL;
continue;
}
}
else
reg_equiv_init (i) = NULL;
}
}
if (dump_file)
for (i = FIRST_PSEUDO_REGISTER; i < max_regno; i++)
if (reg_equiv_init (i))
{
fprintf (dump_file, "init_insns for %u: ", i);
print_inline_rtx (dump_file, reg_equiv_init (i), 20);
fprintf (dump_file, "\n");
}
}
static void
free_reg_equiv (void)
{
int i;
free (offsets_known_at);
free (offsets_at);
offsets_at = 0;
offsets_known_at = 0;
for (i = 0; i < FIRST_PSEUDO_REGISTER; i++)
if (reg_equiv_alt_mem_list (i))
free_EXPR_LIST_list (&reg_equiv_alt_mem_list (i));
vec_free (reg_equivs);
}

static void
spill_hard_reg (unsigned int regno, int cant_eliminate)
{
int i;
if (cant_eliminate)
{
SET_HARD_REG_BIT (bad_spill_regs_global, regno);
df_set_regs_ever_live (regno, true);
}
for (i = FIRST_PSEUDO_REGISTER; i < max_regno; i++)
if (reg_renumber[i] >= 0
&& (unsigned int) reg_renumber[i] <= regno
&& end_hard_regno (PSEUDO_REGNO_MODE (i), reg_renumber[i]) > regno)
SET_REGNO_REG_SET (&spilled_pseudos, i);
}
static int
finish_spills (int global)
{
struct insn_chain *chain;
int something_changed = 0;
unsigned i;
reg_set_iterator rsi;
n_spills = 0;
for (i = 0; i < FIRST_PSEUDO_REGISTER; i++)
if (TEST_HARD_REG_BIT (used_spill_regs, i))
{
spill_reg_order[i] = n_spills;
spill_regs[n_spills++] = i;
if (num_eliminable && ! df_regs_ever_live_p (i))
something_changed = 1;
df_set_regs_ever_live (i, true);
}
else
spill_reg_order[i] = -1;
EXECUTE_IF_SET_IN_REG_SET (&spilled_pseudos, FIRST_PSEUDO_REGISTER, i, rsi)
if (! ira_conflicts_p || reg_renumber[i] >= 0)
{
gcc_assert (reg_renumber[i] >= 0);
SET_HARD_REG_BIT (pseudo_previous_regs[i], reg_renumber[i]);
reg_renumber[i] = -1;
if (ira_conflicts_p)
ira_mark_allocation_change (i);
something_changed = 1;
}
if (global && ira_conflicts_p)
{
unsigned int n;
memset (pseudo_forbidden_regs, 0, max_regno * sizeof (HARD_REG_SET));
for (chain = insns_need_reload; chain; chain = chain->next_need_reload)
{
EXECUTE_IF_SET_IN_REG_SET
(&chain->live_throughout, FIRST_PSEUDO_REGISTER, i, rsi)
{
IOR_HARD_REG_SET (pseudo_forbidden_regs[i],
chain->used_spill_regs);
}
EXECUTE_IF_SET_IN_REG_SET
(&chain->dead_or_set, FIRST_PSEUDO_REGISTER, i, rsi)
{
IOR_HARD_REG_SET (pseudo_forbidden_regs[i],
chain->used_spill_regs);
}
}
for (n = 0, i = FIRST_PSEUDO_REGISTER; i < (unsigned) max_regno; i++)
if (reg_old_renumber[i] != reg_renumber[i])
{
if (reg_renumber[i] < 0)
temp_pseudo_reg_arr[n++] = i;
else
CLEAR_REGNO_REG_SET (&spilled_pseudos, i);
}
if (ira_reassign_pseudos (temp_pseudo_reg_arr, n,
bad_spill_regs_global,
pseudo_forbidden_regs, pseudo_previous_regs,
&spilled_pseudos))
something_changed = 1;
}
for (chain = reload_insn_chain; chain; chain = chain->next)
{
HARD_REG_SET used_by_pseudos;
HARD_REG_SET used_by_pseudos2;
if (! ira_conflicts_p)
{
AND_COMPL_REG_SET (&chain->live_throughout, &spilled_pseudos);
AND_COMPL_REG_SET (&chain->dead_or_set, &spilled_pseudos);
}
if (chain->need_reload)
{
REG_SET_TO_HARD_REG_SET (used_by_pseudos, &chain->live_throughout);
REG_SET_TO_HARD_REG_SET (used_by_pseudos2, &chain->dead_or_set);
IOR_HARD_REG_SET (used_by_pseudos, used_by_pseudos2);
compute_use_by_pseudos (&used_by_pseudos, &chain->live_throughout);
compute_use_by_pseudos (&used_by_pseudos, &chain->dead_or_set);
COMPL_HARD_REG_SET (chain->used_spill_regs, used_by_pseudos);
AND_HARD_REG_SET (chain->used_spill_regs, used_spill_regs);
}
}
CLEAR_REG_SET (&changed_allocation_pseudos);
for (i = FIRST_PSEUDO_REGISTER; i < (unsigned)max_regno; i++)
{
int regno = reg_renumber[i];
if (reg_old_renumber[i] == regno)
continue;
SET_REGNO_REG_SET (&changed_allocation_pseudos, i);
alter_reg (i, reg_old_renumber[i], false);
reg_old_renumber[i] = regno;
if (dump_file)
{
if (regno == -1)
fprintf (dump_file, " Register %d now on stack.\n\n", i);
else
fprintf (dump_file, " Register %d now in %d.\n\n",
i, reg_renumber[i]);
}
}
return something_changed;
}

static void
scan_paradoxical_subregs (rtx x)
{
int i;
const char *fmt;
enum rtx_code code = GET_CODE (x);
switch (code)
{
case REG:
case CONST:
case SYMBOL_REF:
case LABEL_REF:
CASE_CONST_ANY:
case CC0:
case PC:
case USE:
case CLOBBER:
return;
case SUBREG:
if (REG_P (SUBREG_REG (x)))
{
unsigned int regno = REGNO (SUBREG_REG (x));
if (partial_subreg_p (reg_max_ref_mode[regno], GET_MODE (x)))
{
reg_max_ref_mode[regno] = GET_MODE (x);
mark_home_live_1 (regno, GET_MODE (x));
}
}
return;
default:
break;
}
fmt = GET_RTX_FORMAT (code);
for (i = GET_RTX_LENGTH (code) - 1; i >= 0; i--)
{
if (fmt[i] == 'e')
scan_paradoxical_subregs (XEXP (x, i));
else if (fmt[i] == 'E')
{
int j;
for (j = XVECLEN (x, i) - 1; j >= 0; j--)
scan_paradoxical_subregs (XVECEXP (x, i, j));
}
}
}
static bool
strip_paradoxical_subreg (rtx *op_ptr, rtx *other_ptr)
{
rtx op, inner, other, tem;
op = *op_ptr;
if (!paradoxical_subreg_p (op))
return false;
inner = SUBREG_REG (op);
other = *other_ptr;
tem = gen_lowpart_common (GET_MODE (inner), other);
if (!tem)
return false;
if (GET_CODE (tem) == SUBREG
&& REG_P (other)
&& HARD_REGISTER_P (other))
return false;
*op_ptr = inner;
*other_ptr = tem;
return true;
}

static void
fixup_eh_region_note (rtx_insn *insn, rtx_insn *prev, rtx_insn *next)
{
rtx note = find_reg_note (insn, REG_EH_REGION, NULL_RTX);
if (note == NULL)
return;
if (!insn_could_throw_p (insn))
remove_note (insn, note);
copy_reg_eh_region_note_forward (note, NEXT_INSN (prev), next);
}
static void
reload_as_needed (int live_known)
{
struct insn_chain *chain;
#if AUTO_INC_DEC
int i;
#endif
rtx_note *marker;
memset (spill_reg_rtx, 0, sizeof spill_reg_rtx);
memset (spill_reg_store, 0, sizeof spill_reg_store);
reg_last_reload_reg = XCNEWVEC (rtx, max_regno);
INIT_REG_SET (&reg_has_output_reload);
CLEAR_HARD_REG_SET (reg_reloaded_valid);
CLEAR_HARD_REG_SET (reg_reloaded_call_part_clobbered);
set_initial_elim_offsets ();
marker = emit_note (NOTE_INSN_DELETED);
unlink_insn_chain (marker, marker);
for (chain = reload_insn_chain; chain; chain = chain->next)
{
rtx_insn *prev = 0;
rtx_insn *insn = chain->insn;
rtx_insn *old_next = NEXT_INSN (insn);
#if AUTO_INC_DEC
rtx_insn *old_prev = PREV_INSN (insn);
#endif
if (will_delete_init_insn_p (insn))
continue;
if (LABEL_P (insn))
set_offsets_for_label (insn);
else if (INSN_P (insn))
{
regset_head regs_to_forget;
INIT_REG_SET (&regs_to_forget);
note_stores (PATTERN (insn), forget_old_reloads_1, &regs_to_forget);
if ((GET_CODE (PATTERN (insn)) == USE
|| GET_CODE (PATTERN (insn)) == CLOBBER)
&& MEM_P (XEXP (PATTERN (insn), 0)))
XEXP (XEXP (PATTERN (insn), 0), 0)
= eliminate_regs (XEXP (XEXP (PATTERN (insn), 0), 0),
GET_MODE (XEXP (PATTERN (insn), 0)),
NULL_RTX);
if ((num_eliminable || num_eliminable_invariants) && chain->need_elim)
{
eliminate_regs_in_insn (insn, 1);
if (NOTE_P (insn))
{
update_eliminable_offsets ();
CLEAR_REG_SET (&regs_to_forget);
continue;
}
}
if (! chain->need_elim && ! chain->need_reload
&& ! chain->need_operand_change)
n_reloads = 0;
else
{
CLEAR_REG_SET (&reg_has_output_reload);
CLEAR_HARD_REG_SET (reg_is_output_reload);
find_reloads (insn, 1, spill_indirect_levels, live_known,
spill_reg_order);
}
if (n_reloads > 0)
{
rtx_insn *next = NEXT_INSN (insn);
prev = PREV_INSN (insn);
reorder_insns_nobb (marker, marker, prev);
choose_reload_regs (chain);
emit_reload_insns (chain);
subst_reloads (insn);
prev = PREV_INSN (marker);
unlink_insn_chain (marker, marker);
if (cfun->can_throw_non_call_exceptions && !CALL_P (insn))
fixup_eh_region_note (insn, prev, next);
rtx p = find_reg_note (insn, REG_ARGS_SIZE, NULL_RTX);
if (p)
{
remove_note (insn, p);
fixup_args_size_notes (prev, PREV_INSN (next),
get_args_size (p));
}
if (asm_noperands (PATTERN (insn)) >= 0)
for (rtx_insn *p = NEXT_INSN (prev);
p != next;
p = NEXT_INSN (p))
if (p != insn && INSN_P (p)
&& GET_CODE (PATTERN (p)) != USE
&& (recog_memoized (p) < 0
|| (extract_insn (p),
!(constrain_operands (1,
get_enabled_alternatives (p))))))
{
error_for_asm (insn,
"%<asm%> operand requires "
"impossible reload");
delete_insn (p);
}
}
if (num_eliminable && chain->need_elim)
update_eliminable_offsets ();
forget_marked_reloads (&regs_to_forget);
CLEAR_REG_SET (&regs_to_forget);
for (rtx_insn *x = NEXT_INSN (insn); x != old_next; x = NEXT_INSN (x))
if (NONJUMP_INSN_P (x) && GET_CODE (PATTERN (x)) == CLOBBER)
note_stores (PATTERN (x), forget_old_reloads_1, NULL);
#if AUTO_INC_DEC
for (i = n_reloads - 1; i >= 0; i--)
{
rtx in_reg = rld[i].in_reg;
if (in_reg)
{
enum rtx_code code = GET_CODE (in_reg);
if ((code == POST_INC || code == POST_DEC)
&& TEST_HARD_REG_BIT (reg_reloaded_valid,
REGNO (rld[i].reg_rtx))
&& ((unsigned) reg_reloaded_contents[REGNO (rld[i].reg_rtx)]
== REGNO (XEXP (in_reg, 0))))
{
rtx reload_reg = rld[i].reg_rtx;
machine_mode mode = GET_MODE (reload_reg);
int n = 0;
rtx_insn *p;
for (p = PREV_INSN (old_next); p != prev; p = PREV_INSN (p))
{
if (reg_set_p (reload_reg, PATTERN (p)))
break;
n = count_occurrences (PATTERN (p), reload_reg, 0);
if (! n)
continue;
if (n == 1)
{
rtx replace_reg
= gen_rtx_fmt_e (code, mode, reload_reg);
validate_replace_rtx_group (reload_reg,
replace_reg, p);
n = verify_changes (0);
if (n)
{
extract_insn (p);
n = constrain_operands (1,
get_enabled_alternatives (p));
}
if (!n)
cancel_changes (0);
else
confirm_change_group ();
}
break;
}
if (n == 1)
{
add_reg_note (p, REG_INC, reload_reg);
SET_HARD_REG_BIT (reg_is_output_reload,
REGNO (reload_reg));
SET_REGNO_REG_SET (&reg_has_output_reload,
REGNO (XEXP (in_reg, 0)));
}
else
forget_old_reloads_1 (XEXP (in_reg, 0), NULL_RTX,
NULL);
}
else if ((code == PRE_INC || code == PRE_DEC)
&& TEST_HARD_REG_BIT (reg_reloaded_valid,
REGNO (rld[i].reg_rtx))
&& ((unsigned) reg_reloaded_contents[REGNO (rld[i].reg_rtx)]
== REGNO (XEXP (in_reg, 0))))
{
SET_HARD_REG_BIT (reg_is_output_reload,
REGNO (rld[i].reg_rtx));
SET_REGNO_REG_SET (&reg_has_output_reload,
REGNO (XEXP (in_reg, 0)));
}
else if (code == PRE_INC || code == PRE_DEC
|| code == POST_INC || code == POST_DEC)
{
int in_regno = REGNO (XEXP (in_reg, 0));
if (reg_last_reload_reg[in_regno] != NULL_RTX)
{
int in_hard_regno;
bool forget_p = true;
in_hard_regno = REGNO (reg_last_reload_reg[in_regno]);
if (TEST_HARD_REG_BIT (reg_reloaded_valid,
in_hard_regno))
{
for (rtx_insn *x = (old_prev ?
NEXT_INSN (old_prev) : insn);
x != old_next;
x = NEXT_INSN (x))
if (x == reg_reloaded_insn[in_hard_regno])
{
forget_p = false;
break;
}
}
if (forget_p)
forget_old_reloads_1 (XEXP (in_reg, 0),
NULL_RTX, NULL);
}
}
}
}
for (rtx x = REG_NOTES (insn); x; x = XEXP (x, 1))
if (REG_NOTE_KIND (x) == REG_INC)
{
for (i = 0; i < n_reloads; i++)
if (rld[i].out == XEXP (x, 0))
break;
if (i == n_reloads)
forget_old_reloads_1 (XEXP (x, 0), NULL_RTX, NULL);
}
#endif
}
if (LABEL_P (insn))
CLEAR_HARD_REG_SET (reg_reloaded_valid);
else if (CALL_P (insn))
{
AND_COMPL_HARD_REG_SET (reg_reloaded_valid, call_used_reg_set);
AND_COMPL_HARD_REG_SET (reg_reloaded_valid, reg_reloaded_call_part_clobbered);
if (find_reg_note (insn, REG_SETJMP, NULL_RTX))
CLEAR_HARD_REG_SET (reg_reloaded_valid);
}
}
free (reg_last_reload_reg);
CLEAR_REG_SET (&reg_has_output_reload);
}
static void
forget_old_reloads_1 (rtx x, const_rtx ignored ATTRIBUTE_UNUSED,
void *data)
{
unsigned int regno;
unsigned int nr;
regset regs = (regset) data;
while (GET_CODE (x) == SUBREG)
{
x = SUBREG_REG (x);
}
if (!REG_P (x))
return;
regno = REGNO (x);
if (regno >= FIRST_PSEUDO_REGISTER)
nr = 1;
else
{
unsigned int i;
nr = REG_NREGS (x);
if (!regs)
for (i = 0; i < nr; i++)
if (n_reloads == 0
|| ! TEST_HARD_REG_BIT (reg_is_output_reload, regno + i))
{
CLEAR_HARD_REG_BIT (reg_reloaded_valid, regno + i);
spill_reg_store[regno + i] = 0;
}
}
if (regs)
while (nr-- > 0)
SET_REGNO_REG_SET (regs, regno + nr);
else
{
while (nr-- > 0)
if (n_reloads == 0
|| !REGNO_REG_SET_P (&reg_has_output_reload, regno + nr))
reg_last_reload_reg[regno + nr] = 0;
}
}
static void
forget_marked_reloads (regset regs)
{
unsigned int reg;
reg_set_iterator rsi;
EXECUTE_IF_SET_IN_REG_SET (regs, 0, reg, rsi)
{
if (reg < FIRST_PSEUDO_REGISTER
&& (n_reloads == 0
|| ! TEST_HARD_REG_BIT (reg_is_output_reload, reg)))
{
CLEAR_HARD_REG_BIT (reg_reloaded_valid, reg);
spill_reg_store[reg] = 0;
}
if (n_reloads == 0
|| !REGNO_REG_SET_P (&reg_has_output_reload, reg))
reg_last_reload_reg[reg] = 0;
}
}

static HARD_REG_SET reload_reg_unavailable;
static HARD_REG_SET reload_reg_used;
static HARD_REG_SET reload_reg_used_in_input_addr[MAX_RECOG_OPERANDS];
static HARD_REG_SET reload_reg_used_in_inpaddr_addr[MAX_RECOG_OPERANDS];
static HARD_REG_SET reload_reg_used_in_output_addr[MAX_RECOG_OPERANDS];
static HARD_REG_SET reload_reg_used_in_outaddr_addr[MAX_RECOG_OPERANDS];
static HARD_REG_SET reload_reg_used_in_input[MAX_RECOG_OPERANDS];
static HARD_REG_SET reload_reg_used_in_output[MAX_RECOG_OPERANDS];
static HARD_REG_SET reload_reg_used_in_op_addr;
static HARD_REG_SET reload_reg_used_in_op_addr_reload;
static HARD_REG_SET reload_reg_used_in_insn;
static HARD_REG_SET reload_reg_used_in_other_addr;
static HARD_REG_SET reload_reg_used_at_all;
static HARD_REG_SET reload_reg_used_for_inherit;
static HARD_REG_SET reg_used_in_insn;
static void
mark_reload_reg_in_use (unsigned int regno, int opnum, enum reload_type type,
machine_mode mode)
{
switch (type)
{
case RELOAD_OTHER:
add_to_hard_reg_set (&reload_reg_used, mode, regno);
break;
case RELOAD_FOR_INPUT_ADDRESS:
add_to_hard_reg_set (&reload_reg_used_in_input_addr[opnum], mode, regno);
break;
case RELOAD_FOR_INPADDR_ADDRESS:
add_to_hard_reg_set (&reload_reg_used_in_inpaddr_addr[opnum], mode, regno);
break;
case RELOAD_FOR_OUTPUT_ADDRESS:
add_to_hard_reg_set (&reload_reg_used_in_output_addr[opnum], mode, regno);
break;
case RELOAD_FOR_OUTADDR_ADDRESS:
add_to_hard_reg_set (&reload_reg_used_in_outaddr_addr[opnum], mode, regno);
break;
case RELOAD_FOR_OPERAND_ADDRESS:
add_to_hard_reg_set (&reload_reg_used_in_op_addr, mode, regno);
break;
case RELOAD_FOR_OPADDR_ADDR:
add_to_hard_reg_set (&reload_reg_used_in_op_addr_reload, mode, regno);
break;
case RELOAD_FOR_OTHER_ADDRESS:
add_to_hard_reg_set (&reload_reg_used_in_other_addr, mode, regno);
break;
case RELOAD_FOR_INPUT:
add_to_hard_reg_set (&reload_reg_used_in_input[opnum], mode, regno);
break;
case RELOAD_FOR_OUTPUT:
add_to_hard_reg_set (&reload_reg_used_in_output[opnum], mode, regno);
break;
case RELOAD_FOR_INSN:
add_to_hard_reg_set (&reload_reg_used_in_insn,  mode, regno);
break;
}
add_to_hard_reg_set (&reload_reg_used_at_all, mode, regno);
}
static void
clear_reload_reg_in_use (unsigned int regno, int opnum,
enum reload_type type, machine_mode mode)
{
unsigned int nregs = hard_regno_nregs (regno, mode);
unsigned int start_regno, end_regno, r;
int i;
int check_opnum = 0;
int check_any = 0;
HARD_REG_SET *used_in_set;
switch (type)
{
case RELOAD_OTHER:
used_in_set = &reload_reg_used;
break;
case RELOAD_FOR_INPUT_ADDRESS:
used_in_set = &reload_reg_used_in_input_addr[opnum];
break;
case RELOAD_FOR_INPADDR_ADDRESS:
check_opnum = 1;
used_in_set = &reload_reg_used_in_inpaddr_addr[opnum];
break;
case RELOAD_FOR_OUTPUT_ADDRESS:
used_in_set = &reload_reg_used_in_output_addr[opnum];
break;
case RELOAD_FOR_OUTADDR_ADDRESS:
check_opnum = 1;
used_in_set = &reload_reg_used_in_outaddr_addr[opnum];
break;
case RELOAD_FOR_OPERAND_ADDRESS:
used_in_set = &reload_reg_used_in_op_addr;
break;
case RELOAD_FOR_OPADDR_ADDR:
check_any = 1;
used_in_set = &reload_reg_used_in_op_addr_reload;
break;
case RELOAD_FOR_OTHER_ADDRESS:
used_in_set = &reload_reg_used_in_other_addr;
check_any = 1;
break;
case RELOAD_FOR_INPUT:
used_in_set = &reload_reg_used_in_input[opnum];
break;
case RELOAD_FOR_OUTPUT:
used_in_set = &reload_reg_used_in_output[opnum];
break;
case RELOAD_FOR_INSN:
used_in_set = &reload_reg_used_in_insn;
break;
default:
gcc_unreachable ();
}
start_regno = regno;
end_regno = regno + nregs;
if (check_opnum || check_any)
{
for (i = n_reloads - 1; i >= 0; i--)
{
if (rld[i].when_needed == type
&& (check_any || rld[i].opnum == opnum)
&& rld[i].reg_rtx)
{
unsigned int conflict_start = true_regnum (rld[i].reg_rtx);
unsigned int conflict_end
= end_hard_regno (rld[i].mode, conflict_start);
if (conflict_start <= start_regno && conflict_end > start_regno)
start_regno = conflict_end;
if (conflict_start > start_regno && conflict_start < end_regno)
end_regno = conflict_start;
}
}
}
for (r = start_regno; r < end_regno; r++)
CLEAR_HARD_REG_BIT (*used_in_set, r);
}
static int
reload_reg_free_p (unsigned int regno, int opnum, enum reload_type type)
{
int i;
if (TEST_HARD_REG_BIT (reload_reg_used, regno)
|| TEST_HARD_REG_BIT (reload_reg_unavailable, regno))
return 0;
switch (type)
{
case RELOAD_OTHER:
if (TEST_HARD_REG_BIT (reload_reg_used_in_other_addr, regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_op_addr, regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_op_addr_reload, regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_insn, regno))
return 0;
for (i = 0; i < reload_n_operands; i++)
if (TEST_HARD_REG_BIT (reload_reg_used_in_input_addr[i], regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_inpaddr_addr[i], regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_output_addr[i], regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_outaddr_addr[i], regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_input[i], regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_output[i], regno))
return 0;
return 1;
case RELOAD_FOR_INPUT:
if (TEST_HARD_REG_BIT (reload_reg_used_in_insn, regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_op_addr, regno))
return 0;
if (TEST_HARD_REG_BIT (reload_reg_used_in_op_addr_reload, regno))
return 0;
for (i = 0; i < reload_n_operands; i++)
if (TEST_HARD_REG_BIT (reload_reg_used_in_input[i], regno))
return 0;
for (i = opnum + 1; i < reload_n_operands; i++)
if (TEST_HARD_REG_BIT (reload_reg_used_in_input_addr[i], regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_inpaddr_addr[i], regno))
return 0;
return 1;
case RELOAD_FOR_INPUT_ADDRESS:
if (TEST_HARD_REG_BIT (reload_reg_used_in_input_addr[opnum], regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_inpaddr_addr[opnum], regno))
return 0;
for (i = 0; i < opnum; i++)
if (TEST_HARD_REG_BIT (reload_reg_used_in_input[i], regno))
return 0;
return 1;
case RELOAD_FOR_INPADDR_ADDRESS:
if (TEST_HARD_REG_BIT (reload_reg_used_in_inpaddr_addr[opnum], regno))
return 0;
for (i = 0; i < opnum; i++)
if (TEST_HARD_REG_BIT (reload_reg_used_in_input[i], regno))
return 0;
return 1;
case RELOAD_FOR_OUTPUT_ADDRESS:
if (TEST_HARD_REG_BIT (reload_reg_used_in_output_addr[opnum], regno))
return 0;
for (i = 0; i <= opnum; i++)
if (TEST_HARD_REG_BIT (reload_reg_used_in_output[i], regno))
return 0;
return 1;
case RELOAD_FOR_OUTADDR_ADDRESS:
if (TEST_HARD_REG_BIT (reload_reg_used_in_outaddr_addr[opnum], regno))
return 0;
for (i = 0; i <= opnum; i++)
if (TEST_HARD_REG_BIT (reload_reg_used_in_output[i], regno))
return 0;
return 1;
case RELOAD_FOR_OPERAND_ADDRESS:
for (i = 0; i < reload_n_operands; i++)
if (TEST_HARD_REG_BIT (reload_reg_used_in_input[i], regno))
return 0;
return (! TEST_HARD_REG_BIT (reload_reg_used_in_insn, regno)
&& ! TEST_HARD_REG_BIT (reload_reg_used_in_op_addr, regno));
case RELOAD_FOR_OPADDR_ADDR:
for (i = 0; i < reload_n_operands; i++)
if (TEST_HARD_REG_BIT (reload_reg_used_in_input[i], regno))
return 0;
return (!TEST_HARD_REG_BIT (reload_reg_used_in_op_addr_reload, regno));
case RELOAD_FOR_OUTPUT:
if (TEST_HARD_REG_BIT (reload_reg_used_in_insn, regno))
return 0;
for (i = 0; i < reload_n_operands; i++)
if (TEST_HARD_REG_BIT (reload_reg_used_in_output[i], regno))
return 0;
for (i = opnum; i < reload_n_operands; i++)
if (TEST_HARD_REG_BIT (reload_reg_used_in_output_addr[i], regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_outaddr_addr[i], regno))
return 0;
return 1;
case RELOAD_FOR_INSN:
for (i = 0; i < reload_n_operands; i++)
if (TEST_HARD_REG_BIT (reload_reg_used_in_input[i], regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_output[i], regno))
return 0;
return (! TEST_HARD_REG_BIT (reload_reg_used_in_insn, regno)
&& ! TEST_HARD_REG_BIT (reload_reg_used_in_op_addr, regno));
case RELOAD_FOR_OTHER_ADDRESS:
return ! TEST_HARD_REG_BIT (reload_reg_used_in_other_addr, regno);
default:
gcc_unreachable ();
}
}
static int
reload_reg_reaches_end_p (unsigned int regno, int reloadnum)
{
int opnum = rld[reloadnum].opnum;
enum reload_type type = rld[reloadnum].when_needed;
int i;
for (i = reloadnum + 1; i < n_reloads; i++)
{
rtx reg;
if (rld[i].opnum != opnum || rld[i].when_needed != type)
continue;
reg = rld[i].reg_rtx;
if (reg == NULL_RTX)
continue;
if (regno >= REGNO (reg) && regno < END_REGNO (reg))
return 0;
}
switch (type)
{
case RELOAD_OTHER:
return 1;
case RELOAD_FOR_OTHER_ADDRESS:
for (i = 0; i < reload_n_operands; i++)
if (TEST_HARD_REG_BIT (reload_reg_used_in_output_addr[i], regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_outaddr_addr[i], regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_output[i], regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_input_addr[i], regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_inpaddr_addr[i], regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_input[i], regno))
return 0;
return (! TEST_HARD_REG_BIT (reload_reg_used_in_op_addr, regno)
&& ! TEST_HARD_REG_BIT (reload_reg_used_in_op_addr_reload, regno)
&& ! TEST_HARD_REG_BIT (reload_reg_used_in_insn, regno)
&& ! TEST_HARD_REG_BIT (reload_reg_used, regno));
case RELOAD_FOR_INPUT_ADDRESS:
case RELOAD_FOR_INPADDR_ADDRESS:
for (i = opnum; i < reload_n_operands; i++)
if (TEST_HARD_REG_BIT (reload_reg_used_in_input[i], regno))
return 0;
if (type == RELOAD_FOR_INPADDR_ADDRESS
&& TEST_HARD_REG_BIT (reload_reg_used_in_input_addr[opnum], regno))
return 0;
for (i = opnum + 1; i < reload_n_operands; i++)
if (TEST_HARD_REG_BIT (reload_reg_used_in_input_addr[i], regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_inpaddr_addr[i], regno))
return 0;
for (i = 0; i < reload_n_operands; i++)
if (TEST_HARD_REG_BIT (reload_reg_used_in_output_addr[i], regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_outaddr_addr[i], regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_output[i], regno))
return 0;
if (TEST_HARD_REG_BIT (reload_reg_used_in_op_addr_reload, regno))
return 0;
return (!TEST_HARD_REG_BIT (reload_reg_used_in_op_addr, regno)
&& !TEST_HARD_REG_BIT (reload_reg_used_in_insn, regno)
&& !TEST_HARD_REG_BIT (reload_reg_used, regno));
case RELOAD_FOR_INPUT:
for (i = opnum + 1; i < reload_n_operands; i++)
if (TEST_HARD_REG_BIT (reload_reg_used_in_input_addr[i], regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_inpaddr_addr[i], regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_input[i], regno))
return 0;
case RELOAD_FOR_OPERAND_ADDRESS:
for (i = 0; i < reload_n_operands; i++)
if (TEST_HARD_REG_BIT (reload_reg_used_in_output_addr[i], regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_outaddr_addr[i], regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_output[i], regno))
return 0;
return (!TEST_HARD_REG_BIT (reload_reg_used, regno));
case RELOAD_FOR_OPADDR_ADDR:
for (i = 0; i < reload_n_operands; i++)
if (TEST_HARD_REG_BIT (reload_reg_used_in_output_addr[i], regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_outaddr_addr[i], regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_output[i], regno))
return 0;
return (!TEST_HARD_REG_BIT (reload_reg_used_in_op_addr, regno)
&& !TEST_HARD_REG_BIT (reload_reg_used_in_insn, regno)
&& !TEST_HARD_REG_BIT (reload_reg_used, regno));
case RELOAD_FOR_INSN:
opnum = reload_n_operands;
case RELOAD_FOR_OUTPUT:
case RELOAD_FOR_OUTPUT_ADDRESS:
case RELOAD_FOR_OUTADDR_ADDRESS:
for (i = 0; i < opnum; i++)
if (TEST_HARD_REG_BIT (reload_reg_used_in_output_addr[i], regno)
|| TEST_HARD_REG_BIT (reload_reg_used_in_outaddr_addr[i], regno))
return 0;
if (type == RELOAD_FOR_OUTADDR_ADDRESS
&& TEST_HARD_REG_BIT (reload_reg_used_in_outaddr_addr[opnum], regno))
return 0;
return 1;
default:
gcc_unreachable ();
}
}
static bool
reload_reg_rtx_reaches_end_p (rtx reg, int reloadnum)
{
unsigned int i;
for (i = REGNO (reg); i < END_REGNO (reg); i++)
if (!reload_reg_reaches_end_p (i, reloadnum))
return false;
return true;
}

static bool
reloads_unique_chain_p (int r1, int r2)
{
int i;
if (! rld[r1].in || ! rld[r2].in)
return false;
if (rld[r1].out || rld[r2].out)
return false;
if (rld[r1].opnum != rld[r2].opnum
|| rtx_equal_p (rld[r1].in, rld[r2].in)
|| rld[r1].optional || rld[r2].optional
|| ! (reg_mentioned_p (rld[r1].in, rld[r2].in)
|| reg_mentioned_p (rld[r2].in, rld[r1].in)))
return false;
if (r1 > r2)
std::swap (r1, r2);
for (i = 0; i < n_reloads; i ++)
if (i != r1 && i != r2 && rld[i].in)
{
if (reg_mentioned_p (rld[r1].in, rld[i].in))
return false;
}
return true;
}
static void
substitute (rtx *where, const_rtx what, rtx repl)
{
const char *fmt;
int i;
enum rtx_code code;
if (*where == 0)
return;
if (*where == what || rtx_equal_p (*where, what))
{
substitute_stack.safe_push (where);
*where = repl;
return;
}
code = GET_CODE (*where);
fmt = GET_RTX_FORMAT (code);
for (i = GET_RTX_LENGTH (code) - 1; i >= 0; i--)
{
if (fmt[i] == 'E')
{
int j;
for (j = XVECLEN (*where, i) - 1; j >= 0; j--)
substitute (&XVECEXP (*where, i, j), what, repl);
}
else if (fmt[i] == 'e')
substitute (&XEXP (*where, i), what, repl);
}
}
static bool
gen_reload_chain_without_interm_reg_p (int r1, int r2)
{
bool result = true;
int regno, code;
rtx out, in;
rtx_insn *insn;
rtx_insn *last = get_last_insn ();
if (reg_mentioned_p (rld[r1].in, rld[r2].in))
std::swap (r1, r2);
gcc_assert (reg_mentioned_p (rld[r2].in, rld[r1].in));
regno = rld[r1].regno >= 0 ? rld[r1].regno : rld[r2].regno;
gcc_assert (regno >= 0);
out = gen_rtx_REG (rld[r1].mode, regno);
in = rld[r1].in;
substitute (&in, rld[r2].in, gen_rtx_REG (rld[r2].mode, regno));
strip_paradoxical_subreg (&in, &out);
if (GET_CODE (in) == PLUS
&& (REG_P (XEXP (in, 0))
|| GET_CODE (XEXP (in, 0)) == SUBREG
|| MEM_P (XEXP (in, 0)))
&& (REG_P (XEXP (in, 1))
|| GET_CODE (XEXP (in, 1)) == SUBREG
|| CONSTANT_P (XEXP (in, 1))
|| MEM_P (XEXP (in, 1))))
{
insn = emit_insn (gen_rtx_SET (out, in));
code = recog_memoized (insn);
result = false;
if (code >= 0)
{
extract_insn (insn);
result = constrain_operands (1, get_enabled_alternatives (insn));
}
delete_insns_since (last);
}
while (!substitute_stack.is_empty ())
{
rtx *where = substitute_stack.pop ();
*where = rld[r2].in;
}
return result;
}
static int
reloads_conflict (int r1, int r2)
{
enum reload_type r1_type = rld[r1].when_needed;
enum reload_type r2_type = rld[r2].when_needed;
int r1_opnum = rld[r1].opnum;
int r2_opnum = rld[r2].opnum;
if (r2_type == RELOAD_OTHER)
return 1;
switch (r1_type)
{
case RELOAD_FOR_INPUT:
return (r2_type == RELOAD_FOR_INSN
|| r2_type == RELOAD_FOR_OPERAND_ADDRESS
|| r2_type == RELOAD_FOR_OPADDR_ADDR
|| r2_type == RELOAD_FOR_INPUT
|| ((r2_type == RELOAD_FOR_INPUT_ADDRESS
|| r2_type == RELOAD_FOR_INPADDR_ADDRESS)
&& r2_opnum > r1_opnum));
case RELOAD_FOR_INPUT_ADDRESS:
return ((r2_type == RELOAD_FOR_INPUT_ADDRESS && r1_opnum == r2_opnum)
|| (r2_type == RELOAD_FOR_INPUT && r2_opnum < r1_opnum));
case RELOAD_FOR_INPADDR_ADDRESS:
return ((r2_type == RELOAD_FOR_INPADDR_ADDRESS && r1_opnum == r2_opnum)
|| (r2_type == RELOAD_FOR_INPUT && r2_opnum < r1_opnum));
case RELOAD_FOR_OUTPUT_ADDRESS:
return ((r2_type == RELOAD_FOR_OUTPUT_ADDRESS && r2_opnum == r1_opnum)
|| (r2_type == RELOAD_FOR_OUTPUT && r2_opnum <= r1_opnum));
case RELOAD_FOR_OUTADDR_ADDRESS:
return ((r2_type == RELOAD_FOR_OUTADDR_ADDRESS && r2_opnum == r1_opnum)
|| (r2_type == RELOAD_FOR_OUTPUT && r2_opnum <= r1_opnum));
case RELOAD_FOR_OPERAND_ADDRESS:
return (r2_type == RELOAD_FOR_INPUT || r2_type == RELOAD_FOR_INSN
|| (r2_type == RELOAD_FOR_OPERAND_ADDRESS
&& (!reloads_unique_chain_p (r1, r2)
|| !gen_reload_chain_without_interm_reg_p (r1, r2))));
case RELOAD_FOR_OPADDR_ADDR:
return (r2_type == RELOAD_FOR_INPUT
|| r2_type == RELOAD_FOR_OPADDR_ADDR);
case RELOAD_FOR_OUTPUT:
return (r2_type == RELOAD_FOR_INSN || r2_type == RELOAD_FOR_OUTPUT
|| ((r2_type == RELOAD_FOR_OUTPUT_ADDRESS
|| r2_type == RELOAD_FOR_OUTADDR_ADDRESS)
&& r2_opnum >= r1_opnum));
case RELOAD_FOR_INSN:
return (r2_type == RELOAD_FOR_INPUT || r2_type == RELOAD_FOR_OUTPUT
|| r2_type == RELOAD_FOR_INSN
|| r2_type == RELOAD_FOR_OPERAND_ADDRESS);
case RELOAD_FOR_OTHER_ADDRESS:
return r2_type == RELOAD_FOR_OTHER_ADDRESS;
case RELOAD_OTHER:
return 1;
default:
gcc_unreachable ();
}
}

static char reload_inherited[MAX_RELOADS];
static rtx_insn *reload_inheritance_insn[MAX_RELOADS];
static rtx reload_override_in[MAX_RELOADS];
static int reload_spill_index[MAX_RELOADS];
static rtx reload_reg_rtx_for_input[MAX_RELOADS];
static rtx reload_reg_rtx_for_output[MAX_RELOADS];
static int
reload_reg_free_for_value_p (int start_regno, int regno, int opnum,
enum reload_type type, rtx value, rtx out,
int reloadnum, int ignore_address_reloads)
{
int time1;
int check_earlyclobber = 0;
int i;
int copy = 0;
if (TEST_HARD_REG_BIT (reload_reg_unavailable, regno))
return 0;
if (out == const0_rtx)
{
copy = 1;
out = NULL_RTX;
}
switch (type)
{
case RELOAD_FOR_OTHER_ADDRESS:
time1 = copy ? 0 : 1;
break;
case RELOAD_OTHER:
time1 = copy ? 1 : MAX_RECOG_OPERANDS * 5 + 5;
break;
case RELOAD_FOR_INPADDR_ADDRESS:
time1 = opnum * 4 + 2;
break;
case RELOAD_FOR_INPUT_ADDRESS:
time1 = opnum * 4 + 3;
break;
case RELOAD_FOR_INPUT:
time1 = copy ? opnum * 4 + 4 : MAX_RECOG_OPERANDS * 4 + 3;
break;
case RELOAD_FOR_OPADDR_ADDR:
time1 = MAX_RECOG_OPERANDS * 4 + 1;
break;
case RELOAD_FOR_OPERAND_ADDRESS:
time1 = copy ? MAX_RECOG_OPERANDS * 4 + 2 : MAX_RECOG_OPERANDS * 4 + 3;
break;
case RELOAD_FOR_OUTADDR_ADDRESS:
time1 = MAX_RECOG_OPERANDS * 4 + 4 + opnum;
break;
case RELOAD_FOR_OUTPUT_ADDRESS:
time1 = MAX_RECOG_OPERANDS * 4 + 5 + opnum;
break;
default:
time1 = MAX_RECOG_OPERANDS * 5 + 5;
}
for (i = 0; i < n_reloads; i++)
{
rtx reg = rld[i].reg_rtx;
if (reg && REG_P (reg)
&& (unsigned) regno - true_regnum (reg) < REG_NREGS (reg)
&& i != reloadnum)
{
rtx other_input = rld[i].in;
if (true_regnum (reg) != start_regno)
other_input = NULL_RTX;
if (! other_input || ! rtx_equal_p (other_input, value)
|| rld[i].out || out)
{
int time2;
switch (rld[i].when_needed)
{
case RELOAD_FOR_OTHER_ADDRESS:
time2 = 0;
break;
case RELOAD_FOR_INPADDR_ADDRESS:
if (type == RELOAD_FOR_INPUT_ADDRESS && reloadnum == i + 1
&& ignore_address_reloads
&& ! rld[reloadnum].out)
continue;
if (type == RELOAD_FOR_INPUT && opnum == rld[i].opnum
&& ignore_address_reloads
&& ! rld[reloadnum].out)
continue;
time2 = rld[i].opnum * 4 + 2;
break;
case RELOAD_FOR_INPUT_ADDRESS:
if (type == RELOAD_FOR_INPUT && opnum == rld[i].opnum
&& ignore_address_reloads
&& ! rld[reloadnum].out)
continue;
time2 = rld[i].opnum * 4 + 3;
break;
case RELOAD_FOR_INPUT:
time2 = rld[i].opnum * 4 + 4;
check_earlyclobber = 1;
break;
case RELOAD_FOR_OPADDR_ADDR:
if (type == RELOAD_FOR_OPERAND_ADDRESS && reloadnum == i + 1
&& ignore_address_reloads
&& ! rld[reloadnum].out)
continue;
time2 = MAX_RECOG_OPERANDS * 4 + 1;
break;
case RELOAD_FOR_OPERAND_ADDRESS:
time2 = MAX_RECOG_OPERANDS * 4 + 2;
check_earlyclobber = 1;
break;
case RELOAD_FOR_INSN:
time2 = MAX_RECOG_OPERANDS * 4 + 3;
break;
case RELOAD_FOR_OUTPUT:
time2 = MAX_RECOG_OPERANDS * 4 + 4;
break;
case RELOAD_FOR_OUTADDR_ADDRESS:
if (type == RELOAD_FOR_OUTPUT_ADDRESS && reloadnum == i + 1
&& ignore_address_reloads
&& ! rld[reloadnum].out)
continue;
time2 = MAX_RECOG_OPERANDS * 4 + 4 + rld[i].opnum;
break;
case RELOAD_FOR_OUTPUT_ADDRESS:
time2 = MAX_RECOG_OPERANDS * 4 + 5 + rld[i].opnum;
break;
case RELOAD_OTHER:
if (! rld[i].in || rtx_equal_p (other_input, value))
{
time2 = MAX_RECOG_OPERANDS * 4 + 4;
if (earlyclobber_operand_p (rld[i].out))
time2 = MAX_RECOG_OPERANDS * 4 + 3;
break;
}
time2 = 1;
if (out)
return 0;
break;
default:
return 0;
}
if ((time1 >= time2
&& (! rld[i].in || rld[i].out
|| ! rtx_equal_p (other_input, value)))
|| (out && rld[reloadnum].out_reg
&& time2 >= MAX_RECOG_OPERANDS * 4 + 3))
return 0;
}
}
}
if (check_earlyclobber && out && earlyclobber_operand_p (out))
return 0;
return 1;
}
static int
free_for_value_p (int regno, machine_mode mode, int opnum,
enum reload_type type, rtx value, rtx out, int reloadnum,
int ignore_address_reloads)
{
int nregs = hard_regno_nregs (regno, mode);
while (nregs-- > 0)
if (! reload_reg_free_for_value_p (regno, regno + nregs, opnum, type,
value, out, reloadnum,
ignore_address_reloads))
return 0;
return 1;
}
int
function_invariant_p (const_rtx x)
{
if (CONSTANT_P (x))
return 1;
if (x == frame_pointer_rtx || x == arg_pointer_rtx)
return 1;
if (GET_CODE (x) == PLUS
&& (XEXP (x, 0) == frame_pointer_rtx || XEXP (x, 0) == arg_pointer_rtx)
&& GET_CODE (XEXP (x, 1)) == CONST_INT)
return 1;
return 0;
}
static int
conflicts_with_override (rtx x)
{
int i;
for (i = 0; i < n_reloads; i++)
if (reload_override_in[i]
&& reg_overlap_mentioned_p (x, reload_override_in[i]))
return 1;
return 0;
}

static void
failed_reload (rtx_insn *insn, int r)
{
if (asm_noperands (PATTERN (insn)) < 0)
fatal_insn ("could not find a spill register", insn);
error_for_asm (insn,
"%<asm%> operand constraint incompatible with operand size");
rld[r].in = 0;
rld[r].out = 0;
rld[r].reg_rtx = 0;
rld[r].optional = 1;
rld[r].secondary_p = 1;
}
static int
set_reload_reg (int i, int r)
{
int regno;
rtx reg = spill_reg_rtx[i];
if (reg == 0 || GET_MODE (reg) != rld[r].mode)
spill_reg_rtx[i] = reg
= gen_rtx_REG (rld[r].mode, spill_regs[i]);
regno = true_regnum (reg);
if (targetm.hard_regno_mode_ok (regno, rld[r].mode))
{
machine_mode test_mode = VOIDmode;
if (rld[r].in)
test_mode = GET_MODE (rld[r].in);
if (! (rld[r].in != 0 && test_mode != VOIDmode
&& !targetm.hard_regno_mode_ok (regno, test_mode)))
if (! (rld[r].out != 0
&& !targetm.hard_regno_mode_ok (regno, GET_MODE (rld[r].out))))
{
last_spill_reg = i;
mark_reload_reg_in_use (spill_regs[i], rld[r].opnum,
rld[r].when_needed, rld[r].mode);
rld[r].reg_rtx = reg;
reload_spill_index[r] = spill_regs[i];
return 1;
}
}
return 0;
}
static int
allocate_reload_reg (struct insn_chain *chain ATTRIBUTE_UNUSED, int r,
int last_reload)
{
int i, pass, count;
int force_group = rld[r].nregs > 1 && ! last_reload;
for (pass = 0; pass < 3; pass++)
{
i = last_spill_reg;
for (count = 0; count < n_spills; count++)
{
int rclass = (int) rld[r].rclass;
int regnum;
i++;
if (i >= n_spills)
i -= n_spills;
regnum = spill_regs[i];
if ((reload_reg_free_p (regnum, rld[r].opnum,
rld[r].when_needed)
|| (rld[r].in
&& ! TEST_HARD_REG_BIT (reload_reg_used, regnum)
&& free_for_value_p (regnum, rld[r].mode, rld[r].opnum,
rld[r].when_needed, rld[r].in,
rld[r].out, r, 1)))
&& TEST_HARD_REG_BIT (reg_class_contents[rclass], regnum)
&& targetm.hard_regno_mode_ok (regnum, rld[r].mode)
&& (pass
|| (TEST_HARD_REG_BIT (reload_reg_used_at_all,
regnum)
&& ! TEST_HARD_REG_BIT (reload_reg_used_for_inherit,
regnum))))
{
int nr = hard_regno_nregs (regnum, rld[r].mode);
if (pass == 1
&& ira_bad_reload_regno (regnum, rld[r].in, rld[r].out))
continue;
if (force_group)
nr = rld[r].nregs;
if (nr == 1)
{
if (force_group)
continue;
break;
}
while (nr > 1)
{
int regno = regnum + nr - 1;
if (!(TEST_HARD_REG_BIT (reg_class_contents[rclass], regno)
&& spill_reg_order[regno] >= 0
&& reload_reg_free_p (regno, rld[r].opnum,
rld[r].when_needed)))
break;
nr--;
}
if (nr == 1)
break;
}
}
if (count < n_spills)
break;
}
if (count >= n_spills)
return 0;
return set_reload_reg (i, r);
}

static void
choose_reload_regs_init (struct insn_chain *chain, rtx *save_reload_reg_rtx)
{
int i;
for (i = 0; i < n_reloads; i++)
rld[i].reg_rtx = save_reload_reg_rtx[i];
memset (reload_inherited, 0, MAX_RELOADS);
memset (reload_inheritance_insn, 0, MAX_RELOADS * sizeof (rtx));
memset (reload_override_in, 0, MAX_RELOADS * sizeof (rtx));
CLEAR_HARD_REG_SET (reload_reg_used);
CLEAR_HARD_REG_SET (reload_reg_used_at_all);
CLEAR_HARD_REG_SET (reload_reg_used_in_op_addr);
CLEAR_HARD_REG_SET (reload_reg_used_in_op_addr_reload);
CLEAR_HARD_REG_SET (reload_reg_used_in_insn);
CLEAR_HARD_REG_SET (reload_reg_used_in_other_addr);
CLEAR_HARD_REG_SET (reg_used_in_insn);
{
HARD_REG_SET tmp;
REG_SET_TO_HARD_REG_SET (tmp, &chain->live_throughout);
IOR_HARD_REG_SET (reg_used_in_insn, tmp);
REG_SET_TO_HARD_REG_SET (tmp, &chain->dead_or_set);
IOR_HARD_REG_SET (reg_used_in_insn, tmp);
compute_use_by_pseudos (&reg_used_in_insn, &chain->live_throughout);
compute_use_by_pseudos (&reg_used_in_insn, &chain->dead_or_set);
}
for (i = 0; i < reload_n_operands; i++)
{
CLEAR_HARD_REG_SET (reload_reg_used_in_output[i]);
CLEAR_HARD_REG_SET (reload_reg_used_in_input[i]);
CLEAR_HARD_REG_SET (reload_reg_used_in_input_addr[i]);
CLEAR_HARD_REG_SET (reload_reg_used_in_inpaddr_addr[i]);
CLEAR_HARD_REG_SET (reload_reg_used_in_output_addr[i]);
CLEAR_HARD_REG_SET (reload_reg_used_in_outaddr_addr[i]);
}
COMPL_HARD_REG_SET (reload_reg_unavailable, chain->used_spill_regs);
CLEAR_HARD_REG_SET (reload_reg_used_for_inherit);
for (i = 0; i < n_reloads; i++)
if (rld[i].reg_rtx)
mark_reload_reg_in_use (REGNO (rld[i].reg_rtx), rld[i].opnum,
rld[i].when_needed, rld[i].mode);
}
static rtx
replaced_subreg (rtx x)
{
if (GET_CODE (x) == SUBREG)
return find_replacement (&SUBREG_REG (x));
return x;
}
static poly_int64
compute_reload_subreg_offset (machine_mode outermode,
rtx subreg,
machine_mode innermode)
{
poly_int64 outer_offset;
machine_mode middlemode;
if (!subreg)
return subreg_lowpart_offset (outermode, innermode);
outer_offset = SUBREG_BYTE (subreg);
middlemode = GET_MODE (SUBREG_REG (subreg));
if (paradoxical_subreg_p (outermode, middlemode))
return subreg_lowpart_offset (outermode, innermode);
return outer_offset + subreg_lowpart_offset (middlemode, innermode);
}
static void
choose_reload_regs (struct insn_chain *chain)
{
rtx_insn *insn = chain->insn;
int i, j;
unsigned int max_group_size = 1;
enum reg_class group_class = NO_REGS;
int pass, win, inheritance;
rtx save_reload_reg_rtx[MAX_RELOADS];
for (j = 0; j < n_reloads; j++)
{
reload_order[j] = j;
if (rld[j].reg_rtx != NULL_RTX)
{
gcc_assert (REG_P (rld[j].reg_rtx)
&& HARD_REGISTER_P (rld[j].reg_rtx));
reload_spill_index[j] = REGNO (rld[j].reg_rtx);
}
else
reload_spill_index[j] = -1;
if (rld[j].nregs > 1)
{
max_group_size = MAX (rld[j].nregs, max_group_size);
group_class
= reg_class_superunion[(int) rld[j].rclass][(int) group_class];
}
save_reload_reg_rtx[j] = rld[j].reg_rtx;
}
if (n_reloads > 1)
qsort (reload_order, n_reloads, sizeof (short), reload_reg_class_lower);
win = 0;
for (inheritance = optimize > 0; inheritance >= 0; inheritance--)
{
choose_reload_regs_init (chain, save_reload_reg_rtx);
for (j = 0; j < n_reloads; j++)
{
int r = reload_order[j];
rtx search_equiv = NULL_RTX;
if (rld[r].out == 0 && rld[r].in == 0
&& ! rld[r].secondary_p)
continue;
if (rld[r].in != 0 && rld[r].reg_rtx != 0
&& (rtx_equal_p (rld[r].in, rld[r].reg_rtx)
|| (rtx_equal_p (rld[r].out, rld[r].reg_rtx)
&& !MEM_P (rld[r].in)
&& true_regnum (rld[r].in) < FIRST_PSEUDO_REGISTER)))
continue;
#if 0 
if (rld[r].optional != 0)
for (i = 0; i < j; i++)
if ((rld[reload_order[i]].out != 0
|| rld[reload_order[i]].in != 0
|| rld[reload_order[i]].secondary_p)
&& ! rld[reload_order[i]].optional
&& rld[reload_order[i]].reg_rtx == 0)
allocate_reload_reg (chain, reload_order[i], 0);
#endif
if (inheritance)
{
poly_int64 byte = 0;
int regno = -1;
machine_mode mode = VOIDmode;
rtx subreg = NULL_RTX;
if (rld[r].in == 0)
;
else if (REG_P (rld[r].in))
{
regno = REGNO (rld[r].in);
mode = GET_MODE (rld[r].in);
}
else if (REG_P (rld[r].in_reg))
{
regno = REGNO (rld[r].in_reg);
mode = GET_MODE (rld[r].in_reg);
}
else if (GET_CODE (rld[r].in_reg) == SUBREG
&& REG_P (SUBREG_REG (rld[r].in_reg)))
{
regno = REGNO (SUBREG_REG (rld[r].in_reg));
if (regno < FIRST_PSEUDO_REGISTER)
regno = subreg_regno (rld[r].in_reg);
else
{
subreg = rld[r].in_reg;
byte = SUBREG_BYTE (subreg);
}
mode = GET_MODE (rld[r].in_reg);
}
#if AUTO_INC_DEC
else if (GET_RTX_CLASS (GET_CODE (rld[r].in_reg)) == RTX_AUTOINC
&& REG_P (XEXP (rld[r].in_reg, 0)))
{
regno = REGNO (XEXP (rld[r].in_reg, 0));
mode = GET_MODE (XEXP (rld[r].in_reg, 0));
rld[r].out = rld[r].in;
}
#endif
#if 0
else if (GET_CODE (rld[r].in) == SUBREG
&& REG_P (SUBREG_REG (rld[r].in)))
regno = subreg_regno (rld[r].in);
#endif
if (regno >= 0
&& reg_last_reload_reg[regno] != 0
&& (known_ge
(GET_MODE_SIZE (GET_MODE (reg_last_reload_reg[regno])),
GET_MODE_SIZE (mode) + byte))
&& (REG_CAN_CHANGE_MODE_P
(REGNO (reg_last_reload_reg[regno]),
GET_MODE (reg_last_reload_reg[regno]),
mode)))
{
enum reg_class rclass = rld[r].rclass, last_class;
rtx last_reg = reg_last_reload_reg[regno];
i = REGNO (last_reg);
byte = compute_reload_subreg_offset (mode,
subreg,
GET_MODE (last_reg));
i += subreg_regno_offset (i, GET_MODE (last_reg), byte, mode);
last_class = REGNO_REG_CLASS (i);
if (reg_reloaded_contents[i] == regno
&& TEST_HARD_REG_BIT (reg_reloaded_valid, i)
&& targetm.hard_regno_mode_ok (i, rld[r].mode)
&& (TEST_HARD_REG_BIT (reg_class_contents[(int) rclass], i)
|| ((register_move_cost (mode, last_class, rclass)
< memory_move_cost (mode, rclass, true))
&& (secondary_reload_class (1, rclass, mode,
last_reg)
== NO_REGS)
&& !(targetm.secondary_memory_needed
(mode, last_class, rclass))))
&& (rld[r].nregs == max_group_size
|| ! TEST_HARD_REG_BIT (reg_class_contents[(int) group_class],
i))
&& free_for_value_p (i, rld[r].mode, rld[r].opnum,
rld[r].when_needed, rld[r].in,
const0_rtx, r, 1))
{
int nr = hard_regno_nregs (i, rld[r].mode);
int k;
for (k = 1; k < nr; k++)
if (reg_reloaded_contents[i + k] != regno
|| ! TEST_HARD_REG_BIT (reg_reloaded_valid, i + k))
break;
if (k == nr)
{
int i1;
int bad_for_class;
last_reg = (GET_MODE (last_reg) == mode
? last_reg : gen_rtx_REG (mode, i));
bad_for_class = 0;
for (k = 0; k < nr; k++)
bad_for_class |= ! TEST_HARD_REG_BIT (reg_class_contents[(int) rld[r].rclass],
i+k);
for (i1 = 0; i1 < n_earlyclobbers; i1++)
if (reg_overlap_mentioned_for_reload_p
(reg_last_reload_reg[regno],
reload_earlyclobbers[i1]))
break;
if (i1 != n_earlyclobbers
|| ! (free_for_value_p (i, rld[r].mode,
rld[r].opnum,
rld[r].when_needed, rld[r].in,
rld[r].out, r, 1))
|| (TEST_HARD_REG_BIT (reg_used_in_insn, i)
&& rld[r].out
&& ! TEST_HARD_REG_BIT (reg_reloaded_dead, i))
|| (i == HARD_FRAME_POINTER_REGNUM
&& frame_pointer_needed
&& rld[r].out)
|| paradoxical_subreg_p (rld[r].mode, mode)
|| bad_for_class
|| (rld[r].out && rld[r].reg_rtx
&& rtx_equal_p (rld[r].out, rld[r].reg_rtx)))
{
if (! rld[r].optional)
{
reload_override_in[r] = last_reg;
reload_inheritance_insn[r]
= reg_reloaded_insn[i];
}
}
else
{
int k;
mark_reload_reg_in_use (i,
rld[r].opnum,
rld[r].when_needed,
rld[r].mode);
rld[r].reg_rtx = last_reg;
reload_inherited[r] = 1;
reload_inheritance_insn[r]
= reg_reloaded_insn[i];
reload_spill_index[r] = i;
for (k = 0; k < nr; k++)
SET_HARD_REG_BIT (reload_reg_used_for_inherit,
i + k);
}
}
}
}
}
if (inheritance
&& rld[r].in != 0
&& ! reload_inherited[r]
&& rld[r].out == 0
&& (CONSTANT_P (rld[r].in)
|| GET_CODE (rld[r].in) == PLUS
|| REG_P (rld[r].in)
|| MEM_P (rld[r].in))
&& (rld[r].nregs == max_group_size
|| ! reg_classes_intersect_p (rld[r].rclass, group_class)))
search_equiv = rld[r].in;
if (search_equiv)
{
rtx equiv
= find_equiv_reg (search_equiv, insn, rld[r].rclass,
-1, NULL, 0, rld[r].mode);
int regno = 0;
if (equiv != 0)
{
if (REG_P (equiv))
regno = REGNO (equiv);
else
{
gcc_assert (GET_CODE (equiv) == SUBREG);
regno = subreg_regno (equiv);
equiv = gen_rtx_REG (rld[r].mode, regno);
for (i = regno; i < regno + (int) rld[r].nregs; i++)
if (TEST_HARD_REG_BIT (reload_reg_unavailable, i))
equiv = 0;
}
}
if (equiv != 0)
{
int regs_used = 0;
int bad_for_class = 0;
int max_regno = regno + rld[r].nregs;
for (i = regno; i < max_regno; i++)
{
regs_used |= TEST_HARD_REG_BIT (reload_reg_used_at_all,
i);
bad_for_class |= ! TEST_HARD_REG_BIT (reg_class_contents[(int) rld[r].rclass],
i);
}
if ((regs_used
&& ! free_for_value_p (regno, rld[r].mode,
rld[r].opnum, rld[r].when_needed,
rld[r].in, rld[r].out, r, 1))
|| bad_for_class)
equiv = 0;
}
if (equiv != 0
&& !targetm.hard_regno_mode_ok (regno, rld[r].mode))
equiv = 0;
if (equiv != 0)
for (i = 0; i < n_earlyclobbers; i++)
if (reg_overlap_mentioned_for_reload_p (equiv,
reload_earlyclobbers[i]))
{
if (! rld[r].optional)
reload_override_in[r] = equiv;
equiv = 0;
break;
}
if (equiv != 0)
{
if (regno_clobbered_p (regno, insn, rld[r].mode, 2))
switch (rld[r].when_needed)
{
case RELOAD_FOR_OTHER_ADDRESS:
case RELOAD_FOR_INPADDR_ADDRESS:
case RELOAD_FOR_INPUT_ADDRESS:
case RELOAD_FOR_OPADDR_ADDR:
break;
case RELOAD_OTHER:
case RELOAD_FOR_INPUT:
case RELOAD_FOR_OPERAND_ADDRESS:
if (! rld[r].optional)
reload_override_in[r] = equiv;
default:
equiv = 0;
break;
}
else if (regno_clobbered_p (regno, insn, rld[r].mode, 1))
switch (rld[r].when_needed)
{
case RELOAD_FOR_OTHER_ADDRESS:
case RELOAD_FOR_INPADDR_ADDRESS:
case RELOAD_FOR_INPUT_ADDRESS:
case RELOAD_FOR_OPADDR_ADDR:
case RELOAD_FOR_OPERAND_ADDRESS:
case RELOAD_FOR_INPUT:
break;
case RELOAD_OTHER:
if (! rld[r].optional)
reload_override_in[r] = equiv;
default:
equiv = 0;
break;
}
}
if (equiv != 0
&& (regno != HARD_FRAME_POINTER_REGNUM
|| !frame_pointer_needed))
{
int nr = hard_regno_nregs (regno, rld[r].mode);
int k;
rld[r].reg_rtx = equiv;
reload_spill_index[r] = regno;
reload_inherited[r] = 1;
if (! TEST_HARD_REG_BIT (reg_reloaded_valid, regno))
spill_reg_store[regno] = NULL;
for (k = 0; k < nr; k++)
{
i = spill_reg_order[regno + k];
if (i >= 0)
{
mark_reload_reg_in_use (regno, rld[r].opnum,
rld[r].when_needed,
rld[r].mode);
SET_HARD_REG_BIT (reload_reg_used_for_inherit,
regno + k);
}
}
}
}
if (rld[r].reg_rtx != 0 || rld[r].optional != 0)
continue;
#if 0
for (i = j + 1; i < n_reloads; i++)
{
int s = reload_order[i];
if ((rld[s].in == 0 && rld[s].out == 0
&& ! rld[s].secondary_p)
|| rld[s].optional)
continue;
if ((rld[s].rclass != rld[r].rclass
&& reg_classes_intersect_p (rld[r].rclass,
rld[s].rclass))
|| rld[s].nregs < rld[r].nregs)
break;
}
if (i == n_reloads)
continue;
allocate_reload_reg (chain, r, j == n_reloads - 1);
#endif
}
for (j = 0; j < n_reloads; j++)
{
int r = reload_order[j];
if (rld[r].out == 0 && rld[r].in == 0 && ! rld[r].secondary_p)
continue;
if (rld[r].reg_rtx != 0 || rld[r].optional)
continue;
if (! allocate_reload_reg (chain, r, j == n_reloads - 1))
break;
}
if (j == n_reloads)
{
win = 1;
break;
}
}
if (! win)
{
choose_reload_regs_init (chain, save_reload_reg_rtx);
gcc_assert (chain->n_reloads == n_reloads);
for (i = 0; i < n_reloads; i++)
{
if (chain->rld[i].regno < 0 || chain->rld[i].reg_rtx != 0)
continue;
gcc_assert (chain->rld[i].when_needed == rld[i].when_needed);
for (j = 0; j < n_spills; j++)
if (spill_regs[j] == chain->rld[i].regno)
if (! set_reload_reg (j, i))
failed_reload (chain->insn, i);
}
}
for (pass = flag_expensive_optimizations; pass >= 0; pass--)
{
for (j = 0; j < n_reloads; j++)
{
int r = reload_order[j];
rtx check_reg;
rtx tem;
if (reload_inherited[r] && rld[r].reg_rtx)
check_reg = rld[r].reg_rtx;
else if (reload_override_in[r]
&& (REG_P (reload_override_in[r])
|| GET_CODE (reload_override_in[r]) == SUBREG))
check_reg = reload_override_in[r];
else
continue;
if (! free_for_value_p (true_regnum (check_reg), rld[r].mode,
rld[r].opnum, rld[r].when_needed, rld[r].in,
(reload_inherited[r]
? rld[r].out : const0_rtx),
r, 1))
{
if (pass)
continue;
reload_inherited[r] = 0;
reload_override_in[r] = 0;
}
else if (rld[r].in
&& rld[r].out != rld[r].in
&& remove_address_replacements (rld[r].in))
{
if (pass)
pass = 2;
}
else if (rld[r].in
&& rld[r].out != rld[r].in
&& (tem = replaced_subreg (rld[r].in), REG_P (tem))		   
&& REGNO (tem) < FIRST_PSEUDO_REGISTER
&& (targetm.secondary_memory_needed
(rld[r].inmode, REGNO_REG_CLASS (REGNO (tem)),
rld[r].rclass))
&& remove_address_replacements
(get_secondary_mem (tem, rld[r].inmode, rld[r].opnum,
rld[r].when_needed)))
{
if (pass)
pass = 2;
}
}
}
for (j = 0; j < n_reloads; j++)
if (reload_override_in[j])
rld[j].in = reload_override_in[j];
for (j = 0; j < n_reloads; j++)
if (rld[j].reg_rtx != 0
&& ((rld[j].optional && ! reload_inherited[j])
|| (rld[j].in == 0 && rld[j].out == 0
&& ! rld[j].secondary_p)))
{
int regno = true_regnum (rld[j].reg_rtx);
if (spill_reg_order[regno] >= 0)
clear_reload_reg_in_use (regno, rld[j].opnum,
rld[j].when_needed, rld[j].mode);
rld[j].reg_rtx = 0;
reload_spill_index[j] = -1;
}
for (j = 0; j < n_reloads; j++)
{
int r = reload_order[j];
i = reload_spill_index[r];
if (rld[r].out_reg != 0 && REG_P (rld[r].out_reg)
&& rld[r].reg_rtx != 0)
{
int nregno = REGNO (rld[r].out_reg);
int nr = 1;
if (nregno < FIRST_PSEUDO_REGISTER)
nr = hard_regno_nregs (nregno, rld[r].mode);
while (--nr >= 0)
SET_REGNO_REG_SET (&reg_has_output_reload,
nregno + nr);
if (i >= 0)
add_to_hard_reg_set (&reg_is_output_reload, rld[r].mode, i);
gcc_assert (rld[r].when_needed == RELOAD_OTHER
|| rld[r].when_needed == RELOAD_FOR_OUTPUT
|| rld[r].when_needed == RELOAD_FOR_INSN);
}
}
}
void
deallocate_reload_reg (int r)
{
int regno;
if (! rld[r].reg_rtx)
return;
regno = true_regnum (rld[r].reg_rtx);
rld[r].reg_rtx = 0;
if (spill_reg_order[regno] >= 0)
clear_reload_reg_in_use (regno, rld[r].opnum, rld[r].when_needed,
rld[r].mode);
reload_spill_index[r] = -1;
}

static rtx_insn *input_reload_insns[MAX_RECOG_OPERANDS];
static rtx_insn *other_input_address_reload_insns = 0;
static rtx_insn *other_input_reload_insns = 0;
static rtx_insn *input_address_reload_insns[MAX_RECOG_OPERANDS];
static rtx_insn *inpaddr_address_reload_insns[MAX_RECOG_OPERANDS];
static rtx_insn *output_reload_insns[MAX_RECOG_OPERANDS];
static rtx_insn *output_address_reload_insns[MAX_RECOG_OPERANDS];
static rtx_insn *outaddr_address_reload_insns[MAX_RECOG_OPERANDS];
static rtx_insn *operand_reload_insns = 0;
static rtx_insn *other_operand_reload_insns = 0;
static rtx_insn *other_output_reload_insns[MAX_RECOG_OPERANDS];
static rtx_insn *new_spill_reg_store[FIRST_PSEUDO_REGISTER];
static HARD_REG_SET reg_reloaded_died;
static bool
reload_adjust_reg_for_temp (rtx *reload_reg, rtx alt_reload_reg,
enum reg_class new_class,
machine_mode new_mode)
{
rtx reg;
for (reg = *reload_reg; reg; reg = alt_reload_reg, alt_reload_reg = 0)
{
unsigned regno = REGNO (reg);
if (!TEST_HARD_REG_BIT (reg_class_contents[(int) new_class], regno))
continue;
if (GET_MODE (reg) != new_mode)
{
if (!targetm.hard_regno_mode_ok (regno, new_mode))
continue;
if (hard_regno_nregs (regno, new_mode) > REG_NREGS (reg))
continue;
reg = reload_adjust_reg_for_mode (reg, new_mode);
}
*reload_reg = reg;
return true;
}
return false;
}
static bool
reload_adjust_reg_for_icode (rtx *reload_reg, rtx alt_reload_reg,
enum insn_code icode)
{
enum reg_class new_class = scratch_reload_class (icode);
machine_mode new_mode = insn_data[(int) icode].operand[2].mode;
return reload_adjust_reg_for_temp (reload_reg, alt_reload_reg,
new_class, new_mode);
}
static void
emit_input_reload_insns (struct insn_chain *chain, struct reload *rl,
rtx old, int j)
{
rtx_insn *insn = chain->insn;
rtx reloadreg;
rtx oldequiv_reg = 0;
rtx oldequiv = 0;
int special = 0;
machine_mode mode;
rtx_insn **where;
if (reload_override_in[j]
&& (REG_P (rl->in_reg)
|| (GET_CODE (rl->in_reg) == SUBREG
&& REG_P (SUBREG_REG (rl->in_reg)))))
{
oldequiv = old;
old = rl->in_reg;
}
if (oldequiv == 0)
oldequiv = old;
else if (REG_P (oldequiv))
oldequiv_reg = oldequiv;
else if (GET_CODE (oldequiv) == SUBREG)
oldequiv_reg = SUBREG_REG (oldequiv);
reloadreg = reload_reg_rtx_for_input[j];
mode = GET_MODE (reloadreg);
if (optimize && REG_P (oldequiv)
&& REGNO (oldequiv) < FIRST_PSEUDO_REGISTER
&& spill_reg_store[REGNO (oldequiv)]
&& REG_P (old)
&& (dead_or_set_p (insn, spill_reg_stored_to[REGNO (oldequiv)])
|| rtx_equal_p (spill_reg_stored_to[REGNO (oldequiv)],
rl->out_reg)))
delete_output_reload (insn, j, REGNO (oldequiv), reloadreg);
while (GET_CODE (oldequiv) == SUBREG && GET_MODE (oldequiv) != mode)
oldequiv = SUBREG_REG (oldequiv);
if (GET_MODE (oldequiv) != VOIDmode
&& mode != GET_MODE (oldequiv))
oldequiv = gen_lowpart_SUBREG (mode, oldequiv);
switch (rl->when_needed)
{
case RELOAD_OTHER:
where = &other_input_reload_insns;
break;
case RELOAD_FOR_INPUT:
where = &input_reload_insns[rl->opnum];
break;
case RELOAD_FOR_INPUT_ADDRESS:
where = &input_address_reload_insns[rl->opnum];
break;
case RELOAD_FOR_INPADDR_ADDRESS:
where = &inpaddr_address_reload_insns[rl->opnum];
break;
case RELOAD_FOR_OUTPUT_ADDRESS:
where = &output_address_reload_insns[rl->opnum];
break;
case RELOAD_FOR_OUTADDR_ADDRESS:
where = &outaddr_address_reload_insns[rl->opnum];
break;
case RELOAD_FOR_OPERAND_ADDRESS:
where = &operand_reload_insns;
break;
case RELOAD_FOR_OPADDR_ADDR:
where = &other_operand_reload_insns;
break;
case RELOAD_FOR_OTHER_ADDRESS:
where = &other_input_address_reload_insns;
break;
default:
gcc_unreachable ();
}
push_to_sequence (*where);
if (rl->out && ! rl->out_reg)
{
gcc_assert (rl->secondary_in_reload < 0);
if (reload_inherited[j])
oldequiv = reloadreg;
old = XEXP (rl->in_reg, 0);
special = 1;
inc_for_reload (reloadreg, oldequiv, rl->out, rl->inc);
}
else if (optimize && REG_P (old)
&& REGNO (old) >= FIRST_PSEUDO_REGISTER
&& dead_or_set_p (insn, old)
&& ! conflicts_with_override (reloadreg)
&& free_for_value_p (REGNO (reloadreg), rl->mode, rl->opnum,
rl->when_needed, old, rl->out, j, 0))
{
rtx_insn *temp = PREV_INSN (insn);
while (temp && (NOTE_P (temp) || DEBUG_INSN_P (temp)))
temp = PREV_INSN (temp);
if (temp
&& NONJUMP_INSN_P (temp)
&& GET_CODE (PATTERN (temp)) == SET
&& SET_DEST (PATTERN (temp)) == old
&& asm_noperands (PATTERN (temp)) < 0
&& count_occurrences (PATTERN (insn), old, 0) == 1)
{
rtx old = SET_DEST (PATTERN (temp));
SET_DEST (PATTERN (temp)) = reloadreg;
extract_insn (temp);
if (constrain_operands (1, get_enabled_alternatives (temp))
&& (!AUTO_INC_DEC || ! find_reg_note (temp, REG_INC, reloadreg)))
{
if (REG_P (SET_SRC (PATTERN (temp)))
&& REGNO (SET_SRC (PATTERN (temp))) < FIRST_PSEUDO_REGISTER)
{
spill_reg_store[REGNO (SET_SRC (PATTERN (temp)))] = 0;
spill_reg_stored_to[REGNO (SET_SRC (PATTERN (temp)))] = 0;
}
if (REG_N_DEATHS (REGNO (old)) == 1
&& REG_N_SETS (REGNO (old)) == 1)
{
reg_renumber[REGNO (old)] = REGNO (reloadreg);
if (ira_conflicts_p)
ira_mark_allocation_change (REGNO (old));
alter_reg (REGNO (old), -1, false);
}
special = 1;
while ((temp = NEXT_INSN (temp)) != insn)
if (DEBUG_BIND_INSN_P (temp))
INSN_VAR_LOCATION_LOC (temp)
= simplify_replace_rtx (INSN_VAR_LOCATION_LOC (temp),
old, reloadreg);
else
gcc_assert (DEBUG_INSN_P (temp) || NOTE_P (temp));
}
else
{
SET_DEST (PATTERN (temp)) = old;
}
}
}
if (! special && rl->secondary_in_reload >= 0)
{
rtx second_reload_reg = 0;
rtx third_reload_reg = 0;
int secondary_reload = rl->secondary_in_reload;
rtx real_oldequiv = oldequiv;
rtx real_old = old;
rtx tmp;
enum insn_code icode;
enum insn_code tertiary_icode = CODE_FOR_nothing;
tmp = oldequiv;
if (GET_CODE (tmp) == SUBREG)
tmp = SUBREG_REG (tmp);
if (REG_P (tmp)
&& REGNO (tmp) >= FIRST_PSEUDO_REGISTER
&& (reg_equiv_memory_loc (REGNO (tmp)) != 0
|| reg_equiv_constant (REGNO (tmp)) != 0))
{
if (! reg_equiv_mem (REGNO (tmp))
|| num_not_at_initial_offset
|| GET_CODE (oldequiv) == SUBREG)
real_oldequiv = rl->in;
else
real_oldequiv = reg_equiv_mem (REGNO (tmp));
}
tmp = old;
if (GET_CODE (tmp) == SUBREG)
tmp = SUBREG_REG (tmp);
if (REG_P (tmp)
&& REGNO (tmp) >= FIRST_PSEUDO_REGISTER
&& (reg_equiv_memory_loc (REGNO (tmp)) != 0
|| reg_equiv_constant (REGNO (tmp)) != 0))
{
if (! reg_equiv_mem (REGNO (tmp))
|| num_not_at_initial_offset
|| GET_CODE (old) == SUBREG)
real_old = rl->in;
else
real_old = reg_equiv_mem (REGNO (tmp));
}
second_reload_reg = rld[secondary_reload].reg_rtx;
if (rld[secondary_reload].secondary_in_reload >= 0)
{
int tertiary_reload = rld[secondary_reload].secondary_in_reload;
third_reload_reg = rld[tertiary_reload].reg_rtx;
tertiary_icode = rld[secondary_reload].secondary_in_icode;
gcc_assert (rld[tertiary_reload].secondary_in_reload < 0);
}
icode = rl->secondary_in_icode;
if ((old != oldequiv && ! rtx_equal_p (old, oldequiv))
|| (rl->in != 0 && rl->out != 0))
{
secondary_reload_info sri, sri2;
enum reg_class new_class, new_t_class;
sri.icode = CODE_FOR_nothing;
sri.prev_sri = NULL;
new_class
= (enum reg_class) targetm.secondary_reload (1, real_oldequiv,
rl->rclass, mode,
&sri);
if (new_class == NO_REGS && sri.icode == CODE_FOR_nothing)
second_reload_reg = 0;
else if (new_class == NO_REGS)
{
if (reload_adjust_reg_for_icode (&second_reload_reg,
third_reload_reg,
(enum insn_code) sri.icode))
{
icode = (enum insn_code) sri.icode;
third_reload_reg = 0;
}
else
{
oldequiv = old;
real_oldequiv = real_old;
}
}
else if (sri.icode != CODE_FOR_nothing)
gcc_unreachable ();
else
{
sri2.icode = CODE_FOR_nothing;
sri2.prev_sri = &sri;
new_t_class
= (enum reg_class) targetm.secondary_reload (1, real_oldequiv,
new_class, mode,
&sri);
if (new_t_class == NO_REGS && sri2.icode == CODE_FOR_nothing)
{
if (reload_adjust_reg_for_temp (&second_reload_reg,
third_reload_reg,
new_class, mode))
{
third_reload_reg = 0;
tertiary_icode = (enum insn_code) sri2.icode;
}
else
{
oldequiv = old;
real_oldequiv = real_old;
}
}
else if (new_t_class == NO_REGS && sri2.icode != CODE_FOR_nothing)
{
rtx intermediate = second_reload_reg;
if (reload_adjust_reg_for_temp (&intermediate, NULL,
new_class, mode)
&& reload_adjust_reg_for_icode (&third_reload_reg, NULL,
((enum insn_code)
sri2.icode)))
{
second_reload_reg = intermediate;
tertiary_icode = (enum insn_code) sri2.icode;
}
else
{
oldequiv = old;
real_oldequiv = real_old;
}
}
else if (new_t_class != NO_REGS && sri2.icode == CODE_FOR_nothing)
{
rtx intermediate = second_reload_reg;
if (reload_adjust_reg_for_temp (&intermediate, NULL,
new_class, mode)
&& reload_adjust_reg_for_temp (&third_reload_reg, NULL,
new_t_class, mode))
{
second_reload_reg = intermediate;
tertiary_icode = (enum insn_code) sri2.icode;
}
else
{
oldequiv = old;
real_oldequiv = real_old;
}
}
else
{
oldequiv = old;
real_oldequiv = real_old;
}
}
}
if (second_reload_reg)
{
if (icode != CODE_FOR_nothing)
{
gcc_assert (!third_reload_reg);
emit_insn (GEN_FCN (icode) (reloadreg, real_oldequiv,
second_reload_reg));
special = 1;
}
else
{
if (tertiary_icode != CODE_FOR_nothing)
{
emit_insn ((GEN_FCN (tertiary_icode)
(second_reload_reg, real_oldequiv,
third_reload_reg)));
}
else if (third_reload_reg)
{
gen_reload (third_reload_reg, real_oldequiv,
rl->opnum,
rl->when_needed);
gen_reload (second_reload_reg, third_reload_reg,
rl->opnum,
rl->when_needed);
}
else
gen_reload (second_reload_reg, real_oldequiv,
rl->opnum,
rl->when_needed);
oldequiv = second_reload_reg;
}
}
}
if (! special && ! rtx_equal_p (reloadreg, oldequiv))
{
rtx real_oldequiv = oldequiv;
if ((REG_P (oldequiv)
&& REGNO (oldequiv) >= FIRST_PSEUDO_REGISTER
&& (reg_equiv_memory_loc (REGNO (oldequiv)) != 0
|| reg_equiv_constant (REGNO (oldequiv)) != 0))
|| (GET_CODE (oldequiv) == SUBREG
&& REG_P (SUBREG_REG (oldequiv))
&& (REGNO (SUBREG_REG (oldequiv))
>= FIRST_PSEUDO_REGISTER)
&& ((reg_equiv_memory_loc (REGNO (SUBREG_REG (oldequiv))) != 0)
|| (reg_equiv_constant (REGNO (SUBREG_REG (oldequiv))) != 0)))
|| (CONSTANT_P (oldequiv)
&& (targetm.preferred_reload_class (oldequiv,
REGNO_REG_CLASS (REGNO (reloadreg)))
== NO_REGS)))
real_oldequiv = rl->in;
gen_reload (reloadreg, real_oldequiv, rl->opnum,
rl->when_needed);
}
if (cfun->can_throw_non_call_exceptions)
copy_reg_eh_region_note_forward (insn, get_insns (), NULL);
*where = get_insns ();
end_sequence ();
if (oldequiv_reg)
reload_override_in[j] = oldequiv;
}
static void
emit_output_reload_insns (struct insn_chain *chain, struct reload *rl,
int j)
{
rtx reloadreg;
rtx_insn *insn = chain->insn;
int special = 0;
rtx old = rl->out;
machine_mode mode;
rtx_insn *p;
rtx rl_reg_rtx;
if (rl->when_needed == RELOAD_OTHER)
start_sequence ();
else
push_to_sequence (output_reload_insns[rl->opnum]);
rl_reg_rtx = reload_reg_rtx_for_output[j];
mode = GET_MODE (rl_reg_rtx);
reloadreg = rl_reg_rtx;
if (rl->secondary_out_reload >= 0)
{
rtx real_old = old;
int secondary_reload = rl->secondary_out_reload;
int tertiary_reload = rld[secondary_reload].secondary_out_reload;
if (REG_P (old) && REGNO (old) >= FIRST_PSEUDO_REGISTER
&& reg_equiv_mem (REGNO (old)) != 0)
real_old = reg_equiv_mem (REGNO (old));
if (secondary_reload_class (0, rl->rclass, mode, real_old) != NO_REGS)
{
rtx second_reloadreg = reloadreg;
reloadreg = rld[secondary_reload].reg_rtx;
if (rl->secondary_out_icode != CODE_FOR_nothing)
{
gcc_assert (tertiary_reload < 0);
emit_insn ((GEN_FCN (rl->secondary_out_icode)
(real_old, second_reloadreg, reloadreg)));
special = 1;
}
else
{
enum insn_code tertiary_icode
= rld[secondary_reload].secondary_out_icode;
gcc_assert (tertiary_reload < 0
|| rld[tertiary_reload].secondary_out_reload < 0);
if (GET_MODE (reloadreg) != mode)
reloadreg = reload_adjust_reg_for_mode (reloadreg, mode);
if (tertiary_icode != CODE_FOR_nothing)
{
rtx third_reloadreg = rld[tertiary_reload].reg_rtx;
strip_paradoxical_subreg (&real_old, &reloadreg);
gen_reload (reloadreg, second_reloadreg,
rl->opnum, rl->when_needed);
emit_insn ((GEN_FCN (tertiary_icode)
(real_old, reloadreg, third_reloadreg)));
special = 1;
}
else
{
gen_reload (reloadreg, second_reloadreg,
rl->opnum, rl->when_needed);
if (tertiary_reload >= 0)
{
rtx third_reloadreg = rld[tertiary_reload].reg_rtx;
gen_reload (third_reloadreg, reloadreg,
rl->opnum, rl->when_needed);
reloadreg = third_reloadreg;
}
}
}
}
}
if (! special)
{
rtx set;
if (! flag_expensive_optimizations
|| !REG_P (old)
|| !(set = single_set (insn))
|| rtx_equal_p (old, SET_DEST (set))
|| !reg_mentioned_p (old, SET_SRC (set))
|| !((REGNO (old) < FIRST_PSEUDO_REGISTER)
&& regno_clobbered_p (REGNO (old), insn, rl->mode, 0)))
gen_reload (old, reloadreg, rl->opnum,
rl->when_needed);
}
for (p = get_insns (); p; p = NEXT_INSN (p))
if (INSN_P (p))
{
rtx pat = PATTERN (p);
note_stores (pat, forget_old_reloads_1, NULL);
if (reg_mentioned_p (rl_reg_rtx, pat))
{
rtx set = single_set (insn);
if (reload_spill_index[j] < 0
&& set
&& SET_SRC (set) == rl_reg_rtx)
{
int src = REGNO (SET_SRC (set));
reload_spill_index[j] = src;
SET_HARD_REG_BIT (reg_is_output_reload, src);
if (find_regno_note (insn, REG_DEAD, src))
SET_HARD_REG_BIT (reg_reloaded_died, src);
}
if (HARD_REGISTER_P (rl_reg_rtx))
{
int s = rl->secondary_out_reload;
set = single_set (p);
if (s >= 0 && set == NULL_RTX)
;
else if (s >= 0
&& SET_SRC (set) == rl_reg_rtx
&& SET_DEST (set) == rld[s].reg_rtx)
{
rtx s_reg = rld[s].reg_rtx;
rtx_insn *next = NEXT_INSN (p);
rld[s].out = rl->out;
rld[s].out_reg = rl->out_reg;
set = single_set (next);
if (set && SET_SRC (set) == s_reg
&& reload_reg_rtx_reaches_end_p (s_reg, s))
{
SET_HARD_REG_BIT (reg_is_output_reload,
REGNO (s_reg));
new_spill_reg_store[REGNO (s_reg)] = next;
}
}
else if (reload_reg_rtx_reaches_end_p (rl_reg_rtx, j))
new_spill_reg_store[REGNO (rl_reg_rtx)] = p;
}
}
}
if (rl->when_needed == RELOAD_OTHER)
{
emit_insn (other_output_reload_insns[rl->opnum]);
other_output_reload_insns[rl->opnum] = get_insns ();
}
else
output_reload_insns[rl->opnum] = get_insns ();
if (cfun->can_throw_non_call_exceptions)
copy_reg_eh_region_note_forward (insn, get_insns (), NULL);
end_sequence ();
}
static void
do_input_reload (struct insn_chain *chain, struct reload *rl, int j)
{
rtx_insn *insn = chain->insn;
rtx old = (rl->in && MEM_P (rl->in)
? rl->in_reg : rl->in);
rtx reg_rtx = rl->reg_rtx;
if (old && reg_rtx)
{
machine_mode mode;
mode = GET_MODE (old);
if (mode == VOIDmode)
mode = rl->inmode;
if (GET_MODE (reg_rtx) != mode)
reg_rtx = reload_adjust_reg_for_mode (reg_rtx, mode);
}
reload_reg_rtx_for_input[j] = reg_rtx;
if (old != 0
&& (! reload_inherited[j] || (rl->out && ! rl->out_reg))
&& ! rtx_equal_p (reg_rtx, old)
&& reg_rtx != 0)
emit_input_reload_insns (chain, rld + j, old, j);
if (optimize && reload_inherited[j] && rl->in
&& MEM_P (rl->in)
&& MEM_P (rl->in_reg)
&& reload_spill_index[j] >= 0
&& TEST_HARD_REG_BIT (reg_reloaded_valid, reload_spill_index[j]))
rl->in = regno_reg_rtx[reg_reloaded_contents[reload_spill_index[j]]];
if (optimize
&& (reload_inherited[j] || reload_override_in[j])
&& reg_rtx
&& REG_P (reg_rtx)
&& spill_reg_store[REGNO (reg_rtx)] != 0
#if 0
&& !HARD_REGISTER_P (spill_reg_stored_to[REGNO (reg_rtx)])
#endif
&& (dead_or_set_p (insn, spill_reg_stored_to[REGNO (reg_rtx)])
|| rtx_equal_p (spill_reg_stored_to[REGNO (reg_rtx)], rl->out_reg)))
delete_output_reload (insn, j, REGNO (reg_rtx), reg_rtx);
}
static void
do_output_reload (struct insn_chain *chain, struct reload *rl, int j)
{
rtx note, old;
rtx_insn *insn = chain->insn;
rtx pseudo = rl->out_reg;
rtx reg_rtx = rl->reg_rtx;
if (rl->out && reg_rtx)
{
machine_mode mode;
mode = GET_MODE (rl->out);
if (mode == VOIDmode)
{
if (asm_noperands (PATTERN (insn)) < 0)
fatal_insn ("VOIDmode on an output", insn);
error_for_asm (insn, "output operand is constant in %<asm%>");
mode = word_mode;
rl->out = gen_rtx_REG (mode, REGNO (reg_rtx));
}
if (GET_MODE (reg_rtx) != mode)
reg_rtx = reload_adjust_reg_for_mode (reg_rtx, mode);
}
reload_reg_rtx_for_output[j] = reg_rtx;
if (pseudo
&& optimize
&& REG_P (pseudo)
&& ! rtx_equal_p (rl->in_reg, pseudo)
&& REGNO (pseudo) >= FIRST_PSEUDO_REGISTER
&& reg_last_reload_reg[REGNO (pseudo)])
{
int pseudo_no = REGNO (pseudo);
int last_regno = REGNO (reg_last_reload_reg[pseudo_no]);
if (TEST_HARD_REG_BIT (reg_reloaded_valid, last_regno)
&& reg_reloaded_contents[last_regno] == pseudo_no
&& spill_reg_store[last_regno]
&& rtx_equal_p (pseudo, spill_reg_stored_to[last_regno]))
delete_output_reload (insn, j, last_regno, reg_rtx);
}
old = rl->out_reg;
if (old == 0
|| reg_rtx == 0
|| rtx_equal_p (old, reg_rtx))
return;
if ((REG_P (old) || GET_CODE (old) == SCRATCH)
&& (note = find_reg_note (insn, REG_UNUSED, old)) != 0)
{
XEXP (note, 0) = reg_rtx;
return;
}
else if (GET_CODE (old) == SUBREG
&& REG_P (SUBREG_REG (old))
&& (note = find_reg_note (insn, REG_UNUSED,
SUBREG_REG (old))) != 0)
{
XEXP (note, 0) = gen_lowpart_common (GET_MODE (old), reg_rtx);
return;
}
else if (GET_CODE (old) == SCRATCH)
return;
gcc_assert (NONJUMP_INSN_P (insn));
emit_output_reload_insns (chain, rld + j, j);
}
static bool
inherit_piecemeal_p (int dest ATTRIBUTE_UNUSED,
int src ATTRIBUTE_UNUSED,
machine_mode mode ATTRIBUTE_UNUSED)
{
return (REG_CAN_CHANGE_MODE_P (dest, mode, reg_raw_mode[dest])
&& REG_CAN_CHANGE_MODE_P (src, mode, reg_raw_mode[src]));
}
static void
emit_reload_insns (struct insn_chain *chain)
{
rtx_insn *insn = chain->insn;
int j;
CLEAR_HARD_REG_SET (reg_reloaded_died);
for (j = 0; j < reload_n_operands; j++)
input_reload_insns[j] = input_address_reload_insns[j]
= inpaddr_address_reload_insns[j]
= output_reload_insns[j] = output_address_reload_insns[j]
= outaddr_address_reload_insns[j]
= other_output_reload_insns[j] = 0;
other_input_address_reload_insns = 0;
other_input_reload_insns = 0;
operand_reload_insns = 0;
other_operand_reload_insns = 0;
if (dump_file)
{
fprintf (dump_file, "\nReloads for insn # %d\n", INSN_UID (insn));
debug_reload_to_stream (dump_file);
}
for (j = 0; j < n_reloads; j++)
if (rld[j].reg_rtx && HARD_REGISTER_P (rld[j].reg_rtx))
{
unsigned int i;
for (i = REGNO (rld[j].reg_rtx); i < END_REGNO (rld[j].reg_rtx); i++)
new_spill_reg_store[i] = 0;
}
for (j = 0; j < n_reloads; j++)
{
do_input_reload (chain, rld + j, j);
do_output_reload (chain, rld + j, j);
}
emit_insn_before (other_input_address_reload_insns, insn);
emit_insn_before (other_input_reload_insns, insn);
for (j = 0; j < reload_n_operands; j++)
{
emit_insn_before (inpaddr_address_reload_insns[j], insn);
emit_insn_before (input_address_reload_insns[j], insn);
emit_insn_before (input_reload_insns[j], insn);
}
emit_insn_before (other_operand_reload_insns, insn);
emit_insn_before (operand_reload_insns, insn);
for (j = 0; j < reload_n_operands; j++)
{
rtx_insn *x = emit_insn_after (outaddr_address_reload_insns[j], insn);
x = emit_insn_after (output_address_reload_insns[j], x);
x = emit_insn_after (output_reload_insns[j], x);
emit_insn_after (other_output_reload_insns[j], x);
}
for (j = 0; j < n_reloads; j++)
{
int r = reload_order[j];
int i = reload_spill_index[r];
if (rld[r].in_reg != 0
&& ! (reload_inherited[r] || reload_override_in[r]))
{
rtx reg = rld[r].in_reg;
if (GET_CODE (reg) == SUBREG)
reg = SUBREG_REG (reg);
if (REG_P (reg)
&& REGNO (reg) >= FIRST_PSEUDO_REGISTER
&& !REGNO_REG_SET_P (&reg_has_output_reload, REGNO (reg)))
{
int nregno = REGNO (reg);
if (reg_last_reload_reg[nregno])
{
int last_regno = REGNO (reg_last_reload_reg[nregno]);
if (reg_reloaded_contents[last_regno] == nregno)
spill_reg_store[last_regno] = 0;
}
}
}
if (i >= 0 && rld[r].reg_rtx != 0)
{
int nr = hard_regno_nregs (i, GET_MODE (rld[r].reg_rtx));
int k;
for (k = 0; k < nr; k++)
if (reload_reg_reaches_end_p (i + k, r))
CLEAR_HARD_REG_BIT (reg_reloaded_valid, i + k);
if (rld[r].out != 0
&& (REG_P (rld[r].out)
|| (rld[r].out_reg
? REG_P (rld[r].out_reg)
: (GET_CODE (rld[r].out) != POST_MODIFY
&& GET_CODE (rld[r].out) != PRE_MODIFY))))
{
rtx reg;
reg = reload_reg_rtx_for_output[r];
if (reload_reg_rtx_reaches_end_p (reg, r))
{
machine_mode mode = GET_MODE (reg);
int regno = REGNO (reg);
int nregs = REG_NREGS (reg);
rtx out = (REG_P (rld[r].out)
? rld[r].out
: rld[r].out_reg
? rld[r].out_reg
: XEXP (rld[r].in_reg, 0));
int out_regno = REGNO (out);
int out_nregs = (!HARD_REGISTER_NUM_P (out_regno) ? 1
: hard_regno_nregs (out_regno, mode));
bool piecemeal;
spill_reg_store[regno] = new_spill_reg_store[regno];
spill_reg_stored_to[regno] = out;
reg_last_reload_reg[out_regno] = reg;
piecemeal = (HARD_REGISTER_NUM_P (out_regno)
&& nregs == out_nregs
&& inherit_piecemeal_p (out_regno, regno, mode));
if (HARD_REGISTER_NUM_P (out_regno))
for (k = 1; k < out_nregs; k++)
reg_last_reload_reg[out_regno + k]
= (piecemeal ? regno_reg_rtx[regno + k] : 0);
for (k = 0; k < nregs; k++)
{
CLEAR_HARD_REG_BIT (reg_reloaded_dead, regno + k);
reg_reloaded_contents[regno + k]
= (!HARD_REGISTER_NUM_P (out_regno) || !piecemeal
? out_regno
: out_regno + k);
reg_reloaded_insn[regno + k] = insn;
SET_HARD_REG_BIT (reg_reloaded_valid, regno + k);
if (targetm.hard_regno_call_part_clobbered (regno + k,
mode))
SET_HARD_REG_BIT (reg_reloaded_call_part_clobbered,
regno + k);
else
CLEAR_HARD_REG_BIT (reg_reloaded_call_part_clobbered,
regno + k);
}
}
}
else if (rld[r].out_reg == 0
&& rld[r].in != 0
&& ((REG_P (rld[r].in)
&& !HARD_REGISTER_P (rld[r].in)
&& !REGNO_REG_SET_P (&reg_has_output_reload,
REGNO (rld[r].in)))
|| (REG_P (rld[r].in_reg)
&& !REGNO_REG_SET_P (&reg_has_output_reload,
REGNO (rld[r].in_reg))))
&& !reg_set_p (reload_reg_rtx_for_input[r], PATTERN (insn)))
{
rtx reg;
reg = reload_reg_rtx_for_input[r];
if (reload_reg_rtx_reaches_end_p (reg, r))
{
machine_mode mode;
int regno;
int nregs;
int in_regno;
int in_nregs;
rtx in;
bool piecemeal;
mode = GET_MODE (reg);
regno = REGNO (reg);
nregs = REG_NREGS (reg);
if (REG_P (rld[r].in)
&& REGNO (rld[r].in) >= FIRST_PSEUDO_REGISTER)
in = rld[r].in;
else if (REG_P (rld[r].in_reg))
in = rld[r].in_reg;
else
in = XEXP (rld[r].in_reg, 0);
in_regno = REGNO (in);
in_nregs = (!HARD_REGISTER_NUM_P (in_regno) ? 1
: hard_regno_nregs (in_regno, mode));
reg_last_reload_reg[in_regno] = reg;
piecemeal = (HARD_REGISTER_NUM_P (in_regno)
&& nregs == in_nregs
&& inherit_piecemeal_p (regno, in_regno, mode));
if (HARD_REGISTER_NUM_P (in_regno))
for (k = 1; k < in_nregs; k++)
reg_last_reload_reg[in_regno + k]
= (piecemeal ? regno_reg_rtx[regno + k] : 0);
if (! reload_inherited[r]
|| (rld[r].out && ! rld[r].out_reg))
spill_reg_store[regno] = 0;
for (k = 0; k < nregs; k++)
{
CLEAR_HARD_REG_BIT (reg_reloaded_dead, regno + k);
reg_reloaded_contents[regno + k]
= (!HARD_REGISTER_NUM_P (in_regno) || !piecemeal
? in_regno
: in_regno + k);
reg_reloaded_insn[regno + k] = insn;
SET_HARD_REG_BIT (reg_reloaded_valid, regno + k);
if (targetm.hard_regno_call_part_clobbered (regno + k,
mode))
SET_HARD_REG_BIT (reg_reloaded_call_part_clobbered,
regno + k);
else
CLEAR_HARD_REG_BIT (reg_reloaded_call_part_clobbered,
regno + k);
}
}
}
}
if (i < 0
&& ((rld[r].out != 0
&& (REG_P (rld[r].out)
|| (MEM_P (rld[r].out)
&& REG_P (rld[r].out_reg))))
|| (rld[r].out == 0 && rld[r].out_reg
&& REG_P (rld[r].out_reg))))
{
rtx out = ((rld[r].out && REG_P (rld[r].out))
? rld[r].out : rld[r].out_reg);
int out_regno = REGNO (out);
machine_mode mode = GET_MODE (out);
if (rld[r].reg_rtx && rld[r].reg_rtx != out)
forget_old_reloads_1 (rld[r].reg_rtx, NULL_RTX, NULL);
if (!HARD_REGISTER_NUM_P (out_regno))
{
rtx src_reg;
rtx_insn *store_insn = NULL;
reg_last_reload_reg[out_regno] = 0;
src_reg = reload_reg_rtx_for_output[r];
if (src_reg)
{
if (reload_reg_rtx_reaches_end_p (src_reg, r))
store_insn = new_spill_reg_store[REGNO (src_reg)];
else
src_reg = NULL_RTX;
}
else
{
rtx set = single_set (insn);
if (set && SET_DEST (set) == rld[r].out)
{
int k;
src_reg = SET_SRC (set);
store_insn = insn;
for (k = 0; k < n_reloads; k++)
{
if (rld[k].in == src_reg)
{
src_reg = reload_reg_rtx_for_input[k];
break;
}
}
}
}
if (src_reg && REG_P (src_reg)
&& REGNO (src_reg) < FIRST_PSEUDO_REGISTER)
{
int src_regno, src_nregs, k;
rtx note;
gcc_assert (GET_MODE (src_reg) == mode);
src_regno = REGNO (src_reg);
src_nregs = hard_regno_nregs (src_regno, mode);
note = find_regno_note (insn, REG_DEAD, src_regno);
if (! note && store_insn)
note = find_regno_note (store_insn, REG_DEAD, src_regno);
for (k = 0; k < src_nregs; k++)
{
spill_reg_store[src_regno + k] = store_insn;
spill_reg_stored_to[src_regno + k] = out;
reg_reloaded_contents[src_regno + k] = out_regno;
reg_reloaded_insn[src_regno + k] = store_insn;
CLEAR_HARD_REG_BIT (reg_reloaded_dead, src_regno + k);
SET_HARD_REG_BIT (reg_reloaded_valid, src_regno + k);
if (targetm.hard_regno_call_part_clobbered
(src_regno + k, mode))
SET_HARD_REG_BIT (reg_reloaded_call_part_clobbered,
src_regno + k);
else
CLEAR_HARD_REG_BIT (reg_reloaded_call_part_clobbered,
src_regno + k);
SET_HARD_REG_BIT (reg_is_output_reload, src_regno + k);
if (note)
SET_HARD_REG_BIT (reg_reloaded_died, src_regno);
else
CLEAR_HARD_REG_BIT (reg_reloaded_died, src_regno);
}
reg_last_reload_reg[out_regno] = src_reg;
SET_REGNO_REG_SET (&reg_has_output_reload,
out_regno);
}
}
else
{
int k, out_nregs = hard_regno_nregs (out_regno, mode);
for (k = 0; k < out_nregs; k++)
reg_last_reload_reg[out_regno + k] = 0;
}
}
}
IOR_HARD_REG_SET (reg_reloaded_dead, reg_reloaded_died);
}

static rtx_insn *
emit_insn_if_valid_for_reload (rtx pat)
{
rtx_insn *last = get_last_insn ();
int code;
rtx_insn *insn = emit_insn (pat);
code = recog_memoized (insn);
if (code >= 0)
{
extract_insn (insn);
if (constrain_operands (1, get_enabled_alternatives (insn)))
return insn;
}
delete_insns_since (last);
return NULL;
}
static rtx_insn *
gen_reload (rtx out, rtx in, int opnum, enum reload_type type)
{
rtx_insn *last = get_last_insn ();
rtx_insn *tem;
rtx tem1, tem2;
if (!strip_paradoxical_subreg (&in, &out))
strip_paradoxical_subreg (&out, &in);
if (GET_CODE (in) == PLUS
&& (REG_P (XEXP (in, 0))
|| GET_CODE (XEXP (in, 0)) == SUBREG
|| MEM_P (XEXP (in, 0)))
&& (REG_P (XEXP (in, 1))
|| GET_CODE (XEXP (in, 1)) == SUBREG
|| CONSTANT_P (XEXP (in, 1))
|| MEM_P (XEXP (in, 1))))
{
rtx op0, op1, tem;
rtx_insn *insn;
enum insn_code code;
op0 = find_replacement (&XEXP (in, 0));
op1 = find_replacement (&XEXP (in, 1));
if (REG_P (XEXP (in, 1))
&& REGNO (out) == REGNO (XEXP (in, 1)))
tem = op0, op0 = op1, op1 = tem;
if (op0 != XEXP (in, 0) || op1 != XEXP (in, 1))
in = gen_rtx_PLUS (GET_MODE (in), op0, op1);
insn = emit_insn_if_valid_for_reload (gen_rtx_SET (out, in));
if (insn)
return insn;
code = optab_handler (add_optab, GET_MODE (out));
if (CONSTANT_P (op1) || MEM_P (op1) || GET_CODE (op1) == SUBREG
|| (REG_P (op1)
&& REGNO (op1) >= FIRST_PSEUDO_REGISTER)
|| (code != CODE_FOR_nothing
&& !insn_operand_matches (code, 2, op1)))
tem = op0, op0 = op1, op1 = tem;
gen_reload (out, op0, opnum, type);
if (rtx_equal_p (op0, op1))
op1 = out;
insn = emit_insn_if_valid_for_reload (gen_add2_insn (out, op1));
if (insn)
{
set_dst_reg_note (insn, REG_EQUIV, in, out);
return insn;
}
gcc_assert (!reg_overlap_mentioned_p (out, op0));
gen_reload (out, op1, opnum, type);
insn = emit_insn (gen_add2_insn (out, op0));
set_dst_reg_note (insn, REG_EQUIV, in, out);
}
else if ((tem1 = replaced_subreg (in), tem2 = replaced_subreg (out),
(REG_P (tem1) && REG_P (tem2)))
&& REGNO (tem1) < FIRST_PSEUDO_REGISTER
&& REGNO (tem2) < FIRST_PSEUDO_REGISTER
&& targetm.secondary_memory_needed (GET_MODE (out),
REGNO_REG_CLASS (REGNO (tem1)),
REGNO_REG_CLASS (REGNO (tem2))))
{
rtx loc = get_secondary_mem (in, GET_MODE (out), opnum, type);
if (GET_MODE (loc) != GET_MODE (out))
out = gen_rtx_REG (GET_MODE (loc), reg_or_subregno (out));
if (GET_MODE (loc) != GET_MODE (in))
in = gen_rtx_REG (GET_MODE (loc), reg_or_subregno (in));
gen_reload (loc, in, opnum, type);
gen_reload (out, loc, opnum, type);
}
else if (REG_P (out) && UNARY_P (in))
{
rtx op1;
rtx out_moded;
rtx_insn *set;
op1 = find_replacement (&XEXP (in, 0));
if (op1 != XEXP (in, 0))
in = gen_rtx_fmt_e (GET_CODE (in), GET_MODE (in), op1);
set = emit_insn_if_valid_for_reload (gen_rtx_SET (out, in));
if (set)
return set;
if (GET_MODE (op1) != GET_MODE (out))
out_moded = gen_rtx_REG (GET_MODE (op1), REGNO (out));
else
out_moded = out;
gen_reload (out_moded, op1, opnum, type);
rtx temp = gen_rtx_SET (out, gen_rtx_fmt_e (GET_CODE (in), GET_MODE (in),
out_moded));
rtx_insn *insn = emit_insn_if_valid_for_reload (temp);
if (insn)
{
set_unique_reg_note (insn, REG_EQUIV, in);
return insn;
}
fatal_insn ("failure trying to reload:", set);
}
else if (OBJECT_P (in) || GET_CODE (in) == SUBREG)
{
tem = emit_insn (gen_move_insn (out, in));
mark_jump_label (in, tem, 0);
}
else if (targetm.have_reload_load_address ())
emit_insn (targetm.gen_reload_load_address (out, in));
else
emit_insn (gen_rtx_SET (out, in));
return last ? NEXT_INSN (last) : get_insns ();
}

static void
delete_output_reload (rtx_insn *insn, int j, int last_reload_reg,
rtx new_reload_reg)
{
rtx_insn *output_reload_insn = spill_reg_store[last_reload_reg];
rtx reg = spill_reg_stored_to[last_reload_reg];
int k;
int n_occurrences;
int n_inherited = 0;
rtx substed;
unsigned regno;
int nregs;
if (output_reload_insn->deleted ())
return;
while (GET_CODE (reg) == SUBREG)
reg = SUBREG_REG (reg);
substed = reg_equiv_memory_loc (REGNO (reg));
for (k = n_reloads - 1; k >= 0; k--)
{
rtx reg2 = rld[k].in;
if (! reg2)
continue;
if (MEM_P (reg2) || reload_override_in[k])
reg2 = rld[k].in_reg;
if (AUTO_INC_DEC && rld[k].out && ! rld[k].out_reg)
reg2 = XEXP (rld[k].in_reg, 0);
while (GET_CODE (reg2) == SUBREG)
reg2 = SUBREG_REG (reg2);
if (rtx_equal_p (reg2, reg))
{
if (reload_inherited[k] || reload_override_in[k] || k == j)
n_inherited++;
else
return;
}
}
n_occurrences = count_occurrences (PATTERN (insn), reg, 0);
if (CALL_P (insn) && CALL_INSN_FUNCTION_USAGE (insn))
n_occurrences += count_occurrences (CALL_INSN_FUNCTION_USAGE (insn),
reg, 0);
if (substed)
n_occurrences += count_occurrences (PATTERN (insn),
eliminate_regs (substed, VOIDmode,
NULL_RTX), 0);
for (rtx i1 = reg_equiv_alt_mem_list (REGNO (reg)); i1; i1 = XEXP (i1, 1))
{
gcc_assert (!rtx_equal_p (XEXP (i1, 0), substed));
n_occurrences += count_occurrences (PATTERN (insn), XEXP (i1, 0), 0);
}
if (n_occurrences > n_inherited)
return;
regno = REGNO (reg);
nregs = REG_NREGS (reg);
for (rtx_insn *i1 = NEXT_INSN (output_reload_insn);
i1 != insn; i1 = NEXT_INSN (i1))
{
if (NOTE_INSN_BASIC_BLOCK_P (i1))
return;
if ((NONJUMP_INSN_P (i1) || CALL_P (i1))
&& refers_to_regno_p (regno, regno + nregs, PATTERN (i1), NULL))
{
while (NONJUMP_INSN_P (i1) && GET_CODE (PATTERN (i1)) == USE)
{
n_occurrences += rtx_equal_p (reg, XEXP (PATTERN (i1), 0)) != 0;
i1 = NEXT_INSN (i1);
}
if (n_occurrences <= n_inherited && i1 == insn)
break;
return;
}
}
for (k = hard_regno_nregs (last_reload_reg, GET_MODE (reg)); k-- > 0; )
{
spill_reg_store[last_reload_reg + k] = 0;
spill_reg_stored_to[last_reload_reg + k] = 0;
}
if (rld[j].out != rld[j].in
&& REG_N_DEATHS (REGNO (reg)) == 1
&& REG_N_SETS (REGNO (reg)) == 1
&& REG_BASIC_BLOCK (REGNO (reg)) >= NUM_FIXED_BLOCKS
&& find_regno_note (insn, REG_DEAD, REGNO (reg)))
{
rtx_insn *i2;
for (i2 = PREV_INSN (insn); i2; i2 = PREV_INSN (i2))
{
rtx set = single_set (i2);
if (set != 0 && SET_DEST (set) == reg)
continue;
if (LABEL_P (i2) || JUMP_P (i2))
break;
if ((NONJUMP_INSN_P (i2) || CALL_P (i2))
&& reg_mentioned_p (reg, PATTERN (i2)))
{
delete_address_reloads (output_reload_insn, insn);
delete_insn (output_reload_insn);
return;
}
}
for (i2 = PREV_INSN (insn); i2; i2 = PREV_INSN (i2))
{
rtx set = single_set (i2);
if (set != 0 && SET_DEST (set) == reg)
{
delete_address_reloads (i2, insn);
delete_insn (i2);
}
if (LABEL_P (i2) || JUMP_P (i2))
break;
}
reg_renumber[REGNO (reg)] = REGNO (new_reload_reg);
if (ira_conflicts_p)
ira_mark_allocation_change (REGNO (reg));
alter_reg (REGNO (reg), -1, false);
}
else
{
delete_address_reloads (output_reload_insn, insn);
delete_insn (output_reload_insn);
}
}
static void
delete_address_reloads (rtx_insn *dead_insn, rtx_insn *current_insn)
{
rtx set = single_set (dead_insn);
rtx set2, dst;
rtx_insn *prev, *next;
if (set)
{
rtx dst = SET_DEST (set);
if (MEM_P (dst))
delete_address_reloads_1 (dead_insn, XEXP (dst, 0), current_insn);
}
prev = PREV_INSN (dead_insn);
next = NEXT_INSN (dead_insn);
if (! prev || ! next)
return;
set = single_set (next);
set2 = single_set (prev);
if (! set || ! set2
|| GET_CODE (SET_SRC (set)) != PLUS || GET_CODE (SET_SRC (set2)) != PLUS
|| !CONST_INT_P (XEXP (SET_SRC (set), 1))
|| !CONST_INT_P (XEXP (SET_SRC (set2), 1)))
return;
dst = SET_DEST (set);
if (! rtx_equal_p (dst, SET_DEST (set2))
|| ! rtx_equal_p (dst, XEXP (SET_SRC (set), 0))
|| ! rtx_equal_p (dst, XEXP (SET_SRC (set2), 0))
|| (INTVAL (XEXP (SET_SRC (set), 1))
!= -INTVAL (XEXP (SET_SRC (set2), 1))))
return;
delete_related_insns (prev);
delete_related_insns (next);
}
static void
delete_address_reloads_1 (rtx_insn *dead_insn, rtx x, rtx_insn *current_insn)
{
rtx_insn *prev, *i2;
rtx set, dst;
int i, j;
enum rtx_code code = GET_CODE (x);
if (code != REG)
{
const char *fmt = GET_RTX_FORMAT (code);
for (i = GET_RTX_LENGTH (code) - 1; i >= 0; i--)
{
if (fmt[i] == 'e')
delete_address_reloads_1 (dead_insn, XEXP (x, i), current_insn);
else if (fmt[i] == 'E')
{
for (j = XVECLEN (x, i) - 1; j >= 0; j--)
delete_address_reloads_1 (dead_insn, XVECEXP (x, i, j),
current_insn);
}
}
return;
}
if (spill_reg_order[REGNO (x)] < 0)
return;
for (prev = PREV_INSN (dead_insn); prev; prev = PREV_INSN (prev))
{
code = GET_CODE (prev);
if (code == CODE_LABEL || code == JUMP_INSN)
return;
if (!INSN_P (prev))
continue;
if (reg_set_p (x, PATTERN (prev)))
break;
if (reg_referenced_p (x, PATTERN (prev)))
return;
}
if (! prev || INSN_UID (prev) < reload_first_uid)
return;
set = single_set (prev);
if (! set)
return;
dst = SET_DEST (set);
if (!REG_P (dst)
|| ! rtx_equal_p (dst, x))
return;
if (! reg_set_p (dst, PATTERN (dead_insn)))
{
for (i2 = NEXT_INSN (dead_insn); i2; i2 = NEXT_INSN (i2))
{
if (LABEL_P (i2))
break;
if (! INSN_P (i2))
continue;
if (reg_referenced_p (dst, PATTERN (i2)))
{
if (i2 == current_insn)
{
for (j = n_reloads - 1; j >= 0; j--)
if ((rld[j].reg_rtx == dst && reload_inherited[j])
|| reload_override_in[j] == dst)
return;
for (j = n_reloads - 1; j >= 0; j--)
if (rld[j].in && rld[j].reg_rtx == dst)
break;
if (j >= 0)
break;
}
return;
}
if (JUMP_P (i2))
break;
if (i2 == current_insn)
{
for (j = n_reloads - 1; j >= 0; j--)
if ((rld[j].reg_rtx == dst && reload_inherited[j])
|| reload_override_in[j] == dst)
return;
}
if (reg_set_p (dst, PATTERN (i2)))
break;
}
}
delete_address_reloads_1 (prev, SET_SRC (set), current_insn);
reg_reloaded_contents[REGNO (dst)] = -1;
delete_insn (prev);
}

static void
inc_for_reload (rtx reloadreg, rtx in, rtx value, poly_int64 inc_amount)
{
rtx incloc = find_replacement (&XEXP (value, 0));
int post = (GET_CODE (value) == POST_DEC || GET_CODE (value) == POST_INC
|| GET_CODE (value) == POST_MODIFY);
rtx_insn *last;
rtx inc;
rtx_insn *add_insn;
int code;
rtx real_in = in == value ? incloc : in;
if (REG_P (incloc))
reg_last_reload_reg[REGNO (incloc)] = 0;
if (GET_CODE (value) == PRE_MODIFY || GET_CODE (value) == POST_MODIFY)
{
gcc_assert (GET_CODE (XEXP (value, 1)) == PLUS);
inc = find_replacement (&XEXP (XEXP (value, 1), 1));
}
else
{
if (GET_CODE (value) == PRE_DEC || GET_CODE (value) == POST_DEC)
inc_amount = -inc_amount;
inc = gen_int_mode (inc_amount, Pmode);
}
if (post && real_in != reloadreg)
emit_insn (gen_move_insn (reloadreg, real_in));
if (in == value)
{
last = get_last_insn ();
add_insn = emit_insn (gen_rtx_SET (incloc,
gen_rtx_PLUS (GET_MODE (incloc),
incloc, inc)));
code = recog_memoized (add_insn);
if (code >= 0)
{
extract_insn (add_insn);
if (constrain_operands (1, get_enabled_alternatives (add_insn)))
{
if (! post)
emit_insn (gen_move_insn (reloadreg, incloc));
return;
}
}
delete_insns_since (last);
}
if (! post)
{
if (in != reloadreg)
emit_insn (gen_move_insn (reloadreg, real_in));
emit_insn (gen_add2_insn (reloadreg, inc));
emit_insn (gen_move_insn (incloc, reloadreg));
}
else
{
emit_insn (gen_add2_insn (reloadreg, inc));
emit_insn (gen_move_insn (incloc, reloadreg));
if (CONST_INT_P (inc))
emit_insn (gen_add2_insn (reloadreg,
gen_int_mode (-INTVAL (inc),
GET_MODE (reloadreg))));
else
emit_insn (gen_sub2_insn (reloadreg, inc));
}
}

static void
add_auto_inc_notes (rtx_insn *insn, rtx x)
{
enum rtx_code code = GET_CODE (x);
const char *fmt;
int i, j;
if (code == MEM && auto_inc_p (XEXP (x, 0)))
{
add_reg_note (insn, REG_INC, XEXP (XEXP (x, 0), 0));
return;
}
fmt = GET_RTX_FORMAT (code);
for (i = GET_RTX_LENGTH (code) - 1; i >= 0; i--)
{
if (fmt[i] == 'e')
add_auto_inc_notes (insn, XEXP (x, i));
else if (fmt[i] == 'E')
for (j = XVECLEN (x, i) - 1; j >= 0; j--)
add_auto_inc_notes (insn, XVECEXP (x, i, j));
}
}

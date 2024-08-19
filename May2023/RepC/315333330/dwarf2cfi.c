#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "target.h"
#include "function.h"
#include "rtl.h"
#include "tree.h"
#include "tree-pass.h"
#include "memmodel.h"
#include "tm_p.h"
#include "emit-rtl.h"
#include "stor-layout.h"
#include "cfgbuild.h"
#include "dwarf2out.h"
#include "dwarf2asm.h"
#include "common/common-target.h"
#include "except.h"		
#include "profile-count.h"	
#include "expr.h"		
#include "output.h"		
#include "debug.h"		
#undef DWARF2_UNWIND_INFO
#undef DWARF2_FRAME_INFO
#if (GCC_VERSION >= 3000)
#pragma GCC poison DWARF2_UNWIND_INFO DWARF2_FRAME_INFO
#endif
#ifndef INCOMING_RETURN_ADDR_RTX
#define INCOMING_RETURN_ADDR_RTX  (gcc_unreachable (), NULL_RTX)
#endif
#ifndef DEFAULT_INCOMING_FRAME_SP_OFFSET
#define DEFAULT_INCOMING_FRAME_SP_OFFSET INCOMING_FRAME_SP_OFFSET
#endif

struct GTY(()) dw_cfi_row
{
dw_cfa_location cfa;
dw_cfi_ref cfa_cfi;
cfi_vec reg_save;
};
struct GTY(()) reg_saved_in_data {
rtx orig_reg;
rtx saved_in_reg;
};
struct dw_trace_info
{
rtx_insn *head;
dw_cfi_row *beg_row, *end_row;
poly_int64_pod beg_true_args_size, end_true_args_size;
poly_int64_pod beg_delay_args_size, end_delay_args_size;
rtx_insn *eh_head;
dw_cfa_location cfa_store;
dw_cfa_location cfa_temp;
vec<reg_saved_in_data> regs_saved_in_regs;
unsigned id;
bool switch_sections;
bool args_size_undefined;
};
struct trace_info_hasher : nofree_ptr_hash <dw_trace_info>
{
static inline hashval_t hash (const dw_trace_info *);
static inline bool equal (const dw_trace_info *, const dw_trace_info *);
};
inline hashval_t
trace_info_hasher::hash (const dw_trace_info *ti)
{
return INSN_UID (ti->head);
}
inline bool
trace_info_hasher::equal (const dw_trace_info *a, const dw_trace_info *b)
{
return a->head == b->head;
}
static vec<dw_trace_info> trace_info;
static vec<dw_trace_info *> trace_work_list;
static hash_table<trace_info_hasher> *trace_index;
cfi_vec cie_cfi_vec;
static GTY(()) dw_cfi_row *cie_cfi_row;
static GTY(()) reg_saved_in_data *cie_return_save;
static GTY(()) unsigned long dwarf2out_cfi_label_num;
static rtx_insn *add_cfi_insn;
static cfi_vec *add_cfi_vec;
static dw_trace_info *cur_trace;
static dw_cfi_row *cur_row;
static dw_cfa_location *cur_cfa;
struct queued_reg_save {
rtx reg;
rtx saved_reg;
poly_int64_pod cfa_offset;
};
static vec<queued_reg_save> queued_reg_saves;
static bool any_cfis_emitted;
static unsigned dw_stack_pointer_regnum;
static unsigned dw_frame_pointer_regnum;

rtx
expand_builtin_dwarf_sp_column (void)
{
unsigned int dwarf_regnum = DWARF_FRAME_REGNUM (STACK_POINTER_REGNUM);
return GEN_INT (DWARF2_FRAME_REG_OUT (dwarf_regnum, 1));
}
static void
init_return_column_size (scalar_int_mode mode, rtx mem, unsigned int c)
{
HOST_WIDE_INT offset = c * GET_MODE_SIZE (mode);
HOST_WIDE_INT size = GET_MODE_SIZE (Pmode);
emit_move_insn (adjust_address (mem, mode, offset),
gen_int_mode (size, mode));
}
struct init_one_dwarf_reg_state
{
bool wrote_return_column;
bool processed_regno [FIRST_PSEUDO_REGISTER];
};
static
void init_one_dwarf_reg_size (int regno, machine_mode regmode,
rtx table, machine_mode slotmode,
init_one_dwarf_reg_state *init_state)
{
const unsigned int dnum = DWARF_FRAME_REGNUM (regno);
const unsigned int rnum = DWARF2_FRAME_REG_OUT (dnum, 1);
const unsigned int dcol = DWARF_REG_TO_UNWIND_COLUMN (rnum);
poly_int64 slotoffset = dcol * GET_MODE_SIZE (slotmode);
poly_int64 regsize = GET_MODE_SIZE (regmode);
init_state->processed_regno[regno] = true;
if (rnum >= DWARF_FRAME_REGISTERS)
return;
if (dnum == DWARF_FRAME_RETURN_COLUMN)
{
if (regmode == VOIDmode)
return;
init_state->wrote_return_column = true;
}
if (maybe_lt (slotoffset, 0))
return;
emit_move_insn (adjust_address (table, slotmode, slotoffset),
gen_int_mode (regsize, slotmode));
}
void
expand_builtin_init_dwarf_reg_sizes (tree address)
{
unsigned int i;
scalar_int_mode mode = SCALAR_INT_TYPE_MODE (char_type_node);
rtx addr = expand_normal (address);
rtx mem = gen_rtx_MEM (BLKmode, addr);
init_one_dwarf_reg_state init_state;
memset ((char *)&init_state, 0, sizeof (init_state));
for (i = 0; i < FIRST_PSEUDO_REGISTER; i++)
{
machine_mode save_mode;
rtx span;
if (init_state.processed_regno[i])
continue;
save_mode = targetm.dwarf_frame_reg_mode (i);
span = targetm.dwarf_register_span (gen_rtx_REG (save_mode, i));
if (!span)
init_one_dwarf_reg_size (i, save_mode, mem, mode, &init_state);
else
{
for (int si = 0; si < XVECLEN (span, 0); si++)
{
rtx reg = XVECEXP (span, 0, si);
init_one_dwarf_reg_size
(REGNO (reg), GET_MODE (reg), mem, mode, &init_state);
}
}
}
if (!init_state.wrote_return_column)
init_return_column_size (mode, mem, DWARF_FRAME_RETURN_COLUMN);
#ifdef DWARF_ALT_FRAME_RETURN_COLUMN
init_return_column_size (mode, mem, DWARF_ALT_FRAME_RETURN_COLUMN);
#endif
targetm.init_dwarf_reg_sizes_extra (address);
}

static dw_trace_info *
get_trace_info (rtx_insn *insn)
{
dw_trace_info dummy;
dummy.head = insn;
return trace_index->find_with_hash (&dummy, INSN_UID (insn));
}
static bool
save_point_p (rtx_insn *insn)
{
if (LABEL_P (insn))
return inside_basic_block_p (insn);
if (NOTE_P (insn))
switch (NOTE_KIND (insn))
{
case NOTE_INSN_PROLOGUE_END:
case NOTE_INSN_EPILOGUE_BEG:
return true;
}
return false;
}
static inline HOST_WIDE_INT
div_data_align (HOST_WIDE_INT off)
{
HOST_WIDE_INT r = off / DWARF_CIE_DATA_ALIGNMENT;
gcc_assert (r * DWARF_CIE_DATA_ALIGNMENT == off);
return r;
}
static inline bool
need_data_align_sf_opcode (HOST_WIDE_INT off)
{
return DWARF_CIE_DATA_ALIGNMENT < 0 ? off > 0 : off < 0;
}
static inline dw_cfi_ref
new_cfi (void)
{
dw_cfi_ref cfi = ggc_alloc<dw_cfi_node> ();
cfi->dw_cfi_oprnd1.dw_cfi_reg_num = 0;
cfi->dw_cfi_oprnd2.dw_cfi_reg_num = 0;
return cfi;
}
static dw_cfi_row *
new_cfi_row (void)
{
dw_cfi_row *row = ggc_cleared_alloc<dw_cfi_row> ();
row->cfa.reg = INVALID_REGNUM;
return row;
}
static dw_cfi_row *
copy_cfi_row (dw_cfi_row *src)
{
dw_cfi_row *dst = ggc_alloc<dw_cfi_row> ();
*dst = *src;
dst->reg_save = vec_safe_copy (src->reg_save);
return dst;
}
static dw_cfa_location *
copy_cfa (dw_cfa_location *src)
{
dw_cfa_location *dst = ggc_alloc<dw_cfa_location> ();
*dst = *src;
return dst;
}
static char *
dwarf2out_cfi_label (void)
{
int num = dwarf2out_cfi_label_num++;
char label[20];
ASM_GENERATE_INTERNAL_LABEL (label, "LCFI", num);
return xstrdup (label);
}
static void
add_cfi (dw_cfi_ref cfi)
{
any_cfis_emitted = true;
if (add_cfi_insn != NULL)
{
add_cfi_insn = emit_note_after (NOTE_INSN_CFI, add_cfi_insn);
NOTE_CFI (add_cfi_insn) = cfi;
}
if (add_cfi_vec != NULL)
vec_safe_push (*add_cfi_vec, cfi);
}
static void
add_cfi_args_size (poly_int64 size)
{
HOST_WIDE_INT const_size = size.to_constant ();
dw_cfi_ref cfi = new_cfi ();
gcc_assert (const_size >= 0);
cfi->dw_cfi_opc = DW_CFA_GNU_args_size;
cfi->dw_cfi_oprnd1.dw_cfi_offset = const_size;
add_cfi (cfi);
}
static void
add_cfi_restore (unsigned reg)
{
dw_cfi_ref cfi = new_cfi ();
cfi->dw_cfi_opc = (reg & ~0x3f ? DW_CFA_restore_extended : DW_CFA_restore);
cfi->dw_cfi_oprnd1.dw_cfi_reg_num = reg;
add_cfi (cfi);
}
static void
update_row_reg_save (dw_cfi_row *row, unsigned column, dw_cfi_ref cfi)
{
if (vec_safe_length (row->reg_save) <= column)
vec_safe_grow_cleared (row->reg_save, column + 1);
(*row->reg_save)[column] = cfi;
}
static void
get_cfa_from_loc_descr (dw_cfa_location *cfa, struct dw_loc_descr_node *loc)
{
struct dw_loc_descr_node *ptr;
cfa->offset = 0;
cfa->base_offset = 0;
cfa->indirect = 0;
cfa->reg = -1;
for (ptr = loc; ptr != NULL; ptr = ptr->dw_loc_next)
{
enum dwarf_location_atom op = ptr->dw_loc_opc;
switch (op)
{
case DW_OP_reg0:
case DW_OP_reg1:
case DW_OP_reg2:
case DW_OP_reg3:
case DW_OP_reg4:
case DW_OP_reg5:
case DW_OP_reg6:
case DW_OP_reg7:
case DW_OP_reg8:
case DW_OP_reg9:
case DW_OP_reg10:
case DW_OP_reg11:
case DW_OP_reg12:
case DW_OP_reg13:
case DW_OP_reg14:
case DW_OP_reg15:
case DW_OP_reg16:
case DW_OP_reg17:
case DW_OP_reg18:
case DW_OP_reg19:
case DW_OP_reg20:
case DW_OP_reg21:
case DW_OP_reg22:
case DW_OP_reg23:
case DW_OP_reg24:
case DW_OP_reg25:
case DW_OP_reg26:
case DW_OP_reg27:
case DW_OP_reg28:
case DW_OP_reg29:
case DW_OP_reg30:
case DW_OP_reg31:
cfa->reg = op - DW_OP_reg0;
break;
case DW_OP_regx:
cfa->reg = ptr->dw_loc_oprnd1.v.val_int;
break;
case DW_OP_breg0:
case DW_OP_breg1:
case DW_OP_breg2:
case DW_OP_breg3:
case DW_OP_breg4:
case DW_OP_breg5:
case DW_OP_breg6:
case DW_OP_breg7:
case DW_OP_breg8:
case DW_OP_breg9:
case DW_OP_breg10:
case DW_OP_breg11:
case DW_OP_breg12:
case DW_OP_breg13:
case DW_OP_breg14:
case DW_OP_breg15:
case DW_OP_breg16:
case DW_OP_breg17:
case DW_OP_breg18:
case DW_OP_breg19:
case DW_OP_breg20:
case DW_OP_breg21:
case DW_OP_breg22:
case DW_OP_breg23:
case DW_OP_breg24:
case DW_OP_breg25:
case DW_OP_breg26:
case DW_OP_breg27:
case DW_OP_breg28:
case DW_OP_breg29:
case DW_OP_breg30:
case DW_OP_breg31:
cfa->reg = op - DW_OP_breg0;
cfa->base_offset = ptr->dw_loc_oprnd1.v.val_int;
break;
case DW_OP_bregx:
cfa->reg = ptr->dw_loc_oprnd1.v.val_int;
cfa->base_offset = ptr->dw_loc_oprnd2.v.val_int;
break;
case DW_OP_deref:
cfa->indirect = 1;
break;
case DW_OP_plus_uconst:
cfa->offset = ptr->dw_loc_oprnd1.v.val_unsigned;
break;
default:
gcc_unreachable ();
}
}
}
void
lookup_cfa_1 (dw_cfi_ref cfi, dw_cfa_location *loc, dw_cfa_location *remember)
{
switch (cfi->dw_cfi_opc)
{
case DW_CFA_def_cfa_offset:
case DW_CFA_def_cfa_offset_sf:
loc->offset = cfi->dw_cfi_oprnd1.dw_cfi_offset;
break;
case DW_CFA_def_cfa_register:
loc->reg = cfi->dw_cfi_oprnd1.dw_cfi_reg_num;
break;
case DW_CFA_def_cfa:
case DW_CFA_def_cfa_sf:
loc->reg = cfi->dw_cfi_oprnd1.dw_cfi_reg_num;
loc->offset = cfi->dw_cfi_oprnd2.dw_cfi_offset;
break;
case DW_CFA_def_cfa_expression:
if (cfi->dw_cfi_oprnd2.dw_cfi_cfa_loc)
*loc = *cfi->dw_cfi_oprnd2.dw_cfi_cfa_loc;
else
get_cfa_from_loc_descr (loc, cfi->dw_cfi_oprnd1.dw_cfi_loc);
break;
case DW_CFA_remember_state:
gcc_assert (!remember->in_use);
*remember = *loc;
remember->in_use = 1;
break;
case DW_CFA_restore_state:
gcc_assert (remember->in_use);
*loc = *remember;
remember->in_use = 0;
break;
default:
break;
}
}
bool
cfa_equal_p (const dw_cfa_location *loc1, const dw_cfa_location *loc2)
{
return (loc1->reg == loc2->reg
&& known_eq (loc1->offset, loc2->offset)
&& loc1->indirect == loc2->indirect
&& (loc1->indirect == 0
|| known_eq (loc1->base_offset, loc2->base_offset)));
}
static bool
cfi_oprnd_equal_p (enum dw_cfi_oprnd_type t, dw_cfi_oprnd *a, dw_cfi_oprnd *b)
{
switch (t)
{
case dw_cfi_oprnd_unused:
return true;
case dw_cfi_oprnd_reg_num:
return a->dw_cfi_reg_num == b->dw_cfi_reg_num;
case dw_cfi_oprnd_offset:
return a->dw_cfi_offset == b->dw_cfi_offset;
case dw_cfi_oprnd_addr:
return (a->dw_cfi_addr == b->dw_cfi_addr
|| strcmp (a->dw_cfi_addr, b->dw_cfi_addr) == 0);
case dw_cfi_oprnd_loc:
return loc_descr_equal_p (a->dw_cfi_loc, b->dw_cfi_loc);
case dw_cfi_oprnd_cfa_loc:
return cfa_equal_p (a->dw_cfi_cfa_loc, b->dw_cfi_cfa_loc);
}
gcc_unreachable ();
}
static bool
cfi_equal_p (dw_cfi_ref a, dw_cfi_ref b)
{
enum dwarf_call_frame_info opc;
if (a == b)
return true;
if (a == NULL || b == NULL)
return false;
opc = a->dw_cfi_opc;
if (opc != b->dw_cfi_opc)
return false;
return (cfi_oprnd_equal_p (dw_cfi_oprnd1_desc (opc),
&a->dw_cfi_oprnd1, &b->dw_cfi_oprnd1)
&& cfi_oprnd_equal_p (dw_cfi_oprnd2_desc (opc),
&a->dw_cfi_oprnd2, &b->dw_cfi_oprnd2));
}
static bool
cfi_row_equal_p (dw_cfi_row *a, dw_cfi_row *b)
{
size_t i, n_a, n_b, n_max;
if (a->cfa_cfi)
{
if (!cfi_equal_p (a->cfa_cfi, b->cfa_cfi))
return false;
}
else if (!cfa_equal_p (&a->cfa, &b->cfa))
return false;
n_a = vec_safe_length (a->reg_save);
n_b = vec_safe_length (b->reg_save);
n_max = MAX (n_a, n_b);
for (i = 0; i < n_max; ++i)
{
dw_cfi_ref r_a = NULL, r_b = NULL;
if (i < n_a)
r_a = (*a->reg_save)[i];
if (i < n_b)
r_b = (*b->reg_save)[i];
if (!cfi_equal_p (r_a, r_b))
return false;
}
return true;
}
static dw_cfi_ref
def_cfa_0 (dw_cfa_location *old_cfa, dw_cfa_location *new_cfa)
{
dw_cfi_ref cfi;
if (cfa_equal_p (old_cfa, new_cfa))
return NULL;
cfi = new_cfi ();
HOST_WIDE_INT const_offset;
if (new_cfa->reg == old_cfa->reg
&& !new_cfa->indirect
&& !old_cfa->indirect
&& new_cfa->offset.is_constant (&const_offset))
{
if (const_offset < 0)
cfi->dw_cfi_opc = DW_CFA_def_cfa_offset_sf;
else
cfi->dw_cfi_opc = DW_CFA_def_cfa_offset;
cfi->dw_cfi_oprnd1.dw_cfi_offset = const_offset;
}
else if (new_cfa->offset.is_constant ()
&& known_eq (new_cfa->offset, old_cfa->offset)
&& old_cfa->reg != INVALID_REGNUM
&& !new_cfa->indirect
&& !old_cfa->indirect)
{
cfi->dw_cfi_opc = DW_CFA_def_cfa_register;
cfi->dw_cfi_oprnd1.dw_cfi_reg_num = new_cfa->reg;
}
else if (new_cfa->indirect == 0
&& new_cfa->offset.is_constant (&const_offset))
{
if (const_offset < 0)
cfi->dw_cfi_opc = DW_CFA_def_cfa_sf;
else
cfi->dw_cfi_opc = DW_CFA_def_cfa;
cfi->dw_cfi_oprnd1.dw_cfi_reg_num = new_cfa->reg;
cfi->dw_cfi_oprnd2.dw_cfi_offset = const_offset;
}
else
{
struct dw_loc_descr_node *loc_list;
cfi->dw_cfi_opc = DW_CFA_def_cfa_expression;
loc_list = build_cfa_loc (new_cfa, 0);
cfi->dw_cfi_oprnd1.dw_cfi_loc = loc_list;
if (!new_cfa->offset.is_constant ()
|| !new_cfa->base_offset.is_constant ())
cfi->dw_cfi_oprnd2.dw_cfi_cfa_loc = copy_cfa (new_cfa);
else
cfi->dw_cfi_oprnd2.dw_cfi_cfa_loc = NULL;
}
return cfi;
}
static void
def_cfa_1 (dw_cfa_location *new_cfa)
{
dw_cfi_ref cfi;
if (cur_trace->cfa_store.reg == new_cfa->reg && new_cfa->indirect == 0)
cur_trace->cfa_store.offset = new_cfa->offset;
cfi = def_cfa_0 (&cur_row->cfa, new_cfa);
if (cfi)
{
cur_row->cfa = *new_cfa;
cur_row->cfa_cfi = (cfi->dw_cfi_opc == DW_CFA_def_cfa_expression
? cfi : NULL);
add_cfi (cfi);
}
}
static void
reg_save (unsigned int reg, unsigned int sreg, poly_int64 offset)
{
dw_fde_ref fde = cfun ? cfun->fde : NULL;
dw_cfi_ref cfi = new_cfi ();
cfi->dw_cfi_oprnd1.dw_cfi_reg_num = reg;
if (sreg == INVALID_REGNUM)
{
HOST_WIDE_INT const_offset;
if (fde && fde->stack_realign)
{
cfi->dw_cfi_opc = DW_CFA_expression;
cfi->dw_cfi_oprnd1.dw_cfi_reg_num = reg;
cfi->dw_cfi_oprnd2.dw_cfi_loc
= build_cfa_aligned_loc (&cur_row->cfa, offset,
fde->stack_realignment);
}
else if (offset.is_constant (&const_offset))
{
if (need_data_align_sf_opcode (const_offset))
cfi->dw_cfi_opc = DW_CFA_offset_extended_sf;
else if (reg & ~0x3f)
cfi->dw_cfi_opc = DW_CFA_offset_extended;
else
cfi->dw_cfi_opc = DW_CFA_offset;
cfi->dw_cfi_oprnd2.dw_cfi_offset = const_offset;
}
else
{
cfi->dw_cfi_opc = DW_CFA_expression;
cfi->dw_cfi_oprnd1.dw_cfi_reg_num = reg;
cfi->dw_cfi_oprnd2.dw_cfi_loc
= build_cfa_loc (&cur_row->cfa, offset);
}
}
else if (sreg == reg)
{
gcc_unreachable ();
}
else
{
cfi->dw_cfi_opc = DW_CFA_register;
cfi->dw_cfi_oprnd2.dw_cfi_reg_num = sreg;
}
add_cfi (cfi);
update_row_reg_save (cur_row, reg, cfi);
}
static void
notice_args_size (rtx_insn *insn)
{
poly_int64 args_size, delta;
rtx note;
note = find_reg_note (insn, REG_ARGS_SIZE, NULL);
if (note == NULL)
return;
args_size = get_args_size (note);
delta = args_size - cur_trace->end_true_args_size;
if (known_eq (delta, 0))
return;
cur_trace->end_true_args_size = args_size;
if (cur_cfa->reg == dw_stack_pointer_regnum)
{
gcc_assert (!cur_cfa->indirect);
if (!STACK_GROWS_DOWNWARD)
delta = -delta;
cur_cfa->offset += delta;
}
}
static void
notice_eh_throw (rtx_insn *insn)
{
poly_int64 args_size = cur_trace->end_true_args_size;
if (cur_trace->eh_head == NULL)
{
cur_trace->eh_head = insn;
cur_trace->beg_delay_args_size = args_size;
cur_trace->end_delay_args_size = args_size;
}
else if (maybe_ne (cur_trace->end_delay_args_size, args_size))
{
cur_trace->end_delay_args_size = args_size;
add_cfi_args_size (args_size);
}
}
static inline unsigned
dwf_regno (const_rtx reg)
{
gcc_assert (REGNO (reg) < FIRST_PSEUDO_REGISTER);
return DWARF_FRAME_REGNUM (REGNO (reg));
}
static bool
compare_reg_or_pc (rtx x, rtx y)
{
if (REG_P (x) && REG_P (y))
return REGNO (x) == REGNO (y);
return x == y;
}
static void
record_reg_saved_in_reg (rtx dest, rtx src)
{
reg_saved_in_data *elt;
size_t i;
FOR_EACH_VEC_ELT (cur_trace->regs_saved_in_regs, i, elt)
if (compare_reg_or_pc (elt->orig_reg, src))
{
if (dest == NULL)
cur_trace->regs_saved_in_regs.unordered_remove (i);
else
elt->saved_in_reg = dest;
return;
}
if (dest == NULL)
return;
reg_saved_in_data e = {src, dest};
cur_trace->regs_saved_in_regs.safe_push (e);
}
static void
queue_reg_save (rtx reg, rtx sreg, poly_int64 offset)
{
queued_reg_save *q;
queued_reg_save e = {reg, sreg, offset};
size_t i;
FOR_EACH_VEC_ELT (queued_reg_saves, i, q)
if (compare_reg_or_pc (q->reg, reg))
{
*q = e;
return;
}
queued_reg_saves.safe_push (e);
}
static void
dwarf2out_flush_queued_reg_saves (void)
{
queued_reg_save *q;
size_t i;
FOR_EACH_VEC_ELT (queued_reg_saves, i, q)
{
unsigned int reg, sreg;
record_reg_saved_in_reg (q->saved_reg, q->reg);
if (q->reg == pc_rtx)
reg = DWARF_FRAME_RETURN_COLUMN;
else
reg = dwf_regno (q->reg);
if (q->saved_reg)
sreg = dwf_regno (q->saved_reg);
else
sreg = INVALID_REGNUM;
reg_save (reg, sreg, q->cfa_offset);
}
queued_reg_saves.truncate (0);
}
static bool
clobbers_queued_reg_save (const_rtx insn)
{
queued_reg_save *q;
size_t iq;
FOR_EACH_VEC_ELT (queued_reg_saves, iq, q)
{
size_t ir;
reg_saved_in_data *rir;
if (modified_in_p (q->reg, insn))
return true;
FOR_EACH_VEC_ELT (cur_trace->regs_saved_in_regs, ir, rir)
if (compare_reg_or_pc (q->reg, rir->orig_reg)
&& modified_in_p (rir->saved_in_reg, insn))
return true;
}
return false;
}
static rtx
reg_saved_in (rtx reg)
{
unsigned int regn = REGNO (reg);
queued_reg_save *q;
reg_saved_in_data *rir;
size_t i;
FOR_EACH_VEC_ELT (queued_reg_saves, i, q)
if (q->saved_reg && regn == REGNO (q->saved_reg))
return q->reg;
FOR_EACH_VEC_ELT (cur_trace->regs_saved_in_regs, i, rir)
if (regn == REGNO (rir->saved_in_reg))
return rir->orig_reg;
return NULL_RTX;
}
static void
dwarf2out_frame_debug_def_cfa (rtx pat)
{
memset (cur_cfa, 0, sizeof (*cur_cfa));
pat = strip_offset (pat, &cur_cfa->offset);
if (MEM_P (pat))
{
cur_cfa->indirect = 1;
pat = strip_offset (XEXP (pat, 0), &cur_cfa->base_offset);
}
gcc_assert (REG_P (pat));
cur_cfa->reg = dwf_regno (pat);
}
static void
dwarf2out_frame_debug_adjust_cfa (rtx pat)
{
rtx src, dest;
gcc_assert (GET_CODE (pat) == SET);
dest = XEXP (pat, 0);
src = XEXP (pat, 1);
switch (GET_CODE (src))
{
case PLUS:
gcc_assert (dwf_regno (XEXP (src, 0)) == cur_cfa->reg);
cur_cfa->offset -= rtx_to_poly_int64 (XEXP (src, 1));
break;
case REG:
break;
default:
gcc_unreachable ();
}
cur_cfa->reg = dwf_regno (dest);
gcc_assert (cur_cfa->indirect == 0);
}
static void
dwarf2out_frame_debug_cfa_offset (rtx set)
{
poly_int64 offset;
rtx src, addr, span;
unsigned int sregno;
src = XEXP (set, 1);
addr = XEXP (set, 0);
gcc_assert (MEM_P (addr));
addr = XEXP (addr, 0);
switch (GET_CODE (addr))
{
case REG:
gcc_assert (dwf_regno (addr) == cur_cfa->reg);
offset = -cur_cfa->offset;
break;
case PLUS:
gcc_assert (dwf_regno (XEXP (addr, 0)) == cur_cfa->reg);
offset = rtx_to_poly_int64 (XEXP (addr, 1)) - cur_cfa->offset;
break;
default:
gcc_unreachable ();
}
if (src == pc_rtx)
{
span = NULL;
sregno = DWARF_FRAME_RETURN_COLUMN;
}
else
{
span = targetm.dwarf_register_span (src);
sregno = dwf_regno (src);
}
if (!span)
reg_save (sregno, INVALID_REGNUM, offset);
else
{
poly_int64 span_offset = offset;
gcc_assert (GET_CODE (span) == PARALLEL);
const int par_len = XVECLEN (span, 0);
for (int par_index = 0; par_index < par_len; par_index++)
{
rtx elem = XVECEXP (span, 0, par_index);
sregno = dwf_regno (src);
reg_save (sregno, INVALID_REGNUM, span_offset);
span_offset += GET_MODE_SIZE (GET_MODE (elem));
}
}
}
static void
dwarf2out_frame_debug_cfa_register (rtx set)
{
rtx src, dest;
unsigned sregno, dregno;
src = XEXP (set, 1);
dest = XEXP (set, 0);
record_reg_saved_in_reg (dest, src);
if (src == pc_rtx)
sregno = DWARF_FRAME_RETURN_COLUMN;
else
sregno = dwf_regno (src);
dregno = dwf_regno (dest);
reg_save (sregno, dregno, 0);
}
static void
dwarf2out_frame_debug_cfa_expression (rtx set)
{
rtx src, dest, span;
dw_cfi_ref cfi = new_cfi ();
unsigned regno;
dest = SET_DEST (set);
src = SET_SRC (set);
gcc_assert (REG_P (src));
gcc_assert (MEM_P (dest));
span = targetm.dwarf_register_span (src);
gcc_assert (!span);
regno = dwf_regno (src);
cfi->dw_cfi_opc = DW_CFA_expression;
cfi->dw_cfi_oprnd1.dw_cfi_reg_num = regno;
cfi->dw_cfi_oprnd2.dw_cfi_loc
= mem_loc_descriptor (XEXP (dest, 0), get_address_mode (dest),
GET_MODE (dest), VAR_INIT_STATUS_INITIALIZED);
add_cfi (cfi);
update_row_reg_save (cur_row, regno, cfi);
}
static void
dwarf2out_frame_debug_cfa_val_expression (rtx set)
{
rtx dest = SET_DEST (set);
gcc_assert (REG_P (dest));
rtx span = targetm.dwarf_register_span (dest);
gcc_assert (!span);
rtx src = SET_SRC (set);
dw_cfi_ref cfi = new_cfi ();
cfi->dw_cfi_opc = DW_CFA_val_expression;
cfi->dw_cfi_oprnd1.dw_cfi_reg_num = dwf_regno (dest);
cfi->dw_cfi_oprnd2.dw_cfi_loc
= mem_loc_descriptor (src, GET_MODE (src),
GET_MODE (dest), VAR_INIT_STATUS_INITIALIZED);
add_cfi (cfi);
update_row_reg_save (cur_row, dwf_regno (dest), cfi);
}
static void
dwarf2out_frame_debug_cfa_restore (rtx reg)
{
gcc_assert (REG_P (reg));
rtx span = targetm.dwarf_register_span (reg);
if (!span)
{
unsigned int regno = dwf_regno (reg);
add_cfi_restore (regno);
update_row_reg_save (cur_row, regno, NULL);
}
else
{
gcc_assert (GET_CODE (span) == PARALLEL);
const int par_len = XVECLEN (span, 0);
for (int par_index = 0; par_index < par_len; par_index++)
{
reg = XVECEXP (span, 0, par_index);
gcc_assert (REG_P (reg));
unsigned int regno = dwf_regno (reg);
add_cfi_restore (regno);
update_row_reg_save (cur_row, regno, NULL);
}
}
}
static void
dwarf2out_frame_debug_cfa_window_save (void)
{
dw_cfi_ref cfi = new_cfi ();
cfi->dw_cfi_opc = DW_CFA_GNU_window_save;
add_cfi (cfi);
}
static void
dwarf2out_frame_debug_expr (rtx expr)
{
rtx src, dest, span;
poly_int64 offset;
dw_fde_ref fde;
if (GET_CODE (expr) == PARALLEL || GET_CODE (expr) == SEQUENCE)
{
int par_index;
int limit = XVECLEN (expr, 0);
rtx elem;
if (GET_CODE (expr) == PARALLEL)
for (par_index = 0; par_index < limit; par_index++)
{
elem = XVECEXP (expr, 0, par_index);
if (GET_CODE (elem) == SET
&& MEM_P (SET_DEST (elem))
&& (RTX_FRAME_RELATED_P (elem) || par_index == 0))
dwarf2out_frame_debug_expr (elem);
}
for (par_index = 0; par_index < limit; par_index++)
{
elem = XVECEXP (expr, 0, par_index);
if (GET_CODE (elem) == SET
&& (!MEM_P (SET_DEST (elem)) || GET_CODE (expr) == SEQUENCE)
&& (RTX_FRAME_RELATED_P (elem) || par_index == 0))
dwarf2out_frame_debug_expr (elem);
}
return;
}
gcc_assert (GET_CODE (expr) == SET);
src = SET_SRC (expr);
dest = SET_DEST (expr);
if (REG_P (src))
{
rtx rsi = reg_saved_in (src);
if (rsi)
src = rsi;
}
fde = cfun->fde;
switch (GET_CODE (dest))
{
case REG:
switch (GET_CODE (src))
{
case REG:
if (cur_cfa->reg == dwf_regno (src))
{
cur_cfa->reg = dwf_regno (dest);
cur_trace->cfa_temp.reg = cur_cfa->reg;
cur_trace->cfa_temp.offset = cur_cfa->offset;
}
else
{
gcc_assert (!fixed_regs [REGNO (dest)]
|| (dwf_regno (src) == DWARF_FRAME_RETURN_COLUMN));
if (fde
&& fde->stack_realign
&& REGNO (src) == STACK_POINTER_REGNUM)
gcc_assert (REGNO (dest) == HARD_FRAME_POINTER_REGNUM
&& fde->drap_reg != INVALID_REGNUM
&& cur_cfa->reg != dwf_regno (src));
else
queue_reg_save (src, dest, 0);
}
break;
case PLUS:
case MINUS:
case LO_SUM:
if (dest == stack_pointer_rtx)
{
if (REG_P (XEXP (src, 1)))
{
gcc_assert (dwf_regno (XEXP (src, 1))
== cur_trace->cfa_temp.reg);
offset = cur_trace->cfa_temp.offset;
}
else if (!poly_int_rtx_p (XEXP (src, 1), &offset))
gcc_unreachable ();
if (XEXP (src, 0) == hard_frame_pointer_rtx)
{
gcc_assert (cur_cfa->reg == dw_frame_pointer_regnum);
cur_cfa->reg = dw_stack_pointer_regnum;
}
else if (GET_CODE (src) == LO_SUM)
;
else
gcc_assert (XEXP (src, 0) == stack_pointer_rtx);
if (GET_CODE (src) != MINUS)
offset = -offset;
if (cur_cfa->reg == dw_stack_pointer_regnum)
cur_cfa->offset += offset;
if (cur_trace->cfa_store.reg == dw_stack_pointer_regnum)
cur_trace->cfa_store.offset += offset;
}
else if (dest == hard_frame_pointer_rtx)
{
gcc_assert (frame_pointer_needed);
gcc_assert (REG_P (XEXP (src, 0))
&& dwf_regno (XEXP (src, 0)) == cur_cfa->reg);
offset = rtx_to_poly_int64 (XEXP (src, 1));
if (GET_CODE (src) != MINUS)
offset = -offset;
cur_cfa->offset += offset;
cur_cfa->reg = dw_frame_pointer_regnum;
}
else
{
gcc_assert (GET_CODE (src) != MINUS);
if (REG_P (XEXP (src, 0))
&& dwf_regno (XEXP (src, 0)) == cur_cfa->reg
&& poly_int_rtx_p (XEXP (src, 1), &offset))
{
offset = -offset;
cur_cfa->offset += offset;
cur_cfa->reg = dwf_regno (dest);
cur_trace->cfa_temp.reg = cur_cfa->reg;
cur_trace->cfa_temp.offset = cur_cfa->offset;
}
else if (REG_P (XEXP (src, 0))
&& dwf_regno (XEXP (src, 0)) == cur_trace->cfa_temp.reg
&& XEXP (src, 1) == stack_pointer_rtx)
{
gcc_assert (cur_cfa->reg == dw_stack_pointer_regnum);
cur_trace->cfa_store.reg = dwf_regno (dest);
cur_trace->cfa_store.offset
= cur_cfa->offset - cur_trace->cfa_temp.offset;
}
else if (GET_CODE (src) == LO_SUM
&& poly_int_rtx_p (XEXP (src, 1),
&cur_trace->cfa_temp.offset))
cur_trace->cfa_temp.reg = dwf_regno (dest);
else
gcc_unreachable ();
}
break;
case CONST_INT:
case POLY_INT_CST:
cur_trace->cfa_temp.reg = dwf_regno (dest);
cur_trace->cfa_temp.offset = rtx_to_poly_int64 (src);
break;
case IOR:
gcc_assert (REG_P (XEXP (src, 0))
&& dwf_regno (XEXP (src, 0)) == cur_trace->cfa_temp.reg
&& CONST_INT_P (XEXP (src, 1)));
cur_trace->cfa_temp.reg = dwf_regno (dest);
if (!can_ior_p (cur_trace->cfa_temp.offset, INTVAL (XEXP (src, 1)),
&cur_trace->cfa_temp.offset))
gcc_unreachable ();
break;
case HIGH:
break;
case UNSPEC:
case UNSPEC_VOLATILE:
gcc_unreachable ();
return;
case AND:
if (fde && XEXP (src, 0) == stack_pointer_rtx)
{
dwarf2out_flush_queued_reg_saves ();
gcc_assert (cur_trace->cfa_store.reg
== dwf_regno (XEXP (src, 0)));
fde->stack_realign = 1;
fde->stack_realignment = INTVAL (XEXP (src, 1));
cur_trace->cfa_store.offset = 0;
if (cur_cfa->reg != dw_stack_pointer_regnum
&& cur_cfa->reg != dw_frame_pointer_regnum)
fde->drap_reg = cur_cfa->reg;
}
return;
default:
gcc_unreachable ();
}
break;
case MEM:
switch (GET_CODE (XEXP (dest, 0)))
{
case PRE_MODIFY:
case POST_MODIFY:
offset = -rtx_to_poly_int64 (XEXP (XEXP (XEXP (dest, 0), 1), 1));
gcc_assert (REGNO (XEXP (XEXP (dest, 0), 0)) == STACK_POINTER_REGNUM
&& cur_trace->cfa_store.reg == dw_stack_pointer_regnum);
cur_trace->cfa_store.offset += offset;
if (cur_cfa->reg == dw_stack_pointer_regnum)
cur_cfa->offset = cur_trace->cfa_store.offset;
if (GET_CODE (XEXP (dest, 0)) == POST_MODIFY)
offset -= cur_trace->cfa_store.offset;
else
offset = -cur_trace->cfa_store.offset;
break;
case PRE_INC:
case PRE_DEC:
case POST_DEC:
offset = GET_MODE_SIZE (GET_MODE (dest));
if (GET_CODE (XEXP (dest, 0)) == PRE_INC)
offset = -offset;
gcc_assert ((REGNO (XEXP (XEXP (dest, 0), 0))
== STACK_POINTER_REGNUM)
&& cur_trace->cfa_store.reg == dw_stack_pointer_regnum);
cur_trace->cfa_store.offset += offset;
if (fde
&& fde->stack_realign
&& REG_P (src)
&& REGNO (src) == HARD_FRAME_POINTER_REGNUM)
{
gcc_assert (cur_cfa->reg != dw_frame_pointer_regnum);
cur_trace->cfa_store.offset = 0;
}
if (cur_cfa->reg == dw_stack_pointer_regnum)
cur_cfa->offset = cur_trace->cfa_store.offset;
if (GET_CODE (XEXP (dest, 0)) == POST_DEC)
offset += -cur_trace->cfa_store.offset;
else
offset = -cur_trace->cfa_store.offset;
break;
case PLUS:
case MINUS:
case LO_SUM:
{
unsigned int regno;
gcc_assert (REG_P (XEXP (XEXP (dest, 0), 0)));
offset = rtx_to_poly_int64 (XEXP (XEXP (dest, 0), 1));
if (GET_CODE (XEXP (dest, 0)) == MINUS)
offset = -offset;
regno = dwf_regno (XEXP (XEXP (dest, 0), 0));
if (cur_cfa->reg == regno)
offset -= cur_cfa->offset;
else if (cur_trace->cfa_store.reg == regno)
offset -= cur_trace->cfa_store.offset;
else
{
gcc_assert (cur_trace->cfa_temp.reg == regno);
offset -= cur_trace->cfa_temp.offset;
}
}
break;
case REG:
{
unsigned int regno = dwf_regno (XEXP (dest, 0));
if (cur_cfa->reg == regno)
offset = -cur_cfa->offset;
else if (cur_trace->cfa_store.reg == regno)
offset = -cur_trace->cfa_store.offset;
else
{
gcc_assert (cur_trace->cfa_temp.reg == regno);
offset = -cur_trace->cfa_temp.offset;
}
}
break;
case POST_INC:
gcc_assert (cur_trace->cfa_temp.reg
== dwf_regno (XEXP (XEXP (dest, 0), 0)));
offset = -cur_trace->cfa_temp.offset;
cur_trace->cfa_temp.offset -= GET_MODE_SIZE (GET_MODE (dest));
break;
default:
gcc_unreachable ();
}
if (MEM_P (src))
break;
if (REG_P (src)
&& REGNO (src) != STACK_POINTER_REGNUM
&& REGNO (src) != HARD_FRAME_POINTER_REGNUM
&& dwf_regno (src) == cur_cfa->reg)
{
if (known_eq (cur_cfa->offset, 0))
{
if (fde
&& fde->stack_realign
&& cur_cfa->indirect == 0
&& cur_cfa->reg != dw_frame_pointer_regnum)
{
gcc_assert (fde->drap_reg == cur_cfa->reg);
cur_cfa->indirect = 1;
cur_cfa->reg = dw_frame_pointer_regnum;
cur_cfa->base_offset = offset;
cur_cfa->offset = 0;
fde->drap_reg_saved = 1;
break;
}
queue_reg_save (stack_pointer_rtx, NULL_RTX, offset);
break;
}
else
{
rtx x = XEXP (dest, 0);
if (!REG_P (x))
x = XEXP (x, 0);
gcc_assert (REG_P (x));
cur_cfa->reg = dwf_regno (x);
cur_cfa->base_offset = offset;
cur_cfa->indirect = 1;
break;
}
}
if (REG_P (src))
span = targetm.dwarf_register_span (src);
else
span = NULL;
if (!span)
queue_reg_save (src, NULL_RTX, offset);
else
{
poly_int64 span_offset = offset;
gcc_assert (GET_CODE (span) == PARALLEL);
const int par_len = XVECLEN (span, 0);
for (int par_index = 0; par_index < par_len; par_index++)
{
rtx elem = XVECEXP (span, 0, par_index);
queue_reg_save (elem, NULL_RTX, span_offset);
span_offset += GET_MODE_SIZE (GET_MODE (elem));
}
}
break;
default:
gcc_unreachable ();
}
}
static void
dwarf2out_frame_debug (rtx_insn *insn)
{
rtx note, n, pat;
bool handled_one = false;
for (note = REG_NOTES (insn); note; note = XEXP (note, 1))
switch (REG_NOTE_KIND (note))
{
case REG_FRAME_RELATED_EXPR:
pat = XEXP (note, 0);
goto do_frame_expr;
case REG_CFA_DEF_CFA:
dwarf2out_frame_debug_def_cfa (XEXP (note, 0));
handled_one = true;
break;
case REG_CFA_ADJUST_CFA:
n = XEXP (note, 0);
if (n == NULL)
{
n = PATTERN (insn);
if (GET_CODE (n) == PARALLEL)
n = XVECEXP (n, 0, 0);
}
dwarf2out_frame_debug_adjust_cfa (n);
handled_one = true;
break;
case REG_CFA_OFFSET:
n = XEXP (note, 0);
if (n == NULL)
n = single_set (insn);
dwarf2out_frame_debug_cfa_offset (n);
handled_one = true;
break;
case REG_CFA_REGISTER:
n = XEXP (note, 0);
if (n == NULL)
{
n = PATTERN (insn);
if (GET_CODE (n) == PARALLEL)
n = XVECEXP (n, 0, 0);
}
dwarf2out_frame_debug_cfa_register (n);
handled_one = true;
break;
case REG_CFA_EXPRESSION:
case REG_CFA_VAL_EXPRESSION:
n = XEXP (note, 0);
if (n == NULL)
n = single_set (insn);
if (REG_NOTE_KIND (note) == REG_CFA_EXPRESSION)
dwarf2out_frame_debug_cfa_expression (n);
else
dwarf2out_frame_debug_cfa_val_expression (n);
handled_one = true;
break;
case REG_CFA_RESTORE:
n = XEXP (note, 0);
if (n == NULL)
{
n = PATTERN (insn);
if (GET_CODE (n) == PARALLEL)
n = XVECEXP (n, 0, 0);
n = XEXP (n, 0);
}
dwarf2out_frame_debug_cfa_restore (n);
handled_one = true;
break;
case REG_CFA_SET_VDRAP:
n = XEXP (note, 0);
if (REG_P (n))
{
dw_fde_ref fde = cfun->fde;
if (fde)
{
gcc_assert (fde->vdrap_reg == INVALID_REGNUM);
if (REG_P (n))
fde->vdrap_reg = dwf_regno (n);
}
}
handled_one = true;
break;
case REG_CFA_TOGGLE_RA_MANGLE:
case REG_CFA_WINDOW_SAVE:
dwarf2out_frame_debug_cfa_window_save ();
handled_one = true;
break;
case REG_CFA_FLUSH_QUEUE:
handled_one = true;
break;
default:
break;
}
if (!handled_one)
{
pat = PATTERN (insn);
do_frame_expr:
dwarf2out_frame_debug_expr (pat);
if (clobbers_queued_reg_save (pat))
dwarf2out_flush_queued_reg_saves ();
}
}
static void
change_cfi_row (dw_cfi_row *old_row, dw_cfi_row *new_row)
{
size_t i, n_old, n_new, n_max;
dw_cfi_ref cfi;
if (new_row->cfa_cfi && !cfi_equal_p (old_row->cfa_cfi, new_row->cfa_cfi))
add_cfi (new_row->cfa_cfi);
else
{
cfi = def_cfa_0 (&old_row->cfa, &new_row->cfa);
if (cfi)
add_cfi (cfi);
}
n_old = vec_safe_length (old_row->reg_save);
n_new = vec_safe_length (new_row->reg_save);
n_max = MAX (n_old, n_new);
for (i = 0; i < n_max; ++i)
{
dw_cfi_ref r_old = NULL, r_new = NULL;
if (i < n_old)
r_old = (*old_row->reg_save)[i];
if (i < n_new)
r_new = (*new_row->reg_save)[i];
if (r_old == r_new)
;
else if (r_new == NULL)
add_cfi_restore (i);
else if (!cfi_equal_p (r_old, r_new))
add_cfi (r_new);
}
}
static bool
cfi_label_required_p (dw_cfi_ref cfi)
{
if (!dwarf2out_do_cfi_asm ())
return true;
if (dwarf_version == 2
&& debug_info_level > DINFO_LEVEL_TERSE
&& (write_symbols == DWARF2_DEBUG
|| write_symbols == VMS_AND_DWARF2_DEBUG))
{
switch (cfi->dw_cfi_opc)
{
case DW_CFA_def_cfa_offset:
case DW_CFA_def_cfa_offset_sf:
case DW_CFA_def_cfa_register:
case DW_CFA_def_cfa:
case DW_CFA_def_cfa_sf:
case DW_CFA_def_cfa_expression:
case DW_CFA_restore_state:
return true;
default:
return false;
}
}
return false;
}
static void
add_cfis_to_fde (void)
{
dw_fde_ref fde = cfun->fde;
rtx_insn *insn, *next;
for (insn = get_insns (); insn; insn = next)
{
next = NEXT_INSN (insn);
if (NOTE_P (insn) && NOTE_KIND (insn) == NOTE_INSN_SWITCH_TEXT_SECTIONS)
fde->dw_fde_switch_cfi_index = vec_safe_length (fde->dw_fde_cfi);
if (NOTE_P (insn) && NOTE_KIND (insn) == NOTE_INSN_CFI)
{
bool required = cfi_label_required_p (NOTE_CFI (insn));
while (next)
if (NOTE_P (next) && NOTE_KIND (next) == NOTE_INSN_CFI)
{
required |= cfi_label_required_p (NOTE_CFI (next));
next = NEXT_INSN (next);
}
else if (active_insn_p (next)
|| (NOTE_P (next) && (NOTE_KIND (next)
== NOTE_INSN_SWITCH_TEXT_SECTIONS)))
break;
else
next = NEXT_INSN (next);
if (required)
{
int num = dwarf2out_cfi_label_num;
const char *label = dwarf2out_cfi_label ();
dw_cfi_ref xcfi;
xcfi = new_cfi ();
xcfi->dw_cfi_opc = DW_CFA_advance_loc4;
xcfi->dw_cfi_oprnd1.dw_cfi_addr = label;
vec_safe_push (fde->dw_fde_cfi, xcfi);
rtx_note *tmp = emit_note_before (NOTE_INSN_CFI_LABEL, insn);
NOTE_LABEL_NUMBER (tmp) = num;
}
do
{
if (NOTE_P (insn) && NOTE_KIND (insn) == NOTE_INSN_CFI)
vec_safe_push (fde->dw_fde_cfi, NOTE_CFI (insn));
insn = NEXT_INSN (insn);
}
while (insn != next);
}
}
}
static void dump_cfi_row (FILE *f, dw_cfi_row *row);
static void
maybe_record_trace_start (rtx_insn *start, rtx_insn *origin)
{
dw_trace_info *ti;
ti = get_trace_info (start);
gcc_assert (ti != NULL);
if (dump_file)
{
fprintf (dump_file, "   saw edge from trace %u to %u (via %s %d)\n",
cur_trace->id, ti->id,
(origin ? rtx_name[(int) GET_CODE (origin)] : "fallthru"),
(origin ? INSN_UID (origin) : 0));
}
poly_int64 args_size = cur_trace->end_true_args_size;
if (ti->beg_row == NULL)
{
ti->beg_row = copy_cfi_row (cur_row);
ti->beg_true_args_size = args_size;
ti->cfa_store = cur_trace->cfa_store;
ti->cfa_temp = cur_trace->cfa_temp;
ti->regs_saved_in_regs = cur_trace->regs_saved_in_regs.copy ();
trace_work_list.safe_push (ti);
if (dump_file)
fprintf (dump_file, "\tpush trace %u to worklist\n", ti->id);
}
else
{
#if CHECKING_P
if (!cfi_row_equal_p (cur_row, ti->beg_row))
{
if (dump_file)
{
fprintf (dump_file, "Inconsistent CFI state!\n");
fprintf (dump_file, "SHOULD have:\n");
dump_cfi_row (dump_file, ti->beg_row);
fprintf (dump_file, "DO have:\n");
dump_cfi_row (dump_file, cur_row);
}
gcc_unreachable ();
}
#endif
if (maybe_ne (ti->beg_true_args_size, args_size))
ti->args_size_undefined = true;
}
}
static void
maybe_record_trace_start_abnormal (rtx_insn *start, rtx_insn *origin)
{
poly_int64 save_args_size, delta;
dw_cfa_location save_cfa;
save_args_size = cur_trace->end_true_args_size;
if (known_eq (save_args_size, 0))
{
maybe_record_trace_start (start, origin);
return;
}
delta = -save_args_size;
cur_trace->end_true_args_size = 0;
save_cfa = cur_row->cfa;
if (cur_row->cfa.reg == dw_stack_pointer_regnum)
{
if (!STACK_GROWS_DOWNWARD)
delta = -delta;
cur_row->cfa.offset += delta;
}
maybe_record_trace_start (start, origin);
cur_trace->end_true_args_size = save_args_size;
cur_row->cfa = save_cfa;
}
static void
create_trace_edges (rtx_insn *insn)
{
rtx tmp;
int i, n;
if (JUMP_P (insn))
{
rtx_jump_table_data *table;
if (find_reg_note (insn, REG_NON_LOCAL_GOTO, NULL_RTX))
return;
if (tablejump_p (insn, NULL, &table))
{
rtvec vec = table->get_labels ();
n = GET_NUM_ELEM (vec);
for (i = 0; i < n; ++i)
{
rtx_insn *lab = as_a <rtx_insn *> (XEXP (RTVEC_ELT (vec, i), 0));
maybe_record_trace_start (lab, insn);
}
}
else if (computed_jump_p (insn))
{
rtx_insn *temp;
unsigned int i;
FOR_EACH_VEC_SAFE_ELT (forced_labels, i, temp)
maybe_record_trace_start (temp, insn);
}
else if (returnjump_p (insn))
;
else if ((tmp = extract_asm_operands (PATTERN (insn))) != NULL)
{
n = ASM_OPERANDS_LABEL_LENGTH (tmp);
for (i = 0; i < n; ++i)
{
rtx_insn *lab =
as_a <rtx_insn *> (XEXP (ASM_OPERANDS_LABEL (tmp, i), 0));
maybe_record_trace_start (lab, insn);
}
}
else
{
rtx_insn *lab = JUMP_LABEL_AS_INSN (insn);
gcc_assert (lab != NULL);
maybe_record_trace_start (lab, insn);
}
}
else if (CALL_P (insn))
{
if (SIBLING_CALL_P (insn))
return;
if (can_nonlocal_goto (insn))
for (rtx_insn_list *lab = nonlocal_goto_handler_labels;
lab;
lab = lab->next ())
maybe_record_trace_start_abnormal (lab->insn (), insn);
}
else if (rtx_sequence *seq = dyn_cast <rtx_sequence *> (PATTERN (insn)))
{
int i, n = seq->len ();
for (i = 0; i < n; ++i)
create_trace_edges (seq->insn (i));
return;
}
if (CALL_P (insn) || cfun->can_throw_non_call_exceptions)
{
eh_landing_pad lp = get_eh_landing_pad_from_rtx (insn);
if (lp)
maybe_record_trace_start_abnormal (lp->landing_pad, insn);
}
}
static void
scan_insn_after (rtx_insn *insn)
{
if (RTX_FRAME_RELATED_P (insn))
dwarf2out_frame_debug (insn);
notice_args_size (insn);
}
static void
scan_trace (dw_trace_info *trace, bool entry)
{
rtx_insn *prev, *insn = trace->head;
dw_cfa_location this_cfa;
if (dump_file)
fprintf (dump_file, "Processing trace %u : start at %s %d\n",
trace->id, rtx_name[(int) GET_CODE (insn)],
INSN_UID (insn));
trace->end_row = copy_cfi_row (trace->beg_row);
trace->end_true_args_size = trace->beg_true_args_size;
cur_trace = trace;
cur_row = trace->end_row;
this_cfa = cur_row->cfa;
cur_cfa = &this_cfa;
if (entry
&& DEFAULT_INCOMING_FRAME_SP_OFFSET != INCOMING_FRAME_SP_OFFSET)
{
add_cfi_insn = insn;
gcc_assert (NOTE_P (insn) && NOTE_KIND (insn) == NOTE_INSN_DELETED);
this_cfa.offset = INCOMING_FRAME_SP_OFFSET;
def_cfa_1 (&this_cfa);
}
for (prev = insn, insn = NEXT_INSN (insn);
insn;
prev = insn, insn = NEXT_INSN (insn))
{
rtx_insn *control;
add_cfi_insn = prev;
if (BARRIER_P (insn))
{
queued_reg_saves.truncate (0);
break;
}
if (save_point_p (insn))
{
dwarf2out_flush_queued_reg_saves ();
maybe_record_trace_start (insn, NULL);
break;
}
if (DEBUG_INSN_P (insn) || !inside_basic_block_p (insn))
continue;
if (rtx_sequence *pat = dyn_cast <rtx_sequence *> (PATTERN (insn)))
{
rtx_insn *elt;
int i, n = pat->len ();
control = pat->insn (0);
if (can_throw_internal (control))
notice_eh_throw (control);
dwarf2out_flush_queued_reg_saves ();
if (JUMP_P (control) && INSN_ANNULLED_BRANCH_P (control))
{
gcc_assert (n == 2);
gcc_assert (!RTX_FRAME_RELATED_P (control));
gcc_assert (!find_reg_note (control, REG_ARGS_SIZE, NULL));
elt = pat->insn (1);
if (INSN_FROM_TARGET_P (elt))
{
cfi_vec save_row_reg_save;
add_cfi_insn = NULL;
poly_int64 restore_args_size = cur_trace->end_true_args_size;
cur_cfa = &cur_row->cfa;
save_row_reg_save = vec_safe_copy (cur_row->reg_save);
scan_insn_after (elt);
gcc_assert (!queued_reg_saves.length ());
create_trace_edges (control);
cur_trace->end_true_args_size = restore_args_size;
cur_row->cfa = this_cfa;
cur_row->reg_save = save_row_reg_save;
cur_cfa = &this_cfa;
}
else
{
create_trace_edges (control);
add_cfi_insn = insn;
scan_insn_after (elt);
def_cfa_1 (&this_cfa);
}
continue;
}
if (JUMP_P (control))
add_cfi_insn = insn;
for (i = 1; i < n; ++i)
{
elt = pat->insn (i);
scan_insn_after (elt);
}
dwarf2out_flush_queued_reg_saves ();
any_cfis_emitted = false;
add_cfi_insn = insn;
scan_insn_after (control);
}
else
{
if (can_throw_internal (insn))
{
notice_eh_throw (insn);
dwarf2out_flush_queued_reg_saves ();
}
else if (!NONJUMP_INSN_P (insn)
|| clobbers_queued_reg_save (insn)
|| find_reg_note (insn, REG_CFA_FLUSH_QUEUE, NULL))
dwarf2out_flush_queued_reg_saves ();
any_cfis_emitted = false;
add_cfi_insn = insn;
scan_insn_after (insn);
control = insn;
}
def_cfa_1 (&this_cfa);
if (any_cfis_emitted
|| find_reg_note (insn, REG_CFA_FLUSH_QUEUE, NULL))
dwarf2out_flush_queued_reg_saves ();
create_trace_edges (control);
}
add_cfi_insn = NULL;
cur_row = NULL;
cur_trace = NULL;
cur_cfa = NULL;
}
static void
create_cfi_notes (void)
{
dw_trace_info *ti;
gcc_checking_assert (!queued_reg_saves.exists ());
gcc_checking_assert (!trace_work_list.exists ());
ti = &trace_info[0];
scan_trace (ti, true);
while (!trace_work_list.is_empty ())
{
ti = trace_work_list.pop ();
scan_trace (ti, false);
}
queued_reg_saves.release ();
trace_work_list.release ();
}
static rtx_insn *
before_next_cfi_note (rtx_insn *start)
{
rtx_insn *prev = start;
while (start)
{
if (NOTE_P (start) && NOTE_KIND (start) == NOTE_INSN_CFI)
return prev;
prev = start;
start = NEXT_INSN (start);
}
gcc_unreachable ();
}
static void
connect_traces (void)
{
unsigned i, n = trace_info.length ();
dw_trace_info *prev_ti, *ti;
for (i = n - 1; i > 0; --i)
{
ti = &trace_info[i];
if (ti->beg_row == NULL)
{
trace_info.ordered_remove (i);
n -= 1;
}
else
gcc_assert (ti->end_row != NULL);
}
prev_ti = &trace_info[n - 1];
for (i = n - 1; i > 0; --i)
{
dw_cfi_row *old_row;
ti = prev_ti;
prev_ti = &trace_info[i - 1];
add_cfi_insn = ti->head;
if (ti->switch_sections)
old_row = cie_cfi_row;
else
{
old_row = prev_ti->end_row;
if (cfi_row_equal_p (old_row, ti->beg_row))
;
else if (cfi_row_equal_p (prev_ti->beg_row, ti->beg_row))
{
dw_cfi_ref cfi;
add_cfi_insn = before_next_cfi_note (prev_ti->head);
cfi = new_cfi ();
cfi->dw_cfi_opc = DW_CFA_remember_state;
add_cfi (cfi);
add_cfi_insn = ti->head;
cfi = new_cfi ();
cfi->dw_cfi_opc = DW_CFA_restore_state;
add_cfi (cfi);
old_row = prev_ti->beg_row;
}
}
change_cfi_row (old_row, ti->beg_row);
if (dump_file && add_cfi_insn != ti->head)
{
rtx_insn *note;
fprintf (dump_file, "Fixup between trace %u and %u:\n",
prev_ti->id, ti->id);
note = ti->head;
do
{
note = NEXT_INSN (note);
gcc_assert (NOTE_P (note) && NOTE_KIND (note) == NOTE_INSN_CFI);
output_cfi_directive (dump_file, NOTE_CFI (note));
}
while (note != add_cfi_insn);
}
}
if (cfun->eh->lp_array)
{
poly_int64 prev_args_size = 0;
for (i = 0; i < n; ++i)
{
ti = &trace_info[i];
if (ti->switch_sections)
prev_args_size = 0;
if (ti->eh_head == NULL)
continue;
gcc_assert (!ti->args_size_undefined);
if (maybe_ne (ti->beg_delay_args_size, prev_args_size))
{
add_cfi_insn = PREV_INSN (ti->eh_head);
add_cfi_args_size (ti->beg_delay_args_size);
}
prev_args_size = ti->end_delay_args_size;
}
}
}
static void
create_pseudo_cfg (void)
{
bool saw_barrier, switch_sections;
dw_trace_info ti;
rtx_insn *insn;
unsigned i;
trace_info.create (16);
memset (&ti, 0, sizeof (ti));
ti.head = get_insns ();
ti.beg_row = cie_cfi_row;
ti.cfa_store = cie_cfi_row->cfa;
ti.cfa_temp.reg = INVALID_REGNUM;
trace_info.quick_push (ti);
if (cie_return_save)
ti.regs_saved_in_regs.safe_push (*cie_return_save);
saw_barrier = false;
switch_sections = false;
for (insn = get_insns (); insn; insn = NEXT_INSN (insn))
{
if (BARRIER_P (insn))
saw_barrier = true;
else if (NOTE_P (insn)
&& NOTE_KIND (insn) == NOTE_INSN_SWITCH_TEXT_SECTIONS)
{
gcc_assert (saw_barrier);
switch_sections = true;
}
else if (save_point_p (insn)
&& (LABEL_P (insn) || !saw_barrier))
{
memset (&ti, 0, sizeof (ti));
ti.head = insn;
ti.switch_sections = switch_sections;
ti.id = trace_info.length ();
trace_info.safe_push (ti);
saw_barrier = false;
switch_sections = false;
}
}
trace_index
= new hash_table<trace_info_hasher> (trace_info.length ());
dw_trace_info *tp;
FOR_EACH_VEC_ELT (trace_info, i, tp)
{
dw_trace_info **slot;
if (dump_file)
fprintf (dump_file, "Creating trace %u : start at %s %d%s\n", tp->id,
rtx_name[(int) GET_CODE (tp->head)], INSN_UID (tp->head),
tp->switch_sections ? " (section switch)" : "");
slot = trace_index->find_slot_with_hash (tp, INSN_UID (tp->head), INSERT);
gcc_assert (*slot == NULL);
*slot = tp;
}
}
static void
initial_return_save (rtx rtl)
{
unsigned int reg = INVALID_REGNUM;
poly_int64 offset = 0;
switch (GET_CODE (rtl))
{
case REG:
reg = dwf_regno (rtl);
break;
case MEM:
rtl = XEXP (rtl, 0);
switch (GET_CODE (rtl))
{
case REG:
gcc_assert (REGNO (rtl) == STACK_POINTER_REGNUM);
offset = 0;
break;
case PLUS:
gcc_assert (REGNO (XEXP (rtl, 0)) == STACK_POINTER_REGNUM);
offset = rtx_to_poly_int64 (XEXP (rtl, 1));
break;
case MINUS:
gcc_assert (REGNO (XEXP (rtl, 0)) == STACK_POINTER_REGNUM);
offset = -rtx_to_poly_int64 (XEXP (rtl, 1));
break;
default:
gcc_unreachable ();
}
break;
case PLUS:
gcc_assert (CONST_INT_P (XEXP (rtl, 1)));
initial_return_save (XEXP (rtl, 0));
return;
default:
gcc_unreachable ();
}
if (reg != DWARF_FRAME_RETURN_COLUMN)
{
if (reg != INVALID_REGNUM)
record_reg_saved_in_reg (rtl, pc_rtx);
reg_save (DWARF_FRAME_RETURN_COLUMN, reg, offset - cur_row->cfa.offset);
}
}
static void
create_cie_data (void)
{
dw_cfa_location loc;
dw_trace_info cie_trace;
dw_stack_pointer_regnum = DWARF_FRAME_REGNUM (STACK_POINTER_REGNUM);
memset (&cie_trace, 0, sizeof (cie_trace));
cur_trace = &cie_trace;
add_cfi_vec = &cie_cfi_vec;
cie_cfi_row = cur_row = new_cfi_row ();
memset (&loc, 0, sizeof (loc));
loc.reg = dw_stack_pointer_regnum;
loc.offset = DEFAULT_INCOMING_FRAME_SP_OFFSET;
def_cfa_1 (&loc);
if (targetm.debug_unwind_info () == UI_DWARF2
|| targetm_common.except_unwind_info (&global_options) == UI_DWARF2)
{
initial_return_save (INCOMING_RETURN_ADDR_RTX);
switch (cie_trace.regs_saved_in_regs.length ())
{
case 0:
break;
case 1:
cie_return_save = ggc_alloc<reg_saved_in_data> ();
*cie_return_save = cie_trace.regs_saved_in_regs[0];
cie_trace.regs_saved_in_regs.release ();
break;
default:
gcc_unreachable ();
}
}
add_cfi_vec = NULL;
cur_row = NULL;
cur_trace = NULL;
}
static unsigned int
execute_dwarf2_frame (void)
{
dw_frame_pointer_regnum = DWARF_FRAME_REGNUM (HARD_FRAME_POINTER_REGNUM);
if (cie_cfi_vec == NULL)
create_cie_data ();
dwarf2out_alloc_current_fde ();
create_pseudo_cfg ();
create_cfi_notes ();
connect_traces ();
add_cfis_to_fde ();
{
size_t i;
dw_trace_info *ti;
FOR_EACH_VEC_ELT (trace_info, i, ti)
ti->regs_saved_in_regs.release ();
}
trace_info.release ();
delete trace_index;
trace_index = NULL;
return 0;
}

static const char *
dwarf_cfi_name (unsigned int cfi_opc)
{
const char *name = get_DW_CFA_name (cfi_opc);
if (name != NULL)
return name;
return "DW_CFA_<unknown>";
}
static void
output_cfa_loc (dw_cfi_ref cfi, int for_eh)
{
dw_loc_descr_ref loc;
unsigned long size;
if (cfi->dw_cfi_opc == DW_CFA_expression
|| cfi->dw_cfi_opc == DW_CFA_val_expression)
{
unsigned r =
DWARF2_FRAME_REG_OUT (cfi->dw_cfi_oprnd1.dw_cfi_reg_num, for_eh);
dw2_asm_output_data (1, r, NULL);
loc = cfi->dw_cfi_oprnd2.dw_cfi_loc;
}
else
loc = cfi->dw_cfi_oprnd1.dw_cfi_loc;
size = size_of_locs (loc);
dw2_asm_output_data_uleb128 (size, NULL);
output_loc_sequence (loc, for_eh);
}
static void
output_cfa_loc_raw (dw_cfi_ref cfi)
{
dw_loc_descr_ref loc;
unsigned long size;
if (cfi->dw_cfi_opc == DW_CFA_expression
|| cfi->dw_cfi_opc == DW_CFA_val_expression)
{
unsigned r =
DWARF2_FRAME_REG_OUT (cfi->dw_cfi_oprnd1.dw_cfi_reg_num, 1);
fprintf (asm_out_file, "%#x,", r);
loc = cfi->dw_cfi_oprnd2.dw_cfi_loc;
}
else
loc = cfi->dw_cfi_oprnd1.dw_cfi_loc;
size = size_of_locs (loc);
dw2_asm_output_data_uleb128_raw (size);
fputc (',', asm_out_file);
output_loc_sequence_raw (loc);
}
void
output_cfi (dw_cfi_ref cfi, dw_fde_ref fde, int for_eh)
{
unsigned long r;
HOST_WIDE_INT off;
if (cfi->dw_cfi_opc == DW_CFA_advance_loc)
dw2_asm_output_data (1, (cfi->dw_cfi_opc
| (cfi->dw_cfi_oprnd1.dw_cfi_offset & 0x3f)),
"DW_CFA_advance_loc " HOST_WIDE_INT_PRINT_HEX,
((unsigned HOST_WIDE_INT)
cfi->dw_cfi_oprnd1.dw_cfi_offset));
else if (cfi->dw_cfi_opc == DW_CFA_offset)
{
r = DWARF2_FRAME_REG_OUT (cfi->dw_cfi_oprnd1.dw_cfi_reg_num, for_eh);
dw2_asm_output_data (1, (cfi->dw_cfi_opc | (r & 0x3f)),
"DW_CFA_offset, column %#lx", r);
off = div_data_align (cfi->dw_cfi_oprnd2.dw_cfi_offset);
dw2_asm_output_data_uleb128 (off, NULL);
}
else if (cfi->dw_cfi_opc == DW_CFA_restore)
{
r = DWARF2_FRAME_REG_OUT (cfi->dw_cfi_oprnd1.dw_cfi_reg_num, for_eh);
dw2_asm_output_data (1, (cfi->dw_cfi_opc | (r & 0x3f)),
"DW_CFA_restore, column %#lx", r);
}
else
{
dw2_asm_output_data (1, cfi->dw_cfi_opc,
"%s", dwarf_cfi_name (cfi->dw_cfi_opc));
switch (cfi->dw_cfi_opc)
{
case DW_CFA_set_loc:
if (for_eh)
dw2_asm_output_encoded_addr_rtx (
ASM_PREFERRED_EH_DATA_FORMAT (1, 0),
gen_rtx_SYMBOL_REF (Pmode, cfi->dw_cfi_oprnd1.dw_cfi_addr),
false, NULL);
else
dw2_asm_output_addr (DWARF2_ADDR_SIZE,
cfi->dw_cfi_oprnd1.dw_cfi_addr, NULL);
fde->dw_fde_current_label = cfi->dw_cfi_oprnd1.dw_cfi_addr;
break;
case DW_CFA_advance_loc1:
dw2_asm_output_delta (1, cfi->dw_cfi_oprnd1.dw_cfi_addr,
fde->dw_fde_current_label, NULL);
fde->dw_fde_current_label = cfi->dw_cfi_oprnd1.dw_cfi_addr;
break;
case DW_CFA_advance_loc2:
dw2_asm_output_delta (2, cfi->dw_cfi_oprnd1.dw_cfi_addr,
fde->dw_fde_current_label, NULL);
fde->dw_fde_current_label = cfi->dw_cfi_oprnd1.dw_cfi_addr;
break;
case DW_CFA_advance_loc4:
dw2_asm_output_delta (4, cfi->dw_cfi_oprnd1.dw_cfi_addr,
fde->dw_fde_current_label, NULL);
fde->dw_fde_current_label = cfi->dw_cfi_oprnd1.dw_cfi_addr;
break;
case DW_CFA_MIPS_advance_loc8:
dw2_asm_output_delta (8, cfi->dw_cfi_oprnd1.dw_cfi_addr,
fde->dw_fde_current_label, NULL);
fde->dw_fde_current_label = cfi->dw_cfi_oprnd1.dw_cfi_addr;
break;
case DW_CFA_offset_extended:
r = DWARF2_FRAME_REG_OUT (cfi->dw_cfi_oprnd1.dw_cfi_reg_num, for_eh);
dw2_asm_output_data_uleb128 (r, NULL);
off = div_data_align (cfi->dw_cfi_oprnd2.dw_cfi_offset);
dw2_asm_output_data_uleb128 (off, NULL);
break;
case DW_CFA_def_cfa:
r = DWARF2_FRAME_REG_OUT (cfi->dw_cfi_oprnd1.dw_cfi_reg_num, for_eh);
dw2_asm_output_data_uleb128 (r, NULL);
dw2_asm_output_data_uleb128 (cfi->dw_cfi_oprnd2.dw_cfi_offset, NULL);
break;
case DW_CFA_offset_extended_sf:
r = DWARF2_FRAME_REG_OUT (cfi->dw_cfi_oprnd1.dw_cfi_reg_num, for_eh);
dw2_asm_output_data_uleb128 (r, NULL);
off = div_data_align (cfi->dw_cfi_oprnd2.dw_cfi_offset);
dw2_asm_output_data_sleb128 (off, NULL);
break;
case DW_CFA_def_cfa_sf:
r = DWARF2_FRAME_REG_OUT (cfi->dw_cfi_oprnd1.dw_cfi_reg_num, for_eh);
dw2_asm_output_data_uleb128 (r, NULL);
off = div_data_align (cfi->dw_cfi_oprnd2.dw_cfi_offset);
dw2_asm_output_data_sleb128 (off, NULL);
break;
case DW_CFA_restore_extended:
case DW_CFA_undefined:
case DW_CFA_same_value:
case DW_CFA_def_cfa_register:
r = DWARF2_FRAME_REG_OUT (cfi->dw_cfi_oprnd1.dw_cfi_reg_num, for_eh);
dw2_asm_output_data_uleb128 (r, NULL);
break;
case DW_CFA_register:
r = DWARF2_FRAME_REG_OUT (cfi->dw_cfi_oprnd1.dw_cfi_reg_num, for_eh);
dw2_asm_output_data_uleb128 (r, NULL);
r = DWARF2_FRAME_REG_OUT (cfi->dw_cfi_oprnd2.dw_cfi_reg_num, for_eh);
dw2_asm_output_data_uleb128 (r, NULL);
break;
case DW_CFA_def_cfa_offset:
case DW_CFA_GNU_args_size:
dw2_asm_output_data_uleb128 (cfi->dw_cfi_oprnd1.dw_cfi_offset, NULL);
break;
case DW_CFA_def_cfa_offset_sf:
off = div_data_align (cfi->dw_cfi_oprnd1.dw_cfi_offset);
dw2_asm_output_data_sleb128 (off, NULL);
break;
case DW_CFA_GNU_window_save:
break;
case DW_CFA_def_cfa_expression:
case DW_CFA_expression:
case DW_CFA_val_expression:
output_cfa_loc (cfi, for_eh);
break;
case DW_CFA_GNU_negative_offset_extended:
gcc_unreachable ();
default:
break;
}
}
}
void
output_cfi_directive (FILE *f, dw_cfi_ref cfi)
{
unsigned long r, r2;
switch (cfi->dw_cfi_opc)
{
case DW_CFA_advance_loc:
case DW_CFA_advance_loc1:
case DW_CFA_advance_loc2:
case DW_CFA_advance_loc4:
case DW_CFA_MIPS_advance_loc8:
case DW_CFA_set_loc:
gcc_assert (f != asm_out_file);
fprintf (f, "\t.cfi_advance_loc\n");
break;
case DW_CFA_offset:
case DW_CFA_offset_extended:
case DW_CFA_offset_extended_sf:
r = DWARF2_FRAME_REG_OUT (cfi->dw_cfi_oprnd1.dw_cfi_reg_num, 1);
fprintf (f, "\t.cfi_offset %lu, " HOST_WIDE_INT_PRINT_DEC"\n",
r, cfi->dw_cfi_oprnd2.dw_cfi_offset);
break;
case DW_CFA_restore:
case DW_CFA_restore_extended:
r = DWARF2_FRAME_REG_OUT (cfi->dw_cfi_oprnd1.dw_cfi_reg_num, 1);
fprintf (f, "\t.cfi_restore %lu\n", r);
break;
case DW_CFA_undefined:
r = DWARF2_FRAME_REG_OUT (cfi->dw_cfi_oprnd1.dw_cfi_reg_num, 1);
fprintf (f, "\t.cfi_undefined %lu\n", r);
break;
case DW_CFA_same_value:
r = DWARF2_FRAME_REG_OUT (cfi->dw_cfi_oprnd1.dw_cfi_reg_num, 1);
fprintf (f, "\t.cfi_same_value %lu\n", r);
break;
case DW_CFA_def_cfa:
case DW_CFA_def_cfa_sf:
r = DWARF2_FRAME_REG_OUT (cfi->dw_cfi_oprnd1.dw_cfi_reg_num, 1);
fprintf (f, "\t.cfi_def_cfa %lu, " HOST_WIDE_INT_PRINT_DEC"\n",
r, cfi->dw_cfi_oprnd2.dw_cfi_offset);
break;
case DW_CFA_def_cfa_register:
r = DWARF2_FRAME_REG_OUT (cfi->dw_cfi_oprnd1.dw_cfi_reg_num, 1);
fprintf (f, "\t.cfi_def_cfa_register %lu\n", r);
break;
case DW_CFA_register:
r = DWARF2_FRAME_REG_OUT (cfi->dw_cfi_oprnd1.dw_cfi_reg_num, 1);
r2 = DWARF2_FRAME_REG_OUT (cfi->dw_cfi_oprnd2.dw_cfi_reg_num, 1);
fprintf (f, "\t.cfi_register %lu, %lu\n", r, r2);
break;
case DW_CFA_def_cfa_offset:
case DW_CFA_def_cfa_offset_sf:
fprintf (f, "\t.cfi_def_cfa_offset "
HOST_WIDE_INT_PRINT_DEC"\n",
cfi->dw_cfi_oprnd1.dw_cfi_offset);
break;
case DW_CFA_remember_state:
fprintf (f, "\t.cfi_remember_state\n");
break;
case DW_CFA_restore_state:
fprintf (f, "\t.cfi_restore_state\n");
break;
case DW_CFA_GNU_args_size:
if (f == asm_out_file)
{
fprintf (f, "\t.cfi_escape %#x,", DW_CFA_GNU_args_size);
dw2_asm_output_data_uleb128_raw (cfi->dw_cfi_oprnd1.dw_cfi_offset);
if (flag_debug_asm)
fprintf (f, "\t%s args_size " HOST_WIDE_INT_PRINT_DEC,
ASM_COMMENT_START, cfi->dw_cfi_oprnd1.dw_cfi_offset);
fputc ('\n', f);
}
else
{
fprintf (f, "\t.cfi_GNU_args_size " HOST_WIDE_INT_PRINT_DEC "\n",
cfi->dw_cfi_oprnd1.dw_cfi_offset);
}
break;
case DW_CFA_GNU_window_save:
fprintf (f, "\t.cfi_window_save\n");
break;
case DW_CFA_def_cfa_expression:
case DW_CFA_expression:
case DW_CFA_val_expression:
if (f != asm_out_file)
{
fprintf (f, "\t.cfi_%scfa_%sexpression ...\n",
cfi->dw_cfi_opc == DW_CFA_def_cfa_expression ? "def_" : "",
cfi->dw_cfi_opc == DW_CFA_val_expression ? "val_" : "");
break;
}
fprintf (f, "\t.cfi_escape %#x,", cfi->dw_cfi_opc);
output_cfa_loc_raw (cfi);
fputc ('\n', f);
break;
default:
gcc_unreachable ();
}
}
void
dwarf2out_emit_cfi (dw_cfi_ref cfi)
{
if (dwarf2out_do_cfi_asm ())
output_cfi_directive (asm_out_file, cfi);
}
static void
dump_cfi_row (FILE *f, dw_cfi_row *row)
{
dw_cfi_ref cfi;
unsigned i;
cfi = row->cfa_cfi;
if (!cfi)
{
dw_cfa_location dummy;
memset (&dummy, 0, sizeof (dummy));
dummy.reg = INVALID_REGNUM;
cfi = def_cfa_0 (&dummy, &row->cfa);
}
output_cfi_directive (f, cfi);
FOR_EACH_VEC_SAFE_ELT (row->reg_save, i, cfi)
if (cfi)
output_cfi_directive (f, cfi);
}
void debug_cfi_row (dw_cfi_row *row);
void
debug_cfi_row (dw_cfi_row *row)
{
dump_cfi_row (stderr, row);
}

static GTY(()) signed char saved_do_cfi_asm = 0;
bool
dwarf2out_do_eh_frame (void)
{
return
(flag_unwind_tables || flag_exceptions)
&& targetm_common.except_unwind_info (&global_options) == UI_DWARF2;
}
bool
dwarf2out_do_frame (void)
{
if (write_symbols == DWARF2_DEBUG || write_symbols == VMS_AND_DWARF2_DEBUG)
return true;
if (saved_do_cfi_asm > 0)
return true;
if (targetm.debug_unwind_info () == UI_DWARF2)
return true;
if (dwarf2out_do_eh_frame ())
return true;
return false;
}
bool
dwarf2out_do_cfi_asm (void)
{
int enc;
if (saved_do_cfi_asm != 0)
return saved_do_cfi_asm > 0;
saved_do_cfi_asm = -1;
if (!flag_dwarf2_cfi_asm || !dwarf2out_do_frame ())
return false;
if (!HAVE_GAS_CFI_PERSONALITY_DIRECTIVE)
return false;
enc = ASM_PREFERRED_EH_DATA_FORMAT (2,1);
if ((enc & 0x70) != 0 && (enc & 0x70) != DW_EH_PE_pcrel)
return false;
enc = ASM_PREFERRED_EH_DATA_FORMAT (0,0);
if ((enc & 0x70) != 0 && (enc & 0x70) != DW_EH_PE_pcrel)
return false;
if (!HAVE_GAS_CFI_SECTIONS_DIRECTIVE && !dwarf2out_do_eh_frame ())
return false;
saved_do_cfi_asm = 1;
return true;
}
namespace {
const pass_data pass_data_dwarf2_frame =
{
RTL_PASS, 
"dwarf2", 
OPTGROUP_NONE, 
TV_FINAL, 
0, 
0, 
0, 
0, 
0, 
};
class pass_dwarf2_frame : public rtl_opt_pass
{
public:
pass_dwarf2_frame (gcc::context *ctxt)
: rtl_opt_pass (pass_data_dwarf2_frame, ctxt)
{}
virtual bool gate (function *);
virtual unsigned int execute (function *) { return execute_dwarf2_frame (); }
}; 
bool
pass_dwarf2_frame::gate (function *)
{
if (!targetm.have_prologue ())
return false;
return dwarf2out_do_frame ();
}
} 
rtl_opt_pass *
make_pass_dwarf2_frame (gcc::context *ctxt)
{
return new pass_dwarf2_frame (ctxt);
}
#include "gt-dwarf2cfi.h"

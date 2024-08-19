#define IN_TARGET_CODE 1
#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "backend.h"
#include "target.h"
#include "rtl.h"
#include "tree.h"
#include "df.h"
#include "memmodel.h"
#include "tm_p.h"
#include "stringpool.h"
#include "attribs.h"
#include "optabs.h"
#include "regs.h"
#include "emit-rtl.h"
#include "recog.h"
#include "diagnostic-core.h"
#include "alias.h"
#include "stor-layout.h"
#include "varasm.h"
#include "calls.h"
#include "conditions.h"
#include "output.h"
#include "insn-attr.h"
#include "flags.h"
#include "explow.h"
#include "expr.h"
#include "tm-constrs.h"
#include "builtins.h"
#include "target-def.h"
enum h8300_operand_class
{
H8OP_IMMEDIATE,
H8OP_REGISTER,
H8OP_MEM_ABSOLUTE,
H8OP_MEM_BASE,
H8OP_MEM_COMPLEX,
NUM_H8OPS
};
typedef unsigned char h8300_length_table[NUM_H8OPS - 1][NUM_H8OPS];
static const char *byte_reg (rtx, int);
static int h8300_interrupt_function_p (tree);
static int h8300_saveall_function_p (tree);
static int h8300_monitor_function_p (tree);
static int h8300_os_task_function_p (tree);
static void h8300_emit_stack_adjustment (int, HOST_WIDE_INT, bool);
static HOST_WIDE_INT round_frame_size (HOST_WIDE_INT);
static unsigned int compute_saved_regs (void);
static const char *cond_string (enum rtx_code);
static unsigned int h8300_asm_insn_count (const char *);
static tree h8300_handle_fndecl_attribute (tree *, tree, tree, int, bool *);
static tree h8300_handle_eightbit_data_attribute (tree *, tree, tree, int, bool *);
static tree h8300_handle_tiny_data_attribute (tree *, tree, tree, int, bool *);
static void h8300_print_operand_address (FILE *, machine_mode, rtx);
static void h8300_print_operand (FILE *, rtx, int);
static bool h8300_print_operand_punct_valid_p (unsigned char code);
#ifndef OBJECT_FORMAT_ELF
static void h8300_asm_named_section (const char *, unsigned int, tree);
#endif
static int h8300_register_move_cost (machine_mode, reg_class_t, reg_class_t);
static int h8300_and_costs (rtx);
static int h8300_shift_costs (rtx);
static void          h8300_push_pop               (int, int, bool, bool);
static int           h8300_stack_offset_p         (rtx, int);
static int           h8300_ldm_stm_regno          (rtx, int, int, int);
static void          h8300_reorg                  (void);
static unsigned int  h8300_constant_length        (rtx);
static unsigned int  h8300_displacement_length    (rtx, int);
static unsigned int  h8300_classify_operand       (rtx, int, enum h8300_operand_class *);
static unsigned int  h8300_length_from_table      (rtx, rtx, const h8300_length_table *);
static unsigned int  h8300_unary_length           (rtx);
static unsigned int  h8300_short_immediate_length (rtx);
static unsigned int  h8300_bitfield_length        (rtx, rtx);
static unsigned int  h8300_binary_length          (rtx_insn *, const h8300_length_table *);
static bool          h8300_short_move_mem_p       (rtx, enum rtx_code);
static unsigned int  h8300_move_length            (rtx *, const h8300_length_table *);
static bool	     h8300_hard_regno_scratch_ok  (unsigned int);
static rtx	     h8300_get_index (rtx, machine_mode mode, int *);
int cpu_type;
static int pragma_interrupt;
static int pragma_saveall;
static const char *const names_big[] =
{ "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7" };
static const char *const names_extended[] =
{ "er0", "er1", "er2", "er3", "er4", "er5", "er6", "er7" };
static const char *const names_upper_extended[] =
{ "e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7" };
const char * const *h8_reg_names;
const char *h8_push_op, *h8_pop_op, *h8_mov_op;
int h8300_move_ratio;

enum shift_alg
{
SHIFT_INLINE,
SHIFT_ROT_AND,
SHIFT_SPECIAL,
SHIFT_LOOP
};
enum shift_type
{
SHIFT_ASHIFT, SHIFT_LSHIFTRT, SHIFT_ASHIFTRT
};
#define INL SHIFT_INLINE
#define ROT SHIFT_ROT_AND
#define LOP SHIFT_LOOP
#define SPC SHIFT_SPECIAL
static enum shift_alg shift_alg_qi[3][3][8] = {
{
{ INL, INL, INL, INL, INL, ROT, ROT, ROT }, 
{ INL, INL, INL, INL, INL, ROT, ROT, ROT }, 
{ INL, INL, INL, INL, INL, LOP, LOP, SPC }  
},
{
{ INL, INL, INL, INL, INL, ROT, ROT, ROT }, 
{ INL, INL, INL, INL, INL, ROT, ROT, ROT }, 
{ INL, INL, INL, INL, INL, LOP, LOP, SPC }  
},
{
{ INL, INL, INL, INL, INL, INL, ROT, ROT }, 
{ INL, INL, INL, INL, INL, INL, ROT, ROT }, 
{ INL, INL, INL, INL, INL, INL, INL, SPC }  
}
};
static enum shift_alg shift_alg_hi[3][3][16] = {
{
{ INL, INL, INL, INL, INL, INL, INL, SPC,
SPC, SPC, SPC, SPC, SPC, SPC, SPC, SPC }, 
{ INL, INL, INL, INL, INL, LOP, LOP, SPC,
SPC, SPC, SPC, SPC, SPC, SPC, SPC, SPC }, 
{ INL, INL, INL, INL, INL, LOP, LOP, SPC,
SPC, SPC, SPC, SPC, SPC, SPC, SPC, SPC }, 
},
{
{ INL, INL, INL, INL, INL, INL, INL, SPC,
SPC, SPC, SPC, SPC, SPC, ROT, ROT, ROT }, 
{ INL, INL, INL, INL, INL, INL, INL, SPC,
SPC, SPC, SPC, SPC, SPC, ROT, ROT, ROT }, 
{ INL, INL, INL, INL, INL, INL, INL, SPC,
SPC, SPC, SPC, SPC, SPC, SPC, SPC, SPC }, 
},
{
{ INL, INL, INL, INL, INL, INL, INL, INL,
SPC, SPC, SPC, SPC, SPC, ROT, ROT, ROT }, 
{ INL, INL, INL, INL, INL, INL, INL, INL,
SPC, SPC, SPC, SPC, SPC, ROT, ROT, ROT }, 
{ INL, INL, INL, INL, INL, INL, INL, INL,
SPC, SPC, SPC, SPC, SPC, SPC, SPC, SPC }, 
}
};
static enum shift_alg shift_alg_si[3][3][32] = {
{
{ INL, INL, INL, LOP, LOP, LOP, LOP, LOP,
SPC, LOP, LOP, LOP, LOP, LOP, LOP, LOP,
SPC, SPC, SPC, SPC, SPC, LOP, LOP, LOP,
SPC, SPC, SPC, SPC, LOP, LOP, LOP, SPC }, 
{ INL, INL, INL, LOP, LOP, LOP, LOP, LOP,
SPC, SPC, LOP, LOP, LOP, LOP, LOP, SPC,
SPC, SPC, SPC, LOP, LOP, LOP, LOP, LOP,
SPC, SPC, SPC, SPC, SPC, LOP, LOP, SPC }, 
{ INL, INL, INL, LOP, LOP, LOP, LOP, LOP,
SPC, LOP, LOP, LOP, LOP, LOP, LOP, SPC,
SPC, SPC, LOP, LOP, LOP, LOP, LOP, LOP,
SPC, SPC, SPC, LOP, LOP, LOP, LOP, SPC }, 
},
{
{ INL, INL, INL, INL, INL, LOP, LOP, LOP,
SPC, LOP, LOP, LOP, LOP, LOP, LOP, SPC,
SPC, SPC, SPC, SPC, LOP, LOP, LOP, LOP,
SPC, LOP, LOP, LOP, SPC, SPC, SPC, SPC }, 
{ INL, INL, INL, INL, INL, LOP, LOP, LOP,
SPC, LOP, LOP, LOP, LOP, LOP, LOP, SPC,
SPC, SPC, SPC, SPC, LOP, LOP, LOP, LOP,
SPC, LOP, LOP, LOP, SPC, SPC, SPC, SPC }, 
{ INL, INL, INL, INL, INL, LOP, LOP, LOP,
SPC, LOP, LOP, LOP, LOP, LOP, LOP, LOP,
SPC, SPC, SPC, SPC, LOP, LOP, LOP, LOP,
SPC, LOP, LOP, LOP, LOP, LOP, LOP, SPC }, 
},
{
{ INL, INL, INL, INL, INL, INL, INL, INL,
INL, INL, INL, LOP, LOP, LOP, LOP, SPC,
SPC, SPC, SPC, SPC, SPC, SPC, LOP, LOP,
SPC, SPC, LOP, LOP, SPC, SPC, SPC, SPC }, 
{ INL, INL, INL, INL, INL, INL, INL, INL,
INL, INL, INL, LOP, LOP, LOP, LOP, SPC,
SPC, SPC, SPC, SPC, SPC, SPC, LOP, LOP,
SPC, SPC, LOP, LOP, SPC, SPC, SPC, SPC }, 
{ INL, INL, INL, INL, INL, INL, INL, INL,
INL, INL, INL, LOP, LOP, LOP, LOP, LOP,
SPC, SPC, SPC, SPC, SPC, SPC, LOP, LOP,
SPC, SPC, LOP, LOP, LOP, LOP, LOP, SPC }, 
}
};
#undef INL
#undef ROT
#undef LOP
#undef SPC
enum h8_cpu
{
H8_300,
H8_300H,
H8_S
};
static void
h8300_option_override (void)
{
static const char *const h8_push_ops[2] = { "push" , "push.l" };
static const char *const h8_pop_ops[2]  = { "pop"  , "pop.l"  };
static const char *const h8_mov_ops[2]  = { "mov.w", "mov.l"  };
#ifndef OBJECT_FORMAT_ELF
if (TARGET_H8300SX)
{
error ("-msx is not supported in coff");
target_flags |= MASK_H8300S;
}
#endif
if (TARGET_H8300)
{
cpu_type = (int) CPU_H8300;
h8_reg_names = names_big;
}
else
{
cpu_type = (int) CPU_H8300H;
h8_reg_names = names_extended;
}
h8_push_op = h8_push_ops[cpu_type];
h8_pop_op = h8_pop_ops[cpu_type];
h8_mov_op = h8_mov_ops[cpu_type];
if (!TARGET_H8300S && TARGET_MAC)
{
error ("-ms2600 is used without -ms");
target_flags |= MASK_H8300S_1;
}
if (TARGET_H8300 && TARGET_NORMAL_MODE)
{
error ("-mn is used without -mh or -ms or -msx");
target_flags ^= MASK_NORMAL_MODE;
}
if (! TARGET_H8300S &&  TARGET_EXR)
{
error ("-mexr is used without -ms");
target_flags |= MASK_H8300S_1;
}
if (TARGET_H8300 && TARGET_INT32)
{
error ("-mint32 is not supported for H8300 and H8300L targets");
target_flags ^= MASK_INT32;
}
if ((!TARGET_H8300S  &&  TARGET_EXR) && (!TARGET_H8300SX && TARGET_EXR))
{
error ("-mexr is used without -ms or -msx");
target_flags |= MASK_H8300S_1;
}
if ((!TARGET_H8300S  &&  TARGET_NEXR) && (!TARGET_H8300SX && TARGET_NEXR))
{
warning (OPT_mno_exr, "-mno-exr valid only with -ms or -msx    \
- Option ignored!");
}
#ifdef H8300_LINUX 
if ((TARGET_NORMAL_MODE))
{
error ("-mn is not supported for linux targets");
target_flags ^= MASK_NORMAL_MODE;
}
#endif
if (optimize_size)
{
shift_alg_hi[H8_300][SHIFT_ASHIFT][5] = SHIFT_LOOP;
shift_alg_hi[H8_300][SHIFT_ASHIFT][6] = SHIFT_LOOP;
shift_alg_hi[H8_300][SHIFT_ASHIFT][13] = SHIFT_LOOP;
shift_alg_hi[H8_300][SHIFT_ASHIFT][14] = SHIFT_LOOP;
shift_alg_hi[H8_300][SHIFT_LSHIFTRT][13] = SHIFT_LOOP;
shift_alg_hi[H8_300][SHIFT_LSHIFTRT][14] = SHIFT_LOOP;
shift_alg_hi[H8_300][SHIFT_ASHIFTRT][13] = SHIFT_LOOP;
shift_alg_hi[H8_300][SHIFT_ASHIFTRT][14] = SHIFT_LOOP;
shift_alg_hi[H8_300H][SHIFT_ASHIFT][5] = SHIFT_LOOP;
shift_alg_hi[H8_300H][SHIFT_ASHIFT][6] = SHIFT_LOOP;
shift_alg_hi[H8_300H][SHIFT_LSHIFTRT][5] = SHIFT_LOOP;
shift_alg_hi[H8_300H][SHIFT_LSHIFTRT][6] = SHIFT_LOOP;
shift_alg_hi[H8_300H][SHIFT_ASHIFTRT][5] = SHIFT_LOOP;
shift_alg_hi[H8_300H][SHIFT_ASHIFTRT][6] = SHIFT_LOOP;
shift_alg_hi[H8_300H][SHIFT_ASHIFTRT][13] = SHIFT_LOOP;
shift_alg_hi[H8_300H][SHIFT_ASHIFTRT][14] = SHIFT_LOOP;
shift_alg_hi[H8_S][SHIFT_ASHIFTRT][14] = SHIFT_LOOP;
}
if (!TARGET_H8300SX)
{
h8300_move_ratio = 3;
}
else if (flag_omit_frame_pointer)
{
h8300_move_ratio = 4;
}
else if (optimize_size)
{
h8300_move_ratio = 4;
}
else
{
h8300_move_ratio = 6;
}
if (flag_strict_volatile_bitfields < 0 && abi_version_at_least(2))
flag_strict_volatile_bitfields = 1;
}
static const char *
byte_reg (rtx x, int b)
{
static const char *const names_small[] = {
"r0l", "r0h", "r1l", "r1h", "r2l", "r2h", "r3l", "r3h",
"r4l", "r4h", "r5l", "r5h", "r6l", "r6h", "r7l", "r7h"
};
gcc_assert (REG_P (x));
return names_small[REGNO (x) * 2 + b];
}
#define WORD_REG_USED(regno)						\
(regno < SP_REG							\
\
&& ! TREE_THIS_VOLATILE (current_function_decl)			\
&& (h8300_saveall_function_p (current_function_decl)			\
\
|| (df_regs_ever_live_p (regno) && !call_used_regs[regno])	\
\
|| (regno == HARD_FRAME_POINTER_REGNUM && df_regs_ever_live_p (regno)) \
\
|| (h8300_current_function_interrupt_function_p ()		\
&& df_regs_ever_live_p (regno))				\
\
|| (h8300_current_function_interrupt_function_p ()		\
&& call_used_regs[regno]					\
&& !crtl->is_leaf)))
static rtx_insn *
F (rtx_insn *x, bool set_it)
{
if (set_it)
RTX_FRAME_RELATED_P (x) = 1;
return x;
}
static rtx
Fpa (rtx par)
{
int len = XVECLEN (par, 0);
int i;
for (i = 0; i < len; i++)
RTX_FRAME_RELATED_P (XVECEXP (par, 0, i)) = 1;
return par;
}
static void
h8300_emit_stack_adjustment (int sign, HOST_WIDE_INT size, bool in_prologue)
{
if (size == 0)
return;
if (TARGET_H8300
&& size > 4
&& !h8300_current_function_interrupt_function_p ()
&& !(cfun->static_chain_decl != NULL && sign < 0))
{
rtx r3 = gen_rtx_REG (Pmode, 3);
F (emit_insn (gen_movhi (r3, GEN_INT (sign * size))), in_prologue);
F (emit_insn (gen_addhi3 (stack_pointer_rtx,
stack_pointer_rtx, r3)), in_prologue);
}
else
{
if (Pmode == HImode)
{
rtx_insn *x = emit_insn (gen_addhi3 (stack_pointer_rtx,
stack_pointer_rtx,
GEN_INT (sign * size)));
if (size < 4)
F (x, in_prologue);
}
else
F (emit_insn (gen_addsi3 (stack_pointer_rtx,
stack_pointer_rtx, GEN_INT (sign * size))), in_prologue);
}
}
static HOST_WIDE_INT
round_frame_size (HOST_WIDE_INT size)
{
return ((size + STACK_BOUNDARY / BITS_PER_UNIT - 1)
& -STACK_BOUNDARY / BITS_PER_UNIT);
}
static unsigned int
compute_saved_regs (void)
{
unsigned int saved_regs = 0;
int regno;
for (regno = 0; regno <= HARD_FRAME_POINTER_REGNUM; regno++)
{
if (WORD_REG_USED (regno))
saved_regs |= 1 << regno;
}
if (frame_pointer_needed)
saved_regs &= ~(1 << HARD_FRAME_POINTER_REGNUM);
return saved_regs;
}
static rtx
push (int rn, bool in_prologue)
{
rtx reg = gen_rtx_REG (word_mode, rn);
rtx x;
if (TARGET_H8300)
x = gen_push_h8300 (reg);
else if (!TARGET_NORMAL_MODE)
x = gen_push_h8300hs_advanced (reg);
else
x = gen_push_h8300hs_normal (reg);
x = F (emit_insn (x), in_prologue);
add_reg_note (x, REG_INC, stack_pointer_rtx);
return x;
}
static rtx
pop (int rn)
{
rtx reg = gen_rtx_REG (word_mode, rn);
rtx x;
if (TARGET_H8300)
x = gen_pop_h8300 (reg);
else if (!TARGET_NORMAL_MODE)
x = gen_pop_h8300hs_advanced (reg);
else
x = gen_pop_h8300hs_normal (reg);
x = emit_insn (x);
add_reg_note (x, REG_INC, stack_pointer_rtx);
return x;
}
static void
h8300_push_pop (int regno, int nregs, bool pop_p, bool return_p)
{
int i, j;
rtvec vec;
rtx sp, offset, x;
if (!return_p && nregs == 1)
{
if (pop_p)
pop (regno);
else
push (regno, false);
return;
}
vec = rtvec_alloc ((return_p ? 1 : 0) + nregs + 1);
sp = stack_pointer_rtx;
i = 0;
if (return_p)
{
RTVEC_ELT (vec, i) = ret_rtx;
i++;
}
for (j = 0; j < nregs; j++)
{
rtx lhs, rhs;
if (pop_p)
{
lhs = gen_rtx_REG (SImode, regno + j);
rhs = gen_rtx_MEM (SImode, plus_constant (Pmode, sp,
(nregs - j - 1) * 4));
}
else
{
lhs = gen_rtx_MEM (SImode, plus_constant (Pmode, sp, (j + 1) * -4));
rhs = gen_rtx_REG (SImode, regno + j);
}
RTVEC_ELT (vec, i + j) = gen_rtx_SET (lhs, rhs);
}
offset = GEN_INT ((pop_p ? nregs : -nregs) * 4);
RTVEC_ELT (vec, i + j) = gen_rtx_SET (sp, gen_rtx_PLUS (Pmode, sp, offset));
x = gen_rtx_PARALLEL (VOIDmode, vec);
if (!pop_p)
x = Fpa (x);
if (return_p)
emit_jump_insn (x);
else
emit_insn (x);
}
static int
h8300_stack_offset_p (rtx x, int offset)
{
if (offset == 0)
return x == stack_pointer_rtx;
return (GET_CODE (x) == PLUS
&& XEXP (x, 0) == stack_pointer_rtx
&& GET_CODE (XEXP (x, 1)) == CONST_INT
&& INTVAL (XEXP (x, 1)) == offset);
}
static int
h8300_ldm_stm_regno (rtx x, int load_p, int index, int nregs)
{
int regindex, memindex, offset;
if (load_p)
regindex = 0, memindex = 1, offset = (nregs - index - 1) * 4;
else
memindex = 0, regindex = 1, offset = (index + 1) * -4;
if (GET_CODE (x) == SET
&& GET_CODE (XEXP (x, regindex)) == REG
&& GET_CODE (XEXP (x, memindex)) == MEM
&& h8300_stack_offset_p (XEXP (XEXP (x, memindex), 0), offset))
return REGNO (XEXP (x, regindex));
return -1;
}
int
h8300_ldm_stm_parallel (rtvec vec, int load_p, int first)
{
rtx last;
int nregs, i, regno, adjust;
if (GET_NUM_ELEM (vec) < 3)
return false;
nregs = GET_NUM_ELEM (vec) - first - 1;
regno = h8300_ldm_stm_regno (RTVEC_ELT (vec, first), load_p, 0, nregs);
if (regno < 0 || regno + nregs > 8)
return false;
if (!TARGET_H8300SX)
{
if ((regno & 1) != 0)
return false;
if (nregs > 2 && (regno & 3) != 0)
return false;
}
for (i = 1; i < nregs; i++)
if (h8300_ldm_stm_regno (RTVEC_ELT (vec, first + i), load_p, i, nregs)
!= regno + i)
return false;
last = RTVEC_ELT (vec, first + nregs);
adjust = (load_p ? nregs : -nregs) * 4;
return (GET_CODE (last) == SET
&& SET_DEST (last) == stack_pointer_rtx
&& h8300_stack_offset_p (SET_SRC (last), adjust));
}
void
h8300_expand_prologue (void)
{
int regno;
int saved_regs;
int n_regs;
if (h8300_os_task_function_p (current_function_decl))
return;
if (h8300_monitor_function_p (current_function_decl))
emit_insn (gen_monitor_prologue ());
if (frame_pointer_needed)
{
push (HARD_FRAME_POINTER_REGNUM, true);
F (emit_move_insn (hard_frame_pointer_rtx, stack_pointer_rtx), true);
}
saved_regs = compute_saved_regs ();
for (regno = 0; regno < FIRST_PSEUDO_REGISTER; regno += n_regs)
{
n_regs = 1;
if (saved_regs & (1 << regno))
{
if (TARGET_H8300S)
{
if ((!TARGET_H8300SX || (regno & 3) == 0)
&& ((saved_regs >> regno) & 0x0f) == 0x0f)
n_regs = 4;
else if ((!TARGET_H8300SX || (regno & 3) == 0)
&& ((saved_regs >> regno) & 0x07) == 0x07)
n_regs = 3;
else if ((!TARGET_H8300SX || (regno & 1) == 0)
&& ((saved_regs >> regno) & 0x03) == 0x03)
n_regs = 2;
}
h8300_push_pop (regno, n_regs, false, false);
}
}
h8300_emit_stack_adjustment (-1, round_frame_size (get_frame_size ()), true);
if (flag_stack_usage_info)
current_function_static_stack_size
= round_frame_size (get_frame_size ())
+ (__builtin_popcount (saved_regs) * UNITS_PER_WORD)
+ (frame_pointer_needed ? UNITS_PER_WORD : 0);
}
int
h8300_can_use_return_insn_p (void)
{
return (reload_completed
&& !frame_pointer_needed
&& get_frame_size () == 0
&& compute_saved_regs () == 0);
}
void
h8300_expand_epilogue (void)
{
int regno;
int saved_regs;
int n_regs;
HOST_WIDE_INT frame_size;
bool returned_p;
if (h8300_os_task_function_p (current_function_decl))
return;
frame_size = round_frame_size (get_frame_size ());
returned_p = false;
h8300_emit_stack_adjustment (1, frame_size, false);
saved_regs = compute_saved_regs ();
for (regno = FIRST_PSEUDO_REGISTER - 1; regno >= 0; regno -= n_regs)
{
n_regs = 1;
if (saved_regs & (1 << regno))
{
if (TARGET_H8300S)
{
if ((TARGET_H8300SX || (regno & 3) == 3)
&& ((saved_regs << 3 >> regno) & 0x0f) == 0x0f)
n_regs = 4;
else if ((TARGET_H8300SX || (regno & 3) == 2)
&& ((saved_regs << 2 >> regno) & 0x07) == 0x07)
n_regs = 3;
else if ((TARGET_H8300SX || (regno & 1) == 1)
&& ((saved_regs << 1 >> regno) & 0x03) == 0x03)
n_regs = 2;
}
if (TARGET_H8300SX
&& !frame_pointer_needed
&& frame_size == 0
&& (saved_regs & ((1 << (regno - n_regs + 1)) - 1)) == 0)
returned_p = true;
h8300_push_pop (regno - n_regs + 1, n_regs, true, returned_p);
}
}
if (frame_pointer_needed)
{
if (TARGET_H8300SX)
returned_p = true;
h8300_push_pop (HARD_FRAME_POINTER_REGNUM, 1, true, returned_p);
}
if (!returned_p)
emit_jump_insn (ret_rtx);
}
int
h8300_current_function_interrupt_function_p (void)
{
return (h8300_interrupt_function_p (current_function_decl));
}
int
h8300_current_function_monitor_function_p ()
{
return (h8300_monitor_function_p (current_function_decl));
}
static void
h8300_file_start (void)
{
default_file_start ();
if (TARGET_H8300SX)
fputs (TARGET_NORMAL_MODE ? "\t.h8300sxn\n" : "\t.h8300sx\n", asm_out_file);
else if (TARGET_H8300S)
fputs (TARGET_NORMAL_MODE ? "\t.h8300sn\n" : "\t.h8300s\n", asm_out_file);
else if (TARGET_H8300H)
fputs (TARGET_NORMAL_MODE ? "\t.h8300hn\n" : "\t.h8300h\n", asm_out_file);
}
static void
h8300_file_end (void)
{
fputs ("\t.end\n", asm_out_file);
}

void
split_adds_subs (machine_mode mode, rtx *operands)
{
HOST_WIDE_INT val = INTVAL (operands[1]);
rtx reg = operands[0];
HOST_WIDE_INT sign = 1;
HOST_WIDE_INT amount;
rtx (*gen_add) (rtx, rtx, rtx);
if (val < 0)
{
val = -val;
sign = -1;
}
switch (mode)
{
case E_HImode:
gen_add = gen_addhi3;
break;
case E_SImode:
gen_add = gen_addsi3;
break;
default:
gcc_unreachable ();
}
for (amount = (TARGET_H8300H || TARGET_H8300S) ? 4 : 2;
amount > 0;
amount /= 2)
{
for (; val >= amount; val -= amount)
emit_insn (gen_add (reg, reg, GEN_INT (sign * amount)));
}
return;
}
void
h8300_pr_interrupt (struct cpp_reader *pfile ATTRIBUTE_UNUSED)
{
pragma_interrupt = 1;
}
void
h8300_pr_saveall (struct cpp_reader *pfile ATTRIBUTE_UNUSED)
{
pragma_saveall = 1;
}
static rtx
h8300_function_arg (cumulative_args_t cum_v, machine_mode mode,
const_tree type, bool named)
{
CUMULATIVE_ARGS *cum = get_cumulative_args (cum_v);
static const char *const hand_list[] = {
"__main",
"__cmpsi2",
"__divhi3",
"__modhi3",
"__udivhi3",
"__umodhi3",
"__divsi3",
"__modsi3",
"__udivsi3",
"__umodsi3",
"__mulhi3",
"__mulsi3",
"__reg_memcpy",
"__reg_memset",
"__ucmpsi2",
0,
};
rtx result = NULL_RTX;
const char *fname;
int regpass = 0;
if (!named)
return NULL_RTX;
if (TARGET_QUICKCALL)
regpass = 3;
if (cum->libcall)
{
const char * const *p;
fname = XSTR (cum->libcall, 0);
for (p = hand_list; *p && strcmp (*p, fname) != 0; p++)
;
if (*p)
regpass = 4;
}
if (regpass)
{
int size;
if (mode == BLKmode)
size = int_size_in_bytes (type);
else
size = GET_MODE_SIZE (mode);
if (size + cum->nbytes <= regpass * UNITS_PER_WORD
&& cum->nbytes / UNITS_PER_WORD <= 3)
result = gen_rtx_REG (mode, cum->nbytes / UNITS_PER_WORD);
}
return result;
}
static void
h8300_function_arg_advance (cumulative_args_t cum_v, machine_mode mode,
const_tree type, bool named ATTRIBUTE_UNUSED)
{
CUMULATIVE_ARGS *cum = get_cumulative_args (cum_v);
cum->nbytes += (mode != BLKmode
? (GET_MODE_SIZE (mode) + UNITS_PER_WORD - 1) & -UNITS_PER_WORD
: (int_size_in_bytes (type) + UNITS_PER_WORD - 1) & -UNITS_PER_WORD);
}

static int
h8300_register_move_cost (machine_mode mode ATTRIBUTE_UNUSED,
reg_class_t from, reg_class_t to)
{
if (from == MAC_REGS || to == MAC_REG)
return 6;
else
return 3;
}
static int
h8300_and_costs (rtx x)
{
rtx operands[4];
if (GET_MODE (x) == QImode)
return 1;
if (GET_MODE (x) != HImode
&& GET_MODE (x) != SImode)
return 100;
operands[0] = NULL;
operands[1] = XEXP (x, 0);
operands[2] = XEXP (x, 1);
operands[3] = x;
return compute_logical_op_length (GET_MODE (x), operands) / 2;
}
static int
h8300_shift_costs (rtx x)
{
rtx operands[4];
if (GET_MODE (x) != QImode
&& GET_MODE (x) != HImode
&& GET_MODE (x) != SImode)
return 100;
operands[0] = NULL;
operands[1] = NULL;
operands[2] = XEXP (x, 1);
operands[3] = x;
return compute_a_shift_length (NULL, operands) / 2;
}
static bool
h8300_rtx_costs (rtx x, machine_mode mode ATTRIBUTE_UNUSED, int outer_code,
int opno ATTRIBUTE_UNUSED, int *total, bool speed)
{
int code = GET_CODE (x);
if (TARGET_H8300SX && outer_code == MEM)
{
if (register_operand (x, VOIDmode)
|| GET_CODE (x) == POST_INC
|| GET_CODE (x) == POST_DEC
|| CONSTANT_P (x))
*total = 0;
else
*total = COSTS_N_INSNS (1);
return true;
}
switch (code)
{
case CONST_INT:
{
HOST_WIDE_INT n = INTVAL (x);
if (TARGET_H8300SX)
{
*total = 0;
return true;
}
if (n >= -4 && n <= 4)
{
switch ((int) n)
{
case 0:
*total = 0;
return true;
case 1:
case 2:
case -1:
case -2:
*total = 0 + (outer_code == SET);
return true;
case 4:
case -4:
if (TARGET_H8300H || TARGET_H8300S)
*total = 0 + (outer_code == SET);
else
*total = 1;
return true;
}
}
*total = 1;
return true;
}
case CONST:
case LABEL_REF:
case SYMBOL_REF:
if (TARGET_H8300SX)
{
*total = 0;
return true;
}
*total = 3;
return true;
case CONST_DOUBLE:
*total = 20;
return true;
case COMPARE:
if (XEXP (x, 1) == const0_rtx)
*total = 0;
return false;
case AND:
if (!h8300_dst_operand (XEXP (x, 0), VOIDmode)
|| !h8300_src_operand (XEXP (x, 1), VOIDmode))
return false;
*total = COSTS_N_INSNS (h8300_and_costs (x));
return true;
case MOD:
case DIV:
case UMOD:
case UDIV:
if (TARGET_H8300SX)
switch (GET_MODE (x))
{
case E_QImode:
case E_HImode:
*total = COSTS_N_INSNS (!speed ? 4 : 10);
return false;
case E_SImode:
*total = COSTS_N_INSNS (!speed ? 4 : 18);
return false;
default:
break;
}
*total = COSTS_N_INSNS (12);
return true;
case MULT:
if (TARGET_H8300SX)
switch (GET_MODE (x))
{
case E_QImode:
case E_HImode:
*total = COSTS_N_INSNS (2);
return false;
case E_SImode:
*total = COSTS_N_INSNS (5);
return false;
default:
break;
}
*total = COSTS_N_INSNS (4);
return true;
case ASHIFT:
case ASHIFTRT:
case LSHIFTRT:
if (h8sx_binary_shift_operator (x, VOIDmode))
{
*total = COSTS_N_INSNS (2);
return false;
}
else if (h8sx_unary_shift_operator (x, VOIDmode))
{
*total = COSTS_N_INSNS (1);
return false;
}
*total = COSTS_N_INSNS (h8300_shift_costs (x));
return true;
case ROTATE:
case ROTATERT:
if (GET_MODE (x) == HImode)
*total = 2;
else
*total = 8;
return true;
default:
*total = COSTS_N_INSNS (1);
return false;
}
}

static const char *
cond_string (enum rtx_code code)
{
switch (code)
{
case NE:
return "ne";
case EQ:
return "eq";
case GE:
return "ge";
case GT:
return "gt";
case LE:
return "le";
case LT:
return "lt";
case GEU:
return "hs";
case GTU:
return "hi";
case LEU:
return "ls";
case LTU:
return "lo";
default:
gcc_unreachable ();
}
}
static void
h8300_print_operand (FILE *file, rtx x, int code)
{
static int bitint;
switch (code)
{
case 'C':
if (h8300_constant_length (x) == 2)
fprintf (file, ":16");
else
fprintf (file, ":32");
return;
case 'E':
switch (GET_CODE (x))
{
case REG:
fprintf (file, "%sl", names_big[REGNO (x)]);
break;
case CONST_INT:
fprintf (file, "#%ld", (-INTVAL (x)) & 0xff);
break;
default:
gcc_unreachable ();
}
break;
case 'F':
switch (GET_CODE (x))
{
case REG:
fprintf (file, "%sh", names_big[REGNO (x)]);
break;
case CONST_INT:
fprintf (file, "#%ld", ((-INTVAL (x)) & 0xff00) >> 8);
break;
default:
gcc_unreachable ();
}
break;
case 'G':
gcc_assert (GET_CODE (x) == CONST_INT);
fprintf (file, "#%ld", 0xff & (-INTVAL (x)));
break;
case 'S':
if (GET_CODE (x) == REG)
fprintf (file, "%s", names_extended[REGNO (x)]);
else
goto def;
break;
case 'T':
if (GET_CODE (x) == REG)
fprintf (file, "%s", names_big[REGNO (x)]);
else
goto def;
break;
case 'V':
bitint = (INTVAL (x) & 0xffff);
if ((exact_log2 ((bitint >> 8) & 0xff)) == -1)
bitint = exact_log2 (bitint & 0xff);
else
bitint = exact_log2 ((bitint >> 8) & 0xff);	      
gcc_assert (bitint >= 0);
fprintf (file, "#%d", bitint);
break;
case 'W':
bitint = ((~INTVAL (x)) & 0xffff);
if ((exact_log2 ((bitint >> 8) & 0xff)) == -1 )
bitint = exact_log2 (bitint & 0xff);
else
bitint = (exact_log2 ((bitint >> 8) & 0xff));      
gcc_assert (bitint >= 0);
fprintf (file, "#%d", bitint);
break;
case 'R':
case 'X':
if (GET_CODE (x) == REG)
fprintf (file, "%s", byte_reg (x, 0));
else
goto def;
break;
case 'Y':
gcc_assert (bitint >= 0);
if (GET_CODE (x) == REG)
fprintf (file, "%s%c", names_big[REGNO (x)], bitint > 7 ? 'h' : 'l');
else
h8300_print_operand (file, x, 'R');
bitint = -1;
break;
case 'Z':
bitint = INTVAL (x);
fprintf (file, "#%d", bitint & 7);
break;
case 'c':
switch (GET_CODE (x))
{
case IOR:
fprintf (file, "or");
break;
case XOR:
fprintf (file, "xor");
break;
case AND:
fprintf (file, "and");
break;
default:
break;
}
break;
case 'e':
switch (GET_CODE (x))
{
case REG:
if (TARGET_H8300)
fprintf (file, "%s", names_big[REGNO (x)]);
else
fprintf (file, "%s", names_upper_extended[REGNO (x)]);
break;
case MEM:
h8300_print_operand (file, x, 0);
break;
case CONST_INT:
fprintf (file, "#%ld", ((INTVAL (x) >> 16) & 0xffff));
break;
case CONST_DOUBLE:
{
long val;
REAL_VALUE_TO_TARGET_SINGLE (*CONST_DOUBLE_REAL_VALUE (x), val);
fprintf (file, "#%ld", ((val >> 16) & 0xffff));
break;
}
default:
gcc_unreachable ();
break;
}
break;
case 'f':
switch (GET_CODE (x))
{
case REG:
if (TARGET_H8300)
fprintf (file, "%s", names_big[REGNO (x) + 1]);
else
fprintf (file, "%s", names_big[REGNO (x)]);
break;
case MEM:
x = adjust_address (x, HImode, 2);
h8300_print_operand (file, x, 0);
break;
case CONST_INT:
fprintf (file, "#%ld", INTVAL (x) & 0xffff);
break;
case CONST_DOUBLE:
{
long val;
REAL_VALUE_TO_TARGET_SINGLE (*CONST_DOUBLE_REAL_VALUE (x), val);
fprintf (file, "#%ld", (val & 0xffff));
break;
}
default:
gcc_unreachable ();
}
break;
case 'j':
fputs (cond_string (GET_CODE (x)), file);
break;
case 'k':
fputs (cond_string (reverse_condition (GET_CODE (x))), file);
break;
case 'm':
gcc_assert (GET_CODE (x) == CONST_INT);
switch (INTVAL (x))
{
case 1:
fputs (".b", file);
break;
case 2:
fputs (".w", file);
break;
case 4:
fputs (".l", file);
break;
default:
gcc_unreachable ();
}
break;
case 'o':
h8300_print_operand_address (file, VOIDmode, x);
break;
case 's':
if (GET_CODE (x) == CONST_INT)
fprintf (file, "#%ld", (INTVAL (x)) & 0xff);
else
fprintf (file, "%s", byte_reg (x, 0));
break;
case 't':
if (GET_CODE (x) == CONST_INT)
fprintf (file, "#%ld", (INTVAL (x) >> 8) & 0xff);
else
fprintf (file, "%s", byte_reg (x, 1));
break;
case 'w':
if (GET_CODE (x) == CONST_INT)
fprintf (file, "#%ld", INTVAL (x) & 0xff);
else
fprintf (file, "%s",
byte_reg (x, TARGET_H8300 ? 2 : 0));
break;
case 'x':
if (GET_CODE (x) == CONST_INT)
fprintf (file, "#%ld", (INTVAL (x) >> 8) & 0xff);
else
fprintf (file, "%s",
byte_reg (x, TARGET_H8300 ? 3 : 1));
break;
case 'y':
if (GET_CODE (x) == CONST_INT)
fprintf (file, "#%ld", (INTVAL (x) >> 16) & 0xff);
else
fprintf (file, "%s", byte_reg (x, 0));
break;
case 'z':
if (GET_CODE (x) == CONST_INT)
fprintf (file, "#%ld", (INTVAL (x) >> 24) & 0xff);
else
fprintf (file, "%s", byte_reg (x, 1));
break;
default:
def:
switch (GET_CODE (x))
{
case REG:
switch (GET_MODE (x))
{
case E_QImode:
#if 0 
fprintf (file, "%s", byte_reg (x, 0));
#else 
fprintf (file, "%s", names_big[REGNO (x)]);
#endif
break;
case E_HImode:
fprintf (file, "%s", names_big[REGNO (x)]);
break;
case E_SImode:
case E_SFmode:
fprintf (file, "%s", names_extended[REGNO (x)]);
break;
default:
gcc_unreachable ();
}
break;
case MEM:
{
rtx addr = XEXP (x, 0);
fprintf (file, "@");
output_address (GET_MODE (x), addr);
if (CONSTANT_P (addr))
switch (code)
{
case 'R':
if (h8300_eightbit_constant_address_p (addr))
{
fprintf (file, ":8");
break;
}
case 'X':
case 'T':
case 'S':
if (h8300_constant_length (addr) == 2)
fprintf (file, ":16");
else
fprintf (file, ":32");
break;
default:
break;
}
}
break;
case CONST_INT:
case SYMBOL_REF:
case CONST:
case LABEL_REF:
fprintf (file, "#");
h8300_print_operand_address (file, VOIDmode, x);
break;
case CONST_DOUBLE:
{
long val;
REAL_VALUE_TO_TARGET_SINGLE (*CONST_DOUBLE_REAL_VALUE (x), val);
fprintf (file, "#%ld", val);
break;
}
default:
break;
}
}
}
static bool
h8300_print_operand_punct_valid_p (unsigned char code)
{
return (code == '#');
}
static void
h8300_print_operand_address (FILE *file, machine_mode mode, rtx addr)
{
rtx index;
int size;
switch (GET_CODE (addr))
{
case REG:
fprintf (file, "%s", h8_reg_names[REGNO (addr)]);
break;
case PRE_DEC:
fprintf (file, "-%s", h8_reg_names[REGNO (XEXP (addr, 0))]);
break;
case POST_INC:
fprintf (file, "%s+", h8_reg_names[REGNO (XEXP (addr, 0))]);
break;
case PRE_INC:
fprintf (file, "+%s", h8_reg_names[REGNO (XEXP (addr, 0))]);
break;
case POST_DEC:
fprintf (file, "%s-", h8_reg_names[REGNO (XEXP (addr, 0))]);
break;
case PLUS:
fprintf (file, "(");
index = h8300_get_index (XEXP (addr, 0), VOIDmode, &size);
if (GET_CODE (index) == REG)
{
h8300_print_operand_address (file, mode, XEXP (addr, 1));
fprintf (file, ",");
switch (size)
{
case 0:
h8300_print_operand_address (file, mode, index);
break;
case 1:
h8300_print_operand (file, index, 'X');
fputs (".b", file);
break;
case 2:
h8300_print_operand (file, index, 'T');
fputs (".w", file);
break;
case 4:
h8300_print_operand (file, index, 'S');
fputs (".l", file);
break;
}
}
else
{
h8300_print_operand_address (file, mode, XEXP (addr, 0));
fprintf (file, "+");
h8300_print_operand_address (file, mode, XEXP (addr, 1));
}
fprintf (file, ")");
break;
case CONST_INT:
{
int n = INTVAL (addr);
if (TARGET_H8300)
n = (int) (short) n;
fprintf (file, "%d", n);
break;
}
default:
output_addr_const (file, addr);
break;
}
}

void
final_prescan_insn (rtx_insn *insn, rtx *operand ATTRIBUTE_UNUSED,
int num_operands ATTRIBUTE_UNUSED)
{
static int last_insn_address = 0;
const int uid = INSN_UID (insn);
if (TARGET_ADDRESSES)
{
fprintf (asm_out_file, "; 0x%x %d\n", INSN_ADDRESSES (uid),
INSN_ADDRESSES (uid) - last_insn_address);
last_insn_address = INSN_ADDRESSES (uid);
}
}
int
h8300_expand_movsi (rtx operands[])
{
rtx src = operands[1];
rtx dst = operands[0];
if (!reload_in_progress && !reload_completed)
{
if (!register_operand (dst, GET_MODE (dst)))
{
rtx tmp = gen_reg_rtx (GET_MODE (dst));
emit_move_insn (tmp, src);
operands[1] = tmp;
}
}
return 0;
}
static bool
h8300_can_eliminate (const int from ATTRIBUTE_UNUSED, const int to)
{
return (to == STACK_POINTER_REGNUM ? ! frame_pointer_needed : true);
}
static void
h8300_conditional_register_usage (void)
{
if (!TARGET_MAC)
fixed_regs[MAC_REG] = call_used_regs[MAC_REG] = 1;
}
int
h8300_initial_elimination_offset (int from, int to)
{
int pc_size = POINTER_SIZE / BITS_PER_UNIT;
int fp_size = frame_pointer_needed * UNITS_PER_WORD;
int saved_regs_size = 0;
int frame_size = round_frame_size (get_frame_size ());
int regno;
for (regno = 0; regno <= HARD_FRAME_POINTER_REGNUM; regno++)
if (WORD_REG_USED (regno))
saved_regs_size += UNITS_PER_WORD;
saved_regs_size -= fp_size;
switch (to)
{
case HARD_FRAME_POINTER_REGNUM:
switch (from)
{
case ARG_POINTER_REGNUM:
return pc_size + fp_size;
case RETURN_ADDRESS_POINTER_REGNUM:
return fp_size;
case FRAME_POINTER_REGNUM:
return -saved_regs_size;
default:
gcc_unreachable ();
}
break;
case STACK_POINTER_REGNUM:
switch (from)
{
case ARG_POINTER_REGNUM:
return pc_size + saved_regs_size + frame_size;
case RETURN_ADDRESS_POINTER_REGNUM:
return saved_regs_size + frame_size;
case FRAME_POINTER_REGNUM:
return frame_size;
default:
gcc_unreachable ();
}
break;
default:
gcc_unreachable ();
}
gcc_unreachable ();
}
rtx
h8300_return_addr_rtx (int count, rtx frame)
{
rtx ret;
if (count == 0)
ret = gen_rtx_MEM (Pmode,
gen_rtx_REG (Pmode, RETURN_ADDRESS_POINTER_REGNUM));
else if (flag_omit_frame_pointer)
return (rtx) 0;
else
ret = gen_rtx_MEM (Pmode,
memory_address (Pmode,
plus_constant (Pmode, frame,
UNITS_PER_WORD)));
set_mem_alias_set (ret, get_frame_alias_set ());
return ret;
}
void
notice_update_cc (rtx body, rtx_insn *insn)
{
rtx set;
switch (get_attr_cc (insn))
{
case CC_NONE:
break;
case CC_NONE_0HIT:
if (cc_status.value1 != 0
&& reg_overlap_mentioned_p (recog_data.operand[0], cc_status.value1))
cc_status.value1 = 0;
if (cc_status.value2 != 0
&& reg_overlap_mentioned_p (recog_data.operand[0], cc_status.value2))
cc_status.value2 = 0;
break;
case CC_SET_ZN:
CC_STATUS_INIT;
cc_status.flags |= CC_OVERFLOW_UNUSABLE | CC_NO_CARRY;
set = single_set (insn);
cc_status.value1 = SET_SRC (set);
if (SET_DEST (set) != cc0_rtx)
cc_status.value2 = SET_DEST (set);
break;
case CC_SET_ZNV:
CC_STATUS_INIT;
cc_status.flags |= CC_NO_CARRY;
set = single_set (insn);
cc_status.value1 = SET_SRC (set);
if (SET_DEST (set) != cc0_rtx)
{
if (GET_CODE (SET_DEST (set)) == STRICT_LOW_PART)
cc_status.value2 = XEXP (SET_DEST (set), 0);
else
cc_status.value2 = SET_DEST (set);
}
break;
case CC_COMPARE:
CC_STATUS_INIT;
cc_status.value1 = SET_SRC (body);
break;
case CC_CLOBBER:
CC_STATUS_INIT;
break;
}
}

static rtx
h8300_get_index (rtx x, machine_mode mode, int *size)
{
int dummy, factor;
if (size == 0)
size = &dummy;
factor = (mode == VOIDmode ? 0 : GET_MODE_SIZE (mode));
if (TARGET_H8300SX
&& factor <= 4
&& (mode == VOIDmode
|| GET_MODE_CLASS (mode) == MODE_INT
|| GET_MODE_CLASS (mode) == MODE_FLOAT))
{
if (factor <= 1 && GET_CODE (x) == ZERO_EXTEND)
{
*size = GET_MODE_SIZE (GET_MODE (XEXP (x, 0)));
return XEXP (x, 0);
}
else
{
rtx index;
if (GET_CODE (x) == AND
&& GET_CODE (XEXP (x, 1)) == CONST_INT
&& (factor == 0
|| INTVAL (XEXP (x, 1)) == 0xff * factor
|| INTVAL (XEXP (x, 1)) == 0xffff * factor))
{
index = XEXP (x, 0);
*size = (INTVAL (XEXP (x, 1)) >= 0xffff ? 2 : 1);
}
else
{
index = x;
*size = 4;
}
if (GET_CODE (index) == MULT
&& GET_CODE (XEXP (index, 1)) == CONST_INT
&& (factor == 0 || factor == INTVAL (XEXP (index, 1))))
return XEXP (index, 0);
}
}
*size = 0;
return x;
}

static bool
h8300_mode_dependent_address_p (const_rtx addr,
addr_space_t as ATTRIBUTE_UNUSED)
{
if (GET_CODE (addr) == PLUS
&& h8300_get_index (XEXP (addr, 0), VOIDmode, 0) != XEXP (addr, 0))
return true;
return false;
}

static const h8300_length_table addb_length_table =
{
{  2,   2,   4,   4,   4  }, 
{  4,   4,   4,   4,   6  }, 
{  4,   4,   4,   4,   6  }, 
{  6,   4,   4,   4,   6  }  
};
static const h8300_length_table addw_length_table =
{
{  2,   2,   4,   4,   4  }, 
{  4,   4,   4,   4,   6  }, 
{  4,   4,   4,   4,   6  }, 
{  4,   4,   4,   4,   6  }  
};
static const h8300_length_table addl_length_table =
{
{  2,   2,   4,   4,   4  }, 
{  4,   4,   6,   6,   6  }, 
{  4,   4,   6,   6,   6  }, 
{  4,   4,   6,   6,   6  }  
};
#define logicb_length_table addb_length_table
#define logicw_length_table addw_length_table
static const h8300_length_table logicl_length_table =
{
{  2,   4,   4,   4,   4  }, 
{  4,   4,   6,   6,   6  }, 
{  4,   4,   6,   6,   6  }, 
{  4,   4,   6,   6,   6  }  
};
static const h8300_length_table movb_length_table =
{
{  2,   2,   2,   2,   4  }, 
{  4,   2,   4,   4,   4  }, 
{  4,   2,   4,   4,   4  }, 
{  4,   4,   4,   4,   4  }  
};
#define movw_length_table movb_length_table
static const h8300_length_table movl_length_table =
{
{  2,   2,   4,   4,   4  }, 
{  4,   4,   4,   4,   4  }, 
{  4,   4,   4,   4,   4  }, 
{  4,   4,   4,   4,   4  }  
};
static unsigned int
h8300_constant_length (rtx constant)
{
if (GET_CODE (constant) == CONST_INT
&& IN_RANGE (INTVAL (constant), -0x8000, 0x7fff))
return 2;
if (Pmode == HImode || h8300_tiny_constant_address_p (constant))
return 2;
return 4;
}
static unsigned int
h8300_displacement_length (rtx addr, int size)
{
rtx offset;
offset = XEXP (addr, 1);
if (register_operand (XEXP (addr, 0), VOIDmode)
&& GET_CODE (offset) == CONST_INT
&& (INTVAL (offset) == size
|| INTVAL (offset) == size * 2
|| INTVAL (offset) == size * 3))
return 0;
return h8300_constant_length (offset);
}
static unsigned int
h8300_classify_operand (rtx op, int size, enum h8300_operand_class *opclass)
{
enum h8300_operand_class dummy;
if (opclass == 0)
opclass = &dummy;
if (CONSTANT_P (op))
{
*opclass = H8OP_IMMEDIATE;
if (size == 1)
return 0;
if (TARGET_H8300SX
&& size == 4
&& GET_CODE (op) == CONST_INT
&& IN_RANGE (INTVAL (op), 0, 0xffff))
return 2;
return size;
}
else if (GET_CODE (op) == MEM)
{
op = XEXP (op, 0);
if (CONSTANT_P (op))
{
*opclass = H8OP_MEM_ABSOLUTE;
return h8300_constant_length (op);
}
else if (GET_CODE (op) == PLUS && CONSTANT_P (XEXP (op, 1)))
{
*opclass = H8OP_MEM_COMPLEX;
return h8300_displacement_length (op, size);
}
else if (GET_RTX_CLASS (GET_CODE (op)) == RTX_AUTOINC)
{
*opclass = H8OP_MEM_COMPLEX;
return 0;
}
else if (register_operand (op, VOIDmode))
{
*opclass = H8OP_MEM_BASE;
return 0;
}
}
gcc_assert (register_operand (op, VOIDmode));
*opclass = H8OP_REGISTER;
return 0;
}
static unsigned int
h8300_length_from_table (rtx op1, rtx op2, const h8300_length_table *table)
{
enum h8300_operand_class op1_class, op2_class;
unsigned int size, immediate_length;
size = GET_MODE_SIZE (GET_MODE (op1));
immediate_length = (h8300_classify_operand (op1, size, &op1_class)
+ h8300_classify_operand (op2, size, &op2_class));
return immediate_length + (*table)[op1_class - 1][op2_class];
}
unsigned int
h8300_unary_length (rtx op)
{
enum h8300_operand_class opclass;
unsigned int size, operand_length;
size = GET_MODE_SIZE (GET_MODE (op));
operand_length = h8300_classify_operand (op, size, &opclass);
switch (opclass)
{
case H8OP_REGISTER:
return 2;
case H8OP_MEM_BASE:
return (size == 4 ? 6 : 4);
case H8OP_MEM_ABSOLUTE:
return operand_length + (size == 4 ? 6 : 4);
case H8OP_MEM_COMPLEX:
return operand_length + 6;
default:
gcc_unreachable ();
}
}
static unsigned int
h8300_short_immediate_length (rtx op)
{
enum h8300_operand_class opclass;
unsigned int size, operand_length;
size = GET_MODE_SIZE (GET_MODE (op));
operand_length = h8300_classify_operand (op, size, &opclass);
switch (opclass)
{
case H8OP_REGISTER:
return 2;
case H8OP_MEM_BASE:
case H8OP_MEM_ABSOLUTE:
case H8OP_MEM_COMPLEX:
return 4 + operand_length;
default:
gcc_unreachable ();
}
}
static unsigned int
h8300_bitfield_length (rtx op, rtx op2)
{
enum h8300_operand_class opclass;
unsigned int size, operand_length;
if (GET_CODE (op) == REG)
op = op2;
gcc_assert (GET_CODE (op) != REG);
size = GET_MODE_SIZE (GET_MODE (op));
operand_length = h8300_classify_operand (op, size, &opclass);
switch (opclass)
{
case H8OP_MEM_BASE:
case H8OP_MEM_ABSOLUTE:
case H8OP_MEM_COMPLEX:
return 4 + operand_length;
default:
gcc_unreachable ();
}
}
static unsigned int
h8300_binary_length (rtx_insn *insn, const h8300_length_table *table)
{
rtx set;
set = single_set (insn);
gcc_assert (set);
if (BINARY_P (SET_SRC (set)))
return h8300_length_from_table (XEXP (SET_SRC (set), 0),
XEXP (SET_SRC (set), 1), table);
else
{
gcc_assert (GET_RTX_CLASS (GET_CODE (SET_SRC (set))) == RTX_TERNARY);
return h8300_length_from_table (XEXP (XEXP (SET_SRC (set), 1), 0),
XEXP (XEXP (SET_SRC (set), 1), 1),
table);
}
}
static bool
h8300_short_move_mem_p (rtx op, enum rtx_code inc_code)
{
rtx addr;
unsigned int size;
if (GET_CODE (op) != MEM)
return false;
addr = XEXP (op, 0);
size = GET_MODE_SIZE (GET_MODE (op));
if (size != 1 && size != 2)
return false;
return (GET_CODE (addr) == inc_code
|| (GET_CODE (addr) == PLUS
&& GET_CODE (XEXP (addr, 0)) == REG
&& h8300_displacement_length (addr, size) == 2));
}
static unsigned int
h8300_move_length (rtx *operands, const h8300_length_table *table)
{
unsigned int size;
size = h8300_length_from_table (operands[0], operands[1], table);
if (REG_P (operands[0]) && h8300_short_move_mem_p (operands[1], POST_INC))
size -= 2;
if (REG_P (operands[1]) && h8300_short_move_mem_p (operands[0], PRE_DEC))
size -= 2;
return size;
}
static unsigned int
h8300_mova_length (rtx dest, rtx src, rtx offset)
{
unsigned int size;
size = (2
+ h8300_constant_length (offset)
+ h8300_classify_operand (src, GET_MODE_SIZE (GET_MODE (src)), 0));
if (!REG_P (dest) || !REG_P (src) || REGNO (src) != REGNO (dest))
size += 2;
return size;
}
unsigned int
h8300_insn_length_from_table (rtx_insn *insn, rtx * operands)
{
switch (get_attr_length_table (insn))
{
case LENGTH_TABLE_NONE:
gcc_unreachable ();
case LENGTH_TABLE_ADDB:
return h8300_binary_length (insn, &addb_length_table);
case LENGTH_TABLE_ADDW:
return h8300_binary_length (insn, &addw_length_table);
case LENGTH_TABLE_ADDL:
return h8300_binary_length (insn, &addl_length_table);
case LENGTH_TABLE_LOGICB:
return h8300_binary_length (insn, &logicb_length_table);
case LENGTH_TABLE_MOVB:
return h8300_move_length (operands, &movb_length_table);
case LENGTH_TABLE_MOVW:
return h8300_move_length (operands, &movw_length_table);
case LENGTH_TABLE_MOVL:
return h8300_move_length (operands, &movl_length_table);
case LENGTH_TABLE_MOVA:
return h8300_mova_length (operands[0], operands[1], operands[2]);
case LENGTH_TABLE_MOVA_ZERO:
return h8300_mova_length (operands[0], operands[1], const0_rtx);
case LENGTH_TABLE_UNARY:
return h8300_unary_length (operands[0]);
case LENGTH_TABLE_MOV_IMM4:
return 2 + h8300_classify_operand (operands[0], 0, 0);
case LENGTH_TABLE_SHORT_IMMEDIATE:
return h8300_short_immediate_length (operands[0]);
case LENGTH_TABLE_BITFIELD:
return h8300_bitfield_length (operands[0], operands[1]);
case LENGTH_TABLE_BITBRANCH:
return h8300_bitfield_length (operands[1], operands[2]) - 2;
default:
gcc_unreachable ();
}
}
bool
h8sx_mergeable_memrefs_p (rtx lhs, rtx rhs)
{
if (GET_CODE (rhs) == MEM && GET_CODE (lhs) == MEM)
{
rhs = XEXP (rhs, 0);
lhs = XEXP (lhs, 0);
if (GET_CODE (rhs) == PRE_INC || GET_CODE (rhs) == PRE_DEC)
return rtx_equal_p (XEXP (rhs, 0), lhs);
if (GET_CODE (lhs) == POST_INC || GET_CODE (lhs) == POST_DEC)
return rtx_equal_p (rhs, XEXP (lhs, 0));
if (rtx_equal_p (rhs, lhs))
return true;
}
return false;
}
bool
h8300_operands_match_p (rtx *operands)
{
if (register_operand (operands[0], VOIDmode)
&& register_operand (operands[1], VOIDmode))
return true;
if (h8sx_mergeable_memrefs_p (operands[0], operands[1]))
return true;
return false;
}

bool
h8sx_emit_movmd (rtx dest, rtx src, rtx length,
HOST_WIDE_INT alignment)
{
if (!flag_omit_frame_pointer && optimize_size)
return false;
if (GET_CODE (length) == CONST_INT)
{
rtx dest_reg, src_reg, first_dest, first_src;
HOST_WIDE_INT n;
int factor;
factor = (alignment >= 2 ? 4 : 1);
n = INTVAL (length);
if (n <= 0 || n / factor > 65536)
return false;
dest_reg = copy_addr_to_reg (XEXP (dest, 0));
src_reg = copy_addr_to_reg (XEXP (src, 0));
first_dest = replace_equiv_address (dest, dest_reg);
first_src = replace_equiv_address (src, src_reg);
set_mem_size (first_dest, n & -factor);
set_mem_size (first_src, n & -factor);
length = copy_to_mode_reg (HImode, gen_int_mode (n / factor, HImode));
emit_insn (gen_movmd (first_dest, first_src, length, GEN_INT (factor)));
if ((n & -factor) != n)
{
dest = adjust_address (dest, BLKmode, n & -factor);
src = adjust_address (src, BLKmode, n & -factor);
dest = replace_equiv_address (dest, dest_reg);
src = replace_equiv_address (src, src_reg);
if (n & 2)
emit_move_insn (adjust_address (dest, HImode, 0),
adjust_address (src, HImode, 0));
if (n & 1)
emit_move_insn (adjust_address (dest, QImode, n & 2),
adjust_address (src, QImode, n & 2));
}
return true;
}
return false;
}
void
h8300_swap_into_er6 (rtx addr)
{
rtx insn = push (HARD_FRAME_POINTER_REGNUM, false);
if (frame_pointer_needed)
add_reg_note (insn, REG_CFA_DEF_CFA,
plus_constant (Pmode, gen_rtx_MEM (Pmode, stack_pointer_rtx),
2 * UNITS_PER_WORD));
else
add_reg_note (insn, REG_CFA_ADJUST_CFA,
gen_rtx_SET (stack_pointer_rtx,
plus_constant (Pmode, stack_pointer_rtx, 4)));
emit_move_insn (hard_frame_pointer_rtx, addr);
if (REGNO (addr) == SP_REG)
emit_move_insn (hard_frame_pointer_rtx,
plus_constant (Pmode, hard_frame_pointer_rtx,
GET_MODE_SIZE (word_mode)));
}
void
h8300_swap_out_of_er6 (rtx addr)
{
rtx insn;
if (REGNO (addr) != SP_REG)
emit_move_insn (addr, hard_frame_pointer_rtx);
insn = pop (HARD_FRAME_POINTER_REGNUM);
if (frame_pointer_needed)
add_reg_note (insn, REG_CFA_DEF_CFA,
plus_constant (Pmode, hard_frame_pointer_rtx,
2 * UNITS_PER_WORD));
else
add_reg_note (insn, REG_CFA_ADJUST_CFA,
gen_rtx_SET (stack_pointer_rtx,
plus_constant (Pmode, stack_pointer_rtx, -4)));
}

unsigned int
compute_mov_length (rtx *operands)
{
machine_mode mode = GET_MODE (operands[0]);
rtx dest = operands[0];
rtx src = operands[1];
rtx addr;
if (GET_CODE (src) == MEM)
addr = XEXP (src, 0);
else if (GET_CODE (dest) == MEM)
addr = XEXP (dest, 0);
else
addr = NULL_RTX;
if (TARGET_H8300)
{
unsigned int base_length;
switch (mode)
{
case E_QImode:
if (addr == NULL_RTX)
return 2;
if (h8300_eightbit_constant_address_p (addr))
return 2;
base_length = 4;
break;
case E_HImode:
if (addr == NULL_RTX)
{
if (REG_P (src))
return 2;
if (src == const0_rtx)
return 2;
return 4;
}
base_length = 4;
break;
case E_SImode:
if (addr == NULL_RTX)
{
if (REG_P (src))
return 4;
if (GET_CODE (src) == CONST_INT)
{
if (src == const0_rtx)
return 4;
if ((INTVAL (src) & 0xffff) == 0)
return 6;
if ((INTVAL (src) & 0xffff) == 0)
return 6;
if ((INTVAL (src) & 0xffff)
== ((INTVAL (src) >> 16) & 0xffff))
return 6;
}
return 8;
}
base_length = 8;
break;
case E_SFmode:
if (addr == NULL_RTX)
{
if (REG_P (src))
return 4;
if (satisfies_constraint_G (src))
return 4;
return 8;
}
base_length = 8;
break;
default:
gcc_unreachable ();
}
if (GET_CODE (addr) == PRE_DEC
|| GET_CODE (addr) == POST_INC)
{
if (mode == QImode || mode == HImode)
return base_length - 2;
else
return base_length - 4;
}
if (GET_CODE (addr) == REG)
return base_length - 2;
return base_length;
}
else
{
unsigned int base_length;
switch (mode)
{
case E_QImode:
if (addr == NULL_RTX)
return 2;
if (h8300_eightbit_constant_address_p (addr))
return 2;
base_length = 8;
break;
case E_HImode:
if (addr == NULL_RTX)
{
if (REG_P (src))
return 2;
if (src == const0_rtx)
return 2;
return 4;
}
base_length = 8;
break;
case E_SImode:
if (addr == NULL_RTX)
{
if (REG_P (src))
{
if (REGNO (src) == MAC_REG || REGNO (dest) == MAC_REG)
return 4;
else
return 2;
}
if (GET_CODE (src) == CONST_INT)
{
int val = INTVAL (src);
if (val == 0)
return 2;
if (val == (val & 0x00ff) || val == (val & 0xff00))
return 4;
switch (val & 0xffffffff)
{
case 0xffffffff:
case 0xfffffffe:
case 0xfffffffc:
case 0x0000ffff:
case 0x0000fffe:
case 0xffff0000:
case 0xfffe0000:
case 0x00010000:
case 0x00020000:
return 4;
}
}
return 6;
}
base_length = 10;
break;
case E_SFmode:
if (addr == NULL_RTX)
{
if (REG_P (src))
return 2;
if (satisfies_constraint_G (src))
return 2;
return 6;
}
base_length = 10;
break;
default:
gcc_unreachable ();
}
if (GET_CODE (addr) == PRE_DEC
|| GET_CODE (addr) == POST_INC)
return base_length - 6;
if (GET_CODE (addr) == REG)
return base_length - 6;
if (GET_CODE (addr) == PLUS
&& GET_CODE (XEXP (addr, 0)) == REG
&& GET_CODE (XEXP (addr, 1)) == CONST_INT
&& INTVAL (XEXP (addr, 1)) > -32768
&& INTVAL (XEXP (addr, 1)) < 32767)
return base_length - 4;
if (h8300_tiny_constant_address_p (addr))
return base_length - 4;
if (CONSTANT_P (addr))
return base_length - 2;
return base_length;
}
}

const char *
output_plussi (rtx *operands)
{
machine_mode mode = GET_MODE (operands[0]);
gcc_assert (mode == SImode);
if (TARGET_H8300)
{
if (GET_CODE (operands[2]) == REG)
return "add.w\t%f2,%f0\n\taddx\t%y2,%y0\n\taddx\t%z2,%z0";
if (GET_CODE (operands[2]) == CONST_INT)
{
HOST_WIDE_INT n = INTVAL (operands[2]);
if ((n & 0xffffff) == 0)
return "add\t%z2,%z0";
if ((n & 0xffff) == 0)
return "add\t%y2,%y0\n\taddx\t%z2,%z0";
if ((n & 0xff) == 0)
return "add\t%x2,%x0\n\taddx\t%y2,%y0\n\taddx\t%z2,%z0";
}
return "add\t%w2,%w0\n\taddx\t%x2,%x0\n\taddx\t%y2,%y0\n\taddx\t%z2,%z0";
}
else
{
if (GET_CODE (operands[2]) == CONST_INT
&& register_operand (operands[1], VOIDmode))
{
HOST_WIDE_INT intval = INTVAL (operands[2]);
if (TARGET_H8300SX && (intval >= 1 && intval <= 7))
return "add.l\t%S2,%S0";
if (TARGET_H8300SX && (intval >= -7 && intval <= -1))
return "sub.l\t%G2,%S0";
switch ((unsigned int) intval & 0xffffffff)
{
case 0x00000001:
case 0x00000002:
case 0x00000004:
return "adds\t%2,%S0";
case 0xffffffff:
case 0xfffffffe:
case 0xfffffffc:
return "subs\t%G2,%S0";
case 0x00010000:
case 0x00020000:
operands[2] = GEN_INT (intval >> 16);
return "inc.w\t%2,%e0";
case 0xffff0000:
case 0xfffe0000:
operands[2] = GEN_INT (intval >> 16);
return "dec.w\t%G2,%e0";
}
if ((intval & 0xffff) == 0)
{
operands[2] = GEN_INT (intval >> 16);
return "add.w\t%2,%e0";
}
}
if (GET_CODE (operands[2]) == CONST_INT && INTVAL (operands[2]) < 0)
{
operands[2] = GEN_INT (-INTVAL (operands[2]));
return "sub.l\t%S2,%S0";
}
return "add.l\t%S2,%S0";
}
}
unsigned int
compute_plussi_length (rtx *operands)
{
machine_mode mode = GET_MODE (operands[0]);
gcc_assert (mode == SImode);
if (TARGET_H8300)
{
if (GET_CODE (operands[2]) == REG)
return 6;
if (GET_CODE (operands[2]) == CONST_INT)
{
HOST_WIDE_INT n = INTVAL (operands[2]);
if ((n & 0xffffff) == 0)
return 2;
if ((n & 0xffff) == 0)
return 4;
if ((n & 0xff) == 0)
return 6;
}
return 8;
}
else
{
if (GET_CODE (operands[2]) == CONST_INT
&& register_operand (operands[1], VOIDmode))
{
HOST_WIDE_INT intval = INTVAL (operands[2]);
if (TARGET_H8300SX && (intval >= 1 && intval <= 7))
return 2;
if (TARGET_H8300SX && (intval >= -7 && intval <= -1))
return 2;
switch ((unsigned int) intval & 0xffffffff)
{
case 0x00000001:
case 0x00000002:
case 0x00000004:
return 2;
case 0xffffffff:
case 0xfffffffe:
case 0xfffffffc:
return 2;
case 0x00010000:
case 0x00020000:
return 2;
case 0xffff0000:
case 0xfffe0000:
return 2;
}
if ((intval & 0xffff) == 0)
return 4;
}
if (GET_CODE (operands[2]) == CONST_INT && INTVAL (operands[2]) < 0)
return h8300_length_from_table (operands[0],
GEN_INT (-INTVAL (operands[2])),
&addl_length_table);
else
return h8300_length_from_table (operands[0], operands[2],
&addl_length_table);
return 6;
}
}
enum attr_cc
compute_plussi_cc (rtx *operands)
{
machine_mode mode = GET_MODE (operands[0]);
gcc_assert (mode == SImode);
if (TARGET_H8300)
{
return CC_CLOBBER;
}
else
{
if (GET_CODE (operands[2]) == CONST_INT
&& register_operand (operands[1], VOIDmode))
{
HOST_WIDE_INT intval = INTVAL (operands[2]);
if (TARGET_H8300SX && (intval >= 1 && intval <= 7))
return CC_SET_ZN;
if (TARGET_H8300SX && (intval >= -7 && intval <= -1))
return CC_SET_ZN;
switch ((unsigned int) intval & 0xffffffff)
{
case 0x00000001:
case 0x00000002:
case 0x00000004:
return CC_NONE_0HIT;
case 0xffffffff:
case 0xfffffffe:
case 0xfffffffc:
return CC_NONE_0HIT;
case 0x00010000:
case 0x00020000:
return CC_CLOBBER;
case 0xffff0000:
case 0xfffe0000:
return CC_CLOBBER;
}
if ((intval & 0xffff) == 0)
return CC_CLOBBER;
}
return CC_SET_ZN;
}
}

const char *
output_logical_op (machine_mode mode, rtx *operands)
{
enum rtx_code code = GET_CODE (operands[3]);
const unsigned HOST_WIDE_INT intval =
(unsigned HOST_WIDE_INT) ((GET_CODE (operands[2]) == CONST_INT)
&& register_operand (operands[1], VOIDmode)
? INTVAL (operands[2]) : 0x55555555);
const unsigned HOST_WIDE_INT det = (code != AND) ? intval : ~intval;
const unsigned HOST_WIDE_INT b0 = (det >>  0) & 0xff;
const unsigned HOST_WIDE_INT b1 = (det >>  8) & 0xff;
const unsigned HOST_WIDE_INT b2 = (det >> 16) & 0xff;
const unsigned HOST_WIDE_INT b3 = (det >> 24) & 0xff;
const unsigned HOST_WIDE_INT w0 = (det >>  0) & 0xffff;
const unsigned HOST_WIDE_INT w1 = (det >> 16) & 0xffff;
int lower_half_easy_p = 0;
int upper_half_easy_p = 0;
const char *opname;
char insn_buf[100];
switch (code)
{
case AND:
opname = "and";
break;
case IOR:
opname = "or";
break;
case XOR:
opname = "xor";
break;
default:
gcc_unreachable ();
}
switch (mode)
{
case E_HImode:
if ((TARGET_H8300H || TARGET_H8300S)
&& b0 != 0
&& b1 != 0)
{
sprintf (insn_buf, "%s.w\t%%T2,%%T0", opname);
output_asm_insn (insn_buf, operands);
}
else
{
if (b0 != 0)
{
sprintf (insn_buf, "%s\t%%s2,%%s0", opname);
output_asm_insn (insn_buf, operands);
}
if (b1 != 0)
{
sprintf (insn_buf, "%s\t%%t2,%%t0", opname);
output_asm_insn (insn_buf, operands);
}
}
break;
case E_SImode:
if (TARGET_H8300H || TARGET_H8300S)
{
lower_half_easy_p = (b0 == 0
|| b1 == 0
|| (code != IOR && w0 == 0xffff));
upper_half_easy_p = ((code != IOR && w1 == 0xffff)
|| (code == AND && w1 == 0xff00));
}
if ((TARGET_H8300H || TARGET_H8300S)
&& w0 != 0 && w1 != 0
&& !(lower_half_easy_p && upper_half_easy_p)
&& !(code == IOR && w1 == 0xffff
&& (w0 & 0x8000) != 0 && lower_half_easy_p))
{
sprintf (insn_buf, "%s.l\t%%S2,%%S0", opname);
output_asm_insn (insn_buf, operands);
}
else
{
if (w0 == 0xffff
&& (TARGET_H8300 ? (code == AND) : (code != IOR)))
output_asm_insn ((code == AND)
? "sub.w\t%f0,%f0" : "not.w\t%f0",
operands);
else if ((TARGET_H8300H || TARGET_H8300S)
&& (b0 != 0)
&& (b1 != 0))
{
sprintf (insn_buf, "%s.w\t%%f2,%%f0", opname);
output_asm_insn (insn_buf, operands);
}
else
{
if (b0 != 0)
{
sprintf (insn_buf, "%s\t%%w2,%%w0", opname);
output_asm_insn (insn_buf, operands);
}
if (b1 != 0)
{
sprintf (insn_buf, "%s\t%%x2,%%x0", opname);
output_asm_insn (insn_buf, operands);
}
}
if ((w1 == 0xffff)
&& (TARGET_H8300 ? (code == AND) : (code != IOR)))
output_asm_insn ((code == AND)
? "sub.w\t%e0,%e0" : "not.w\t%e0",
operands);
else if ((TARGET_H8300H || TARGET_H8300S)
&& code == IOR
&& w1 == 0xffff
&& (w0 & 0x8000) != 0)
{
output_asm_insn ("exts.l\t%S0", operands);
}
else if ((TARGET_H8300H || TARGET_H8300S)
&& code == AND
&& w1 == 0xff00)
{
output_asm_insn ("extu.w\t%e0", operands);
}
else if (TARGET_H8300H || TARGET_H8300S)
{
if (w1 != 0)
{
sprintf (insn_buf, "%s.w\t%%e2,%%e0", opname);
output_asm_insn (insn_buf, operands);
}
}
else
{
if (b2 != 0)
{
sprintf (insn_buf, "%s\t%%y2,%%y0", opname);
output_asm_insn (insn_buf, operands);
}
if (b3 != 0)
{
sprintf (insn_buf, "%s\t%%z2,%%z0", opname);
output_asm_insn (insn_buf, operands);
}
}
}
break;
default:
gcc_unreachable ();
}
return "";
}
unsigned int
compute_logical_op_length (machine_mode mode, rtx *operands)
{
enum rtx_code code = GET_CODE (operands[3]);
const unsigned HOST_WIDE_INT intval =
(unsigned HOST_WIDE_INT) ((GET_CODE (operands[2]) == CONST_INT)
&& register_operand (operands[1], VOIDmode)
? INTVAL (operands[2]) : 0x55555555);
const unsigned HOST_WIDE_INT det = (code != AND) ? intval : ~intval;
const unsigned HOST_WIDE_INT b0 = (det >>  0) & 0xff;
const unsigned HOST_WIDE_INT b1 = (det >>  8) & 0xff;
const unsigned HOST_WIDE_INT b2 = (det >> 16) & 0xff;
const unsigned HOST_WIDE_INT b3 = (det >> 24) & 0xff;
const unsigned HOST_WIDE_INT w0 = (det >>  0) & 0xffff;
const unsigned HOST_WIDE_INT w1 = (det >> 16) & 0xffff;
int lower_half_easy_p = 0;
int upper_half_easy_p = 0;
unsigned int length = 0;
switch (mode)
{
case E_HImode:
if ((TARGET_H8300H || TARGET_H8300S)
&& b0 != 0
&& b1 != 0)
{
length = h8300_length_from_table (operands[1], operands[2],
&logicw_length_table);
}
else
{
if (b0 != 0)
length += 2;
if (b1 != 0)
length += 2;
}
break;
case E_SImode:
if (TARGET_H8300H || TARGET_H8300S)
{
lower_half_easy_p = (b0 == 0
|| b1 == 0
|| (code != IOR && w0 == 0xffff));
upper_half_easy_p = ((code != IOR && w1 == 0xffff)
|| (code == AND && w1 == 0xff00));
}
if ((TARGET_H8300H || TARGET_H8300S)
&& w0 != 0 && w1 != 0
&& !(lower_half_easy_p && upper_half_easy_p)
&& !(code == IOR && w1 == 0xffff
&& (w0 & 0x8000) != 0 && lower_half_easy_p))
{
length = h8300_length_from_table (operands[1], operands[2],
&logicl_length_table);
}
else
{
if (w0 == 0xffff
&& (TARGET_H8300 ? (code == AND) : (code != IOR)))
{
length += 2;
}
else if ((TARGET_H8300H || TARGET_H8300S)
&& (b0 != 0)
&& (b1 != 0))
{
length += 4;
}
else
{
if (b0 != 0)
length += 2;
if (b1 != 0)
length += 2;
}
if (w1 == 0xffff
&& (TARGET_H8300 ? (code == AND) : (code != IOR)))
{
length += 2;
}
else if ((TARGET_H8300H || TARGET_H8300S)
&& code == IOR
&& w1 == 0xffff
&& (w0 & 0x8000) != 0)
{
length += 2;
}
else if ((TARGET_H8300H || TARGET_H8300S)
&& code == AND
&& w1 == 0xff00)
{
length += 2;
}
else if (TARGET_H8300H || TARGET_H8300S)
{
if (w1 != 0)
length += 4;
}
else
{
if (b2 != 0)
length += 2;
if (b3 != 0)
length += 2;
}
}
break;
default:
gcc_unreachable ();
}
return length;
}
enum attr_cc
compute_logical_op_cc (machine_mode mode, rtx *operands)
{
enum rtx_code code = GET_CODE (operands[3]);
const unsigned HOST_WIDE_INT intval =
(unsigned HOST_WIDE_INT) ((GET_CODE (operands[2]) == CONST_INT)
&& register_operand (operands[1], VOIDmode)
? INTVAL (operands[2]) : 0x55555555);
const unsigned HOST_WIDE_INT det = (code != AND) ? intval : ~intval;
const unsigned HOST_WIDE_INT b0 = (det >>  0) & 0xff;
const unsigned HOST_WIDE_INT b1 = (det >>  8) & 0xff;
const unsigned HOST_WIDE_INT w0 = (det >>  0) & 0xffff;
const unsigned HOST_WIDE_INT w1 = (det >> 16) & 0xffff;
int lower_half_easy_p = 0;
int upper_half_easy_p = 0;
enum attr_cc cc = CC_CLOBBER;
switch (mode)
{
case E_HImode:
if ((TARGET_H8300H || TARGET_H8300S)
&& b0 != 0
&& b1 != 0)
{
cc = CC_SET_ZNV;
}
break;
case E_SImode:
if (TARGET_H8300H || TARGET_H8300S)
{
lower_half_easy_p = (b0 == 0
|| b1 == 0
|| (code != IOR && w0 == 0xffff));
upper_half_easy_p = ((code != IOR && w1 == 0xffff)
|| (code == AND && w1 == 0xff00));
}
if ((TARGET_H8300H || TARGET_H8300S)
&& w0 != 0 && w1 != 0
&& !(lower_half_easy_p && upper_half_easy_p)
&& !(code == IOR && w1 == 0xffff
&& (w0 & 0x8000) != 0 && lower_half_easy_p))
{
cc = CC_SET_ZNV;
}
else
{
if ((TARGET_H8300H || TARGET_H8300S)
&& code == IOR
&& w1 == 0xffff
&& (w0 & 0x8000) != 0)
{
cc = CC_SET_ZNV;
}
}
break;
default:
gcc_unreachable ();
}
return cc;
}

void
h8300_expand_branch (rtx operands[])
{
enum rtx_code code = GET_CODE (operands[0]);
rtx op0 = operands[1];
rtx op1 = operands[2];
rtx label = operands[3];
rtx tmp;
tmp = gen_rtx_COMPARE (VOIDmode, op0, op1);
emit_insn (gen_rtx_SET (cc0_rtx, tmp));
tmp = gen_rtx_fmt_ee (code, VOIDmode, cc0_rtx, const0_rtx);
tmp = gen_rtx_IF_THEN_ELSE (VOIDmode, tmp,
gen_rtx_LABEL_REF (VOIDmode, label),
pc_rtx);
emit_jump_insn (gen_rtx_SET (pc_rtx, tmp));
}
void
h8300_expand_store (rtx operands[])
{
rtx dest = operands[0];
enum rtx_code code = GET_CODE (operands[1]);
rtx op0 = operands[2];
rtx op1 = operands[3];
rtx tmp;
tmp = gen_rtx_COMPARE (VOIDmode, op0, op1);
emit_insn (gen_rtx_SET (cc0_rtx, tmp));
tmp = gen_rtx_fmt_ee (code, GET_MODE (dest), cc0_rtx, const0_rtx);
emit_insn (gen_rtx_SET (dest, tmp));
}

enum h8sx_shift_type
h8sx_classify_shift (machine_mode mode, enum rtx_code code, rtx op)
{
if (!TARGET_H8300SX)
return H8SX_SHIFT_NONE;
switch (code)
{
case ASHIFT:
case LSHIFTRT:
if (GET_CODE (op) != CONST_INT)
return H8SX_SHIFT_BINARY;
if (INTVAL (op) <= 0 || INTVAL (op) >= GET_MODE_BITSIZE (mode))
return H8SX_SHIFT_NONE;
if (exact_log2 (INTVAL (op)) >= 0)
return H8SX_SHIFT_UNARY;
return H8SX_SHIFT_BINARY;
case ASHIFTRT:
if (op == const1_rtx || op == const2_rtx)
return H8SX_SHIFT_UNARY;
return H8SX_SHIFT_NONE;
case ROTATE:
if (GET_CODE (op) == CONST_INT
&& (INTVAL (op) == 1
|| INTVAL (op) == 2
|| INTVAL (op) == GET_MODE_BITSIZE (mode) - 2
|| INTVAL (op) == GET_MODE_BITSIZE (mode) - 1))
return H8SX_SHIFT_UNARY;
return H8SX_SHIFT_NONE;
default:
return H8SX_SHIFT_NONE;
}
}
const char *
output_h8sx_shift (rtx *operands, int suffix, int optype)
{
static char buffer[16];
const char *stem;
switch (GET_CODE (operands[3]))
{
case ASHIFT:
stem = "shll";
break;
case ASHIFTRT:
stem = "shar";
break;
case LSHIFTRT:
stem = "shlr";
break;
case ROTATE:
stem = "rotl";
if (INTVAL (operands[2]) > 2)
{
operands[2] = GEN_INT (GET_MODE_BITSIZE (GET_MODE (operands[0]))
- INTVAL (operands[2]));
stem = "rotr";
}
break;
default:
gcc_unreachable ();
}
if (operands[2] == const1_rtx)
sprintf (buffer, "%s.%c\t%%%c0", stem, suffix, optype);
else
sprintf (buffer, "%s.%c\t%%X2,%%%c0", stem, suffix, optype);
return buffer;
}
bool
expand_a_shift (machine_mode mode, enum rtx_code code, rtx operands[])
{
switch (h8sx_classify_shift (mode, code, operands[2]))
{
case H8SX_SHIFT_BINARY:
operands[1] = force_reg (mode, operands[1]);
return false;
case H8SX_SHIFT_UNARY:
return false;
case H8SX_SHIFT_NONE:
break;
}
emit_move_insn (copy_rtx (operands[0]), operands[1]);
emit_insn (gen_rtx_PARALLEL
(VOIDmode,
gen_rtvec (2,
gen_rtx_SET (copy_rtx (operands[0]),
gen_rtx_fmt_ee (code, mode,
copy_rtx (operands[0]), operands[2])),
gen_rtx_CLOBBER (VOIDmode,
gen_rtx_SCRATCH (QImode)))));
return true;
}
enum shift_mode
{
QIshift, HIshift, SIshift
};
struct shift_insn
{
const char *const assembler;
const enum attr_cc cc_valid;
};
static const struct shift_insn shift_one[2][3][3] =
{
{
{
{ "shll\t%X0", CC_SET_ZNV },
{ "add.w\t%T0,%T0", CC_SET_ZN },
{ "add.w\t%f0,%f0\n\taddx\t%y0,%y0\n\taddx\t%z0,%z0", CC_CLOBBER }
},
{
{ "shlr\t%X0", CC_SET_ZNV },
{ "shlr\t%t0\n\trotxr\t%s0", CC_CLOBBER },
{ "shlr\t%z0\n\trotxr\t%y0\n\trotxr\t%x0\n\trotxr\t%w0", CC_CLOBBER }
},
{
{ "shar\t%X0", CC_SET_ZNV },
{ "shar\t%t0\n\trotxr\t%s0", CC_CLOBBER },
{ "shar\t%z0\n\trotxr\t%y0\n\trotxr\t%x0\n\trotxr\t%w0", CC_CLOBBER }
}
},
{
{
{ "shll.b\t%X0", CC_SET_ZNV },
{ "shll.w\t%T0", CC_SET_ZNV },
{ "shll.l\t%S0", CC_SET_ZNV }
},
{
{ "shlr.b\t%X0", CC_SET_ZNV },
{ "shlr.w\t%T0", CC_SET_ZNV },
{ "shlr.l\t%S0", CC_SET_ZNV }
},
{
{ "shar.b\t%X0", CC_SET_ZNV },
{ "shar.w\t%T0", CC_SET_ZNV },
{ "shar.l\t%S0", CC_SET_ZNV }
}
}
};
static const struct shift_insn shift_two[3][3] =
{
{
{ "shll.b\t#2,%X0", CC_SET_ZNV },
{ "shll.w\t#2,%T0", CC_SET_ZNV },
{ "shll.l\t#2,%S0", CC_SET_ZNV }
},
{
{ "shlr.b\t#2,%X0", CC_SET_ZNV },
{ "shlr.w\t#2,%T0", CC_SET_ZNV },
{ "shlr.l\t#2,%S0", CC_SET_ZNV }
},
{
{ "shar.b\t#2,%X0", CC_SET_ZNV },
{ "shar.w\t#2,%T0", CC_SET_ZNV },
{ "shar.l\t#2,%S0", CC_SET_ZNV }
}
};
static const char *const rotate_one[2][3][3] =
{
{
{
"rotr\t%X0",
"shlr\t%t0\n\trotxr\t%s0\n\tbst\t#7,%t0",
0
},
{
"rotl\t%X0",
"shll\t%s0\n\trotxl\t%t0\n\tbst\t#0,%s0",
0
},
{
"rotl\t%X0",
"shll\t%s0\n\trotxl\t%t0\n\tbst\t#0,%s0",
0
}
},
{
{
"rotr.b\t%X0",
"rotr.w\t%T0",
"rotr.l\t%S0"
},
{
"rotl.b\t%X0",
"rotl.w\t%T0",
"rotl.l\t%S0"
},
{
"rotl.b\t%X0",
"rotl.w\t%T0",
"rotl.l\t%S0"
}
}
};
static const char *const rotate_two[3][3] =
{
{
"rotr.b\t#2,%X0",
"rotr.w\t#2,%T0",
"rotr.l\t#2,%S0"
},
{
"rotl.b\t#2,%X0",
"rotl.w\t#2,%T0",
"rotl.l\t#2,%S0"
},
{
"rotl.b\t#2,%X0",
"rotl.w\t#2,%T0",
"rotl.l\t#2,%S0"
}
};
struct shift_info {
enum shift_alg alg;
unsigned int remainder;
const char *special;
const char *shift1;
const char *shift2;
enum attr_cc cc_inline;
enum attr_cc cc_special;
};
static void get_shift_alg (enum shift_type,
enum shift_mode, unsigned int,
struct shift_info *);
static void
get_shift_alg (enum shift_type shift_type, enum shift_mode shift_mode,
unsigned int count, struct shift_info *info)
{
enum h8_cpu cpu;
if (TARGET_H8300)
cpu = H8_300;
else if (TARGET_H8300S)
cpu = H8_S;
else
cpu = H8_300H;
info->alg = SHIFT_LOOP;
switch (shift_mode)
{
case QIshift:
if (count < GET_MODE_BITSIZE (QImode))
info->alg = shift_alg_qi[cpu][shift_type][count];
break;
case HIshift:
if (count < GET_MODE_BITSIZE (HImode))
info->alg = shift_alg_hi[cpu][shift_type][count];
break;
case SIshift:
if (count < GET_MODE_BITSIZE (SImode))
info->alg = shift_alg_si[cpu][shift_type][count];
break;
default:
gcc_unreachable ();
}
switch (info->alg)
{
case SHIFT_INLINE:
info->remainder = count;
case SHIFT_LOOP:
info->shift1 = shift_one[cpu_type][shift_type][shift_mode].assembler;
info->shift2 = shift_two[shift_type][shift_mode].assembler;
info->cc_inline = shift_one[cpu_type][shift_type][shift_mode].cc_valid;
goto end;
case SHIFT_ROT_AND:
info->shift1 = rotate_one[cpu_type][shift_type][shift_mode];
info->shift2 = rotate_two[shift_type][shift_mode];
info->cc_inline = CC_CLOBBER;
goto end;
case SHIFT_SPECIAL:
info->remainder = 0;
info->shift1 = shift_one[cpu_type][shift_type][shift_mode].assembler;
info->shift2 = shift_two[shift_type][shift_mode].assembler;
info->cc_inline = shift_one[cpu_type][shift_type][shift_mode].cc_valid;
info->cc_special = CC_CLOBBER;
break;
}
switch (shift_mode)
{
case QIshift:
gcc_assert (shift_type == SHIFT_ASHIFTRT && count == 7);
info->special = "shll\t%X0\n\tsubx\t%X0,%X0";
goto end;
case HIshift:
if (count == 7)
{
switch (shift_type)
{
case SHIFT_ASHIFT:
if (TARGET_H8300)
info->special = "shar.b\t%t0\n\tmov.b\t%s0,%t0\n\trotxr.b\t%t0\n\trotr.b\t%s0\n\tand.b\t#0x80,%s0";
else
info->special = "shar.b\t%t0\n\tmov.b\t%s0,%t0\n\trotxr.w\t%T0\n\tand.b\t#0x80,%s0";
goto end;
case SHIFT_LSHIFTRT:
if (TARGET_H8300)
info->special = "shal.b\t%s0\n\tmov.b\t%t0,%s0\n\trotxl.b\t%s0\n\trotl.b\t%t0\n\tand.b\t#0x01,%t0";
else
info->special = "shal.b\t%s0\n\tmov.b\t%t0,%s0\n\trotxl.w\t%T0\n\tand.b\t#0x01,%t0";
goto end;
case SHIFT_ASHIFTRT:
info->special = "shal.b\t%s0\n\tmov.b\t%t0,%s0\n\trotxl.b\t%s0\n\tsubx\t%t0,%t0";
goto end;
}
}
else if ((count >= 8 && count <= 13)
|| (TARGET_H8300S && count == 14))
{
info->remainder = count - 8;
switch (shift_type)
{
case SHIFT_ASHIFT:
info->special = "mov.b\t%s0,%t0\n\tsub.b\t%s0,%s0";
goto end;
case SHIFT_LSHIFTRT:
if (TARGET_H8300)
{
info->special = "mov.b\t%t0,%s0\n\tsub.b\t%t0,%t0";
info->shift1  = "shlr.b\t%s0";
info->cc_inline = CC_SET_ZNV;
}
else
{
info->special = "mov.b\t%t0,%s0\n\textu.w\t%T0";
info->cc_special = CC_SET_ZNV;
}
goto end;
case SHIFT_ASHIFTRT:
if (TARGET_H8300)
{
info->special = "mov.b\t%t0,%s0\n\tbld\t#7,%s0\n\tsubx\t%t0,%t0";
info->shift1  = "shar.b\t%s0";
}
else
{
info->special = "mov.b\t%t0,%s0\n\texts.w\t%T0";
info->cc_special = CC_SET_ZNV;
}
goto end;
}
}
else if (count == 14)
{
switch (shift_type)
{
case SHIFT_ASHIFT:
if (TARGET_H8300)
info->special = "mov.b\t%s0,%t0\n\trotr.b\t%t0\n\trotr.b\t%t0\n\tand.b\t#0xC0,%t0\n\tsub.b\t%s0,%s0";
goto end;
case SHIFT_LSHIFTRT:
if (TARGET_H8300)
info->special = "mov.b\t%t0,%s0\n\trotl.b\t%s0\n\trotl.b\t%s0\n\tand.b\t#3,%s0\n\tsub.b\t%t0,%t0";
goto end;
case SHIFT_ASHIFTRT:
if (TARGET_H8300)
info->special = "mov.b\t%t0,%s0\n\tshll.b\t%s0\n\tsubx.b\t%t0,%t0\n\tshll.b\t%s0\n\tmov.b\t%t0,%s0\n\tbst.b\t#0,%s0";
else if (TARGET_H8300H)
{
info->special = "shll.b\t%t0\n\tsubx.b\t%s0,%s0\n\tshll.b\t%t0\n\trotxl.b\t%s0\n\texts.w\t%T0";
info->cc_special = CC_SET_ZNV;
}
else 
gcc_unreachable ();
goto end;
}
}
else if (count == 15)
{
switch (shift_type)
{
case SHIFT_ASHIFT:
info->special = "bld\t#0,%s0\n\txor\t%s0,%s0\n\txor\t%t0,%t0\n\tbst\t#7,%t0";
goto end;
case SHIFT_LSHIFTRT:
info->special = "bld\t#7,%t0\n\txor\t%s0,%s0\n\txor\t%t0,%t0\n\tbst\t#0,%s0";
goto end;
case SHIFT_ASHIFTRT:
info->special = "shll\t%t0\n\tsubx\t%t0,%t0\n\tmov.b\t%t0,%s0";
goto end;
}
}
gcc_unreachable ();
case SIshift:
if (TARGET_H8300 && count >= 8 && count <= 9)
{
info->remainder = count - 8;
switch (shift_type)
{
case SHIFT_ASHIFT:
info->special = "mov.b\t%y0,%z0\n\tmov.b\t%x0,%y0\n\tmov.b\t%w0,%x0\n\tsub.b\t%w0,%w0";
goto end;
case SHIFT_LSHIFTRT:
info->special = "mov.b\t%x0,%w0\n\tmov.b\t%y0,%x0\n\tmov.b\t%z0,%y0\n\tsub.b\t%z0,%z0";
info->shift1  = "shlr\t%y0\n\trotxr\t%x0\n\trotxr\t%w0";
goto end;
case SHIFT_ASHIFTRT:
info->special = "mov.b\t%x0,%w0\n\tmov.b\t%y0,%x0\n\tmov.b\t%z0,%y0\n\tshll\t%z0\n\tsubx\t%z0,%z0";
goto end;
}
}
else if (count == 8 && !TARGET_H8300)
{
switch (shift_type)
{
case SHIFT_ASHIFT:
info->special = "mov.w\t%e0,%f4\n\tmov.b\t%s4,%t4\n\tmov.b\t%t0,%s4\n\tmov.b\t%s0,%t0\n\tsub.b\t%s0,%s0\n\tmov.w\t%f4,%e0";
goto end;
case SHIFT_LSHIFTRT:
info->special = "mov.w\t%e0,%f4\n\tmov.b\t%t0,%s0\n\tmov.b\t%s4,%t0\n\tmov.b\t%t4,%s4\n\textu.w\t%f4\n\tmov.w\t%f4,%e0";
goto end;
case SHIFT_ASHIFTRT:
info->special = "mov.w\t%e0,%f4\n\tmov.b\t%t0,%s0\n\tmov.b\t%s4,%t0\n\tmov.b\t%t4,%s4\n\texts.w\t%f4\n\tmov.w\t%f4,%e0";
goto end;
}
}
else if (count == 15 && TARGET_H8300)
{
switch (shift_type)
{
case SHIFT_ASHIFT:
gcc_unreachable ();
case SHIFT_LSHIFTRT:
info->special = "bld\t#7,%z0\n\tmov.w\t%e0,%f0\n\txor\t%y0,%y0\n\txor\t%z0,%z0\n\trotxl\t%w0\n\trotxl\t%x0\n\trotxl\t%y0";
goto end;
case SHIFT_ASHIFTRT:
info->special = "bld\t#7,%z0\n\tmov.w\t%e0,%f0\n\trotxl\t%w0\n\trotxl\t%x0\n\tsubx\t%y0,%y0\n\tsubx\t%z0,%z0";
goto end;
}
}
else if (count == 15 && !TARGET_H8300)
{
switch (shift_type)
{
case SHIFT_ASHIFT:
info->special = "shlr.w\t%e0\n\tmov.w\t%f0,%e0\n\txor.w\t%f0,%f0\n\trotxr.l\t%S0";
info->cc_special = CC_SET_ZNV;
goto end;
case SHIFT_LSHIFTRT:
info->special = "shll.w\t%f0\n\tmov.w\t%e0,%f0\n\txor.w\t%e0,%e0\n\trotxl.l\t%S0";
info->cc_special = CC_SET_ZNV;
goto end;
case SHIFT_ASHIFTRT:
gcc_unreachable ();
}
}
else if ((TARGET_H8300 && count >= 16 && count <= 20)
|| (TARGET_H8300H && count >= 16 && count <= 19)
|| (TARGET_H8300S && count >= 16 && count <= 21))
{
info->remainder = count - 16;
switch (shift_type)
{
case SHIFT_ASHIFT:
info->special = "mov.w\t%f0,%e0\n\tsub.w\t%f0,%f0";
if (TARGET_H8300)
info->shift1 = "add.w\t%e0,%e0";
goto end;
case SHIFT_LSHIFTRT:
if (TARGET_H8300)
{
info->special = "mov.w\t%e0,%f0\n\tsub.w\t%e0,%e0";
info->shift1  = "shlr\t%x0\n\trotxr\t%w0";
}
else
{
info->special = "mov.w\t%e0,%f0\n\textu.l\t%S0";
info->cc_special = CC_SET_ZNV;
}
goto end;
case SHIFT_ASHIFTRT:
if (TARGET_H8300)
{
info->special = "mov.w\t%e0,%f0\n\tshll\t%z0\n\tsubx\t%z0,%z0\n\tmov.b\t%z0,%y0";
info->shift1  = "shar\t%x0\n\trotxr\t%w0";
}
else
{
info->special = "mov.w\t%e0,%f0\n\texts.l\t%S0";
info->cc_special = CC_SET_ZNV;
}
goto end;
}
}
else if (TARGET_H8300 && count >= 24 && count <= 28)
{
info->remainder = count - 24;
switch (shift_type)
{
case SHIFT_ASHIFT:
info->special = "mov.b\t%w0,%z0\n\tsub.b\t%y0,%y0\n\tsub.w\t%f0,%f0";
info->shift1  = "shll.b\t%z0";
info->cc_inline = CC_SET_ZNV;
goto end;
case SHIFT_LSHIFTRT:
info->special = "mov.b\t%z0,%w0\n\tsub.b\t%x0,%x0\n\tsub.w\t%e0,%e0";
info->shift1  = "shlr.b\t%w0";
info->cc_inline = CC_SET_ZNV;
goto end;
case SHIFT_ASHIFTRT:
info->special = "mov.b\t%z0,%w0\n\tbld\t#7,%w0\n\tsubx\t%x0,%x0\n\tsubx\t%y0,%y0\n\tsubx\t%z0,%z0";
info->shift1  = "shar.b\t%w0";
info->cc_inline = CC_SET_ZNV;
goto end;
}
}
else if ((TARGET_H8300H && count == 24)
|| (TARGET_H8300S && count >= 24 && count <= 25))
{
info->remainder = count - 24;
switch (shift_type)
{
case SHIFT_ASHIFT:
info->special = "mov.b\t%s0,%t0\n\tsub.b\t%s0,%s0\n\tmov.w\t%f0,%e0\n\tsub.w\t%f0,%f0";
goto end;
case SHIFT_LSHIFTRT:
info->special = "mov.w\t%e0,%f0\n\tmov.b\t%t0,%s0\n\textu.w\t%f0\n\textu.l\t%S0";
info->cc_special = CC_SET_ZNV;
goto end;
case SHIFT_ASHIFTRT:
info->special = "mov.w\t%e0,%f0\n\tmov.b\t%t0,%s0\n\texts.w\t%f0\n\texts.l\t%S0";
info->cc_special = CC_SET_ZNV;
goto end;
}
}
else if (!TARGET_H8300 && count == 28)
{
switch (shift_type)
{
case SHIFT_ASHIFT:
if (TARGET_H8300H)
info->special = "sub.w\t%e0,%e0\n\trotr.l\t%S0\n\trotr.l\t%S0\n\trotr.l\t%S0\n\trotr.l\t%S0\n\tsub.w\t%f0,%f0";
else
info->special = "sub.w\t%e0,%e0\n\trotr.l\t#2,%S0\n\trotr.l\t#2,%S0\n\tsub.w\t%f0,%f0";
goto end;
case SHIFT_LSHIFTRT:
if (TARGET_H8300H)
{
info->special = "sub.w\t%f0,%f0\n\trotl.l\t%S0\n\trotl.l\t%S0\n\trotl.l\t%S0\n\trotl.l\t%S0\n\textu.l\t%S0";
info->cc_special = CC_SET_ZNV;
}
else
info->special = "sub.w\t%f0,%f0\n\trotl.l\t#2,%S0\n\trotl.l\t#2,%S0\n\textu.l\t%S0";
goto end;
case SHIFT_ASHIFTRT:
gcc_unreachable ();
}
}
else if (!TARGET_H8300 && count == 29)
{
switch (shift_type)
{
case SHIFT_ASHIFT:
if (TARGET_H8300H)
info->special = "sub.w\t%e0,%e0\n\trotr.l\t%S0\n\trotr.l\t%S0\n\trotr.l\t%S0\n\tsub.w\t%f0,%f0";
else
info->special = "sub.w\t%e0,%e0\n\trotr.l\t#2,%S0\n\trotr.l\t%S0\n\tsub.w\t%f0,%f0";
goto end;
case SHIFT_LSHIFTRT:
if (TARGET_H8300H)
{
info->special = "sub.w\t%f0,%f0\n\trotl.l\t%S0\n\trotl.l\t%S0\n\trotl.l\t%S0\n\textu.l\t%S0";
info->cc_special = CC_SET_ZNV;
}
else
{
info->special = "sub.w\t%f0,%f0\n\trotl.l\t#2,%S0\n\trotl.l\t%S0\n\textu.l\t%S0";
info->cc_special = CC_SET_ZNV;
}
goto end;
case SHIFT_ASHIFTRT:
gcc_unreachable ();
}
}
else if (!TARGET_H8300 && count == 30)
{
switch (shift_type)
{
case SHIFT_ASHIFT:
if (TARGET_H8300H)
info->special = "sub.w\t%e0,%e0\n\trotr.l\t%S0\n\trotr.l\t%S0\n\tsub.w\t%f0,%f0";
else
info->special = "sub.w\t%e0,%e0\n\trotr.l\t#2,%S0\n\tsub.w\t%f0,%f0";
goto end;
case SHIFT_LSHIFTRT:
if (TARGET_H8300H)
info->special = "sub.w\t%f0,%f0\n\trotl.l\t%S0\n\trotl.l\t%S0\n\textu.l\t%S0";
else
info->special = "sub.w\t%f0,%f0\n\trotl.l\t#2,%S0\n\textu.l\t%S0";
goto end;
case SHIFT_ASHIFTRT:
gcc_unreachable ();
}
}
else if (count == 31)
{
if (TARGET_H8300)
{
switch (shift_type)
{
case SHIFT_ASHIFT:
info->special = "sub.w\t%e0,%e0\n\tshlr\t%w0\n\tmov.w\t%e0,%f0\n\trotxr\t%z0";
goto end;
case SHIFT_LSHIFTRT:
info->special = "sub.w\t%f0,%f0\n\tshll\t%z0\n\tmov.w\t%f0,%e0\n\trotxl\t%w0";
goto end;
case SHIFT_ASHIFTRT:
info->special = "shll\t%z0\n\tsubx\t%w0,%w0\n\tmov.b\t%w0,%x0\n\tmov.w\t%f0,%e0";
goto end;
}
}
else
{
switch (shift_type)
{
case SHIFT_ASHIFT:
info->special = "shlr.l\t%S0\n\txor.l\t%S0,%S0\n\trotxr.l\t%S0";
info->cc_special = CC_SET_ZNV;
goto end;
case SHIFT_LSHIFTRT:
info->special = "shll.l\t%S0\n\txor.l\t%S0,%S0\n\trotxl.l\t%S0";
info->cc_special = CC_SET_ZNV;
goto end;
case SHIFT_ASHIFTRT:
info->special = "shll\t%e0\n\tsubx\t%w0,%w0\n\texts.w\t%T0\n\texts.l\t%S0";
info->cc_special = CC_SET_ZNV;
goto end;
}
}
}
gcc_unreachable ();
default:
gcc_unreachable ();
}
end:
if (!TARGET_H8300S)
info->shift2 = NULL;
}
int
h8300_shift_needs_scratch_p (int count, machine_mode mode)
{
enum h8_cpu cpu;
int a, lr, ar;
if (GET_MODE_BITSIZE (mode) <= count)
return 1;
if (TARGET_H8300)
cpu = H8_300;
else if (TARGET_H8300S)
cpu = H8_S;
else
cpu = H8_300H;
switch (mode)
{
case E_QImode:
a  = shift_alg_qi[cpu][SHIFT_ASHIFT][count];
lr = shift_alg_qi[cpu][SHIFT_LSHIFTRT][count];
ar = shift_alg_qi[cpu][SHIFT_ASHIFTRT][count];
break;
case E_HImode:
a  = shift_alg_hi[cpu][SHIFT_ASHIFT][count];
lr = shift_alg_hi[cpu][SHIFT_LSHIFTRT][count];
ar = shift_alg_hi[cpu][SHIFT_ASHIFTRT][count];
break;
case E_SImode:
a  = shift_alg_si[cpu][SHIFT_ASHIFT][count];
lr = shift_alg_si[cpu][SHIFT_LSHIFTRT][count];
ar = shift_alg_si[cpu][SHIFT_ASHIFTRT][count];
break;
default:
gcc_unreachable ();
}
return (a == SHIFT_LOOP || lr == SHIFT_LOOP || ar == SHIFT_LOOP
|| (TARGET_H8300H && mode == SImode && count == 8));
}
const char *
output_a_shift (rtx *operands)
{
static int loopend_lab;
rtx shift = operands[3];
machine_mode mode = GET_MODE (shift);
enum rtx_code code = GET_CODE (shift);
enum shift_type shift_type;
enum shift_mode shift_mode;
struct shift_info info;
int n;
loopend_lab++;
switch (mode)
{
case E_QImode:
shift_mode = QIshift;
break;
case E_HImode:
shift_mode = HIshift;
break;
case E_SImode:
shift_mode = SIshift;
break;
default:
gcc_unreachable ();
}
switch (code)
{
case ASHIFTRT:
shift_type = SHIFT_ASHIFTRT;
break;
case LSHIFTRT:
shift_type = SHIFT_LSHIFTRT;
break;
case ASHIFT:
shift_type = SHIFT_ASHIFT;
break;
default:
gcc_unreachable ();
}
gcc_assert (GET_CODE (operands[2]) == CONST_INT);
n = INTVAL (operands[2]);
if (n < 0)
n = 0;
else if ((unsigned int) n > GET_MODE_BITSIZE (mode))
n = GET_MODE_BITSIZE (mode);
get_shift_alg (shift_type, shift_mode, n, &info);
switch (info.alg)
{
case SHIFT_SPECIAL:
output_asm_insn (info.special, operands);
case SHIFT_INLINE:
n = info.remainder;
if (info.shift2 != NULL)
{
for (; n > 1; n -= 2)
output_asm_insn (info.shift2, operands);
}
for (; n > 0; n--)
output_asm_insn (info.shift1, operands);
return "";
case SHIFT_ROT_AND:
{
int m = GET_MODE_BITSIZE (mode) - n;
const int mask = (shift_type == SHIFT_ASHIFT
? ((1 << m) - 1) << n
: (1 << m) - 1);
char insn_buf[200];
gcc_assert (info.shift1);
if (info.shift2 != NULL)
{
for (; m > 1; m -= 2)
output_asm_insn (info.shift2, operands);
}
for (; m > 0; m--)
output_asm_insn (info.shift1, operands);
switch (mode)
{
case E_QImode:
sprintf (insn_buf, "and\t#%d,%%X0", mask);
break;
case E_HImode:
gcc_assert (TARGET_H8300H || TARGET_H8300S);
sprintf (insn_buf, "and.w\t#%d,%%T0", mask);
break;
default:
gcc_unreachable ();
}
output_asm_insn (insn_buf, operands);
return "";
}
case SHIFT_LOOP:
if (info.shift2 != NULL)
{
fprintf (asm_out_file, "\tmov.b	#%d,%sl\n", n / 2,
names_big[REGNO (operands[4])]);
fprintf (asm_out_file, ".Llt%d:\n", loopend_lab);
output_asm_insn (info.shift2, operands);
output_asm_insn ("add	#0xff,%X4", operands);
fprintf (asm_out_file, "\tbne	.Llt%d\n", loopend_lab);
if (n % 2)
output_asm_insn (info.shift1, operands);
}
else
{
fprintf (asm_out_file, "\tmov.b	#%d,%sl\n", n,
names_big[REGNO (operands[4])]);
fprintf (asm_out_file, ".Llt%d:\n", loopend_lab);
output_asm_insn (info.shift1, operands);
output_asm_insn ("add	#0xff,%X4", operands);
fprintf (asm_out_file, "\tbne	.Llt%d\n", loopend_lab);
}
return "";
default:
gcc_unreachable ();
}
}
static unsigned int
h8300_asm_insn_count (const char *templ)
{
unsigned int count = 1;
for (; *templ; templ++)
if (*templ == '\n')
count++;
return count;
}
unsigned int
compute_a_shift_length (rtx insn ATTRIBUTE_UNUSED, rtx *operands)
{
rtx shift = operands[3];
machine_mode mode = GET_MODE (shift);
enum rtx_code code = GET_CODE (shift);
enum shift_type shift_type;
enum shift_mode shift_mode;
struct shift_info info;
unsigned int wlength = 0;
switch (mode)
{
case E_QImode:
shift_mode = QIshift;
break;
case E_HImode:
shift_mode = HIshift;
break;
case E_SImode:
shift_mode = SIshift;
break;
default:
gcc_unreachable ();
}
switch (code)
{
case ASHIFTRT:
shift_type = SHIFT_ASHIFTRT;
break;
case LSHIFTRT:
shift_type = SHIFT_LSHIFTRT;
break;
case ASHIFT:
shift_type = SHIFT_ASHIFT;
break;
default:
gcc_unreachable ();
}
if (GET_CODE (operands[2]) != CONST_INT)
{
get_shift_alg (shift_type, shift_mode, 1, &info);
return (4 + h8300_asm_insn_count (info.shift1)) * 2;
}
else
{
int n = INTVAL (operands[2]);
if (n < 0)
n = 0;
else if ((unsigned int) n > GET_MODE_BITSIZE (mode))
n = GET_MODE_BITSIZE (mode);
get_shift_alg (shift_type, shift_mode, n, &info);
switch (info.alg)
{
case SHIFT_SPECIAL:
wlength += h8300_asm_insn_count (info.special);
if (strstr (info.special, "xor.l") != NULL)
wlength++;
case SHIFT_INLINE:
n = info.remainder;
if (info.shift2 != NULL)
{
wlength += h8300_asm_insn_count (info.shift2) * (n / 2);
n = n % 2;
}
wlength += h8300_asm_insn_count (info.shift1) * n;
return 2 * wlength;
case SHIFT_ROT_AND:
{
int m = GET_MODE_BITSIZE (mode) - n;
gcc_assert (info.shift1);
if (info.shift2 != NULL)
{
wlength += h8300_asm_insn_count (info.shift2) * (m / 2);
m = m % 2;
}
wlength += h8300_asm_insn_count (info.shift1) * m;
switch (mode)
{
case E_QImode:
wlength += 1;
break;
case E_HImode:
wlength += 2;
break;
case E_SImode:
gcc_assert (!TARGET_H8300);
wlength += 3;
break;
default:
gcc_unreachable ();
}
return 2 * wlength;
}
case SHIFT_LOOP:
if (info.shift2 != NULL)
{
wlength += 3 + h8300_asm_insn_count (info.shift2);
if (n % 2)
wlength += h8300_asm_insn_count (info.shift1);
}
else
{
wlength += 3 + h8300_asm_insn_count (info.shift1);
}
return 2 * wlength;
default:
gcc_unreachable ();
}
}
}
enum attr_cc
compute_a_shift_cc (rtx insn ATTRIBUTE_UNUSED, rtx *operands)
{
rtx shift = operands[3];
machine_mode mode = GET_MODE (shift);
enum rtx_code code = GET_CODE (shift);
enum shift_type shift_type;
enum shift_mode shift_mode;
struct shift_info info;
int n;
switch (mode)
{
case E_QImode:
shift_mode = QIshift;
break;
case E_HImode:
shift_mode = HIshift;
break;
case E_SImode:
shift_mode = SIshift;
break;
default:
gcc_unreachable ();
}
switch (code)
{
case ASHIFTRT:
shift_type = SHIFT_ASHIFTRT;
break;
case LSHIFTRT:
shift_type = SHIFT_LSHIFTRT;
break;
case ASHIFT:
shift_type = SHIFT_ASHIFT;
break;
default:
gcc_unreachable ();
}
gcc_assert (GET_CODE (operands[2]) == CONST_INT);
n = INTVAL (operands[2]);
if (n < 0)
n = 0;
else if ((unsigned int) n > GET_MODE_BITSIZE (mode))
n = GET_MODE_BITSIZE (mode);
get_shift_alg (shift_type, shift_mode, n, &info);
switch (info.alg)
{
case SHIFT_SPECIAL:
if (info.remainder == 0)
return info.cc_special;
case SHIFT_INLINE:
return info.cc_inline;
case SHIFT_ROT_AND:
return CC_SET_ZNV;
case SHIFT_LOOP:
if (info.shift2 != NULL)
{
if (n % 2)
return info.cc_inline;
}
return CC_CLOBBER;
default:
gcc_unreachable ();
}
}

int
expand_a_rotate (rtx operands[])
{
rtx dst = operands[0];
rtx src = operands[1];
rtx rotate_amount = operands[2];
machine_mode mode = GET_MODE (dst);
if (h8sx_classify_shift (mode, ROTATE, rotate_amount) == H8SX_SHIFT_UNARY)
return false;
emit_move_insn (dst, src);
if (GET_CODE (rotate_amount) != CONST_INT)
{
rtx counter = gen_reg_rtx (QImode);
rtx_code_label *start_label = gen_label_rtx ();
rtx_code_label *end_label = gen_label_rtx ();
emit_cmp_and_jump_insns (rotate_amount, const0_rtx, LE, NULL_RTX,
QImode, 0, end_label);
emit_move_insn (counter, rotate_amount);
emit_label (start_label);
switch (mode)
{
case E_QImode:
emit_insn (gen_rotlqi3_1 (dst, dst, const1_rtx));
break;
case E_HImode:
emit_insn (gen_rotlhi3_1 (dst, dst, const1_rtx));
break;
case E_SImode:
emit_insn (gen_rotlsi3_1 (dst, dst, const1_rtx));
break;
default:
gcc_unreachable ();
}
emit_insn (gen_addqi3 (counter, counter, constm1_rtx));
emit_cmp_and_jump_insns (counter, const0_rtx, NE, NULL_RTX, QImode, 1,
start_label);
emit_label (end_label);
}
else
{
switch (mode)
{
case E_QImode:
emit_insn (gen_rotlqi3_1 (dst, dst, rotate_amount));
break;
case E_HImode:
emit_insn (gen_rotlhi3_1 (dst, dst, rotate_amount));
break;
case E_SImode:
emit_insn (gen_rotlsi3_1 (dst, dst, rotate_amount));
break;
default:
gcc_unreachable ();
}
}
return 1;
}
const char *
output_a_rotate (enum rtx_code code, rtx *operands)
{
rtx dst = operands[0];
rtx rotate_amount = operands[2];
enum shift_mode rotate_mode;
enum shift_type rotate_type;
const char *insn_buf;
int bits;
int amount;
machine_mode mode = GET_MODE (dst);
gcc_assert (GET_CODE (rotate_amount) == CONST_INT);
switch (mode)
{
case E_QImode:
rotate_mode = QIshift;
break;
case E_HImode:
rotate_mode = HIshift;
break;
case E_SImode:
rotate_mode = SIshift;
break;
default:
gcc_unreachable ();
}
switch (code)
{
case ROTATERT:
rotate_type = SHIFT_ASHIFT;
break;
case ROTATE:
rotate_type = SHIFT_LSHIFTRT;
break;
default:
gcc_unreachable ();
}
amount = INTVAL (rotate_amount);
if (amount < 0)
amount = 0;
if ((unsigned int) amount > GET_MODE_BITSIZE (mode))
amount = GET_MODE_BITSIZE (mode);
if ((unsigned int) amount > GET_MODE_BITSIZE (mode) / (unsigned) 2)
{
amount = GET_MODE_BITSIZE (mode) - amount;
rotate_type =
(rotate_type == SHIFT_ASHIFT) ? SHIFT_LSHIFTRT : SHIFT_ASHIFT;
}
if ((mode == HImode && TARGET_H8300 && amount >= 5)
|| (mode == HImode && TARGET_H8300H && amount >= 6)
|| (mode == HImode && TARGET_H8300S && amount == 8)
|| (mode == SImode && TARGET_H8300H && amount >= 10)
|| (mode == SImode && TARGET_H8300S && amount >= 13))
{
switch (mode)
{
case E_HImode:
insn_buf = "xor.b\t%s0,%t0\n\txor.b\t%t0,%s0\n\txor.b\t%s0,%t0";
output_asm_insn (insn_buf, operands);
break;
case E_SImode:
insn_buf = "xor.w\t%e0,%f0\n\txor.w\t%f0,%e0\n\txor.w\t%e0,%f0";
output_asm_insn (insn_buf, operands);
break;
default:
gcc_unreachable ();
}
amount = GET_MODE_BITSIZE (mode) / 2 - amount;
rotate_type =
(rotate_type == SHIFT_ASHIFT) ? SHIFT_LSHIFTRT : SHIFT_ASHIFT;
}
for (bits = TARGET_H8300S ? 2 : 1; bits > 0; bits /= 2)
{
if (bits == 2)
insn_buf = rotate_two[rotate_type][rotate_mode];
else
insn_buf = rotate_one[cpu_type][rotate_type][rotate_mode];
for (; amount >= bits; amount -= bits)
output_asm_insn (insn_buf, operands);
}
return "";
}
unsigned int
compute_a_rotate_length (rtx *operands)
{
rtx src = operands[1];
rtx amount_rtx = operands[2];
machine_mode mode = GET_MODE (src);
int amount;
unsigned int length = 0;
gcc_assert (GET_CODE (amount_rtx) == CONST_INT);
amount = INTVAL (amount_rtx);
if (amount < 0)
amount = 0;
if ((unsigned int) amount > GET_MODE_BITSIZE (mode))
amount = GET_MODE_BITSIZE (mode);
if ((unsigned int) amount > GET_MODE_BITSIZE (mode) / (unsigned) 2)
amount = GET_MODE_BITSIZE (mode) - amount;
if ((mode == HImode && TARGET_H8300 && amount >= 5)
|| (mode == HImode && TARGET_H8300H && amount >= 6)
|| (mode == HImode && TARGET_H8300S && amount == 8)
|| (mode == SImode && TARGET_H8300H && amount >= 10)
|| (mode == SImode && TARGET_H8300S && amount >= 13))
{
amount = GET_MODE_BITSIZE (mode) / 2 - amount;
length += 6;
}
if (TARGET_H8300S)
amount = amount / 2 + amount % 2;
length += amount * ((TARGET_H8300 && mode == HImode) ? 6 : 2);
return length;
}

int
fix_bit_operand (rtx *operands, enum rtx_code code)
{
if (code == AND
? single_zero_operand (operands[2], QImode)
: single_one_operand (operands[2], QImode))
{
if (GET_CODE (operands[0]) == MEM
&& !satisfies_constraint_U (operands[0]))
{
rtx mem = gen_rtx_MEM (GET_MODE (operands[0]),
copy_to_mode_reg (Pmode,
XEXP (operands[0], 0)));
MEM_COPY_ATTRIBUTES (mem, operands[0]);
operands[0] = mem;
}
if (GET_CODE (operands[1]) == MEM
&& !satisfies_constraint_U (operands[1]))
{
rtx mem = gen_rtx_MEM (GET_MODE (operands[1]),
copy_to_mode_reg (Pmode,
XEXP (operands[1], 0)));
MEM_COPY_ATTRIBUTES (mem, operands[0]);
operands[1] = mem;
}
return 0;
}
operands[1] = force_reg (QImode, operands[1]);
{
rtx res = gen_reg_rtx (QImode);
switch (code)
{
case AND:
emit_insn (gen_andqi3_1 (res, operands[1], operands[2]));
break;
case IOR:
emit_insn (gen_iorqi3_1 (res, operands[1], operands[2]));
break;
case XOR:
emit_insn (gen_xorqi3_1 (res, operands[1], operands[2]));
break;
default:
gcc_unreachable ();
}
emit_insn (gen_movqi (operands[0], res));
}
return 1;
}
static int
h8300_interrupt_function_p (tree func)
{
tree a;
if (TREE_CODE (func) != FUNCTION_DECL)
return 0;
a = lookup_attribute ("interrupt_handler", DECL_ATTRIBUTES (func));
return a != NULL_TREE;
}
static int
h8300_saveall_function_p (tree func)
{
tree a;
if (TREE_CODE (func) != FUNCTION_DECL)
return 0;
a = lookup_attribute ("saveall", DECL_ATTRIBUTES (func));
return a != NULL_TREE;
}
static int
h8300_os_task_function_p (tree func)
{
tree a;
if (TREE_CODE (func) != FUNCTION_DECL)
return 0;
a = lookup_attribute ("OS_Task", DECL_ATTRIBUTES (func));
return a != NULL_TREE;
}
static int
h8300_monitor_function_p (tree func)
{
tree a;
if (TREE_CODE (func) != FUNCTION_DECL)
return 0;
a = lookup_attribute ("monitor", DECL_ATTRIBUTES (func));
return a != NULL_TREE;
}
int
h8300_funcvec_function_p (tree func)
{
tree a;
if (TREE_CODE (func) != FUNCTION_DECL)
return 0;
a = lookup_attribute ("function_vector", DECL_ATTRIBUTES (func));
return a != NULL_TREE;
}
int
h8300_eightbit_data_p (tree decl)
{
tree a;
if (TREE_CODE (decl) != VAR_DECL)
return 0;
a = lookup_attribute ("eightbit_data", DECL_ATTRIBUTES (decl));
return a != NULL_TREE;
}
int
h8300_tiny_data_p (tree decl)
{
tree a;
if (TREE_CODE (decl) != VAR_DECL)
return 0;
a = lookup_attribute ("tiny_data", DECL_ATTRIBUTES (decl));
return a != NULL_TREE;
}
static void
h8300_insert_attributes (tree node, tree *attributes)
{
if (TREE_CODE (node) == FUNCTION_DECL)
{
if (pragma_interrupt)
{
pragma_interrupt = 0;
*attributes = tree_cons (get_identifier ("interrupt_handler"),
NULL, *attributes);
}
if (pragma_saveall)
{
pragma_saveall = 0;
*attributes = tree_cons (get_identifier ("saveall"),
NULL, *attributes);
}
}
}
static const struct attribute_spec h8300_attribute_table[] =
{
{ "interrupt_handler", 0, 0, true,  false, false, false,
h8300_handle_fndecl_attribute, NULL },
{ "saveall",           0, 0, true,  false, false, false,
h8300_handle_fndecl_attribute, NULL },
{ "OS_Task",           0, 0, true,  false, false, false,
h8300_handle_fndecl_attribute, NULL },
{ "monitor",           0, 0, true,  false, false, false,
h8300_handle_fndecl_attribute, NULL },
{ "function_vector",   0, 0, true,  false, false, false,
h8300_handle_fndecl_attribute, NULL },
{ "eightbit_data",     0, 0, true,  false, false, false,
h8300_handle_eightbit_data_attribute, NULL },
{ "tiny_data",         0, 0, true,  false, false, false,
h8300_handle_tiny_data_attribute, NULL },
{ NULL,                0, 0, false, false, false, false, NULL, NULL }
};
static tree
h8300_handle_fndecl_attribute (tree *node, tree name,
tree args ATTRIBUTE_UNUSED,
int flags ATTRIBUTE_UNUSED,
bool *no_add_attrs)
{
if (TREE_CODE (*node) != FUNCTION_DECL)
{
warning (OPT_Wattributes, "%qE attribute only applies to functions",
name);
*no_add_attrs = true;
}
return NULL_TREE;
}
static tree
h8300_handle_eightbit_data_attribute (tree *node, tree name,
tree args ATTRIBUTE_UNUSED,
int flags ATTRIBUTE_UNUSED,
bool *no_add_attrs)
{
tree decl = *node;
if (TREE_STATIC (decl) || DECL_EXTERNAL (decl))
{
set_decl_section_name (decl, ".eight");
}
else
{
warning (OPT_Wattributes, "%qE attribute ignored",
name);
*no_add_attrs = true;
}
return NULL_TREE;
}
static tree
h8300_handle_tiny_data_attribute (tree *node, tree name,
tree args ATTRIBUTE_UNUSED,
int flags ATTRIBUTE_UNUSED,
bool *no_add_attrs)
{
tree decl = *node;
if (TREE_STATIC (decl) || DECL_EXTERNAL (decl))
{
set_decl_section_name (decl, ".tiny");
}
else
{
warning (OPT_Wattributes, "%qE attribute ignored",
name);
*no_add_attrs = true;
}
return NULL_TREE;
}
static void
h8300_encode_section_info (tree decl, rtx rtl, int first)
{
int extra_flags = 0;
default_encode_section_info (decl, rtl, first);
if (TREE_CODE (decl) == FUNCTION_DECL
&& h8300_funcvec_function_p (decl))
extra_flags = SYMBOL_FLAG_FUNCVEC_FUNCTION;
else if (TREE_CODE (decl) == VAR_DECL
&& (TREE_STATIC (decl) || DECL_EXTERNAL (decl)))
{
if (h8300_eightbit_data_p (decl))
extra_flags = SYMBOL_FLAG_EIGHTBIT_DATA;
else if (first && h8300_tiny_data_p (decl))
extra_flags = SYMBOL_FLAG_TINY_DATA;
}
if (extra_flags)
SYMBOL_REF_FLAGS (XEXP (rtl, 0)) |= extra_flags;
}
const char *
output_simode_bld (int bild, rtx operands[])
{
if (TARGET_H8300)
{
output_asm_insn ("sub.w\t%e0,%e0\n\tsub.w\t%f0,%f0", operands);
if (bild)
output_asm_insn ("bild\t%Z2,%Y1", operands);
else
output_asm_insn ("bld\t%Z2,%Y1", operands);
output_asm_insn ("bst\t#0,%w0", operands);
}
else
{
int clear_first = (REG_P (operands[0]) && REG_P (operands[1])
&& REGNO (operands[0]) != REGNO (operands[1]));
if (clear_first)
output_asm_insn ("sub.l\t%S0,%S0", operands);
if (bild)
output_asm_insn ("bild\t%Z2,%Y1", operands);
else
output_asm_insn ("bld\t%Z2,%Y1", operands);
if (!clear_first)
output_asm_insn ("xor.l\t%S0,%S0", operands);
output_asm_insn ("rotxl.l\t%S0", operands);
}
return "";
}
static void
h8300_reorg (void)
{
if (flag_delayed_branch)
shorten_branches (get_insns ());
}
#ifndef OBJECT_FORMAT_ELF
static void
h8300_asm_named_section (const char *name, unsigned int flags ATTRIBUTE_UNUSED,
tree decl)
{
fprintf (asm_out_file, "\t.section %s\n", name);
}
#endif 
int
h8300_eightbit_constant_address_p (rtx x)
{
const unsigned HOST_WIDE_INT n1 = trunc_int_for_mode (0xff00, HImode);
const unsigned HOST_WIDE_INT n2 = trunc_int_for_mode (0xffff, HImode);
const unsigned HOST_WIDE_INT h1 = trunc_int_for_mode (0x00ffff00, SImode);
const unsigned HOST_WIDE_INT h2 = trunc_int_for_mode (0x00ffffff, SImode);
const unsigned HOST_WIDE_INT s1 = trunc_int_for_mode (0xffffff00, SImode);
const unsigned HOST_WIDE_INT s2 = trunc_int_for_mode (0xffffffff, SImode);
unsigned HOST_WIDE_INT addr;
if (GET_CODE (x) == SYMBOL_REF)
return (SYMBOL_REF_FLAGS (x) & SYMBOL_FLAG_EIGHTBIT_DATA) != 0;
if (GET_CODE (x) == CONST
&& GET_CODE (XEXP (x, 0)) == PLUS
&& GET_CODE (XEXP (XEXP (x, 0), 0)) == SYMBOL_REF
&& (SYMBOL_REF_FLAGS (XEXP (XEXP (x, 0), 0)) & SYMBOL_FLAG_EIGHTBIT_DATA) != 0)
return 1;
if (GET_CODE (x) != CONST_INT)
return 0;
addr = INTVAL (x);
return (0
|| ((TARGET_H8300 || TARGET_NORMAL_MODE) && IN_RANGE (addr, n1, n2))
|| (TARGET_H8300H && IN_RANGE (addr, h1, h2))
|| (TARGET_H8300S && IN_RANGE (addr, s1, s2)));
}
int
h8300_tiny_constant_address_p (rtx x)
{
const unsigned HOST_WIDE_INT h1 = trunc_int_for_mode (0x00000000, SImode);
const unsigned HOST_WIDE_INT h2 = trunc_int_for_mode (0x00007fff, SImode);
const unsigned HOST_WIDE_INT h3 = trunc_int_for_mode (0x00ff8000, SImode);
const unsigned HOST_WIDE_INT h4 = trunc_int_for_mode (0x00ffffff, SImode);
const unsigned HOST_WIDE_INT s1 = trunc_int_for_mode (0x00000000, SImode);
const unsigned HOST_WIDE_INT s2 = trunc_int_for_mode (0x00007fff, SImode);
const unsigned HOST_WIDE_INT s3 = trunc_int_for_mode (0xffff8000, SImode);
const unsigned HOST_WIDE_INT s4 = trunc_int_for_mode (0xffffffff, SImode);
unsigned HOST_WIDE_INT addr;
switch (GET_CODE (x))
{
case SYMBOL_REF:
return (TARGET_NORMAL_MODE
|| (SYMBOL_REF_FLAGS (x) & SYMBOL_FLAG_TINY_DATA) != 0);
case CONST_INT:
addr = INTVAL (x);
return (TARGET_NORMAL_MODE
|| (TARGET_H8300H
&& (IN_RANGE (addr, h1, h2) || IN_RANGE (addr, h3, h4)))
|| (TARGET_H8300S
&& (IN_RANGE (addr, s1, s2) || IN_RANGE (addr, s3, s4))));
case CONST:
return TARGET_NORMAL_MODE;
default:
return 0;
}
}
int
byte_accesses_mergeable_p (rtx addr1, rtx addr2)
{
HOST_WIDE_INT offset1, offset2;
rtx reg1, reg2;
if (REG_P (addr1))
{
reg1 = addr1;
offset1 = 0;
}
else if (GET_CODE (addr1) == PLUS
&& REG_P (XEXP (addr1, 0))
&& GET_CODE (XEXP (addr1, 1)) == CONST_INT)
{
reg1 = XEXP (addr1, 0);
offset1 = INTVAL (XEXP (addr1, 1));
}
else
return 0;
if (REG_P (addr2))
{
reg2 = addr2;
offset2 = 0;
}
else if (GET_CODE (addr2) == PLUS
&& REG_P (XEXP (addr2, 0))
&& GET_CODE (XEXP (addr2, 1)) == CONST_INT)
{
reg2 = XEXP (addr2, 0);
offset2 = INTVAL (XEXP (addr2, 1));
}
else
return 0;
if (((reg1 == stack_pointer_rtx && reg2 == stack_pointer_rtx)
|| (reg1 == frame_pointer_rtx && reg2 == frame_pointer_rtx))
&& offset1 % 2 == 0
&& offset1 + 1 == offset2)
return 1;
return 0;
}
int
same_cmp_preceding_p (rtx_insn *i3)
{
rtx_insn *i1, *i2;
i2 = prev_nonnote_insn (i3);
if (i2 == NULL)
return 0;
i1 = prev_nonnote_insn (i2);
if (i1 == NULL)
return 0;
return (INSN_P (i1) && rtx_equal_p (PATTERN (i1), PATTERN (i3))
&& any_condjump_p (i2) && onlyjump_p (i2));
}
int
same_cmp_following_p (rtx_insn *i1)
{
rtx_insn *i2, *i3;
i2 = next_nonnote_insn (i1);
if (i2 == NULL)
return 0;
i3 = next_nonnote_insn (i2);
if (i3 == NULL)
return 0;
return (INSN_P (i3) && rtx_equal_p (PATTERN (i1), PATTERN (i3))
&& any_condjump_p (i2) && onlyjump_p (i2));
}
int
h8300_regs_ok_for_stm (int n, rtx operands[])
{
switch (n)
{
case 2:
return ((REGNO (operands[0]) == 0 && REGNO (operands[1]) == 1)
|| (REGNO (operands[0]) == 2 && REGNO (operands[1]) == 3)
|| (REGNO (operands[0]) == 4 && REGNO (operands[1]) == 5));
case 3:
return ((REGNO (operands[0]) == 0
&& REGNO (operands[1]) == 1
&& REGNO (operands[2]) == 2)
|| (REGNO (operands[0]) == 4
&& REGNO (operands[1]) == 5
&& REGNO (operands[2]) == 6));
case 4:
return (REGNO (operands[0]) == 0
&& REGNO (operands[1]) == 1
&& REGNO (operands[2]) == 2
&& REGNO (operands[3]) == 3);
default:
gcc_unreachable ();
}
}
int
h8300_hard_regno_rename_ok (unsigned int old_reg ATTRIBUTE_UNUSED,
unsigned int new_reg)
{
if (h8300_current_function_interrupt_function_p ()
&& !df_regs_ever_live_p (new_reg))
return 0;
return 1;
}
static bool
h8300_hard_regno_scratch_ok (unsigned int regno)
{
if (h8300_current_function_interrupt_function_p ()
&& ! WORD_REG_USED (regno))
return false;
return true;
}
static int
h8300_rtx_ok_for_base_p (rtx x, int strict)
{
if (GET_CODE (x) == SUBREG)
x = SUBREG_REG (x);
return (REG_P (x)
&& (strict
? REG_OK_FOR_BASE_STRICT_P (x)
: REG_OK_FOR_BASE_NONSTRICT_P (x)));
}
static bool
h8300_legitimate_address_p (machine_mode mode, rtx x, bool strict)
{
if (h8300_rtx_ok_for_base_p (x, strict))
return 1;
if (CONSTANT_ADDRESS_P (x))
return 1;
if (TARGET_H8300SX
&& (   GET_CODE (x) == PRE_INC
|| GET_CODE (x) == PRE_DEC
|| GET_CODE (x) == POST_INC
|| GET_CODE (x) == POST_DEC)
&& h8300_rtx_ok_for_base_p (XEXP (x, 0), strict))
return 1;
if (GET_CODE (x) == PLUS
&& CONSTANT_ADDRESS_P (XEXP (x, 1))
&& h8300_rtx_ok_for_base_p (h8300_get_index (XEXP (x, 0),
mode, 0), strict))
return 1;
return 0;
}
static bool
h8300_hard_regno_mode_ok (unsigned int regno, machine_mode mode)
{
if (TARGET_H8300)
return ((regno & 1) == 0) || (mode == HImode) || (mode == QImode);
else
return regno == MAC_REG ? mode == SImode : 1;
}
static bool
h8300_modes_tieable_p (machine_mode mode1, machine_mode mode2)
{
return (mode1 == mode2
|| ((mode1 == QImode
|| mode1 == HImode
|| ((TARGET_H8300H || TARGET_H8300S) && mode1 == SImode))
&& (mode2 == QImode
|| mode2 == HImode
|| ((TARGET_H8300H || TARGET_H8300S) && mode2 == SImode))));
}
bool
h8300_move_ok (rtx dest, rtx src)
{
rtx addr, other;
if (MEM_P (dest))
{
if (MEM_P (src) || CONSTANT_P (src))
return false;
addr = XEXP (dest, 0);
other = src;
}
else if (MEM_P (src))
{
addr = XEXP (src, 0);
other = dest;
}
else
return true;
if (GET_RTX_CLASS (GET_CODE (addr)) != RTX_AUTOINC)
return true;
addr = XEXP (addr, 0);
if (addr == stack_pointer_rtx)
return register_no_sp_elim_operand (other, VOIDmode);
else
return !reg_overlap_mentioned_p(other, addr);
}

static void
h8300_init_libfuncs (void)
{
set_optab_libfunc (smul_optab, HImode, "__mulhi3");
set_optab_libfunc (sdiv_optab, HImode, "__divhi3");
set_optab_libfunc (udiv_optab, HImode, "__udivhi3");
set_optab_libfunc (smod_optab, HImode, "__modhi3");
set_optab_libfunc (umod_optab, HImode, "__umodhi3");
}

static rtx
h8300_function_value (const_tree ret_type,
const_tree fn_decl_or_type ATTRIBUTE_UNUSED,
bool outgoing ATTRIBUTE_UNUSED)
{
return gen_rtx_REG (TYPE_MODE (ret_type), R0_REG);
}
static rtx
h8300_libcall_value (machine_mode mode, const_rtx fun ATTRIBUTE_UNUSED)
{
return gen_rtx_REG (mode, R0_REG);
}
static bool
h8300_function_value_regno_p (const unsigned int regno)
{
return (regno == R0_REG);
}
static bool
h8300_return_in_memory (const_tree type, const_tree fntype ATTRIBUTE_UNUSED)
{
return (TYPE_MODE (type) == BLKmode
|| GET_MODE_SIZE (TYPE_MODE (type)) > (TARGET_H8300 ? 4 : 8));
}

static void
h8300_trampoline_init (rtx m_tramp, tree fndecl, rtx cxt)
{
rtx fnaddr = XEXP (DECL_RTL (fndecl), 0);
rtx mem;
if (Pmode == HImode)
{
mem = adjust_address (m_tramp, HImode, 0);
emit_move_insn (mem, GEN_INT (0x7903));
mem = adjust_address (m_tramp, Pmode, 2);
emit_move_insn (mem, cxt);
mem = adjust_address (m_tramp, HImode, 4);
emit_move_insn (mem, GEN_INT (0x5a00));
mem = adjust_address (m_tramp, Pmode, 6);
emit_move_insn (mem, fnaddr);
}
else
{
rtx tem;
mem = adjust_address (m_tramp, HImode, 0);
emit_move_insn (mem, GEN_INT (0x7a03));
mem = adjust_address (m_tramp, Pmode, 2);
emit_move_insn (mem, cxt);
tem = copy_to_reg (fnaddr);
emit_insn (gen_andsi3 (tem, tem, GEN_INT (0x00ffffff)));
emit_insn (gen_iorsi3 (tem, tem, GEN_INT (0x5a000000)));
mem = adjust_address (m_tramp, SImode, 6);
emit_move_insn (mem, tem);
}
}
poly_int64
h8300_push_rounding (poly_int64 bytes)
{
return ((bytes + PARM_BOUNDARY / 8 - 1) & (-PARM_BOUNDARY / 8));
}

#undef TARGET_ATTRIBUTE_TABLE
#define TARGET_ATTRIBUTE_TABLE h8300_attribute_table
#undef TARGET_ASM_ALIGNED_HI_OP
#define TARGET_ASM_ALIGNED_HI_OP "\t.word\t"
#undef TARGET_ASM_FILE_START
#define TARGET_ASM_FILE_START h8300_file_start
#undef TARGET_ASM_FILE_START_FILE_DIRECTIVE
#define TARGET_ASM_FILE_START_FILE_DIRECTIVE true
#undef TARGET_ASM_FILE_END
#define TARGET_ASM_FILE_END h8300_file_end
#undef TARGET_PRINT_OPERAND
#define TARGET_PRINT_OPERAND h8300_print_operand
#undef TARGET_PRINT_OPERAND_ADDRESS
#define TARGET_PRINT_OPERAND_ADDRESS h8300_print_operand_address
#undef TARGET_PRINT_OPERAND_PUNCT_VALID_P
#define TARGET_PRINT_OPERAND_PUNCT_VALID_P h8300_print_operand_punct_valid_p
#undef TARGET_ENCODE_SECTION_INFO
#define TARGET_ENCODE_SECTION_INFO h8300_encode_section_info
#undef TARGET_INSERT_ATTRIBUTES
#define TARGET_INSERT_ATTRIBUTES h8300_insert_attributes
#undef TARGET_REGISTER_MOVE_COST
#define TARGET_REGISTER_MOVE_COST h8300_register_move_cost
#undef TARGET_RTX_COSTS
#define TARGET_RTX_COSTS h8300_rtx_costs
#undef TARGET_INIT_LIBFUNCS
#define TARGET_INIT_LIBFUNCS h8300_init_libfuncs
#undef TARGET_FUNCTION_VALUE
#define TARGET_FUNCTION_VALUE h8300_function_value
#undef TARGET_LIBCALL_VALUE
#define TARGET_LIBCALL_VALUE h8300_libcall_value
#undef TARGET_FUNCTION_VALUE_REGNO_P
#define TARGET_FUNCTION_VALUE_REGNO_P h8300_function_value_regno_p
#undef TARGET_RETURN_IN_MEMORY
#define TARGET_RETURN_IN_MEMORY h8300_return_in_memory
#undef TARGET_FUNCTION_ARG
#define TARGET_FUNCTION_ARG h8300_function_arg
#undef TARGET_FUNCTION_ARG_ADVANCE
#define TARGET_FUNCTION_ARG_ADVANCE h8300_function_arg_advance
#undef  TARGET_MACHINE_DEPENDENT_REORG
#define TARGET_MACHINE_DEPENDENT_REORG h8300_reorg
#undef TARGET_HARD_REGNO_SCRATCH_OK
#define TARGET_HARD_REGNO_SCRATCH_OK h8300_hard_regno_scratch_ok
#undef TARGET_HARD_REGNO_MODE_OK
#define TARGET_HARD_REGNO_MODE_OK h8300_hard_regno_mode_ok
#undef TARGET_MODES_TIEABLE_P
#define TARGET_MODES_TIEABLE_P h8300_modes_tieable_p
#undef TARGET_LRA_P
#define TARGET_LRA_P hook_bool_void_false
#undef TARGET_LEGITIMATE_ADDRESS_P
#define TARGET_LEGITIMATE_ADDRESS_P	h8300_legitimate_address_p
#undef TARGET_CAN_ELIMINATE
#define TARGET_CAN_ELIMINATE h8300_can_eliminate
#undef TARGET_CONDITIONAL_REGISTER_USAGE
#define TARGET_CONDITIONAL_REGISTER_USAGE h8300_conditional_register_usage
#undef TARGET_TRAMPOLINE_INIT
#define TARGET_TRAMPOLINE_INIT h8300_trampoline_init
#undef TARGET_OPTION_OVERRIDE
#define TARGET_OPTION_OVERRIDE h8300_option_override
#undef TARGET_MODE_DEPENDENT_ADDRESS_P
#define TARGET_MODE_DEPENDENT_ADDRESS_P h8300_mode_dependent_address_p
struct gcc_target targetm = TARGET_INITIALIZER;

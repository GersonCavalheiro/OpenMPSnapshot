#define IN_TARGET_CODE 1
#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "target.h"
#include "c-family/c-common.h"
#include "stor-layout.h"
#include "langhooks.h"
#include "memmodel.h"
#include "tm_p.h"
enum avr_builtin_id
{
#define DEF_BUILTIN(NAME, N_ARGS, TYPE, CODE, LIBNAME)  \
AVR_BUILTIN_ ## NAME,
#include "builtins.def"
#undef DEF_BUILTIN
AVR_BUILTIN_COUNT
};
static tree
avr_resolve_overloaded_builtin (unsigned int iloc, tree fndecl, void *vargs)
{
tree type0, type1, fold = NULL_TREE;
enum avr_builtin_id id = AVR_BUILTIN_COUNT;
location_t loc = (location_t) iloc;
vec<tree, va_gc> &args = * (vec<tree, va_gc>*) vargs;
switch (DECL_FUNCTION_CODE (fndecl))
{
default:
break;
case AVR_BUILTIN_ABSFX:
if (args.length() != 1)
{
error_at (loc, "%qs expects 1 argument but %d given",
"absfx", (int) args.length());
fold = error_mark_node;
break;
}
type0 = TREE_TYPE (args[0]);
if (!FIXED_POINT_TYPE_P (type0))
{
error_at (loc, "%qs expects a fixed-point value as argument",
"absfx");
fold = error_mark_node;
}
switch (TYPE_MODE (type0))
{
case E_QQmode: id = AVR_BUILTIN_ABSHR; break;
case E_HQmode: id = AVR_BUILTIN_ABSR; break;
case E_SQmode: id = AVR_BUILTIN_ABSLR; break;
case E_DQmode: id = AVR_BUILTIN_ABSLLR; break;
case E_HAmode: id = AVR_BUILTIN_ABSHK; break;
case E_SAmode: id = AVR_BUILTIN_ABSK; break;
case E_DAmode: id = AVR_BUILTIN_ABSLK; break;
case E_TAmode: id = AVR_BUILTIN_ABSLLK; break;
case E_UQQmode:
case E_UHQmode:
case E_USQmode:
case E_UDQmode:
case E_UHAmode:
case E_USAmode:
case E_UDAmode:
case E_UTAmode:
warning_at (loc, 0, "using %qs with unsigned type has no effect",
"absfx");
return args[0];
default:
error_at (loc, "no matching fixed-point overload found for %qs",
"absfx");
fold = error_mark_node;
break;
}
fold = targetm.builtin_decl (id, true);
if (fold != error_mark_node)
fold = build_function_call_vec (loc, vNULL, fold, &args, NULL);
break; 
case AVR_BUILTIN_ROUNDFX:
if (args.length() != 2)
{
error_at (loc, "%qs expects 2 arguments but %d given",
"roundfx", (int) args.length());
fold = error_mark_node;
break;
}
type0 = TREE_TYPE (args[0]);
type1 = TREE_TYPE (args[1]);
if (!FIXED_POINT_TYPE_P (type0))
{
error_at (loc, "%qs expects a fixed-point value as first argument",
"roundfx");
fold = error_mark_node;
}
if (!INTEGRAL_TYPE_P (type1))
{
error_at (loc, "%qs expects an integer value as second argument",
"roundfx");
fold = error_mark_node;
}
switch (TYPE_MODE (type0))
{
case E_QQmode: id = AVR_BUILTIN_ROUNDHR; break;
case E_HQmode: id = AVR_BUILTIN_ROUNDR; break;
case E_SQmode: id = AVR_BUILTIN_ROUNDLR; break;
case E_DQmode: id = AVR_BUILTIN_ROUNDLLR; break;
case E_UQQmode: id = AVR_BUILTIN_ROUNDUHR; break;
case E_UHQmode: id = AVR_BUILTIN_ROUNDUR; break;
case E_USQmode: id = AVR_BUILTIN_ROUNDULR; break;
case E_UDQmode: id = AVR_BUILTIN_ROUNDULLR; break;
case E_HAmode: id = AVR_BUILTIN_ROUNDHK; break;
case E_SAmode: id = AVR_BUILTIN_ROUNDK; break;
case E_DAmode: id = AVR_BUILTIN_ROUNDLK; break;
case E_TAmode: id = AVR_BUILTIN_ROUNDLLK; break;
case E_UHAmode: id = AVR_BUILTIN_ROUNDUHK; break;
case E_USAmode: id = AVR_BUILTIN_ROUNDUK; break;
case E_UDAmode: id = AVR_BUILTIN_ROUNDULK; break;
case E_UTAmode: id = AVR_BUILTIN_ROUNDULLK; break;
default:
error_at (loc, "no matching fixed-point overload found for %qs",
"roundfx");
fold = error_mark_node;
break;
}
fold = targetm.builtin_decl (id, true);
if (fold != error_mark_node)
fold = build_function_call_vec (loc, vNULL, fold, &args, NULL);
break; 
case AVR_BUILTIN_COUNTLSFX:
if (args.length() != 1)
{
error_at (loc, "%qs expects 1 argument but %d given",
"countlsfx", (int) args.length());
fold = error_mark_node;
break;
}
type0 = TREE_TYPE (args[0]);
if (!FIXED_POINT_TYPE_P (type0))
{
error_at (loc, "%qs expects a fixed-point value as first argument",
"countlsfx");
fold = error_mark_node;
}
switch (TYPE_MODE (type0))
{
case E_QQmode: id = AVR_BUILTIN_COUNTLSHR; break;
case E_HQmode: id = AVR_BUILTIN_COUNTLSR; break;
case E_SQmode: id = AVR_BUILTIN_COUNTLSLR; break;
case E_DQmode: id = AVR_BUILTIN_COUNTLSLLR; break;
case E_UQQmode: id = AVR_BUILTIN_COUNTLSUHR; break;
case E_UHQmode: id = AVR_BUILTIN_COUNTLSUR; break;
case E_USQmode: id = AVR_BUILTIN_COUNTLSULR; break;
case E_UDQmode: id = AVR_BUILTIN_COUNTLSULLR; break;
case E_HAmode: id = AVR_BUILTIN_COUNTLSHK; break;
case E_SAmode: id = AVR_BUILTIN_COUNTLSK; break;
case E_DAmode: id = AVR_BUILTIN_COUNTLSLK; break;
case E_TAmode: id = AVR_BUILTIN_COUNTLSLLK; break;
case E_UHAmode: id = AVR_BUILTIN_COUNTLSUHK; break;
case E_USAmode: id = AVR_BUILTIN_COUNTLSUK; break;
case E_UDAmode: id = AVR_BUILTIN_COUNTLSULK; break;
case E_UTAmode: id = AVR_BUILTIN_COUNTLSULLK; break;
default:
error_at (loc, "no matching fixed-point overload found for %qs",
"countlsfx");
fold = error_mark_node;
break;
}
fold = targetm.builtin_decl (id, true);
if (fold != error_mark_node)
fold = build_function_call_vec (loc, vNULL, fold, &args, NULL);
break; 
}
return fold;
}
void
avr_register_target_pragmas (void)
{
gcc_assert (ADDR_SPACE_GENERIC == ADDR_SPACE_RAM);
for (int i = 0; i < ADDR_SPACE_COUNT; i++)
{
gcc_assert (i == avr_addrspace[i].id);
if (!ADDR_SPACE_GENERIC_P (i))
c_register_addr_space (avr_addrspace[i].name, avr_addrspace[i].id);
}
targetm.resolve_overloaded_builtin = avr_resolve_overloaded_builtin;
}
static char*
avr_toupper (char *up, const char *lo)
{
char *up0 = up;
for (; *lo; lo++, up++)
*up = TOUPPER (*lo);
*up = '\0';
return up0;
}
void
avr_cpu_cpp_builtins (struct cpp_reader *pfile)
{
builtin_define_std ("AVR");
if (avr_arch->macro)
cpp_define_formatted (pfile, "__AVR_ARCH__=%s", avr_arch->macro);
if (AVR_HAVE_RAMPD)    cpp_define (pfile, "__AVR_HAVE_RAMPD__");
if (AVR_HAVE_RAMPX)    cpp_define (pfile, "__AVR_HAVE_RAMPX__");
if (AVR_HAVE_RAMPY)    cpp_define (pfile, "__AVR_HAVE_RAMPY__");
if (AVR_HAVE_RAMPZ)    cpp_define (pfile, "__AVR_HAVE_RAMPZ__");
if (AVR_HAVE_ELPM)     cpp_define (pfile, "__AVR_HAVE_ELPM__");
if (AVR_HAVE_ELPMX)    cpp_define (pfile, "__AVR_HAVE_ELPMX__");
if (AVR_HAVE_MOVW)     cpp_define (pfile, "__AVR_HAVE_MOVW__");
if (AVR_HAVE_LPMX)     cpp_define (pfile, "__AVR_HAVE_LPMX__");
if (avr_arch->asm_only)
cpp_define (pfile, "__AVR_ASM_ONLY__");
if (AVR_HAVE_MUL)
{
cpp_define (pfile, "__AVR_ENHANCED__");
cpp_define (pfile, "__AVR_HAVE_MUL__");
}
if (AVR_HAVE_JMP_CALL)
cpp_define (pfile, "__AVR_HAVE_JMP_CALL__");
if (avr_arch->have_jmp_call)
cpp_define (pfile, "__AVR_MEGA__");
if (AVR_SHORT_CALLS)
cpp_define (pfile, "__AVR_SHORT_CALLS__");
if (AVR_XMEGA)
cpp_define (pfile, "__AVR_XMEGA__");
if (AVR_TINY)
{
cpp_define (pfile, "__AVR_TINY__");
cpp_define_formatted (pfile, "__AVR_TINY_PM_BASE_ADDRESS__=0x%x",
avr_arch->flash_pm_offset);
}
if (avr_arch->flash_pm_offset)
cpp_define_formatted (pfile, "__AVR_PM_BASE_ADDRESS__=0x%x",
avr_arch->flash_pm_offset);
if (AVR_HAVE_EIJMP_EICALL)
{
cpp_define (pfile, "__AVR_HAVE_EIJMP_EICALL__");
cpp_define (pfile, "__AVR_3_BYTE_PC__");
}
else
{
cpp_define (pfile, "__AVR_2_BYTE_PC__");
}
if (AVR_HAVE_8BIT_SP)
cpp_define (pfile, "__AVR_HAVE_8BIT_SP__");
else
cpp_define (pfile, "__AVR_HAVE_16BIT_SP__");
if (AVR_HAVE_SPH)
cpp_define (pfile, "__AVR_HAVE_SPH__");
else
cpp_define (pfile, "__AVR_SP8__");
if (TARGET_NO_INTERRUPTS)
cpp_define (pfile, "__NO_INTERRUPTS__");
if (TARGET_SKIP_BUG)
{
cpp_define (pfile, "__AVR_ERRATA_SKIP__");
if (AVR_HAVE_JMP_CALL)
cpp_define (pfile, "__AVR_ERRATA_SKIP_JMP_CALL__");
}
if (TARGET_RMW)
cpp_define (pfile, "__AVR_ISA_RMW__");
cpp_define_formatted (pfile, "__AVR_SFR_OFFSET__=0x%x",
avr_arch->sfr_offset);
#ifdef WITH_AVRLIBC
cpp_define (pfile, "__WITH_AVRLIBC__");
#endif 
if (lang_GNU_C ())
{
for (int i = 0; i < ADDR_SPACE_COUNT; i++)
if (!ADDR_SPACE_GENERIC_P (i)
&& avr_addr_space_supported_p ((addr_space_t) i))
{
const char *name = avr_addrspace[i].name;
char *Name = (char*) alloca (1 + strlen (name));
cpp_define (pfile, avr_toupper (Name, name));
}
}
#define DEF_BUILTIN(NAME, N_ARGS, TYPE, CODE, LIBNAME)  \
cpp_define (pfile, "__BUILTIN_AVR_" #NAME);
#include "builtins.def"
#undef DEF_BUILTIN
cpp_define_formatted (pfile, "__INT24_MAX__=8388607%s",
INT_TYPE_SIZE == 8 ? "LL" : "L");
cpp_define (pfile, "__INT24_MIN__=(-__INT24_MAX__-1)");
cpp_define_formatted (pfile, "__UINT24_MAX__=16777215%s",
INT_TYPE_SIZE == 8 ? "ULL" : "UL");
}

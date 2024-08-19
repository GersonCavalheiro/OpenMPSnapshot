#define IN_TARGET_CODE 1
#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "target.h"
#include "tree.h"
#include "stringpool.h"
#include "alias.h"
#include "vms-protos.h"
#include "output.h"
#include "dwarf2out.h"
#define VMS_CRTL_MALLOC	(1 << 0)
#define VMS_CRTL_64	(1 << 1)
#define VMS_CRTL_FLOAT32  (1 << 2)
#define VMS_CRTL_FLOAT64  (1 << 3)
#define VMS_CRTL_FLOAT64_VAXD  (1 << 4)
#define VMS_CRTL_FLOAT128 (1 << 5)
#define VMS_CRTL_DPML (1 << 6)
#define VMS_CRTL_NODPML (1 << 7)
#define VMS_CRTL_BSD44	(1 << 8)
#define VMS_CRTL_32ONLY (1 << 9)
#define VMS_CRTL_G_MASK (7 << 10)
#define VMS_CRTL_G_NONE (0 << 10)
#define VMS_CRTL_GA	(1 << 10)
#define VMS_CRTL_GL	(2 << 10)
#define VMS_CRTL_FLOATV2 (1 << 13)
struct vms_crtl_name
{
const char *const name;
unsigned int flags;
};
static const struct vms_crtl_name vms_crtl_names[] =
{
#include "vms-crtlmap.h"
};
#define NBR_CRTL_NAMES (sizeof (vms_crtl_names) / sizeof (*vms_crtl_names))
static GTY(()) vec<tree, va_gc> *aliases_id;
static void
vms_add_crtl_xlat (const char *name, size_t nlen,
const char *id_str, size_t id_len)
{
tree targ;
targ = get_identifier_with_length (name, nlen);
gcc_assert (!IDENTIFIER_TRANSPARENT_ALIAS (targ));
IDENTIFIER_TRANSPARENT_ALIAS (targ) = 1;
TREE_CHAIN (targ) = get_identifier_with_length (id_str, id_len);
vec_safe_push (aliases_id, targ);
}
void
vms_patch_builtins (void)
{
unsigned int i;
if (builtin_decl_implicit_p (BUILT_IN_FWRITE))
set_builtin_decl_implicit_p (BUILT_IN_FWRITE, false);
if (builtin_decl_implicit_p (BUILT_IN_FWRITE_UNLOCKED))
set_builtin_decl_implicit_p (BUILT_IN_FWRITE_UNLOCKED, false);
for (i = 0; i < NBR_CRTL_NAMES; i++)
{
const struct vms_crtl_name *n = &vms_crtl_names[i];
char res[VMS_CRTL_MAXLEN + 3 + 9 + 1 + 1];
int rlen;
int nlen = strlen (n->name);
if ((n->flags & VMS_CRTL_32ONLY)
&& flag_vms_pointer_size == VMS_POINTER_SIZE_64)
continue;
if ((n->flags & VMS_CRTL_DPML)
&& !(n->flags & VMS_CRTL_NODPML))
{
const char *p;
char alt[VMS_CRTL_MAXLEN + 3];
memcpy (res, "MATH$", 5);
rlen = 5;
for (p = n->name; *p; p++)
res[rlen++] = TOUPPER (*p);
res[rlen++] = '_';
res[rlen++] = 'T';
if (!(n->flags & VMS_CRTL_FLOAT64))
vms_add_crtl_xlat (n->name, nlen, res, rlen);
res[rlen - 1] = 'S';
memcpy (alt, n->name, nlen);
alt[nlen] = 'f';
vms_add_crtl_xlat (alt, nlen + 1, res, rlen);
res[rlen - 1] = (LONG_DOUBLE_TYPE_SIZE == 128 ? 'X' : 'T');
alt[nlen] = 'l';
vms_add_crtl_xlat (alt, nlen + 1, res, rlen);
if (!(n->flags & (VMS_CRTL_FLOAT32 | VMS_CRTL_FLOAT64)))
continue;
}
if (n->flags & VMS_CRTL_FLOAT64_VAXD)
continue;
memcpy (res, "decc$", 5);
rlen = 5;
if (n->flags & VMS_CRTL_BSD44)
{
memcpy (res + rlen, "__bsd44_", 8);
rlen += 8;
}
if ((n->flags & VMS_CRTL_G_MASK) != VMS_CRTL_G_NONE)
{
res[rlen++] = 'g';
switch (n->flags & VMS_CRTL_G_MASK)
{
case VMS_CRTL_GA:
res[rlen++] = 'a';
break;
case VMS_CRTL_GL:
res[rlen++] = 'l';
break;
default:
gcc_unreachable ();
}
res[rlen++] = '_';
}
if (n->flags & VMS_CRTL_FLOAT32)
res[rlen++] = 'f';
if (n->flags & VMS_CRTL_FLOAT64)
res[rlen++] = 't';
if ((n->flags & VMS_CRTL_FLOAT128) && LONG_DOUBLE_TYPE_SIZE == 128)
res[rlen++] = 'x';
memcpy (res + rlen, n->name, nlen);
if ((n->flags & VMS_CRTL_64) == 0)
{
rlen += nlen;
if (n->flags & VMS_CRTL_FLOATV2)
{
res[rlen++] = '_';
res[rlen++] = '2';
}
vms_add_crtl_xlat (n->name, nlen, res, rlen);
}
else
{
char alt[VMS_CRTL_MAXLEN + 3];
bool use_64;
alt[0] = '_';
memcpy (alt + 1, n->name, nlen);
alt[1 + nlen + 0] = '3';
alt[1 + nlen + 1] = '2';
alt[1 + nlen + 2] = 0;
vms_add_crtl_xlat (alt, nlen + 3, res, rlen + nlen);
use_64 = (((n->flags & VMS_CRTL_64)
&& flag_vms_pointer_size == VMS_POINTER_SIZE_64)
|| ((n->flags & VMS_CRTL_MALLOC)
&& flag_vms_malloc64
&& flag_vms_pointer_size != VMS_POINTER_SIZE_NONE));
if (!use_64)
vms_add_crtl_xlat (n->name, nlen, res, rlen + nlen);
res[rlen++] = '_';
memcpy (res + rlen, n->name, nlen);
res[rlen + nlen + 0] = '6';
res[rlen + nlen + 1] = '4';
if (use_64)
vms_add_crtl_xlat (n->name, nlen, res, rlen + nlen + 2);
alt[1 + nlen + 0] = '6';
alt[1 + nlen + 1] = '4';
vms_add_crtl_xlat (alt, nlen + 3, res, rlen + nlen + 2);
}
}
}
section *
vms_function_section (tree decl ATTRIBUTE_UNUSED,
enum node_frequency freq ATTRIBUTE_UNUSED,
bool startup ATTRIBUTE_UNUSED,
bool exit ATTRIBUTE_UNUSED)
{
return NULL;
}
#define VMS_MAIN_FLAGS_SYMBOL "__gcc_main_flags"
#define MAIN_FLAG_64BIT (1 << 0)
#define MAIN_FLAG_POSIX (1 << 1)
void
vms_start_function (const char *fnname)
{
#if VMS_DEBUGGING_INFO
if (vms_debug_main
&& debug_info_level > DINFO_LEVEL_NONE
&& strncmp (vms_debug_main, fnname, strlen (vms_debug_main)) == 0)
{
targetm.asm_out.globalize_label (asm_out_file, VMS_DEBUG_MAIN_POINTER);
ASM_OUTPUT_DEF (asm_out_file, VMS_DEBUG_MAIN_POINTER, fnname);
dwarf2out_vms_debug_main_pointer ();
vms_debug_main = 0;
}
#endif
if (strcmp (fnname, "main") == 0)
{
unsigned int flags = 0;
if (flag_vms_pointer_size == VMS_POINTER_SIZE_64)
flags |= MAIN_FLAG_64BIT;
if (!flag_vms_return_codes)
flags |= MAIN_FLAG_POSIX;
targetm.asm_out.globalize_label (asm_out_file, VMS_MAIN_FLAGS_SYMBOL);
assemble_name (asm_out_file, VMS_MAIN_FLAGS_SYMBOL);
fprintf (asm_out_file, " = %u\n", flags);
}
}
#include "gt-vms.h"

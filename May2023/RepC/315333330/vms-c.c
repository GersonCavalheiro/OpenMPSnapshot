#define IN_TARGET_CODE 1
#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "tm.h"
#include "tree.h"
#include "c-family/c-common.h"
#include "c/c-tree.h"
#include "memmodel.h"
#include "tm_p.h"
#include "c-family/c-pragma.h"
#include "toplev.h"
#include "incpath.h"
static void
vms_pragma_nostandard (cpp_reader *pfile ATTRIBUTE_UNUSED)
{
tree x;
if (pragma_lex (&x) != CPP_EOF)
warning (OPT_Wpragmas, "junk at end of #pragma __nostandard");
}
static void
vms_pragma_standard (cpp_reader *pfile ATTRIBUTE_UNUSED)
{
tree x;
if (pragma_lex (&x) != CPP_EOF)
warning (OPT_Wpragmas, "junk at end of #pragma __standard");
}
static int saved_member_alignment;
static void
vms_pragma_member_alignment (cpp_reader *pfile ATTRIBUTE_UNUSED)
{
tree x;
int tok;
const char *arg;
tok = pragma_lex (&x);
if (tok == CPP_EOF)
{
maximum_field_alignment = initial_max_fld_align;
return;
}
if (tok != CPP_NAME)
{
warning (OPT_Wpragmas, "malformed '#pragma member_alignment', ignoring");
return;
}
arg = IDENTIFIER_POINTER (x);
if (arg[0] == '_' && arg[1] == '_')
arg += 2;
if (strcmp (arg, "save") == 0)
saved_member_alignment = maximum_field_alignment;
else if (strcmp (arg, "restore") == 0)
maximum_field_alignment = saved_member_alignment;
else
{
error ("unknown '#pragma member_alignment' name %s", arg);
return;
}
if (pragma_lex (&x) != CPP_EOF)
{
error ("malformed '#pragma member_alignment'");
return;
}
}
static void
vms_pragma_nomember_alignment (cpp_reader *pfile ATTRIBUTE_UNUSED)
{
tree x;
int tok;
tok = pragma_lex (&x);
if (tok == CPP_NAME)
{
const char *arg = IDENTIFIER_POINTER (x);
if (arg[0] == '_' && arg[1] == '_')
arg += 2;
if (strcmp (arg, "byte") == 0)
maximum_field_alignment = 1 * BITS_PER_UNIT;
else if (strcmp (arg, "word") == 0)
maximum_field_alignment = 2 * BITS_PER_UNIT;
else if (strcmp (arg, "longword") == 0)
maximum_field_alignment = 4 * BITS_PER_UNIT;
else if (strcmp (arg, "quadword") == 0)
maximum_field_alignment = 8 * BITS_PER_UNIT;
else if (strcmp (arg, "octaword") == 0)
maximum_field_alignment = 16 * BITS_PER_UNIT;
else
{
error ("unhandled alignment for '#pragma nomember_alignment'");
}
tok = pragma_lex (&x);
}
else
{
maximum_field_alignment = BITS_PER_UNIT;
}
if (tok != CPP_EOF)
{
error ("garbage at end of '#pragma nomember_alignment'");
return;
}
}
enum extern_model_kind
{
extern_model_common_block,
extern_model_relaxed_refdef,
extern_model_strict_refdef,
extern_model_globalvalue
};
static enum extern_model_kind current_extern_model;
static enum extern_model_kind saved_extern_model;
static void
vms_pragma_extern_model (cpp_reader *pfile ATTRIBUTE_UNUSED)
{
tree x;
int tok;
const char *arg;
tok = pragma_lex (&x);
if (tok != CPP_NAME)
{
warning (OPT_Wpragmas, "malformed '#pragma extern_model', ignoring");
return;
}
arg = IDENTIFIER_POINTER (x);
if (arg[0] == '_' && arg[1] == '_')
arg += 2;
if (strcmp (arg, "save") == 0)
saved_extern_model = current_extern_model;
else if (strcmp (arg, "restore") == 0)
current_extern_model = saved_extern_model;
else if (strcmp (arg, "relaxed_refdef") == 0)
current_extern_model = extern_model_relaxed_refdef;
else if (strcmp (arg, "strict_refdef") == 0)
current_extern_model = extern_model_strict_refdef;
else if (strcmp (arg, "common_block") == 0)
current_extern_model = extern_model_common_block;
else if (strcmp (arg, "globalvalue") == 0)
{
sorry ("extern model globalvalue");
return;
}
else
{
error ("unknown '#pragma extern_model' model '%s'", arg);
return;
}
#if 0
if (pragma_lex (&x) != CPP_EOF)
{
permerror (input_location, "junk at end of '#pragma extern_model'");
return;
}
#endif
}
static void
vms_pragma_message (cpp_reader *pfile ATTRIBUTE_UNUSED)
{
#if 0
pedwarn (input_location, OPT_Wpragmas,
"vms '#pragma __message' is ignored");
#endif
}
static GTY(()) tree saved_extern_prefix;
static void
vms_pragma_extern_prefix (cpp_reader * ARG_UNUSED (dummy))
{
enum cpp_ttype tok;
tree x;
tok = pragma_lex (&x);
if (tok == CPP_NAME)
{
const char *op = IDENTIFIER_POINTER (x);
if (!strcmp (op, "__save"))
saved_extern_prefix = pragma_extern_prefix;
else if (!strcmp (op, "__restore"))
pragma_extern_prefix = saved_extern_prefix;
else
warning (OPT_Wpragmas,
"malformed '#pragma __extern_prefix', ignoring");
return;
}
else if (tok != CPP_STRING)
{
warning (OPT_Wpragmas,
"malformed '#pragma __extern_prefix', ignoring");
}
else
{
pragma_extern_prefix = (TREE_STRING_LENGTH (x) > 1 ? x : NULL);
}
}
static machine_mode saved_pointer_mode;
static void
handle_pragma_pointer_size (const char *pragma_name)
{
enum cpp_ttype tok;
tree x;
tok = pragma_lex (&x);
if (tok == CPP_NAME)
{
const char *op = IDENTIFIER_POINTER (x);
if (!strcmp (op, "__save"))
saved_pointer_mode = c_default_pointer_mode;
else if (!strcmp (op, "__restore"))
c_default_pointer_mode = saved_pointer_mode;
else if (!strcmp (op, "__short"))
c_default_pointer_mode = SImode;
else if (!strcmp (op, "__long"))
c_default_pointer_mode = DImode;
else
error ("malformed %<#pragma %s%>, ignoring", pragma_name);
}
else if (tok == CPP_NUMBER)
{
int val;
if (TREE_CODE (x) == INTEGER_CST)
val = TREE_INT_CST_LOW (x);
else
val = -1;
if (val == 32)
c_default_pointer_mode = SImode;
else if (val == 64)
c_default_pointer_mode = DImode;
else
error ("invalid constant in %<#pragma %s%>", pragma_name);
}
else
{
error ("malformed %<#pragma %s%>, ignoring", pragma_name);
}
}
static void
vms_pragma_pointer_size (cpp_reader * ARG_UNUSED (dummy))
{
if (flag_vms_pointer_size == VMS_POINTER_SIZE_NONE)
return;
handle_pragma_pointer_size ("pointer_size");
}
static void
vms_pragma_required_pointer_size (cpp_reader * ARG_UNUSED (dummy))
{
handle_pragma_pointer_size ("required_pointer_size");
}
void
vms_c_register_pragma (void)
{
c_register_pragma (NULL, "__nostandard", vms_pragma_nostandard);
c_register_pragma (NULL, "nostandard", vms_pragma_nostandard);
c_register_pragma (NULL, "__standard", vms_pragma_standard);
c_register_pragma (NULL, "standard", vms_pragma_standard);
c_register_pragma (NULL, "__member_alignment", vms_pragma_member_alignment);
c_register_pragma (NULL, "member_alignment", vms_pragma_member_alignment);
c_register_pragma_with_expansion (NULL, "__nomember_alignment",
vms_pragma_nomember_alignment);
c_register_pragma_with_expansion (NULL, "nomember_alignment",
vms_pragma_nomember_alignment);
c_register_pragma (NULL, "__pointer_size",
vms_pragma_pointer_size);
c_register_pragma (NULL, "__required_pointer_size",
vms_pragma_required_pointer_size);
c_register_pragma (NULL, "__extern_model", vms_pragma_extern_model);
c_register_pragma (NULL, "extern_model", vms_pragma_extern_model);
c_register_pragma (NULL, "__message", vms_pragma_message);
c_register_pragma (NULL, "__extern_prefix", vms_pragma_extern_prefix);
}
static char *
vms_construct_include_filename (const char *fname, cpp_dir *dir)
{
size_t dlen, flen;
char *path;
const char *fbasename = lbasename (fname);
size_t i;
dlen = dir->len;
flen = strlen (fbasename) + 2;
path = XNEWVEC (char, dlen + 1 + flen + 1);
memcpy (path, dir->name, dlen);
if (dlen && !IS_DIR_SEPARATOR (path[dlen - 1]))
path[dlen++] = '/';
for (i = 0; i < flen; i++)
if (fbasename[i] == '.')
break;
else
path[dlen + i] = TOLOWER (fbasename[i]);
path[dlen + i + 0] = '.';
path[dlen + i + 1] = 'h';
path[dlen + i + 2] = 0;
return path;
}
static const char * const vms_std_modules[] = { "rtldef", "starlet_c", NULL };
void
vms_c_register_includes (const char *sysroot,
const char *iprefix ATTRIBUTE_UNUSED, int stdinc)
{
static const char dir_separator_str[] = { DIR_SEPARATOR, 0 };
struct cpp_dir *dir;
if (!stdinc)
return;
for (dir = get_added_cpp_dirs (INC_SYSTEM); dir != NULL; dir = dir->next)
{
const char * const *lib;
for (lib = vms_std_modules; *lib != NULL; lib++)
{
char *path;
struct stat st;
if (sysroot != NULL)
path = concat (sysroot, dir->name, dir_separator_str, *lib, NULL);
else
path = concat (dir->name, dir_separator_str, *lib, NULL);
if (stat (path, &st) == 0 && S_ISDIR (st.st_mode))
{
cpp_dir *p;
p = XNEW (cpp_dir);
p->next = NULL;
p->name = path;
p->sysp = 1;
p->construct = vms_construct_include_filename;
p->user_supplied_p = 0;
add_cpp_dir_path (p, INC_SYSTEM);
}
else
free (path);
}
}
}
void
vms_c_common_override_options (void)
{
flag_allow_parameterless_variadic_functions = TRUE;
switch (flag_vms_pointer_size)
{
case VMS_POINTER_SIZE_NONE:
break;
case VMS_POINTER_SIZE_32:
c_default_pointer_mode = SImode;
break;
case VMS_POINTER_SIZE_64:
c_default_pointer_mode = DImode;
break;
}
}
int
vms_c_get_crtl_ver (void)
{
return VMS_DEFAULT_CRTL_VER;
}
int
vms_c_get_vms_ver (void)
{
return VMS_DEFAULT_VMS_VER;
}

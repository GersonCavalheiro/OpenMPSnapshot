#undef TARGET_SOM
#define TARGET_SOM 1
#undef DBX_USE_BINCL
#define DBX_LINES_FUNCTION_RELATIVE 1
#define DBX_OUTPUT_NULL_N_SO_AT_MAIN_SOURCE_FILE_END
#define LDD_SUFFIX "chatr"
#define PARSE_LDD_OUTPUT(PTR)					\
do {								\
static int in_shlib_list = 0;					\
while (*PTR == ' ') PTR++;					\
if (strncmp (PTR, "shared library list:",			\
sizeof ("shared library list:") - 1) == 0)	\
{								\
PTR = 0;							\
in_shlib_list = 1;					\
}								\
else if (strncmp (PTR, "shared library binding:",		\
sizeof ("shared library binding:") - 1) == 0)\
{								\
PTR = 0;							\
in_shlib_list = 0;					\
}								\
else if (strncmp (PTR, "static branch prediction disabled",	\
sizeof ("static branch prediction disabled") - 1) == 0)\
{								\
PTR = 0;							\
in_shlib_list = 0;					\
}								\
else if (in_shlib_list					\
&&  strncmp (PTR, "dynamic", sizeof ("dynamic") - 1) == 0) \
{								\
PTR += sizeof ("dynamic") - 1;				\
while (*p == ' ') PTR++;					\
}								\
else if (in_shlib_list					\
&& strncmp (PTR, "static", sizeof ("static") - 1) == 0) \
{								\
PTR += sizeof ("static") - 1;				\
while (*p == ' ') PTR++;					\
}								\
else								\
PTR = 0;							\
} while (0)
#ifndef HP_FP_ARG_DESCRIPTOR_REVERSED
#define ASM_DOUBLE_ARG_DESCRIPTORS(FILE, ARG0, ARG1)	\
do { fprintf (FILE, ",ARGW%d=FR", (ARG0));		\
fprintf (FILE, ",ARGW%d=FU", (ARG1));} while (0)
#define DFMODE_RETURN_STRING ",RTNVAL=FU"
#define SFMODE_RETURN_STRING ",RTNVAL=FR"
#else
#define ASM_DOUBLE_ARG_DESCRIPTORS(FILE, ARG0, ARG1)	\
do { fprintf (FILE, ",ARGW%d=FU", (ARG0));		\
fprintf (FILE, ",ARGW%d=FR", (ARG1));} while (0)
#define DFMODE_RETURN_STRING ",RTNVAL=FR"
#define SFMODE_RETURN_STRING ",RTNVAL=FU"
#endif

#define ASM_DECLARE_FUNCTION_NAME(FILE, NAME, DECL) \
do { tree fntype = TREE_TYPE (TREE_TYPE (DECL));			\
tree tree_type = TREE_TYPE (DECL);				\
tree parm;							\
int i;								\
if (TREE_PUBLIC (DECL) || TARGET_GAS)				\
{ 								\
if (TREE_PUBLIC (DECL))					\
{							\
fputs ("\t.EXPORT ", FILE);				\
assemble_name (FILE, NAME);				\
fputs (",ENTRY,PRIV_LEV=3", FILE);			\
}							\
else							\
{							\
fputs ("\t.PARAM ", FILE);				\
assemble_name (FILE, NAME);				\
fputs (",PRIV_LEV=3", FILE);				\
}							\
for (parm = DECL_ARGUMENTS (DECL), i = 0; parm && i < 4;	\
parm = DECL_CHAIN (parm))				\
{							\
tree type = DECL_ARG_TYPE (parm);			\
machine_mode mode = TYPE_MODE (type);			\
if (mode == SFmode && ! TARGET_SOFT_FLOAT)		\
fprintf (FILE, ",ARGW%d=FR", i++);			\
else if (mode == DFmode && ! TARGET_SOFT_FLOAT)	\
{							\
if (i <= 2)					\
{						\
if (i == 1) i++;				\
ASM_DOUBLE_ARG_DESCRIPTORS (FILE, i++, i++);	\
}						\
else						\
break;						\
}							\
else							\
{							\
int arg_size = pa_function_arg_size (mode, type);	\
\
if (arg_size > 2 || TREE_ADDRESSABLE (type))	\
arg_size = 1;					\
if (arg_size == 2 && i <= 2)			\
{						\
if (i == 1) i++;				\
fprintf (FILE, ",ARGW%d=GR", i++);		\
fprintf (FILE, ",ARGW%d=GR", i++);		\
}						\
else if (arg_size == 1)				\
fprintf (FILE, ",ARGW%d=GR", i++);		\
else						\
i += arg_size;					\
}							\
}							\
\
if (stdarg_p (tree_type))					\
{							\
for (; i < 4; i++)					\
fprintf (FILE, ",ARGW%d=GR", i);			\
}							\
if (TYPE_MODE (fntype) == DFmode && ! TARGET_SOFT_FLOAT)	\
fputs (DFMODE_RETURN_STRING, FILE);			\
else if (TYPE_MODE (fntype) == SFmode && ! TARGET_SOFT_FLOAT) \
fputs (SFMODE_RETURN_STRING, FILE);			\
else if (fntype != void_type_node)				\
fputs (",RTNVAL=GR", FILE);				\
fputs ("\n", FILE);					\
}} while (0)
#define TARGET_ASM_FILE_START pa_som_file_start
#define TARGET_ASM_INIT_SECTIONS pa_som_asm_init_sections
#define DATA_SECTION_ASM_OP "\t.SPACE $PRIVATE$\n\t.SUBSPA $DATA$\n"
#define BSS_SECTION_ASM_OP "\t.SPACE $PRIVATE$\n\t.SUBSPA $BSS$\n"
#define ASM_OUTPUT_EXTERNAL(FILE, DECL, NAME) \
pa_hpux_asm_output_external ((FILE), (DECL), (NAME))
#define ASM_OUTPUT_EXTERNAL_REAL(FILE, DECL, NAME) \
do { fputs ("\t.IMPORT ", FILE);					\
assemble_name_raw (FILE, NAME);					\
if (FUNCTION_NAME_P (NAME))					\
fputs (",CODE\n", FILE);					\
else								\
fputs (",DATA\n", FILE);					\
} while (0)
#define ASM_OUTPUT_EXTERNAL_LIBCALL(FILE, RTL) \
do { const char *name;						\
tree id;								\
\
if (!function_label_operand (RTL, VOIDmode))			\
pa_encode_label (RTL);						\
\
name = targetm.strip_name_encoding (XSTR ((RTL), 0));		\
id = maybe_get_identifier (name);				\
if (!id || !TREE_SYMBOL_REFERENCED (id))				\
{								\
fputs ("\t.IMPORT ", FILE);					\
assemble_name_raw (FILE, XSTR ((RTL), 0));		       	\
fputs (",CODE\n", FILE);					\
}								\
} while (0)
#define DO_GLOBAL_DTORS_BODY			\
do {						\
extern void __gcc_plt_call (void);		\
void (*reference)(void) = &__gcc_plt_call;	\
func_ptr *p;					\
__asm__ ("" : : "r" (reference));		\
for (p = __DTOR_LIST__ + 1; *p; )		\
(*p++) ();					\
} while (0)
#define MAX_OFILE_ALIGNMENT 32768
#undef TARGET_ALWAYS_STRIP_DOTDOT
#define TARGET_ALWAYS_STRIP_DOTDOT true
#ifdef HAVE_GAS_WEAK
#define TARGET_SUPPORTS_WEAK (TARGET_SOM_SDEF && TARGET_GAS)
#else
#define TARGET_SUPPORTS_WEAK 0
#endif
#ifdef HAVE_GAS_NSUBSPA_COMDAT
#define SUPPORTS_SOM_COMDAT (TARGET_GAS)
#else
#define SUPPORTS_SOM_COMDAT 0
#endif
#define SUPPORTS_ONE_ONLY (TARGET_SUPPORTS_WEAK || SUPPORTS_SOM_COMDAT)
#define MAKE_DECL_ONE_ONLY(DECL) \
do {									\
if (TREE_CODE (DECL) == VAR_DECL					\
&& (DECL_INITIAL (DECL) == 0					\
|| DECL_INITIAL (DECL) == error_mark_node))			\
DECL_COMMON (DECL) = 1;						\
else if (TARGET_SUPPORTS_WEAK)					\
DECL_WEAK (DECL) = 1;						\
} while (0)
#define ASM_WEAKEN_LABEL(FILE,NAME) \
do { fputs ("\t.weak\t", FILE);				\
assemble_name (FILE, NAME);				\
fputc ('\n', FILE);					\
targetm.asm_out.globalize_label (FILE, NAME);		\
} while (0)
#define GTHREAD_USE_WEAK 0
#define SHLIB_SUFFIX ".sl"
#define TARGET_HAVE_NAMED_SECTIONS false
#define TARGET_ASM_TM_CLONE_TABLE_SECTION pa_som_tm_clone_table_section
#define EH_FRAME_THROUGH_COLLECT2

#ifndef GCC_NVPTX_PROTOS_H
#define GCC_NVPTX_PROTOS_H
enum nvptx_shuffle_kind
{
SHUFFLE_UP,
SHUFFLE_DOWN,
SHUFFLE_BFLY,
SHUFFLE_IDX,
SHUFFLE_MAX
};
extern void nvptx_declare_function_name (FILE *, const char *, const_tree decl);
extern void nvptx_declare_object_name (FILE *file, const char *name,
const_tree decl);
extern void nvptx_output_aligned_decl (FILE *file, const char *name,
const_tree decl,
HOST_WIDE_INT size, unsigned align);
extern void nvptx_function_end (FILE *);
extern void nvptx_output_skip (FILE *, unsigned HOST_WIDE_INT);
extern void nvptx_output_ascii (FILE *, const char *, unsigned HOST_WIDE_INT);
extern void nvptx_register_pragmas (void);
extern unsigned int nvptx_data_alignment (const_tree, unsigned int);
#ifdef RTX_CODE
extern void nvptx_expand_oacc_fork (unsigned);
extern void nvptx_expand_oacc_join (unsigned);
extern void nvptx_expand_call (rtx, rtx);
extern rtx nvptx_gen_shuffle (rtx, rtx, rtx, nvptx_shuffle_kind);
extern rtx nvptx_expand_compare (rtx);
extern const char *nvptx_ptx_type_from_mode (machine_mode, bool);
extern const char *nvptx_output_mov_insn (rtx, rtx);
extern const char *nvptx_output_call_insn (rtx_insn *, rtx, rtx);
extern const char *nvptx_output_return (void);
extern const char *nvptx_output_set_softstack (unsigned);
extern const char *nvptx_output_simt_enter (rtx, rtx, rtx);
extern const char *nvptx_output_simt_exit (rtx);
#endif
#endif


const char *    rl78_addsi3_internal (rtx *, unsigned int);
void		rl78_emit_eh_epilogue (rtx);
void		rl78_expand_compare (rtx *);
void		rl78_expand_movsi (rtx *);
void		rl78_split_movsi (rtx *, machine_mode);
void 		rl78_split_movdi (rtx *, enum machine_mode);
int		rl78_force_nonfar_2 (rtx *, rtx (*gen)(rtx,rtx));
int		rl78_force_nonfar_3 (rtx *, rtx (*gen)(rtx,rtx,rtx));
void		rl78_expand_eh_epilogue (rtx);
void		rl78_expand_epilogue (void);
void		rl78_expand_prologue (void);
int		rl78_far_p (rtx x);
bool		rl78_hl_b_c_addr_p (rtx);
int		rl78_initial_elimination_offset (int, int);
bool		rl78_as_legitimate_address (machine_mode, rtx,
bool, addr_space_t);
int		rl78_legitimize_reload_address (rtx *, machine_mode, int,int, int);
enum reg_class	rl78_mode_code_base_reg_class (machine_mode, addr_space_t, int, int);
bool		rl78_peep_movhi_p (rtx *);
bool		rl78_real_insns_ok (void);
void		rl78_register_pragmas (void);
bool		rl78_regno_mode_code_ok_for_base_p (int, machine_mode, addr_space_t, int, int);
void		rl78_setup_peep_movhi (rtx *);
bool		rl78_virt_insns_ok (void);
bool		rl78_es_addr (rtx);
rtx		rl78_es_base (rtx);
bool		rl78_flags_already_set (rtx, rtx);
void		rl78_output_symbol_ref (FILE *, rtx);
void		rl78_output_labelref (FILE *, const char *);
int		rl78_saddr_p (rtx x);
int		rl78_sfr_p (rtx x);
void		rl78_output_aligned_common (FILE *, tree, const char *,
int, int, int);
int		rl78_one_far_p (rtx *operands, int num_operands);
#ifdef RTX_CODE
#ifdef HAVE_MACHINE_MODES
rtx rl78_emit_libcall (const char*, enum rtx_code,
enum machine_mode, enum machine_mode,
int, rtx*);
#endif
#endif

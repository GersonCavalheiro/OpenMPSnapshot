#ifndef GCC_MSP430_PROTOS_H
#define GCC_MSP430_PROTOS_H
bool	msp430_do_not_relax_short_jumps (void);
rtx	msp430_eh_return_stackadj_rtx (void);
void	msp430_expand_eh_return (rtx);
void	msp430_expand_epilogue (int);
void	msp430_expand_helper (rtx *operands, const char *, bool);
void	msp430_expand_prologue (void);
const char * msp430x_extendhisi (rtx *);
void	msp430_fixup_compare_operands (machine_mode, rtx *);
int	msp430_hard_regno_nregs_has_padding (int, machine_mode);
int	msp430_hard_regno_nregs_with_padding (int, machine_mode);
bool    msp430_hwmult_enabled (void);
rtx	msp430_incoming_return_addr_rtx (void);
void	msp430_init_cumulative_args (CUMULATIVE_ARGS *, tree, rtx, tree, int);
int	msp430_initial_elimination_offset (int, int);
bool    msp430_is_interrupt_func (void);
const char * msp430x_logical_shift_right (rtx);
const char * msp430_mcu_name (void);
void    msp430_output_aligned_decl_common (FILE *, const tree, const char *, unsigned HOST_WIDE_INT, unsigned);
void	msp430_output_labelref (FILE *, const char *);
void	msp430_register_pragmas (void);
rtx	msp430_return_addr_rtx (int);
void	msp430_split_movsi (rtx *);
void    msp430_start_function (FILE *, const char *, tree);
rtx	msp430_subreg (machine_mode, rtx, machine_mode, int);
bool    msp430_use_f5_series_hwmult (void);
#endif 

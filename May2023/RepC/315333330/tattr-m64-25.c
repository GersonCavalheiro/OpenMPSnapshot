void fn_default_start (void) { }
__attribute__ ((target ("small-exec")))
void fn_att_1 (void) { }
void fn_att_1_default (void) { }
__attribute__ ((target ("no-small-exec")))
void fn_att_0 (void) { }
void fn_att_0_default (void) { }
__attribute__ ((target ("no-small-exec,small-exec")))
void fn_att_0_1 (void) { }
__attribute__ ((target ("small-exec,no-small-exec")))
void fn_att_1_0 (void) { }
#pragma GCC target ("small-exec")
void fn_pragma_1 (void) { }
#pragma GCC reset_options
void fn_pragma_1_default (void) { }
#pragma GCC target ("no-small-exec")
void fn_pragma_0 (void) { }
#pragma GCC reset_options
void fn_pragma_0_default (void) { }
#pragma GCC target ("no-small-exec")
#pragma GCC target ("small-exec")
void fn_pragma_0_1 (void) { }
#pragma GCC reset_options
#pragma GCC target ("small-exec")
#pragma GCC target ("no-small-exec")
void fn_pragma_1_0 (void) { }
#pragma GCC reset_options
#pragma GCC target ("small-exec")
__attribute__ ((target ("small-exec")))
void fn_pragma_1_att_1 (void) { }
#pragma GCC reset_options
#pragma GCC target ("small-exec")
__attribute__ ((target ("small-exec")))
void fn_pragma_0_att_1 (void) { }
#pragma GCC reset_options
#pragma GCC target ("small-exec")
__attribute__ ((target ("no-small-exec")))
void fn_pragma_1_att_0 (void) { }
#pragma GCC reset_options
#pragma GCC target ("small-exec")
__attribute__ ((target ("no-small-exec")))
void fn_pragma_0_att_0 (void) { }
#pragma GCC reset_options
void fn_default_end (void) { }

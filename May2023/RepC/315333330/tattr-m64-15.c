void fn_default_start (void) { }
__attribute__ ((target ("hard-dfp")))
void fn_att_1 (void) { }
void fn_att_1_default (void) { }
__attribute__ ((target ("no-hard-dfp")))
void fn_att_0 (void) { }
void fn_att_0_default (void) { }
__attribute__ ((target ("no-hard-dfp,hard-dfp")))
void fn_att_0_1 (void) { }
__attribute__ ((target ("hard-dfp,no-hard-dfp")))
void fn_att_1_0 (void) { }
#pragma GCC target ("hard-dfp")
void fn_pragma_1 (void) { }
#pragma GCC reset_options
void fn_pragma_1_default (void) { }
#pragma GCC target ("no-hard-dfp")
void fn_pragma_0 (void) { }
#pragma GCC reset_options
void fn_pragma_0_default (void) { }
#pragma GCC target ("no-hard-dfp")
#pragma GCC target ("hard-dfp")
void fn_pragma_0_1 (void) { }
#pragma GCC reset_options
#pragma GCC target ("hard-dfp")
#pragma GCC target ("no-hard-dfp")
void fn_pragma_1_0 (void) { }
#pragma GCC reset_options
#pragma GCC target ("hard-dfp")
__attribute__ ((target ("hard-dfp")))
void fn_pragma_1_att_1 (void) { }
#pragma GCC reset_options
#pragma GCC target ("hard-dfp")
__attribute__ ((target ("hard-dfp")))
void fn_pragma_0_att_1 (void) { }
#pragma GCC reset_options
#pragma GCC target ("hard-dfp")
__attribute__ ((target ("no-hard-dfp")))
void fn_pragma_1_att_0 (void) { }
#pragma GCC reset_options
#pragma GCC target ("hard-dfp")
__attribute__ ((target ("no-hard-dfp")))
void fn_pragma_0_att_0 (void) { }
#pragma GCC reset_options
void fn_default_end (void) { }

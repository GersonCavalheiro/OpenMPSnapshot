void fn_default_start (void) { }
__attribute__ ((target ("mvcle")))
void fn_att_1 (void) { }
void fn_att_1_default (void) { }
__attribute__ ((target ("no-mvcle")))
void fn_att_0 (void) { }
void fn_att_0_default (void) { }
__attribute__ ((target ("no-mvcle,mvcle")))
void fn_att_0_1 (void) { }
__attribute__ ((target ("mvcle,no-mvcle")))
void fn_att_1_0 (void) { }
#pragma GCC target ("mvcle")
void fn_pragma_1 (void) { }
#pragma GCC reset_options
void fn_pragma_1_default (void) { }
#pragma GCC target ("no-mvcle")
void fn_pragma_0 (void) { }
#pragma GCC reset_options
void fn_pragma_0_default (void) { }
#pragma GCC target ("no-mvcle")
#pragma GCC target ("mvcle")
void fn_pragma_0_1 (void) { }
#pragma GCC reset_options
#pragma GCC target ("mvcle")
#pragma GCC target ("no-mvcle")
void fn_pragma_1_0 (void) { }
#pragma GCC reset_options
#pragma GCC target ("mvcle")
__attribute__ ((target ("mvcle")))
void fn_pragma_1_att_1 (void) { }
#pragma GCC reset_options
#pragma GCC target ("mvcle")
__attribute__ ((target ("mvcle")))
void fn_pragma_0_att_1 (void) { }
#pragma GCC reset_options
#pragma GCC target ("mvcle")
__attribute__ ((target ("no-mvcle")))
void fn_pragma_1_att_0 (void) { }
#pragma GCC reset_options
#pragma GCC target ("mvcle")
__attribute__ ((target ("no-mvcle")))
void fn_pragma_0_att_0 (void) { }
#pragma GCC reset_options
void fn_default_end (void) { }
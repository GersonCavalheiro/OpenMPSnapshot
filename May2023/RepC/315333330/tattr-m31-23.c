void fn_default_start (void) { }
__attribute__ ((target ("packed-stack")))
void fn_att_1 (void) { }
void fn_att_1_default (void) { }
__attribute__ ((target ("no-packed-stack")))
void fn_att_0 (void) { }
void fn_att_0_default (void) { }
__attribute__ ((target ("no-packed-stack,packed-stack")))
void fn_att_0_1 (void) { }
__attribute__ ((target ("packed-stack,no-packed-stack")))
void fn_att_1_0 (void) { }
#pragma GCC target ("packed-stack")
void fn_pragma_1 (void) { }
#pragma GCC reset_options
void fn_pragma_1_default (void) { }
#pragma GCC target ("no-packed-stack")
void fn_pragma_0 (void) { }
#pragma GCC reset_options
void fn_pragma_0_default (void) { }
#pragma GCC target ("no-packed-stack")
#pragma GCC target ("packed-stack")
void fn_pragma_0_1 (void) { }
#pragma GCC reset_options
#pragma GCC target ("packed-stack")
#pragma GCC target ("no-packed-stack")
void fn_pragma_1_0 (void) { }
#pragma GCC reset_options
#pragma GCC target ("packed-stack")
__attribute__ ((target ("packed-stack")))
void fn_pragma_1_att_1 (void) { }
#pragma GCC reset_options
#pragma GCC target ("packed-stack")
__attribute__ ((target ("packed-stack")))
void fn_pragma_0_att_1 (void) { }
#pragma GCC reset_options
#pragma GCC target ("packed-stack")
__attribute__ ((target ("no-packed-stack")))
void fn_pragma_1_att_0 (void) { }
#pragma GCC reset_options
#pragma GCC target ("packed-stack")
__attribute__ ((target ("no-packed-stack")))
void fn_pragma_0_att_0 (void) { }
#pragma GCC reset_options
void fn_default_end (void) { }

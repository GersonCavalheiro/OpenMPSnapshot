void fn_default_start (void) { }
__attribute__ ((target ("arch=arch11")))
void fn_att_1 (void) { }
void fn_att_1_default (void) { }
__attribute__ ((target ("arch=arch8")))
void fn_att_0 (void) { }
void fn_att_0_default (void) { }
__attribute__ ((target ("arch=arch8,arch=arch11")))
void fn_att_0_1 (void) { }
__attribute__ ((target ("arch=arch11,arch=arch8")))
void fn_att_1_0 (void) { }
#pragma GCC target ("arch=arch11")
void fn_pragma_1 (void) { }
#pragma GCC reset_options
void fn_pragma_1_default (void) { }
#pragma GCC target ("arch=arch8")
void fn_pragma_0 (void) { }
#pragma GCC reset_options
void fn_pragma_0_default (void) { }
#pragma GCC target ("arch=arch8")
#pragma GCC target ("arch=arch11")
void fn_pragma_0_1 (void) { }
#pragma GCC reset_options
#pragma GCC target ("arch=arch11")
#pragma GCC target ("arch=arch8")
void fn_pragma_1_0 (void) { }
#pragma GCC reset_options
#pragma GCC target ("arch=arch11")
__attribute__ ((target ("arch=arch11")))
void fn_pragma_1_att_1 (void) { }
#pragma GCC reset_options
#pragma GCC target ("arch=arch11")
__attribute__ ((target ("arch=arch11")))
void fn_pragma_0_att_1 (void) { }
#pragma GCC reset_options
#pragma GCC target ("arch=arch11")
__attribute__ ((target ("arch=arch8")))
void fn_pragma_1_att_0 (void) { }
#pragma GCC reset_options
#pragma GCC target ("arch=arch11")
__attribute__ ((target ("arch=arch8")))
void fn_pragma_0_att_0 (void) { }
#pragma GCC reset_options
void fn_default_end (void) { }

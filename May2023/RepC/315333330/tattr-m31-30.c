void fn_default_start (void) { }
#pragma GCC target ("no-zvector")
void fn_pragma_0 (void) { }
#pragma GCC reset_options
void fn_pragma_0_default (void) { }
#pragma GCC target ("zvector")
void fn_pragma_1 (void) { }
#pragma GCC reset_options
void fn_pragma_1_default (void) { }
#pragma GCC target ("zvector")
#pragma GCC target ("no-zvector")
void fn_pragma_1_0 (void) { }
#pragma GCC reset_options
#pragma GCC target ("no-zvector")
#pragma GCC target ("zvector")
void fn_pragma_0_1 (void) { }
#pragma GCC reset_options
void fn_default_end (void) { }

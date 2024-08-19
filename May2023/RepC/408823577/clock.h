#if __INTEL_COMPILER
#pragma warning ( disable : 1418 )
#endif
void start_counter();
void start_counter_copy();
double get_counter();
double get_counter_copy();
double ovhd();
double mhz(int verbose);
double mhz_full(int verbose, int sleeptime);
void start_comp_counter();
double get_comp_counter();

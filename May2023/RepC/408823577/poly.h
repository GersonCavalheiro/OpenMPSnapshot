#if __INTEL_COMPILER
#pragma warning ( disable : 1418 )
#endif
typedef double (*poly_t)(double*, double, int);
void add_function(poly_t f, char *description);
void set_check_function(poly_t f);
void register_functions(void);

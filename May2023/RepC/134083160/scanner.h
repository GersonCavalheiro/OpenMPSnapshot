#ifndef __SCANNER_H__
#define __SCANNER_H__
#include <stdio.h>
extern FILE *yyin;                
extern int   yylex(void);
extern void  sc_set_filename(char *fn);
extern int   __has_omp;            
extern int   __has_ompix;          
extern int   __has_affinitysched;  
extern char *sc_original_file(void);
extern int   sc_original_line(void);
extern int   sc_line(void);
extern int   sc_column(void);
extern int sc_scan_attribute(char **string);
extern void sc_set_start_token(int t);
extern void sc_scan_string(char *s);
extern void sc_pause_openmp();
extern void sc_start_openmp();
#endif

#pragma once


struct radial_grid_t {
int   n = 0; 
float rmax = 0.f; 
double const*    r = nullptr; 
double const*   dr = nullptr; 
double const*  rdr = nullptr; 
double const* r2dr = nullptr; 
double const* rinv = nullptr; 
double anisotropy = 0.;
bool  memory_owner = true;
char  equation = '\0';
}; 


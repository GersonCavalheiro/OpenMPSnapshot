#ifndef DIAGNOSTICSCALAR_H
#define DIAGNOSTICSCALAR_H

#include <fstream>

#include "Diagnostic.h"

class Patch;
class Params;
class SmileiMPI;


struct val_index {
double val;
int index;
};


class Scalar
{
public:
Scalar( std::string name, std::string secondname, unsigned int width, bool allowed ):
name_( name ), secondname_( secondname ), width_( width ), allowed_( allowed )
{};
virtual ~Scalar() {};
virtual inline operator double() const
{
return 0.;
}
virtual inline void reset() { };
std::string name_, secondname_;
unsigned int width_;
bool allowed_;
};

class Scalar_value : public Scalar
{
public:
Scalar_value( std::string name, unsigned int width, bool allowed, std::vector<double> *values ):
Scalar( name, "", width, allowed ), values_( values ), index( values->size() )
{};
~Scalar_value() {};
inline Scalar_value &operator= ( double v )
{
( *values_ )[index] = v;
return *this;
};
inline Scalar_value &operator+= ( double v )
{
#pragma omp atomic
( *values_ )[index]+=v;
return *this;
};
inline operator double() const override
{
return ( *values_ )[index];
} ;
inline void reset() override
{
( *values_ )[index]=0.;
};
std::vector<double> *values_;
unsigned int index;
};

class Scalar_value_location : public Scalar
{
public:
Scalar_value_location( std::string name, std::string secondname, unsigned int width, bool allowed, std::vector<val_index> *values, double reset_value ):
Scalar( name, secondname, width, allowed ), values_( values ), index( values->size() ), reset_value_( reset_value )
{};
~Scalar_value_location() {};
inline Scalar_value_location &operator= ( val_index v )
{
( *values_ )[index] = v;
return *this;
};
inline operator double() const override
{
return ( *values_ )[index].val;
}
inline operator int() const
{
return ( *values_ )[index].index;
}
inline void reset() override
{
( *values_ )[index].val=reset_value_;
( *values_ )[index].index=-1;
};
std::vector<val_index> *values_;
unsigned int index;
double reset_value_;
};

class DiagnosticScalar : public Diagnostic
{
friend class SmileiMPI;
friend class Checkpoint;
public :
DiagnosticScalar( Params &params, SmileiMPI *smpi, Patch *patch );

~DiagnosticScalar() override;

void openFile( Params &params, SmileiMPI *smpi ) override;

void closeFile() override;

void init( Params &params, SmileiMPI *smpi, VectorPatch &vecPatches ) override;

bool prepare( int itime ) override;

void run( Patch *patch, int itime, SimWindow *simWindow ) override;

void write( int itime, SmileiMPI *smpi ) override;

virtual bool needsRhoJs( int itime ) override;

double getScalar( std::string name );

double Energy_time_zero;

double EnergyUsedForNorm;

void compute( Patch *patch, int itime );

int latest_timestep;

int getMemFootPrint() override
{
return 0;
}

uint64_t getDiskFootPrint( int istart, int istop, Patch *patch ) override;

private :

unsigned int calculateWidth( std::string key );

Scalar_value *newScalar_SUM( std::string name );
Scalar_value_location *newScalar_MINLOC( std::string name );
Scalar_value_location *newScalar_MAXLOC( std::string name );

bool allowedKey( std::string );

unsigned int precision;

std::vector<std::string> vars;

std::vector<Scalar *> allScalars;
std::vector<double> values_SUM;
std::vector<val_index> values_MINLOC;
std::vector<val_index> values_MAXLOC;

double cell_volume;

double res_time;

double dt;

std::vector<unsigned int> n_space;

std::vector<unsigned int> n_space_global;

std::ofstream fout;

Scalar_value *Utot, *Uexp, *Ubal, *Ubal_norm;
Scalar_value *Uelm, *Ukin, *Uelm_bnd, *Ukin_bnd;
Scalar_value *Ukin_out_mvw, *Ukin_inj_mvw, *Ukin_new, *Uelm_out_mvw, *Uelm_inj_mvw;
Scalar_value *Urad;
Scalar_value *UmBWpairs;

std::vector<Scalar_value *> sDens, sNtot, sZavg, sUkin, fieldUelm;
std::vector<Scalar_value *> sUrad;
std::vector<Scalar_value_location *> fieldMin, fieldMax;
std::vector<Scalar_value *> poy, poyInst;

bool necessary_Ubal_norm, necessary_Ubal, necessary_Utot, necessary_Uexp;
bool necessary_Ukin, necessary_Ukin_BC;
bool necessary_Uelm, necessary_Uelm_BC;
bool necessary_Urad;
bool necessary_UmBWpairs;
bool necessary_fieldMinMax_any;
std::vector<bool> necessary_species, necessary_fieldUelm, necessary_fieldMinMax, necessary_poy;
};

#endif


#ifndef Laser_H
#define Laser_H

#include "PyTools.h"
#include "Profile.h"
#include "Field.h"
#include "Field1D.h"
#include "Field2D.h"
#include "Field3D.h"

#include <vector>
#include <string>
#include <cmath>

class Params;
class Patch;


class LaserProfile
{
friend class SmileiMPI;
public:
LaserProfile() {};
virtual ~LaserProfile() {};
virtual double getAmplitude( std::vector<double> pos, double t, int j, int k ) = 0;
virtual std::complex<double> getAmplitudecomplex( std::vector<double> pos, double t, int j, int k )
{
return 0.;
};

virtual std::string getInfo()
{
return "?";
};
virtual void createFields( Params &params, Patch *patch ) {};
virtual void initFields( Params &params, Patch *patch ) {};
};


class Laser
{
friend class SmileiMPI;
friend class Patch;
friend class Checkpoint;
public:
Laser( Params &params, int ilaser, Patch *patch, bool verbose = false );
Laser( Laser *, Params & );
~Laser();
void clean();

inline double getAmplitude0( std::vector<double> pos, double t, int j, int k )
{
return profiles[0]->getAmplitude( pos, t, j, k );
}
inline double getAmplitude1( std::vector<double> pos, double t, int j, int k )
{
return profiles[1]->getAmplitude( pos, t, j, k );
}

inline std::complex<double> getAmplitudecomplexN( std::vector<double> pos, double t, int j, int k, int imode )
{
return profiles[imode]->getAmplitudecomplex( pos, t, j, k );
}

void createFields( Params &params, Patch *patch )
{
profiles[0]->createFields( params, patch );
profiles[1]->createFields( params, patch );
};
void initFields( Params &params, Patch *patch )
{
profiles[0]->initFields( params, patch );
profiles[1]->initFields( params, patch );
};

unsigned int i_boundary_;

void disable();

std::vector<bool> spacetime;

protected:
std::vector<LaserProfile *> profiles;

private:

std::string file;

};



class LaserProfileSeparable : public LaserProfile
{
friend class SmileiMPI;
friend class Patch;
public:
LaserProfileSeparable( double, Profile *, Profile *, Profile *, Profile *, double, bool, unsigned int );
LaserProfileSeparable( LaserProfileSeparable * );
~LaserProfileSeparable();
void createFields( Params &params, Patch *patch );
void initFields( Params &params, Patch *patch );
double getAmplitude( std::vector<double> pos, double t, int j, int k );
protected:
Field *space_envelope, *phase;
private:
bool primal_;
double omega_;
Profile *timeProfile_, *chirpProfile_, *spaceProfile_, *phaseProfile_;
double delay_phase_;
unsigned int axis_;
};

class LaserProfileNonSeparable : public LaserProfile
{
friend class SmileiMPI;
public:
LaserProfileNonSeparable( Profile *spaceAndTimeProfile )
: spaceAndTimeProfile_( spaceAndTimeProfile ) {};
LaserProfileNonSeparable( LaserProfileNonSeparable *lp )
: spaceAndTimeProfile_( new Profile( lp->spaceAndTimeProfile_ ) ) {};
~LaserProfileNonSeparable();
inline double getAmplitude( std::vector<double> pos, double t, int j, int k )
{
double amp;
#pragma omp critical
amp = spaceAndTimeProfile_->valueAt( pos, t );
return amp;
}

inline std::complex<double> getAmplitudecomplex( std::vector<double> pos, double t, int j, int k )
{
std::complex<double> amp;
#pragma omp critical
amp = spaceAndTimeProfile_->complexValueAt( pos, t );
return amp;
}

private:
Profile *spaceAndTimeProfile_;
};

class LaserProfileFile : public LaserProfile
{
friend class SmileiMPI;
friend class Checkpoint;
public:
LaserProfileFile( std::string file_, Profile *ep_, bool pr_, unsigned int axis )
: magnitude( NULL ), phase( NULL ), file( file_ ), extraProfile( ep_ ), primal_( pr_ ), axis_( axis ) {};
LaserProfileFile( LaserProfileFile *lp )
: magnitude( NULL ), phase( NULL ), file( lp->file ), extraProfile( new Profile( lp->extraProfile ) ), primal_( lp->primal_ ), axis_( lp->axis_ ) {};
~LaserProfileFile();
void createFields( Params &params, Patch *patch );
void initFields( Params &params, Patch *patch );
double getAmplitude( std::vector<double> pos, double t, int j, int k );
protected:
Field3D *magnitude, *phase;
std::vector<double> omega;
private:
std::string file;
Profile *extraProfile;
bool primal_;
unsigned int axis_;
};

class LaserProfileNULL : public LaserProfile
{
public:
LaserProfileNULL() {};
~LaserProfileNULL() {};

inline double getAmplitude( std::vector<double> pos, double t, int j, int k )
{
return 0.;
}
};


#endif

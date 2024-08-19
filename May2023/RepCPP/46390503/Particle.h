#ifndef _DEM_RECORDS_PARTICLE_H
#define _DEM_RECORDS_PARTICLE_H

#include "peano/utils/Globals.h"
#include "tarch/compiler/CompilerSpecificSettings.h"
#include "peano/utils/PeanoOptimisations.h"
#ifdef Parallel
#include "tarch/parallel/Node.h"
#endif
#ifdef Parallel
#include <mpi.h>
#endif
#include "tarch/logging/Log.h"
#include "tarch/la/Vector.h"
#include <bitset>
#include <complex>
#include <string>
#include <iostream>

namespace dem {
namespace records {
class Particle;
class ParticlePacked;
}
}


class dem::records::Particle { 

public:

typedef dem::records::ParticlePacked Packed;

struct PersistentRecords {
#ifdef UseManualAlignment
tarch::la::Vector<6,double> _vertices __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<6,double> _vertices;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<6,double> _verticesA __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<6,double> _verticesA;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<6,double> _verticesB __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<6,double> _verticesB;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<6,double> _verticesC __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<6,double> _verticesC;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<6,double> _verticesrefA __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<6,double> _verticesrefA;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<6,double> _verticesrefB __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<6,double> _verticesrefB;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<6,double> _verticesrefC __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<6,double> _verticesrefC;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<9,double> _orientation __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<9,double> _orientation;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<9,double> _inertia __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<9,double> _inertia;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<9,double> _inverse __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<9,double> _inverse;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS,double> _centre __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS,double> _centre;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS,double> _centreOfMass __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS,double> _centreOfMass;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS,double> _referentialCentreOfMass __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS,double> _referentialCentreOfMass;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS,double> _velocity __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS,double> _velocity;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS,double> _angular __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS,double> _angular;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS,double> _referentialAngular __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS,double> _referentialAngular;
#endif
double _diameter;
double _haloDiameter;
double _epsilon;
double _mass;
double _hMin;
int _globalParticleId;
int _localParticleId;
int _numberOfTriangles;
int _material;
bool _isObstacle;
bool _friction;

PersistentRecords();


PersistentRecords(const tarch::la::Vector<6,double>& vertices, const tarch::la::Vector<6,double>& verticesA, const tarch::la::Vector<6,double>& verticesB, const tarch::la::Vector<6,double>& verticesC, const tarch::la::Vector<6,double>& verticesrefA, const tarch::la::Vector<6,double>& verticesrefB, const tarch::la::Vector<6,double>& verticesrefC, const tarch::la::Vector<9,double>& orientation, const tarch::la::Vector<9,double>& inertia, const tarch::la::Vector<9,double>& inverse, const tarch::la::Vector<DIMENSIONS,double>& centre, const tarch::la::Vector<DIMENSIONS,double>& centreOfMass, const tarch::la::Vector<DIMENSIONS,double>& referentialCentreOfMass, const tarch::la::Vector<DIMENSIONS,double>& velocity, const tarch::la::Vector<DIMENSIONS,double>& angular, const tarch::la::Vector<DIMENSIONS,double>& referentialAngular, const double& diameter, const double& haloDiameter, const double& epsilon, const double& mass, const double& hMin, const int& globalParticleId, const int& localParticleId, const int& numberOfTriangles, const int& material, const bool& isObstacle, const bool& friction);



inline tarch::la::Vector<6,double> getVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _vertices;
}




inline void setVertices(const tarch::la::Vector<6,double>& vertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_vertices = (vertices);
}




inline tarch::la::Vector<6,double> getVerticesA() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _verticesA;
}




inline void setVerticesA(const tarch::la::Vector<6,double>& verticesA) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_verticesA = (verticesA);
}




inline tarch::la::Vector<6,double> getVerticesB() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _verticesB;
}




inline void setVerticesB(const tarch::la::Vector<6,double>& verticesB) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_verticesB = (verticesB);
}




inline tarch::la::Vector<6,double> getVerticesC() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _verticesC;
}




inline void setVerticesC(const tarch::la::Vector<6,double>& verticesC) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_verticesC = (verticesC);
}




inline tarch::la::Vector<6,double> getVerticesrefA() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _verticesrefA;
}




inline void setVerticesrefA(const tarch::la::Vector<6,double>& verticesrefA) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_verticesrefA = (verticesrefA);
}




inline tarch::la::Vector<6,double> getVerticesrefB() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _verticesrefB;
}




inline void setVerticesrefB(const tarch::la::Vector<6,double>& verticesrefB) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_verticesrefB = (verticesrefB);
}




inline tarch::la::Vector<6,double> getVerticesrefC() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _verticesrefC;
}




inline void setVerticesrefC(const tarch::la::Vector<6,double>& verticesrefC) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_verticesrefC = (verticesrefC);
}




inline tarch::la::Vector<9,double> getOrientation() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _orientation;
}




inline void setOrientation(const tarch::la::Vector<9,double>& orientation) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_orientation = (orientation);
}




inline tarch::la::Vector<9,double> getInertia() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _inertia;
}




inline void setInertia(const tarch::la::Vector<9,double>& inertia) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_inertia = (inertia);
}




inline tarch::la::Vector<9,double> getInverse() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _inverse;
}




inline void setInverse(const tarch::la::Vector<9,double>& inverse) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_inverse = (inverse);
}




inline tarch::la::Vector<DIMENSIONS,double> getCentre() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _centre;
}




inline void setCentre(const tarch::la::Vector<DIMENSIONS,double>& centre) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_centre = (centre);
}




inline tarch::la::Vector<DIMENSIONS,double> getCentreOfMass() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _centreOfMass;
}




inline void setCentreOfMass(const tarch::la::Vector<DIMENSIONS,double>& centreOfMass) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_centreOfMass = (centreOfMass);
}




inline tarch::la::Vector<DIMENSIONS,double> getReferentialCentreOfMass() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _referentialCentreOfMass;
}




inline void setReferentialCentreOfMass(const tarch::la::Vector<DIMENSIONS,double>& referentialCentreOfMass) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_referentialCentreOfMass = (referentialCentreOfMass);
}




inline tarch::la::Vector<DIMENSIONS,double> getVelocity() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _velocity;
}




inline void setVelocity(const tarch::la::Vector<DIMENSIONS,double>& velocity) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_velocity = (velocity);
}




inline tarch::la::Vector<DIMENSIONS,double> getAngular() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _angular;
}




inline void setAngular(const tarch::la::Vector<DIMENSIONS,double>& angular) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_angular = (angular);
}




inline tarch::la::Vector<DIMENSIONS,double> getReferentialAngular() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _referentialAngular;
}




inline void setReferentialAngular(const tarch::la::Vector<DIMENSIONS,double>& referentialAngular) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_referentialAngular = (referentialAngular);
}



inline double getDiameter() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _diameter;
}



inline void setDiameter(const double& diameter) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_diameter = diameter;
}



inline double getHaloDiameter() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _haloDiameter;
}



inline void setHaloDiameter(const double& haloDiameter) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_haloDiameter = haloDiameter;
}



inline double getEpsilon() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _epsilon;
}



inline void setEpsilon(const double& epsilon) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_epsilon = epsilon;
}



inline double getMass() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _mass;
}



inline void setMass(const double& mass) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_mass = mass;
}



inline double getHMin() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _hMin;
}



inline void setHMin(const double& hMin) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_hMin = hMin;
}



inline int getGlobalParticleId() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _globalParticleId;
}



inline void setGlobalParticleId(const int& globalParticleId) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_globalParticleId = globalParticleId;
}



inline int getLocalParticleId() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _localParticleId;
}



inline void setLocalParticleId(const int& localParticleId) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_localParticleId = localParticleId;
}



inline int getNumberOfTriangles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfTriangles;
}



inline void setNumberOfTriangles(const int& numberOfTriangles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfTriangles = numberOfTriangles;
}



inline int getMaterial() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _material;
}



inline void setMaterial(const int& material) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_material = material;
}



inline bool getIsObstacle() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isObstacle;
}



inline void setIsObstacle(const bool& isObstacle) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isObstacle = isObstacle;
}



inline bool getFriction() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _friction;
}



inline void setFriction(const bool& friction) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_friction = friction;
}



};
private: 
public:   

PersistentRecords _persistentRecords;
private:


public:

Particle();


Particle(const PersistentRecords& persistentRecords);


Particle(const tarch::la::Vector<6,double>& vertices, const tarch::la::Vector<6,double>& verticesA, const tarch::la::Vector<6,double>& verticesB, const tarch::la::Vector<6,double>& verticesC, const tarch::la::Vector<6,double>& verticesrefA, const tarch::la::Vector<6,double>& verticesrefB, const tarch::la::Vector<6,double>& verticesrefC, const tarch::la::Vector<9,double>& orientation, const tarch::la::Vector<9,double>& inertia, const tarch::la::Vector<9,double>& inverse, const tarch::la::Vector<DIMENSIONS,double>& centre, const tarch::la::Vector<DIMENSIONS,double>& centreOfMass, const tarch::la::Vector<DIMENSIONS,double>& referentialCentreOfMass, const tarch::la::Vector<DIMENSIONS,double>& velocity, const tarch::la::Vector<DIMENSIONS,double>& angular, const tarch::la::Vector<DIMENSIONS,double>& referentialAngular, const double& diameter, const double& haloDiameter, const double& epsilon, const double& mass, const double& hMin, const int& globalParticleId, const int& localParticleId, const int& numberOfTriangles, const int& material, const bool& isObstacle, const bool& friction);


~Particle();



inline tarch::la::Vector<6,double> getVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._vertices;
}




inline void setVertices(const tarch::la::Vector<6,double>& vertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._vertices = (vertices);
}



inline double getVertices(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
return _persistentRecords._vertices[elementIndex];

}



inline void setVertices(int elementIndex, const double& vertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
_persistentRecords._vertices[elementIndex]= vertices;

}




inline tarch::la::Vector<6,double> getVerticesA() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._verticesA;
}




inline void setVerticesA(const tarch::la::Vector<6,double>& verticesA) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._verticesA = (verticesA);
}



inline double getVerticesA(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
return _persistentRecords._verticesA[elementIndex];

}



inline void setVerticesA(int elementIndex, const double& verticesA) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
_persistentRecords._verticesA[elementIndex]= verticesA;

}




inline tarch::la::Vector<6,double> getVerticesB() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._verticesB;
}




inline void setVerticesB(const tarch::la::Vector<6,double>& verticesB) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._verticesB = (verticesB);
}



inline double getVerticesB(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
return _persistentRecords._verticesB[elementIndex];

}



inline void setVerticesB(int elementIndex, const double& verticesB) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
_persistentRecords._verticesB[elementIndex]= verticesB;

}




inline tarch::la::Vector<6,double> getVerticesC() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._verticesC;
}




inline void setVerticesC(const tarch::la::Vector<6,double>& verticesC) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._verticesC = (verticesC);
}



inline double getVerticesC(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
return _persistentRecords._verticesC[elementIndex];

}



inline void setVerticesC(int elementIndex, const double& verticesC) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
_persistentRecords._verticesC[elementIndex]= verticesC;

}




inline tarch::la::Vector<6,double> getVerticesrefA() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._verticesrefA;
}




inline void setVerticesrefA(const tarch::la::Vector<6,double>& verticesrefA) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._verticesrefA = (verticesrefA);
}



inline double getVerticesrefA(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
return _persistentRecords._verticesrefA[elementIndex];

}



inline void setVerticesrefA(int elementIndex, const double& verticesrefA) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
_persistentRecords._verticesrefA[elementIndex]= verticesrefA;

}




inline tarch::la::Vector<6,double> getVerticesrefB() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._verticesrefB;
}




inline void setVerticesrefB(const tarch::la::Vector<6,double>& verticesrefB) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._verticesrefB = (verticesrefB);
}



inline double getVerticesrefB(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
return _persistentRecords._verticesrefB[elementIndex];

}



inline void setVerticesrefB(int elementIndex, const double& verticesrefB) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
_persistentRecords._verticesrefB[elementIndex]= verticesrefB;

}




inline tarch::la::Vector<6,double> getVerticesrefC() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._verticesrefC;
}




inline void setVerticesrefC(const tarch::la::Vector<6,double>& verticesrefC) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._verticesrefC = (verticesrefC);
}



inline double getVerticesrefC(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
return _persistentRecords._verticesrefC[elementIndex];

}



inline void setVerticesrefC(int elementIndex, const double& verticesrefC) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
_persistentRecords._verticesrefC[elementIndex]= verticesrefC;

}




inline tarch::la::Vector<9,double> getOrientation() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._orientation;
}




inline void setOrientation(const tarch::la::Vector<9,double>& orientation) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._orientation = (orientation);
}



inline double getOrientation(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<9);
return _persistentRecords._orientation[elementIndex];

}



inline void setOrientation(int elementIndex, const double& orientation) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<9);
_persistentRecords._orientation[elementIndex]= orientation;

}




inline tarch::la::Vector<9,double> getInertia() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._inertia;
}




inline void setInertia(const tarch::la::Vector<9,double>& inertia) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._inertia = (inertia);
}



inline double getInertia(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<9);
return _persistentRecords._inertia[elementIndex];

}



inline void setInertia(int elementIndex, const double& inertia) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<9);
_persistentRecords._inertia[elementIndex]= inertia;

}




inline tarch::la::Vector<9,double> getInverse() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._inverse;
}




inline void setInverse(const tarch::la::Vector<9,double>& inverse) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._inverse = (inverse);
}



inline double getInverse(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<9);
return _persistentRecords._inverse[elementIndex];

}



inline void setInverse(int elementIndex, const double& inverse) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<9);
_persistentRecords._inverse[elementIndex]= inverse;

}




inline tarch::la::Vector<DIMENSIONS,double> getCentre() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._centre;
}




inline void setCentre(const tarch::la::Vector<DIMENSIONS,double>& centre) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._centre = (centre);
}



inline double getCentre(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._centre[elementIndex];

}



inline void setCentre(int elementIndex, const double& centre) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._centre[elementIndex]= centre;

}




inline tarch::la::Vector<DIMENSIONS,double> getCentreOfMass() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._centreOfMass;
}




inline void setCentreOfMass(const tarch::la::Vector<DIMENSIONS,double>& centreOfMass) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._centreOfMass = (centreOfMass);
}



inline double getCentreOfMass(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._centreOfMass[elementIndex];

}



inline void setCentreOfMass(int elementIndex, const double& centreOfMass) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._centreOfMass[elementIndex]= centreOfMass;

}




inline tarch::la::Vector<DIMENSIONS,double> getReferentialCentreOfMass() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._referentialCentreOfMass;
}




inline void setReferentialCentreOfMass(const tarch::la::Vector<DIMENSIONS,double>& referentialCentreOfMass) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._referentialCentreOfMass = (referentialCentreOfMass);
}



inline double getReferentialCentreOfMass(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._referentialCentreOfMass[elementIndex];

}



inline void setReferentialCentreOfMass(int elementIndex, const double& referentialCentreOfMass) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._referentialCentreOfMass[elementIndex]= referentialCentreOfMass;

}




inline tarch::la::Vector<DIMENSIONS,double> getVelocity() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._velocity;
}




inline void setVelocity(const tarch::la::Vector<DIMENSIONS,double>& velocity) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._velocity = (velocity);
}



inline double getVelocity(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._velocity[elementIndex];

}



inline void setVelocity(int elementIndex, const double& velocity) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._velocity[elementIndex]= velocity;

}




inline tarch::la::Vector<DIMENSIONS,double> getAngular() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._angular;
}




inline void setAngular(const tarch::la::Vector<DIMENSIONS,double>& angular) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._angular = (angular);
}



inline double getAngular(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._angular[elementIndex];

}



inline void setAngular(int elementIndex, const double& angular) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._angular[elementIndex]= angular;

}




inline tarch::la::Vector<DIMENSIONS,double> getReferentialAngular() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._referentialAngular;
}




inline void setReferentialAngular(const tarch::la::Vector<DIMENSIONS,double>& referentialAngular) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._referentialAngular = (referentialAngular);
}



inline double getReferentialAngular(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._referentialAngular[elementIndex];

}



inline void setReferentialAngular(int elementIndex, const double& referentialAngular) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._referentialAngular[elementIndex]= referentialAngular;

}



inline double getDiameter() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._diameter;
}



inline void setDiameter(const double& diameter) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._diameter = diameter;
}



inline double getHaloDiameter() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._haloDiameter;
}



inline void setHaloDiameter(const double& haloDiameter) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._haloDiameter = haloDiameter;
}



inline double getEpsilon() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._epsilon;
}



inline void setEpsilon(const double& epsilon) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._epsilon = epsilon;
}



inline double getMass() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._mass;
}



inline void setMass(const double& mass) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._mass = mass;
}



inline double getHMin() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._hMin;
}



inline void setHMin(const double& hMin) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._hMin = hMin;
}



inline int getGlobalParticleId() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._globalParticleId;
}



inline void setGlobalParticleId(const int& globalParticleId) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._globalParticleId = globalParticleId;
}



inline int getLocalParticleId() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._localParticleId;
}



inline void setLocalParticleId(const int& localParticleId) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._localParticleId = localParticleId;
}



inline int getNumberOfTriangles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfTriangles;
}



inline void setNumberOfTriangles(const int& numberOfTriangles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfTriangles = numberOfTriangles;
}



inline int getMaterial() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._material;
}



inline void setMaterial(const int& material) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._material = material;
}



inline bool getIsObstacle() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isObstacle;
}



inline void setIsObstacle(const bool& isObstacle) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isObstacle = isObstacle;
}



inline bool getFriction() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._friction;
}



inline void setFriction(const bool& friction) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._friction = friction;
}



std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

ParticlePacked convert() const;


#ifdef Parallel
protected:
static tarch::logging::Log _log;

public:


static MPI_Datatype Datatype;
static MPI_Datatype FullDatatype;


static void initDatatype();

static void shutdownDatatype();

enum class ExchangeMode { Blocking, NonblockingWithPollingLoopOverTests, LoopOverProbeWithBlockingReceive };

void send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, ExchangeMode mode );

void receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, ExchangeMode mode );

static bool isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise);

#endif

};

#ifndef DaStGenPackedPadding
#define DaStGenPackedPadding 1      
#endif


#ifdef PackedRecords
#pragma pack (push, DaStGenPackedPadding)
#endif


class dem::records::ParticlePacked { 

public:

struct PersistentRecords {
tarch::la::Vector<6,double> _vertices;
tarch::la::Vector<6,double> _verticesA;
tarch::la::Vector<6,double> _verticesB;
tarch::la::Vector<6,double> _verticesC;
tarch::la::Vector<6,double> _verticesrefA;
tarch::la::Vector<6,double> _verticesrefB;
tarch::la::Vector<6,double> _verticesrefC;
tarch::la::Vector<9,double> _orientation;
tarch::la::Vector<9,double> _inertia;
tarch::la::Vector<9,double> _inverse;
tarch::la::Vector<DIMENSIONS,double> _centre;
tarch::la::Vector<DIMENSIONS,double> _centreOfMass;
tarch::la::Vector<DIMENSIONS,double> _referentialCentreOfMass;
tarch::la::Vector<DIMENSIONS,double> _velocity;
tarch::la::Vector<DIMENSIONS,double> _angular;
tarch::la::Vector<DIMENSIONS,double> _referentialAngular;
double _diameter;
double _haloDiameter;
double _epsilon;
double _mass;
double _hMin;
int _globalParticleId;
int _localParticleId;
int _numberOfTriangles;
int _material;
bool _isObstacle;
bool _friction;

PersistentRecords();


PersistentRecords(const tarch::la::Vector<6,double>& vertices, const tarch::la::Vector<6,double>& verticesA, const tarch::la::Vector<6,double>& verticesB, const tarch::la::Vector<6,double>& verticesC, const tarch::la::Vector<6,double>& verticesrefA, const tarch::la::Vector<6,double>& verticesrefB, const tarch::la::Vector<6,double>& verticesrefC, const tarch::la::Vector<9,double>& orientation, const tarch::la::Vector<9,double>& inertia, const tarch::la::Vector<9,double>& inverse, const tarch::la::Vector<DIMENSIONS,double>& centre, const tarch::la::Vector<DIMENSIONS,double>& centreOfMass, const tarch::la::Vector<DIMENSIONS,double>& referentialCentreOfMass, const tarch::la::Vector<DIMENSIONS,double>& velocity, const tarch::la::Vector<DIMENSIONS,double>& angular, const tarch::la::Vector<DIMENSIONS,double>& referentialAngular, const double& diameter, const double& haloDiameter, const double& epsilon, const double& mass, const double& hMin, const int& globalParticleId, const int& localParticleId, const int& numberOfTriangles, const int& material, const bool& isObstacle, const bool& friction);



inline tarch::la::Vector<6,double> getVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _vertices;
}




inline void setVertices(const tarch::la::Vector<6,double>& vertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_vertices = (vertices);
}




inline tarch::la::Vector<6,double> getVerticesA() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _verticesA;
}




inline void setVerticesA(const tarch::la::Vector<6,double>& verticesA) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_verticesA = (verticesA);
}




inline tarch::la::Vector<6,double> getVerticesB() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _verticesB;
}




inline void setVerticesB(const tarch::la::Vector<6,double>& verticesB) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_verticesB = (verticesB);
}




inline tarch::la::Vector<6,double> getVerticesC() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _verticesC;
}




inline void setVerticesC(const tarch::la::Vector<6,double>& verticesC) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_verticesC = (verticesC);
}




inline tarch::la::Vector<6,double> getVerticesrefA() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _verticesrefA;
}




inline void setVerticesrefA(const tarch::la::Vector<6,double>& verticesrefA) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_verticesrefA = (verticesrefA);
}




inline tarch::la::Vector<6,double> getVerticesrefB() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _verticesrefB;
}




inline void setVerticesrefB(const tarch::la::Vector<6,double>& verticesrefB) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_verticesrefB = (verticesrefB);
}




inline tarch::la::Vector<6,double> getVerticesrefC() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _verticesrefC;
}




inline void setVerticesrefC(const tarch::la::Vector<6,double>& verticesrefC) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_verticesrefC = (verticesrefC);
}




inline tarch::la::Vector<9,double> getOrientation() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _orientation;
}




inline void setOrientation(const tarch::la::Vector<9,double>& orientation) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_orientation = (orientation);
}




inline tarch::la::Vector<9,double> getInertia() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _inertia;
}




inline void setInertia(const tarch::la::Vector<9,double>& inertia) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_inertia = (inertia);
}




inline tarch::la::Vector<9,double> getInverse() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _inverse;
}




inline void setInverse(const tarch::la::Vector<9,double>& inverse) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_inverse = (inverse);
}




inline tarch::la::Vector<DIMENSIONS,double> getCentre() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _centre;
}




inline void setCentre(const tarch::la::Vector<DIMENSIONS,double>& centre) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_centre = (centre);
}




inline tarch::la::Vector<DIMENSIONS,double> getCentreOfMass() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _centreOfMass;
}




inline void setCentreOfMass(const tarch::la::Vector<DIMENSIONS,double>& centreOfMass) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_centreOfMass = (centreOfMass);
}




inline tarch::la::Vector<DIMENSIONS,double> getReferentialCentreOfMass() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _referentialCentreOfMass;
}




inline void setReferentialCentreOfMass(const tarch::la::Vector<DIMENSIONS,double>& referentialCentreOfMass) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_referentialCentreOfMass = (referentialCentreOfMass);
}




inline tarch::la::Vector<DIMENSIONS,double> getVelocity() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _velocity;
}




inline void setVelocity(const tarch::la::Vector<DIMENSIONS,double>& velocity) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_velocity = (velocity);
}




inline tarch::la::Vector<DIMENSIONS,double> getAngular() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _angular;
}




inline void setAngular(const tarch::la::Vector<DIMENSIONS,double>& angular) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_angular = (angular);
}




inline tarch::la::Vector<DIMENSIONS,double> getReferentialAngular() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _referentialAngular;
}




inline void setReferentialAngular(const tarch::la::Vector<DIMENSIONS,double>& referentialAngular) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_referentialAngular = (referentialAngular);
}



inline double getDiameter() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _diameter;
}



inline void setDiameter(const double& diameter) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_diameter = diameter;
}



inline double getHaloDiameter() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _haloDiameter;
}



inline void setHaloDiameter(const double& haloDiameter) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_haloDiameter = haloDiameter;
}



inline double getEpsilon() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _epsilon;
}



inline void setEpsilon(const double& epsilon) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_epsilon = epsilon;
}



inline double getMass() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _mass;
}



inline void setMass(const double& mass) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_mass = mass;
}



inline double getHMin() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _hMin;
}



inline void setHMin(const double& hMin) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_hMin = hMin;
}



inline int getGlobalParticleId() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _globalParticleId;
}



inline void setGlobalParticleId(const int& globalParticleId) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_globalParticleId = globalParticleId;
}



inline int getLocalParticleId() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _localParticleId;
}



inline void setLocalParticleId(const int& localParticleId) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_localParticleId = localParticleId;
}



inline int getNumberOfTriangles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfTriangles;
}



inline void setNumberOfTriangles(const int& numberOfTriangles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfTriangles = numberOfTriangles;
}



inline int getMaterial() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _material;
}



inline void setMaterial(const int& material) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_material = material;
}



inline bool getIsObstacle() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isObstacle;
}



inline void setIsObstacle(const bool& isObstacle) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isObstacle = isObstacle;
}



inline bool getFriction() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _friction;
}



inline void setFriction(const bool& friction) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_friction = friction;
}



};
private: 
PersistentRecords _persistentRecords;

public:

ParticlePacked();


ParticlePacked(const PersistentRecords& persistentRecords);


ParticlePacked(const tarch::la::Vector<6,double>& vertices, const tarch::la::Vector<6,double>& verticesA, const tarch::la::Vector<6,double>& verticesB, const tarch::la::Vector<6,double>& verticesC, const tarch::la::Vector<6,double>& verticesrefA, const tarch::la::Vector<6,double>& verticesrefB, const tarch::la::Vector<6,double>& verticesrefC, const tarch::la::Vector<9,double>& orientation, const tarch::la::Vector<9,double>& inertia, const tarch::la::Vector<9,double>& inverse, const tarch::la::Vector<DIMENSIONS,double>& centre, const tarch::la::Vector<DIMENSIONS,double>& centreOfMass, const tarch::la::Vector<DIMENSIONS,double>& referentialCentreOfMass, const tarch::la::Vector<DIMENSIONS,double>& velocity, const tarch::la::Vector<DIMENSIONS,double>& angular, const tarch::la::Vector<DIMENSIONS,double>& referentialAngular, const double& diameter, const double& haloDiameter, const double& epsilon, const double& mass, const double& hMin, const int& globalParticleId, const int& localParticleId, const int& numberOfTriangles, const int& material, const bool& isObstacle, const bool& friction);


~ParticlePacked();



inline tarch::la::Vector<6,double> getVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._vertices;
}




inline void setVertices(const tarch::la::Vector<6,double>& vertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._vertices = (vertices);
}



inline double getVertices(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
return _persistentRecords._vertices[elementIndex];

}



inline void setVertices(int elementIndex, const double& vertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
_persistentRecords._vertices[elementIndex]= vertices;

}




inline tarch::la::Vector<6,double> getVerticesA() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._verticesA;
}




inline void setVerticesA(const tarch::la::Vector<6,double>& verticesA) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._verticesA = (verticesA);
}



inline double getVerticesA(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
return _persistentRecords._verticesA[elementIndex];

}



inline void setVerticesA(int elementIndex, const double& verticesA) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
_persistentRecords._verticesA[elementIndex]= verticesA;

}




inline tarch::la::Vector<6,double> getVerticesB() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._verticesB;
}




inline void setVerticesB(const tarch::la::Vector<6,double>& verticesB) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._verticesB = (verticesB);
}



inline double getVerticesB(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
return _persistentRecords._verticesB[elementIndex];

}



inline void setVerticesB(int elementIndex, const double& verticesB) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
_persistentRecords._verticesB[elementIndex]= verticesB;

}




inline tarch::la::Vector<6,double> getVerticesC() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._verticesC;
}




inline void setVerticesC(const tarch::la::Vector<6,double>& verticesC) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._verticesC = (verticesC);
}



inline double getVerticesC(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
return _persistentRecords._verticesC[elementIndex];

}



inline void setVerticesC(int elementIndex, const double& verticesC) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
_persistentRecords._verticesC[elementIndex]= verticesC;

}




inline tarch::la::Vector<6,double> getVerticesrefA() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._verticesrefA;
}




inline void setVerticesrefA(const tarch::la::Vector<6,double>& verticesrefA) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._verticesrefA = (verticesrefA);
}



inline double getVerticesrefA(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
return _persistentRecords._verticesrefA[elementIndex];

}



inline void setVerticesrefA(int elementIndex, const double& verticesrefA) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
_persistentRecords._verticesrefA[elementIndex]= verticesrefA;

}




inline tarch::la::Vector<6,double> getVerticesrefB() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._verticesrefB;
}




inline void setVerticesrefB(const tarch::la::Vector<6,double>& verticesrefB) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._verticesrefB = (verticesrefB);
}



inline double getVerticesrefB(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
return _persistentRecords._verticesrefB[elementIndex];

}



inline void setVerticesrefB(int elementIndex, const double& verticesrefB) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
_persistentRecords._verticesrefB[elementIndex]= verticesrefB;

}




inline tarch::la::Vector<6,double> getVerticesrefC() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._verticesrefC;
}




inline void setVerticesrefC(const tarch::la::Vector<6,double>& verticesrefC) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._verticesrefC = (verticesrefC);
}



inline double getVerticesrefC(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
return _persistentRecords._verticesrefC[elementIndex];

}



inline void setVerticesrefC(int elementIndex, const double& verticesrefC) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<6);
_persistentRecords._verticesrefC[elementIndex]= verticesrefC;

}




inline tarch::la::Vector<9,double> getOrientation() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._orientation;
}




inline void setOrientation(const tarch::la::Vector<9,double>& orientation) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._orientation = (orientation);
}



inline double getOrientation(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<9);
return _persistentRecords._orientation[elementIndex];

}



inline void setOrientation(int elementIndex, const double& orientation) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<9);
_persistentRecords._orientation[elementIndex]= orientation;

}




inline tarch::la::Vector<9,double> getInertia() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._inertia;
}




inline void setInertia(const tarch::la::Vector<9,double>& inertia) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._inertia = (inertia);
}



inline double getInertia(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<9);
return _persistentRecords._inertia[elementIndex];

}



inline void setInertia(int elementIndex, const double& inertia) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<9);
_persistentRecords._inertia[elementIndex]= inertia;

}




inline tarch::la::Vector<9,double> getInverse() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._inverse;
}




inline void setInverse(const tarch::la::Vector<9,double>& inverse) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._inverse = (inverse);
}



inline double getInverse(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<9);
return _persistentRecords._inverse[elementIndex];

}



inline void setInverse(int elementIndex, const double& inverse) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<9);
_persistentRecords._inverse[elementIndex]= inverse;

}




inline tarch::la::Vector<DIMENSIONS,double> getCentre() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._centre;
}




inline void setCentre(const tarch::la::Vector<DIMENSIONS,double>& centre) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._centre = (centre);
}



inline double getCentre(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._centre[elementIndex];

}



inline void setCentre(int elementIndex, const double& centre) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._centre[elementIndex]= centre;

}




inline tarch::la::Vector<DIMENSIONS,double> getCentreOfMass() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._centreOfMass;
}




inline void setCentreOfMass(const tarch::la::Vector<DIMENSIONS,double>& centreOfMass) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._centreOfMass = (centreOfMass);
}



inline double getCentreOfMass(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._centreOfMass[elementIndex];

}



inline void setCentreOfMass(int elementIndex, const double& centreOfMass) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._centreOfMass[elementIndex]= centreOfMass;

}




inline tarch::la::Vector<DIMENSIONS,double> getReferentialCentreOfMass() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._referentialCentreOfMass;
}




inline void setReferentialCentreOfMass(const tarch::la::Vector<DIMENSIONS,double>& referentialCentreOfMass) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._referentialCentreOfMass = (referentialCentreOfMass);
}



inline double getReferentialCentreOfMass(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._referentialCentreOfMass[elementIndex];

}



inline void setReferentialCentreOfMass(int elementIndex, const double& referentialCentreOfMass) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._referentialCentreOfMass[elementIndex]= referentialCentreOfMass;

}




inline tarch::la::Vector<DIMENSIONS,double> getVelocity() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._velocity;
}




inline void setVelocity(const tarch::la::Vector<DIMENSIONS,double>& velocity) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._velocity = (velocity);
}



inline double getVelocity(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._velocity[elementIndex];

}



inline void setVelocity(int elementIndex, const double& velocity) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._velocity[elementIndex]= velocity;

}




inline tarch::la::Vector<DIMENSIONS,double> getAngular() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._angular;
}




inline void setAngular(const tarch::la::Vector<DIMENSIONS,double>& angular) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._angular = (angular);
}



inline double getAngular(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._angular[elementIndex];

}



inline void setAngular(int elementIndex, const double& angular) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._angular[elementIndex]= angular;

}




inline tarch::la::Vector<DIMENSIONS,double> getReferentialAngular() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._referentialAngular;
}




inline void setReferentialAngular(const tarch::la::Vector<DIMENSIONS,double>& referentialAngular) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._referentialAngular = (referentialAngular);
}



inline double getReferentialAngular(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._referentialAngular[elementIndex];

}



inline void setReferentialAngular(int elementIndex, const double& referentialAngular) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._referentialAngular[elementIndex]= referentialAngular;

}



inline double getDiameter() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._diameter;
}



inline void setDiameter(const double& diameter) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._diameter = diameter;
}



inline double getHaloDiameter() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._haloDiameter;
}



inline void setHaloDiameter(const double& haloDiameter) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._haloDiameter = haloDiameter;
}



inline double getEpsilon() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._epsilon;
}



inline void setEpsilon(const double& epsilon) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._epsilon = epsilon;
}



inline double getMass() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._mass;
}



inline void setMass(const double& mass) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._mass = mass;
}



inline double getHMin() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._hMin;
}



inline void setHMin(const double& hMin) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._hMin = hMin;
}



inline int getGlobalParticleId() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._globalParticleId;
}



inline void setGlobalParticleId(const int& globalParticleId) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._globalParticleId = globalParticleId;
}



inline int getLocalParticleId() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._localParticleId;
}



inline void setLocalParticleId(const int& localParticleId) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._localParticleId = localParticleId;
}



inline int getNumberOfTriangles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfTriangles;
}



inline void setNumberOfTriangles(const int& numberOfTriangles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfTriangles = numberOfTriangles;
}



inline int getMaterial() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._material;
}



inline void setMaterial(const int& material) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._material = material;
}



inline bool getIsObstacle() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isObstacle;
}



inline void setIsObstacle(const bool& isObstacle) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isObstacle = isObstacle;
}



inline bool getFriction() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._friction;
}



inline void setFriction(const bool& friction) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._friction = friction;
}



std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

Particle convert() const;


#ifdef Parallel
protected:
static tarch::logging::Log _log;

public:


static MPI_Datatype Datatype;
static MPI_Datatype FullDatatype;


static void initDatatype();

static void shutdownDatatype();

enum class ExchangeMode { Blocking, NonblockingWithPollingLoopOverTests, LoopOverProbeWithBlockingReceive };

void send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, ExchangeMode mode );

void receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, ExchangeMode mode );

static bool isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise);

#endif

};

#ifdef PackedRecords
#pragma pack (pop)
#endif


#endif


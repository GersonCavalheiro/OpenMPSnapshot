


#include "Engine.h"

delta::core::Engine::Engine()
{

}

delta::core::Engine::Engine(
delta::world::World	world,
delta::core::data::Meta::Simulation	meta)
{
_overlapCheck		= meta.overlapPreCheck;
_collisionModel	= meta.modelScheme;
_plot				= meta.plotScheme;
_gravity			= world.hasGravity();
_boundary			= world.getBoundary();
_data				= delta::core::data::Structure(world.getObjects());
_state			= delta::core::State(_data, meta);
}

delta::core::Engine::Engine(
std::vector<delta::world::structure::Object>    	particles,
std::array<iREAL, 6>								boundary,
delta::core::data::Meta::Simulation					meta)
{
_overlapCheck     = meta.overlapPreCheck;
_collisionModel   = meta.modelScheme;
_plot             = meta.plotScheme;
_gravity          = meta.gravity;
_state            = delta::core::State(_data, meta);
_boundary         = boundary;

if(meta.maxPrescribedRefinement > 0.0) {
_data = delta::core::data::Structure(
particles,
meta.maxPrescribedRefinement,
true
);
} else {
_data = delta::core::data::Structure(particles);
}
}

delta::core::Engine::~Engine()
{

}


void delta::core::Engine::hyperContacts(
double epsilonA,
double epsilonB,
std::vector<iREAL>& ex,
std::vector<iREAL>& ey,
std::vector<iREAL>& ez,
std::vector<std::array<iREAL,4>>& d)
{
std::array<iREAL, 6> bboxA = delta::geometry::operators::vertex::computeBbox(ex, ey, ez);

for(int ii=0; ii<getParticleRecords().size() &&
getState().getCurrentStepIteration() > 0; ii++)
{
std::array<iREAL, 6> bboxB = getParticleRecords()[ii].getBbox();
if (delta::contact::detection::box(bboxA, epsilonA, bboxB, epsilonB))
{
for(int h=0; h<ex.size(); h++)
{
iREAL xx[3] = {ex[h], ey[h], ez[h]};

std::vector<iREAL> xCoordinatesPartial, yCoordinatesPartial, zCoordinatesPartial;
getParticleRecords()[ii].getSubsetOfMesh(  xx, epsilonA,
xCoordinatesPartial,
yCoordinatesPartial,
zCoordinatesPartial);

auto newContactPoints = delta::contact::detection::pointToGeometry(
xx[0], xx[1], xx[2], 0, epsilonA,
xCoordinatesPartial.data(),
yCoordinatesPartial.data(),
zCoordinatesPartial.data(),
zCoordinatesPartial.size()/3,
1, epsilonB);

iREAL distance = 1E99;
int index = 0;

for(int i=0; i<newContactPoints.size(); i++)
{
iREAL contactDistance = newContactPoints[i].getDistance();
if(distance > contactDistance)
{
distance = contactDistance;
index = i;
}
}

if(newContactPoints.size() > 0)
{
d[h][0] = newContactPoints[index].Q[1];
d[h][1] = newContactPoints[index].Q[2];
d[h][2] = newContactPoints[index].Q[3];
d[h][3] = distance;

if(newContactPoints[index].penetration < 0.0)
{ 	
d[h][3] = 0.0;
}
}
else
{
d[h][0] = 0.0;
d[h][1] = 0.0;
d[h][2] = 0.0;
d[h][3] = epsilonA; 
}

}
}
}
}

void delta::core::Engine::iterate()
{
delta::core::Engine::plot("../output/");
delta::core::Engine::updatePosition();

_state.update();
}

void delta::core::Engine::addCollision(
std::vector<delta::contact::contactpoint>& 		newContactPoints,
delta::core::data::ParticleRecord&				particleA,
delta::core::data::ParticleRecord&          	particleB,
bool sphere
) {

if( _collisionsOfNextTraversal.count(particleA.getGlobalParticleID())==0 ) {
_collisionsOfNextTraversal.insert(
std::pair<int,std::vector<Collisions>>(particleA.getGlobalParticleID(), std::vector<Collisions>()));
}

if( _collisionsOfNextTraversal.count(particleB.getGlobalParticleID())==0 ) {
_collisionsOfNextTraversal.insert(
std::pair<int,std::vector<Collisions>>(particleB.getGlobalParticleID(), std::vector<Collisions>()));
}

Collisions* dataSetA = nullptr;
Collisions* dataSetB = nullptr;

for (std::vector<delta::core::data::Meta::Collisions>::iterator p=_collisionsOfNextTraversal[particleA.getGlobalParticleID()].begin();
p!=_collisionsOfNextTraversal[particleA.getGlobalParticleID()].end(); p++)
{
if(p->_copyOfPartnerParticle.getGlobalParticleID()==particleB.getGlobalParticleID())
{
dataSetA = &(*p);
}
}

for(std::vector<delta::core::data::Meta::Collisions>::iterator p=_collisionsOfNextTraversal[particleB.getGlobalParticleID()].begin();
p!=_collisionsOfNextTraversal[particleB.getGlobalParticleID()].end(); p++)
{
if(p->_copyOfPartnerParticle.getGlobalParticleID()==particleA.getGlobalParticleID())
{
dataSetB = &(*p);
}
}

if((dataSetA!=nullptr && dataSetB!=nullptr)) {
newContactPoints.clear();
return;
}

if(dataSetA==nullptr)
{

}



dataSetA->_contactPoints.insert(dataSetA->_contactPoints.end(), newContactPoints.begin(), newContactPoints.end());

for (std::vector<delta::contact::contactpoint>::iterator p = newContactPoints.begin(); p != newContactPoints.end(); p++)
{
p->normal[0] = -1.0 * p->normal[0];
p->normal[1] = -1.0 * p->normal[1];
p->normal[2] = -1.0 * p->normal[2];
}

dataSetB->_contactPoints.insert(dataSetB->_contactPoints.end(), newContactPoints.begin(), newContactPoints.end());
}

delta::core::State delta::core::Engine::getState()
{
return _state;
}

int delta::core::Engine::getNumberOfCollisions()
{
return _activeCollisions.size();
}

void delta::core::Engine::contactDetection()
{
for(unsigned j=0; j<_data.getNumberOfParticles(); j++)
{
delta::core::data::ParticleRecord& particleA = _data.getParticle(j);


for(unsigned k=j+1; k<_data.getNumberOfParticles(); k++)
{
delta::core::data::ParticleRecord& particleB = _data.getParticle(k);
if(_overlapCheck)
{
bool overlap = delta::contact::detection::isSphereOverlayInContact(
particleA._centre[0],
particleA._centre[1],
particleA._centre[2],
particleA.getHaloDiameter(),

particleB._centre[0],
particleB._centre[1],
particleB._centre[2],
particleB.getHaloDiameter());

if (!overlap) continue;
}

std::vector<delta::contact::contactpoint> newContactPoints;
#if defined(SharedTBB) && defined(peanoCall)

#else
switch (_collisionModel)
{
case CollisionModel::Sphere:
if(	particleA.getIsObstacle() &&
!particleB.getIsObstacle())
{
newContactPoints = delta::contact::detection::sphereWithBarrierBA(
particleB._centre[0],
particleB._centre[1],
particleB._centre[2],
particleB.getDiameter(),
particleB.getEpsilon(),
particleB.getIsFriction(),
particleB.getGlobalParticleID(),

particleA._xCoordinates.data(),
particleA._yCoordinates.data(),
particleA._zCoordinates.data(),
particleA.getNumberOfTriangles(),
particleA.getEpsilon(),
particleA.getIsFriction(),
particleA.getGlobalParticleID());
} else if (
!particleA.getIsObstacle() &&
particleB.getIsObstacle()
) {
newContactPoints = delta::contact::detection::sphereWithBarrierAB(
particleA._centre[0],
particleA._centre[1],
particleA._centre[2],
particleA.getDiameter(),
particleA.getEpsilon(),
particleA.getIsFriction(),
particleA.getGlobalParticleID(),

particleB._xCoordinates.data(),
particleB._yCoordinates.data(),
particleB._zCoordinates.data(),
particleB.getNumberOfTriangles(),
particleB.getEpsilon(),
particleB.getIsFriction(),
particleB.getGlobalParticleID());
}
else {
newContactPoints = delta::contact::detection::sphere(
particleA._centre[0],
particleA._centre[1],
particleA._centre[2],
particleA.getDiameter(),
particleA.getEpsilon(),
particleA.getIsFriction(),
particleA.getGlobalParticleID(),

particleB._centre[0],
particleB._centre[1],
particleB._centre[2],
particleB.getDiameter(),
particleB.getEpsilon(),
particleB.getIsFriction(),
particleB.getGlobalParticleID());
}
break;
case CollisionModel::BruteForce:

newContactPoints = delta::contact::detection::bf(
particleA._xCoordinates.data(),
particleA._yCoordinates.data(),
particleA._zCoordinates.data(),
particleA.getNumberOfTriangles(),
particleA.getEpsilon(),
particleA.getIsFriction(),
particleA.getGlobalParticleID(),

particleB._xCoordinates.data(),
particleB._yCoordinates.data(),
particleB._zCoordinates.data(),
particleB.getNumberOfTriangles(),
particleB.getEpsilon(),
particleB.getIsFriction(),
particleB.getGlobalParticleID());

break;
case CollisionModel::Penalty:

newContactPoints = delta::contact::detection::penalty(
particleA._xCoordinates.data(),
particleA._yCoordinates.data(),
particleA._zCoordinates.data(),
particleA.getNumberOfTriangles(),
particleA.getEpsilon(),
particleA.getIsFriction(),
particleA.getGlobalParticleID(),

particleB._xCoordinates.data(),
particleB._yCoordinates.data(),
particleB._zCoordinates.data(),
particleB.getNumberOfTriangles(),
particleB.getEpsilon(),
particleB.getIsFriction(),
particleB.getGlobalParticleID());

break;
case CollisionModel::HybridOnBatches:

newContactPoints = delta::contact::detection::hybridWithPerBatchFallBack(
particleA._xCoordinates.data(),
particleA._yCoordinates.data(),
particleA._zCoordinates.data(),
particleA.getNumberOfTriangles(),
particleA.getEpsilon(),
particleA.getIsFriction(),
particleA.getGlobalParticleID(),

particleB._xCoordinates.data(),
particleB._yCoordinates.data(),
particleB._zCoordinates.data(),
particleB.getNumberOfTriangles(),
particleB.getEpsilon(),
particleB.getIsFriction(),
particleB.getGlobalParticleID());

break;
case CollisionModel::HybridOnTrianglePairs:

newContactPoints = delta::contact::detection::hybridWithPerTriangleFallBack(
particleA._xCoordinates.data(),
particleA._yCoordinates.data(),
particleA._zCoordinates.data(),
particleA.getNumberOfTriangles(),
particleA.getEpsilon(),
particleA.getIsFriction(),
particleA.getGlobalParticleID(),

particleB._xCoordinates.data(),
particleB._yCoordinates.data(),
particleB._zCoordinates.data(),
particleB.getNumberOfTriangles(),
particleB.getEpsilon(),
particleB.getIsFriction(),
particleB.getGlobalParticleID());

break;
case CollisionModel::none:
break;
}
#endif
if(!newContactPoints.empty()) {
delta::core::Engine::addCollision(newContactPoints, particleA, particleB, _collisionModel == CollisionModel::Sphere);
}

_state.incNumberOfTriangleComparisons(particleA.getNumberOfTriangles() * particleB.getNumberOfTriangles());
_state.incNumberOfParticleComparisons(1);
}
}
}

void delta::core::Engine::deriveForces()
{
for(int i=0; i<_data.getNumberOfParticles(); i++)
{
delta::core::data::ParticleRecord& particle = _data.getParticle(i);

if(_activeCollisions.count(particle.getGlobalParticleID())==0) {continue;}

iREAL force[3]  = {0.0,0.0,0.0};
iREAL torque[3] = {0.0,0.0,0.0};

for(std::vector<Collisions>::iterator p = _activeCollisions[particle.getGlobalParticleID()].begin();
p != _activeCollisions[particle.getGlobalParticleID()].end();
p++)
{
iREAL rforce[3]  = {0.0,0.0,0.0};
iREAL rtorque[3] = {0.0,0.0,0.0};

delta::contact::forces::getContactsForces(
p->_contactPoints,
particle._centreOfMass.data(),
particle._refCentreOfMass.data(),
particle._angularVelocity.data(),
particle._refAngularVelocity.data(),
particle._linearVelocity.data(),
particle.getMass(),
particle.getInverse().data(),
particle._orientation.data(),
int(particle.getMaterial()),
p->_copyOfPartnerParticle._centreOfMass.data(),
p->_copyOfPartnerParticle._refCentreOfMass.data(),
p->_copyOfPartnerParticle._angularVelocity.data(),
p->_copyOfPartnerParticle._refAngularVelocity.data(),
p->_copyOfPartnerParticle._linearVelocity.data(),
p->_copyOfPartnerParticle.getMass(),
p->_copyOfPartnerParticle.getInverse().data(),
p->_copyOfPartnerParticle._orientation.data(),
int(p->_copyOfPartnerParticle.getMaterial()),
rforce, rtorque,
(_collisionModel == CollisionModel::Sphere));

force[0] += rforce[0];
force[1] += rforce[1];
force[2] += rforce[2];

torque[0] += rtorque[0];
torque[1] += rtorque[1];
torque[2] += rtorque[2];
}

if(!particle.getIsObstacle())
{
particle._linearVelocity[0] += _state.getStepSize() * (force[0] / particle.getMass());
particle._linearVelocity[1] += _state.getStepSize() * (force[1] / particle.getMass());
particle._linearVelocity[2] += _state.getStepSize() * (force[2] / particle.getMass());

delta::dynamics::updateAngular(
particle._refAngularVelocity.data(),
particle._orientation.data(),
particle.getInertia().data(),
particle.getInverse().data(),
torque, _state.getStepSize()
);
}
}
}

void delta::core::Engine::updatePosition()
{
for(unsigned i=0; i<_data.getNumberOfParticles(); i++)
{
delta::core::data::ParticleRecord& particle = _data.getParticle(i);

if(particle.getIsObstacle()) continue;

particle._linearVelocity[1] += _state.getStepSize()*(int(_gravity)*-9.8); 

particle._centre[0] += _state.getStepSize()*particle._linearVelocity[0];
particle._centre[1] += _state.getStepSize()*particle._linearVelocity[1];
particle._centre[2] += _state.getStepSize()*particle._linearVelocity[2];

particle._centreOfMass[0] += _state.getStepSize()*particle._linearVelocity[0];
particle._centreOfMass[1] += _state.getStepSize()*particle._linearVelocity[1];
particle._centreOfMass[2] += _state.getStepSize()*particle._linearVelocity[2];

delta::dynamics::updateRotationMatrix(
particle._angularVelocity.data(),
particle._refAngularVelocity.data(),
particle._orientation.data(), 
_state.getStepSize()
);

#pragma omp parallel for
for(int j=0; j<particle.getNumberOfTriangles()*3; j++)
{
delta::dynamics::updateVertices(
&particle._xCoordinates.data()[j],
&particle._yCoordinates.data()[j],
&particle._zCoordinates.data()[j],
&particle._refxCoordinates.data()[j],
&particle._refyCoordinates.data()[j],
&particle._refzCoordinates.data()[j],
particle._orientation.data(),
particle._centreOfMass.data(),
particle._refCentreOfMass.data()
);
}
}
}

std::vector<delta::core::data::ParticleRecord>& delta::core::Engine::getParticleRecords()
{
return _data.getAll();
}

void delta::core::Engine::plot(std::string path)
{
if(_plot == delta::core::data::Meta::Plot::EveryIteration)
{
delta::core::io::writeGeometryToVTKVTK(path, _state.getCurrentStepIteration(), _data.getAll());
}
}

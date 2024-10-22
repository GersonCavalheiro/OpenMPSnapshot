#pragma once

#include "geodesic_mesh_elements.h"
#include <vector>
#include <cmath>
#include <assert.h>
#include <algorithm>

namespace geodesic {

class Interval;
class IntervalList;
typedef Interval* interval_pointer;
typedef IntervalList* list_pointer;

class Interval 
{
public:
Interval() {}
~Interval() {}

enum DirectionType
{
FROM_FACE_0,
FROM_FACE_1,
FROM_SOURCE,
UNDEFINED_DIRECTION
};

double signal(double x) const 
{
assert(x >= 0.0 && x <= m_edge->length());

if (m_d == GEODESIC_INF) {
return GEODESIC_INF;
} else {
double dx = x - m_pseudo_x;
if (m_pseudo_y == 0.0) {
return m_d + std::abs(dx);
} else {
return m_d + std::sqrt(dx * dx + m_pseudo_y * m_pseudo_y);
}
}
}

double max_distance(double end) const
{
if (m_d == GEODESIC_INF) {
return GEODESIC_INF;
} else {
double a = std::abs(m_start - m_pseudo_x);
double b = std::abs(end - m_pseudo_x);

return a > b ? m_d + std::sqrt(a * a + m_pseudo_y * m_pseudo_y)
: m_d + std::sqrt(b * b + m_pseudo_y * m_pseudo_y);
}
}

void compute_min_distance(double stop) 
{
assert(stop > m_start);

if (m_d == GEODESIC_INF) {
m_min = GEODESIC_INF;
} else if (m_start > m_pseudo_x) {
m_min = signal(m_start);
} else if (stop < m_pseudo_x) {
m_min = signal(stop);
} else {
assert(m_pseudo_y <= 0);
m_min = m_d - m_pseudo_y;
}
}
bool operator()(const interval_pointer x, const interval_pointer y) const
{
if (x->min() != y->min()) {
return x->min() < y->min();
} else if (x->start() != y->start()) {
return x->start() < y->start();
} else {
return x->edge()->id() < y->edge()->id();
}
}

double stop() const 
{
return m_next ? m_next->start() : m_edge->length();
}

double hypotenuse(double a, double b) const { return std::sqrt(a * a + b * b); }

void find_closest_point(double const x,
double const y,
double& offset,
double& distance)
const; 

double& start() { return m_start; }
double& d() { return m_d; }
double& pseudo_x() { return m_pseudo_x; }
double& pseudo_y() { return m_pseudo_y; }
double& min() { return m_min; }
interval_pointer& next() { return m_next; }
edge_pointer& edge() { return m_edge; }
DirectionType& direction() { return m_direction; }
bool visible_from_source() { return m_direction == FROM_SOURCE; }
unsigned& source_index() { return m_source_index; }

void initialize(edge_pointer edge,
const SurfacePoint* point = nullptr,
unsigned source_index = 0);

protected:
double m_start;    
double m_d;        
double m_pseudo_x; 
double m_pseudo_y; 
double m_min;      

interval_pointer m_next;   
edge_pointer m_edge;       
unsigned m_source_index;   
DirectionType m_direction; 
};

struct IntervalWithStop : public Interval
{
public:
double& stop() { return m_stop; }

protected:
double m_stop;
};

class IntervalList 
{
public:
IntervalList() { m_first = nullptr; }
~IntervalList() {}

void clear() { m_first = nullptr; }

void initialize(edge_pointer e)
{
m_edge = e;
m_first = nullptr;
}

interval_pointer covering_interval(
double offset) const 
{
assert(offset >= 0.0 && offset <= m_edge->length());

interval_pointer p = m_first;
while (p && p->stop() < offset) {
p = p->next();
}

return p; 
}

void find_closest_point(const SurfacePoint* point,
double& offset,
double& distance,
interval_pointer& interval) const
{
interval_pointer p = m_first;
distance = GEODESIC_INF;
interval = nullptr;

double x, y;
m_edge->local_coordinates(point, x, y);

while (p) {
if (p->min() < GEODESIC_INF) {
double o, d;
p->find_closest_point(x, y, o, d);
if (d < distance) {
distance = d;
offset = o;
interval = p;
}
}
p = p->next();
}
}

unsigned number_of_intervals() const
{
interval_pointer p = m_first;
unsigned count = 0;
while (p) {
++count;
p = p->next();
}
return count;
}

interval_pointer last()
{
interval_pointer p = m_first;
if (p) {
while (p->next()) {
p = p->next();
}
}
return p;
}

double signal(double x) const
{
const interval_pointer interval = covering_interval(x);

return interval ? interval->signal(x) : GEODESIC_INF;
}

interval_pointer& first() { return m_first; }
edge_pointer& edge() { return m_edge; }

private:
interval_pointer m_first; 
edge_pointer m_edge;      
};

class SurfacePointWithIndex : public SurfacePoint
{
public:
unsigned index() const { return m_index; }

void initialize(const SurfacePoint& p, unsigned index)
{
SurfacePoint::initialize(p);
m_index = index;
}

bool operator()(const SurfacePointWithIndex* x,
const SurfacePointWithIndex* y) const 
{
assert(x->type() != UNDEFINED_POINT && y->type() != UNDEFINED_POINT);

if (x->type() != y->type()) {
return x->type() < y->type();
} else {
return x->base_element()->id() < y->base_element()->id();
}
}

private:
unsigned m_index;
};

class SortedSources : public std::vector<SurfacePointWithIndex>
{
private:
typedef std::vector<SurfacePointWithIndex*> sorted_vector_type;

public:
typedef sorted_vector_type::iterator sorted_iterator;
typedef std::pair<sorted_iterator, sorted_iterator> sorted_iterator_pair;

sorted_iterator_pair sources(base_pointer mesh_element)
{
m_search_dummy.base_element() = mesh_element;

return equal_range(m_sorted.begin(), m_sorted.end(), &m_search_dummy, m_compare_less);
}

void initialize(const std::vector<SurfacePoint>& sources) 
{
resize(sources.size());
m_sorted.resize(sources.size());
for (unsigned i = 0; i < sources.size(); ++i) {
SurfacePointWithIndex& p = *(begin() + i);

p.initialize(sources[i], i);
m_sorted[i] = &p;
}

std::sort(m_sorted.begin(), m_sorted.end(), m_compare_less);
}

SurfacePointWithIndex& operator[](unsigned i)
{
assert(i < size());
return *(begin() + i);
}

private:
sorted_vector_type m_sorted;
SurfacePointWithIndex m_search_dummy; 
SurfacePointWithIndex m_compare_less; 
};

inline void
Interval::find_closest_point(
double const rs,
double const hs,
double& r,
double& d_out) const 
{
if (m_d == GEODESIC_INF) {
r = GEODESIC_INF;
d_out = GEODESIC_INF;
return;
}

double hc = -m_pseudo_y;
double rc = m_pseudo_x;
double end = stop();

double local_epsilon = SMALLEST_INTERVAL_RATIO * m_edge->length();
if (std::abs(hs + hc) < local_epsilon) {
if (rs <= m_start) {
r = m_start;
d_out = signal(m_start) + std::abs(rs - m_start);
} else if (rs >= end) {
r = end;
d_out = signal(end) + fabs(end - rs);
} else {
r = rs;
d_out = signal(rs);
}
} else {
double ri = (rs * hc + hs * rc) / (hs + hc);

if (ri < m_start) {
r = m_start;
d_out = signal(m_start) + hypotenuse(m_start - rs, hs);
} else if (ri > end) {
r = end;
d_out = signal(end) + hypotenuse(end - rs, hs);
} else {
r = ri;
d_out = m_d + hypotenuse(rc - rs, hc + hs);
}
}
}

inline void
Interval::initialize(edge_pointer edge, const SurfacePoint* source, unsigned source_index)
{
m_next = nullptr;
m_direction = UNDEFINED_DIRECTION;
m_edge = edge;
m_source_index = source_index;

m_start = 0.0;
if (!source) {
m_d = GEODESIC_INF;
m_min = GEODESIC_INF;
return;
}
m_d = 0;

if (source->base_element()->type() == VERTEX) {
if (source->base_element()->id() == edge->v0()->id()) {
m_pseudo_x = 0.0;
m_pseudo_y = 0.0;
m_min = 0.0;
return;
} else if (source->base_element()->id() == edge->v1()->id()) {
m_pseudo_x = stop();
m_pseudo_y = 0.0;
m_min = 0.0;
return;
}
}

edge->local_coordinates(source, m_pseudo_x, m_pseudo_y);
m_pseudo_y = -m_pseudo_y;

compute_min_distance(stop());
}

} 

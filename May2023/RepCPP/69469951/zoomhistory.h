


#pragma once


#include <vector>

enum class TZoomPosition { PREV_ZOOM, CURRENT_ZOOM, NEXT_ZOOM };

template <typename Dimension1, typename Dimension2>
class ZoomHistory
{
public:

ZoomHistory();
~ZoomHistory();

void addZoom( Dimension1 begin1, Dimension1 end1,
Dimension2 begin2, Dimension2 end2 );
void addZoom( Dimension1 begin, Dimension1 end );
void addZoom( Dimension2 begin, Dimension2 end );

void setFirstDimension( std::pair<Dimension1, Dimension1> &dim );
void setSecondDimension( std::pair<Dimension2, Dimension2> &dim );

std::pair<Dimension1, Dimension1> getFirstDimension( TZoomPosition pos = TZoomPosition::CURRENT_ZOOM ) const;
std::pair<Dimension2, Dimension2> getSecondDimension( TZoomPosition pos = TZoomPosition::CURRENT_ZOOM ) const;

bool isEmpty( TZoomPosition pos = TZoomPosition::CURRENT_ZOOM ) const;

void firstZoom();
void nextZoom();
void prevZoom();

void clear();

private:
int currentZoom;
std::vector< std::pair< std::pair<Dimension1,Dimension1>, std::pair<Dimension2, Dimension2> > > zooms;

bool sameZoomAsCurrent( Dimension1 begin1, Dimension1 end1,
Dimension2 begin2, Dimension2 end2 ) const;
bool sameZoomAsCurrent( Dimension1 begin, Dimension1 end ) const;
bool sameZoomAsCurrent( Dimension2 begin, Dimension2 end ) const;
};

#include "zoomhistory_impl.h"



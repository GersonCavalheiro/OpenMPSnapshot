#pragma once

#define MERGE_SORT

typedef struct CUSTOM_POSITION
{
double m_dX;
double m_dY;
static double m_sdTol;

CUSTOM_POSITION(double dX = 0., double dY = 0.)
{
m_dX = dX;
m_dY = dY;
}
}CUSTOM_POS;
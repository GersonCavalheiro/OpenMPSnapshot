


#pragma once


#include <string>
#include "progresscontroller.h"

class Timeline;
class Histogram;

enum class TOutput
{
TEXT = 0,
GNUPLOT
};

class Output
{
public: 
static Output *createOutput( TOutput whichOutput );

Output() {}
virtual ~Output() {}

virtual void dumpWindow( Timeline *whichWindow,
std::string& strOutputFile,
ProgressController *progress = nullptr ) = 0;

virtual void dumpHistogram( Histogram *whichHisto,
std::string& strOutputFile,
bool onlySelectedPlane = false,
bool hideEmptyColumns = false,
bool withLabels = true,
bool withPreferencesPrecision = true,
bool recalcHisto = true,
ProgressController *progress = nullptr ) = 0;

virtual bool getMultipleFiles() const = 0;
virtual void setMultipleFiles( bool newValue ) = 0;

virtual bool getObjectHierarchy() const
{
return false;
}
virtual void setObjectHierarchy( bool newValue )
{}

bool getWindowTimeUnits() const
{
return true;
}
void setWindowTimeUnits( bool newValue )
{}

bool getTextualSemantic() const
{
return false;
}
void setTextualSemantic( bool newValue )
{}

protected:

private:

};







#pragma once


#include "output.h"

class GNUPlotOutput: public Output
{
public:
GNUPlotOutput() {}
virtual ~GNUPlotOutput() {}

virtual void dumpWindow( Timeline *whichWindow,
std::string& strOutputFile,
ProgressController *progress = nullptr ) override;
virtual void dumpHistogram( Histogram *whichHisto,
std::string& strOutputFile,
bool onlySelectedPlane = false,
bool hideEmptyColumns = false,
bool withLabels = true,
bool withPreferencesPrecision = true,
bool recalcHisto = true,
ProgressController *progress = nullptr ) override;

virtual bool getMultipleFiles() const override;
virtual void setMultipleFiles( bool newValue ) override;

protected:

private:

};




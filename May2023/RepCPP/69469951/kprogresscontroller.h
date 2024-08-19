


#pragma once


#include "progresscontroller.h"

class KProgressController: public ProgressController
{
public:
KProgressController();
~KProgressController();

void setHandler( ProgressHandler whichHandler, void *callerWindow ) override;
void callHandler( ProgressController *not_used ) override;
double getEndLimit() const override;
void setEndLimit( double limit ) override;
double getCurrentProgress() const override;
void setCurrentProgress( double progress ) override;
void setPartner( ProgressController* partner ) override;
void setStop( bool value ) override;
bool getStop() const override;
void setMessage( std::string whichMessage ) override;
std::string getMessage() const override;
void clearMessageChanged() override;
bool getMessageChanged() const override;

private:
ProgressController *myPartner;

ProgressHandler handler;
void *window;
double endLimit;
double currentProgress;
bool stop;
};




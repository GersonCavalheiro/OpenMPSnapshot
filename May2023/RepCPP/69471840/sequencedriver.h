

#pragma once


#include "traceeditactions.h"

class gTimeline;


class RunAppClusteringAction: public TraceToTraceAction
{
public:
RunAppClusteringAction( TraceEditSequence *whichSequence ) : TraceToTraceAction( whichSequence )
{}
~RunAppClusteringAction()
{}

virtual std::vector<TSequenceStates> getStateDependencies() const override;

virtual bool execute( std::string whichTrace ) override;

protected:

private:

};



class RunAppCutterAction: public TraceToTraceAction
{
public:
RunAppCutterAction( TraceEditSequence *whichSequence ) : TraceToTraceAction( whichSequence )
{}
~RunAppCutterAction()
{}

virtual std::vector<TSequenceStates> getStateDependencies() const override;

virtual bool execute( std::string whichTrace ) override;

protected:

private:

};



class RunAppDimemasAction: public TraceToTraceAction
{
public:
RunAppDimemasAction( TraceEditSequence *whichSequence ) : TraceToTraceAction( whichSequence )
{}
~RunAppDimemasAction()
{}

virtual std::vector<TSequenceStates> getStateDependencies() const override;

virtual bool execute( std::string whichTrace ) override;

protected:

private:

};



class RunAppFoldingAction: public TraceToTraceAction
{
public:
RunAppFoldingAction( TraceEditSequence *whichSequence ) : TraceToTraceAction( whichSequence )
{}
~RunAppFoldingAction()
{}

virtual std::vector<TSequenceStates> getStateDependencies() const override;

virtual bool execute( std::string whichTrace ) override;

protected:

private:

};



class RunSpectralAction: public TraceToTraceAction
{
public:
RunSpectralAction( TraceEditSequence *whichSequence ) : TraceToTraceAction( whichSequence )
{}
~RunSpectralAction()
{}

virtual std::vector<TSequenceStates> getStateDependencies() const override;

virtual bool execute( std::string whichTrace ) override;

protected:

private:

};



class RunProfetAction: public TraceToTraceAction
{
public:
RunProfetAction( TraceEditSequence *whichSequence ) : TraceToTraceAction( whichSequence )
{}
~RunProfetAction()
{}

virtual std::vector<TSequenceStates> getStateDependencies() const override;

virtual bool execute( std::string whichTrace ) override;

protected:

private:

};



class RunAppUserCommandAction: public TraceToTraceAction
{
public:
RunAppUserCommandAction( TraceEditSequence *whichSequence ) : TraceToTraceAction( whichSequence )
{}
~RunAppUserCommandAction()
{}

virtual std::vector<TSequenceStates> getStateDependencies() const override;

virtual bool execute( std::string whichTrace ) override;

protected:

private:

};



class ExternalSortAction: public TraceToTraceAction
{
public:
ExternalSortAction( TraceEditSequence *whichSequence ) : TraceToTraceAction( whichSequence )
{}
~ExternalSortAction()
{}

virtual std::vector<TSequenceStates> getStateDependencies() const override;

virtual bool execute( std::string whichTrace ) override;

protected:

private:

};



class SequenceDriver
{
public:
static void sequenceClustering( gTimeline *whichTimeline );
static void sequenceCutter( gTimeline *whichTimeline );
static void sequenceDimemas( gTimeline *whichTimeline );
static void sequenceFolding( gTimeline *whichTimeline );
static void sequenceSpectral( gTimeline *whichTimeline );
static void sequenceProfet( gTimeline *whichTimeline );
static void sequenceUserCommand( gTimeline *whichTimeline );

};



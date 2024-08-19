

#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

#include <string>
#include <algorithm>

#include <wx/filename.h>
#include <wx/numdlg.h>

#include "sequencedriver.h"
#include "kernelconnection.h"
#include "gtimeline.h"
#include "traceeditsequence.h"
#include "traceeditstates.h"
#include "traceoptions.h"
#include "runscript.h"
#include "wxparaverapp.h"
#include "cutfilterdialog.h"

using namespace std;


vector<TSequenceStates> RunAppClusteringAction::getStateDependencies() const
{
vector<TSequenceStates> tmpStates;
return tmpStates;
}

bool RunAppClusteringAction::execute( std::string whichTrace )
{
bool errorFound = false;

TraceEditSequence *tmpSequence = (TraceEditSequence *)mySequence;
std::string tmpFileName = ( (CSVFileNameState *)tmpSequence->getState( TSequenceStates::csvFileNameState ) )->getData();
RunScript *runAppDialog = wxparaverApp::mainWindow->GetRunApplication();
if( runAppDialog == nullptr )
{
runAppDialog = new RunScript( wxparaverApp::mainWindow );
wxparaverApp::mainWindow->SetRunApplication( runAppDialog );
}
runAppDialog->setTrace( wxString::FromUTF8( whichTrace.c_str() ) );
runAppDialog->setClustering( wxString::FromUTF8( tmpFileName.c_str() ) );

runAppDialog->Show();
runAppDialog->Raise();

return errorFound;
}



vector<TSequenceStates> RunAppFoldingAction::getStateDependencies() const
{
vector<TSequenceStates> tmpStates;
return tmpStates;
}

bool RunAppFoldingAction::execute( std::string whichTrace )
{
bool errorFound = false;

TraceEditSequence *tmpSequence = (TraceEditSequence *)mySequence;
std::string tmpFileName = ( (CSVFileNameState *)tmpSequence->getState( TSequenceStates::csvFileNameState ) )->getData();
RunScript *runAppDialog = wxparaverApp::mainWindow->GetRunApplication();
if( runAppDialog == nullptr )
{
runAppDialog = new RunScript( wxparaverApp::mainWindow );
wxparaverApp::mainWindow->SetRunApplication( runAppDialog );
}
runAppDialog->setTrace( wxString::FromUTF8( whichTrace.c_str() ) );
runAppDialog->setFolding( wxString::FromUTF8( tmpFileName.c_str() ) );

runAppDialog->Show();
runAppDialog->Raise();

return errorFound;
}



vector<TSequenceStates> RunAppDimemasAction::getStateDependencies() const
{
vector<TSequenceStates> tmpStates;
return tmpStates;
}

bool RunAppDimemasAction::execute( std::string whichTrace )
{
bool errorFound = false;

RunScript *runAppDialog = wxparaverApp::mainWindow->GetRunApplication();
if( runAppDialog == nullptr )
{
runAppDialog = new RunScript( wxparaverApp::mainWindow );
wxparaverApp::mainWindow->SetRunApplication( runAppDialog );
}
runAppDialog->setTrace( wxString::FromUTF8( whichTrace.c_str() ) );
runAppDialog->setDimemas();

runAppDialog->Show();
runAppDialog->Raise();

return errorFound;
}



vector<TSequenceStates> RunAppCutterAction::getStateDependencies() const
{
vector<TSequenceStates> tmpStates;
return tmpStates;
}


bool RunAppCutterAction::execute( std::string whichTrace )
{
bool errorFound = false;

CutFilterDialog *cutFilterDialog = new CutFilterDialog( wxparaverApp::mainWindow );

wxparaverApp::mainWindow->MainSettingsCutFilterDialog( cutFilterDialog, whichTrace, true );

TraceEditSequence *tmpSequence = (TraceEditSequence *)mySequence;
TraceOptions *traceOptions = ( (TraceOptionsState *)tmpSequence->getState( TSequenceStates::traceOptionsState ) )->getData();
string dummyXmlName = "";
vector< string > toolOrder;
toolOrder.push_back( TraceCutter::getID() );
wxparaverApp::mainWindow->OptionsSettingCutFilterDialog( cutFilterDialog, traceOptions, dummyXmlName, toolOrder );

cutFilterDialog->Show();

return errorFound;
}



vector<TSequenceStates> RunSpectralAction::getStateDependencies() const
{
vector<TSequenceStates> tmpStates;
return tmpStates;
}


bool RunSpectralAction::execute( std::string whichTrace )
{
bool errorFound = true;
wxString errorMsg;

TraceEditSequence *tmpSequence = (TraceEditSequence *)mySequence;
std::string tmpFileName = ( (CSVFileNameState *)tmpSequence->getState( TSequenceStates::csvFileNameState ) )->getData();

wxString spectralEnvVar = wxString( wxT("SPECTRAL_HOME") );
wxString spectralPath;
if ( wxGetEnv( spectralEnvVar, &spectralPath ) )
{
wxString tmpSep = wxFileName::GetPathSeparator();
wxString spectralBin = spectralPath + tmpSep + _("bin") + tmpSep + _("csv-analysis");
if ( wxFileName::IsFileExecutable( spectralBin ) )
{
wxString traceFileName( _("\"") + wxString::FromUTF8( whichTrace.c_str() ) + _("\"") );
wxString csvFileName( _("\"") + wxString::FromUTF8( tmpFileName.c_str() ) + _("\"") );

long tmpValue = wxGetNumberFromUser( _( "Please enter the number of iterations" ), _( "Iterations" ),
_( "Iterations" ), 3, 0, 100 );
wxString numericParameter;
if( tmpValue == -1 )
numericParameter = _("0");
else
numericParameter = wxString( wxT( "%d" ), tmpValue );

wxString command = _( "/bin/sh -c '") + 
spectralBin + _(" ") +
traceFileName + _(" ") +
csvFileName + _(" ") +
numericParameter + _(" ");
_(" 1>&- 2>&-'");

wxExecute( command, wxEXEC_SYNC );

std::string tmpIterTrace = whichTrace;
size_t lastDot = tmpIterTrace.find_last_of(".");
tmpIterTrace = tmpIterTrace.substr( 0, lastDot ) + std::string( ".iterations.prv" );
wxString tmpIterTrace_wx = wxString::FromUTF8( tmpIterTrace.c_str() );

std::string tmpCFG = wxparaverApp::mainWindow->GetLocalKernel()->getDistributedCFGsPath() + PATH_SEP +
std::string("spectral") + PATH_SEP +
std::string("periodicity.cfg");
wxString tmpCFG_wx = wxString::FromUTF8( tmpCFG.c_str() );

if ( wxFileName::FileExists( tmpIterTrace_wx ) )
{
if ( wxFileName::FileExists( tmpCFG_wx ) )
{
wxparaverApp::mainWindow->DoLoadTrace( tmpIterTrace );
wxparaverApp::mainWindow->DoLoadCFG( tmpCFG );
errorFound = false;
}
else
errorMsg = wxString( _("Missing file:\n\n") ) + tmpCFG_wx;
}
else
errorMsg = wxString( _("Missing file:\n\n") ) + tmpIterTrace_wx;
}
else
errorMsg = wxString( _("Unable to find/execute file:\n\n") ) + spectralBin;
}
else
errorMsg =  wxString( _("Undeclared environment variable:\n\n$") ) + spectralEnvVar;

if ( errorFound )
{
errorMsg += wxString( _("\n\nSpectral sequence aborted.") );
wxMessageBox( errorMsg, _( "Warning" ), wxOK | wxICON_WARNING );
}

return errorFound;
}



vector<TSequenceStates> RunProfetAction::getStateDependencies() const
{
vector<TSequenceStates> tmpStates;
return tmpStates;
}

bool RunProfetAction::execute( std::string whichTrace )
{
bool errorFound = false;

RunScript *runAppDialog = wxparaverApp::mainWindow->GetRunApplication();
if( runAppDialog == nullptr )
{
runAppDialog = new RunScript( wxparaverApp::mainWindow );
wxparaverApp::mainWindow->SetRunApplication( runAppDialog );
}
runAppDialog->setTrace( wxString::FromUTF8( whichTrace.c_str() ) );
runAppDialog->setProfet();

runAppDialog->Show();
runAppDialog->Raise();

return errorFound;
}




vector<TSequenceStates> ExternalSortAction::getStateDependencies() const
{
vector<TSequenceStates> tmpStates;
return tmpStates;
}

bool ExternalSortAction::execute( std::string whichTrace )
{
bool errorFound = false;


return errorFound;
}



vector<TSequenceStates> RunAppUserCommandAction::getStateDependencies() const
{
vector<TSequenceStates> tmpStates;
return tmpStates;
}

bool RunAppUserCommandAction::execute( std::string whichTrace )
{
bool errorFound = false;

RunScript *runAppDialog = wxparaverApp::mainWindow->GetRunApplication();
if( runAppDialog == nullptr )
{
runAppDialog = new RunScript( wxparaverApp::mainWindow );
wxparaverApp::mainWindow->SetRunApplication( runAppDialog );
}
runAppDialog->setTrace( wxString::FromUTF8( whichTrace.c_str() ) );
runAppDialog->setUserCommand();

runAppDialog->Show();
runAppDialog->Raise();

return errorFound;
}



void SequenceDriver::sequenceClustering( gTimeline *whichTimeline )
{
KernelConnection *myKernel =  whichTimeline->GetMyWindow()->getKernel();
TraceEditSequence *mySequence = TraceEditSequence::create( myKernel );

mySequence->pushbackAction( TSequenceActions::csvOutputAction );
mySequence->pushbackAction( TSequenceActions::traceCutterAction );
mySequence->pushbackAction( new RunAppClusteringAction( mySequence ) );

TraceOptions *tmpOptions = TraceOptions::create( myKernel );
tmpOptions->set_by_time( true );
tmpOptions->set_min_cutting_time( whichTimeline->GetMyWindow()->getWindowBeginTime() );
tmpOptions->set_max_cutting_time( whichTimeline->GetMyWindow()->getWindowEndTime() );
tmpOptions->set_original_time( false );
tmpOptions->set_break_states( false );

TraceOptionsState *tmpOptionsState = new TraceOptionsState( mySequence );
tmpOptionsState->setData( tmpOptions );
mySequence->addState( TSequenceStates::traceOptionsState, tmpOptionsState );

TextOutput output;
output.setObjectHierarchy( true );
output.setWindowTimeUnits( false );
CSVOutputState *tmpOutputState = new CSVOutputState( mySequence );
tmpOutputState->setData( output );
mySequence->addState( TSequenceStates::csvOutputState, tmpOutputState );

SourceTimelineState *tmpWindowState = new SourceTimelineState( mySequence );
tmpWindowState->setData( whichTimeline->GetMyWindow() );
mySequence->addState( TSequenceStates::sourceTimelineState, tmpWindowState );

CSVFileNameState *tmpCSVFilenameState = new CSVFileNameState( mySequence );
std::string tmpFileName;
wxFileName tmpTraceName( wxString::FromUTF8( whichTimeline->GetMyWindow()->getTrace()->getFileName().c_str() ) );
tmpTraceName.ClearExt();
tmpTraceName.AppendDir( wxString::FromUTF8( TraceEditSequence::dirNameClustering.c_str() ) );

if( !tmpTraceName.DirExists() )
tmpTraceName.Mkdir();
std::string auxName = whichTimeline->GetMyWindow()->getName() + "_";
std::replace( auxName.begin(), auxName.end(), ',', '-' );
tmpFileName = std::string( tmpTraceName.GetPath( wxPATH_GET_SEPARATOR ).mb_str() ) + auxName.c_str() + std::string( tmpTraceName.GetFullName().mb_str() ) + std::string( ".csv" );

tmpCSVFilenameState->setData( tmpFileName );
mySequence->addState( TSequenceStates::csvFileNameState, tmpCSVFilenameState );

OutputDirSuffixState *tmpOutputDirSuffixState = new OutputDirSuffixState( mySequence );
tmpOutputDirSuffixState->setData( TraceEditSequence::dirNameClustering );
mySequence->addState( TSequenceStates::outputDirSuffixState, tmpOutputDirSuffixState );

vector<std::string> traces;
traces.push_back( whichTimeline->GetMyWindow()->getTrace()->getFileName() );
mySequence->execute( traces );

delete mySequence;
}


void SequenceDriver::sequenceCutter( gTimeline *whichTimeline )
{
KernelConnection *myKernel =  whichTimeline->GetMyWindow()->getKernel();
TraceEditSequence *mySequence = TraceEditSequence::create( myKernel );

mySequence->pushbackAction( new RunAppCutterAction( mySequence ) );

TraceOptions *tmpOptions = TraceOptions::create( myKernel );
tmpOptions->set_by_time( true );
tmpOptions->set_min_cutting_time( whichTimeline->GetMyWindow()->getWindowBeginTime() );
tmpOptions->set_max_cutting_time( whichTimeline->GetMyWindow()->getWindowEndTime() );
tmpOptions->set_original_time( false );
tmpOptions->set_break_states( false );
tmpOptions->set_remLastStates( true );
tmpOptions->set_keep_boundary_events( true );

TraceOptionsState *tmpOptionsState = new TraceOptionsState( mySequence );
tmpOptionsState->setData( tmpOptions );
mySequence->addState( TSequenceStates::traceOptionsState, tmpOptionsState );

vector<std::string> traces;
traces.push_back( whichTimeline->GetMyWindow()->getTrace()->getFileName() );
mySequence->execute( traces );

delete mySequence;
}


void SequenceDriver::sequenceDimemas( gTimeline *whichTimeline )
{
KernelConnection *myKernel =  whichTimeline->GetMyWindow()->getKernel();
TraceEditSequence *mySequence = TraceEditSequence::create( myKernel );

mySequence->pushbackAction( TSequenceActions::traceCutterAction );
mySequence->pushbackAction( new RunAppDimemasAction( mySequence ) );

TraceOptions *tmpOptions = TraceOptions::create( myKernel );
tmpOptions->set_by_time( true );
tmpOptions->set_min_cutting_time( whichTimeline->GetMyWindow()->getWindowBeginTime() );
tmpOptions->set_max_cutting_time( whichTimeline->GetMyWindow()->getWindowEndTime() );
tmpOptions->set_original_time( false );
tmpOptions->set_break_states( false );

TraceOptionsState *tmpOptionsState = new TraceOptionsState( mySequence );
tmpOptionsState->setData( tmpOptions );
mySequence->addState( TSequenceStates::traceOptionsState, tmpOptionsState );

SourceTimelineState *tmpWindowState = new SourceTimelineState( mySequence );
tmpWindowState->setData( whichTimeline->GetMyWindow() );
mySequence->addState( TSequenceStates::sourceTimelineState, tmpWindowState );

std::string tmpFileName;
wxFileName tmpTraceName( wxString::FromUTF8( whichTimeline->GetMyWindow()->getTrace()->getFileName().c_str() ) );
tmpTraceName.ClearExt();
tmpTraceName.AppendDir( wxString::FromUTF8( TraceEditSequence::dirNameDimemas.c_str() ) );

if( !tmpTraceName.DirExists() )
tmpTraceName.Mkdir();

OutputDirSuffixState *tmpOutputDirSuffixState = new OutputDirSuffixState( mySequence );
tmpOutputDirSuffixState->setData( TraceEditSequence::dirNameDimemas );
mySequence->addState( TSequenceStates::outputDirSuffixState, tmpOutputDirSuffixState );

vector<std::string> traces;
traces.push_back( whichTimeline->GetMyWindow()->getTrace()->getFileName() );
mySequence->execute( traces );

delete mySequence;
}


void SequenceDriver::sequenceFolding( gTimeline *whichTimeline )
{
KernelConnection *myKernel =  whichTimeline->GetMyWindow()->getKernel();
TraceEditSequence *mySequence = TraceEditSequence::create( myKernel );

mySequence->pushbackAction( TSequenceActions::csvOutputAction );
mySequence->pushbackAction( TSequenceActions::traceCutterAction );
mySequence->pushbackAction( new RunAppFoldingAction( mySequence ) );

TraceOptions *tmpOptions = TraceOptions::create( myKernel );
tmpOptions->set_by_time( true );
tmpOptions->set_min_cutting_time( whichTimeline->GetMyWindow()->getWindowBeginTime() );
tmpOptions->set_max_cutting_time( whichTimeline->GetMyWindow()->getWindowEndTime() );
tmpOptions->set_original_time( false );
tmpOptions->set_break_states( false );

TraceOptionsState *tmpOptionsState = new TraceOptionsState( mySequence );
tmpOptionsState->setData( tmpOptions );
mySequence->addState( TSequenceStates::traceOptionsState, tmpOptionsState );

TextOutput output;
output.setObjectHierarchy( true );
output.setWindowTimeUnits( false );
output.setTextualSemantic( true );
CSVOutputState *tmpOutputState = new CSVOutputState( mySequence );
tmpOutputState->setData( output );
mySequence->addState( TSequenceStates::csvOutputState, tmpOutputState );

SourceTimelineState *tmpWindowState = new SourceTimelineState( mySequence );
tmpWindowState->setData( whichTimeline->GetMyWindow() );
mySequence->addState( TSequenceStates::sourceTimelineState, tmpWindowState );

CSVFileNameState *tmpCSVFilenameState = new CSVFileNameState( mySequence );
std::string tmpFileName;
wxFileName tmpTraceName( wxString::FromUTF8( whichTimeline->GetMyWindow()->getTrace()->getFileName().c_str() ) );
tmpTraceName.ClearExt();
tmpTraceName.AppendDir( wxString::FromUTF8( TraceEditSequence::dirNameFolding.c_str() ) );

if( !tmpTraceName.DirExists() )
tmpTraceName.Mkdir();
std::string auxName = whichTimeline->GetMyWindow()->getName() + "_";
tmpFileName = std::string( tmpTraceName.GetPath( wxPATH_GET_SEPARATOR ).mb_str() ) +
auxName.c_str() + std::string( tmpTraceName.GetFullName().mb_str() ) +
std::string( ".csv" );

tmpCSVFilenameState->setData( tmpFileName );
mySequence->addState( TSequenceStates::csvFileNameState, tmpCSVFilenameState );

OutputDirSuffixState *tmpOutputDirSuffixState = new OutputDirSuffixState( mySequence );
tmpOutputDirSuffixState->setData( TraceEditSequence::dirNameFolding );
mySequence->addState( TSequenceStates::outputDirSuffixState, tmpOutputDirSuffixState );

vector<std::string> traces;
traces.push_back( whichTimeline->GetMyWindow()->getTrace()->getFileName() );
mySequence->execute( traces );

delete mySequence;
}


void SequenceDriver::sequenceSpectral( gTimeline *whichTimeline )
{
KernelConnection *myKernel =  whichTimeline->GetMyWindow()->getKernel();
TraceEditSequence *mySequence = TraceEditSequence::create( myKernel );

mySequence->pushbackAction( TSequenceActions::csvOutputAction );
mySequence->pushbackAction( new RunSpectralAction( mySequence ) );

Timeline *tmpWindow = whichTimeline->GetMyWindow()->clone();
tmpWindow->setLevel( TTraceLevel::APPLICATION );
tmpWindow->setTimeUnit( NS );

TraceOptions *tmpOptions = TraceOptions::create( myKernel );
tmpOptions->set_by_time( true );
tmpOptions->set_min_cutting_time( tmpWindow->getWindowBeginTime() );
tmpOptions->set_max_cutting_time( tmpWindow->getWindowEndTime() );
tmpOptions->set_original_time( false );
tmpOptions->set_break_states( false );


TraceOptionsState *tmpOptionsState = new TraceOptionsState( mySequence );
tmpOptionsState->setData( tmpOptions );
mySequence->addState( TSequenceStates::traceOptionsState, tmpOptionsState );

TextOutput output;
output.setObjectHierarchy( true );
output.setWindowTimeUnits( false );
output.setTextualSemantic( true );
CSVOutputState *tmpOutputState = new CSVOutputState( mySequence );
tmpOutputState->setData( output );
mySequence->addState( TSequenceStates::csvOutputState, tmpOutputState );

SourceTimelineState *tmpWindowState = new SourceTimelineState( mySequence );
tmpWindowState->setData( tmpWindow );
mySequence->addState( TSequenceStates::sourceTimelineState, tmpWindowState );

CSVFileNameState *tmpCSVFilenameState = new CSVFileNameState( mySequence );
std::string tmpFileName;
wxFileName tmpTraceName( wxString::FromUTF8( tmpWindow->getTrace()->getFileName().c_str() ) );
tmpTraceName.ClearExt();
tmpTraceName.AppendDir( wxString::FromUTF8( TraceEditSequence::dirNameSpectral.c_str() ) );
if( !tmpTraceName.DirExists() )
tmpTraceName.Mkdir();
std::string auxName = tmpWindow->getName() + "_";
tmpFileName = std::string( tmpTraceName.GetPath( wxPATH_GET_SEPARATOR ).mb_str() ) + 
auxName.c_str() + std::string( tmpTraceName.GetFullName().mb_str() ) +
std::string( ".csv" );

tmpCSVFilenameState->setData( tmpFileName );
mySequence->addState( TSequenceStates::csvFileNameState, tmpCSVFilenameState );

OutputDirSuffixState *tmpOutputDirSuffixState = new OutputDirSuffixState( mySequence );
tmpOutputDirSuffixState->setData( TraceEditSequence::dirNameSpectral );
mySequence->addState( TSequenceStates::outputDirSuffixState, tmpOutputDirSuffixState );

vector<std::string> traces;
traces.push_back( tmpWindow->getTrace()->getFileName() );
mySequence->execute( traces );

delete tmpWindow;
delete mySequence;
}


void SequenceDriver::sequenceProfet( gTimeline *whichTimeline )
{
KernelConnection *myKernel =  whichTimeline->GetMyWindow()->getKernel();
TraceEditSequence *mySequence = TraceEditSequence::create( myKernel );

mySequence->pushbackAction( TSequenceActions::traceCutterAction );
mySequence->pushbackAction( new RunProfetAction( mySequence ) );

TraceOptions *tmpOptions = TraceOptions::create( myKernel );
tmpOptions->set_by_time( true );
tmpOptions->set_min_cutting_time( whichTimeline->GetMyWindow()->getWindowBeginTime() );
tmpOptions->set_max_cutting_time( whichTimeline->GetMyWindow()->getWindowEndTime() );
tmpOptions->set_original_time( false );
tmpOptions->set_break_states( false );
tmpOptions->set_remFirstStates( false );
tmpOptions->set_remLastStates( true );
tmpOptions->set_keep_all_events( true );
tmpOptions->set_max_cut_time_to_finish_of_first_appl( true );
TraceOptionsState *tmpOptionsState = new TraceOptionsState( mySequence );
tmpOptionsState->setData( tmpOptions );
mySequence->addState( TSequenceStates::traceOptionsState, tmpOptionsState );

SourceTimelineState *tmpWindowState = new SourceTimelineState( mySequence );
tmpWindowState->setData( whichTimeline->GetMyWindow() );
mySequence->addState( TSequenceStates::sourceTimelineState, tmpWindowState );

std::string tmpFileName;
wxFileName tmpTraceName( wxString::FromUTF8( whichTimeline->GetMyWindow()->getTrace()->getFileName().c_str() ) );
tmpTraceName.ClearExt();
tmpTraceName.AppendDir( wxString::FromUTF8( TraceEditSequence::dirNameProfet.c_str() ) );  
if( !tmpTraceName.DirExists() )
tmpTraceName.Mkdir();

OutputDirSuffixState *tmpOutputDirSuffixState = new OutputDirSuffixState( mySequence );
tmpOutputDirSuffixState->setData( TraceEditSequence::dirNameProfet );
mySequence->addState( TSequenceStates::outputDirSuffixState, tmpOutputDirSuffixState );

vector<std::string> traces;
traces.push_back( whichTimeline->GetMyWindow()->getTrace()->getFileName() );
mySequence->execute( traces );

delete mySequence;
}


void SequenceDriver::sequenceUserCommand( gTimeline *whichTimeline )
{
KernelConnection *myKernel =  whichTimeline->GetMyWindow()->getKernel();
TraceEditSequence *mySequence = TraceEditSequence::create( myKernel );

mySequence->pushbackAction( TSequenceActions::traceCutterAction );
mySequence->pushbackAction( new RunAppUserCommandAction( mySequence ) );

TraceOptions *tmpOptions = TraceOptions::create( myKernel );
tmpOptions->set_by_time( true );
tmpOptions->set_min_cutting_time( whichTimeline->GetMyWindow()->getWindowBeginTime() );
tmpOptions->set_max_cutting_time( whichTimeline->GetMyWindow()->getWindowEndTime() );
tmpOptions->set_original_time( false );
tmpOptions->set_break_states( false );

TraceOptionsState *tmpOptionsState = new TraceOptionsState( mySequence );
tmpOptionsState->setData( tmpOptions );
mySequence->addState( TSequenceStates::traceOptionsState, tmpOptionsState );

SourceTimelineState *tmpWindowState = new SourceTimelineState( mySequence );
tmpWindowState->setData( whichTimeline->GetMyWindow() );
mySequence->addState( TSequenceStates::sourceTimelineState, tmpWindowState );

std::string tmpFileName;
wxFileName tmpTraceName( wxString::FromUTF8( whichTimeline->GetMyWindow()->getTrace()->getFileName().c_str() ) );
tmpTraceName.ClearExt();
tmpTraceName.AppendDir( wxString::FromUTF8( TraceEditSequence::dirNameUserCommand.c_str() ) );

if( !tmpTraceName.DirExists() )
tmpTraceName.Mkdir();

OutputDirSuffixState *tmpOutputDirSuffixState = new OutputDirSuffixState( mySequence );
tmpOutputDirSuffixState->setData( TraceEditSequence::dirNameUserCommand );
mySequence->addState( TSequenceStates::outputDirSuffixState, tmpOutputDirSuffixState );

vector<std::string> traces;
traces.push_back( whichTimeline->GetMyWindow()->getTrace()->getFileName() );
mySequence->execute( traces );

delete mySequence;
}


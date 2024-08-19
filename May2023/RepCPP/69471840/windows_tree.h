


#pragma once


#include <wx/treectrl.h>
#include <wx/choicebk.h>
#include <vector>
#include <string>

class gTimeline;
class gHistogram;
class paraverMain;
class Timeline;
class Histogram;
class Trace;

class gWindow
{
public:
gWindow()
{
enableButtonDestroy = true;
}

bool getEnableDestroyButton() const { return enableButtonDestroy ; }
virtual void setEnableDestroyButton( bool value ) { enableButtonDestroy = value ; }

private:
bool enableButtonDestroy;
};


wxTreeCtrl * createTree( wxImageList *imageList );
wxTreeCtrl *getAllTracesTree();
wxTreeCtrl *getSelectedTraceTree( Trace *trace );

void appendHistogram2Tree( gHistogram *ghistogram );

wxTreeItemId getItemIdFromWindow( wxTreeItemId root, Timeline *wanted, bool &found );
wxTreeItemId getItemIdFromGTimeline( wxTreeItemId root, gTimeline *wanted, bool &found );
gTimeline *getGTimelineFromWindow( wxTreeItemId root, Timeline *wanted, bool &found );
gHistogram *getGHistogramFromWindow( wxTreeItemId root, Histogram *wanted );
void getParentGTimeline( gTimeline *current, std::vector< gTimeline * > & children );

void BuildTree( paraverMain *parent,
wxTreeCtrl *root1, wxTreeItemId idRoot1,
wxTreeCtrl *root2, wxTreeItemId idRoot2,
Timeline *window,
std::string nameSuffix = std::string("") );

bool updateTreeItem( wxTreeCtrl *tree,
wxTreeItemId& id,
std::vector< Timeline * > &allWindows,
std::vector< Histogram * > &allHistograms,
wxWindow **currentWindow,
bool allTracesTree );

void iconizeWindows( wxTreeCtrl *tree,
wxTreeItemId& id,
bool iconize );

int getIconNumber( Timeline *whichWindow );



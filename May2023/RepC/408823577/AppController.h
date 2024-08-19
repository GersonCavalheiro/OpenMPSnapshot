#import <Cocoa/Cocoa.h>
#import <Carbon/Carbon.h>
#import "FractalView.h"
#include "ui.h"
@class VideatorProxy;
@interface AppController:NSObject {
FractalView *view;
NSWindow *window;
BOOL applicationIsLaunched;
}
#pragma mark Accessors
-(FractalView *) view;
#pragma mark Driver Initialization
-(void) initLocale;
-(int) initDriver:(struct ui_driver *)driver fullscreen:(BOOL) fullscreen;
-(void) uninitDriver;
#pragma mark Menus
-(void) localizeApplicationMenu;
-(void) performMenuAction:(NSMenuItem *) sender;
-(NSString *) keyEquivalentForName:(NSString *) name;
-(void) buildMenuWithContext:(struct uih_context *)context name:(CONST char *) name;
-(void) buildMenuWithContext:(struct uih_context *)context name:(CONST char *)menuName parent:(NSMenu *) parentMenu;
-(void) buildMenuWithContext:(struct uih_context *)context name:(CONST char *)menuName parent:(NSMenu *)parentMenu isNumbered:(BOOL) isNumbered;
-(void) showPopUpMenuWithContext:(struct uih_context *)context name:(CONST char *) name;
#pragma mark Dialogs
-(void) showDialogWithContext:(struct uih_context *)context name:(CONST char *) name;
#pragma mark Help
-(void) showHelpWithContext:(struct uih_context *)context name:(CONST char *) name;
@end extern AppController *controller;

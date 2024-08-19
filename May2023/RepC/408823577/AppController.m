#import "AppController.h"
#import "CustomDialog.h"
#import "VideatorProxy.h"
#import "ui.h"
#ifdef HAVE_GETTEXT
#include <libintl.h>
#include <locale.h>
#define _(string) gettext(string)
#else
#define _(string) (string)
#endif
@implementation NSWindow (CanBecomeKeyWindowOverride)
- (BOOL)canBecomeKeyWindow {
return YES;
}
@end
AppController *controller;
@implementation AppController
#pragma mark Initialization
- (id)init {
self = [super init];
if (self) {
applicationIsLaunched = NO;
[self initLocale];
}
return self;
}
#pragma mark Accessors
- (FractalView *)view {
return view;
}
#pragma mark Driver Initialization
- (void)initLocale {
NSString *myLocalePath = [[[NSBundle mainBundle] resourcePath] 
stringByAppendingPathComponent:@"locale"];
#ifdef USE_LOCALEPATH
localepath = (char *)[myLocalePath UTF8String];
#endif
NSMutableArray *supportedLanguages = [[[NSFileManager defaultManager]
directoryContentsAtPath:myLocalePath] 
mutableCopy];
[supportedLanguages addObject:@"en"];
NSUserDefaults * defaults = [NSUserDefaults standardUserDefaults];
NSArray *preferredLanguages = [defaults objectForKey:@"AppleLanguages"];
NSString *lang = [preferredLanguages firstObjectCommonWithArray:supportedLanguages];
if (lang) setenv("LANG", [lang UTF8String],  0);
[supportedLanguages release];
}    
- (int)initDriver:(struct ui_driver *)driver 
fullscreen:(BOOL)fullscreen {
CGDirectDisplayID displayID = (CGDirectDisplayID)[[[[NSScreen mainScreen] deviceDescription] objectForKey:@"NSScreenNumber"] intValue];
CGSize displaySize = CGDisplayScreenSize(displayID);
NSSize displayResolution = [[NSScreen mainScreen] frame].size;
driver->width = (displaySize.width/displayResolution.width)/10;
driver->height = (displaySize.height/displayResolution.height)/10;
if (fullscreen) {
SetSystemUIMode(kUIModeAllHidden, kUIOptionAutoShowMenuBar);
window = [[NSWindow alloc] initWithContentRect:[[NSScreen mainScreen] frame]
styleMask:NSBorderlessWindowMask 
backing:NSBackingStoreBuffered
defer:YES];
} else {
window = [[NSWindow alloc] initWithContentRect:NSMakeRect(50, 50, 640, 480) 
styleMask:(NSTitledWindowMask | NSClosableWindowMask | NSMiniaturizableWindowMask | NSResizableWindowMask) 
backing:NSBackingStoreBuffered 
defer:YES];
[window setFrameAutosaveName:@"XaoSWindow"];
}
view = [[FractalView alloc] initWithFrame:[[window contentView] frame]];
[view setAutoresizingMask:(NSViewWidthSizable | NSViewHeightSizable)];
[[window contentView] addSubview:view];
[window makeFirstResponder:view];
[window setDelegate:self];
[window setTitle:@"XaoS"];
[window makeKeyAndOrderFront:self];
[NSApp setDelegate:self];
if (!applicationIsLaunched) {
[self localizeApplicationMenu];
[NSApp finishLaunching];
applicationIsLaunched = YES;
}
return 1; 
}
- (void)uninitDriver {
SetSystemUIMode(kUIModeNormal, 0);
[view release];
[window release];
}
#pragma mark Menus
- (void)localizeApplicationMenu {
NSMenu *appMenu = [[[NSApp mainMenu] itemAtIndex:0] submenu];
[[appMenu itemWithTitle:@"About XaoS"] 
setTitle:[NSString stringWithUTF8String:_("About XaoS")]];
[[appMenu itemWithTitle:@"Services"] 
setTitle:[NSString stringWithUTF8String:_("Services")]];
[[appMenu itemWithTitle:@"Hide XaoS"] 
setTitle:[NSString stringWithUTF8String:_("Hide XaoS")]];
[[appMenu itemWithTitle:@"Hide Others"] 
setTitle:[NSString stringWithUTF8String:_("Hide Others")]];
[[appMenu itemWithTitle:@"Show All"] 
setTitle:[NSString stringWithUTF8String:_("Show All")]];
[[appMenu itemWithTitle:@"Quit XaoS"] 
setTitle:[NSString stringWithUTF8String:_("Quit XaoS")]];
}
- (void)performMenuAction:(NSMenuItem *)sender {
NSString *name = [sender representedObject];
CONST menuitem *item = menu_findcommand([name UTF8String]);
ui_menuactivate(item, NULL);
}
- (NSString *)keyEquivalentForName:(NSString *)name {
if ([name isEqualToString:@"undo"]) return @"z";
if ([name isEqualToString:@"redo"]) return @"Z";
if ([name isEqualToString:@"loadpos"]) return @"o";
if ([name isEqualToString:@"savepos"]) return @"s";
return @"";
}
- (void)buildMenuWithContext:(struct uih_context *)context 
name:(CONST char *)name {
NSMenu *menu = [NSApp mainMenu];
while ([menu numberOfItems] > 1)
[menu removeItemAtIndex:1];
[self buildMenuWithContext:context name:name parent:menu];
}
- (void)buildMenuWithContext:(struct uih_context *)context 
name:(CONST char *)menuName 
parent:(NSMenu *)parentMenu {
[self buildMenuWithContext:context
name:menuName
parent:parentMenu
isNumbered:NO];
}
- (void)buildMenuWithContext:(struct uih_context *)context 
name:(CONST char *)menuName 
parent:(NSMenu *)parentMenu
isNumbered:(BOOL)isNumbered {
int i, n;
CONST menuitem *item;
for (i=0,n=1; (item = menu_item(menuName, i)) != NULL; i++)	{
if (item->type == MENU_SEPARATOR) {
[parentMenu addItem:[NSMenuItem separatorItem]];
} else {
NSString *menuTitle = [NSString stringWithUTF8String:item->name];
if (item->type == MENU_CUSTOMDIALOG || item->type == MENU_DIALOG)
menuTitle = [menuTitle stringByAppendingString:@"..."];
NSString *menuShortName = [NSString stringWithUTF8String:item->shortname];
NSString *keyEquiv = [self keyEquivalentForName:menuShortName];
if (item->key && parentMenu != [NSApp mainMenu])
menuTitle = [NSString stringWithFormat:@"%@ (%s)", menuTitle, item->key];
NSMenuItem *newItem = [[NSMenuItem allocWithZone:[NSMenu menuZone]] initWithTitle:menuTitle action:nil keyEquivalent:keyEquiv];
if (isNumbered && item->type != MENU_SUBMENU) {
if (n < 10)
keyEquiv = [NSString stringWithFormat:@"%d", n];
else if (n == 10)
keyEquiv = @"0";
else if (n < 36)
keyEquiv = [NSString stringWithFormat:@"%c", 'a' + n - 11];
[newItem setKeyEquivalent:keyEquiv];
[newItem setKeyEquivalentModifierMask:0];
n++;
}
if (item->type == MENU_SUBMENU) {
NSMenu *newMenu = [[NSMenu allocWithZone:[NSMenu menuZone]] initWithTitle:menuTitle];
[newMenu setDelegate:self];
[newItem setSubmenu:newMenu];
[self buildMenuWithContext:context name:item->shortname parent:newMenu];
if ([menuShortName isEqualToString:@"edit"]) {
[newMenu addItem:[NSMenuItem separatorItem]];
[newMenu addItemWithTitle:[NSString stringWithUTF8String:_("Cut")]
action:@selector(cut:) keyEquivalent:@"x"];
[newMenu addItemWithTitle:[NSString stringWithUTF8String:_("Copy")]
action:@selector(copy:) keyEquivalent:@"c"];
[newMenu addItemWithTitle:[NSString stringWithUTF8String:_("Paste")]
action:@selector(paste:) keyEquivalent:@"v"];
[newMenu addItemWithTitle:[NSString stringWithUTF8String:_("Delete")]
action:@selector(delete:) keyEquivalent:@""];
[newMenu addItemWithTitle:[NSString stringWithUTF8String:_("Select All")]
action:@selector(selectAll:) keyEquivalent:@"a"];
}
if ([menuShortName isEqualToString:@"window"]) {
[newMenu addItemWithTitle:[NSString stringWithUTF8String:_("Minimize")]
action:@selector(performMiniaturize:) keyEquivalent:@"m"];
[newMenu addItemWithTitle:[NSString stringWithUTF8String:_("Zoom")]
action:@selector(performZoom:) keyEquivalent:@""];
[newMenu addItem:[NSMenuItem separatorItem]];
[newMenu addItemWithTitle:[NSString stringWithUTF8String:_("Bring All to Front")]
action:@selector(arrangeInFront:) keyEquivalent:@""];
}
if ([menuShortName isEqualToString:@"file"]) {
int i = [newMenu indexOfItemWithRepresentedObject:@"savepos"];
[newMenu insertItemWithTitle:[NSString stringWithUTF8String:_("Close")] 
action:@selector(performClose:)
keyEquivalent:@"w"
atIndex:i];
[newMenu insertItem:[NSMenuItem separatorItem] atIndex:i];
}
#ifdef VIDEATOR_SUPPORT
if ([menuShortName isEqualToString:@"ui"]) {
int i = [newMenu indexOfItemWithRepresentedObject:@"inhibittextoutput"]+1;
NSMenuItem *item = [newMenu insertItemWithTitle:[NSString stringWithUTF8String:_("Videator Output")] 
action:@selector(toggleVideator:) 
keyEquivalent:@""
atIndex:i];
[item setTarget:[view videatorProxy]];
[item setRepresentedObject:@"videator"];
}
#endif
[newMenu release];
} else {
[newItem setTarget:self];
[newItem setAction:@selector(performMenuAction:)];
[newItem setRepresentedObject:menuShortName];
if (item->flags & (MENUFLAG_RADIO | MENUFLAG_CHECKBOX) && menu_enabled (item, context))
[newItem setState:NSOnState];
}
[parentMenu addItem:newItem];
[newItem release];
}
}
}
- (void)menuNeedsUpdate:(NSMenu *)menu {
CONST struct menuitem *xaosItem;
NSMenuItem *menuItem;
NSEnumerator *itemEnumerator = [[menu itemArray] objectEnumerator];
while (menuItem = [itemEnumerator nextObject]) {
if ([menuItem representedObject]) {
xaosItem = menu_findcommand([[menuItem representedObject] UTF8String]);
if (xaosItem)
[menuItem setState:(menu_enabled(xaosItem, globaluih) ? NSOnState : NSOffState)];
#ifdef VIDEATOR_SUPPORT
else if ([[menuItem representedObject] isEqualToString:@"videator"])
[menuItem setState:([[view videatorProxy] videatorEnabled] ? NSOnState : NSOffState)];
#endif
}
}
}
- (void)showPopUpMenuWithContext:(struct uih_context *)context 
name:(CONST char *)name {
NSMenu *popUpMenu = [[NSMenu alloc] initWithTitle:@"Popup Menu"];
NSPopUpButtonCell *popUpButtonCell = [[NSPopUpButtonCell alloc] initTextCell:@"" pullsDown:NO];
NSRect frame = {{0.0, 0.0}, {0.0, 0.0}};
frame.origin = [window mouseLocationOutsideOfEventStream];
[self buildMenuWithContext:context name:name parent:popUpMenu isNumbered:YES];
int state = [[popUpMenu itemAtIndex:0] state];
[popUpButtonCell setMenu:popUpMenu];
[[popUpMenu itemAtIndex:0] setState:state];
[popUpButtonCell performClickWithFrame:frame inView:view];
[popUpButtonCell release];
[popUpMenu release];
}
#pragma mark Dialogs
- (void)showDialogWithContext:(struct uih_context *)context 
name:(CONST char *)name {
CONST menuitem *item = menu_findcommand (name);
if (!item) return;
CONST menudialog *dialog = menu_getdialog (context, item);
if (!dialog) return;
int nitems;
for (nitems = 0; dialog[nitems].question; nitems++);
if (nitems == 1 && (dialog[0].type == DIALOG_IFILE || dialog[0].type == DIALOG_OFILE)) {
NSString *extension = [[NSString stringWithUTF8String:dialog[0].defstr] pathExtension];
NSString *fileName = nil;
switch(dialog[0].type) {
case DIALOG_IFILE:
{
NSOpenPanel *oPanel = [NSOpenPanel openPanel];
int result = [oPanel runModalForDirectory:nil
file:nil 
types:[NSArray arrayWithObject:extension]];
if (result == NSOKButton)
fileName = [oPanel filename];
break;
}
case DIALOG_OFILE:
{
NSSavePanel *sPanel = [NSSavePanel savePanel];
[sPanel setRequiredFileType:extension];
int result = [sPanel runModalForDirectory:nil file:@"untitled"];
if (result == NSOKButton)
fileName = [sPanel filename];
break;
}
}
[window makeKeyAndOrderFront:self];
if (fileName) {
dialogparam *param = malloc (sizeof (dialogparam));
param->dstring = strdup([fileName UTF8String]);
ui_menuactivate (item, param);
}
} else {
CustomDialog *customDialog = [[CustomDialog alloc] initWithContext:context 
menuItem:item 
dialog:dialog];
[NSApp beginSheet:customDialog 
modalForWindow:window 
modalDelegate:nil 
didEndSelector:nil 
contextInfo:nil];
[NSApp runModalForWindow:customDialog];
[NSApp endSheet:customDialog];
[customDialog orderOut:self];
[window makeKeyAndOrderFront:self];
if ([customDialog params])
ui_menuactivate(item, [customDialog params]);
[customDialog release];
}
}
#pragma mark Help
- (void)showHelpWithContext:(struct uih_context *)context 
name:(CONST char *)name {
NSString *anchor = [NSString stringWithUTF8String:name];
[[NSHelpManager sharedHelpManager] openHelpAnchor:anchor inBook:@"XaoS Help"];
}
#pragma mark Window Delegates
- (void)windowWillClose:(NSNotification *)notification {
[NSApp terminate:self];
}
- (void)windowDidResize:(NSNotification *)notification {
if (![view inLiveResize])
ui_resize();
}
#pragma mark Application Delegates
- (BOOL)application:(NSApplication *)theApplication 
openFile:(NSString *)filename {
if ([[filename pathExtension] isEqualToString:@"xpf"]) {
uih_loadfile(globaluih, [filename UTF8String]);
return YES;
} else if ([[filename pathExtension] isEqualToString:@"xaf"]) {
uih_playfile(globaluih, [filename UTF8String]);
return YES;
} else {
return NO;
}
}
@end
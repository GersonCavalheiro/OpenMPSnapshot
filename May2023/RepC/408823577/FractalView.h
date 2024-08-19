#import <Cocoa/Cocoa.h>
#include "ui.h"
#ifdef VIDEATOR_SUPPORT
@ class VideatorProxy;
#endif
@interface FractalView: NSOpenGLView {
int mouseX, mouseY;
int mouseButton, rightMouseButton, otherMouseButton, mouseScrollWheel;
int keysDown;
int cursorType;
int width, height;
int currentBuffer;
unsigned char *buffer[2];
NSString *messageText;
NSPoint messageLocation;
#ifdef VIDEATOR_SUPPORT
VideatorProxy *videatorProxy;
#endif
}
#pragma mark Buffers
-(int) allocBuffer1:(char **)b1 buffer2:(char **) b2;
-(void) freeBuffers;
-(void) flipBuffers;
#pragma mark Accessors
#ifdef VIDEATOR_SUPPORT
-(VideatorProxy *) videatorProxy;
#endif
-(void) getWidth:(int *)w height:(int *) h;
-(void) getMouseX:(int *)mx mouseY:(int *)my mouseButton:(int *) mb;
-(void) getMouseX:(int *)mx mouseY:(int *)my mouseButton:(int *)mb keys:(int *) k;
#pragma mark Cursor
-(void) setCursorType:(int) type;
#pragma mark Text
-(void) printText:(CONST char *)text atX:(int)x y:(int) y;
- (NSDictionary *) textAttributes;
@end

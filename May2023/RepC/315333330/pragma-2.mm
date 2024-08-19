@interface A
{
int p;
}
+(int) foo;
-(int) bar;
@end
@implementation A
#pragma mark -
#pragma mark init / dealloc
+ (int)foo {
return 1;
}
#pragma mark -
#pragma mark Private Functions
- (int)bar {
return 2;
}
@end

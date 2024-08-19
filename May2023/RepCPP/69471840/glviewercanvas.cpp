





#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

#include "wx/imaglist.h"

#include "glviewercanvas.h"
#include "paravermain.h"
#include "semanticcolor.h"

#include "wx/sizer.h"
#include "window.h"



#ifdef PARALLEL_ENABLED 
#include "omp.h"
#endif

#include <GL/glu.h> 

using std::map;


BEGIN_EVENT_TABLE(GLViewerCanvas, wxGLCanvas)  
EVT_MOTION(GLViewerCanvas::mouseMoved)
EVT_LEFT_DOWN(GLViewerCanvas::mouseDown)
EVT_LEFT_UP(GLViewerCanvas::mouseReleased)
EVT_RIGHT_DOWN(GLViewerCanvas::rightClick)
EVT_LEAVE_WINDOW(GLViewerCanvas::mouseLeftWindow)
EVT_SIZE(GLViewerCanvas::resized)
EVT_KEY_DOWN(GLViewerCanvas::keyPressed)
EVT_KEY_UP(GLViewerCanvas::keyReleased)
EVT_MOUSEWHEEL(GLViewerCanvas::mouseWheelMoved)
EVT_PAINT(GLViewerCanvas::render)
END_EVENT_TABLE()


const char *vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"layout (location = 1) in vec3 aColour;\n"
"layout (location = 2) in vec2 aTexCoord;\n"
"out vec3 colour;\n"
"out vec2 texCoord;\n"
"uniform mat4 model;\n"
"uniform mat4 view;\n"
"uniform mat4 projection;\n"
"void main()\n"
"{\n"
"   
"   texCoord = aTexCoord;\n"
"   gl_Position =  vec4( aPos.xyz, 1.0 );\n"
"}\0";

const char *fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"in vec3 colour;\n"
"in vec2 texCoord;\n"
"uniform sampler2D texture1;\n"
"void main()\n"
"{\n"
"   FragColor = vec4( 1.0f, 0.0f, 0.2f, 1.0f );\n"
"   FragColor = texture( texture1, texCoord );\n"
"}\n\0";





int FPS = 0;
time_t iniTime;
time_t endTime;

void GLViewerCanvas::mouseMoved(wxMouseEvent& event) 
{
float xMouse = event.GetX();
float yMouse = event.GetY();
float factor = 0.1f;

if ( event.Dragging() )
spin( (yMouse - yPrevMouse)*factor, (xMouse - xPrevMouse)*factor );

xPrevMouse = xMouse;
yPrevMouse = yMouse;
}

void GLViewerCanvas::mouseDown(wxMouseEvent& event) 
{
float xMouse = event.GetX();
float yMouse = event.GetY();
}

void GLViewerCanvas::mouseReleased(wxMouseEvent& event) {}
void GLViewerCanvas::rightClick(wxMouseEvent& event) {}
void GLViewerCanvas::mouseLeftWindow(wxMouseEvent& event) {}
void GLViewerCanvas::keyReleased(wxKeyEvent& event) {}

void GLViewerCanvas::mouseWheelMoved(wxMouseEvent& event) 
{
float zoom = event.GetWheelRotation();
float xrotrad = (xAngle / 180 * 3.141592654f);
camY -= float(sin(xrotrad)) * zoom * 0.0001f;

Refresh( false );
}

void GLViewerCanvas::keyPressed(wxKeyEvent& event) 
{
float factor = 0.2f;
float mx, my, mz,   ax, ay;
float xrotrad, yrotrad;
mx = my = mz = ax = ay = 0.0f; 
switch ( event.GetKeyCode() )
{ 
case (long) 'W':

xrotrad = (xAngle / 180 * 3.141592654f);
yrotrad = (yAngle / 180 * 3.141592654f);
mx -= float(sin(yrotrad)) * factor;
mz += float(cos(yrotrad)) * factor;  
break;

case (long) 'S':  
xrotrad = (xAngle / 180 * 3.141592654f);
yrotrad = (yAngle / 180 * 3.141592654f);
mx += float(sin(yrotrad)) * factor;
mz -= float(cos(yrotrad)) * factor;  
break;

case (long)'A': 
yrotrad = (yAngle / 180 * 3.141592654f);
mx += float(cos(yrotrad)) * factor;
mz += float(sin(yrotrad)) * factor; 
break;

case (long)'D': 
yrotrad = (yAngle / 180 * 3.141592654f);
mx -= float(cos(yrotrad)) * factor;
mz -= float(sin(yrotrad)) * factor; 
break;

case (long)'Q':
xrotrad = (xAngle / 180 * 3.141592654f);
my += float(sin(xrotrad));
break;

case (long) 'E':
xrotrad = (xAngle / 180 * 3.141592654f); 
my -= float(sin(xrotrad));
break;


case WXK_UP:
ax -= factor;
break;

case WXK_DOWN:
ax += factor;
break;

case WXK_LEFT:
ay -= factor;
break;

case WXK_RIGHT:
ay += factor;
break;


case WXK_SPACE:
setPerspective( !usePerspective );
break;

case (long) 'V':
setStackedRacks( !stackedRacks );
break; 

case (long) 'G':
setAdjustMinGradient( !adjustedMinGradient );
break; 


case (long) 'T':
setView3DColumns( !view3DColumns );
break; 

case (long) 'R': 
camX = -1.1f * ( (nNodes/nRacks) - 1 ) * (groupRacks * 0.25);
camY = -3.0f;
camZ = -3.0f;
xAngle = 30.0f;
yAngle = 0.0f;
zAngle = 0.0f;
xPrevMouse = 0;
yPrevMouse = 0;
break; 

default:
event.Skip();
}
move( mx, my, mz );
spin( ax, ay );
}




GLViewerCanvas::GLViewerCanvas( wxFrame* parent, int* args ) :
wxGLCanvas( parent, wxID_ANY, args, wxDefaultPosition, wxDefaultSize, wxFULL_REPAINT_ON_RESIZE )
{
m_context = new wxGLContext( this );

camX =  0.0f;  
camY = -3.0f;  
camZ = -3.0f;  

xAngle = 30.0f;
yAngle = 0.0f;
zAngle = 0.0f;

xPrevMouse = 0;
yPrevMouse = 0;

redrawTextures = true;
verticalCores = false;

usePerspective = false;
stackedRacks = false;
view3DColumns = false;

labelTextMode = 0;
adjustedMinGradient = false;
valMin = 0;
valMax = 0;


SetBackgroundStyle( wxBG_STYLE_CUSTOM );


nRacks   = 1;
nSockets = 1;
nGroupedCores = 1;

histo  = paraverMain::myParaverMain->GetCurrentHisto();
nNodes = histo->getTrace()->totalNodes(); 
nCores = histo->getTrace()->totalCPUs() / nGroupedCores;


computerID = "";
assignRacks();
camX = -1.1f * ((nNodes/nRacks) - 1);  
camY = -3.0f;  
camZ = -3.0f;  


nodeSize = 1.0f;
nodeSep  = 1.0f;
viewType = 0;
getHistogramMinMaxValueFromType();

textureData = ( GLuint* ) malloc( nNodes * sizeof( GLuint ) );

initializedGL = false; 
redrawGeometry = true; 
VAO = std::vector< uint >( 1 );
VBO = std::vector< uint >( 3 );
glBuffers = std::vector< uint >( 3 );
}


GLViewerCanvas::~GLViewerCanvas()
{
glDeleteVertexArrays( nNodes, &VAO[0] );
glDeleteBuffers( nNodes, &VBO[0] );
glDeleteProgram( shaderProgram );

free( textureData );
delete m_context;
}


void GLViewerCanvas::resized(wxSizeEvent& evt)
{
Refresh();
}




bool DEBUG_FPS = false;
void GLViewerCanvas::Idle( )
{
if ( DEBUG_FPS )
{  
++FPS;
endTime = time( NULL ); 
if ( endTime > iniTime ) {
std::cout << "FPS = " << FPS / (endTime - iniTime) << std::endl;
iniTime = endTime;
FPS = 0;
}
}
}




void GLViewerCanvas::prepare3DViewport(int topleft_x, int topleft_y, int bottomrigth_x, int bottomrigth_y)
{

glViewport( topleft_x, topleft_y, bottomrigth_x - topleft_x, bottomrigth_y - topleft_y );




if ( redrawTextures )
{
glGenTextures( nNodes, textureData ); 

k = 0; 
for ( unsigned i = 0; i < nNodes; ++i ) 
{
glBindTexture( GL_TEXTURE_2D, textureData[ i ] );

glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );
glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

wxImage img( DrawNodeTexture( TEX_RES, i + 1 ) );
img.SetMask( ); 



GLubyte *bitmapData = img.GetData();
GLubyte *alphaData = img.GetAlpha();

int bytesPerPixel = img.HasAlpha() ?  4 : 3;

int imageSize = TEX_RES * TEX_RES * bytesPerPixel;
GLubyte *imageData = new GLubyte[ imageSize ];

for ( int y = 0; y < TEX_RES; ++y )
{
for ( int x = 0; x < TEX_RES; ++x )
{
imageData[ ( x + y*TEX_RES )*bytesPerPixel + 0 ] = bitmapData[ ( x + y*TEX_RES)*3 ];      
imageData[ ( x + y*TEX_RES )*bytesPerPixel + 1 ] = bitmapData[ ( x + y*TEX_RES)*3 + 1 ];  
imageData[ ( x + y*TEX_RES )*bytesPerPixel + 2 ] = bitmapData[ ( x + y*TEX_RES)*3 + 2 ];  

if ( bytesPerPixel == 4 ) 
imageData[ ( x + y*TEX_RES )*bytesPerPixel + 3 ] = alphaData[ x + y*TEX_RES ];        
}
}

glTexImage2D( GL_TEXTURE_2D,
0,
bytesPerPixel,
TEX_RES, TEX_RES,
0,
GL_RGBA, 
GL_UNSIGNED_BYTE,
imageData );

delete [] imageData;
}
redrawTextures = false;
}
}

static const GLfloat g_vertex_buffer_data[] = {
0.5f,  0.5f, 0.0f,  
0.5f, -0.5f, 0.0f,  
-0.5f,  0.5f, 0.0f,  
0.5f, -0.5f, 0.0f,  
-0.5f, -0.5f, 0.0f,  
-0.5f,  0.5f, 0.0f   
}; 

static const GLfloat g_texture_buffer_data[] = {
1.0f, 1.0f,
0.0f, 1.0f,
1.0f, 0.0f,
0.0f, 1.0f,
0.0f, 0.0f,
1.0f, 0.0f 
}; 

void GLViewerCanvas::render( wxPaintEvent& evt )
{
if ( !IsShown() ) return;
if ( histo != NULL )
histo->setOnlyTotals( true );

if ( redrawTextures )
{
free( textureData );
textureData = ( GLuint* ) malloc( nNodes * sizeof( GLuint ) );
TEX_RES = setTextureResolution(); 
}


if ( !initializedGL )
{ 

wxGLCanvas::SetCurrent( *m_context );
wxPaintDC( this ); 

glLoadIdentity();
GLenum err = glewInit();
if ( err != GL_NO_ERROR )
{
wxMessageBox(
wxString( "GLEW Error: " ) +
wxString( glewGetErrorString( err ) ),
_( "OpenGl ERROR" ),
wxOK | wxICON_EXCLAMATION
);
exit( 4001 );
}
else std::cout << "\t --> GL initialized successfully!" << std::endl;

initializedGL = true;
}

prepare3DViewport( 0, 0, getWidth(), getHeight() );

if ( redrawGeometry ) 
{

assignRacks(); 
map< int, std::vector< int > >::reverse_iterator it;
int i = NodesPerRack.size() - 1;
k = 0;

for ( it = NodesPerRack.rbegin() ; it != NodesPerRack.rend(); ++it )
{
int RackSize = ( (*it).second ).size();
for ( int j = RackSize-1; j >= 0; --j )
{
k = (*it).second[ j ] * ( nCores / nNodes );
DrawRack( i, j, RackSize ); 
}
--i;
}

LoadShadersGL();



glGenVertexArrays( 1, &VAO[ 0 ] );
glGenBuffers( 1, &VBO[ 0 ] );                

glBindVertexArray( VAO[ 0 ] );
glBindBuffer( GL_ARRAY_BUFFER, VBO[ 0 ] );   



glBufferData( GL_ARRAY_BUFFER, sizeof( g_vertex_buffer_data ), g_vertex_buffer_data, GL_STATIC_DRAW );  
glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0 );
glEnableVertexAttribArray( 0 );

glGenBuffers( 1, &VBO[ 1 ] );                
glBindBuffer( GL_ARRAY_BUFFER, VBO[ 1 ] );   
glBufferData( GL_ARRAY_BUFFER, sizeof( colData ), &colData[ 0 ], GL_STATIC_DRAW );  
glVertexAttribPointer( 1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0 );
glEnableVertexAttribArray( 1 );

glGenBuffers( 2, &VBO[ 2 ] );                
glBindBuffer( GL_ARRAY_BUFFER, VBO[ 2 ] );   
glBufferData( GL_ARRAY_BUFFER, sizeof( g_texture_buffer_data ), g_texture_buffer_data, GL_STATIC_DRAW );  
glVertexAttribPointer( 2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0 );
glEnableVertexAttribArray( 2 );






redrawGeometry = false;
}


rgb bgCol = ParaverConfig::getInstance()->getColorsTimelineBackground();
glClearColor( bgCol.red / 255.0f, bgCol.green / 255.0f, bgCol.blue / 255.0f, 1.0f ); 
glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

glUseProgram( shaderProgram );

model = glm::mat4( 1.0f );
view  = glm::mat4( 1.0f );
proj  = glm::mat4( 1.0f );
float ratio_w_h = getWidth() / getHeight();

model = glm::rotate( model, glm::radians( -45.0f ), glm::vec3( 1.0f, 0.0f, 0.0f ) );  
view = glm::translate( view, glm::vec3(0.0f, 0.0f, -1.0f)); 
proj = glm::perspective( glm::radians( 45.0f ), ratio_w_h, 0.1f, 200.0f ); 

int modelLoc = glGetUniformLocation( shaderProgram, "model" );
glUniformMatrix4fv( modelLoc, 1, GL_FALSE, glm::value_ptr( model ) );

int viewLoc = glGetUniformLocation( shaderProgram, "view" );
glUniformMatrix4fv( viewLoc, 1, GL_FALSE, glm::value_ptr( view ) );

int projLoc = glGetUniformLocation( shaderProgram, "proj" );
glUniformMatrix4fv( projLoc, 1, GL_FALSE, glm::value_ptr( proj ) );




for ( auto i : VAO )
{
glActiveTexture( GL_TEXTURE0 );
glBindTexture( GL_TEXTURE_2D, textureData[ 0 ] );

glBindVertexArray( VAO[ i ] ); 
glDrawArrays ( GL_TRIANGLES, 0, 6 ); 

}
SwapBuffers(); 
}





void GLViewerCanvas::LoadShadersGL()
{
int success;
char infoLog[512];

unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
glCompileShader(vertexShader);
glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
if (!success)
{
glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n\t" << infoLog << std::endl;
}
else std::cout << "\t --> [1/3] Vrtx Shader compiled successfully!" << std::endl;


unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
glCompileShader(fragmentShader);
glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
if (!success)
{
glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n\t" << infoLog << std::endl;
}
else std::cout << "\t --> [2/3] Frag Shader compiled successfully!" << std::endl;


shaderProgram = glCreateProgram();
glAttachShader( shaderProgram, vertexShader );
glAttachShader( shaderProgram, fragmentShader );
glLinkProgram( shaderProgram );

glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
if (!success) {
glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n\t" << infoLog << std::endl;
}
else std::cout << "\t --> [3/3] Shader linked successfully!" << std::endl << std::endl;

glDeleteShader( vertexShader ); glDeleteShader( fragmentShader );

}



#if 1
void GLViewerCanvas::DrawRack( int rackIdx, int pir, int rackSize ) 
{ 
vtxData = vector< double >( 0 );
colData = vector< double >( 0 );
texData = vector< double >( 0 );

texData.insert( texData.end(), { 0, 0,   0, 1,   1, 0,   1, 1 } );

colData.insert( colData.end(), { 1.0f, 0.0f, 0.0f,    1.0f, 1.0f, 0.0f,    1.0f, 0.0f, 1.0f,    1.0f, 1.0f, 1.0f } );

if ( !groupRacks )
{
int texPos = k / ( nCores / nNodes );
glBindTexture( GL_TEXTURE_2D, textureData[ texPos ] );  
if ( stackedRacks )
{
vtxData.insert( vtxData.end(), { -nodeSize,   nodeSize - pir*nodeSep, -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx } );
vtxData.insert( vtxData.end(), {  nodeSize,   nodeSize - pir*nodeSep,  nodeSize - 3.3f*nodeSep*nodeSize*rackIdx } );
vtxData.insert( vtxData.end(), { -nodeSize,   nodeSize - pir*nodeSep,  nodeSize - 3.3f*nodeSep*nodeSize*rackIdx } );
vtxData.insert( vtxData.end(), {  nodeSize,   nodeSize - pir*nodeSep, -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx } );
}
else
{
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*pir*nodeSize,    nodeSize,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*pir*nodeSize,    nodeSize,   nodeSize - 3.3f*nodeSep*nodeSize*rackIdx } );
vtxData.insert( vtxData.end(), {  nodeSize + 2.2f*nodeSep*pir*nodeSize,    nodeSize,   nodeSize - 3.3f*nodeSep*nodeSize*rackIdx } );
vtxData.insert( vtxData.end(), {  nodeSize + 2.2f*nodeSep*pir*nodeSize,    nodeSize,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx } );
}
if ( view3DColumns )
{ 
TSemanticValue val = 0;
DrawCoreLevels( rackIdx, pir, texPos );
}
}
else
{
int N, M;
string cIDaux = computerID;
computerID = "";

getCoreSquaresFromCoreCount( rackSize, N, M );
computerID = cIDaux;

int texPos = k / ( nCores / nNodes );
int i = pir / M;
int j = pir % M;

glBindTexture( GL_TEXTURE_2D, textureData[ texPos ] ); 
if ( stackedRacks )
{ 
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*i*nodeSize,   nodeSize - j*nodeSep,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*i*nodeSize,   nodeSize - j*nodeSep,   nodeSize - 3.3f*nodeSep*nodeSize*rackIdx } );
vtxData.insert( vtxData.end(), {  nodeSize + 2.2f*nodeSep*i*nodeSize,   nodeSize - j*nodeSep,   nodeSize - 3.3f*nodeSep*nodeSize*rackIdx } );
vtxData.insert( vtxData.end(), {  nodeSize + 2.2f*nodeSep*i*nodeSize,   nodeSize - j*nodeSep,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx } );
}
else 
{  
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*j*nodeSize,    nodeSize,  -nodeSize - N*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*i*nodeSize } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*j*nodeSize,    nodeSize,   nodeSize - N*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*i*nodeSize } );
vtxData.insert( vtxData.end(), {  nodeSize + 2.2f*nodeSep*j*nodeSize,    nodeSize,   nodeSize - N*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*i*nodeSize } );
vtxData.insert( vtxData.end(), {  nodeSize + 2.2f*nodeSep*j*nodeSize,    nodeSize,  -nodeSize - N*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*i*nodeSize } );
}
if ( view3DColumns )
{ 
TSemanticValue val = 0;
DrawBlockCoreLevels( rackIdx, i, j, N, texPos );
}
}
}


void GLViewerCanvas::DrawCoreLevels( int rackIdx, int nodeIdx, int texIdx )
{
TSemanticValue val;
float SC = nodeSize;
int N, M;
int nCores = histo->getTrace()->totalCPUs(); 

getCoreSquaresFromCoreCount( nCores/nNodes, N, M );

int sizeX = TEX_RES / N;
int sizeY = TEX_RES / M;

for ( int i = 0; i < N; ++i )
{
for ( int j = 0; j < M; ++j )
{
rgb colour; 
getHistogramValueFromType( k, val, colour );

float X0,X1, Y0,Y1; 
X0 = float( i * sizeX )     / float( TEX_RES );     
X1 = float( (i+1) * sizeX ) / float( TEX_RES );     
Y0 = float( j * sizeY )     / float( TEX_RES );     
Y1 = float( (j+1) * sizeY ) / float( TEX_RES );     

val = ( val - valMin ) / ( valMax - valMin );


glBindTexture( GL_TEXTURE_2D, textureData[ texIdx ] );  
texData.insert( texData.end(), { X0, Y0,  X0, Y1,  X1, Y0,  X1, Y1 } ); 
texData.insert( texData.end(), { -1, -1,  -1, -1,  -1, -1,  -1, -1 } ); 
texData.insert( texData.end(), { -1, -1,  -1, -1,  -1, -1,  -1, -1 } );
texData.insert( texData.end(), { -1, -1,  -1, -1,  -1, -1,  -1, -1 } );
texData.insert( texData.end(), { -1, -1,  -1, -1,  -1, -1,  -1, -1 } );

colData.insert( colData.end(), { 1.0f, 0.0f, 0.0f,    1.0f, 1.0f, 0.0f,    1.0f, 0.0f, 1.0f,    1.0f, 1.0f, 1.0f } );
colData.insert( colData.end(), { float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f,   float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f,   float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f,   float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f } );
colData.insert( colData.end(), { float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f,   float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f,   float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f,   float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f } );
colData.insert( colData.end(), { float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f,   float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f,   float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f,   float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f } );
colData.insert( colData.end(), { float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f,   float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f,   float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f,   float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f } );

if ( stackedRacks ) 
{
vtxData.insert( vtxData.end(), { -nodeSize + X0*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X0*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X1*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X1*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );

glBindTexture( GL_TEXTURE_2D, 0 );     

vtxData.insert( vtxData.end(), { -nodeSize + X0*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X0*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X0*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X0*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );

vtxData.insert( vtxData.end(), { -nodeSize + X1*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X1*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X1*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X1*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );

vtxData.insert( vtxData.end(), { -nodeSize + X0*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X1*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X1*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X0*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );

vtxData.insert( vtxData.end(), { -nodeSize + X0*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X0*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X1*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X1*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );

}
else 
{
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );

glBindTexture( GL_TEXTURE_2D, 0 );     
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X0*2.0f*(nodeSize),        nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X0*2.0f*(nodeSize),        nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );


vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X1*2.0f*(nodeSize),        nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X1*2.0f*(nodeSize),        nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );

vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X0*2.0f*(nodeSize),        nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X1*2.0f*(nodeSize),        nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );

vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X0*2.0f*(nodeSize),        nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X1*2.0f*(nodeSize),        nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );
}
++k;
}
}
}


void GLViewerCanvas::DrawBlockCoreLevels( int rackIdx, int nodeIdxI, int nodeIdxJ, int Nloc, int texIdx )
{
TSemanticValue val;
float SC = nodeSize;
int N, M;
int nCores = histo->getTrace()->totalCPUs(); 

getCoreSquaresFromCoreCount( nCores/nNodes, N, M );

int sizeX = TEX_RES / N;
int sizeY = TEX_RES / M;

for ( int i = 0; i < N; ++i )
{
for ( int j = 0; j < M; ++j )
{
rgb colour; 
getHistogramValueFromType( k, val, colour );

float X0,X1, Y0,Y1; 
X0 = float( i * sizeX )     / float( TEX_RES );
X1 = float( (i+1) * sizeX ) / float( TEX_RES );
Y0 = float( j * sizeY )     / float( TEX_RES );
Y1 = float( (j+1) * sizeY ) / float( TEX_RES );

val = ( val - valMin ) / ( valMax - valMin );

glBindTexture( GL_TEXTURE_2D, textureData[ texIdx ] );  
texData.insert( texData.end(), { X0, Y0,  X0, Y1,  X1, Y0,  X1, Y1 } ); 
texData.insert( texData.end(), { -1, -1,  -1, -1,  -1, -1,  -1, -1 } ); 
texData.insert( texData.end(), { -1, -1,  -1, -1,  -1, -1,  -1, -1 } );
texData.insert( texData.end(), { -1, -1,  -1, -1,  -1, -1,  -1, -1 } );
texData.insert( texData.end(), { -1, -1,  -1, -1,  -1, -1,  -1, -1 } );

colData.insert( colData.end(), { 1.0f, 0.0f, 0.0f,    1.0f, 1.0f, 0.0f,    1.0f, 0.0f, 1.0f,    1.0f, 1.0f, 1.0f } );
colData.insert( colData.end(), { float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f,   float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f,   float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f,   float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f } );
colData.insert( colData.end(), { float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f,   float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f,   float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f,   float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f } );
colData.insert( colData.end(), { float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f,   float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f,   float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f,   float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f } );
colData.insert( colData.end(), { float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f,   float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f,   float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f,   float( colour.red )/255.0f, float( colour.green )/255.0f, float( colour.blue )/255.0f } );

if ( stackedRacks ) 
{
vtxData.insert( vtxData.end(), { -nodeSize + X0*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X0*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X1*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X1*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );
glBindTexture( GL_TEXTURE_2D, 0 );     

vtxData.insert( vtxData.end(), { -nodeSize + X0*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X0*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X0*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X0*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );

vtxData.insert( vtxData.end(), { -nodeSize + X1*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X1*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X1*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X1*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );

vtxData.insert( vtxData.end(), { -nodeSize + X0*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X1*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X1*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X0*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) } );

vtxData.insert( vtxData.end(), { -nodeSize + X0*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X0*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X1*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + X1*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) } );

}
else 
{ 
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y0*2.0f*( nodeSize ) } );

glBindTexture( GL_TEXTURE_2D, 0 );     

vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X0*2.0f*(nodeSize),        nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X0*2.0f*(nodeSize),        nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y0*2.0f*( nodeSize ) } );

vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X1*2.0f*(nodeSize),        nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X1*2.0f*(nodeSize),        nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y1*2.0f*( nodeSize ) } );

vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X0*2.0f*(nodeSize),        nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X1*2.0f*(nodeSize),        nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y1*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y1*2.0f*( nodeSize ) } );

vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X0*2.0f*(nodeSize),        nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y0*2.0f*( nodeSize ) } );
vtxData.insert( vtxData.end(), { -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X1*2.0f*(nodeSize),        nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y0*2.0f*( nodeSize ) } );

}
++k;
}
}
}
#else
void GLViewerCanvas::DrawRack( int rackIdx, int pir, int rackSize ) 
{

if ( !groupRacks )
{
int texPos = k / ( nCores / nNodes );
glBindTexture( GL_TEXTURE_2D, textureData[ texPos ] ); 
glBegin( GL_QUADS );
glNormal3f( 0.0f, 1.0f, 0.0f ); 
if ( stackedRacks )
{
glTexCoord2f( 0, 0 ); glVertex3f( -nodeSize,   nodeSize - pir*nodeSep, -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx );
glTexCoord2f( 0, 1 ); glVertex3f( -nodeSize,   nodeSize - pir*nodeSep,  nodeSize - 3.3f*nodeSep*nodeSize*rackIdx );
glTexCoord2f( 1, 1 ); glVertex3f(  nodeSize,   nodeSize - pir*nodeSep,  nodeSize - 3.3f*nodeSep*nodeSize*rackIdx );
glTexCoord2f( 1, 0 ); glVertex3f(  nodeSize,   nodeSize - pir*nodeSep, -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx );
glEnd();
}
else
{
glTexCoord2f( 0, 0 ); glVertex3f( -nodeSize + 2.2f*nodeSep*pir*nodeSize,    nodeSize,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx );
glTexCoord2f( 0, 1 ); glVertex3f( -nodeSize + 2.2f*nodeSep*pir*nodeSize,    nodeSize,   nodeSize - 3.3f*nodeSep*nodeSize*rackIdx );
glTexCoord2f( 1, 1 ); glVertex3f(  nodeSize + 2.2f*nodeSep*pir*nodeSize,    nodeSize,   nodeSize - 3.3f*nodeSep*nodeSize*rackIdx );
glTexCoord2f( 1, 0 ); glVertex3f(  nodeSize + 2.2f*nodeSep*pir*nodeSize,    nodeSize,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx );
glEnd();
}

if ( view3DColumns )
{ 
TSemanticValue val = 0;
DrawCoreLevels( rackIdx, pir, texPos );
}
}
else
{
int N, M;
string cIDaux = computerID;
computerID = "";

getCoreSquaresFromCoreCount( rackSize, N, M );
computerID = cIDaux;

int texPos = k / ( nCores / nNodes );
int i = pir / M;
int j = pir % M;


glBindTexture( GL_TEXTURE_2D, textureData[ texPos ] ); 
glBegin( GL_QUADS );
glNormal3f( 0.0f, 1.0f, 0.0f );  
if ( stackedRacks )
{ 
glTexCoord2f( 0, 0 ); glVertex3f( -nodeSize + 2.2f*nodeSep*i*nodeSize,   nodeSize - j*nodeSep,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx );
glTexCoord2f( 0, 1 ); glVertex3f( -nodeSize + 2.2f*nodeSep*i*nodeSize,   nodeSize - j*nodeSep,   nodeSize - 3.3f*nodeSep*nodeSize*rackIdx );
glTexCoord2f( 1, 1 ); glVertex3f(  nodeSize + 2.2f*nodeSep*i*nodeSize,   nodeSize - j*nodeSep,   nodeSize - 3.3f*nodeSep*nodeSize*rackIdx );
glTexCoord2f( 1, 0 ); glVertex3f(  nodeSize + 2.2f*nodeSep*i*nodeSize,   nodeSize - j*nodeSep,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx );

glEnd();
}
else 
{  
glTexCoord2f( 0, 0 ); glVertex3f( -nodeSize + 2.2f*nodeSep*j*nodeSize,    nodeSize,  -nodeSize - N*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*i*nodeSize );
glTexCoord2f( 0, 1 ); glVertex3f( -nodeSize + 2.2f*nodeSep*j*nodeSize,    nodeSize,   nodeSize - N*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*i*nodeSize );
glTexCoord2f( 1, 1 ); glVertex3f(  nodeSize + 2.2f*nodeSep*j*nodeSize,    nodeSize,   nodeSize - N*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*i*nodeSize );
glTexCoord2f( 1, 0 ); glVertex3f(  nodeSize + 2.2f*nodeSep*j*nodeSize,    nodeSize,  -nodeSize - N*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*i*nodeSize );
glEnd();

}
if ( view3DColumns )
{ 
TSemanticValue val = 0;
DrawBlockCoreLevels( rackIdx, i, j, N, texPos );
} 
}
}



void GLViewerCanvas::DrawCoreLevels( int rackIdx, int nodeIdx, int texIdx )
{
TSemanticValue val;
float SC = nodeSize;
int N, M;
int nCores = histo->getTrace()->totalCPUs(); 

getCoreSquaresFromCoreCount( nCores/nNodes, N, M );

int sizeX = TEX_RES / N;
int sizeY = TEX_RES / M;

for ( int i = 0; i < N; ++i )
{
for ( int j = 0; j < M; ++j )
{
rgb colour; 
getHistogramValueFromType( k, val, colour );

float X0,X1, Y0,Y1; 
X0 = float( i * sizeX )     / float( TEX_RES );     
X1 = float( (i+1) * sizeX ) / float( TEX_RES );     
Y0 = float( j * sizeY )     / float( TEX_RES );     
Y1 = float( (j+1) * sizeY ) / float( TEX_RES );     

val = ( val - valMin ) / ( valMax - valMin );


glBindTexture( GL_TEXTURE_2D, textureData[ texIdx ] );  
if ( stackedRacks ) 
{
glBegin( GL_QUADS );
glNormal3f( 0.0f, 1.0f, 0.0f ); 
glTexCoord2f( X0, Y0 ); glVertex3f( -nodeSize + X0*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );
glTexCoord2f( X0, Y1 ); glVertex3f( -nodeSize + X0*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );
glTexCoord2f( X1, Y1 ); glVertex3f( -nodeSize + X1*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );
glTexCoord2f( X1, Y0 ); glVertex3f( -nodeSize + X1*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );
glEnd();   


glBindTexture( GL_TEXTURE_2D, 0 );     

glBegin( GL_QUADS );
glColor4ub( int( colour.red ), int( colour.green ), int( colour.blue ), 255 ); 
glNormal3f( -1.0f, 0.0f, 0.0f ); 
glVertex3f( -nodeSize + X0*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X0*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X0*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X0*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );
glEnd();

glBegin( GL_QUADS );
glColor4ub( int( colour.red ), int( colour.green ), int( colour.blue ), 255 ); 
glNormal3f( 1.0f, 0.0f, 0.0f ); 
glVertex3f( -nodeSize + X1*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X1*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X1*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X1*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );
glEnd();

glBegin( GL_QUADS );
glColor4ub( int( colour.red ), int( colour.green ), int( colour.blue ), 255 ); 
glNormal3f( 0.0f, 0.0f, 1.0f ); 
glVertex3f( -nodeSize + X0*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X1*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X1*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X0*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );
glEnd();

glBegin( GL_QUADS );
glColor4ub( int( colour.red ), int( colour.green ), int( colour.blue ), 255 ); 
glNormal3f( 0.0f, 0.0f, -1.0f ); 
glVertex3f( -nodeSize + X0*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X0*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X1*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X1*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );
glEnd();


glBegin( GL_LINES );
glLineWidth( 2.0f );

glVertex3f( -nodeSize + X0*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X0*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X1*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X1*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );

glVertex3f( -nodeSize + X0*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X0*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X1*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X1*2.0f*(nodeSize),   nodeSize - nodeIdx*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );

glEnd();
}
else 
{
glBegin( GL_QUADS );
glNormal3f( 0.0f, 1.0f, 0.0f ); 
glTexCoord2f( X0, Y0 ); glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) );
glTexCoord2f( X0, Y1 ); glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) );
glTexCoord2f( X1, Y1 ); glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) );
glTexCoord2f( X1, Y0 ); glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) );
glEnd();   

glBindTexture( GL_TEXTURE_2D, 0 );     
glBegin( GL_QUADS );
glColor4ub( int( colour.red ), int( colour.green ), int( colour.blue ), 255 ); 
glNormal3f( -1.0f, 0.0f, 0.0f ); 
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X0*2.0f*(nodeSize),        nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X0*2.0f*(nodeSize),        nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) );
glEnd();


glBegin( GL_QUADS );
glColor4ub( int( colour.red ), int( colour.green ), int( colour.blue ), 255 ); 
glNormal3f( 1.0f, 0.0f, 0.0f ); 
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X1*2.0f*(nodeSize),        nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X1*2.0f*(nodeSize),        nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) );
glEnd();

glBegin( GL_QUADS );
glColor4ub( int( colour.red ), int( colour.green ), int( colour.blue ), 255 ); 
glNormal3f( 0.0f, 0.0f, 1.0f ); 
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X0*2.0f*(nodeSize),        nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X1*2.0f*(nodeSize),        nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) );
glEnd();

glBegin( GL_QUADS );
glColor4ub( int( colour.red ), int( colour.green ), int( colour.blue ), 255 ); 
glNormal3f( 0.0f, 0.0f, -1.0f ); 
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X0*2.0f*(nodeSize),        nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X1*2.0f*(nodeSize),        nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) );
glEnd();

glBegin( GL_LINES );
glLineWidth( 2.0f );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X0*2.0f*(nodeSize),        nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X1*2.0f*(nodeSize),        nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X0*2.0f*(nodeSize),        nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X1*2.0f*(nodeSize),        nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdx*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize ) );

glEnd();
}
glColor4ub( 255, 255, 255, 255 ); 
++k;
}
}
}


void GLViewerCanvas::DrawBlockCoreLevels( int rackIdx, int nodeIdxI, int nodeIdxJ, int Nloc, int texIdx )
{
TSemanticValue val;
float SC = nodeSize;
int N, M;
int nCores = histo->getTrace()->totalCPUs(); 

getCoreSquaresFromCoreCount( nCores/nNodes, N, M );

int sizeX = TEX_RES / N;
int sizeY = TEX_RES / M;

for ( int i = 0; i < N; ++i )
{
for ( int j = 0; j < M; ++j )
{
rgb colour; 
getHistogramValueFromType( k, val, colour );


float X0,X1, Y0,Y1; 
X0 = float( i * sizeX )     / float( TEX_RES );
X1 = float( (i+1) * sizeX ) / float( TEX_RES );
Y0 = float( j * sizeY )     / float( TEX_RES );
Y1 = float( (j+1) * sizeY ) / float( TEX_RES );

val = ( val - valMin ) / ( valMax - valMin );

glBindTexture( GL_TEXTURE_2D, textureData[ texIdx ] );  
if ( stackedRacks ) 
{
glBegin( GL_QUADS );
glNormal3f( 0.0f, 1.0f, 0.0f ); 
glTexCoord2f( X0, Y0 ); glVertex3f( -nodeSize + X0*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );
glTexCoord2f( X0, Y1 ); glVertex3f( -nodeSize + X0*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );
glTexCoord2f( X1, Y1 ); glVertex3f( -nodeSize + X1*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );
glTexCoord2f( X1, Y0 ); glVertex3f( -nodeSize + X1*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );
glEnd();   


glBindTexture( GL_TEXTURE_2D, 0 );     

glBegin( GL_QUADS );
glColor4ub( int( colour.red ), int( colour.green ), int( colour.blue ), 255 ); 
glNormal3f( -1.0f, 0.0f, 0.0f ); 
glVertex3f( -nodeSize + X0*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X0*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X0*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X0*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );
glEnd();

glBegin( GL_QUADS );
glColor4ub( int( colour.red ), int( colour.green ), int( colour.blue ), 255 ); 
glNormal3f( 1.0f, 0.0f, 0.0f ); 
glVertex3f( -nodeSize + X1*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X1*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X1*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X1*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );
glEnd();

glBegin( GL_QUADS );
glColor4ub( int( colour.red ), int( colour.green ), int( colour.blue ), 255 ); 
glNormal3f( 0.0f, 0.0f, 1.0f ); 
glVertex3f( -nodeSize + X0*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X1*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X1*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X0*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );
glEnd();

glBegin( GL_QUADS );
glColor4ub( int( colour.red ), int( colour.green ), int( colour.blue ), 255 ); 
glNormal3f( 0.0f, 0.0f, -1.0f ); 
glVertex3f( -nodeSize + X0*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X0*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X1*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X1*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );
glEnd();


glBegin( GL_LINES );
glLineWidth( 2.0f );

glVertex3f( -nodeSize + X0*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X0*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X1*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X1*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y0*2.0f*( nodeSize )  );

glVertex3f( -nodeSize + X0*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X0*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X1*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep,        -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );
glVertex3f( -nodeSize + X1*2.0f*(nodeSize) + 2.2f*nodeSep*nodeIdxI*nodeSize,   nodeSize - nodeIdxJ*nodeSep + val,  -nodeSize - 3.3f*nodeSep*nodeSize*rackIdx + Y1*2.0f*( nodeSize )  );

glEnd();
}
else 
{
glBegin( GL_QUADS );
glNormal3f( 0.0f, 1.0f, 0.0f ); 
glTexCoord2f( X0, Y0 ); glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y0*2.0f*( nodeSize ) );
glTexCoord2f( X0, Y1 ); glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y1*2.0f*( nodeSize ) );
glTexCoord2f( X1, Y1 ); glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y1*2.0f*( nodeSize ) );
glTexCoord2f( X1, Y0 ); glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y0*2.0f*( nodeSize ) );
glEnd();   

glBindTexture( GL_TEXTURE_2D, 0 );     
glBegin( GL_QUADS );
glColor4ub( int( colour.red ), int( colour.green ), int( colour.blue ), 255 ); 
glNormal3f( -1.0f, 0.0f, 0.0f ); 
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X0*2.0f*(nodeSize),        nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y0*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X0*2.0f*(nodeSize),        nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y1*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y1*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y0*2.0f*( nodeSize ) );
glEnd();


glBegin( GL_QUADS );
glColor4ub( int( colour.red ), int( colour.green ), int( colour.blue ), 255 ); 
glNormal3f( 1.0f, 0.0f, 0.0f ); 
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X1*2.0f*(nodeSize),        nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y0*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y0*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y1*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X1*2.0f*(nodeSize),        nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y1*2.0f*( nodeSize ) );
glEnd();

glBegin( GL_QUADS );
glColor4ub( int( colour.red ), int( colour.green ), int( colour.blue ), 255 ); 
glNormal3f( 0.0f, 0.0f, 1.0f ); 
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X0*2.0f*(nodeSize),        nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y1*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X1*2.0f*(nodeSize),        nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y1*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y1*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y1*2.0f*( nodeSize ) );
glEnd();

glBegin( GL_QUADS );
glColor4ub( int( colour.red ), int( colour.green ), int( colour.blue ), 255 ); 
glNormal3f( 0.0f, 0.0f, -1.0f ); 
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X0*2.0f*(nodeSize),        nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y0*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y0*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y0*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X1*2.0f*(nodeSize),        nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y0*2.0f*( nodeSize ) );
glEnd();

glBegin( GL_LINES );
glLineWidth( 2.0f );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X0*2.0f*(nodeSize),        nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y0*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y0*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X1*2.0f*(nodeSize),        nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y0*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y0*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X0*2.0f*(nodeSize),        nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y1*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X0*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y1*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X1*2.0f*(nodeSize),        nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y1*2.0f*( nodeSize ) );
glVertex3f( -nodeSize + 2.2f*nodeSep*nodeIdxJ*nodeSize + X1*2.0f*(nodeSize),    val+nodeSize,   -nodeSize - Nloc*3.3f*nodeSep*nodeSize*rackIdx + 2.2f*nodeSep*nodeIdxI*nodeSize + Y1*2.0f*( nodeSize ) );
glEnd();
}
glColor4ub( 255, 255, 255, 255 ); 
++k;
}
}
}
#endif







int GLViewerCanvas::getWidth()
{
return GetSize().x;
}

int GLViewerCanvas::getHeight()
{
return GetSize().y;
}

int GLViewerCanvas::setTextureResolution()
{ 
float coreToNode = float( nCores ) / float( nNodes );
if ( coreToNode != 7 && coreToNode <= 16 ) return 256;
else if ( coreToNode <= 49 ) return 512;

return 512;
}


void GLViewerCanvas::assignRacks()
{
bool FAILSAFE = false;
if ( FAILSAFE )
{
std::vector<int> v( nNodes ); 
int idx = -1;
std::generate( v.begin(), v.end(), [ &idx ] { return idx+=1; } ); 

NodesPerRack[ 0 ] = v;
nRacks = 1;
return;
}


k = 0;  

std::vector< std::string > NodeSet; 
for ( TThreadOrder i = 0; i < nCores; i += (nCores/nNodes) )
{ 
TNodeOrder NID = histo->getTrace()->getNodeFromThread( i );
NodeSet.push_back( histo->getTrace()->getRowLabel( NODE, NID ) );
}

wxRegEx reMN4( "^s[0-9]+r[0-9]+b[0-9]+" );
int idx = 0;

NodesPerRack.clear();
for ( std::vector< std::string >::iterator it = NodeSet.begin() ; it != NodeSet.end(); ++it )
{
wxString nodeName( ( ( *it ) ).c_str(), wxConvUTF8 );

long switchID, rackID, nodeID;
if ( reMN4.Matches( nodeName ) )
{
computerID = "MN4";
nodeName.AfterFirst( 's' ).BeforeFirst( 'r' ).ToLong( &switchID );
nodeName.AfterFirst( 'r' ).BeforeFirst( 'b' ).ToLong( &rackID );
nodeName.AfterFirst( 'b' ).ToLong( &nodeID );
}
else  
{
std::vector<int> v( nNodes ); 
idx = -1;
std::generate( v.begin(), v.end(), [ &idx ] { return idx+=1; } ); 

NodesPerRack[ 0 ] = v;
nRacks = 1;
return;
}

NodesPerRack[ (int) rackID ].push_back( idx );
++idx;
}

nRacks = NodesPerRack.size();
} 


void GLViewerCanvas::move( float xMove, float yMove, float zMove )
{
camX += xMove;
camY += yMove;
camZ += zMove;


Refresh( false );
}

void GLViewerCanvas::spin( float xSpin, float ySpin )
{
xAngle += xSpin;
yAngle += ySpin;


Refresh( false );
}


void GLViewerCanvas::getCoreSquaresFromCoreCount( const int &numCores, int& N, int& M )
{
if ( numCores == 48 && computerID == "MN4" ) 
{
N = 4;
M = 12;
return;
}
for (int i = floor( sqrt( numCores ) ); i <= numCores; i++) {
if ( numCores % i == 0 ) 
{
N = numCores / i;
M = i; 
if (N > M) 
{
N = i;
M = numCores / i;
}
return;
}
}
N = 64;
M = 64;
}

wxImage GLViewerCanvas::DrawNodeTexture( int size, unsigned num )
{
wxASSERT_MSG( num >= 1, "invalid index" );

int nSockets = 1; 

wxBitmap bmp( size, size*nSockets, 32 );

wxMemoryDC dc;
dc.SelectObject( bmp );
dc.SetBackgroundMode( wxTRANSPARENT );
dc.SetBackground( *wxTRANSPARENT_BRUSH );
dc.Clear();
dc.SetBrush( *wxBLACK_BRUSH );

wxColour cpuColour = wxColour( 0, 0, 0, 255 );

if ( histo != NULL ) 
{
histo->setOnlyTotals( true ); 

int N, M;
getCoreSquaresFromCoreCount( nCores/nNodes, N, M );

int Nx = N;
int Mx = M;
if ( verticalCores )
{
Nx = M;
Mx = N;
}
int sizeX = TEX_RES / Nx;
int sizeY = TEX_RES / Mx;


for ( int i = 0; i < Nx; ++i )
{
for ( int j = 0; j < Mx; ++j )
{
TSemanticValue val = 0;
if ( k >= nCores*nGroupedCores )
cpuColour = wxColour( 255, 255, 255, 0 );
else
{
rgb valCol;
if ( nGroupedCores > 1 )
{
TSemanticValue valAux;
rgb valColAux;
for ( int ii = 0; ii < nGroupedCores; ++ii )
{
getHistogramValueFromType( k+ii, valAux, valColAux );
val += valAux;
valCol.red   += valColAux.red;
valCol.green += valColAux.green;
valCol.blue  += valColAux.blue;
}
val = val / nGroupedCores;
valCol.red   = valColAux.red / nGroupedCores;
valCol.green = valColAux.green / nGroupedCores;
valCol.blue  = valColAux.blue / nGroupedCores;
}
else
{
getHistogramValueFromType( k, val, valCol );

}


if ( isnan( val ) ) val = 0;
int alpha = 255*( val > 0 ) + 5*( val <= 0 );
cpuColour = wxColour( valCol.red, valCol.green, valCol.blue, alpha ); 
}

dc.SetBrush( wxBrush( cpuColour ) );
dc.DrawRectangle( 0 + i*sizeX, 0 + j*sizeY, sizeX, sizeY );

const wxColour *textCol = paraverMain::myParaverMain->GetSelectedHistogram()->GetTableBase()->getTextColorFromLuminance( cpuColour );
dc.SetTextForeground( *textCol );

if ( k < nCores ) 
{
wxString textToWrite, maxValText;
wxCoord width, height;
if ( labelTextMode == 0 ) 
{
if ( nSockets > 1 )
textToWrite = wxString::Format( wxT( "%is" ), k+1 / nGroupedCores ); 
else
textToWrite = wxString::Format( wxT( "%i" ), k+1 / nGroupedCores );

dc.GetTextExtent( textToWrite, &width, &height );
wxFont fnt    = dc.GetFont();
wxFont fntNew = fnt.Scale( std::min( float( sizeX - 10.0 ) / float( width ), float( sizeY - 10.0 ) / float( height ) ) );
dc.SetFont( fntNew ); 
dc.DrawText( textToWrite,  10 + i*sizeX,  10 + j*sizeY );
dc.SetFont( fnt );
} 
else if ( labelTextMode == 1 ) 
{ 
if ( nSockets > 1 )
textToWrite = wxString::Format( wxT( "%.2fs" ), val ); 
else
textToWrite = wxString::Format( wxT( "%.2f" ), val );

maxValText = wxString::Format( wxT( "%.2f" ), valMax );

dc.GetTextExtent( maxValText, &width, &height );
wxFont fnt    = dc.GetFont();
wxFont fntNew = fnt.Scale( std::min( float( sizeX - 20.0 ) / float( width ), float( sizeY - 5.0 ) / float( height ) ) );
dc.SetFont( fntNew );
dc.DrawText( textToWrite,  10 + i*sizeX,  10 + j*sizeY );
dc.SetFont( fnt );
} 
}

if ( nGroupedCores == 1 )
++k;
else
k += nGroupedCores;


}
}
}
dc.SelectObject( wxNullBitmap );
return bmp.ConvertToImage();
}



void GLViewerCanvas::getHistogramValueFromType( int k, TSemanticValue &val, rgb &valCol )
{
PRV_UINT16 currentStat;
histo->getIdStat( histo->getCurrentStat(), currentStat );

switch ( viewType )
{ 
case 0: 
val = histo->getRowTotals()->getTotal( currentStat, k );
valCol = gc.calcColor( val, valMin, valMax );
break;

case 1: 
val = histo->getRowTotals()->getAverage( currentStat, k );
valCol = gc.calcColor( val, valMin, valMax );
break;

case 2: 
val = histo->getRowTotals()->getMaximum( currentStat, k );
valCol = gc.calcColor( val, valMin, valMax );
break;

case 3: 
val = histo->getRowTotals()->getMinimum( currentStat, k );
valCol = gc.calcColor( val, valMin, valMax );
break;

case 4: 
val = histo->getRowTotals()->getStdev( currentStat, k ); 
valCol = gc.calcColor( val, valMin, valMax );
break;

case 5: 
val = histo->getRowTotals()->getAvgDivMax( currentStat, k );
valCol = gc.calcColor( val, valMin, valMax ); 
break;

default: 
val = histo->getRowTotals()->getAvgDivMax( currentStat, k );
valCol = gc.calcColor( val, valMin, valMax );
break;
}
}



void GLViewerCanvas::getHistogramMinMaxValueFromType(  )
{
std::pair< TSemanticValue, TSemanticValue > vals;

PRV_UINT16 currentStat;
histo->getIdStat( histo->getCurrentStat(), currentStat );

switch ( viewType )
{ 
case 0: 
valMin = histo->getRowTotals()->getMinTotal( currentStat );
valMax = histo->getRowTotals()->getMaxTotal( currentStat );
break;

case 1: 
valMin = histo->getRowTotals()->getMinAverage( currentStat );
valMax = histo->getRowTotals()->getMaxAverage( currentStat ); 
break;

case 2: 
valMin = histo->getRowTotals()->getMinMaximum( currentStat );
valMax = histo->getRowTotals()->getMaxMaximum( currentStat ); 
break;

case 3: 
valMin = histo->getRowTotals()->getMinMinimum( currentStat );
valMax = histo->getRowTotals()->getMaxMinimum( currentStat ); 
break;

case 4: 
valMin = histo->getRowTotals()->getMinStdev( currentStat );
valMax = histo->getRowTotals()->getMaxStdev( currentStat );
break;

case 5: 
vals = histo->getRowTotals()->getMinMaxAvgDivMax( currentStat );
valMin = vals.first;
valMax = vals.second;
break;

default: 
valMin = histo->getRowTotals()->getMinTotal( currentStat );
valMax = histo->getRowTotals()->getMaxTotal( currentStat );
break;
}
}



int GLViewerCanvas::getRacks(  )
{ 
return nRacks;
}

void GLViewerCanvas::setRacks( int numRacks )
{
if ( numRacks == -1 )
numRacks = 1536; 

nRacks   = numRacks;
redrawTextures = true;
redrawGeometry = true;
Refresh( false );
}

int GLViewerCanvas::getNodes(  )
{ 
return nNodes;
}

void GLViewerCanvas::setNodes( int numNodes )
{
nNodes = numNodes;
redrawTextures = true;
redrawGeometry = true;
Refresh( false );
}

int GLViewerCanvas::getCores(  )
{ 
return nCores;
}

void GLViewerCanvas::setCores( int numCores )
{ 
nCores = numCores / nGroupedCores;
redrawTextures = true;
redrawGeometry = true;
Refresh( false );
}


int GLViewerCanvas::getSockets(  )
{ 
return nSockets;
}


void GLViewerCanvas::setSockets( int numSockets )
{ 
nSockets = numSockets;
redrawTextures = true;
Refresh( false );
}

void GLViewerCanvas::setPerspective( bool isPerspective )
{ 
usePerspective = isPerspective;
redrawGeometry = true;
Refresh( false );
}

void GLViewerCanvas::setStackedRacks( bool areRacksStacked )
{
stackedRacks = areRacksStacked;
redrawGeometry = true;
Refresh( false );
}

void GLViewerCanvas::setView3DColumns( bool viewing3Dcols )
{
view3DColumns = viewing3Dcols;
redrawGeometry = true;
Refresh( false );
}


int GLViewerCanvas::getNodeSize(  )
{ 
return nNodes;
}

void GLViewerCanvas::setNodeSize( int newNodeSize )
{
nodeSize = (float) newNodeSize / 100.0f;
redrawGeometry = true;
Refresh( false );
}


int GLViewerCanvas::getNodeSep(  )
{ 
return nNodes;
}

void GLViewerCanvas::setNodeSep( int newNodeSep )
{
nodeSep = (float) newNodeSep / 100.0f;
redrawGeometry = true;
Refresh( false );
}


void GLViewerCanvas::setViewType( int newViewType )
{
viewType = newViewType;

getHistogramMinMaxValueFromType();
if ( !adjustedMinGradient ) 
valMin = 0;

redrawTextures = true;
redrawGeometry = true;
Refresh( false );
}


void GLViewerCanvas::setAdjustMinGradient( int adjMinGrad )
{
adjustedMinGradient = adjMinGrad; 

getHistogramMinMaxValueFromType();
if ( !adjustedMinGradient ) 
valMin = 0;

redrawTextures = true;
redrawGeometry = true;
Refresh( false );
}



void GLViewerCanvas::setLabelTextMode( int newLabel )
{
labelTextMode = newLabel; 
redrawTextures = true;
redrawGeometry = true;
Refresh( false );
}

int GLViewerCanvas::getLabelTextMode()
{
return labelTextMode;
}



void GLViewerCanvas::setGroupedCores( int avgNodes )
{
nCores = nCores * nGroupedCores; 
nGroupedCores = avgNodes; 
nCores = nCores / nGroupedCores; 
Refresh( false );
}

int GLViewerCanvas::getGroupedCores()
{
return nGroupedCores;
}



void GLViewerCanvas::setGroupRacks( bool groupedRacks )
{
groupRacks = groupedRacks; 
redrawGeometry = true;
Refresh( false );
}


void GLViewerCanvas::setHistogramToView( Histogram* myHisto )
{
histo = myHisto;
paraverMain::myParaverMain->selectTrace( histo->getTrace() );

nNodes = histo->getTrace()->totalNodes(); 
nCores = histo->getTrace()->totalCPUs() / nGroupedCores;

getHistogramMinMaxValueFromType();
if ( !adjustedMinGradient ) 
valMin = 0;

redrawTextures = true;
redrawGeometry = true;
Refresh( false );
}

Histogram* GLViewerCanvas::getHistogramToView()
{
return histo;
}
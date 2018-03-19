#pragma once

#if defined(__APPLE__)
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined(_WIN32)
#    include <GL/wglew.h>
#    include <GL/freeglut.h>
#  endif
#  include <GL/glut.h>
#endif

#include <OptiXMesh.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sutil.h>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <list>
#include <map>

using namespace optix;

// Optix context
extern Context context;

// Window size
extern const int width, height;

// List of compiled cuda files
extern std::map<std::string, const char*> cudaFiles;

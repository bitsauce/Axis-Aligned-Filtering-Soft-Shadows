#include "util.h"

#include <sutil/sutil.h>
#include <sutil/HDRLoader.h>
#include <sutil/PPMLoader.h>
#include <sampleConfig.h>

#include <nvrtc.h>

#include <cstring>
#include <iostream>
#include <fstream>
#include <stdint.h>
#include <sstream>
#include <map>
#include <memory>

using namespace optix;

#define STRINGIFY(x) STRINGIFY2(x)
#define STRINGIFY2(x) #x
#define LINE_STR STRINGIFY(__LINE__)

// Error check/report helper for users of the C API
#define NVRTC_CHECK_ERROR( func )                                  \
  do {                                                             \
    nvrtcResult code = func;                                       \
    if( code != NVRTC_SUCCESS )                                    \
      throw Exception( "ERROR: " __FILE__ "(" LINE_STR "): " +     \
          std::string( nvrtcGetErrorString( code ) ) );            \
  } while( 0 )

static bool readSourceFile(std::string &str, const std::string &filename)
{
	// Try to open file
	std::ifstream file(filename.c_str());
	if(file.good())
	{
		// Found usable source file
		std::stringstream source_buffer;
		source_buffer << file.rdbuf();
		str = source_buffer.str();
		return true;
	}
	return false;
}

static std::string g_nvrtcLog;

static void getPtxFromCuString(std::string &ptx, const char* cu_source, const char* name, const char** log_string)
{
	// Create program
	nvrtcProgram prog = 0;
	NVRTC_CHECK_ERROR(nvrtcCreateProgram(&prog, cu_source, name, 0, NULL, NULL));

	// Gather NVRTC options
	std::vector<const char *> options;
	std::vector<std::string> include_dirs;

	// Collect include dirs
	const char *abs_dirs[] = { SAMPLES_ABSOLUTE_INCLUDE_DIRS };

	const size_t n_abs_dirs = sizeof(abs_dirs) / sizeof(abs_dirs[0]);
	for(size_t i = 0; i < n_abs_dirs; i++) {
		include_dirs.push_back(std::string("-I") + abs_dirs[i]);
		options.push_back(include_dirs.back().c_str());
	}

	// Collect NVRTC options
	const char *compiler_options[] = { CUDA_NVRTC_OPTIONS };
	const size_t n_compiler_options = sizeof(compiler_options) / sizeof(compiler_options[0]);
	for(size_t i = 0; i < n_compiler_options - 1; i++)
		options.push_back(compiler_options[i]);

	// JIT compile CU to PTX
	const nvrtcResult compileRes = nvrtcCompileProgram(prog, (int)options.size(), options.data());

	// Retrieve log output
	size_t log_size = 0;
	NVRTC_CHECK_ERROR(nvrtcGetProgramLogSize(prog, &log_size));
	g_nvrtcLog.resize(log_size);
	if(log_size > 1)
	{
		NVRTC_CHECK_ERROR(nvrtcGetProgramLog(prog, &g_nvrtcLog[0]));
		if(log_string)
			*log_string = g_nvrtcLog.c_str();
	}
	if(compileRes != NVRTC_SUCCESS)
		throw Exception("NVRTC Compilation failed.\n" + g_nvrtcLog);

	// Retrieve PTX code
	size_t ptx_size = 0;
	NVRTC_CHECK_ERROR(nvrtcGetPTXSize(prog, &ptx_size));
	ptx.resize(ptx_size);
	NVRTC_CHECK_ERROR(nvrtcGetPTX(prog, &ptx[0]));

	// Cleanup
	NVRTC_CHECK_ERROR(nvrtcDestroyProgram(&prog));
}


const char* loadCudaFile(
	const char* filename,
	const char** log)
{
	if(log) {
		*log = NULL;
	}

	std::string *ptx, cu;
	ptx = new std::string();
	readSourceFile(cu, filename);
	getPtxFromCuString(*ptx, cu.c_str(), filename, log);

	return ptx->c_str();
}
#pragma once

#include <string>
#include <stdio.h>
#include <time.h>


const char* loadCudaFile(
	const char* filename,
	const char** log = nullptr);

std::string getTimeStamp();
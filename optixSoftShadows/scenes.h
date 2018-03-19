#pragma once

#include "common.h"
#include "structs.h"

class Scene
{
public:
	virtual void update() = 0;

	bool animate = true;
};

class DefaultScene : public Scene
{
public:
	DefaultScene();
	void update();

private:
	ParallelogramLight light;
	Buffer lightBuffer;
};

class GridScene : public Scene
{
public:
	GridScene();
	void update();

private:
	ParallelogramLight light;
	Buffer lightBuffer;
};

OutDir = ./Debug/
CXX = g++
DEBUG = -g
CXXFLAGS = $(DEBUG) -O -O0 -O1 -O2 -O3 -std=c++11 -Wall
IncludePath = -I. -I/usr/include/CL/
LibPaths = -L. -L/usr/lib/
Libs = -lOpenCL
LFLAGS = $(LibPaths) $(Libs)

cl_conv: Directories main.cpp.o
	$(CXX) -o $(OutDir)cl_conv $(OutDir)main.cpp.o $(LFLAGS)

main.cpp.o: main.cpp
	$(CXX) -c "main.cpp" $(CXXFLAGS) -o $(OutDir)main.cpp.o $(IncludePath)

Directories:
	mkdir -p $(OutDir)

##
## Clean
##
clean:
	$(RM) -r $(OutDir)
# Host smoke test for the generated quad MPC (Eigen-free via eigen_shim/).
#   make            -> ./mpc_smoke
#   make run        -> build + run
CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -I. -Ieigen_shim

mpc_smoke: mpc_smoke.cpp quad_mpc.h generated/*.h eigen_shim/Eigen/Core
	$(CXX) $(CXXFLAGS) $< -o $@

run: mpc_smoke
	./mpc_smoke

clean:
	rm -f mpc_smoke

.PHONY: run clean

NVCC = nvcc
CFLAG = -O3 -g -c
LFLAG = -O3
LIB = -lcudart -lpthread 
all : main

main: main.o cpu_interface.o data_interface.o frontier.o frontier_node.o frontier_preexpand.o global.o gpu_interface.o job_manager.o candidate_collection.o mem_controller.o 
	$(NVCC) $(LFLAG) $(INC) $(LIB) -o dist/frontier_expansion main.o cpu_interface.o data_interface.o frontier.o frontier_node.o frontier_preexpand.o global.o gpu_interface.o job_manager.o mem_controller.o candidate_collection.o

gpu_interface.o : gpu_interface.cu
	$(NVCC) $(CFLAG) $(INC) $(LIB) -arch=sm_13 -o gpu_interface.o gpu_interface.cu

cpu_interface.o : cpu_interface.cpp cpu_interface.h
	$(NVCC) $(CFLAG) $(INC) $(LIB) -o cpu_interface.o cpu_interface.cpp

main.o : main.cpp
	$(NVCC) $(CFLAG) $(INC) $(LIB) -o main.o main.cpp

data_interface.o : data_interface.cpp data_interface.h
	$(NVCC) $(CFLAG) $(INC) $(LIB) -o data_interface.o data_interface.cpp

frontier.o : frontier.cpp frontier.h
	$(NVCC) $(CFLAG) $(INC) $(LIB) -o frontier.o frontier.cpp

frontier_node.o : frontier_node.cpp frontier_node.h
	$(NVCC) $(CFLAG) $(INC) $(LIB) -o frontier_node.o frontier_node.cpp

frontier_preexpand.o : frontier_preexpand.cpp frontier_preexpand.h
	$(NVCC) $(CFLAG) $(INC) $(LIB) -o frontier_preexpand.o frontier_preexpand.cpp

global.o : global.cpp global.h
	$(NVCC) $(CFLAG) $(INC) $(LIB) -o global.o global.cpp

job_manager.o : job_manager.cpp job_manager.h
	$(NVCC) $(CFLAG) $(INC) $(LIB) -o job_manager.o job_manager.cpp

mem_controller.o : mem_controller.cpp mem_controller.h
	$(NVCC) $(CFLAG) $(INC) $(LIB) -o mem_controller.o mem_controller.cpp

time_analysis.o : time_analysis.cpp time_analysis.h
	$(NVCC) $(CFLAG) $(INC) $(LIB) -o time_analysis.o time_analysis.cpp

candidate_collection.o : candidate_collection.cpp candidate_collection.h
	$(NVCC) $(CFLAG) $(INC) $(LIB) -o candidate_collection.o candidate_collection.cpp

test : test.o test_case.o cpu_interface.o data_interface.o frontier.o frontier_node.o frontier_preexpand.o global.o gpu_interface.o job_manager.o candidate_collection.o mem_controller.o 
	$(NVCC) $(LFLAG) $(INC) $(LIB) -o dist/test test.o test_case.o cpu_interface.o data_interface.o frontier.o frontier_node.o frontier_preexpand.o global.o gpu_interface.o job_manager.o mem_controller.o candidate_collection.o

test.o : test.cpp
	$(NVCC) $(CFLAG) $(INC) $(LIB) -o test.o test.cpp	
test_case.o : test_case.cpp test_case.h
	$(NVCC) $(CFLAG) $(INC) $(LIB) -o test_case.o test_case.cpp

clean :
	rm *.o *~ dist/*
 

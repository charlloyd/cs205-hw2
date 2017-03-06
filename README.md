# cs205-hw2
##4.  Parallel Graph Algorithms in Semi-ring [50%]

Note: in this task you do not explicitly code parallel programs but use the directives of OpenACC. You only need to implement the logic of the algorithm in ’C’ and understand the inherent parallelism to guide the OpenACC compiler to generate the parallel programs with work-sharing semantics. You can use the knowledge of using directives from the Matrix Multiplication implementation in OpenACC above to implement the parallel graph algorithms.
					
Graph models are fundamental to many computational and data science applications and scalable parallel graph algorithms will impact on emerging exascale computing applications.
					
The purpose of this task is three-fold:					
Develop practical skills in graph algorithms that are fundamental to many computational and data science applications.
Learn to apply semi-implict programming techniques (OpenAccc) to rapidly transform data parallel applications. 
Gain insights into algebraic formulations of parallel graph algorithms from a practical perspective.
In this problem you will implement parallel algorithms for two fundamental graph problems. 
					
###Semi-ring Parallel BFS
Implement a parallel BFS algorithm in the boolean semi-ring (Lecture 11 slide 18) using OpenAcc.
This requires a sparse Matrix Vector computation. Note that multiplication is logical AND and addition is logical OR operation in the semi-ring model.
The sparsity can be used to optimize the computation (xt having zero elements can be skipped).
First test your implementation works correctly on simple graphs (lecture slide). Then benchmark your implementation with RMAT graphs from Graph500. 
		 	 	 			
###Semi-ring Parallel All Pair Shortest Path (APSP)					
First, parallelise the Floyd-Warshall algorithm given in Lecture 11, slide 23 using pragma in OpenAcc.	
Next implement the tropical semi-ring version of APSP as follows:

- Note that multiplication is ADD and addition is min operation in this semi-ring model.
- The set of equations in the semi-ring formulation (Lecture 11, slide 26) for APSP can be implemented as a recursive divide-and-conquer algorithm by the following psuedo-code[solomonik, ipdps, 2013]:		–  Translate the semi-ring pseudo-code into a sequential ’C’ program and test it with simple examples.	
- Then implement a parallel APSP algorithm by annotating the sequential program using OpenAcc parallelisation pragmas as follows: 
  * Use directives to implement the recursive calls.
  * Implement equations 4,5 in parallel, similarly 8 and 9 keeping the computations within each equation sequential in the first instance.
  * Consider further optimizations such as tiling to use shared memory in GPU.
First test that your implementation works correctly on simple graphs. Then benchmark the two versions of the implementations with RMAT graphs from Graph500. 	 						
						
###Submission: A write-up covering the following aspects uploaded as a single PDF file (per group)
 A short one page description of the different design principles that were used to develop successive versions of the parallel program.					
The optimizations used and the results obtained with reference to perf ormance plots. 
Code listing and test cases.
Performance plots of throughput (GFlop/s). The plot should indicate the peak throughput achievable on the GPU node. 

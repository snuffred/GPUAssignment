# VU ACCE Course (XM_0171) - GPU Group Assignment

## Assignment Description

The problem for this assignment is grid-based, iterative simulation of rainwater flood over a 30x30 KM terrain.
It features a cloud front moving across a region of land characterized by a heightmap.
The input parameters include the grid size (finer grid size will increase the resolution and the level of details of the simulation), ground configuration (one of four presets), termination conditions and cloud configuration.

The algorithm works as follows:

1. Move all clouds across the grid based on their direction and velocity and add their rain to the water level.
2. The water is redistributed among neighbors. Given a cell we:
   1. Compute the potential water flow to adjacent cells in the grid.
   2. Distribute the water across the adjacent cells, proportionally to the to the relative differences of their total height (ground + water level).
3. Update the water array with the computed spillage.

The algorithms repeats steps 1-3 until one of the following conditions is satisfied:
- None of the grid cells have a water flow exceeding a threshold.
- A maximum number of iterations (minutes) is reached.

Step 2 lets the water flow the from higher regions to lower ones, gradually reducing height differences and tending towards equalizing levels. Out-of-bounds cells are treated as dry cells with the same height as their in-bounds neighbor.

**Note:** Further details can be found in the Assignment description provided as PDF.

We want you to implement a parallel implementation of this algorithm with CUDA. We suggest you not change this algorithm in your parallel code. There might be floating point errors if you parallelize the algorithm in different ways, but the workflow should be (mathematically) the same.


## Structure
The structure of this project template is as follows:

The `flood.c` contains the main and I/O components (e.g., reading the input file, writing the results) of the program. In principle, this file does not need to be modified.
The `flood_seq.c` file is the sequential reference implementation, and should be used as the ground truth for correctness.

Some initialization logic is implemented in `flood.c`. While the default implementation is sufficient for completing the assignment, you are allowed to modify the data structures used by your custom implementation and make the corresponding changes in this file if needed to support optimizations.

You need to implement your simulation function in `do_compute` with CUDA in the file `flood_cuda.cu`.
In the template, the function `do_compute`, which actually performs the algorithm, is just a copy of the CPU sequential version.

### Input
The input consists of a series of command line arguments

- `<rows>`: Number of rows in the cell array.
- `<columns>`: Number of columns in the cell array.
- `<ground scenario (M|V|D|d)>`: The heights of the terrain cells are stored in an array.
  The program includes four predefined terrain scenarios, which are selected using a single-character code:
  - `M`: Mountain lakes.
  - `V`: Valley.
  - `D`: Valley with a dam at slightly higher elevations.
  - `d`: Valley with a dam at lower elevations.
- `<threshold>` The simulation stops when the highest amount of water discharged from one cell to another falls below this threshold.
- `<num minutes>` The simulation stops after this number of minutes has been simulated.
- `<exaggeration factor>` Multiplier for the rainfall discharge intensity.
  It allows for a less realistic but faster simulation.
  For example, a value of 60 means that in one minute, the amount of rainwater corresponding to one hour is discharged.
- `<front distance>` Distance between the center of the simulation domain and the center of the cloud front.
- `<front width>` Width of the cloud front.
- `<front depth>` Depth of the cloud front.
  These two parameters (width and depth) define the dimensions of the rectangular region within which random clouds are generated.
  This region is rotated and translated based on the other parameters in order to correctly position the front, so that its direction leads the clouds into the domain.
- `<front direction (degrees)>` Direction of the cloud front in degrees.
- `<num random clouds>` Number of clouds generated in the front.
- `<cloud max radius (km)>` Maximum radius of the clouds.
  For each cloud, a radius is generated randomly between this value and its half.
- `<cloud max intensity (mm/h)>` Maximum rainfall intensity.
  For each cloud, an intensity is generated randomly between this value and its half.
  Rainfall intensity is considered:
  - `Normal`: up to 15 mm/h,
  - `Heavy`: 15–30 mm/h,
  - `Very Heavy`: 30–60 mm/h,
  - `Torrential`: above 60 mm/h.  Torrential rainfall exceeding 120–140 mm/h is rare but realistic.
- `<cloud max speed (km/h)>` Maximum speed.
  Each cloud is assigned a speed randomly between this value and its half.
- `<cloud max angle aperture (degrees)>` Maximum aperture angle.
  Each cloud’s movement direction is randomly selected within the range defined by the front direction plus or minus half of this angle.
- `<clouds rnd seed>` Random seed used to reproduce the cloud generation in the front.

Additional clouds can be added by providing additional optional arguments (6 per cloud).
You can write your own script to generate arbitrary cloud formations and use these arguments to test them within your simulation.

- `<cloud start x (km)>` Initial x-coordinate of the cloud centre.
- `<cloud start y (km)>` Initial y-coordinate of the cloud centre.
- `<cloud radius (km)>` Radius of the cloud.
- `<cloud intensity (mm/h)>` Rainfall intensity.
- `<cloud speed (km/h)>` Cloud speed.
- `<cloud angle (degrees)>` Direction of cloud movement, in degrees.

These arguments enable the creation of custom scenarios with different spatial and temporal dynamics, as well as varying computational demands in the different phases of the simulation. 

For your convenience, a few test sequences are provided within the `test_files` folder, which can be run using 
```bash
./flood_seq $(< test_files/debug.in)
```

These includes:
- `debug.in` a low resolution scenario useful for debugging.
- `small_mountains.in` a low resolution mountain scenario.
- `custom_clouds.in` a low resolution dam scenario with custom clouds.
- `medium_lower_dam.in` and `medium_higher_dam.in` two medium resolution dam scenarios.
- `large_mountains.in` a higher resolution mountain scenario (expected sequential running time on DAS5 is approximately 3 minutes).

### Output

At the end of execution, the program prints the execution time (in seconds, excluding the initialization phase) and a set of aggregated statistics that helps verify the correctness of the results. For instance:
```bash
Time: 0.069946
Result: 350, 55,   1.093845,  11.164813, 7373.742188, 5309.645996, 2063.282471

Check precision loss:   0.813839
```

The statistics reported under `Result` include:
1. the total number of iterations executed
2. the iteration at which the highest amount of water was transferred between two cells
3. the maximum amount of water transferred in a single step
4. the highest water level reached in any cell
5. the total amount of rainwater discharged by the clouds
6. the total amount of water remaining on the ground at the end of the simulation
7. the amount of water lost through the boundaries of the terrain

To verify correctness, the results of the parallel implementation are compared against those of the provided sequential reference implementation (ground truth):
- the first statistic must match exactly
- statistics [3-7] may exhibit small numerical differences due to floating-point rounding effects that can accumulate during the execution. For these values, a small relative error is tolerated.

The reported "Check precision loss" summarizes the deviation in the last 3 statistics, and is acceptable if those 3 statistics are within the described margin.

To automate correctness checking, you can use the provided `test_files/check_correctness.py` verification script. The scripts expects two input files:
1. the output of the sequential (reference) program
2. the output of your CUDA implementation.

A typical workflow is the following
```bash
# Run the sequential version
$ prun -np 1 -native '-C gpunode' ./flood_seq $(cat test_files/small_mountains.in) > res_seq.out

# Run the CUDA version
$ prun -np 1 -native '-C gpunode' ./flood_cuda $(cat test_files/small_mountains.in) > res_cuda.out

# Compare the results
$ python3 test_files/check_correctness.py res_seq.out res_cuda.out
```

The script verifies that results are within the allowed tolerance and, if your implementation is correct, it will report:
```bash
Your output matches the reference.
```

### Compilation
To compile the source code on DAS-5, load `cuda12.6/toolkit` first: 

```bash
module load cuda12.6/toolkit
```

Then compile with the makefile by running `make all` or the specific targets. 

### Execution
You can run the sequential code with the following command:

```bash
./flood_seq <rows> <columns> <ground scenario> <threshold> <time> <ex_factor> <cloud dist> <cloud width> <cloud depth> <front direction> <rng cloud count> <radius> <intensity> <speed> <angle aperture> <seed> <optional additional clouds>
```

You may consider adding a script to assist with running these tests, using either the Makefile, bash, python or another approach of your choosing.
For an example, look at the arguments in the previous section or run `make test_seq` (note, depending on your shell you might need using double `$$` signs).



**NOTE: Always run with a compute node on DAS-5 to evaluate the performance, and do not execute with the head node. You need to use `prun`, `sbatch`, or other approaches to run your code on a DAS-5 compute node:**

It is recommended that all your experiments run on the same compute node.
For example, to run on a TitanX GPU you can append that to the gpunode argument:

```bash
prun -np 1 -native '-C gpunode,TitanX' ./flood_cuda [arguments...]
```
### Debug

To more comprehensively test correctness you may compile with the `-DDEBUG` flag (for example, using `make debug`) and compare the output to the reference implementation. **Note:** currently `make` rules are implemented only for the sequential version, but you can make your own.

The `DEBUG` flag, activates sections of the code that print to the standard output the arguments read, the terrain height matrix, the list of generated clouds with their corresponding data, and, at each simulation step, the changes in the terrain water level matrix. 
This mode can be useful for detecting possible errors or for gaining a better understanding of the program’s behavior with very small test cases.

Alternatively, you can set p.final_matrix to 1 at the start of the main function in `flood.c`, which print the water levels only once at the end of the simulation.

If the program is compiled with both `-DDEBUG` and `-DANIMATION` (e.g., using `make animation`), additional parts of the code are activated that write to the standard output the data and matrices required to generate an animation of the simulation. To do so, redirect the program output to a file and run the provided Python script (`test_files/animation.py`) with that file as input.
The script first displays the terrain; after a key is pressed, it proceeds to generate the simulation. The resulting animation is saved as an MP4 file.
**Note:** the `animation.py` script requires external packages not installed on DAS5 and considerable time also for small scenarios. Therefore run the script on your own computer.


## Resources

CUDA documents: https://docs.nvidia.com/cuda/archive/12.6.0/ 

CUDA debugging: https://docs.nvidia.com/cuda/cuda-gdb/index.html

CUDA profiling: https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof



## Evaluation

### Correctness
The provided input data files cover various grid and cloud configurations.
We will check the correctness with at least the 4 input files (`small_mountains.in`, `custom_clouds.in`, `medium_lower_dam.in`, and `medium_higher_dam.in`,) available under `test_files` and compare with the outputs from the provided sequential versions, using the provided `check_correctness.py` script.

Since the program uses single-precision floating-point arithmetic, small numerical differences may arise between CPU and GPU implementations, especially for larger or more complex scenarios (e.g., higher-resolution grids). For this reason, a limited deviation is permitted in the final results. **We therefore recommend performing correctness checks on smaller test cases, such as the four provided scenarios, where these discrepancies remain minimal and easier to interpret.**


### Performance
Provided that your code is correct (as defined above), you can show the performance with other input files (containing more points, higher dimensions, more clusters, etc.).
To do this, make your own `test_files/*.in` files using the parameters described in the execution section.

To analyze the speedup, you need to find a set of representative test cases. For instance, to test your speedup for larger grids and more clouds you can run the `large_mountains.in` arguments, which have higher numbers for these parameters:

```bash
prun -np 1 -native '-C gpunode,TitanX' ./flood_cuda $(cat test_files/large_mountains.in) > res_cuda.out
```
For reference, with this particular input file, a naive GPU implementation will achieve >2x speedup, while we expect you to achieve at least a 20x speedup over the sequential code.
A good implementation can achieve a speedup of >50x. **Can you do better than this?**

We will check your performance with your `flood_cuda.cu` source code and a set of (hidden) test files and arguments.


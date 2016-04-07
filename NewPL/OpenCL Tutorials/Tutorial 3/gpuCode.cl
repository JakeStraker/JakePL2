__kernel void averageTemperature(__global const int* temperature, __global int* output, __local int* scratch){ 
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	//cache all values from global memory to local memory
	scratch[lid] = temperature[id];

	barrier(CLK_LOCAL_MEM_FENCE); //Lock threads, wait for full set to finish

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (!lid) {
		atom_add(&output[0],scratch[lid]); //atomic add function locks resources
	}
}
/*
histogram calculation code, automatically calculate bin ranges using provided min and max
*/
__kernel void histogram(__global const int* temperature, __global int* output, int bincount, int minval, int maxval) { 
	int id = get_global_id(0);
	int bin_index = temperature[id];
	int range = maxval-minval; //range is important for calculating bin ranges
	int i = bin_index;
	int n = 0;
	int increment = range/bincount;
	int compareval = minval + increment;
	while (i > compareval)
	{
		compareval += increment;
		n++;
	}
	atomic_inc(&output[n]);
}
__kernel void minTemperature(__global const int* temperature, __global int* output, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all values from global memory to local memory
	scratch[lid] = temperature[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) {
		if (!(scratch[lid] <= scratch[lid + i]))
			scratch[lid] = scratch[lid + i];
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	if (!lid) {
		atom_min(&output[0], scratch[lid]);
	}

}
__kernel void maxTemperature(__global const int* temperature, __global int* output, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	//cache all values from global memory to local memory
	scratch[lid] = temperature[id];
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int i = 1; i < N; i *= 2) {
		if (!(scratch[lid] >= scratch[lid + i]))
			scratch[lid] = scratch[lid + i];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (!lid) {
		atom_max(&output[0], scratch[lid]);
	}
}

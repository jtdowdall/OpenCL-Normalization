kernel void Normalize
(
  const int num_weights,
  const int workGroupSize,
  __global float* matrix,
  __local float* weights,
  __local float* reduction
)
{
  int vector_index = get_global_id(0) * num_weights;
  int local_index = get_local_id(1);
  int workgroup_index = local_index * workGroupSize;
  float epsilon = 0.0011;
  
  
  /* Compute Difference */
  // Initialize minimum
  	reduction[workgroup_index] = INFINITY;
  	
  // Initialize maximum
  reduction[workgroup_index + 1] = -INFINITY;

  // Load data into local memory
  for (int local_weight = 0; local_weight < workGroupSize; local_weight++)
  {
  	int weight_index = workgroup_index + local_weight;
  	
  	// Find weight in global matrix;
  	if (weight_index < num_weights)
  	{
  		int weight = matrix[vector_index + weight_index];
  		weights[weight_index] = weight;
  		
  		if (weight < reduction[workgroup_index]) 
  			reduction[workgroup_index] = weight;
  		if (weight > reduction[workgroup_index + 1]) 
  			reduction[workgroup_index + 1] = weight;
  	}
  	// To satisfy power of two for reduction,
  	// pad vector with zeros
  	else weights[weight_index] = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  
  // Perform minimum and maximum reduction
  for(int offset = 1; offset < get_local_size(1); offset <<= 1) {
    int mask = (offset << 1) - 1;
    if ((local_index & mask) == 0) {      
      float minimumA = reduction[workgroup_index];
      float minimumB = reduction[workgroup_index + offset * workGroupSize];
      reduction[workgroup_index] = (minimumA < minimumB) ? minimumA : minimumB;
      
      float maximumA = reduction[workgroup_index + 1];
      float maximumB = reduction[workgroup_index + offset * workGroupSize + 1];
      reduction[workgroup_index + 1] = (maximumA > maximumB) ? maximumA : maximumB;
      
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float difference_value = reduction[1] - reduction[0] + epsilon;
  
  
  /* Update Elements */
  for (int local_weight = 0; local_weight < workGroupSize; local_weight++)
  {
	
	weights[workgroup_index + local_weight] = (weights[workgroup_index + local_weight] - reduction[0]) / difference_value;
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  /* Compute Mean */
  // Load data into local memory
  for (int local_weight = 0; local_weight < workGroupSize; local_weight++)
  {
  	
  	reduction[workgroup_index] += weights[workgroup_index + local_weight];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  
  
  // Perform sum reduction
  for(int offset = 1; offset < get_local_size(1); offset <<= 1) {
    int mask = (offset << 1) - 1;
    if ((local_index & mask) == 0) {
      
      //float sumA = reduction[workgroup_index];
      //float sumB = reduction[workgroup_index + offset * workGroupSize];
      reduction[workgroup_index] = (reduction[workgroup_index]+ reduction[workgroup_index + offset * workGroupSize]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float mean = reduction[0] / num_weights;
  
  
  /* Update Elements */
  for (int local_weight = 0; local_weight < workGroupSize; local_weight++)
  {
	weights[workgroup_index + local_weight] = weights[workgroup_index + local_weight] - mean + epsilon;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  
  
  /* Compute Norm */
  // Load data into local memory
  for (int local_weight = 0; local_weight < workGroupSize; local_weight++)
  {
  	reduction[workgroup_index] += pow(weights[workgroup_index + local_weight],2);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  
  
  // Perform norm reduction
  for(int offset = 1; offset < get_local_size(1); offset <<= 1) {
    int mask = (offset << 1) - 1;
    if ((local_index & mask) == 0) {
      //float normA = reduction[workgroup_index];
      //float normB = reduction[workgroup_index + offset * workGroupSize];
      reduction[workgroup_index] = (reduction[workgroup_index] + reduction[workgroup_index + offset * workGroupSize]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float norm_val = sqrt(reduction[0]);
  
  
  /* Update Elements in original matrix */
  for (int local_weight = 0; local_weight < workGroupSize; local_weight++)
  {
	matrix[vector_index + workgroup_index + local_weight] = difference_value;//weights[workgroup_index + local_weight] / norm_val;
  }
  
  
}
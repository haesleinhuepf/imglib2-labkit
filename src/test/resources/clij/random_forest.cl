__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void random_forest
(
  __global float* dst,
  __global float* src,
  __global float* thresholds,
  __global float* probabilities,
  __global float* indices,
  const int num_trees,
  const int num_classes
)
{
  const int x = get_global_id(0), y = get_global_id(1);
  for(int tree = 0; tree < num_trees; tree++) {
    int nodeIndex = 0;
    while(nodeIndex >= 0) {
      int attributeIndex = (int) READ_IMAGE_3D(indices, sampler, (int4)(0,nodeIndex,tree,0)).x;
      float attributeValue = READ_IMAGE_3D(src, sampler, (int4)(x,y,attributeIndex,0)).x;
      float threshold = READ_IMAGE_3D(thresholds, sampler, (int4)(0,nodeIndex,tree,0)).x;
      int smaller = attributeValue < threshold ? 1 : 2;
      nodeIndex = (int) READ_IMAGE_3D(indices, sampler, (int4)(smaller,nodeIndex,tree,0)).x;
    }
    int leafIndex = - 1 - nodeIndex;
    for(int i = 0; i < num_classes; i++) {
      float probability = READ_IMAGE_3D(dst, sampler, (int4)(x,y,i,0)).x;
      probability += READ_IMAGE_3D(probabilities, sampler, (int4)(i,leafIndex,tree,0)).x;
      WRITE_IMAGE_3D(dst, (int4)(x,y,i,0), (DTYPE_OUT) probability);
    }
  }

  // calculate sum of the distribution
  float sum = 0;
  for(int i = 0; i < num_classes; i++) {
    sum += READ_IMAGE_3D(dst, sampler, (int4)(x,y,i,0)).x;
  }

  // normalize distribution
  for(int i = 0; i < num_classes; i++) {
    float probability = READ_IMAGE_3D(dst, sampler, (int4)(x,y,i,0)).x;
    probability /= sum;
    WRITE_IMAGE_3D(dst, (int4)(x,y,i,0), (DTYPE_OUT) probability);
  }
}

use burn::tensor::Tensor;
// use burn::backend::Wgpu;
use burn::backend::NdArray;
// type Backend = Wgpu;

type Backend = NdArray;

fn main() {
    let device = Default::default();
    
    let tensor_1 = Tensor::<Backend,2>::from_data([[2., 3.], [4., 5.]], &device);
    let tensor_2 = Tensor::<Backend,2>::ones_like(&tensor_1);
    
    let result = tensor_1 + tensor_2;
    
    println!("Result: {}", result);
}

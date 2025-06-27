use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::vision::MnistItem;
use burn::prelude::*;

#[derive(Clone, Debug, Default)]
pub struct FashionMNISTBatcher {}

#[derive(Clone, Debug)]
pub struct FashionMNISTBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, MnistItem, FashionMNISTBatch<B>> for FashionMNISTBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &B::Device) -> FashionMNISTBatch<B> {
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image))
            .map(|data| {
                // Flatten and normalize f32 pixel values (original in 0-255 range)
                let pixel_data: Vec<f32> = data.into_vec::<f32>().unwrap().iter().map(|&v| v / 255.0).collect();
                // Create 2D tensor data with shape [28, 28]
                let tensor_data = TensorData::new(pixel_data, [28, 28]);
                // Convert dtype and build tensor
                Tensor::<B, 2>::from_data(tensor_data.convert::<B::FloatElem>(), device)
            })
            .map(|tensor| tensor.reshape([1, 28, 28]))
            // Apply normalization: mean=0.1307, std=0.3081
            .map(|tensor| (tensor - 0.1307) / 0.3081)
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(
                    TensorData::from([(item.label as i64).elem::<B::IntElem>()]),
                    device,
                )
            })
            .collect();
        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);
        FashionMNISTBatch { images, targets }
    }
}

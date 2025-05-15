use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
};
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::vision::MnistDataset;
use burn_ndarray::NdArray;
use std::sync::Arc;
use burn::data::dataloader::DataLoader;
use plotters::prelude::*;
use plotters::backend::BitMapBackend;
use std::error::Error;
use std::io;

#[derive(Clone, Debug, Default)]
struct MnistBatcher {}

#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, MnistItem, MnistBatch<B>> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &B::Device) -> MnistBatch<B> {
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image))
            .map(|data| Tensor::<B, 2>::from_data(data.convert::<B::FloatElem>(), device))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
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

        MnistBatch { images, targets }
    }
}

static ARTIFACT_DIR: &str = "/tmp/burn-example-mnist";

#[derive(Config)]
struct MnistTrainingConfig {
    #[config(default = 10)]
    num_epochs: usize,

    #[config(default = 64)]
    batch_size: usize,

    #[config(default = 4)]
    num_workers: usize,

    #[config(default = 42)]
    seed: u64,
}

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

type NdBatch = MnistBatch<NdArray>;
// Alias for readability
type NdDataLoader = Arc<dyn DataLoader<NdArray, NdBatch>>;

fn build_dataloaders(config: &MnistTrainingConfig) -> (NdDataLoader, NdDataLoader) {
    let batcher = MnistBatcher::default();
    let train: NdDataLoader = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::train());
    let test: NdDataLoader = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::test());
    (train, test)
}

// Plot a grayscale image from f32 buffer without extra u8 allocation
fn plot_f32_image(img: &[f32], width: usize, height: usize, path: &str) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(path, (width as u32, height as u32)).into_drawing_area();
    root.fill(&WHITE)?;
    for (i, &v) in img.iter().enumerate() {
        let x = (i % width) as i32;
        let y = (i / width) as i32;
        let gray = (v * 255.0) as u8;
        root.draw_pixel((x, y), &RGBColor(gray, gray, gray))?;
    }
    root.present()?;
    Ok(())
}

fn inspect_and_plot_first_batch(loader: &NdDataLoader) -> Result<(), Box<dyn Error>> {
    if let Some(MnistBatch { images: raw_images, targets: labels }) = loader.iter().next() {
        // Print shapes
        let dims = raw_images.dims(); // [batch, height, width]
        println!("Image shape: {:?}", dims);
        println!("Labels shape: {:?}", labels.to_data().shape);

        // Flatten inputs
        let num_features = dims[1] * dims[2];
        let inputs = raw_images.clone().reshape([dims[0], num_features]);
        println!("Flattened inputs shape: {:?}", inputs.dims());

        // Plot first image
        let data = raw_images.to_data();
        let flat: Vec<f32> = data.into_vec().map_err(|e| io::Error::new(io::ErrorKind::Other, format!("DataError: {:?}", e)))?;
        plot_f32_image(&flat[..num_features], dims[2], dims[1], "mnist.png")?;
        println!("Saved first image to mnist.png");
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    create_artifact_dir(ARTIFACT_DIR);
    let config = MnistTrainingConfig::new();
    let (dataloader_train, dataloader_test) = build_dataloaders(&config);
    println!("Number of train batches: {}", dataloader_train.iter().count());
    println!("Number of test batches: {}", dataloader_test.iter().count());
    inspect_and_plot_first_batch(&dataloader_train)?;
    Ok(())
}
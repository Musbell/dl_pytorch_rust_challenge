//https://github.com/richardyang92/burn-examples/blob/main/fashion_mnist/src/data.rs
use std::{fs::File, io::{Read, Seek, SeekFrom}, path::Path};
use burn::data::dataset::{Dataset, InMemDataset};
use burn::data::dataset::transform::{Mapper, MapperDataset};
use burn::data::dataset::vision::MnistItem;
use burn::serde::Deserialize;

const TRAIN_IMAGES: &str = "train-images-idx3-ubyte";
const TRAIN_LABELS: &str = "train-labels-idx1-ubyte";
const TEST_IMAGES: &str = "t10k-images-idx3-ubyte";
const TEST_LABELS: &str = "t10k-labels-idx1-ubyte";

const WIDTH: usize = 28;
const HEIGHT: usize = 28;

#[derive(Deserialize, Debug, Clone)]
struct MNISTItemRaw {
    pub image_bytes: Vec<u8>,
    pub label: u8,
}

struct BytesToImage;

impl Mapper<MNISTItemRaw, MnistItem> for BytesToImage {
    /// Convert a raw MNIST item (image bytes) to a MNIST item (2D array image).
    fn map(&self, item: &MNISTItemRaw) -> MnistItem {
        // Ensure the image dimensions are correct.
        debug_assert_eq!(item.image_bytes.len(), WIDTH * HEIGHT);

        // Convert the image to a 2D array of floats.
        let mut image_array = [[0f32; WIDTH]; HEIGHT];
        for (i, pixel) in item.image_bytes.iter().enumerate() {
            let x = i % WIDTH;
            let y = i / HEIGHT;
            image_array[y][x] = *pixel as f32;
        }

        MnistItem {
            image: image_array,
            label: item.label,
        }
    }
}

type MappedDataset = MapperDataset<InMemDataset<MNISTItemRaw>, BytesToImage, MNISTItemRaw>;

pub struct FashionMNISTDataset {
    dataset: MappedDataset,
}

impl Dataset<MnistItem> for FashionMNISTDataset {
    fn get(&self, index: usize) -> Option<MnistItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl FashionMNISTDataset {
    /// Creates a new train dataset.
    pub fn train() -> Self {
        // Try different paths to find the data directory
        let paths = vec!["./data", "../fashion-mnist/data", "./fashion-mnist/data"];

        for path in paths {
            if Path::new(path).exists() {
                return Self::train_with_root(path);
            }
        }

        // If none of the paths exist, use the first one and let the error handling show a clear message
        Self::train_with_root("./data")
    }

    /// Creates a new test dataset.
    pub fn test() -> Self {
        // Try different paths to find the data directory
        let paths = vec!["./data", "../fashion-mnist/data", "./fashion-mnist/data"];

        for path in paths {
            if Path::new(path).exists() {
                return Self::test_with_root(path);
            }
        }

        // If none of the paths exist, use the first one and let the error handling show a clear message
        Self::test_with_root("./data")
    }

    /// Creates a new train dataset with a custom root path.
    pub fn train_with_root<P: AsRef<Path>>(root: P) -> Self {
        Self::new("train", root)
    }

    /// Creates a new test dataset with a custom root path.
    pub fn test_with_root<P: AsRef<Path>>(root: P) -> Self {
        Self::new("test", root)
    }

    fn new<P: AsRef<Path>>(split: &str, root: P) -> Self {
        // Use the provided root path

        // MNIST is tiny so we can load it in-memory
        // Train images (u8): 28 * 28 * 60000 = 47.04Mb
        // Test images (u8): 28 * 28 * 10000 = 7.84Mb
        let images = FashionMNISTDataset::read_images(&root, split);
        let labels = FashionMNISTDataset::read_labels(&root, split);

        // Collect as vector of MNISTItemRaw
        let items: Vec<_> = images
            .into_iter()
            .zip(labels)
            .map(|(image_bytes, label)| MNISTItemRaw { image_bytes, label })
            .collect();

        let dataset = InMemDataset::new(items);
        let dataset = MapperDataset::new(dataset, BytesToImage);

        Self { dataset }
    }

    /// Read images at the provided path for the specified split.
    /// Each image is a vector of bytes.
    fn read_images<P: AsRef<Path>>(root: &P, split: &str) -> Vec<Vec<u8>> {
        let file_name = if split == "train" {
            TRAIN_IMAGES
        } else {
            TEST_IMAGES
        };
        let file_path = root.as_ref().join(file_name);

        // Read number of images from 16-byte header metadata
        let mut f = match File::open(&file_path) {
            Ok(file) => file,
            Err(e) => panic!("Failed to open image file at {:?}: {}", file_path, e),
        };

        let mut buf = [0u8; 4];
        match f.seek(SeekFrom::Start(4)) {
            Ok(_) => {},
            Err(e) => panic!("Failed to seek in image file: {}", e),
        }

        f.read_exact(&mut buf)
            .expect("Should be able to read image file header");
        let size = u32::from_be_bytes(buf);

        let mut buf_images: Vec<u8> = vec![0u8; WIDTH * HEIGHT * (size as usize)];
        match f.seek(SeekFrom::Start(16)) {
            Ok(_) => {},
            Err(e) => panic!("Failed to seek in image file: {}", e),
        }

        f.read_exact(&mut buf_images)
            .expect("Should be able to read image file content");

        buf_images
            .chunks(WIDTH * HEIGHT)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    /// Read labels at the provided path for the specified split.
    fn read_labels<P: AsRef<Path>>(root: &P, split: &str) -> Vec<u8> {
        let file_name = if split == "train" {
            TRAIN_LABELS
        } else {
            TEST_LABELS
        };
        let file_path = root.as_ref().join(file_name);

        // Read number of labels from 8-byte header metadata
        let mut f = match File::open(&file_path) {
            Ok(file) => file,
            Err(e) => panic!("Failed to open label file at {:?}: {}", file_path, e),
        };

        let mut buf = [0u8; 4];
        match f.seek(SeekFrom::Start(4)) {
            Ok(_) => {},
            Err(e) => panic!("Failed to seek in label file: {}", e),
        }

        f.read_exact(&mut buf)
            .expect("Should be able to read label file header");
        let size = u32::from_be_bytes(buf);

        let mut buf_labels: Vec<u8> = vec![0u8; size as usize];
        match f.seek(SeekFrom::Start(8)) {
            Ok(_) => {},
            Err(e) => panic!("Failed to seek in label file: {}", e),
        }

        f.read_exact(&mut buf_labels)
            .expect("Should be able to read labels from file");

        buf_labels
    }
}

pub mod error;
pub mod math_helper;
pub mod matrix;
pub(crate) mod vector2;
pub(crate) mod vector3;

pub mod vector {
    pub trait Vector {
        #[allow(non_snake_case)]
        /// All elements of the vector are `1`
        fn ONE() -> Self;
        #[allow(non_snake_case)]
        // All elements of the vector are `0`
        fn ZERO() -> Self;
        /// The square root of the sum of each vector part to the power of 2
        fn magnitude(&self) -> f32;
        /// VectorN to Vec<f32>, N in (2, 3)
        fn to_vector(&self) -> Vec<f32>;
    }

    pub use vector2::Vector2;
    pub use vector3::Vector3;
}

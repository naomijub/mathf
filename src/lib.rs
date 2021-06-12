pub mod error;
pub(crate) mod math_helper;
pub mod matrix;
pub(crate) mod vector2;
pub(crate) mod vector3;

pub mod vector {
    pub trait Vector {
        #[allow(non_snake_case)]
        fn ONE() -> Self;
        #[allow(non_snake_case)]
        fn ZERO() -> Self;
        fn magnitude(&self) -> f32;
        fn to_vector(&self) -> Vec<f32>;
    }

    pub use vector2::Vector2;
    pub use vector3::Vector3;
}

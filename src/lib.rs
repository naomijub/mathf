pub(crate) mod vector2;
pub(crate) mod vector3;
pub(crate) mod math_helper;
pub mod matrix;
pub mod error;


pub mod vector {
    pub use vector2::Vector2;
    pub use vector3::Vector3;
}
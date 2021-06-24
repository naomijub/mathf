pub mod error;
pub mod math_helper;
pub mod matrix;
pub mod quaternion;
pub(crate) mod vector2;
pub(crate) mod vector3;

pub mod vector {
    pub use vector2::Point2;
    pub use vector2::Vector2;
    pub use vector3::Point3;
    pub use vector3::Vector3;
}

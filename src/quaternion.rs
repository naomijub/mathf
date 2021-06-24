use crate::error::Error;
use crate::math_helper::float_eq;
use crate::vector3::Vector3;

use std::ops;

/// A Quaternion with i, j, k, w coordinates: Quaternion
#[derive(Clone, PartialEq, Debug)]
pub struct Quaternion {
    pub i: f32,
    pub j: f32,
    pub k: f32,
    pub w: f32,
}

impl Quaternion {
    /// Instantiates a new quaternion with to be defined values of i, j, k w;
    pub fn new(i: f32, j: f32, k: f32, w: f32) -> Quaternion {
        Quaternion { i, j, k, w }
    }

    #[allow(non_snake_case)]
    /// ```
    /// // | 0 |
    /// // | 1 |
    /// // | 0 |
    /// // | w |
    /// // j=1, i=0, k=0, w=?
    /// ```
    pub fn UP(w: f32) -> Quaternion {
        Quaternion {
            i: 0f32,
            j: 1f32,
            k: 0f32,
            w,
        }
    }

    #[allow(non_snake_case)]
    /// ```
    /// // |  0 |
    /// // | -1 |
    /// // |  0 |
    /// // |  w |
    /// // j=-1, i=0, k=0, w=?
    /// ```
    pub fn DOWN(w: f32) -> Quaternion {
        Quaternion {
            i: 0f32,
            j: -1f32,
            k: 0f32,
            w,
        }
    }

    #[allow(non_snake_case)]
    /// ```
    /// // | 1 |
    /// // | 0 |
    /// // | 0 |
    /// // | w |
    /// // j=0, i=1, k=0, w=?
    /// ```
    pub fn RIGHT(w: f32) -> Quaternion {
        Quaternion {
            i: 1f32,
            j: 0f32,
            k: 0f32,
            w,
        }
    }

    #[allow(non_snake_case)]
    /// ```
    /// // | -1 |
    /// // |  0 |
    /// // |  0 |
    /// // |  w |
    /// // j=0, i=-1, k=0, w=?
    /// ```
    pub fn LEFT(w: f32) -> Quaternion {
        Quaternion {
            i: -1f32,
            j: 0f32,
            k: 0f32,
            w,
        }
    }

    #[allow(non_snake_case)]
    /// ```
    /// // | 0 |
    /// // | 0 |
    /// // | 1 |
    /// // | w |
    /// // j=0, i=0, k=1, w=?
    /// ```
    pub fn FOWARD(w: f32) -> Quaternion {
        Quaternion {
            i: 0f32,
            j: 0f32,
            k: 1f32,
            w,
        }
    }

    #[allow(non_snake_case)]
    /// ```
    /// // |  0 |
    /// // |  0 |
    /// // | -1 |
    /// // |  w |
    /// // j=0, i=0, k=-1, w=?
    /// ```
    pub fn BACK(w: f32) -> Quaternion {
        Quaternion {
            i: 0f32,
            j: 0f32,
            k: -1f32,
            w,
        }
    }

    /// Cross product between two Quaternions.
    /// ```
    /// // | a |   | m |
    /// // | d | x | n |
    /// // | g |   | o |
    /// // | w |   | v |
    /// ```
    pub fn x(&self, new_quat: Quaternion) -> Quaternion {
        Quaternion {
            i: self.w * new_quat.w
                - self.i * new_quat.i
                - self.j * new_quat.j
                - self.k * new_quat.k,
            j: self.w * new_quat.w - self.i * new_quat.i
                + self.j * new_quat.j
                + self.k * new_quat.k,
            k: self.w * new_quat.w + self.i * new_quat.i - self.j * new_quat.j
                + self.k * new_quat.k,
            w: self.w * new_quat.w
                - self.i * new_quat.i
                - self.j * new_quat.j
                - self.k * new_quat.k,
        }
    }

    #[allow(non_snake_case)]
    /// All elements of the quaternion are `1`
    /// ```
    /// // | 1 |
    /// // | 1 |
    /// // | 1 |
    /// // | 1 |
    /// // i=1, j=1, k=1, w=1
    /// ```
    pub fn ONE() -> Quaternion {
        Quaternion {
            i: 1f32,
            j: 1f32,
            k: 1f32,
            w: 1f32,
        }
    }

    #[allow(non_snake_case)]
    /// All elements of the quaternion are `0`
    /// * Defines a Modulus ZERO quaternion (i=0, j=0, k=0), w=?
    /// ```
    /// // | 0 |
    /// // | 0 |
    /// // | 0 |
    /// // | 0 |
    /// ```
    pub fn ZERO() -> Quaternion {
        Quaternion {
            i: 0f32,
            j: 0f32,
            k: 0f32,
            w: 0f32,
        }
    }

    /// quaternion magnitude: the square root of the sum of each quaternion part to the power of 2
    pub fn magnitude(&self) -> f32 {
        f32::sqrt(self.squared())
    }

    fn squared(&self) -> f32 {
        self.i.powi(2) + self.j.powi(2) + self.k.powi(2) + self.w.powi(2)
    }

    pub fn conjugate(&self) -> Quaternion {
        Quaternion {
            i: -self.i,
            j: -self.j,
            k: -self.k,
            w: self.w,
        }
    }

    pub fn scalar_vector_format(&self) -> (f32, Vector3) {
        (
            self.w,
            Vector3 {
                x: self.i,
                y: self.j,
                z: self.k,
            },
        )
    }

    pub fn scalar_vector_conjugate_format(&self) -> (f32, Vector3) {
        (
            self.w,
            Vector3 {
                x: -self.i,
                y: -self.j,
                z: -self.k,
            },
        )
    }

    pub fn inverse(&self) -> Result<Quaternion, Error> {
        let magnitude_squared = self.squared();
        if float_eq(magnitude_squared, 0., 0.0001) {
            Ok(self.conjugate() / magnitude_squared)
        } else {
            Err(Error::QuaternionNotInversible)
        }
    }
}

impl ops::Add for Quaternion {
    type Output = Quaternion;

    /// Implements the Quaternion '+' trait
    fn add(self, new_quaternion: Quaternion) -> Quaternion {
        Quaternion {
            i: self.i + new_quaternion.i,
            j: self.j + new_quaternion.j,
            k: self.k + new_quaternion.k,
            w: self.w + new_quaternion.w,
        }
    }
}

impl ops::Add for &Quaternion {
    type Output = Quaternion;

    /// Implements the &Quaternion '+' trait
    fn add(self, new_quaternion: &Quaternion) -> Quaternion {
        Quaternion {
            i: self.i + new_quaternion.i,
            j: self.j + new_quaternion.j,
            k: self.k + new_quaternion.k,
            w: self.w + new_quaternion.w,
        }
    }
}

impl ops::Mul<f32> for Quaternion {
    type Output = Quaternion;

    /// Implements the scalar multiplication of a Quaternion with a f32. Other numbers should be passed with 'i as f32'
    fn mul(self, value: f32) -> Quaternion {
        Quaternion {
            i: self.i * value,
            j: self.j * value,
            k: self.k * value,
            w: self.w * value,
        }
    }
}

impl ops::Mul<f32> for &Quaternion {
    type Output = Quaternion;

    /// Implements the scalar multiplication of a &Quaternion with a f32. Other numbers should be passed with 'i as f32'
    fn mul(self, value: f32) -> Quaternion {
        Quaternion {
            i: self.i * value,
            j: self.j * value,
            k: self.k * value,
            w: self.w * value,
        }
    }
}

impl ops::Div<f32> for Quaternion {
    type Output = Quaternion;

    /// Implements the scalar division of a Quaternion with a f32. Other numbers should be passed with 'i as f32'
    fn div(self, value: f32) -> Quaternion {
        Quaternion {
            i: self.i / value,
            j: self.j / value,
            k: self.k / value,
            w: self.w / value,
        }
    }
}

impl ops::Div<f32> for &Quaternion {
    type Output = Quaternion;

    /// Implements the scalar division of a &Quaternion with a f32. Other numbers should be passed with 'i as f32'
    fn div(self, value: f32) -> Quaternion {
        Quaternion {
            i: self.i / value,
            j: self.j / value,
            k: self.k / value,
            w: self.w / value,
        }
    }
}

impl ops::Mul<Quaternion> for Quaternion {
    type Output = f32;

    /// Implements the dot product of 2 Quaternion as '*'.
    fn mul(self, new_quat: Quaternion) -> f32 {
        self.w * new_quat.w + self.i * new_quat.i + self.j * new_quat.j + self.k * new_quat.k
    }
}

impl ops::Mul<&Quaternion> for &Quaternion {
    type Output = f32;

    /// Implements the dot product of 2 &Quaternion as '*'.
    fn mul(self, new_quat: &Quaternion) -> f32 {
        self.w * new_quat.w * self.i * new_quat.i * self.j * new_quat.j * self.k * new_quat.k
    }
}

impl ops::Sub for Quaternion {
    type Output = Quaternion;

    /// Implements the Quaternion '-' trait
    fn sub(self, new_quat: Quaternion) -> Quaternion {
        Quaternion {
            i: self.i - new_quat.i,
            j: self.j - new_quat.j,
            k: self.k - new_quat.k,
            w: self.w - new_quat.w,
        }
    }
}

impl ops::Sub for &Quaternion {
    type Output = Quaternion;

    /// Implements the &Quaternion '-' trait
    fn sub(self, new_quat: &Quaternion) -> Quaternion {
        Quaternion {
            i: self.i - new_quat.i,
            j: self.j - new_quat.j,
            k: self.k - new_quat.k,
            w: self.w - new_quat.w,
        }
    }
}

// Indexing
use std::ops::Index;

impl Index<usize> for Quaternion {
    type Output = f32;
    fn index(&self, s: usize) -> &f32 {
        match s {
            0 => &self.i,
            1 => &self.j,
            2 => &self.k,
            3 => &self.w,
            _ => panic!("Index out of bonds"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_quaternion_with_parameters() {
        let actual = Quaternion::new(1f32, 2f32, 4f32, 3f32);
        let expected = Quaternion {
            i: 1f32,
            j: 2f32,
            k: 4f32,
            w: 3f32,
        };
        assert_eq!(expected, actual);
    }

    #[test]
    fn creates_quaternion_up() {
        let actual = Quaternion::UP(0.5f32);
        assert!(actual.i == 0f32 && actual.j == 1f32 && actual.k == 0f32 && actual.w == 0.5f32);
    }

    #[test]
    fn adds_right_and_left_quaternions() {
        let actual = Quaternion::RIGHT(1f32) + Quaternion::LEFT(1f32);
        assert_eq!(actual.i, 0f32);
    }

    #[test]
    fn adds_right_and_left_quaternions_by_ref() {
        let actual = &Quaternion::RIGHT(1f32) + &Quaternion::LEFT(1f32);
        assert_eq!(actual.i, 0f32);
    }

    #[test]
    fn adds_right_and_up() {
        let actual = Quaternion::RIGHT(1f32) + Quaternion::UP(1f32);
        assert_eq!(actual.i, 1f32);
        assert_eq!(actual.j, 1f32);
        assert_eq!(actual.k, 0f32);
        assert_eq!(actual.w, 2f32);
    }

    #[test]
    fn mult_one_by_3() {
        let actual = Quaternion::ONE() * 3f32;
        assert_eq!(actual.i, 3f32);
        assert_eq!(actual.j, 3f32);
        assert_eq!(actual.k, 3f32);
        assert_eq!(actual.w, 3f32);
    }

    #[test]
    fn mult_one_by_3_by_ref() {
        let actual = &Quaternion::ONE() * 3f32;
        assert_eq!(actual.i, 3f32);
        assert_eq!(actual.j, 3f32);
        assert_eq!(actual.k, 3f32);
        assert_eq!(actual.w, 3f32);
    }

    #[test]
    fn sub_right_from_one() {
        let actual = Quaternion::ONE() - Quaternion::RIGHT(1f32);
        assert_eq!(actual.i, 0f32);
        assert_eq!(actual.j, 1f32);
        assert_eq!(actual.k, 1f32);
        assert_eq!(actual.w, 0f32);
    }

    #[test]
    fn sub_right_from_one_by_ref() {
        let actual = &Quaternion::ONE() - &Quaternion::RIGHT(1f32);
        assert_eq!(actual.i, 0f32);
        assert_eq!(actual.j, 1f32);
        assert_eq!(actual.k, 1f32);
        assert_eq!(actual.w, 0f32);
    }

    #[test]
    fn magnitude_of_quaternion() {
        let quat = Quaternion {
            i: 1f32,
            j: 2f32,
            k: 3f32,
            w: 4f32,
        };
        assert_eq!(quat.magnitude(), 5.477226);
    }

    #[test]
    fn magnitude_of_quaternion_is_positive() {
        let quat = Quaternion {
            i: -3f32,
            j: 4f32,
            k: 5f32,
            w: 7f32,
        };
        assert!(quat.magnitude() >= 0f32);
    }

    #[test]
    fn dot_product() {
        let quat1 = Quaternion {
            i: 2f32,
            j: 1f32,
            k: 2f32,
            w: 1f32,
        };
        let quat2 = Quaternion {
            i: 1f32,
            j: 2f32,
            k: 3f32,
            w: 1f32,
        };
        let actual = quat1 * quat2;
        let expected = 11f32;
        assert_eq!(actual, expected);
    }

    #[test]
    fn cross_product_between_2_quaternions() {
        let quat1 = Quaternion::new(1f32, 2f32, 3f32, 1f32);
        let actual = quat1.x(Quaternion::new(2f32, 3f32, 4f32, 1f32));
        let expected = Quaternion::new(-19f32, 17f32, 9f32, -19f32);
        assert_eq!(actual, expected);
    }

    #[test]
    fn indexing_0_returns_i() {
        let quaternion = Quaternion::new(1f32, 2f32, 3f32, 4f32);

        assert_eq!(quaternion[0usize], 1f32);
    }

    #[test]
    fn indexing_1_returns_j() {
        let quaternion = Quaternion::new(1f32, 2f32, 3f32, 4f32);

        assert_eq!(quaternion[1usize], 2f32);
    }

    #[test]
    fn indexing_2_returns_k() {
        let quaternion = Quaternion::new(1f32, 2f32, 3f32, 4f32);

        assert_eq!(quaternion[2usize], 3f32);
    }

    #[test]
    fn indexing_3_returns_w() {
        let quaternion = Quaternion::new(1f32, 2f32, 3f32, 4f32);

        assert_eq!(quaternion[3], 4f32);
    }

    #[test]
    fn arithmetic() {
        let v = Quaternion::ONE();

        assert_eq!(
            v.clone() / 2.0,
            Quaternion::new(0.5f32, 0.5f32, 0.5f32, 0.5)
        );
        assert_eq!(v.clone() * 2.0, Quaternion::new(2f32, 2f32, 2f32, 2f32));
    }

    #[test]
    fn arithmetic_by_ref() {
        let v = Quaternion::ONE();

        assert_eq!(&v / 2.0, Quaternion::new(0.5f32, 0.5f32, 0.5f32, 0.5));
        assert_eq!(&v * 2.0, Quaternion::new(2f32, 2f32, 2f32, 2f32));
    }
}

use crate::error::Error;
use crate::math_helper;
use crate::math_helper::float_eq;
use crate::vector2::Point2;
use crate::vector3::Point3;

use super::vector2::Vector2;
use super::vector3::Vector3;
use std::ops;

/// Implements a matrix 3 x 3
#[derive(Clone, PartialEq, Debug)]
pub struct Matrix3x3 {
    pub r1: Vector3,
    pub r2: Vector3,
    pub r3: Vector3,
}

/// Implements a matrix 2 x 2
#[derive(Clone, PartialEq, Debug)]
pub struct Matrix2x2 {
    pub r1: Vector2,
    pub r2: Vector2,
}

pub enum RotationAxis {
    X,
    Y,
    Z,
}

impl Matrix3x3 {
    #[allow(non_snake_case)]
    /// Identity Matrrix NxN, where all elements Uij that i = j are `1`
    /// ```
    /// // | 1 0 0 |
    /// // | 0 1 0 |
    /// // | 0 0 1 |
    /// ```
    pub fn IDENTITY() -> Self {
        Matrix3x3::new_idx(1f32, 0f32, 0f32, 0f32, 1f32, 0f32, 0f32, 0f32, 1f32)
    }

    /// Matrix determinant
    /// ```
    /// //                | 4 -3 0 |
    /// // let matrix =   | 2 -1 2 |
    /// //                | 1  5 7 |
    /// // matrix.det() = -32
    /// ```
    ///
    /// ```
    /// use mathf::matrix::{Matrix3x3};
    ///
    /// let matrix = Matrix3x3::new_idx(4.0, -3.0, 0.0, 2.0, -1.0, 2.0, 1.0,  5.0, 7.0);
    /// assert_eq!(matrix.det(), -32f32);
    /// ```
    pub fn det(&self) -> f32 {
        (self.r1.x * self.r2.y * self.r3.z
            + self.r1.y * self.r2.z * self.r3.x
            + self.r1.z * self.r2.x * self.r3.y)
            - (self.r1.z * self.r2.y * self.r3.x
                + self.r1.y * self.r2.x * self.r3.z
                + self.r1.x * self.r2.z * self.r3.y)
    }

    /// Transforms a MatrixNxN into a Vec<Vec<f32>> by row
    /// ```
    /// use mathf::matrix::{Matrix3x3};
    ///
    /// let matrix = Matrix3x3::new_idx(4.0, -3.0, 0.0, 2.0, -1.0, 2.0, 1.0,  5.0, 7.0);
    /// assert_eq!(matrix.vectorize(), vec![
    ///    vec![4.0, -3.0, 0.0], vec![2.0, -1.0, 2.0], vec![1.0,  5.0, 7.0]
    /// ]);
    /// ```
    pub fn vectorize(&self) -> Vec<Vec<f32>> {
        vec![
            self.r1.to_vector(),
            self.r2.to_vector(),
            self.r3.to_vector(),
        ]
    }

    /// Modulus of a matrix 3x3
    ///                | 4 -3 0 |
    /// let matrix =   | 2 -1 2 |
    ///                | 1  5 7 |
    /// matrix.modulus() =
    /// -> 4 | -1 2 | - (-3) | 2 2 | + | 2 -1 |
    ///      |  5 7 |        | 1 7 |   | 1  5 |
    pub fn modulus(&self) -> f32 {
        let elements: Vec<f32> = (0..=2)
            .map(|i| {
                let idxs: Vec<usize> = (0..=2).filter(|x| x != &i).collect();
                let factor = self.r1[i];
                let inner2x2 = Matrix2x2::new_idx(
                    self.r2[idxs[0]],
                    self.r2[idxs[1]],
                    self.r3[idxs[0]],
                    self.r3[idxs[1]],
                );
                factor * inner2x2.det()
            })
            .collect();
        elements[0] - elements[1] + elements[2]
    }

    /// Transpose of Matrix NxN.
    /// Transpose switches rows and columns
    pub fn transpose(&self) -> Matrix3x3 {
        Matrix3x3::new_idx(
            self.r1[0], self.r2[0], self.r3[0], self.r1[1], self.r2[1], self.r3[1], self.r1[2],
            self.r2[2], self.r3[2],
        )
    }

    /// Matrix inverse
    /// ```
    /// //               | 1 2 3 |
    /// //let matrix =   | 0 1 5 |
    /// //               | 5 6 0 |
    /// //                           | -30  18  7 |
    /// //let matrix_inverse =  1/5  |  25 -15 -5 |
    /// //                           | -5   4   1 |
    /// ```
    ///
    /// ```
    /// use mathf::matrix::{Matrix3x3};
    ///
    /// let matrix = Matrix3x3::new_idx(1.0, 2.0, 3.0, 0.0, 1.0, 5.0, 5.0,  6.0, 0.0);
    /// let matrix_inverse = Matrix3x3::new_idx(-6.0, 3.6, 1.4, 5.0, -3.0, -1.0, -1.0,  0.8, 0.2);
    /// assert_eq!(matrix.inverse(0.001).unwrap(), matrix_inverse);
    /// ```
    pub fn inverse(&self, delta: f32) -> Result<Matrix3x3, Error> {
        let modulus = self.modulus();
        if float_eq(modulus, 0f32, delta) {
            return Err(Error::SingularMatrixNotInversible);
        }
        let cofactor_matrix = self.cofactor();
        let traspose = cofactor_matrix.transpose();
        Ok(traspose / modulus)
    }

    /// A matrix is orthogonal if and only if transpose(A) == inverse(A)
    pub fn is_orthogonal(&self) -> bool {
        let inv = self.inverse(0.0001);
        if inv.is_ok() {
            self.transpose() == inv.unwrap()
        } else {
            false
        }
    }

    /// Uniform scale matrix is determined as Uij / i==j => i = a ^ i!=j => i = 0
    /// ```
    /// // | a 0 0 |
    /// // | 0 a 0 |
    /// // | 0 0 a |
    /// ```
    pub fn scale_matrix(a: f32) -> Self {
        Matrix3x3::IDENTITY() * a
    }

    /// Creates a new Matrix3x3 from 3 vector3 rows
    pub fn new(r1: Vector3, r2: Vector3, r3: Vector3) -> Matrix3x3 {
        Matrix3x3 { r1, r2, r3 }
    }

    /// Creates a new Matrix3x3 from 9 indexed floats
    pub fn new_idx(
        n1: f32,
        n2: f32,
        n3: f32,
        n4: f32,
        n5: f32,
        n6: f32,
        n7: f32,
        n8: f32,
        n9: f32,
    ) -> Matrix3x3 {
        Matrix3x3 {
            r1: Vector3::new(n1, n2, n3),
            r2: Vector3::new(n4, n5, n6),
            r3: Vector3::new(n7, n8, n9),
        }
    }

    /// Cofactor matrix [wiki](https://en.wikipedia.org/wiki/Minor_(linear_algebra))
    pub fn cofactor(&self) -> Matrix3x3 {
        let elements1: Vec<f32> = (0..=2)
            .map(|i| {
                let idxs: Vec<usize> = (0..=2).filter(|x| x != &i).collect();
                let inner2x2 = Matrix2x2::new_idx(
                    self.r2[idxs[0]],
                    self.r2[idxs[1]],
                    self.r3[idxs[0]],
                    self.r3[idxs[1]],
                );
                inner2x2.det()
            })
            .collect();

        let elements2: Vec<f32> = (0..=2)
            .map(|i| {
                let idxs: Vec<usize> = (0..=2).filter(|x| x != &i).collect();
                let inner2x2 = Matrix2x2::new_idx(
                    self.r1[idxs[0]],
                    self.r1[idxs[1]],
                    self.r3[idxs[0]],
                    self.r3[idxs[1]],
                );
                inner2x2.det()
            })
            .collect();

        let elements3: Vec<f32> = (0..=2)
            .map(|i| {
                let idxs: Vec<usize> = (0..=2).filter(|x| x != &i).collect();
                let inner2x2 = Matrix2x2::new_idx(
                    self.r1[idxs[0]],
                    self.r1[idxs[1]],
                    self.r2[idxs[0]],
                    self.r2[idxs[1]],
                );
                inner2x2.det()
            })
            .collect();
        Matrix3x3::new_idx(
            elements1[0],
            -elements1[1],
            elements1[2],
            -elements2[0],
            elements2[1],
            -elements2[2],
            elements3[0],
            -elements3[1],
            elements3[2],
        )
    }

    /// Non-uniform scale matrix is determined as Uij / i==j => i belongs to {a, b, c}  ^ i!=j => i = 0
    /// ```
    /// // | a 0 0 |
    /// // | 0 b 0 |
    /// // | 0 0 c |
    /// ```
    pub fn scale_matrix_non_uniform(a: f32, b: f32, c: f32) -> Self {
        Matrix3x3::new_idx(a, 0f32, 0f32, 0f32, b, 0f32, 0f32, 0f32, c)
    }

    /// 2D rotation Matrix by angle teta (radians) over RotationAxis { X, Y, Z }
    /// ```
    /// // Matrix (Z) 3x3:
    /// //       | cosθ −sinθ 0 |
    /// // Rz(θ)=| sinθ  cosθ 0 |
    /// //       |    0     0 1 |
    /// // Matrix (X) 3x3:
    /// //       | 1    0     0 |
    /// // Rx(θ)=| 0 cosθ −sinθ |
    /// //       | 0 sinθ  cosθ |
    /// // Matrix (Y) 3x3:
    /// //       | cosθ 0 -sinθ |
    /// // Ry(θ)=|    0 1     0 |
    /// //       | sinθ 0  cosθ |
    /// ```
    pub fn rotation_2d(teta: f32, rotation_axis: RotationAxis) -> Self {
        let (sin, cos) = teta.sin_cos();
        match rotation_axis {
            RotationAxis::X => {
                Matrix3x3::new_idx(1f32, 0f32, 0f32, 0f32, cos, -sin, 0f32, sin, cos)
            }
            RotationAxis::Y => {
                Matrix3x3::new_idx(cos, 0f32, -sin, 0f32, 1f32, 0f32, sin, 0f32, cos)
            }
            RotationAxis::Z => {
                Matrix3x3::new_idx(cos, -sin, 0f32, sin, cos, 0f32, 0f32, 0f32, 1f32)
            }
        }
    }
}

impl Matrix2x2 {
    #[allow(non_snake_case)]
    /// Identity Matrrix NxN, where all elements Uij that i = j are `1`
    /// ```
    /// // | 1 0 |
    /// // | 0 1 |
    /// ```
    pub fn IDENTITY() -> Self {
        Matrix2x2::new_idx(1f32, 0f32, 0f32, 1f32)
    }

    /// Matrix determinant
    /// ```
    /// //                | 4 -3 |
    /// // let matrix =   | 0  2 |
    ///
    /// // matrix.det() = 2
    /// ```
    ///
    /// ```
    /// use mathf::matrix::{Matrix2x2};
    ///
    /// let matrix = Matrix2x2::new_idx(4.0, -3.0, 0.0, 2.0);
    /// assert_eq!(matrix.det(), 8f32);
    /// ```
    pub fn det(&self) -> f32 {
        self.r1.x * self.r2.y - self.r1.y * self.r2.x
    }

    /// Transforms a MatrixNxN into a Vec<Vec<f32>> by row
    /// ```
    /// //                | 4 -3 |`
    /// // let matrix =   | 0  2 |`

    /// // matrix.det() = 2
    /// ```
    ///
    /// ```
    /// use mathf::matrix::{Matrix2x2};
    ///
    /// let matrix = Matrix2x2::new_idx(4.0, -3.0, 0.0, 2.0);
    /// assert_eq!(matrix.vectorize(), vec![
    ///    vec![4.0, -3.0], vec![0.0, 2.0]    
    /// ]);
    /// ```
    pub fn vectorize(&self) -> Vec<Vec<f32>> {
        vec![self.r1.to_vector(), self.r2.to_vector()]
    }

    /// Same as determinant for Matrix 2x2
    pub fn modulus(&self) -> f32 {
        (self.r1.x * self.r2.y) - (self.r2.x * self.r1.y)
    }

    /// Transpose of Matrix NxN.
    /// Transpose switches rows and columns
    pub fn transpose(&self) -> Self {
        Matrix2x2::new_idx(self.r1.x, self.r2.x, self.r1.y, self.r2.y)
    }

    /// Matrix inverse
    /// ```
    /// //                | 4 7 |
    /// // let matrix =   | 2 6 |
    /// //
    /// //                             |  6 -7 |
    /// // let matrix_inverse =  1/10  | -2  4 |
    /// ```
    ///
    /// ```
    /// use mathf::matrix::{Matrix2x2};
    ///
    /// let matrix = Matrix2x2::new_idx(4.0, 7.0, 2.0, 6.0);
    /// let matrix_inverse = Matrix2x2::new_idx(0.6, -0.7, -0.2, 0.4);
    /// assert_eq!(matrix.inverse(0.001).unwrap(), matrix_inverse);
    /// ```
    pub fn inverse(&self, delta: f32) -> Result<Matrix2x2, Error> {
        let det = self.det();

        if math_helper::float_eq(det, 0f32, delta) {
            Err(Error::NonZeroDeterminantMatrix)
        } else {
            Ok(Matrix2x2::new(
                Vector2::new(self.r2.y / det, -self.r1.y / det),
                Vector2::new(-self.r2.x / det, self.r1.x / det),
            ))
        }
    }

    /// A matrix is orthogonal if and only if transpose(A) == inverse(A)
    pub fn is_orthogonal(&self) -> bool {
        let inv = self.inverse(0.0001);
        if inv.is_ok() {
            self.transpose() == inv.unwrap()
        } else {
            false
        }
    }

    // Uniform scale matrix is determined as Uij / i==j => i = a ^ i!=j => i = 0
    /// ```
    /// // | a 0 |
    /// // | 0 a |
    /// ```
    pub fn scale_matrix(a: f32) -> Self {
        Matrix2x2::IDENTITY() * a
    }

    /// Creates a new Matrix2x2 from 2 vector2 rows
    pub fn new(r1: Vector2, r2: Vector2) -> Matrix2x2 {
        Matrix2x2 { r1, r2 }
    }

    /// Creates a new Matrix2x2 from 4 indexed floats
    pub fn new_idx(n1: f32, n2: f32, n3: f32, n4: f32) -> Matrix2x2 {
        Matrix2x2 {
            r1: Vector2::new(n1, n2),
            r2: Vector2::new(n3, n4),
        }
    }

    pub fn scale_matrix_non_uniform(a: f32, b: f32) -> Self {
        Matrix2x2::new_idx(a, 0f32, 0f32, b)
    }

    /// 2D rotation Matrix by angle teta (radians)
    /// For matrix 2x2:
    /// ```
    /// // P′= | cosθ − sinθ | P
    /// //     | sinθ + cosθ |
    /// ```
    pub fn rotation_2d(teta: f32) -> Self {
        let (sin, cos) = teta.sin_cos();
        Matrix2x2::new_idx(cos, -sin, sin, cos)
    }
}

impl ops::Add for Matrix3x3 {
    type Output = Matrix3x3;

    /// Implements the Matrix 3x3 '+' trait
    fn add(self, new: Matrix3x3) -> Matrix3x3 {
        Matrix3x3::new(
            Vector3::new(
                self.r1.x + new.r1.x,
                self.r1.y + new.r1.y,
                self.r1.z + new.r1.z,
            ),
            Vector3::new(
                self.r2.x + new.r2.x,
                self.r2.y + new.r2.y,
                self.r2.z + new.r2.z,
            ),
            Vector3::new(
                self.r3.x + new.r3.x,
                self.r3.y + new.r3.y,
                self.r3.z + new.r3.z,
            ),
        )
    }
}

impl ops::Add for &Matrix3x3 {
    type Output = Matrix3x3;

    /// Implements the &Matrix 3x3 '+' trait
    fn add(self, new: &Matrix3x3) -> Matrix3x3 {
        Matrix3x3::new(
            Vector3::new(
                self.r1.x + new.r1.x,
                self.r1.y + new.r1.y,
                self.r1.z + new.r1.z,
            ),
            Vector3::new(
                self.r2.x + new.r2.x,
                self.r2.y + new.r2.y,
                self.r2.z + new.r2.z,
            ),
            Vector3::new(
                self.r3.x + new.r3.x,
                self.r3.y + new.r3.y,
                self.r3.z + new.r3.z,
            ),
        )
    }
}

impl ops::Mul<Vector3> for Matrix3x3 {
    type Output = Vector3;

    /// Implements the transform matrix of a vector 3 into another vector 3.
    /// The order should be matrix * vector   because of 3x3 * 3x1 = 3x1
    fn mul(self, vec: Vector3) -> Vector3 {
        Vector3 {
            x: &self.r1 * &vec,
            y: &self.r2 * &vec,
            z: &self.r3 * &vec,
        }
    }
}

impl ops::Mul<&Vector3> for Matrix3x3 {
    type Output = Vector3;

    /// Implements the transform matrix of a vector 3 into another vector 3.
    /// The order should be matrix * vector   because of 3x3 * 3x1 = 3x1
    fn mul(self, vec: &Vector3) -> Vector3 {
        Vector3 {
            x: &self.r1 * vec,
            y: &self.r2 * vec,
            z: &self.r3 * vec,
        }
    }
}

impl ops::Mul<&Vector3> for &Matrix3x3 {
    type Output = Vector3;

    /// Implements the transform matrix of a vector 3 into another vector 3.
    /// The order should be matrix * vector   because of 3x3 * 3x1 = 3x1
    fn mul(self, vec: &Vector3) -> Vector3 {
        Vector3 {
            x: &self.r1 * vec,
            y: &self.r2 * vec,
            z: &self.r3 * vec,
        }
    }
}

impl ops::Mul<&Point3> for Matrix3x3 {
    type Output = Point3;

    /// Implements the transform matrix of a point 3 into another point 3.
    /// The order should be matrix * point  because of 3x3 * 3x1 = 3x1
    fn mul(self, vec: &Point3) -> Point3 {
        Point3 {
            x: &self.r1 * vec,
            y: &self.r2 * vec,
            z: &self.r3 * vec,
        }
    }
}

impl ops::Mul<&Point3> for &Matrix3x3 {
    type Output = Point3;

    /// Implements the transform matrix of a point 3 into another point 3.
    /// The order should be matrix * point  because of 3x3 * 3x1 = 3x1
    fn mul(self, vec: &Point3) -> Point3 {
        Point3 {
            x: &self.r1 * vec,
            y: &self.r2 * vec,
            z: &self.r3 * vec,
        }
    }
}

impl ops::Mul<f32> for Matrix3x3 {
    type Output = Matrix3x3;

    /// Implements the Matrix 3x3 '*' trait for `Matrix3x3 * f32` so that `(identity * value).det() == valueˆ3`.
    fn mul(self, value: f32) -> Matrix3x3 {
        Matrix3x3::new(
            Vector3::new(self.r1.x * value, self.r1.y * value, self.r1.z * value),
            Vector3::new(self.r2.x * value, self.r2.y * value, self.r2.z * value),
            Vector3::new(self.r3.x * value, self.r3.y * value, self.r3.z * value),
        )
    }
}

impl ops::Mul<f32> for &Matrix3x3 {
    type Output = Matrix3x3;

    /// Implements the &Matrix 3x3 '*' trait for `&Matrix3x3 * f32` so that `(identity * value).det() == valueˆ3`.
    fn mul(self, value: f32) -> Matrix3x3 {
        Matrix3x3::new(
            Vector3::new(self.r1.x * value, self.r1.y * value, self.r1.z * value),
            Vector3::new(self.r2.x * value, self.r2.y * value, self.r2.z * value),
            Vector3::new(self.r3.x * value, self.r3.y * value, self.r3.z * value),
        )
    }
}

impl ops::Mul<Matrix3x3> for f32 {
    type Output = Matrix3x3;

    /// Implements the Matrix 3x3 '*' trait for `Matrix3x3 * f32` so that `(value * identity).det() == valueˆ3` and `ßM == Mß`˜.
    fn mul(self, m: Matrix3x3) -> Matrix3x3 {
        Matrix3x3::new(
            Vector3::new(m.r1.x * self, m.r1.y * self, m.r1.z * self),
            Vector3::new(m.r2.x * self, m.r2.y * self, m.r2.z * self),
            Vector3::new(m.r3.x * self, m.r3.y * self, m.r3.z * self),
        )
    }
}

impl ops::Mul<&Matrix3x3> for f32 {
    type Output = Matrix3x3;

    /// Implements the &Matrix 3x3 '*' trait for `&Matrix3x3 * f32` so that `(value * identity).det() == valueˆ3` and `ßM == Mß`˜.
    fn mul(self, m: &Matrix3x3) -> Matrix3x3 {
        Matrix3x3::new(
            Vector3::new(m.r1.x * self, m.r1.y * self, m.r1.z * self),
            Vector3::new(m.r2.x * self, m.r2.y * self, m.r2.z * self),
            Vector3::new(m.r3.x * self, m.r3.y * self, m.r3.z * self),
        )
    }
}

impl ops::Mul<Matrix3x3> for Matrix3x3 {
    type Output = Matrix3x3;

    /// Implements the Matrix 2x2 '*' trait for `Matrix3x3 * Matrix3x3`
    fn mul(self, m: Matrix3x3) -> Matrix3x3 {
        Matrix3x3::new(
            Vector3::new(
                self.r1.x * m.r1.x + self.r1.y * m.r2.x + self.r1.z * m.r3.x,
                self.r1.x * m.r1.y + self.r1.y * m.r2.y + self.r1.z * m.r3.y,
                self.r1.x * m.r1.z + self.r1.y * m.r2.z + self.r1.z * m.r3.z,
            ),
            Vector3::new(
                self.r2.x * m.r1.x + self.r2.y * m.r2.x + self.r2.z * m.r3.x,
                self.r2.x * m.r1.y + self.r2.y * m.r2.y + self.r2.z * m.r3.y,
                self.r2.x * m.r1.z + self.r2.y * m.r2.z + self.r2.z * m.r3.z,
            ),
            Vector3::new(
                self.r3.x * m.r1.x + self.r3.y * m.r2.x + self.r3.z * m.r3.x,
                self.r3.x * m.r1.y + self.r3.y * m.r2.y + self.r3.z * m.r3.y,
                self.r3.x * m.r1.z + self.r3.y * m.r2.z + self.r3.z * m.r3.z,
            ),
        )
    }
}

impl ops::Div<f32> for Matrix3x3 {
    type Output = Matrix3x3;

    /// Implements the Matrix 3x3 '/' trait for `Matrix3x3 / f32` so that `(identity / value).det() == valueˆ3`.
    fn div(self, value: f32) -> Matrix3x3 {
        Matrix3x3::new(
            Vector3::new(self.r1.x / value, self.r1.y / value, self.r1.z / value),
            Vector3::new(self.r2.x / value, self.r2.y / value, self.r2.z / value),
            Vector3::new(self.r3.x / value, self.r3.y / value, self.r3.z / value),
        )
    }
}

impl ops::Div<f32> for &Matrix3x3 {
    type Output = Matrix3x3;

    /// Implements the &Matrix 3x3 '/' trait for `&Matrix3x3 / f32` so that `(identity / value).det() == valueˆ3`.
    fn div(self, value: f32) -> Matrix3x3 {
        Matrix3x3::new(
            Vector3::new(self.r1.x / value, self.r1.y / value, self.r1.z / value),
            Vector3::new(self.r2.x / value, self.r2.y / value, self.r2.z / value),
            Vector3::new(self.r3.x / value, self.r3.y / value, self.r3.z / value),
        )
    }
}

impl ops::Add for Matrix2x2 {
    type Output = Matrix2x2;

    /// Implements the Matrix 2x2 '+' trait
    fn add(self, new: Matrix2x2) -> Matrix2x2 {
        Matrix2x2::new(
            Vector2::new(self.r1.x + new.r1.x, self.r1.y + new.r1.y),
            Vector2::new(self.r2.x + new.r2.x, self.r2.y + new.r2.y),
        )
    }
}

impl ops::Add for &Matrix2x2 {
    type Output = Matrix2x2;

    /// Implements the &Matrix 2x2 '+' trait
    fn add(self, new: &Matrix2x2) -> Matrix2x2 {
        Matrix2x2::new(
            Vector2::new(self.r1.x + new.r1.x, self.r1.y + new.r1.y),
            Vector2::new(self.r2.x + new.r2.x, self.r2.y + new.r2.y),
        )
    }
}

impl ops::Mul<f32> for Matrix2x2 {
    type Output = Matrix2x2;

    /// Implements the Matrix 2x2 '*' trait for `Matrix2x2 * f32` so that `(identity * value).det() == valueˆ2`.
    fn mul(self, value: f32) -> Matrix2x2 {
        Matrix2x2::new(
            Vector2::new(self.r1.x * value, self.r1.y * value),
            Vector2::new(self.r2.x * value, self.r2.y * value),
        )
    }
}

impl ops::Mul<f32> for &Matrix2x2 {
    type Output = Matrix2x2;

    /// Implements the &Matrix 2x2 '*' trait for `&Matrix2x2 * f32` so that `(identity * value).det() == valueˆ2`.
    fn mul(self, value: f32) -> Matrix2x2 {
        Matrix2x2::new(
            Vector2::new(self.r1.x * value, self.r1.y * value),
            Vector2::new(self.r2.x * value, self.r2.y * value),
        )
    }
}

impl ops::Mul<Matrix2x2> for f32 {
    type Output = Matrix2x2;

    /// Implements the Matrix 2x2 '*' trait for `Matrix2x2 * f32` so that `(value * identity).det() == valueˆ2` and `ßM == Mß`˜.
    fn mul(self, m: Matrix2x2) -> Matrix2x2 {
        Matrix2x2::new(
            Vector2::new(m.r1.x * self, m.r1.y * self),
            Vector2::new(m.r2.x * self, m.r2.y * self),
        )
    }
}

impl ops::Mul<&Matrix2x2> for f32 {
    type Output = Matrix2x2;

    /// Implements the &Matrix 2x2 '*' trait for `&Matrix2x2 * f32` so that `(value * identity).det() == valueˆ2` and `ßM == Mß`˜.
    fn mul(self, m: &Matrix2x2) -> Matrix2x2 {
        Matrix2x2::new(
            Vector2::new(m.r1.x * self, m.r1.y * self),
            Vector2::new(m.r2.x * self, m.r2.y * self),
        )
    }
}

impl ops::Mul<Matrix2x2> for Matrix2x2 {
    type Output = Matrix2x2;

    /// Implements the Matrix 2x2 '*' trait for `Matrix2x2 * Matrix2x2`
    fn mul(self, m: Matrix2x2) -> Matrix2x2 {
        Matrix2x2::new(
            Vector2::new(
                self.r1.x * m.r1.x + self.r1.y * m.r2.x,
                self.r1.x * m.r1.y + self.r1.y * m.r2.y,
            ),
            Vector2::new(
                self.r2.x * m.r1.x + self.r2.y * m.r2.x,
                self.r2.x * m.r1.y + self.r2.y * m.r2.y,
            ),
        )
    }
}

impl ops::Div<f32> for Matrix2x2 {
    type Output = Matrix2x2;

    /// Implements the Matrix 2x2 '/' trait for `Matrix2x2 / f32` so that `(identity / value).det() == valueˆ2`.
    fn div(self, value: f32) -> Matrix2x2 {
        Matrix2x2::new(
            Vector2::new(self.r1.x / value, self.r1.y / value),
            Vector2::new(self.r2.x / value, self.r2.y / value),
        )
    }
}

impl ops::Div<f32> for &Matrix2x2 {
    type Output = Matrix2x2;

    /// Implements the Matrix 2x2 '/' trait for `&Matrix2x2 / f32` so that `(identity / value).det() == valueˆ2`.
    fn div(self, value: f32) -> Matrix2x2 {
        Matrix2x2::new(
            Vector2::new(self.r1.x / value, self.r1.y / value),
            Vector2::new(self.r2.x / value, self.r2.y / value),
        )
    }
}

impl ops::Mul<Vector2> for Matrix2x2 {
    type Output = Vector2;

    /// Implements the transform matrix of a vector 2 into another vector 2.
    /// The order should be matrix * vector   because of 2x2 * 2x1 = 2x1
    fn mul(self, vec: Vector2) -> Vector2 {
        Vector2 {
            x: &self.r1 * &vec,
            y: &self.r2 * &vec,
        }
    }
}

impl ops::Mul<&Vector2> for Matrix2x2 {
    type Output = Vector2;

    /// Implements the transform matrix of a vector 2 into another vector 2.
    /// The order should be matrix * &vector   because of 2x2 * 2x1 = 2x1
    fn mul(self, vec: &Vector2) -> Vector2 {
        Vector2 {
            x: &self.r1 * vec,
            y: &self.r2 * vec,
        }
    }
}

impl ops::Mul<&Vector2> for &Matrix2x2 {
    type Output = Vector2;

    /// Implements the transform matrix of a vector 2 into another vector 2.
    /// The order should be matrix * vector   because of 2x2 * 2x1 = 2x1
    fn mul(self, vec: &Vector2) -> Vector2 {
        Vector2 {
            x: &self.r1 * vec,
            y: &self.r2 * vec,
        }
    }
}

impl ops::Mul<&Point2> for Matrix2x2 {
    type Output = Point2;

    /// Implements the transform matrix of a vector 2 into another vector 2.
    /// The order should be matrix * &point   because of 2x2 * 2x1 = 2x1
    fn mul(self, vec: &Point2) -> Point2 {
        Point2 {
            x: &self.r1 * vec,
            y: &self.r2 * vec,
        }
    }
}

impl ops::Mul<&Point2> for &Matrix2x2 {
    type Output = Point2;

    /// Implements the transform matrix of a point 2 into another point 2.
    /// The order should be matrix * point   because of 2x2 * 2x1 = 2x1
    fn mul(self, vec: &Point2) -> Point2 {
        Point2 {
            x: &self.r1 * vec,
            y: &self.r2 * vec,
        }
    }
}

// Indexing
use std::ops::Index;

impl Index<usize> for Matrix2x2 {
    type Output = Vector2;
    fn index(&self, s: usize) -> &Vector2 {
        match s {
            0 => &self.r1,
            1 => &self.r2,
            _ => panic!("Index out of bonds"),
        }
    }
}

impl Index<usize> for Matrix3x3 {
    type Output = Vector3;
    fn index(&self, s: usize) -> &Vector3 {
        match s {
            0 => &self.r1,
            1 => &self.r2,
            2 => &self.r3,
            _ => panic!("Index out of bonds"),
        }
    }
}

#[cfg(test)]
mod tests_matrix3x3 {
    use super::*;

    #[test]
    fn matrix_created_new_from_3_vector3() {
        let actual = Matrix3x3::new(
            Vector3::new(1f32, 2f32, 3f32),
            Vector3::new(1f32, 2f32, 3f32),
            Vector3::new(1f32, 2f32, 3f32),
        );
        let expected = Matrix3x3 {
            r1: Vector3::new(1f32, 2f32, 3f32),
            r2: Vector3::new(1f32, 2f32, 3f32),
            r3: Vector3::new(1f32, 2f32, 3f32),
        };
        assert_eq!(expected, actual);
    }

    #[test]
    fn matrix_created_new_from_9_floats() {
        let actual = Matrix3x3::new_idx(1f32, 1f32, 1f32, 1f32, 1f32, 1f32, 1f32, 1f32, 1f32);
        let expected = Matrix3x3 {
            r1: Vector3::new(1f32, 1f32, 1f32),
            r2: Vector3::new(1f32, 1f32, 1f32),
            r3: Vector3::new(1f32, 1f32, 1f32),
        };
        assert_eq!(expected, actual);
    }

    #[test]
    fn vector_transform_by_matrix() {
        let vec = Vector3::new(1f32, 2f32, 3f32);
        let matrix = Matrix3x3::new_idx(1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32, 9f32);
        let actual = matrix * vec;
        let expected = Vector3::new(14f32, 32f32, 50f32);
        assert_eq!(expected, actual);
    }

    #[test]
    fn point_transform_by_matrix() {
        let point = Point3::new(1f32, 2f32, 3f32);
        let matrix = Matrix3x3::new_idx(1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32, 9f32);
        let actual = &matrix * &point;
        let expected = Point3::new(14f32, 32f32, 50f32);
        assert_eq!(expected, actual);
    }

    #[test]
    fn vector_transform_by_matrix_by_ref() {
        let vec = Vector3::new(1f32, 2f32, 3f32);
        let matrix = Matrix3x3::new_idx(1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32, 9f32);
        let actual = &matrix * &vec;
        let expected = Vector3::new(14f32, 32f32, 50f32);
        assert_eq!(expected, actual);
    }

    #[test]
    fn vector_transform_by_matrix_by_ref_vec_only() {
        let vec = Vector3::new(1f32, 2f32, 3f32);
        let matrix = Matrix3x3::new_idx(1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32, 9f32);
        let actual = matrix * &vec;
        let expected = Vector3::new(14f32, 32f32, 50f32);
        assert_eq!(expected, actual);
    }

    #[test]
    fn seq_maxtrix_det() {
        let matrix = Matrix3x3::new_idx(1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32, 5f32);
        let actual = matrix.det();
        let expected = 12f32;
        assert_eq!(expected, actual);
    }

    #[test]
    fn identity_matrix_has_det_1() {
        let identity = Matrix3x3::IDENTITY();
        assert_eq!(1f32, identity.det());
    }

    #[test]
    fn identity_plus_identity_has_det_8() {
        let identities = Matrix3x3::IDENTITY() + Matrix3x3::IDENTITY();
        assert_eq!(8f32, identities.det());
    }

    #[test]
    fn identity_plus_identity_has_det_8_by_ref() {
        let identities = &Matrix3x3::IDENTITY() + &Matrix3x3::IDENTITY();
        assert_eq!(8f32, identities.det());
    }

    #[test]
    fn det_of_identity_times_3_is_27() {
        let identity_3 = Matrix3x3::IDENTITY() * 3f32;
        assert_eq!(27f32, identity_3.det());
    }

    #[test]
    fn det_of_identity_times_3_is_27_by_ref() {
        let identity_3 = &Matrix3x3::IDENTITY() * 3f32;
        assert_eq!(27f32, identity_3.det());
    }

    #[test]
    fn det_of_4_times_identity_is_64() {
        let identity_4 = 4f32 * Matrix3x3::IDENTITY();
        assert_eq!(64f32, identity_4.det());
    }

    #[test]
    fn det_of_4_times_identity_is_64_by_ref() {
        let identity_4 = 4f32 * &Matrix3x3::IDENTITY();
        assert_eq!(64f32, identity_4.det());
    }

    #[test]
    fn uniform_scale() {
        let matrix = Matrix3x3::scale_matrix(4f32);
        let expected = Matrix3x3::new_idx(4f32, 0f32, 0f32, 0f32, 4f32, 0f32, 0f32, 0f32, 4f32);

        assert_eq!(matrix, expected);
    }

    #[test]
    fn non_uniform_scale() {
        let matrix = Matrix3x3::scale_matrix_non_uniform(4f32, 5f32, 6f32);
        let expected = Matrix3x3::new_idx(4f32, 0f32, 0f32, 0f32, 5f32, 0f32, 0f32, 0f32, 6f32);

        assert_eq!(matrix, expected);
    }

    #[test]
    fn rotation_2d_z() {
        let pi_quarters = 0.78539816f32;
        let rotation_matrix = Matrix3x3::rotation_2d(pi_quarters, RotationAxis::Z);
        let expect = Matrix3x3 {
            r1: Vector3 {
                x: 0.70710677,
                y: -0.70710677,
                z: 0.0,
            },
            r2: Vector3 {
                x: 0.70710677,
                y: 0.70710677,
                z: 0.0,
            },
            r3: Vector3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
        };

        assert_eq!(rotation_matrix, expect);
    }

    #[test]
    fn rotation_2d_x() {
        let pi_quarters = 0.78539816f32;
        let rotation_matrix = Matrix3x3::rotation_2d(pi_quarters, RotationAxis::X);
        let expect = Matrix3x3 {
            r1: Vector3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            r2: Vector3 {
                x: 0.0,
                y: 0.70710677,
                z: -0.70710677,
            },
            r3: Vector3 {
                x: 0.0,
                y: 0.70710677,
                z: 0.70710677,
            },
        };

        assert_eq!(rotation_matrix, expect);
    }

    #[test]
    fn rotation_2d_y() {
        let pi_quarters = 0.78539816f32;
        let rotation_matrix = Matrix3x3::rotation_2d(pi_quarters, RotationAxis::Y);
        let expect = Matrix3x3 {
            r1: Vector3 {
                x: 0.70710677,
                y: 0.0,
                z: -0.70710677,
            },
            r2: Vector3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            r3: Vector3 {
                x: 0.70710677,
                y: 0.0,
                z: 0.70710677,
            },
        };

        assert_eq!(rotation_matrix, expect);
    }

    #[test]
    fn matrix_multiplication() {
        let pi_quarters = 0.78539816f32;
        let rotation_matrix = Matrix3x3::rotation_2d(pi_quarters, RotationAxis::Z);
        let matrix = Matrix3x3::new_idx(1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32, 9f32);
        let expect = Matrix3x3 {
            r1: Vector3 {
                x: -2.1213202,
                y: -2.1213202,
                z: -2.1213202,
            },
            r2: Vector3 {
                x: 3.535534,
                y: 4.9497476,
                z: 6.3639607,
            },
            r3: Vector3 {
                x: 7.0,
                y: 8.0,
                z: 9.0,
            },
        };

        assert_eq!(rotation_matrix * matrix, expect);
    }
}

#[cfg(test)]
mod tests_matrix2x2 {
    use super::*;

    #[test]
    fn matrix_created_new_from_2_vector2() {
        let actual = Matrix2x2::new(Vector2::new(1f32, 2f32), Vector2::new(1f32, 2f32));
        let expected = Matrix2x2 {
            r1: Vector2::new(1f32, 2f32),
            r2: Vector2::new(1f32, 2f32),
        };
        assert_eq!(expected, actual);
    }

    #[test]
    fn matrix_created_new_from_4_floats() {
        let actual = Matrix2x2::new_idx(1f32, 1f32, 1f32, 1f32);
        let expected = Matrix2x2 {
            r1: Vector2::new(1f32, 1f32),
            r2: Vector2::new(1f32, 1f32),
        };
        assert_eq!(expected, actual);
    }

    #[test]
    fn seq_maxtrix_det() {
        let matrix = Matrix2x2::new_idx(4f32, 2f32, 3f32, 4f32);
        let actual = matrix.det();
        let expected = 10f32;
        assert_eq!(expected, actual);
    }

    #[test]
    fn identity_matrix_has_det_1() {
        let identity = Matrix2x2::IDENTITY();
        assert_eq!(1f32, identity.det());
    }

    #[test]
    fn identity_plus_identity_has_det_8() {
        let identities = Matrix2x2::IDENTITY() + Matrix2x2::IDENTITY();
        assert_eq!(4f32, identities.det());
    }

    #[test]
    fn identity_plus_identity_has_det_8_by_ref() {
        let identities = &Matrix2x2::IDENTITY() + &Matrix2x2::IDENTITY();
        assert_eq!(4f32, identities.det());
    }

    #[test]
    fn det_of_identity_times_3_is_9() {
        let identity_3 = Matrix2x2::IDENTITY() * 3f32;
        assert_eq!(9f32, identity_3.det());
    }

    #[test]
    fn det_of_identity_times_3_is_9_by_ref() {
        let identity_3 = &Matrix2x2::IDENTITY() * 3f32;
        assert_eq!(9f32, identity_3.det());
    }

    #[test]
    fn det_of_4_times_identity_is_16() {
        let identity_4 = 4f32 * Matrix2x2::IDENTITY();
        assert_eq!(16f32, identity_4.det());
    }

    #[test]
    fn det_of_4_times_identity_is_16_by_ref() {
        let identity_4 = 4f32 * &Matrix2x2::IDENTITY();
        assert_eq!(16f32, identity_4.det());
    }

    #[test]
    fn vector_transform_by_matrix() {
        let vec = Vector2::new(1f32, 2f32);
        let matrix = Matrix2x2::new_idx(1f32, 2f32, 3f32, 4f32);
        let actual = matrix * vec;
        let expected = Vector2::new(5f32, 11f32);
        assert_eq!(expected, actual);
    }

    #[test]
    fn vector_transform_by_matrix_by_ref() {
        let vec = Vector2::new(1f32, 2f32);
        let matrix = Matrix2x2::new_idx(1f32, 2f32, 3f32, 4f32);
        let actual = &matrix * &vec;
        let expected = Vector2::new(5f32, 11f32);
        assert_eq!(expected, actual);
    }

    #[test]
    fn point_transform_by_matrix() {
        let point = Point2::new(1f32, 2f32);
        let matrix = Matrix2x2::new_idx(1f32, 2f32, 3f32, 4f32);
        let actual = &matrix * &point;
        let expected = Point2::new(5f32, 11f32);
        assert_eq!(expected, actual);
    }

    #[test]
    fn vector_transform_by_matrix_by_ref_vec_only() {
        let vec = Vector2::new(1f32, 2f32);
        let matrix = Matrix2x2::new_idx(1f32, 2f32, 3f32, 4f32);
        let actual = matrix * &vec;
        let expected = Vector2::new(5f32, 11f32);
        assert_eq!(expected, actual);
    }

    #[test]
    #[should_panic]
    fn matrix_inverse_panic_for_det_0() {
        let matrix = Matrix2x2::new_idx(1f32, 1f32, 1f32, 1f32);
        matrix.inverse(0.001f32).unwrap();
    }

    #[test]
    fn matrix_inverse() {
        let matrix = Matrix2x2::new_idx(1f32, 2f32, 3f32, 4f32);
        let expected = Matrix2x2::new_idx(-2f32, 1f32, 1.5f32, -0.5f32);
        assert_eq!(expected, matrix.inverse(0.001f32).unwrap());
    }

    #[test]
    fn modulus() {
        let matrix = Matrix2x2::new_idx(1f32, 2f32, 3f32, 4f32);
        let expected = -2f32;
        assert_eq!(expected, matrix.modulus());
    }

    #[test]
    fn transpose() {
        let matrix = Matrix2x2::new_idx(1f32, 2f32, 3f32, 4f32);
        let expected = Matrix2x2::new_idx(1f32, 3f32, 2f32, 4f32);
        assert_eq!(expected, matrix.transpose());
    }

    #[test]
    fn orthogonal() {
        let matrix = Matrix2x2::new_idx(1.0, 0.0, 0.0, -1.0);
        assert!(matrix.is_orthogonal())
    }

    #[test]
    fn not_orthogonal() {
        let matrix = Matrix2x2::new_idx(1.0, 1.0, 1.0, 1.0);
        assert!(!matrix.is_orthogonal())
    }

    #[test]
    fn uniform_scale() {
        let matrix = Matrix2x2::scale_matrix(4f32);
        let expected = Matrix2x2::new_idx(4f32, 0f32, 0f32, 4f32);

        assert_eq!(matrix, expected);
    }

    #[test]
    fn non_uniform_scale() {
        let matrix = Matrix2x2::scale_matrix_non_uniform(4f32, 5f32);
        let expected = Matrix2x2::new_idx(4f32, 0f32, 0f32, 5f32);

        assert_eq!(matrix, expected);
    }

    #[test]
    fn rotation_2d() {
        let pi_quarters = 0.78539816f32;
        let matrix_rot_2d = Matrix2x2::rotation_2d(pi_quarters);
        let expected = Matrix2x2::new_idx(0.70710677, -0.70710677, 0.70710677, 0.70710677);
        assert_eq!(matrix_rot_2d, expected)
    }

    #[test]
    fn matrix_mul() {
        let pi_quarters = 0.78539816f32;
        let matrix_rot_2d = Matrix2x2::rotation_2d(pi_quarters);
        let matrix = Matrix2x2::new_idx(1f32, 2f32, 3f32, 4f32);
        let expected = Matrix2x2::new_idx(-1.4142134, -1.4142135, 2.828427, 4.2426405);
        assert_eq!(matrix_rot_2d * matrix, expected)
    }
}

#[cfg(test)]
mod tests_matrix_generation {
    use super::*;

    #[test]
    fn generate_vectorized_matrix() {
        let matrix = Matrix3x3::new_idx(1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32, 5f32);
        let expected = vec![
            vec![1f32, 2f32, 3f32],
            vec![4f32, 5f32, 6f32],
            vec![7f32, 8f32, 5f32],
        ];
        assert_eq!(expected, matrix.vectorize());
    }

    #[test]
    fn indexing_m3x3() {
        let matrix = Matrix3x3::new_idx(1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32, 5f32);
        let expected_vector = Vector3::new(4f32, 5f32, 6f32);
        let expected_value = 5f32;
        assert_eq!(expected_vector, matrix[1]);
        assert_eq!(expected_value, matrix[1][1]);
    }

    #[test]
    fn indexing_m2x2() {
        let matrix = Matrix2x2::new_idx(1f32, 2f32, 3f32, 4f32);
        let expected_vector = Vector2::new(3f32, 4f32);
        let expected_value = 4f32;
        assert_eq!(expected_vector, matrix[1]);
        assert_eq!(expected_value, matrix[1][1]);
    }
}

#[cfg(test)]
mod tests_matrix3x3_inverse_functions {
    use super::*;

    #[test]
    fn transpose() {
        let matrix = Matrix3x3::new_idx(2f32, 2f32, 6f32, -1f32, -2f32, -5f32, -2f32, -2f32, -8f32);
        let expected =
            Matrix3x3::new_idx(2f32, -1f32, -2f32, 2f32, -2f32, -2f32, 6f32, -5f32, -8f32);

        assert_eq!(matrix.transpose(), expected);
    }

    #[test]
    fn cofactor() {
        let matrix = Matrix3x3::new_idx(3f32, 1f32, -1f32, 2f32, -2f32, 0f32, 1f32, 2f32, -1f32);
        let expected =
            Matrix3x3::new_idx(2f32, 2f32, 6f32, -1f32, -2f32, -5f32, -2f32, -2f32, -8f32);

        assert_eq!(matrix.cofactor(), expected);
    }

    #[test]
    fn inverse() {
        let matrix = Matrix3x3::new_idx(2f32, -1f32, 3f32, 1f32, 1f32, 1f32, 1f32, -1f32, 1f32);
        let expected = Matrix3x3::new_idx(
            -1f32, 1f32, 2f32, 0f32, 0.5f32, -0.5f32, 1f32, -0.5f32, -1.5f32,
        );

        assert_eq!(matrix.inverse(0.0001).unwrap(), expected);
    }

    #[test]
    fn inverse_err() {
        let matrix = Matrix3x3::new_idx(2f32, -1f32, 3f32, 1f32, 1f32, 1f32, 1f32, 1f32, 1f32);

        let err = matrix.inverse(0.0001).err().unwrap();
        assert_eq!(err, Error::SingularMatrixNotInversible);
    }

    #[test]
    fn modulus() {
        let matrix = Matrix3x3::new_idx(2f32, -1f32, 3f32, 1f32, 1f32, 1f32, 1f32, -1f32, 1f32);

        assert_eq!(matrix.modulus(), -2f32);
    }

    #[test]
    fn orthogonal() {
        let matrix = Matrix3x3::new_idx(
            2f32 / 3f32,
            1f32 / 3f32,
            2f32 / 3f32,
            -2f32 / 3f32,
            2f32 / 3f32,
            1f32 / 3f32,
            1f32 / 3f32,
            2f32 / 3f32,
            -2f32 / 3f32,
        );
        assert!(matrix.is_orthogonal())
    }

    #[test]
    fn not_orthogonal() {
        let matrix = Matrix3x3::new_idx(1f32, 1f32, 1f32, 1f32, 1f32, 1f32, 1f32, 1f32, 1f32);
        assert!(!matrix.is_orthogonal())
    }
}

#[cfg(test)]
mod test_arithmetic {
    use super::*;

    #[test]
    fn maxtrix2x2() {
        let matrix = Matrix2x2::IDENTITY();

        assert_eq!(
            matrix.clone() / 2f32,
            Matrix2x2::new_idx(0.5, 0.0, 0.0, 0.5)
        );
        assert_eq!(
            matrix.clone() * 2f32,
            Matrix2x2::new_idx(2.0, 0.0, 0.0, 2.0)
        );
    }

    #[test]
    fn maxtrix3x3() {
        let matrix = Matrix3x3::IDENTITY();

        assert_eq!(
            matrix.clone() / 2f32,
            Matrix3x3::new_idx(0.5f32, 0f32, 0f32, 0f32, 0.5f32, 0f32, 0f32, 0f32, 0.5f32)
        );
        assert_eq!(
            matrix.clone() * 2f32,
            Matrix3x3::new_idx(2f32, 0f32, 0f32, 0f32, 2f32, 0f32, 0f32, 0f32, 2f32)
        );
    }

    #[test]
    fn maxtrix2x2_by_ref() {
        let matrix = Matrix2x2::IDENTITY();

        assert_eq!(&matrix / 2f32, Matrix2x2::new_idx(0.5, 0.0, 0.0, 0.5));
        assert_eq!(&matrix * 2f32, Matrix2x2::new_idx(2.0, 0.0, 0.0, 2.0));
    }

    #[test]
    fn maxtrix3x3_by_ref() {
        let matrix = Matrix3x3::IDENTITY();

        assert_eq!(
            &matrix / 2f32,
            Matrix3x3::new_idx(0.5f32, 0f32, 0f32, 0f32, 0.5f32, 0f32, 0f32, 0f32, 0.5f32)
        );
        assert_eq!(
            &matrix * 2f32,
            Matrix3x3::new_idx(2f32, 0f32, 0f32, 0f32, 2f32, 0f32, 0f32, 0f32, 2f32)
        );
    }
}

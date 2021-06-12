/// Error for mathf types
#[derive(Debug, PartialEq)]
pub enum Error {
    /// Can't inverse a matrix which determinant is zero.
    /// matrix.det() == 0
    NonZeroDeterminantMatrix,
    /// Can't inverse a 3x3 Singular Matrix.
    /// *A square matrix is singular if and only if its determinant is zero.*
    SingularMatrixNotInversible,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NonZeroDeterminantMatrix => {
                write!(f, "Matrix determinant should be different from ZERO")
            }
            Error::SingularMatrixNotInversible => {
                write!(f, "Matrix3x3 cannont be inverse because it is singular")
            }
        }
    }
}

impl std::error::Error for Error {
    fn description(&self) -> &str {
        match self {
            Error::NonZeroDeterminantMatrix => "Matrix determinant should be different from ZERO",
            Error::SingularMatrixNotInversible => {
                "Matrix3x3 cannont be inverse because it is singular"
            }
        }
    }

    fn cause(&self) -> Option<&dyn std::error::Error> {
        Some(self)
    }
}

#[derive(Debug, PartialEq)]
pub enum Error {
    NonZeroDeterminantMatrix,
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

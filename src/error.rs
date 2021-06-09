#[derive(Debug)]
pub enum Error {
    NonZeroDeterminantMatrix
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NonZeroDeterminantMatrix => write!(f, "Matrix determinant should be different from ZERO")
            
        }
    }
}

impl std::error::Error for Error {
    fn description(&self) -> &str {
        match self {
            Error::NonZeroDeterminantMatrix => "Matrix determinant should be different from ZERO"
        }
    }

    fn cause(&self) -> Option<&dyn std::error::Error> {
        Some(self)
    }
}
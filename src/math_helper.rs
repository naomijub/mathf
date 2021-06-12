/// Function to determine possible equality problems between 2 equal floats.
/// `delta` is the equality variance.
/// ```
/// use mathf::math_helper::float_eq;
/// assert!(float_eq(4.00001, 4.00002, 0.0001));
/// assert!(!float_eq(4.00001, 4.00002, 0.00001));
/// ```
pub fn float_eq(a: f32, b: f32, delta: f32) -> bool {
    a == b || (a.abs() - b.abs()).abs().le(&delta)
}

#[cfg(test)]
mod test {
    use super::float_eq;

    #[test]
    fn non_equal_floats() {
        assert!(!float_eq(5f32, 2f32, 1f32));
    }

    #[test]
    fn equal_floats_small_delta() {
        assert!(float_eq(3.0005f32, 3.0005f32, 0.00000001f32));
    }
}

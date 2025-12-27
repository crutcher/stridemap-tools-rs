//! # Vector Operations

/// Returns the maximum of two vectors.
///
/// Requires that the vectors have the same length.
pub fn vcell_max<T>(a: &[T], b: &[T]) -> Vec<T>
where
    T: Copy + Ord,
{
    assert_eq!(
        a.len(),
        b.len(),
        "vectors must have the same length: {} != {}",
        a.len(),
        b.len(),
    );
    a.iter().zip(b.iter()).map(|(&a, &b)| a.max(b)).collect()
}

/// Returns the sum of two vectors.
///
/// Requires that the vectors have the same length.
pub fn vadd<T>(a: &[T], b: &[T]) -> Vec<T>
where
    T: Copy + std::ops::Add<Output = T>,
{
    assert_eq!(
        a.len(),
        b.len(),
        "vectors must have the same length: {} != {}",
        a.len(),
        b.len(),
    );
    a.iter().zip(b.iter()).map(|(&a, &b)| a + b).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vcell_max() {
        assert_eq!(vcell_max(&[1, 3, 4], &[2, 2, 5]), vec![2, 3, 5]);
    }

    #[test]
    #[should_panic(expected = "3 != 2")]
    fn test_vcell_max_panic() {
        vcell_max(&[1, 2, 3], &[4, 5]);
    }

    #[test]
    fn test_vadd() {
        assert_eq!(vadd(&[1, 2, 3], &[4, 5, 6]), vec![5, 7, 9]);
    }

    #[test]
    #[should_panic(expected = "3 != 2")]
    fn test_vadd_panic() {
        vadd(&[1, 2, 3], &[4, 5]);
    }
}

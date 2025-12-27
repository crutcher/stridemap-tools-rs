//! # Vector Operations

use std::cmp::Ordering;
use std::fmt::Debug;

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

/// Return z-space ortho-regular partial ordering of the vectors.
pub fn vcmp<T>(a: &[T], b: &[T]) -> Option<Ordering>
where
    T: Copy + PartialOrd + Debug,
{
    assert_eq!(
        a.len(),
        b.len(),
        "vectors must have the same length: {:?} != {:?}",
        a,
        b,
    );
    if a.is_empty() {
        return Some(Ordering::Equal);
    }

    let mut ordering = Ordering::Equal;
    for (a, b) in a.iter().zip(b.iter()) {
        match a.partial_cmp(b) {
            None => return None,
            Some(Ordering::Equal) => (),
            Some(ord) => match ordering {
                Ordering::Equal => ordering = ord,
                _ => {
                    if ord != ordering {
                        return None;
                    }
                }
            },
        }
    }
    Some(ordering)
}

/// Assert that the point is within the bounds.
pub fn assert_vle<T>(point: &[T], bounds: &[T])
where
    T: Copy + PartialOrd + Debug,
{
    match vcmp(point, bounds) {
        Some(Ordering::Equal) | Some(Ordering::Less) => (),

        _ => panic!("{:?} is not <= {:?}", point, bounds),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vcmp() {
        assert_eq!(vcmp(&[1, 2, 3], &[1, 2, 3]), Some(Ordering::Equal));
        assert_eq!(vcmp(&[1, 2, 3], &[1, 2, 4]), Some(Ordering::Less));
        assert_eq!(vcmp(&[1, 2, 3], &[1, 3, 2]), None);
    }

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

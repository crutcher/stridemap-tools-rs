//! # `ZRange`
/// A axis-aligned dense range of points in n-dimensional space
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ZRange {
    /// The start of the range; inclusive (if non-empty).
    start: Vec<usize>,

    /// The end of the range; exclusive.
    end: Vec<usize>,
}

impl ZRange {
    /// Create a new range.
    pub fn new(start: Vec<usize>, end: Vec<usize>) -> Self {
        assert_eq!(start.len(), end.len());
        for (s, e) in start.iter().zip(end.iter()) {
            assert!(*s <= *e);
        }
        Self { start, end }
    }

    /// The rank of the range.
    pub fn rank(&self) -> usize {
        self.start.len()
    }

    /// The start of the range; inclusive (if non-empty).
    pub fn start(&self) -> &[usize] {
        &self.start
    }

    /// The end of the range; exclusive.
    pub fn end(&self) -> &[usize] {
        &self.end
    }

    /// The number of elements in the range.
    pub fn elem_count(&self) -> usize {
        self.start
            .iter()
            .zip(self.end.iter())
            .map(|(&s, &e)| e - s)
            .product()
    }

    /// Whether the range is empty.
    pub fn is_empty(&self) -> bool {
        self.start
            .iter()
            .zip(self.end.iter())
            .any(|(&s, &e)| s == e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zrange() {
        let zr = ZRange::new(vec![1, 2, 3], vec![4, 5, 6]);

        assert_eq!(zr.rank(), 3);
        assert_eq!(zr.start(), &[1, 2, 3]);
        assert_eq!(zr.end(), &[4, 5, 6]);

        assert_eq!(zr.elem_count(), 27);
    }
}

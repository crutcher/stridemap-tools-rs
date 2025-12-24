//! # `StrideMap` Partitioning
#![warn(missing_docs)]

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

/// A stride map for efficiently accessing elements in a multi-dimensional array.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StrideMap {
    /// The size of each array element, in bytes.
    elem_size: usize,

    /// The shape of the array.
    shape: Vec<usize>,

    /// The strides of the array.
    ///
    /// May have negative strides.
    ///
    /// A stride of 0 indicates that a dimension is a broadcast dimension
    /// and can have any logical shape (using duplicated storage).
    strides: Vec<isize>,
}

impl StrideMap {
    /// Create a new stride map.
    pub fn new(elem_size: usize, shape: Vec<usize>, strides: Vec<isize>) -> Self {
        assert_eq!(shape.len(), strides.len());
        Self {
            elem_size,
            shape,
            strides,
        }
    }

    /// The rank of the array.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// The size of each array element, in bytes.
    pub fn elem_size(&self) -> usize {
        self.elem_size
    }

    /// The shape of the array.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The strides of the map.
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Whether the given dimension is a reversed dimension.
    pub fn is_reversed_dim(&self, dim: usize) -> bool {
        self.strides[dim] < 0
    }

    /// The dimensions that are reversed dimensions.
    pub fn reversed_dims(&self) -> Vec<usize> {
        (0..self.rank())
            .filter(|&dim| self.is_reversed_dim(dim))
            .collect()
    }

    /// Whether the given dimension is a broadcast dimension.
    pub fn is_broadcast_dim(&self, dim: usize) -> bool {
        self.strides[dim] == 0
    }

    /// The dimensions that are broadcast dimensions.
    pub fn broadcast_dims(&self) -> Vec<usize> {
        (0..self.rank())
            .filter(|&dim| self.is_broadcast_dim(dim))
            .collect()
    }

    /// The number of logical elements in the array.
    pub fn broadcast_elem_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// The size needed to store each logical broadcast element.
    pub fn broadcast_size(&self) -> usize {
        self.elem_size * self.broadcast_elem_count()
    }

    /// The number of internal non-broadcast elements in the array.
    pub fn internal_elem_count(&self) -> usize {
        let mut size = 1;
        for (dim, &shape) in self.shape.iter().enumerate() {
            if !self.is_broadcast_dim(dim) {
                size *= shape;
            }
        }
        size
    }

    /// The size needed to store each internal non-broadcast element.
    pub fn internal_size(&self) -> usize {
        self.elem_size * self.internal_elem_count()
    }

    /// Returns the dimensions in ascending order of stride size.
    pub fn asc_dim_order(&self) -> Vec<usize> {
        let mut pairs: Vec<(usize, usize)> = self
            .strides
            .iter()
            .map(|&s| s.unsigned_abs())
            .enumerate()
            .collect();
        pairs.sort_by_key(|&(_, stride)| stride);
        pairs.iter().map(|&(dim, _)| dim).collect()
    }

    /// Returns the dimensions in descending order of stride size.
    pub fn desc_dim_order(&self) -> Vec<usize> {
        let mut order = self.asc_dim_order();
        order.reverse();
        order
    }

    /// Returns the ravel index of the given index.
    pub fn ravel_index(&self, index: &[usize]) -> isize {
        assert_eq!(index.len(), self.rank());
        index
            .iter()
            .zip(self.strides.iter())
            .map(|(&i, &s)| i as isize * s)
            .sum()
    }

    /// Returns the offset of the given index.
    ///
    /// This is the `ravel_index` * element size.
    pub fn ravel_offset(&self, index: &[usize]) -> isize {
        self.ravel_index(index) * (self.elem_size as isize)
    }

    /// Returns the index of the location with least ravel index.
    pub fn least_index(&self) -> Vec<usize> {
        let mut index = vec![0; self.rank()];
        for dim in self.desc_dim_order() {
            let stride = self.strides[dim];
            if stride < 0 {
                index[dim] = self.shape[dim] - 1;
            }
        }
        index
    }
}

/// TBD; iterator over chunks of the array.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChunkIterator {
    sm: StrideMap,
    order: Vec<usize>,
    broadcast: bool,
}

impl ChunkIterator {
    /// Create a new chunk iterator.
    pub fn new(sm: StrideMap, broadcast: bool) -> Self {
        let order = sm.desc_dim_order();
        Self {
            sm,
            order,
            broadcast,
        }
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

    #[test]
    fn stride_map_size() {
        let elem_size = 4;
        let shape = vec![2, 10, 3, 4];
        let strides = vec![12, 0, 4, 1];

        let sm = StrideMap::new(elem_size, shape.clone(), strides.clone());

        assert_eq!(sm.rank(), shape.len());
        assert_eq!(sm.elem_size(), elem_size);
        assert_eq!(sm.shape(), &shape);
        assert_eq!(sm.strides(), &strides);

        assert_eq!(sm.reversed_dims(), vec![]);
        assert_eq!(sm.broadcast_dims(), vec![1]);

        assert_eq!(sm.asc_dim_order(), vec![1, 3, 2, 0]);
        assert_eq!(sm.desc_dim_order(), vec![0, 2, 3, 1]);

        assert_eq!(sm.least_index(), vec![0, 0, 0, 0]);

        assert_eq!(sm.broadcast_elem_count(), 240);
        assert_eq!(sm.internal_elem_count(), 24);

        assert_eq!(sm.broadcast_size(), sm.broadcast_elem_count() * elem_size);
        assert_eq!(sm.internal_size(), sm.internal_elem_count() * elem_size);
    }

    #[test]
    fn stride_map_size_reversed() {
        let elem_size = 4;
        let shape = vec![2, 10, 3, 4];
        let strides = vec![12, 0, -4, 1];

        let sm = StrideMap::new(elem_size, shape.clone(), strides.clone());

        assert_eq!(sm.rank(), shape.len());
        assert_eq!(sm.elem_size(), elem_size);
        assert_eq!(sm.shape(), &shape);
        assert_eq!(sm.strides(), &strides);

        assert_eq!(sm.reversed_dims(), vec![2]);
        assert_eq!(sm.broadcast_dims(), vec![1]);

        assert_eq!(sm.asc_dim_order(), vec![1, 3, 2, 0]);
        assert_eq!(sm.desc_dim_order(), vec![0, 2, 3, 1]);

        assert_eq!(sm.least_index(), vec![0, 0, 2, 0]);
    }
}

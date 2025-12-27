//! # `StrideMap`

use crate::counters::StepCounter;
use crate::vops::{assert_vle, vadd};
use std::cmp::{max, min};

/// Describes a contiguous block of memory.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BlockInfo {
    /// The offset of the block.
    pub offset: isize,

    /// The size of the block in bytes.
    pub size: usize,
}

/// Describes a contiguous tile of an array.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TileInfo {
    /// The inclusive start of the tile.
    pub start: Vec<usize>,

    /// The exclusive end of the tile.
    pub end: Vec<usize>,
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

    /// Given a ``[start, end)` slice, returns the ravel offset and block size.
    pub fn ravel_slice(&self, start: &[usize], end: &[usize]) -> BlockInfo {
        assert_vle(start, end);

        let start_offset = self.ravel_offset(start);
        let end_offset = self.ravel_offset(end);

        let offset = min(start_offset, end_offset);
        let end = max(start_offset, end_offset);

        let size = (end - offset) as usize;

        BlockInfo { offset, size }
    }

    /// Returns the offset of the given index.
    ///
    /// This is the `ravel_index` * element size.
    pub fn ravel_offset(&self, index: &[usize]) -> isize {
        self.ravel_index(index) * (self.elem_size as isize)
    }

    /// Returns the index of the location with the least ravel index.
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

    /// Returns an iterator over the mapped elements with the given step size.
    pub fn step_iterator(&self, step: &[usize]) -> StepCounter {
        let step = step.to_vec();

        let start = vec![0; self.rank()];
        let end = self.shape.to_vec();

        StepCounter::new(start, end, step)
    }

    /// Returns the maximum contiguous stencil for the stride map.
    pub fn max_contiguous_stencil(&self) -> Vec<usize> {
        let mut stencil = vec![1; self.rank()];
        if self.rank() == 0 {
            return stencil;
        }

        let order = self.asc_dim_order();

        let mut first_non_broadcast = 0;

        // scan past broadcast dimensions (stride == 0)
        for &dim in &order {
            if self.strides[dim] != 0 {
                break;
            }

            stencil[dim] = self.shape[dim];
            first_non_broadcast += 1;
        }

        if first_non_broadcast < self.rank() {
            let dim = order[first_non_broadcast];
            if self.strides[dim] > 1 {
                return stencil;
            }

            stencil[dim] = self.shape[dim];
        }

        for idx in (first_non_broadcast + 1)..self.rank() {
            let prev_dim = order[idx - 1];
            let dim = order[idx];

            let expected_abs_stride = self.shape[prev_dim] * self.strides[prev_dim].unsigned_abs();
            let abs_stride = self.strides[dim].unsigned_abs();

            if abs_stride != expected_abs_stride {
                return stencil;
            }

            stencil[dim] = self.shape[dim];
        }

        stencil
    }

    /// Get an iterator over the contiguous tiles in the stride map.
    pub fn contiguous_tiles(&self) -> TileCounter {
        let stencil = self.max_contiguous_stencil();
        TileCounter::new(self.step_iterator(&stencil), stencil)
    }
}

/// `TileCounter` is used to generate tiles.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TileCounter {
    step_counter: StepCounter,
    tile_size: Vec<usize>,
}

impl TileCounter {
    /// Create a new tile counter.
    pub fn new(step_counter: StepCounter, tile_size: Vec<usize>) -> Self {
        assert_eq!(step_counter.rank(), tile_size.len());
        Self {
            step_counter,
            tile_size,
        }
    }

    /// The rank of the counter.
    pub fn rank(&self) -> usize {
        self.step_counter.rank()
    }
}

impl Iterator for TileCounter {
    type Item = TileInfo;

    fn next(&mut self) -> Option<Self::Item> {
        let start = self.step_counter.next()?;
        let end = vadd(&start, &self.tile_size);
        Some(TileInfo { start, end })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contiguous_tiles() {
        let elem_size = 1;

        // Contiguous, Row-Order
        let shape = vec![2, 2, 3];
        let strides = vec![6, 3, 1];
        let sm = StrideMap::new(elem_size, shape.clone(), strides);
        assert_eq!(sm.max_contiguous_stencil(), shape);
        assert_eq!(
            sm.contiguous_tiles().collect::<Vec<_>>(),
            vec![TileInfo {
                start: vec![0, 0, 0],
                end: vec![2, 2, 3]
            }]
        );

        // Non-contiguous at Middle, w/ Broadcast
        let shape = vec![2, 10, 2, 3, 20];
        let strides = vec![8, 0, 4, 1, 0];
        let sm = StrideMap::new(elem_size, shape.clone(), strides);
        assert_eq!(sm.max_contiguous_stencil(), vec![1, 10, 1, 3, 20]);
        assert_eq!(
            sm.contiguous_tiles().collect::<Vec<_>>(),
            vec![
                TileInfo {
                    start: vec![0, 0, 0, 0, 0],
                    end: vec![1, 10, 1, 3, 20]
                },
                TileInfo {
                    start: vec![0, 0, 1, 0, 0],
                    end: vec![1, 10, 2, 3, 20]
                },
                TileInfo {
                    start: vec![1, 0, 0, 0, 0],
                    end: vec![2, 10, 1, 3, 20]
                },
                TileInfo {
                    start: vec![1, 0, 1, 0, 0],
                    end: vec![2, 10, 2, 3, 20]
                },
            ]
        );
    }

    #[test]
    fn test_contiguous_slices() {
        let elem_size = 1;

        // Contiguous, Row-Order
        let shape = vec![2, 2, 3];
        let strides = vec![6, 3, 1];
        let sm = StrideMap::new(elem_size, shape.clone(), strides);

        let stencil = sm.max_contiguous_stencil();
        assert_eq!(&stencil, &shape);
        assert_eq!(
            sm.step_iterator(&stencil).collect::<Vec<_>>(),
            vec![vec![0, 0, 0]]
        );

        // Non-contiguous at Middle, w/ Broadcast
        let shape = vec![2, 10, 2, 3, 20];
        let strides = vec![8, 0, 4, 1, 0];
        let sm = StrideMap::new(elem_size, shape.clone(), strides);

        let stencil = sm.max_contiguous_stencil();
        assert_eq!(&stencil, &vec![1, 10, 1, 3, 20]);
        assert_eq!(
            sm.step_iterator(&stencil).collect::<Vec<_>>(),
            vec![
                vec![0, 0, 0, 0, 0],
                vec![0, 0, 1, 0, 0],
                vec![1, 0, 0, 0, 0],
                vec![1, 0, 1, 0, 0],
            ]
        );
    }

    #[test]
    fn test_max_contiguous_stencil() {
        let elem_size = 1;

        // Contiguous, Row-Order
        let shape = vec![2, 2, 3];
        let strides = vec![6, 3, 1];
        let sm = StrideMap::new(elem_size, shape.clone(), strides);
        assert_eq!(sm.max_contiguous_stencil(), shape);

        // Contiguous, Shuffled
        let shape = vec![2, 3, 2];
        let strides = vec![6, 1, 3];
        let sm = StrideMap::new(elem_size, shape.clone(), strides);
        assert_eq!(sm.max_contiguous_stencil(), shape);

        // Contiguous, Shuffled, w/ Broadcast
        let shape = vec![2, 3, 10, 2];
        let strides = vec![6, 1, 0, 3];
        let sm = StrideMap::new(elem_size, shape.clone(), strides);
        assert_eq!(sm.max_contiguous_stencil(), shape);

        // Contiguous, Shuffled, w/ Broadcast and negative strides
        let shape = vec![2, 3, 10, 2];
        let strides = vec![6, 1, 0, -3];
        let sm = StrideMap::new(elem_size, shape.clone(), strides);
        assert_eq!(sm.max_contiguous_stencil(), shape);

        // Non-contiguous at Least
        let shape = vec![2, 2, 3];
        let strides = vec![12, 6, 2];
        let sm = StrideMap::new(elem_size, shape.clone(), strides);
        assert_eq!(sm.max_contiguous_stencil(), vec![1, 1, 1]);

        // Non-contiguous at Greatest
        let shape = vec![2, 2, 3];
        let strides = vec![7, 3, 1];
        let sm = StrideMap::new(elem_size, shape.clone(), strides);
        assert_eq!(sm.max_contiguous_stencil(), vec![1, 2, 3]);

        // Non-contiguous at Middle
        let shape = vec![2, 2, 3];
        let strides = vec![8, 4, 1];
        let sm = StrideMap::new(elem_size, shape.clone(), strides);
        assert_eq!(sm.max_contiguous_stencil(), vec![1, 1, 3]);

        // Non-contiguous at Middle, w/ Broadcast
        let shape = vec![2, 10, 2, 3, 20];
        let strides = vec![8, 0, 4, 1, 0];
        let sm = StrideMap::new(elem_size, shape.clone(), strides);
        assert_eq!(sm.max_contiguous_stencil(), vec![1, 10, 1, 3, 20]);
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

    #[test]
    fn test_stride_map_step_iterator() {
        let elem_size = 4;
        let shape = vec![2, 10, 3, 4];
        let strides = vec![12, 0, -4, 1];

        let sm = StrideMap::new(elem_size, shape.clone(), strides.clone());

        let points = sm.step_iterator(&[1, 10, 2, 2]).collect::<Vec<_>>();

        assert_eq!(
            points,
            vec![
                vec![0, 0, 0, 0],
                vec![0, 0, 0, 2],
                vec![0, 0, 2, 0],
                vec![0, 0, 2, 2],
                vec![1, 0, 0, 0],
                vec![1, 0, 0, 2],
                vec![1, 0, 2, 0],
                vec![1, 0, 2, 2],
            ]
        );
    }
}

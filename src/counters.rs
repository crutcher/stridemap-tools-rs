//! # Counters
/// Counter over all points in an n-dimensional range.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NDCounter {
    start: Vec<usize>,
    end: Vec<usize>,
    next: Option<Vec<usize>>,
}

impl NDCounter {
    /// Create a new chunk iterator.
    pub fn new(start: Vec<usize>, end: Vec<usize>) -> Self {
        assert_eq!(start.len(), end.len());
        assert!(start.iter().zip(end.iter()).all(|(&s, &e)| s <= e));
        let current = Some(start.clone());

        Self {
            start,
            end,
            next: current,
        }
    }

    /// Get the start of the range, inclusive.
    pub fn start(&self) -> &[usize] {
        &self.start
    }

    /// Get the end of the range, exclusive.
    pub fn end(&self) -> &[usize] {
        &self.end
    }

    fn increment(&mut self, index: Vec<usize>) -> Option<Vec<usize>> {
        let mut index = index;
        let n = index.len();
        index[n - 1] += 1;
        for dim in (1..n).rev() {
            if index[dim] >= self.end[dim] {
                index[dim] = self.start[dim];
                index[dim - 1] += 1;
            } else {
                break;
            }
        }
        if index[0] >= self.end[0] {
            None
        } else {
            Some(index)
        }
    }
}

impl Iterator for NDCounter {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next.as_ref()?;

        let val = self.next.as_ref().unwrap().clone();
        self.next = self.increment(val.clone());
        Some(val)
    }
}

/// Counter over all points in an n-dimensional range with a given step size.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StepCounter {
    step_counter: NDCounter,
    end: Vec<usize>,
    step: Vec<usize>,
}

impl StepCounter {
    /// Create a new step counter.
    pub fn new(start: Vec<usize>, end: Vec<usize>, step: Vec<usize>) -> Self {
        assert_eq!(start.len(), end.len());
        assert_eq!(start.len(), step.len());

        let rank = start.len();
        let mut shape = Vec::with_capacity(rank);
        for dim in 0..rank {
            let size = (end[dim] - start[dim]).div_ceil(step[dim]);
            shape.push(size);
        }

        Self {
            step_counter: NDCounter::new(start, shape),
            end,
            step,
        }
    }

    /// Get the start of the range, inclusive.
    pub fn start(&self) -> &[usize] {
        &self.step_counter.start
    }

    /// Get the end of the range, exclusive.
    pub fn end(&self) -> &[usize] {
        &self.end
    }

    /// Get the step size.
    pub fn step(&self) -> &[usize] {
        &self.step
    }

    /// Get the dimensional step count.
    pub fn step_count(&self) -> Vec<usize> {
        self.step_counter.next.as_ref().unwrap().clone()
    }
}

impl Iterator for StepCounter {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let val = self.step_counter.next()?;

        Some(
            val.into_iter()
                .zip(self.step.iter())
                .map(|(v, s)| v * s)
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndcounter() {
        let start = vec![0, 0];
        let end = vec![2, 3];
        let counter = NDCounter::new(start, end);

        let points: Vec<Vec<usize>> = counter.into_iter().collect();

        assert_eq!(
            points,
            vec![
                vec![0, 0],
                vec![0, 1],
                vec![0, 2],
                vec![1, 0],
                vec![1, 1],
                vec![1, 2]
            ]
        );
    }

    #[test]
    fn test_stepcounter() {
        let start = vec![0, 0];
        let end = vec![2, 3];
        let step = vec![1, 2];
        let counter = StepCounter::new(start, end, step);

        let points: Vec<Vec<usize>> = counter.into_iter().collect();

        assert_eq!(points, vec![vec![0, 0], vec![0, 2], vec![1, 0], vec![1, 2]]);
    }
}

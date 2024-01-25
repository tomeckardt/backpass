use num_traits::Zero;

pub fn diag<T: Zero + Copy>(dim: usize, value: T) -> Vec<Vec<T>> {
    let mut out = Vec::with_capacity(dim);
    for i in 0..dim {
        let mut row = vec![T::zero(); dim];
        row[i] = value;
        out.push(row);
    }
    out
}

This library, which I haven't given a proper name yet, is a simple project I work on in my free time. Inspired by my lectures, I wanted to try my own implementation of backpropagation and learn Rust at the same time.
# Usage
The most important data structure is the `GradVec<T>`, essentially a `Vec<T>` that can compute its gradient and supports basic arithmetic operations and functions. A `GradVec` can be initialized via `GradVec::from`, which takes any `IntoIterator` as an input.
Example:
```
let vusize: GradVec<usize> = GradVec::from(0..10);
let v1: GradVec<f32> = GradVec::from((0..10).map(|x| x as f32));
let v2: GradVec<f32> = GradVec::from([1.2, -0.4, 2.1]);
```
Since arithmetic operations take ownership of their parameters, a `GradVec` must be borrowed to be used. This does not apply to functions.
```
let v3 = &v1 + &v2;
let v4 = (&v3 * &v3).abs();
```
(`v4` computes $|(v^{(1)} + v^{(2)}) \circ (v^{(1)} + v^{(2)})|$ with $\circ$ being the element-wise multiplication and $|\cdot|$ the element-wise absolute value)

The only optimization function currently supported is Gradient Descent with a batch size of 1. It can be initialized with the `gd` macro.
```
let gd = gd!(lr=0.005, &v1, &v2);
gd.step(&v4)
```
This will perform one optimization step for `v1` and `v2` with regard to the loss gradient wrt `v4`


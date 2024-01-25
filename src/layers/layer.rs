use std::marker::PhantomData;

use crate::GradVec;

pub struct Layer<T> {
    next: Option<Box<Layer<T>>>,
    pd: PhantomData<T>
}

